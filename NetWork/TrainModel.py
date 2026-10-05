import copy
import random
import uuid
from collections.abc import Callable, Sequence
import numpy as np
from deap import base, creator, tools
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from . import CreateModel as mdl

CONSOLE = Console()
BOARD_SHAPE = (15, 15)

class TrainModel:
    """Evolve neural-network weights by playing games against the population.

    ``winner_func`` receives a (2, 15, 15) board and returns True for a black
    win, False for a white win, or None while play continues. An optional
    ``winner_after_move_func`` can check only the newly placed stone for faster
    evaluation. ``simulation_budget`` limits extra model.forward evaluations
    per move; initial candidate screening is always completed.
    """

    def __init__(
        self,
        model: mdl.InitModel,
        gen: int,
        pop: int,
        cxpb: float,
        mutpb: float,
        back_funk: Callable[[int, int], object] | None = None,
        winner_func: Callable[[np.ndarray], bool | None] | None = None,
        simulation_budget: int = 8,
        matches_per_individual: int = 2,
        win_threshold: float = 0.3,
        neighborhood: int = 3,
        max_moves: int = 225,
        mutation_sigma: float = 0.1,
        policy_pool_size: int = 12,
        historical_opponent_probability: float = 0.5,
        elo_weight: float = 0.25,
        adaptive_mutation: bool = True,
        seed: int | None = None,
        winner_after_move_func: (
            Callable[[np.ndarray, tuple[int, int]], bool | None] | None
        ) = None,
    ):
        """
        Args:
            model (InitModel): Initial model to evolve.
            gen (int): Number of generations to evolve.
            pop (int): Number of individuals in the population.
            cxpb (float): Probability of crossover.
            mutpb (float): Probability of mutation.
            mutation_sigma (float): Standard deviation of mutation.
            matches_per_individual (int): Number of matches per individual.
            policy_pool_size (int): Maximum number of archived historical policies.
            historical_opponent_probability (float): Chance to play an archived policy.
            elo_weight (float): Weight of normalized Elo in selection fitness.
            adaptive_mutation (bool): Increase mutation when population diversity falls.
        """
        if gen < 1:
            raise ValueError("gen must be at least 1.")
        if pop < 2:
            raise ValueError("pop must be at least 2 for population self-play.")
        if not 0 <= cxpb <= 1 or not 0 <= mutpb <= 1:
            raise ValueError("cxpb and mutpb must be between 0 and 1.")
        if simulation_budget < 0:
            raise ValueError("simulation_budget cannot be negative.")
        if matches_per_individual < 1:
            raise ValueError("matches_per_individual must be at least 1.")
        if not 0 <= win_threshold <= 1:
            raise ValueError("win_threshold must be between 0 and 1.")
        if neighborhood < 1:
            raise ValueError("neighborhood must be at least 1.")
        if not 1 <= max_moves <= BOARD_SHAPE[0] * BOARD_SHAPE[1]:
            raise ValueError("max_moves must be between 1 and 225.")
        if mutation_sigma <= 0:
            raise ValueError("mutation_sigma must be greater than 0.")
        if policy_pool_size < 1:
            raise ValueError("policy_pool_size must be at least 1.")
        if not 0 <= historical_opponent_probability <= 1:
            raise ValueError("historical_opponent_probability must be between 0 and 1.")
        if not 0 <= elo_weight <= 1:
            raise ValueError("elo_weight must be between 0 and 1.")

        self.model = model
        self.gen = gen
        self.pop = pop
        self.cxpb = cxpb
        self.mutpb = mutpb
        self.back_funk = back_funk
        self.winner_func = winner_func
        self.winner_after_move_func = winner_after_move_func
        self.simulation_budget = simulation_budget
        self.matches_per_individual = matches_per_individual
        self.win_threshold = win_threshold
        self.neighborhood = neighborhood
        self.max_moves = max_moves
        self.mutation_sigma = mutation_sigma
        self.policy_pool_size = policy_pool_size
        self.historical_opponent_probability = historical_opponent_probability
        self.elo_weight = elo_weight
        self.adaptive_mutation = adaptive_mutation
        self._rng = random.Random(seed)
        self._mutation_multiplier = 1.0
        self._active_genome_id: int | None = None
        self.best_individual: list[float] | None = None
        self.best_fitness: float | None = None
        self.best_win_score: float | None = None
        self.best_elo_rating: float | None = None
        self.policy_pool: list[tuple[list[float], float]] = []

        if not getattr(model, "model", None):
            raise ValueError(
                "The model has no weights. Call model.CreateModel(layers) before training."
            )

        self._weight_keys = tuple(model.model.keys())
        if any(not isinstance(model.model[key], np.ndarray) for key in self._weight_keys):
            raise TypeError("Every model weight must be a NumPy array.")
        self._weight_shapes = tuple(model.model[key].shape for key in self._weight_keys)
        if any(not np.issubdtype(model.model[key].dtype, np.number) for key in self._weight_keys):
            raise TypeError("Every model weight must be a numeric NumPy array.")
        if any(not np.all(np.isfinite(model.model[key])) for key in self._weight_keys):
            raise ValueError("Model weights must contain only finite values.")

        self._base_genome = np.concatenate(
            [model.model[key].astype(float, copy=False).ravel() for key in self._weight_keys]
        ).tolist()
        if not self._base_genome:
            raise ValueError("The model must contain at least one weight.")

        suffix = uuid.uuid4().hex
        fitness_name = f"TrainModelFitness_{suffix}"
        individual_name = f"TrainModelIndividual_{suffix}"
        creator.create(fitness_name, base.Fitness, weights=(1.0,))
        fitness_type = getattr(creator, fitness_name)
        creator.create(
            individual_name,
            list,
            fitness=fitness_type,
            win_score=0.0,
            elo_rating=1000.0,
        )
        individual_type = getattr(creator, individual_name)

        self.toolbox = base.Toolbox()
        self.toolbox.register("individual", individual_type)
        self.toolbox.register("clone", copy.deepcopy)
        self.toolbox.register("mate", self._mate)
        self.toolbox.register("mutate", self._mutate)
        self.toolbox.register("select", self._select)
        self.hall_of_fame = tools.HallOfFame(1)

    def _activate_genome(self, genome: Sequence[float]) -> None:
        if self._active_genome_id == id(genome):
            return

        flat_weights = np.asarray(genome, dtype=float)
        if flat_weights.size != len(self._base_genome):
            raise ValueError(
                f"Genome has {flat_weights.size} values; "
                f"expected {len(self._base_genome)}."
            )

        offset = 0
        for key, shape in zip(self._weight_keys, self._weight_shapes):
            size = int(np.prod(shape))
            self.model.model[key] = flat_weights[offset : offset + size].reshape(shape).copy()
            offset += size
        self._active_genome_id = id(genome)

    def _predict_black_win(self, board: np.ndarray, genome: Sequence[float]) -> float:
        return float(self._predict_black_wins(board[np.newaxis, ...], genome)[0])

    def _predict_black_wins(
        self, boards: np.ndarray, genome: Sequence[float]
    ) -> np.ndarray:
        self._activate_genome(genome)
        output = np.asarray(self.model.forward(boards), dtype=float).reshape(-1)
        if output.size != len(boards):
            raise ValueError(
                "model.forward must return one black-win probability per board; "
                f"got {output.size} probabilities for {len(boards)} boards."
            )
        if not np.all(np.isfinite(output)):
            raise ValueError("model.forward returned a non-finite win probability.")
        if np.any((output < 0) | (output > 1)):
            raise ValueError(
                "model.forward must return probabilities in [0, 1]."
            )
        return output

    def _candidate_moves(self, board: np.ndarray) -> list[tuple[int, int]]:
        occupied = np.any(board != 0, axis=0)
        rows, columns = np.nonzero(occupied)
        if len(rows) == 0:
            center = (BOARD_SHAPE[0] // 2, BOARD_SHAPE[1] // 2)
            return [center]

        candidates = np.zeros(BOARD_SHAPE, dtype=bool)
        for row, column in zip(rows, columns):
            row_start = max(0, row - self.neighborhood)
            row_end = min(BOARD_SHAPE[0], row + self.neighborhood + 1)
            column_start = max(0, column - self.neighborhood)
            column_end = min(BOARD_SHAPE[1], column + self.neighborhood + 1)
            candidates[row_start:row_end, column_start:column_end] = True

        candidates &= ~occupied
        return [tuple(point) for point in np.argwhere(candidates)]

    def _score_moves(
        self,
        board: np.ndarray,
        is_black: bool,
        genome: Sequence[float],
        progress_callback: Callable[[str], None] | None = None,
    ) -> list[tuple[float, tuple[int, int], np.ndarray]]:
        plane = 0 if is_black else 1
        candidates = self._candidate_moves(board)
        if not candidates:
            return []

        candidate_boards = np.broadcast_to(
            board, (len(candidates), *board.shape)
        ).copy()
        rows, columns = np.asarray(candidates).T
        candidate_boards[np.arange(len(candidates)), plane, rows, columns] = 1
        probabilities = self._predict_black_wins(
            candidate_boards, genome
        )

        scored_moves = []
        for (row, column), next_board, black_probability in zip(
            candidates, candidate_boards, probabilities
        ):
            player_probability = black_probability if is_black else 1 - black_probability
            scored_moves.append((player_probability, (row, column), next_board))
        if progress_callback is not None:
            progress_callback(f"scored {len(candidates)} candidate moves")
        scored_moves.sort(key=lambda item: item[0], reverse=True)
        return scored_moves

    def _get_winner(
        self, board: np.ndarray, last_move: tuple[int, int] | None = None
    ) -> bool | None:
        if self.winner_func is None:
            raise ValueError("winner_func is required to run self-play training.")
        if last_move is not None and self.winner_after_move_func is not None:
            winner = self.winner_after_move_func(board.copy(), last_move)
        else:
            winner = self.winner_func(board.copy())
        if winner is not None and not isinstance(winner, (bool, np.bool_)):
            raise TypeError("winner_func must return True, False, or None.")
        return None if winner is None else bool(winner)

    def _choose_move(
        self,
        board: np.ndarray,
        is_black: bool,
        genome: Sequence[float],
        progress_callback: Callable[[str], None] | None = None,
    ) -> tuple[int, int] | None:
        scored_moves = self._score_moves(
            board, is_black, genome, progress_callback
        )
        if not scored_moves:
            return None

        winning_move = self._first_winning_move(scored_moves, is_black)
        if winning_move is not None:
            return winning_move

        promising = [item for item in scored_moves if item[0] > self.win_threshold]
        if not promising or self.simulation_budget == 0:
            return scored_moves[0][1]

        simulations = [
            {
                "board": item[2],
                "turn_is_black": not is_black,
                "score": item[0],
                "finished": False,
                "initial_move": item[1],
            }
            for item in promising
        ]
        budget_per_move, extra_budget = divmod(
            self.simulation_budget, len(simulations)
        )
        for index, simulation in enumerate(simulations):
            remaining = [budget_per_move + (index < extra_budget)]
            while remaining[0] > 0 and not simulation["finished"]:
                if not self._simulate_step(
                    simulation,
                    is_black,
                    genome,
                    remaining,
                    progress_callback,
                ):
                    break

        best_simulation = max(simulations, key=lambda item: item["score"])
        return best_simulation["initial_move"]

    def _simulate_step(
        self,
        simulation: dict[str, object],
        original_is_black: bool,
        genome: Sequence[float],
        remaining: list[int],
        progress_callback: Callable[[str], None] | None = None,
    ) -> bool:
        board = simulation["board"]
        is_black = simulation["turn_is_black"]
        best_move = None
        best_score = float("-inf")
        best_probability = 0.5
        best_board = None
        plane = 0 if is_black else 1
        candidate_moves = self._candidate_moves(board)
        self._rng.shuffle(candidate_moves)

        for index, (row, column) in enumerate(candidate_moves, start=1):
            next_board = board.copy()
            next_board[plane, row, column] = 1
            winner = self._get_winner(next_board, (row, column))
            if winner is not None:
                simulation["score"] = float(winner == original_is_black)
                simulation["finished"] = True
                if progress_callback is not None:
                    progress_callback(
                        f"simulation found a win among {index} candidate moves"
                    )
                return True
            if remaining[0] == 0:
                break

            probability = self._predict_black_win(next_board, genome)
            remaining[0] -= 1
            score = probability if is_black else 1 - probability
            if score > best_score:
                best_move = (row, column)
                best_score = score
                best_probability = probability
                best_board = next_board

        if best_move is None:
            if remaining[0] > 0:
                simulation["score"] = 0.5
                simulation["finished"] = True
                return True
            return False

        if progress_callback is not None:
            progress_callback(
                f"simulation scored candidates through {index}/{len(candidate_moves)}"
            )
        simulation["board"] = best_board
        simulation["turn_is_black"] = not is_black
        simulation["score"] = (
            best_probability if original_is_black else 1 - best_probability
        )
        return True

    def _first_winning_move(
        self,
        scored_moves: Sequence[tuple[float, tuple[int, int], np.ndarray]],
        is_black: bool,
    ) -> tuple[int, int] | None:
        for _, move, next_board in scored_moves:
            if self._get_winner(next_board, move) is is_black:
                return move
        return None

    def _play_game(
        self,
        black_genome: Sequence[float],
        white_genome: Sequence[float],
        progress_callback: Callable[[str, int], None] | None = None,
    ) -> bool | None:
        board = np.zeros((2, *BOARD_SHAPE), dtype=float)
        is_black = True

        for move_index in range(self.max_moves):
            side = "Black" if is_black else "White"
            if progress_callback is not None:
                progress_callback(
                    f"{side} to play | turn {move_index + 1}/{self.max_moves}",
                    move_index,
                )
            active_genome = black_genome if is_black else white_genome
            move = self._choose_move(
                board,
                is_black,
                active_genome,
                (
                    lambda status: progress_callback(
                        f"{side} to play | turn {move_index + 1}/{self.max_moves} "
                        f"| {status}",
                        move_index,
                    )
                )
                if progress_callback is not None
                else None,
            )
            if move is None:
                if progress_callback is not None:
                    progress_callback("No legal move; game drawn", move_index)
                return None

            board[0 if is_black else 1, move[0], move[1]] = 1
            if progress_callback is not None:
                progress_callback(
                    f"{side} played ({move[0] + 1}, {move[1] + 1})",
                    move_index + 1,
                )
            winner = self._get_winner(board, move)
            if winner is not None:
                if progress_callback is not None:
                    winner_name = "Black" if winner else "White"
                    progress_callback(
                        f"{winner_name} won on turn {move_index + 1}",
                        move_index + 1,
                    )
                return winner
            is_black = not is_black

        if progress_callback is not None:
            progress_callback(f"Draw after {self.max_moves} turns", self.max_moves)
        return None

    def _evaluate_individual(
        self,
        individual: Sequence[float],
        population: Sequence[Sequence[float]],
        index: int,
        progress_callback: Callable[[str, int, int], None] | None = None,
    ) -> float:
        points = 0.0
        rating = float(getattr(individual, "elo_rating", 1000.0))
        opponent_indices = [other for other in range(len(population)) if other != index]

        for match_index in range(self.matches_per_individual):
            if progress_callback is not None:
                progress_callback(
                    f"Match {match_index + 1}/{self.matches_per_individual}: starting",
                    0,
                    match_index,
                )
            use_historical = (
                self.policy_pool
                and self._rng.random() < self.historical_opponent_probability
            )
            if use_historical:
                opponent, opponent_rating = self._rng.choice(self.policy_pool)
            else:
                opponent = population[self._rng.choice(opponent_indices)]
                opponent_rating = float(getattr(opponent, "elo_rating", 1000.0))
            individual_is_black = match_index % 2 == 0
            game_progress_callback = (
                lambda status, moves: progress_callback(
                    f"Match {match_index + 1}/{self.matches_per_individual} "
                    f"| {status}",
                    moves,
                    match_index,
                )
            ) if progress_callback is not None else None
            if individual_is_black:
                winner = self._play_game(
                    individual, opponent, game_progress_callback
                )
            else:
                winner = self._play_game(
                    opponent, individual, game_progress_callback
                )

            match_score = (
                0.5
                if winner is None
                else 1.0
                if winner == individual_is_black
                else 0.0
            )
            points += match_score
            expected_score = self._elo_probability(opponent_rating - rating)
            rating += 32 * (match_score - expected_score)
            if progress_callback is not None:
                result = (
                    "draw"
                    if winner is None
                    else f"{'Black' if winner else 'White'} won"
                )
                progress_callback(
                    f"Match {match_index + 1}/{self.matches_per_individual} "
                    f"complete: {result}",
                    0,
                    match_index + 1,
                )

        win_score = points / self.matches_per_individual
        elo_score = self._elo_probability(1000 - rating)
        individual.win_score = win_score
        individual.elo_rating = rating
        return (1 - self.elo_weight) * win_score + self.elo_weight * elo_score

    @staticmethod
    def _elo_probability(rating_difference: float) -> float:
        exponent = min(20.0, max(-20.0, rating_difference / 400))
        return 1 / (1 + 10**exponent)

    def _population_diversity(self, population: Sequence[Sequence[float]]) -> float:
        genomes = np.asarray(population, dtype=float)
        average_spread = float(np.mean(np.std(genomes, axis=0)))
        average_scale = float(np.mean(np.sqrt(np.mean(np.square(genomes), axis=0))))
        return average_spread / max(average_scale, 1e-8)

    def _update_policy_pool(self, population: Sequence[Sequence[float]]) -> None:
        ranked = sorted(
            population,
            key=lambda item: item.fitness.values[0],
            reverse=True,
        )
        for individual in ranked[:2]:
            self.policy_pool.append(
                (list(individual), float(individual.elo_rating))
            )
        self.policy_pool = self.policy_pool[-self.policy_pool_size :]

    def _initialize_population(self) -> list[list[float]]:
        population = [self.toolbox.individual(self._base_genome.copy())]
        for _ in range(self.pop - 1):
            genome = [
                min(10.0, max(-10.0, value + self._rng.gauss(0, self.mutation_sigma)))
                for value in self._base_genome
            ]
            population.append(self.toolbox.individual(genome))
        return population

    def _mate(
        self, first: list[float], second: list[float]
    ) -> tuple[list[float], list[float]]:
        for index, (first_value, second_value) in enumerate(zip(first, second)):
            gamma = 2 * self._rng.random() - 0.5
            first[index] = min(
                10.0, max(-10.0, (1 - gamma) * first_value + gamma * second_value)
            )
            second[index] = min(
                10.0, max(-10.0, gamma * first_value + (1 - gamma) * second_value)
            )
        return first, second

    def _mutate(self, individual: list[float]) -> tuple[list[float]]:
        probability = min(
            1.0,
            max(1 / len(individual), 0.01) * self._mutation_multiplier,
        )
        for index, value in enumerate(individual):
            if self._rng.random() < probability:
                individual[index] = min(
                    10.0,
                    max(-10.0, value + self._rng.gauss(0, self.mutation_sigma)),
                )
        return (individual,)

    def _select(self, population: Sequence[list[float]], count: int) -> list[list[float]]:
        selected = []
        tournament_size = min(3, len(population))
        for _ in range(count):
            competitors = self._rng.sample(list(population), tournament_size)
            selected.append(
                max(competitors, key=lambda individual: individual.fitness.values[0])
            )
        return selected

    def train(self) -> mdl.InitModel:
        """Run evolution and return the model with the best observed weights."""
        if self.winner_func is None:
            raise ValueError(
                "winner_func is required. It receives a (2, 15, 15) board "
                "and returns True for a black win, False for a white win, "
                "or None while the game is undecided."
            )
        if not callable(self.winner_func):
            raise TypeError("winner_func must be callable.")
        if (
            self.winner_after_move_func is not None
            and not callable(self.winner_after_move_func)
        ):
            raise TypeError("winner_after_move_func must be callable.")
        if self.back_funk is not None and not callable(self.back_funk):
            raise TypeError("back_funk must be callable.")

        population = self._initialize_population()
        CONSOLE.print(
            f"[bold]Starting DEAP self-play: {self.pop} individuals, "
            f"{self.gen} generations, {self.matches_per_individual} games per individual.[/bold]"
        )
        CONSOLE.print(
            f"[dim]Move policy: {self.neighborhood}-cell neighborhood, "
            f"win threshold > {self.win_threshold:.0%}, "
            f"{self.simulation_budget} extra forward evaluations per move.[/dim]"
        )

        for generation in range(self.gen):
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=CONSOLE,
            ) as progress:
                task_id = progress.add_task(
                    f"[cyan]GEN {generation + 1:03d}/{self.gen:03d}[/cyan] "
                    f"Evaluating {len(population)} individuals "
                    f"({len(population) * self.matches_per_individual} games)",
                    total=len(population) * self.matches_per_individual,
                )
                for index, individual in enumerate(population):
                    reported_matches = {"count": 0}
                    match_task_id = progress.add_task(
                        f"Individual {index + 1}/{len(population)} matches",
                        total=self.matches_per_individual,
                    )
                    move_task_id = progress.add_task(
                        f"Individual {index + 1}/{len(population)} current game",
                        total=self.max_moves,
                    )

                    def update_game_progress(
                        status: str,
                        moves_completed: int,
                        matches_completed: int,
                    ) -> None:
                        new_matches = matches_completed - reported_matches["count"]
                        if new_matches > 0:
                            progress.advance(task_id, new_matches)
                            reported_matches["count"] = matches_completed
                        progress.update(
                            match_task_id,
                            completed=matches_completed,
                            description=(
                                f"Individual {index + 1}/{len(population)} "
                                f"matches"
                            ),
                        )
                        progress.update(
                            move_task_id,
                            completed=moves_completed,
                            description=(
                                f"Individual {index + 1}/{len(population)} "
                                f"| {status}"
                            ),
                        )

                    individual.fitness.values = (
                        self._evaluate_individual(
                            individual,
                            population,
                            index,
                            update_game_progress,
                        ),
                    )
                    progress.remove_task(match_task_id)
                    progress.remove_task(move_task_id)

            self.hall_of_fame.update(population)
            best = self.hall_of_fame[0]
            generation_best = max(
                population, key=lambda individual: individual.fitness.values[0]
            )
            average_win_score = sum(
                individual.win_score for individual in population
            ) / len(population)
            self._update_policy_pool(population)
            CONSOLE.print(
                f"[green]Generation {generation + 1}/{self.gen} complete[/green] "
                f"| evaluated: {len(population)} individuals "
                f"| generation best win score: {generation_best.win_score:.3f} "
                f"| generation best Elo: {generation_best.elo_rating:.0f} "
                f"| all-time best win score: {best.win_score:.3f} "
                f"| all-time best Elo: {best.elo_rating:.0f} "
                f"| population average win score: {average_win_score:.3f}"
            )

            if self.back_funk is not None:
                self.back_funk(round((generation + 1) / self.gen * 100), generation + 1)

            if generation + 1 == self.gen:
                break

            diversity = self._population_diversity(population)
            self._mutation_multiplier = (
                min(3.0, 1 + 2 * max(0.0, 0.15 - diversity) / 0.15)
                if self.adaptive_mutation
                else 1.0
            )
            CONSOLE.print(
                f"[dim]Population diversity: {diversity:.3f} "
                f"| mutation multiplier: {self._mutation_multiplier:.2f}x[/dim]"
            )
            elite = self.toolbox.clone(best)
            parents = self.toolbox.select(population, self.pop - 1)
            offspring = [self.toolbox.clone(parent) for parent in parents]

            for first, second in zip(offspring[::2], offspring[1::2]):
                if self._rng.random() < self.cxpb:
                    self.toolbox.mate(first, second)
                    del first.fitness.values
                    del second.fitness.values

            for individual in offspring:
                if self._rng.random() < self.mutpb:
                    self.toolbox.mutate(individual)
                    del individual.fitness.values

            population = [elite, *offspring[: self.pop - 1]]
            self._active_genome_id = None

        self.best_individual = list(self.hall_of_fame[0])
        self.best_fitness = float(self.hall_of_fame[0].fitness.values[0])
        self.best_win_score = float(self.hall_of_fame[0].win_score)
        self.best_elo_rating = float(self.hall_of_fame[0].elo_rating)
        self._active_genome_id = None
        self._activate_genome(self.best_individual)
        CONSOLE.print(
            f"[bold green]Training complete[/bold green] "
            f"| best win score: {self.best_win_score:.3f} "
            f"| Elo: {self.best_elo_rating:.0f}"
        )
        return self.model