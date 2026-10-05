import NetWork as nt
import GameRule as gamer

def back_func(*args):
    print(args)

if __name__ == '__main__':
    model = nt.InitModel()
    model.CreateModel([128, 64, 16])
    train = nt.TrainModel(
        model,
        gen=3,
        pop=20,
        cxpb=0.7,
        mutpb=0.2,
        matches_per_individual=50,
        winner_func=gamer.checkInput,
        winner_after_move_func=gamer.checkMove,
        simulation_budget=0,
        historical_opponent_probability=0.5,
        elo_weight=0.25,
    )
    model = train.train()
    model.SaveModel("./", "test", {"info" : "test"})