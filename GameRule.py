def check(board):  
    directions = [(1, 0), (0, 1), (1, 1), (1, -1)]

    for i in range(15):  
        for j in range(15):  
            if board[i][j] == 0:  
                continue
            
            for dx, dy in directions:  #1 0,0 1,1 1,1 -1
                count = 1
                
                for step in range(1, 9):  
                    x, y = i + dx * step, j + dy * step  

                    if 0 <= x < 15 and 0 <= y < 15 and board[x][y] == 1:
                        count += 1
                    else:
                        break
                if count >= 5:
                    return True
                
    return False

def checkInput(board):
    return True if check(board[0]) else False if check(board[1]) else None

def checkMove(board, move):
    """Check whether the stone at ``move`` completed a five-in-a-row."""
    row, column = move
    if len(board) != 2 or not 0 <= row < 15 or not 0 <= column < 15:
        raise ValueError("Expected a two-plane 15x15 board and an in-bounds move.")

    plane = 0 if board[0][row][column] else 1
    if not board[plane][row][column]:
        return None

    for dx, dy in ((1, 0), (0, 1), (1, 1), (1, -1)):
        count = 1
        for direction in (-1, 1):
            for step in range(1, 5):
                x = row + dx * step * direction
                y = column + dy * step * direction
                if not (0 <= x < 15 and 0 <= y < 15):
                    break
                if not board[plane][x][y]:
                    break
                count += 1
        if count >= 5:
            return plane == 0

    return None