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