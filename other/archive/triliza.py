board = [["     ", "     ", "     "], 
         ["     ", "     ", "     "], 
         ["     ", "     ", "     "]]

def print_board(board):
    for rows in board:
        print("+-----+------+------+")
        print("|", end="")
        for element in rows:
            print(element + "|", end=" ")
        print()
    print("+-----+------+------+")

def player(player_name):
    pl_row = int(input(f"\nPlayer {player_name.strip()}, choose row (1-3):"))
    pl_col = int(input(f"Player {player_name.strip()}, choose col (1-3):"))
    print("\nGame on! Here we are: \n")
    return pl_row, pl_col

def score(board):
    score_x = 0
    score_o = 0
    '''needs more work to find and announce winner and start new gme'''
    
    if score_x == 1:
        print(f"Player X won!")
    if score_o == 1:
        print(f"Player O won!")
    else:
        pass

player1 = "  X  "
player2 = "  O  "

print_board(board)

while True:
    
    pl1_row, pl1_col = player(player1)
    board[pl1_row - 1][pl1_col - 1] = "  X  "
    print_board(board)
    score(board)

    pl2_row, pl2_col = player(player2)
    board[pl2_row - 1][pl2_col - 1] = "  O  "
    print_board(board)
    score(board)