import sys

# Function to print the Tic-Tac-Toe board
def print_board(board):
    for row in board:
        print(" | ".join(row))
        if row != board[-1]:
            print("---------")

# Function to check if the game is over
def is_game_over(board):
    # Check for a win
    for row in board:
        if row[0] == row[1] == row[2] and row[0] != ' ':
            return True
    for col in range(3):
        if board[0][col] == board[1][col] == board[2][col] and board[0][col] != ' ':
            return True
    if board[0][0] == board[1][1] == board[2][2] and board[0][0] != ' ':
        return True
    if board[0][2] == board[1][1] == board[2][0] and board[0][2] != ' ':
        return True
    # Check for a tie
    if all(cell != ' ' for row in board for cell in row):
        return True
    return False

# MiniMax algorithm
def minimax(board, depth, is_maximizing):
    if is_game_over(board) or depth == 0:
        return evaluate(board), None

    if is_maximizing:
        max_eval = float("-inf")
        move = None
        for i in range(3):
            for j in range(3):
                if board[i][j] == ' ':
                    board[i][j] = 'X'
                    eval, _ = minimax(board, depth - 1, False)
                    board[i][j] = ' '
                    if eval > max_eval:
                        max_eval = eval
                        move = (i, j)
        return max_eval, move
    else:
        min_eval = float("inf")
        move = None
        for i in range(3):
            for j in range(3):
                if board[i][j] == ' ':
                    board[i][j] = 'O'
                    eval, _ = minimax(board, depth - 1, True)
                    board[i][j] = ' '
                    if eval < min_eval:
                        min_eval = eval
                        move = (i, j)
        return min_eval, move
    
def alpha_beta_pruning(board, depth, alpha, beta, is_maximizing):
    if is_game_over(board) or depth == 0:
        return evaluate(board), None

    if is_maximizing:
        max_eval = float("-inf")
        best_move = None
        for i in range(3):
            for j in range(3):
                if board[i][j] == ' ':
                    board[i][j] = 'X'
                    eval, _ = alpha_beta_pruning(board, depth - 1, alpha, beta, False)
                    board[i][j] = ' '
                    if eval > max_eval:
                        max_eval = eval
                        best_move = (i, j)
                    alpha = max(alpha, eval)
                    if beta <= alpha:
                        break
        return max_eval, best_move
    else:
        min_eval = float("inf")
        best_move = None
        for i in range(3):
            for j in range(3):
                if board[i][j] == ' ':
                    board[i][j] = 'O'
                    eval, _ = alpha_beta_pruning(board, depth - 1, alpha, beta, True)
                    board[i][j] = ' '
                    if eval < min_eval:
                        min_eval = eval
                        best_move = (i, j)
                    beta = min(beta, eval)
                    if beta <= alpha:
                        break
        return min_eval, best_move


# Function to evaluate the board state
def evaluate(board):
    # Check if 'X' or 'O' wins or if it's a tie
    for player in ['X', 'O']:
        for i in range(3):
            if all(board[i][j] == player for j in range(3)):
                return 10 if player == 'X' else -10
            if all(board[j][i] == player for j in range(3)):
                return 10 if player == 'X' else -10
        if all(board[i][i] == player for i in range(3)) or all(board[i][2 - i] == player for i in range(3)):
            return 10 if player == 'X' else -10
    return 0

# Function to update the board with a player's move
def make_move(board, player, position):
    row, col = position
    board[row][col] = player

def main():
    if len(sys.argv) != 4:
        print("ERROR: Not enough/too many/illegal input arguments.")
        return

    algo = int(sys.argv[1])
    first = sys.argv[2]
    mode = int(sys.argv[3])

    # Initialize the Tic-Tac-Toe board
    board = [[' ' for _ in range(3)] for _ in range(3)]

    print("Prabhu Avula, A20522815 Solution:")
    print("Algorithm: " + ("MiniMax" if algo == 1 else "MiniMax with alpha-beta pruning"))
    print("First: " + first)
    print("Mode: " + ("human versus computer" if mode == 1 else "computer versus computer"))

    # Game logic here
    while not is_game_over(board):
        if first == "X":
            if mode == 1:
                print_board(board)
                possible_moves = [(i, j) for i in range(3) for j in range(3) if board[i][j] == ' ']
                print("X's move. What is your move (possible moves at the moment are:", [i * 3 + j + 1 for i, j in possible_moves], "| enter 0 to exit the game)?")
                user_input = int(input())
                if user_input == 0:
                    break
                move = ((user_input - 1) // 3, (user_input - 1) % 3)
                make_move(board, 'X', move)
            else:
                # Implement computer versus computer logic
                if algo == 1:
                    _, move = minimax(board, 9, True)
                elif algo == 2:
                    _, move = alpha_beta_pruning(board, 9, float("-inf"), float("inf"), True)
                make_move(board, 'O', move)
                print("O's selected move:", move[0] * 3 + move[1] + 1, ". Number of search tree nodes generated:", "XXXX")

        else:
            if mode == 1:
                print_board(board)
                possible_moves = [(i, j) for i in range(3) for j in range(3) if board[i][j] == ' ']
                print("O's move. What is your move (possible moves at the moment are:", [i * 3 + j + 1 for i, j in possible_moves], "| enter 0 to exit the game)?")
                user_input = int(input())
                if user_input == 0:
                    break
                move = ((user_input - 1) // 3, (user_input - 1) % 3)
                make_move(board, 'O', move)
            else:
                # Implement computer versus computer logic
                if algo == 1:
                    _, move = minimax(board, 9, True)
                elif algo == 2:
                    _, move = alpha_beta_pruning(board, 9, float("-inf"), float("inf"), True)
                make_move(board, 'X', move)
                print("X's selected move:", move[0] * 3 + move[1] + 1, ". Number of search tree nodes generated:", "XXXX")

    # Game over message
    if is_game_over(board):
        result = evaluate(board)
        if result == 10:
            print("X WON")
        elif result == -10:
            print("O WON")
        else:
            print("TIE")
    else:
        if first == "X":
            print("X LOST")
        else:
            print("O LOST")

if __name__ == "__main__":

    main()
