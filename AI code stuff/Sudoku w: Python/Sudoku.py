import csv
import sys
import os
import time

file_name = sys.argv[1]
mode = int(sys.argv[2])
global counter

def CSP_back_tracking(board):
    def get_unassigned_variable():
        #Find the first empty cell 
        for i in range(9):
            for j in range(9):
                if board[i][j] == 0:
                    return (i, j)
        return None

    def is_valid(variable, value):
        row, col = variable
        #Check the row
        for i in range(9):
            if board[row][i] == value:
                return False
        #Check the column
        for i in range(9):
            if board[i][col] == value:
                return False
        #Check the 3x3 box
        box_row = (row // 3) * 3
        box_col = (col // 3) * 3
        for i in range(3):
            for j in range(3):
                if board[box_row + i][box_col + j] == value:
                    return False
        return True

    def get_values(variable):
        #Generate all possible values for a given variable
        row, col = variable
        values = []
        for value in range(1, 10):
            if is_valid(variable, value):
                values.append(value)
        return values

    def solve():
        #Check if solved
        variable = get_unassigned_variable()
        if variable is None:
            return True
        #Try all possible values
        for value in get_values(variable):
            board[variable[0]][variable[1]] = value
            #Recurse
            if solve():
                return True
            #Revert and try next
            board[variable[0]][variable[1]] = 0
        return False

    solve()
    return board


def Brute_Force(board):
    def valid(row, col, num):
        for i in range(9):
            if board[row][i] == num:
                return False
        for i in range(9):
            if board[i][col] == num:
                return False
        box_row = (row // 3) * 3
        box_col = (col // 3) * 3
        for i in range(3):
            for j in range(3):
                if board[box_row + i][box_col + j] == num:
                    return False
        return True

    def solve():
        for row in range(9):
            for col in range(9):
                if board[row][col] == 0:
                    for num in range(1, 10):
                        if valid(row, col, num):
                            board[row][col] = num
                            if solve():
                                return True
                            board[row][col] = 0
                    return False
        return True

    solve()
    return board


def CSP_forward_checking(board):
    def get_unassigned_variable():
        #Find the variable with the least remaining values
        min_values = float('inf')
        min_variable = None
        for i in range(9):
            for j in range(9):
                if board[i][j] == 0:
                    values = get_values((i, j))
                    if len(values) < min_values:
                        min_values = len(values)
                        min_variable = (i, j)
        return min_variable

    def is_valid(variable, value, pruned):
        row, col = variable
        for i in range(9):
            if board[row][i] == value and (row, i) not in pruned:
                return False
        for i in range(9):
            if board[i][col] == value and (i, col) not in pruned:
                return False
        box_row = (row // 3) * 3
        box_col = (col // 3) * 3
        for i in range(3):
            for j in range(3):
                if board[box_row + i][box_col + j] == value and (box_row + i, box_col + j) not in pruned:
                    return False
        return True

    def get_values(variable):
        #Generate values for a variable
        row, col = variable
        values = []
        for value in range(1, 10):
            if is_valid(variable, value, set()):
                values.append(value)
        return values

    def forward_check(variable, value, pruned):
        #Check for conflicts 
        row, col = variable
        conflicts = []
        for i in range(9):
            if board[row][i] == 0 and (row, i) not in pruned:
                values = get_values((row, i))
                if value in values:
                    values.remove(value)
                    if len(values) == 0:
                        return False, None
                    conflicts.append(((row, i), values))
            if board[i][col] == 0 and (i, col) not in pruned:
                values = get_values((i, col))
                if value in values:
                    values.remove(value)
                    if len(values) == 0:
                        return False, None
                    conflicts.append(((i, col), values))
        return True, conflicts

    def solve():
        # Check to see if solved
        variable = get_unassigned_variable()
        if variable is None:
            return True
        # Try all possible values
        for value in get_values(variable):
            board[variable[0]][variable[1]] = value
            # Check for conflicts and prune 
            success, conflicts = forward_check(variable, value, set())
            if success:
                # Recursively solve 
                if solve():
                    return True
                #Revert
                board[variable[0]][variable[1]] = 0
                for conflict, values in conflicts:
                    if variable in values:
                        values.remove(variable)
            else:
                for conflict, values in conflicts:
                    if variable in values:
                        values.append(variable)
        return False

    #Call it
    solve()
    return board


def is_valid_solution(grid):
    for row in grid:
        if sorted(row) != list(range(1, 10)):
            return False
    for col in range(9):
        if sorted([grid[row][col] for row in range(9)]) != list(range(1, 10)):
            return False
    for i in range(0, 9, 3):
        for j in range(0, 9, 3):
            square = [grid[row][col] for row in range(i, i+3) for col in range(j, j+3)]
            if sorted(square) != list(range(1, 10)):
                return False

    return True

def get_input():
    #Please change the path according to your own. I've left the default as my personal path. It could be a different path for you.
    with open(f'/Users/prabhuavula7/Desktop/IIT-C/Sem 2/CS 480 - Intro to AI/Programming assignment 2/{file_name}', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        data = []
        for row in reader:
            #replace X with 0
            new_row = [0 if x == "X" else int(x) for x in row]
            data.append(new_row)
    return data

def print_board(board):
    for i in range(len(board)):
        if i % 3 == 0 and i != 0:
            print("- - - - - - - - - - - - - - ")
        for j in range(len(board[0])):
            if j % 3 == 0 and j != 0:
                print(" | ", end="")
            if j == 8:
                print(board[i][j])
            else:
                print(str(board[i][j]) + " ", end="")



def write_to_csv(data):
    #Please change the path according to your own. I've left the default as my personal path. It could be a different path for you.
    outputfilename = f'/Users/prabhuavula7/Desktop/IIT-C/Sem 2/CS 480 - Intro to AI/Programming assignment 2/{file_name.strip(".csv")}_solution.csv'
    with open(outputfilename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in data:
            writer.writerow(row)

def main():
    board = get_input()
    if mode == 1:
        print('Avula, Prabhu, A20522815')
        print(f'Input file: {file_name}')
        print('Algorithm: Brute Force Exhaustive Search Algorithm')
        print('\nInput puzzle:\n')
        print_board(board)
        starttime=time.time()
        Brute_Force(board)
        endtime=time.time()
        print(f"Search time:" + str(round(endtime - starttime, 5)) + "seconds")
        print('\nSolved puzzle:\n')
        print_board(board)
        write_to_csv(board)
    elif mode == 2:
        print('Avula, Prabhu Kiran, A20522815')
        print(f'Input file: {file_name}')
        print('Algorithm: Constraint Satisfaction Problem Back-Tracking Search')
        print('\nInput puzzle:\n')
        print_board(board)
        starttime=time.time()
        CSP_back_tracking(board)
        endtime=time.time()
        print(f"Search time:" + str(round(endtime - starttime, 5)) + "seconds")
        print('\nSolved puzzle:\n')
        print_board(board)
        write_to_csv(board)
    elif mode == 3:
        print('Avula, Prabhu Kiran, A20522815')
        print(f'Input file: {file_name}')
        print('Algorithm: Constraint Satisfaction Problem with Forward-Checking and MRV Heuristics')
        print('\nInput puzzle:\n')
        print_board(board)
        starttime=time.time()
        CSP_forward_checking(board)
        endtime=time.time()
        print(f"Search time:" + str(round(endtime - starttime, 5)) + "seconds")
        print('\nSolved puzzle:\n')
        print_board(board)
        write_to_csv(board)
    elif mode == 4:
        if is_valid_solution(board):
            print("This is a valid, solved, Sudoku puzzle.")
        else:
            print("ERROR: This is NOT a solved Sudoku puzzle.")
    else:
        print("Invalid mode, please try again.")


if __name__ == "__main__":
    main()