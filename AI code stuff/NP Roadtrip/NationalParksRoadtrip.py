# Import libraries. Chose to use pandas because its just easier than using csv
import sys
import pandas as pd

def load_data(): #Function to load the data from the three documents 
    driving_distances = pd.read_csv('driving2.csv', index_col='STATE') #Loaded as a matrix dataframe
    parks = pd.read_csv('parks.csv', index_col='STATE').iloc[0] #Loaded as a series for the parks
    zones = pd.read_csv('zones.csv', index_col='STATE').iloc[0] #Loaded as a series for the zones Z1-Z12
    return driving_distances, parks, zones

def get_next_zone(current_zone): #Get the Zone number in our westward moving nature voyage (I personally think Utah Parks or the other West Coast parks are the best)
    return f"Z{int(current_zone[1:]) + 1}" if int(current_zone[1:]) < 12 else None

def get_states_in_zone(zone, zones): #Get all the states in a specific zone. Again, Z12 is visually hard too beat!
    return sorted(zones[zones == int(zone[1:])].index.tolist())

def is_valid_move(current_state, next_state, driving_distances): #Check for a valid node connection in the road network to see if the path can continue forward.  
    return driving_distances.at[current_state, next_state] > 0

def backtrack(current_state, current_zone, visited_parks, path, driving_distances, parks, zones, min_parks): #CSP Backtracking code
    if current_zone == "Z12": #Check if the current zone is Z12 as that is our final destination
        if visited_parks >= min_parks:
            return path, visited_parks
        else:
            return None, 0

    next_zone = get_next_zone(current_zone) #Get the next zone to explore
    if not next_zone:
        return None, 0

    for state in get_states_in_zone(next_zone, zones): #Iterate over each state in the selected zone
        if is_valid_move(current_state, state, driving_distances): #Check for a valid road connection
            new_path = path + [state] #Update the path
            new_visited_parks = visited_parks + parks[state] #Update the states
            result, total_parks = backtrack(state, next_zone, new_visited_parks, new_path, driving_distances, parks, zones, min_parks) #Recursively continue to explore from this state until we hit the goal state
            if result: #Return results
                return result, total_parks

    return None, 0

def main(): #Main driver function 
    if len(sys.argv) != 3: #For the inputs. Has to be two inputs, anything otherwise throws the error below
        print("ERROR: Not enough or too many input arguments.")
        return

    initial_state = sys.argv[1] #Take and parse the inputs
    min_parks = int(sys.argv[2])

    driving_distances, parks, zones = load_data() #Load the data from the csv files

    if initial_state not in zones.index: #Check to see if the input initial state is valid. If not, it will throw the error below
        print(f"ERROR: Invalid initial state '{initial_state}'.")
        return

    initial_zone = f"Z{zones[initial_state]}" #Get the initial, starting zone for the provided input state
    path, visited_parks = backtrack(initial_state, initial_zone, parks[initial_state], [initial_state],
                                    driving_distances, parks, zones, min_parks) #Run the backtracking algorithm with the initial state

    print(f"Avula, Prabhu, A20522815 Solution:") #General print statement with my credentials
    print(f"Initial state: {initial_state}") #Show the input state to the user
    print(f"Minimum number of parks: {min_parks}") #Show the number of parks the user wants to visit
    print()

    if path: #If a valid path is found, print the path, its cost and the number of national parks on that path
        path_cost = sum(driving_distances.at[path[i], path[i+1]] for i in range(len(path) - 1))
        print(f"Solution path: {', '.join(path)}")
        print(f"Number of states on a path: {len(path)}")
        print(f"Path cost: {path_cost}")
        print(f"Number of national parks visited: {visited_parks}")
    else: #If no solution path is found, then these series of statements are to be shown to the user
        print("Solution path: FAILURE: NO PATH FOUND")
        print("Number of states on a path: 0")
        print("Path cost: 0")
        print("Number of national parks visited: 0")

if __name__ == "__main__": #main access point for the entire program
    main()