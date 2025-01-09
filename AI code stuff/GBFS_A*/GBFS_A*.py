# Implementation of Greedy Best First Search algorithm, and A* algorithms to compare which algorithm is better at computing distances between two US state capitals
import pandas as pd
import numpy as np
from queue import PriorityQueue 
import time
import sys

try:
    if(len(sys.argv) == 3):
        initial_state = sys.argv[1] # US Start State
        goal_state = sys.argv[2] # US End State

        #to read both input files (driving.csv with driving distances and straightline.csv for heuristics)
        driving = pd.read_csv("driving.csv", index_col=0)
        straightline = pd.read_csv("straightline.csv", index_col=0)
        state_list = list(driving.index)

        class path():
            def __init__(self, name, algorithm, heuristics, parent=None, path_cost=0):
                self.name = name
                self.connected_nodes = []
                self.path_cost_intial_to_node = path_cost
                self.parent = parent
                self.heuristics = heuristics

                self.find_connected_nodes()

                if algorithm == 'GreedyBestFirstSearch':
                    self.EVAL = self.heuristics 
                elif algorithm == 'Astar':
                    self.EVAL = self.path_cost_intial_to_node + \
                        self.heuristics

            def find_connected_nodes(self):
                for i in state_list:
                    if(driving[self.name][i] != -1 and i != self.name):
                        self.connected_nodes.append(i)

            def getEval(self):
                return self.EVAL

            def __lt__(self, other):
                return self.getEval() < other.getEval()
#For straightline distance
        def straightline_between_nodes(node1, node2):
            if ((node1 in straightline.index) and (node2 in straightline.columns)):
                return (int(straightline[node1][node2]), node1)
            else:
                return False
#For driving distance
        def driving_between_nodes(node1, node2):
            if ((node1 in driving.index) and (node2 in driving.columns)):
                return (int(driving[node1][node2]), node1)
            else:
                return False

        def get_path_solution(end_node):
            nd = end_node
            path_list = []
            while (nd.parent is not None):
                path_list.append(nd.name)
                print(nd.name)
                print(nd.path_cost_intial_to_node)
                nd = nd.parent
            path_list.append(initial_state)
            path_list.reverse()
            return path_list

        def search_path(initial_state, goal_state, alg):
            start = initial_state
            end = goal_state
            alg_used = alg
            frontier = PriorityQueue()
            explored = dict()

            #For cases where the input arguments are same, i.e the same states are given as both inputs
            if(start == end):
                end = path(goal_state, alg_used, 0)
                explored[end.name] = end
                return explored

            else:
                dist = straightline_between_nodes(start, end)
                if(not dist):
                    return False
                else:
                    if(alg_used == "Astar"):
                        dist = dist[0]
                    start = path(initial_state, alg_used, dist)
                    end = path(goal_state, alg_used, 0)

                    frontier.put((dist, start))
                    explored[start.name] = start

                    while (not frontier.empty()):
                        curr_node = frontier.get()[1]

                        #This is the exit code for when the end state is reached
                        if(curr_node.name == end.name):
                            end = curr_node
                            explored[end.name] = end

                            return explored
                        else:
                            for adj_node in curr_node.connected_nodes:
                                p_cost = curr_node.path_cost_intial_to_node
                                d_cost = driving_between_nodes(curr_node.name, adj_node)[0]
                                straightline_cost = straightline_between_nodes(adj_node, end.name)[0]
                                neighbor = path(
                                    adj_node, alg_used, straightline_cost, parent=curr_node, path_cost=p_cost+d_cost)
                                existing_path_cost = False
                                if(neighbor.name in list(explored.keys())):
                                    if(neighbor.path_cost_intial_to_node < explored[neighbor.name].path_cost_intial_to_node):
                                        existing_path_cost = True

                                if(neighbor.name not in explored.keys() or existing_path_cost):
                                    explored[neighbor.name] = neighbor
                                    frontier.put((neighbor.EVAL, neighbor))
                    return False

        #Function to call GreedyBestFirstSearch Algorithm
        gfs_starttime = time.time()
        gfs = search_path(initial_state, goal_state, "GreedyBestFirstSearch")
        gfs_endtime = time.time()

        #Execution time for the algorithm
        gfs_executiontime = gfs_endtime-gfs_starttime

        #Function to call A* Algorithm
        astar_starttime = time.time()
        astar = search_path(initial_state, goal_state, "Astar")
        astar_endtime = time.time()

        #Execution time for the algorithm
        astar_executiontime = astar_endtime-astar_starttime

        print("{}, {}:".format("Avula , Prabhu", "A20522815 Solution"))
        print("Initial State: {}".format(initial_state))
        print("Goal State: {}".format(goal_state), end="\n\n")
        print("Greedy Best First Search:")
 
        if(gfs):
            gfs_end_node = gfs[goal_state]
            gfs_path = get_path_solution(gfs_end_node)
            print("Solution Path: {0}".format(gfs_path))
            print("Number of States on a Path: {0}".format(len(gfs_path)))
            print("Number of Expanded Nodes: {0}".format(len(gfs_path)+1)) #In order to include the extra node
            print("Path Cost: {0}".format(
                gfs_end_node.path_cost_intial_to_node))
            print("Execution Time (in seconds): {0}".format(gfs_executiontime))

        else:
            print("Solution Path: FAILURE: NO PATH FOUND")
            print("Number of States on a Path: 0")
            print("Execution Time (in seconds): {0}".format(gfs_executiontime))

        print("")

        print("A * Search:")

        if(astar):
            astar_end_node = astar[goal_state]
            astar_path = get_path_solution(astar_end_node)
            print("Solution Path: {0}".format(astar_path))
            print("Number of States on a Path: {0}".format(len(astar_path)))
            print("Number of Expanded Nodes: {0}".format(len(astar_path)+1)) #In order to include the extra node
            print("Path Cost: {0}".format(
                astar_end_node.path_cost_intial_to_node))
            print("Execution Time (in seconds): {0}".format(astar_executiontime))
        else:
            print("Solution Path: FAILURE: NO PATH FOUND")
            print("Number of States on a Path: 0")
            print("Execution Time (in seconds): {0}".format(astar_executiontime))

    else:
        print("ERROR: Not enough or too many input arguments.") #In case the user gives too few or too many arguments as inputs


except Exception as ex:
    print("Solution Path: FAILURE: NO PATH FOUND")
    print("Number of States on a Path: 0")
    print("Execution Time (in seconds): {0}".format((astar_executiontime+gfs_executiontime)/2))
