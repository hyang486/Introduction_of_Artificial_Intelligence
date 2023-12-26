# Hyeomin Yang
import heapq
import copy

def get_manhattan_distance(from_state, to_state=[1, 2, 3, 4, 5, 6, 7, 0, 0]):
    """
    TODO: implement this function. This function will not be tested directly by the grader. 

    INPUT: 
        Two states (if second state is omitted then it is assumed that it is the goal state)

    RETURNS:
        A scalar that is the sum of Manhattan distances for all tiles.
    """
    #initialize distance with 0
    distance = 0
    for state_index_from, puzzle_piece in enumerate(from_state):
        # if the puzzle piece is 0, we don't need to calculate the distance (h(n))
        if puzzle_piece == 0:
            continue
        else:
            # calculate the index 
            state_index_to = to_state.index(puzzle_piece)
            # first we need to calcualte the coordinates of the puzzle piece in from state
            col_from = state_index_from % 3
            row_from = state_index_from // 3
            
            # Now we need to calculate the coordinates of the puzzle piece in to state
            col_to = state_index_to % 3
            row_to = state_index_to // 3
            
            # now calcualte each manhattan distance of each puzzle piece. 
            manhattan_distance = abs(col_from - col_to) + abs(row_from - row_to)
            distance = distance + manhattan_distance
    return distance


def print_succ(state):
    """
    TODO: This is based on get_succ function below, so should implement that function.

    INPUT: 
        A state (list of length 9)

    WHAT IT DOES:
        Prints the list of all the valid successors in the puzzle. 
    """
    succ_states = get_succ(state)

    for succ_state in succ_states:
        print(succ_state, "h={}".format(get_manhattan_distance(succ_state)))


def get_succ(state):
    """
    TODO: implement this function.

    INPUT: 
        A state (list of length 9)

    RETURNS:
        A list of all the valid successors in the puzzle (don't forget to sort the result as done below). 
    """
    
    # initialize the list for the sucessors
    succ_states = []
    
    # this is the method for update puzzle state
    def next_state(copy_state, next_index, orig_index):
        # save index of 0 in temp
        temp = copy_state[orig_index]
        # assign not 0 value to position of 0 
        copy_state[orig_index] = copy_state[next_index]
        # assing 0 value to position of not 0
        copy_state[next_index] = temp
        return copy_state
    
    for index_state, state_piece in enumerate(state):
        
        # we only need to care about when the piece is 0 (empty)
        # else(state_piece == 0):
        col_current = index_state % 3
        row_current = index_state // 3
        if(state_piece == 0):
            # moving empty piece to up side 
            if(row_current != 0):
                copy_state_above = copy.deepcopy(state)
                # now change empty space with the piece of above
                above_index = (row_current - 1)  * 3 + (col_current)
                if (state[index_state] != copy_state_above[above_index]):
                    # add successor to succ_states
                    succ_states.append(next_state(copy_state_above, above_index, index_state))
                    
            # moving empty piece to down side
            if(row_current != 2):
                copy_state_down = copy.deepcopy(state)
                # now change empty space with the piece of down
                down_index = (row_current + 1) * 3 + col_current 
                if (state[index_state] != copy_state_down[down_index]):
                    # add successor to succ_states
                    succ_states.append(next_state(copy_state_down, down_index, index_state))
            
            # moving empty piece to right side
            if(col_current != 2):
                copy_state_right = copy.deepcopy(state)
                # now change empty space with the piece of right
                right_index = row_current * 3 + (col_current + 1) 
                if (state[index_state] != copy_state_right[right_index]):
                    # add successor to succ_states
                    succ_states.append(next_state(copy_state_right, right_index, index_state))
                
            # moving empty piece to left side
            if(col_current != 0):
                copy_state_left = copy.deepcopy(state)
                # now change empty space with the piece of left
                left_index = row_current * 3 + (col_current - 1) 
                if (state[index_state] != copy_state_left[left_index]):
                    # add successor to succ_states
                    succ_states.append(next_state(copy_state_left, left_index, index_state))      
        else:
            continue
        

    
   
    return sorted(succ_states)


def solve(state, goal_state=[1, 2, 3, 4, 5, 6, 7, 0, 0]):
    """
    TODO: Implement the A* algorithm here.

    INPUT: 
        An initial state (list of length 9)

    WHAT IT SHOULD DO:
        Prints a path of configurations from initial state to goal state along  h values, number of moves, and max queue number in the format specified in the pdf.
    """
    open = []
    # close disctionay will save all the poped node. it's key will be state and value will be it's id 
    close = {}
    max_length = 0
    # this is open tracker to check if the node is in open or not. it's key will be state and value will be it's id
    open_tracker = {}
    # visited will save all the nodes that is visited at least one tim e
    # the key will be id and value will be node(g+h, state, (g, h, parent_id, my_id))
    visited = {}
    
    # first initialize g and h for the starting points
    start_g = 0
    start_h = get_manhattan_distance(state)
    
    # add to visited 
    start_node = (start_g + start_h, state, (start_g, start_h, -1, 0))
    visited[0] = start_node
    
    # add to open (push)
    heapq.heappush(open, start_node)
    # add to open tracker
    open_tracker[tuple(state)] = 0

    # then handle it's child
    while(open):
        if max_length < len(open):
            max_length = len(open)
        
        # first pop the start node
        node = heapq.heappop(open)
        current_cost, current_state, (current_g, current_h, parent_current, current_id) = node
        
        # also delete it from open tracker to make a sync with open (heapq)
        del open_tracker[tuple(current_state)]
        # then add to close
        close[tuple(current_state)] = current_id
        
        # if current state is matched with goal then terminate the loop with saving all the information of path to state_info_list 
        # and print each step's state and its h and g 
        if (current_state == goal_state):
            # create state_info_list
            state_info_list = []
            # append last current_state's info -> this will be goal's info
            state_info_list.append((current_state, current_h, current_g))
            # then do back tracking uptil it's parent is -1 
            while parent_current != -1:
                # assign each parent's data
                current_cost, current_state, (current_g, current_h, parent_current, current_id) = visited[parent_current]
                # and insert each data at state_info_list's index 0
                state_info_list.insert(0,(current_state, current_h, current_g))


            # This is a format helper.
            # build "state_info_list", for each "state_info" in the list, it contains "current_state", "h" and "move".
            # define and compute max length
            # it can help to avoid any potential format issue.
            for state_info in state_info_list:
                current_state = state_info[0]
                h = state_info[1]
                move = state_info[2]
                print(current_state, "h={}".format(h), "moves: {}".format(move))
            print("Max queue length: {}".format(max_length))
            return
        
        # now handle successor nodes
        successor_nodes = get_succ(current_state)
        
        # iterate all the successors
        for successor in successor_nodes:
            # add 1 to each g 
            new_g = current_g + 1
            # calculate it's h(n)
            new_h = get_manhattan_distance(successor)
            
            # check if the node is already in open or closed or not 

            
                
            # this is the case when the same state is already in open 
            if tuple(successor) in open_tracker:
                # find old node 
                visited_node_id = open_tracker[tuple(successor)]
                visited_node = visited[visited_node_id]
                # check which f(x) is smaller
                if visited_node[0] > new_g + new_h:
                        #update node's data
                    visited[visited_node_id] = (new_g + new_h, successor, (new_g, new_h, current_id, visited_node_id))
                    # get the index of old node that is in open(heapq)
                    visited_node_index = open.index(visited_node)
                    # replace old node to new node
                    open[visited_node_index] = visited[visited_node_id]
                    # because we just replace it now we need to reorder heapq with heapq.heapify()
                    heapq.heapify(open)
                else:
                    pass
                continue
            
            
            # this is the case when the same state is already in close list
            elif tuple(successor) in close:
                # find visited node
                visited_node_id = close[tuple(successor)]
                visited_node = visited[visited_node_id]
                # check which f(x) is smaller
                if visited_node[0] > new_g + new_h:
                    # then update visited node because we no longer need to have old node with higher cost
                    visited[visited_node_id] =  (new_g + new_h, successor, (new_g, new_h, current_id, visited_node_id))
                    # delete it from the close because we need to push it to heapq (open)
                    del close[tuple(successor)]
                    # then push it to open
                    heapq.heappush(open, visited[visited_node_id])
                    # add it to open_tracker for make syncing with open 
                    open_tracker[tuple(successor)] = visited_node_id
                else:
                    pass
                continue
      
                
            else:
                # assign it's new id
                new_succ_id = len(visited) 
                # make a node
                first_seen_node_succesor = (new_g + new_h, successor,(new_g, new_h, current_id, new_succ_id))
                # add it to visited
                visited[new_succ_id] = first_seen_node_succesor
                # add it to open_tracker
                open_tracker[tuple(successor)] = new_succ_id
                # push it to open
                heapq.heappush(open, first_seen_node_succesor)
                
            

if __name__ == "__main__":
    """
    Feel free to write your own test code here to exaime the correctness of your functions. 
    Note that this part will not be graded.
    """
    print_succ([6, 0, 0, 3, 5, 1, 7, 2, 4])
    print()

    print(get_manhattan_distance([6, 0, 0, 3, 5, 1, 7, 2, 4], [1, 2, 3, 4, 5, 6, 7, 0, 0]))
    print()

    solve([6, 0, 0, 3, 5, 1, 7, 2, 4])
    print()
