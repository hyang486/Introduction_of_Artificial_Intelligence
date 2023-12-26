import random
import copy
import time

class TeekoPlayer:
    """ An object representation for an AI game player for the game Teeko.
    """
    board = [[' ' for j in range(5)] for i in range(5)]
    pieces = ['b', 'r']

    def __init__(self):
        """ Initializes a TeekoPlayer object by randomly selecting red or black as its
        piece color.
        """
        self.my_piece = random.choice(self.pieces)
        self.opp = self.pieces[0] if self.my_piece == self.pieces[1] else self.pieces[1]
        # this is for the dept limitation
        self.limit_depth = 2

    def make_move(self, state):
        """ Selects a (row, col) space for the next move. You may assume that whenever
        this function is called, it is this player's turn to move.

        Args:
            state (list of lists): should be the current state of the game as saved in
                this TeekoPlayer object. Note that this is NOT assumed to be a copy of
                the game state and should NOT be modified within this method (use
                place_piece() instead). Any modifications (e.g. to generate successors)
                should be done on a deep copy of the state.

                In the "drop phase", the state will contain less than 8 elements which
                are not ' ' (a single space character).

        Return:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.

        Note that without drop phase behavior, the AI will just keep placing new markers
            and will eventually take over the board. This is not a valid strategy and
            will earn you no points.
        """
    
        alpha = float("-inf")
        beta = float("inf")
        move = []
        start_depth = 0

        # this is for checking drop phase or not 
        drop_phase_info = self.detect_drop_phase(state)
        # store all succesor information to succ_info
        succ_info = self.get_succ(state, self.my_piece)
        
        # iterate all succ info to get max case
        for info in succ_info:
            # call minmax function for this project we need to call max 
            recursive_value = self.max_value(info[4], start_depth , alpha, beta)
            # if ths max value is beigger then alpha, then update alpha
            if recursive_value > alpha:
                alpha = recursive_value
                # keep update next row and col with best value
                # if drop phase, we only need original row and col for drop my piece
                if drop_phase_info:
                    next = (info[0], info[1])
                # if it is not, then we need to get source row and col and new row and col
                else:
                    next = (info[2], info[3])
                    orig = (info[0], info[1])
        # store best case to move list
        if drop_phase_info:
            move.append(next)
        else:
            move.append(next)
            move.append(orig)
        
        return move

            

    """
    This is a helper function of detect current state is drop phase or not
    if drop phase then return true else, return false 
    """
    def detect_drop_phase(self, state):
        # initialize detect_drop with true value
        detect_drop = True
        # this is for counting num of pieces in the board
        count_current_pieces = 0
        for row in range(5):
            for col in range(5):
                # count non-empty position in state
                if state[row][col] != ' ':
                    # update count_current_pieces 
                    count_current_pieces = count_current_pieces + 1
        # if num of piece is equal or greater then 9 this is not drop_phase
        if count_current_pieces >= 8:
            detect_drop = False
        
        return detect_drop

    
    """
    This is a helper function for get successer of current state so if current state is alpha 
    then the successor will be beta and then return it's row, column information 
    if drop_phase only return the postion that piece is dropped else, return originla row,col and 
    next row, col
    state -> current state 
    piece_color -> red or black (also same with alpha beta)
    """
    def get_succ(self, state, piece_color):
        list_of_succ = []
        # check if it is drop_phase or not
        drop_phase = self.detect_drop_phase(state)
        
        # just put piece in the empty space in the board
        if drop_phase:
            # iterate all position in the board
            for current_row in range(5):
                for current_col in range(5):
                    # if it is empty space
                    if state[current_row][current_col] == ' ':
                        # this is for storeing possible next state, so deepcopy state for not modifying anything in original state
                        next_sample = copy.deepcopy(state)
                        # drop the piece 
                        next_sample[current_row][current_col] = piece_color
                        # then store current row and col and sample of next state
                        list_of_succ.append([current_row, current_col, None, None,next_sample])
        else:
            # iterate all position in the board
            for current_row in range(5):
                for current_col in range(5):
                    # if we found my piece
                    if state[current_row][current_col] == piece_color:
                        # then get all the possible row and col's next position
                        possible_move = self.get_possible_move_position(state, current_row, current_col)
                        for next in possible_move:
                            # this is for storeing possible next state, so deepcopy state for not modifying anything in original state
                            next_sample = copy.deepcopy(state)
                            # change original space to empty 
                            next_sample[current_row][current_col] = ' '
                            # then drop the piece on next state
                            next_sample[next[0]][next[1]] = piece_color
                            # then store current row and col and next row and col and sample of next state
                            list_of_succ.append([current_row, current_col, next[0], next[1], next_sample])
        
        return list_of_succ
                      
                
    
    """
    This is the helper method for find possible move postition
    the piece can move with 8 ways (4 diagonal, up, down, right left)
    check if those place is empty and not out of board and return the list that store 
    possible move position (row , col)
    """
    def get_possible_move_position(self, state, original_row, original_col):
        max_row_col = 5
        # this is for return, it will store all possible move positions(row, col)
        possible_move_position = []
        # this is dictionary for all possible move positions difference with original position 'direction' : (row differecne, col difference)
        possible_move = {'up' : (-1, 0), 'down' : (+1, 0), 'right' : (0, +1), 
                         'left' : (0, -1), 'up-left' : (-1, -1), 'up-right' : (-1, +1), 
                         'down-left' : (+1, -1), 'down-right' : (+1, +1)}
        
        # with loop check all the possible move position only check with value not key in dictionary
        for possible_move in possible_move.values():
            # this is all possible move row and col value
            possible_row = original_row + possible_move[0]
            possible_col = original_col + possible_move[1]
            # check out of board or not and place is emtpy
            if 0 <= possible_row < max_row_col and 0 <= possible_col < max_row_col and state[possible_row][possible_col] == ' ':
                    # if it is not then append it to return list
                    possible_move_position.append([possible_row, possible_col])
                    
        return possible_move_position   

    

    """
    this is for max value of successors, it will check drop phase or not and get the all the max value from successors and return the most max value
    it will call min value which is it's successors's value and also it has alpha prunning function
    """
    def max_value(self, state, depth, alpha, beta):
        # if the game is over before the max depth is reached
        if self.game_value(state) != 0:
            return self.game_value(state)

        # if the max depth is reached but game is not over then use heuristic_value
        if depth == self.limit_depth:
            return self.heuristic_value(state)
        # drop info
        drop_phase = self.detect_drop_phase(state)
        # get all the possible and sample of next state
        successors = self.succ_sample(state, drop_phase, self.my_piece) 
        # iterate all the succ info 
        for sample in successors:
             # update alpha so, we can get best state in successors
            alpha = max(alpha, self.min_value(sample, depth+1, alpha, beta)) 
            # this is for alpha pruning
            if alpha >= beta:
                return beta

        return alpha
    
    """
        this is for get all the possible state of successors 
        this method can seems unnecessary method but this is for just get sample state not row and col info
    """
    def succ_sample(self, state, drop_phase, my_piece):
        # return lsit
        sample_list = []
        # get all the information form the get_succ 
        possible_move = self.get_succ(state, my_piece)
        # iterate all the successors
        for move in possible_move:
            # check drop phase or not
            if drop_phase == True:
                # and append all the possible and acceptable state
                sample_list.append(move[4])
            else:
                sample_list.append(move[4])
                
        return sample_list
            

    """
        this is for min value of successors, it will check drop phase or not and get 
        the all the min value from successors and return the most min value
        it will call max value which is it's successors's value and also it has beta prunning function
    """
    def min_value(self, state, depth, alpha, beta):
        # if the game is over before the max depth is reached
        if self.game_value(state) != 0:
            return self.game_value(state)
        
        # if the max depth is reached but game is not over then use heuristic_value
        if depth == self.limit_depth:
            return self.heuristic_value(state)
        
        # drop info
        drop_phase = self.detect_drop_phase(state)
        # get all the possible and sample of next state
        successors = self.succ_sample(state, drop_phase, self.opp) 
        # iterate all the successors sample
        for sample in successors:
            # update beta so, we can get best state in successors
            beta = min(beta, self.max_value(sample, depth+1, alpha, beta)) # update the alpha
            # this is for beta pruning
            if beta <= alpha:
                return alpha

        return beta
    
    
    """
    this is for heuristic value 
    this method will call heuristic_calculator method. so method is just for 
    final calculation
    """
    def heuristic_value(self, state):

        my_piece_score = 0
        opp_score = 0
        # iterate all the board to find start piece
        for row in range(5):
            for col in range(5):
                if state[row][col] == self.my_piece:
                        # this is for my piece's heuristic value
                        my_piece_score = self.heuristic_calculator(state, self.my_piece)
                if state[row][col] == self.opp:
                        # this is for opposit player's heurstic value
                        opp_score = self.heuristic_calculator(state, self.opp)
        # return my_piece's heuristic value - opp_piece's heuristic value because we need to choose best case for my piece
        return my_piece_score - opp_score
    

    """
    this is for real heuristic value calculation, this method will return each defalut value when the piece has no piece in adjacent postion
    otherwise, it will check continuous piece number and give a heuristic score 
    """
    def heuristic_calculator(self, state, piece_color):
        
        # this for my value
        heuristic_value = 0.0
        # this is for opposite value
        opp_value = 0.0
        # this is for extra
        extra_credit = 0.0

        """
        this is referee for calculate all the score this will calculate
        1) when the piece has no piece in adjacent postion
        2) when the piece ans continuous piece (vertical, horizontal, diagonal)
            2-1) when the opp_piece located in adjacent postion of my_piece
        """
        def referee(num_adjacent, opp_count, row , col):
            # this is defalut value of when the piece located alone
            default_value =[
                [0.02, 0.05, 0.05, 0.05, 0.02],
                [0.05, 0.15, 0.15, 0.15, 0.05],
                [0.05, 0.15, 0.35, 0.15, 0.05],
                [0.05, 0.15, 0.15, 0.15, 0.05],
                [0.02, 0.05, 0.05, 0.05, 0.02],
            ]
            # this is case for my_piece
            if num_adjacent < 1:
                heuristic_value == default_value[row][col]
            if num_adjacent == 1:
                heuristic_value == 0.2
            if num_adjacent > 1:
                heuristic_value == 0.7
            # this is case for when opp_piece is located in adjacent position
            if opp_count < 1:
                opp_value == 0.0
            if opp_count == 1:
                opp_value == 0.1
            if opp_count > 1:
                opp_value == 0.3

            return heuristic_value - opp_value 
        
                    
        # check horizontal 
        for row in range(5):
            for col in range(2):
                if state[row][col] == piece_color:
                    count = 0
                    opp_count = 0
                    for index in range(1, 4):
                        # if postion is not out of board and next has same piece
                        if col + index < 5 and state[row][col + index] == piece_color:
                            count = count + 1
                        # if adjacent postion has opp_value (blocking)
                        elif col + index < 5 and state[col + index] != ' ' and state[row][col + index] != piece_color:
                            opp_count += 1
                        # update heuristic_value with considering coninous and blocking and num of possible position
                        heuristic_value += referee(count,opp_count, row , col) + extra_credit
        
        # check vertical 
        for row in range(5):
            for col in range(2):
                if state[col][row] == piece_color:
                    count = 0
                    opp_count = 0
                    for index in range(1, 4):
                        # if postion is not out of board and next has same piece
                        if col+index < 5 and state[col+index][row] == piece_color:
                            count = count + 1
                        # if adjacent postion has opp_value (blocking)
                        elif col+index < 5 and state[col+index][row] != ' ' and state[col+index][row] != piece_color:
                            opp_count += 1
                        # update heuristic_value with considering coninous and blocking and num of possible position
                        heuristic_value += referee(count,opp_count, row, col) + extra_credit
            
        # check \ diagoanl 
        for row in range(2):
            for col in range(2):
                if state[row][col] == piece_color:
                    count = 0
                    opp_count = 0
                    for index in range(1, 4):
                        # if postion is not out of board and next has same piece
                        if row + index < 5 and col + index < 5 and state[row+index][col+index] == piece_color:
                            count = count + 1
                        # if adjacent postion has opp_value (blocking)
                        elif row + index < 5 and col + index < 5 and state[row+index][col+index] != ' ' and state[row+index][col+index] != piece_color:
                            opp_count = opp_count + 1
                        # update heuristic_value with considering coninous and blocking and num of possible position
                        heuristic_value += referee(count, opp_count, row, col) + extra_credit
                    
        # check / diagonal 
        for row in range(2):
            for col in range(3,5):
                if state[row][col] == piece_color:
                    count = 0
                    opp_count = 0
                    for index in range(1, 4):
                        # if postion is not out of board and next has same piece
                        if row+index < 5 and col-index >= 0 and state[row+index][col-index] == piece_color:
                            count = count + 1
                        # if adjacent postion has opp_value (blocking)
                        elif row + index < 5 and col - index >= 0 and state[row+index][col-index] != ' ' and state[row+index][col-index] != piece_color:
                            opp_count = opp_count + 1
                        # update heuristic_value with considering coninous and blocking and num of possible position
                        heuristic_value += referee(count, opp_count, row , col) + extra_credit
        
        # check box
        for row in range(4):
            for col in range(4):
                if state[row][col] == piece_color:
                    count = 0
                    opp_count = 0
                    # this is down side position checking
                    if row+1 < 5 and state[row+1][col] == piece_color:
                        count += 1
                    # this is left side checking
                    if col+1 < 5 and state[row][col+1] == piece_color:
                        count += 1
                        # this is diagonal position checking
                    if row + 1 < 5 and col + 1 <5 and state[row+1][col+1] == piece_color:
                        count += 1
                    # this is same with referee function, the reason why i don't use referee function on it because i don't know the reason 
                    # but if i use referee, the program has infinite loop and i don't give minus point for box case becasue i thougth box position
                    # is best case for game because it can make vertical, horizontal and diagonal position
                    if count < 1: 
                        heuristic_value += 0.0
                        heuristic_value += extra_credit
                    elif count == 1:
                        heuristic_value += 0.5
                        heuristic_value += extra_credit
                    else:
                        heuristic_value += 0.75
                        heuristic_value += extra_credit

        # return heuristic_value
        return heuristic_value
    

    
    
    


    def opponent_move(self, move):
        """ Validates the opponent's next move against the internal board representation.
        You don't need to touch this code.

        Args:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.
        """
        # validate input
        if len(move) > 1:
            source_row = move[1][0]
            source_col = move[1][1]
            if source_row != None and self.board[source_row][source_col] != self.opp:
                self.print_board()
                print(move)
                raise Exception("You don't have a piece there!")
            if abs(source_row - move[0][0]) > 1 or abs(source_col - move[0][1]) > 1:
                self.print_board()
                print(move)
                raise Exception('Illegal move: Can only move to an adjacent space')
        if self.board[move[0][0]][move[0][1]] != ' ':
            raise Exception("Illegal move detected")
        # make move
        self.place_piece(move, self.opp)

    def place_piece(self, move, piece):
        """ Modifies the board representation using the specified move and piece

        Args:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.

                This argument is assumed to have been validated before this method
                is called.
            piece (str): the piece ('b' or 'r') to place on the board
        """
        if len(move) > 1:
            self.board[move[1][0]][move[1][1]] = ' '
        self.board[move[0][0]][move[0][1]] = piece

    def print_board(self):
        """ Formatted printing for the board """
        for row in range(len(self.board)):
            line = str(row)+": "
            for cell in self.board[row]:
                line += cell + " "
            print(line)
        print("   A B C D E")

    def game_value(self, state):
        """ Checks the current board status for a win condition

        Args:
        state (list of lists): either the current state of the game as saved in
            this TeekoPlayer object, or a generated successor state.

        Returns:
            int: 1 if this TeekoPlayer wins, -1 if the opponent wins, 0 if no winner

        TODO: complete checks for diagonal and box wins
        """
        # check horizontal wins
        for row in state:
            for i in range(2):
                if row[i] != ' ' and row[i] == row[i+1] == row[i+2] == row[i+3]:
                    return 1 if row[i]==self.my_piece else -1

        # check vertical wins
        for col in range(5):
            for i in range(2):
                if state[i][col] != ' ' and state[i][col] == state[i+1][col] == state[i+2][col] == state[i+3][col]:
                    return 1 if state[i][col]==self.my_piece else -1

        # TODO: check \ diagonal wins
        for row in range(2):
            for col in range(2):
                if state[row][col] != ' ' and state[row][col] == state[row+1][col+1] == state[row+2][col+2] == state[row+3][col+3]:
                    return 1 if state[row][col] == self.my_piece else -1
                
        # TODO: check / diagonal wins
        for row in range(2):
            for col in range(3,5):
                if state[row][col] != ' ' and state[row][col] == state[row+1][col-1] == state[row+2][col-2] == state[row+3][col-3]:
                    return 1 if state[row][col] == self.my_piece else -1
        # TODO: check box wins
        for row in range(4):
            for col in range(4):
                if state[row][col] != ' ' and state[row][col] == state[row][col+1] == state[row+1][col] == state[row+1][col+1]:
                    return 1 if state[row][col] == self.my_piece else -1
                
        return 0 # no winner yet
    

############################################################################
#
# THE FOLLOWING CODE IS FOR SAMPLE GAMEPLAY ONLY
#
############################################################################
def main():
    print('Hello, this is Samaritan')
    ai = TeekoPlayer()
    piece_count = 0
    turn = 0

    # drop phase
    while piece_count < 8 and ai.game_value(ai.board) == 0:

        # get the player or AI's move
        if ai.my_piece == ai.pieces[turn]:
            ai.print_board()
            move = ai.make_move(ai.board)
            ai.place_piece(move, ai.my_piece)
            print(ai.my_piece+" moved at "+chr(move[0][1]+ord("A"))+str(move[0][0]))
        else:
            move_made = False
            ai.print_board()
            print(ai.opp+"'s turn")
            while not move_made:
                player_move = input("Move (e.g. B3): ")
                while player_move[0] not in "ABCDE" or player_move[1] not in "01234":
                    player_move = input("Move (e.g. B3): ")
                try:
                    ai.opponent_move([(int(player_move[1]), ord(player_move[0])-ord("A"))])
                    move_made = True
                except Exception as e:
                    print(e)

        # update the game variables
        piece_count += 1
        turn += 1
        turn %= 2

    # move phase - can't have a winner until all 8 pieces are on the board
    while ai.game_value(ai.board) == 0:

        # get the player or AI's move
        if ai.my_piece == ai.pieces[turn]:
            ai.print_board()
            move = ai.make_move(ai.board)
            ai.place_piece(move, ai.my_piece)
            print(ai.my_piece+" moved from "+chr(move[1][1]+ord("A"))+str(move[1][0]))
            print("  to "+chr(move[0][1]+ord("A"))+str(move[0][0]))
        else:
            move_made = False
            ai.print_board()
            print(ai.opp+"'s turn")
            while not move_made:
                move_from = input("Move from (e.g. B3): ")
                while move_from[0] not in "ABCDE" or move_from[1] not in "01234":
                    move_from = input("Move from (e.g. B3): ")
                move_to = input("Move to (e.g. B3): ")
                while move_to[0] not in "ABCDE" or move_to[1] not in "01234":
                    move_to = input("Move to (e.g. B3): ")
                try:
                    ai.opponent_move([(int(move_to[1]), ord(move_to[0])-ord("A")),
                                    (int(move_from[1]), ord(move_from[0])-ord("A"))])
                    move_made = True
                except Exception as e:
                    print(e)

        # update the game variables
        turn += 1
        turn %= 2

    ai.print_board()
    if ai.game_value(ai.board) == 1:
        print("AI wins! Game over.")
    else:
        print("You win! Game over.")
        
        


if __name__ == "__main__":
    main()

