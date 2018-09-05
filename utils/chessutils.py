import numpy as np
import cv2
import chess


def get_grid_id(x, y): #get the grid id from x and y
    return 8*y + x

def get_grid_coord(id): #get x and y from grid id defined in python-chess
    return(id%8, id//8)

def peices_position(board): #get the positions where pieces exist 

    def isupper(string):
        return string.isupper()

    def islower(string):
        return string.islower()

    creteria = isupper if board.turn == chess.WHITE else islower
    result = []
    current_state = board.__str__().split('\n')
    for i in range(8):
        pieces = current_state[i].split(' ')
        for j in range(len(pieces)):
            piece = pieces[j]
            if creteria(piece):
                result.append(8*(7-i) + j)

    return result


def king_position(board): #get the position of the king belongs to current player

    creteria = 'K' if board.turn == chess.WHITE else 'k'
    current_state = board.__str__().split('\n')
    for i in range(8):
        pieces = current_state[i].split(' ')
        for j in range(len(pieces)):
            piece = pieces[j]
            if creteria == piece:
                return 8*(7-i) + j

def move_type(board, x, y, last_x, last_y):#check whether a move is valid, etc.
    global selection_flag

    if not selection_flag: #now no grid has been selected
        if get_grid_id(x, y) in peices_position(board):
            return SELECT_A_PIECE
        return SELECT_INVALID_GRID
        
    else:
        if x == last_x and y == last_y:
            return RESELECT_SAME_GRID
        if get_grid_id(x, y) in peices_position(board):
            return SELECT_A_PIECE
        if chess.Move(get_grid_id(last_x, last_y), get_grid_id(x, y)) in board.legal_moves or chess.Move(get_grid_id(last_x, last_y), get_grid_id(x, y), promotion = chess.QUEEN) in board.legal_moves:
            return VALID_MOVE
        return SELECT_INVALID_GRID 

def make_move(board, x, y, last_x, last_y, promotion_type):
    if promotion_type != 0:
        board.push(chess.Move(get_grid_id(last_x, last_y), get_grid_id(x, y), promotion_type))
    else:
        board.push(chess.Move(get_grid_id(last_x, last_y), get_grid_id(x, y)))

    return board.is_game_over(), board.result()

def bottom_line(board):
    return 7 if board.turn == chess.WHITE else 0