from .matrix4x4 import *
from .models import OBJModel

pieces_models = {}


def load_models():
    global pieces_models
    pieces_models = {'K': OBJModel('resources/chess_piece_3d/queen.obj'),
                     'R': OBJModel('resources/chess_piece_3d/rock.obj'),
                     'N': OBJModel('resources/chess_piece_3d/knight.obj'),
                     'B': OBJModel('resources/chess_piece_3d/bishop.obj'),
                     'Q': OBJModel('resources/chess_piece_3d/king.obj'),
                     'P': OBJModel('resources/chess_piece_3d/pawn.obj')
                     }


def draw_3d_piece(board, rvec, tvec, A):
    camera_view = scales([0.5, 0.5, 0.5]) @ translate(tvec) @ rodrigues(rvec)
    
    focal = A[0, 0], A[1, 1]
    principal = A[0, 2], A[1, 2]
    camera_perspective = intrinsics_to_perspective(focal,
                                                   principal,
                                                   0.1, 100,
                                                   1280, 720)

    current_state = board.__str__().split('\n')
    for i in range(8):
        pieces = current_state[i].split(' ')
        for j in range(8):
            piece = pieces[j]
            if piece != '.':
                pieces_models[piece.upper()].render(
                    translate([j - 0.5, 7 - i - 0.5, 0]),
                    camera_view,
                    camera_perspective,
                    [10.0, 10.0, 10.0],
                    np.array([1, 1, 1]) if piece.isupper() else np.array([0, 0, 0]))
