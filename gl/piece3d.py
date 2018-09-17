from .matrix4x4 import *
from .models import OBJModel
import numpy as np

pieces_models = {}


def load_models():
    global pieces_models
    pieces_models = {'Q': OBJModel('resources/chess_piece_3d/queen.obj'),
                     'R': OBJModel('resources/chess_piece_3d/rock.obj'),
                     'N': OBJModel('resources/chess_piece_3d/knight.obj'),
                     'B': OBJModel('resources/chess_piece_3d/bishop.obj'),
                     'K': OBJModel('resources/chess_piece_3d/king.obj'),
                     'P': OBJModel('resources/chess_piece_3d/pawn.obj')
                     }


def draw_3d_piece(board, rvec, tvec, A):
    scales = {
    'K': 1.75,
    'R': 1.0,
    'N': 1.2,
    'B': 1.5,
    'Q': 1.6,
    'P': 1.0,
    }

    camera_view = translate(tvec) @ rodrigues(rvec)
    light_source = - np.linalg.inv(rodrigues(rvec)) @ np.append(tvec, 1)
    light_source = light_source / np.linalg.norm(light_source) * 5

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
                piece_scale = scales[piece.upper()]
                pieces_models[piece.upper()].render(
                    translate([(1/piece_scale) * (j - 0.5), (1/piece_scale) * (-7 + i + 0.5) * (1 if piece.isupper() else -1) , 0]),
                    camera_view @ scale([piece_scale, piece_scale, piece_scale]) @ flip() if piece.isupper() else camera_view @ scale([piece_scale, piece_scale, piece_scale]),
                    camera_perspective,
                    light_source,
                    np.array([1, 1, 1]) if piece.isupper() else np.array([0, 0, 0]))
