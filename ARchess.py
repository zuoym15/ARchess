import argparse
import json

import chess
import cv2
import glfw
import numpy as np
from OpenGL.GL import *

import utils_3d
import utils_chess
from gl.models import BackGroundImage
from gl.piece3d import draw_3d_piece, load_models
from frame_reader import FrameReader

mouse_x, mouse_y = None, None
selected_x, selected_y = -1, -1
selection_flag = False  # If true, currently a grid is seleted.

board = chess.Board()
is_game_over, game_result = False, None

rvecs, tvecs = None, None
with open("./camera_parameters.json", 'r') as f:
    A = np.matrix(json.load(f))


def cursor_pos_callback(window, x, y):
    global mouse_x, mouse_y
    if rvecs is not None:
        bd_flag, mouse_x, mouse_y = utils_3d.within_boundary(x, y, 0, rvecs, tvecs, A)
        if not bd_flag: mouse_x = mouse_y = None


def mouse_button_callback(window, button, action, mods):
    global selected_x, selected_y, selection_flag, is_game_over, game_result
    if button == glfw.MOUSE_BUTTON_LEFT and action == glfw.PRESS:
        if mouse_x is not None:
            move = utils_chess.move_type(board, mouse_x, mouse_y, selected_x, selected_y, selection_flag)

            if not selection_flag:
                if move == utils_chess.SELECT_A_PIECE:
                    selection_flag = True
                    selected_x, selected_y = mouse_x, mouse_y
                elif move == utils_chess.SELECT_INVALID_GRID:
                    pass
            else:  # selected something
                if move == utils_chess.SELECT_A_PIECE:
                    selection_flag = True
                    selected_x, selected_y = mouse_x, mouse_y
                elif move == utils_chess.SELECT_INVALID_GRID:
                    selection_flag = False
                elif move == utils_chess.RESELECT_SAME_GRID:
                    selection_flag = False
                elif move == utils_chess.VALID_MOVE:
                    selection_flag = False
                    if board.piece_type_at(utils_chess.get_grid_id(selected_x,
                                                                   selected_y)) == chess.PAWN and selected_y == utils_chess.bottom_line(
                        board):
                        promotion_type = chess.QUEEN
                    else:
                        promotion_type = 0

                    is_game_over, game_result = utils_chess.make_move(board, mouse_x, mouse_y, selected_x, selected_y,
                                                                      promotion_type)


def main(args):
    global is_game_over, game_result, rvecs, tvecs

    reader = FrameReader(camera_id=args.camera_id)

    image = BackGroundImage(1280, 720)

    while not glfw.window_should_close(window):
        try:
            if is_game_over:
                result_img_folder = './resources/results/'
                if game_result == '1-0':
                    img = cv2.imread(result_img_folder + 'white_won.jpg')
                elif game_result == '0-1':
                    img = cv2.imread(result_img_folder + 'black_won.jpg')
                else:
                    img = cv2.imread(result_img_folder + 'tie.jpg')

            else:

                img = reader.read()

                rvecs, tvecs, corners = utils_3d.get_extrinsic_parameters(img, 7, 7, A)

                contours = np.zeros((9 * 9, 3), np.float32)
                contours[:, :2] = np.mgrid[0:9, 0:9].T.reshape(-1, 2) - 1

                contours_corners, _ = cv2.projectPoints(contours, rvecs, tvecs, A, None)
                contours_corners = contours_corners.squeeze()

                contours_corners_2d = (contours[:, 0:2] + 1) * 60
                filled_grid_img = np.zeros((8 * 60, 8 * 60, 4), dtype=np.float32)

                if selection_flag:
                    filled_grid_img = utils_3d.fill_grid(filled_grid_img, selected_x, selected_y,
                                                         contours_corners_2d,
                                                         (0, 0, 255))
                    for i in range(64):
                        if chess.Move(utils_chess.get_grid_id(selected_x, selected_y),
                                      i) in board.legal_moves or chess.Move(
                            utils_chess.get_grid_id(selected_x, selected_y), i,
                            promotion=chess.QUEEN) in board.legal_moves:
                            filled_grid_img = utils_3d.fill_grid(filled_grid_img, utils_chess.get_grid_coord(i)[0],
                                                                 utils_chess.get_grid_coord(i)[1],
                                                                 contours_corners_2d, (255, 0, 0))

                if board.is_check():
                    filled_grid_img = utils_3d.fill_grid(filled_grid_img,
                                                         utils_chess.get_grid_coord(
                                                             utils_chess.king_position(board))[
                                                             0],
                                                         utils_chess.get_grid_coord(
                                                             utils_chess.king_position(board))[
                                                             1], contours_corners_2d,
                                                         (255, 0, 255))

                if mouse_x is not None:
                    filled_grid_img = utils_3d.fill_grid(filled_grid_img, mouse_x, mouse_y, contours_corners_2d,
                                                         (0, 255, 0))

                if len(filled_grid_img != 0) != 0:
                    mask = filled_grid_img[:, :, 3] != 0

                    for i in range(3):
                        filled_grid_img[:, :, i][mask] /= filled_grid_img[:, :, 3][mask]

                    filled_grid_img[:, :, 3][mask] = 255
                    filled_grid_img = filled_grid_img.astype(np.uint8)

                piece_type = '3d' if args.d3_piece else '2d'

                img = utils_3d.draw_3d_board(img, board, rvecs, tvecs, A, contours_corners, piece_type,
                                             filled_grid_img)

        except RuntimeError as e:
            print(e)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            image.set_image(img)
            image.render()
        else:
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            image.set_image(img)
            image.render()
            if args.d3_piece and not is_game_over:
                draw_3d_piece(board, rvecs, tvecs, A)
        finally:
            glfw.swap_buffers(window)
            glfw.poll_events()

    glfw.terminate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ARchess parameters')
    parser.add_argument('--camera-id', default=0, type=int,
                        help='id of the camera you want to use. Start from 0. Try different value if you have multiple cameras')
    parser.add_argument('--d3-piece', action='store_true', help='Display 3d pieces')  # to be finished

    if not glfw.init():
        exit(-1)

    window = glfw.create_window(1280, 720, "My OpenGL window", None, None)

    if not window:
        glfw.terminate()
        exit(-2)

    glfw.make_context_current(window)
    # glViewport(0, 0, 1280, 720)

    glfw.set_cursor_pos_callback(window, cursor_pos_callback)
    glfw.set_mouse_button_callback(window, mouse_button_callback)

    glClearColor(0, 0, 0, 1)

    load_models()

    main(parser.parse_args())
