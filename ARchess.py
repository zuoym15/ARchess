import argparse
import json

import chess
import cv2
import numpy as np

import utils_3d
import utils_chess

mouse_x, mouse_y = None, None
selected_x, selected_y = -1, -1
selection_flag = False  # If true, currently a grid is seleted.

WINDOW_NAME = 'img'

board = chess.Board()
is_game_over, game_result = False, None

rvecs, tvecs = None, None
with open("./camera_parameters.json", 'r') as f:
    A = np.matrix(json.load(f))


def event_processor(event, x, y, flags, param):
    """
    mouse event
    """
    global mouse_x, mouse_y, selected_x, selected_y, selection_flag, is_game_over, game_result
    if event == cv2.EVENT_MOUSEMOVE:
        if rvecs != None:
            bd_flag, mouse_x, mouse_y = utils_3d.within_boundary(x, y, 0, rvecs, tvecs, A)
            if not bd_flag: mouse_x = mouse_y = None

    if event == cv2.EVENT_LBUTTONDOWN:
        ret, x, y = utils_3d.within_boundary(x, y, 0, rvecs, tvecs, A)

        if ret:
            move = utils_chess.move_type(board, x, y, selected_x, selected_y, selection_flag)

            if not selection_flag:
                if move == utils_chess.SELECT_A_PIECE:
                    selection_flag = True
                    selected_x, selected_y = x, y
                elif move == utils_chess.SELECT_INVALID_GRID:
                    pass
            else:  # selected something
                if move == utils_chess.SELECT_A_PIECE:
                    selection_flag = True
                    selected_x, selected_y = x, y
                elif move == utils_chess.SELECT_INVALID_GRID:
                    selection_flag = False
                elif move == utils_chess.RESELECT_SAME_GRID:
                    selection_flag = False
                elif move == utils_chess.VALID_MOVE:
                    selection_flag = False
                    if board.piece_type_at(utils_chess.get_grid_id(selected_x,
                                                                   selected_y)) == chess.PAWN and y == utils_chess.bottom_line(
                        board):
                        promotion_type = chess.QUEEN
                    else:
                        promotion_type = 0

                    is_game_over, game_result = utils_chess.make_move(board, x, y, selected_x, selected_y,
                                                                      promotion_type)


class FrameReader(object):
    """
    read frame from camera.
    """

    def __init__(self, width=1280, height=720, camera_id=0):
        self.width = width
        self.height = height
        self.capture = cv2.VideoCapture(camera_id)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

    def read(self):
        ret, img = self.capture.read()
        if not ret:
            print("fail to read frame!")
        return ret, img  # img in gbr format


def main(args):
    global is_game_over, game_result, rvecs, tvecs

    reader = FrameReader(camera_id=args.camera_id)

    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, event_processor)

    while True:
        ret, img = reader.read()
        img = cv2.resize(img, (1280, 720))

        if ret:
            ret, rvecs, tvecs, corners = utils_3d.get_extrinsic_parameters(img, 7, 7, A)

            if ret:
                contours = np.zeros((9 * 9, 3), np.float32)
                contours[:, :2] = np.mgrid[0:9, 0:9].T.reshape(-1, 2) - 1

                contours_corners, _ = cv2.projectPoints(contours, rvecs, tvecs, A, None)
                contours_corners = contours_corners.squeeze()

                contours_corners_2d = (contours[:, 0:2] + 1) * 60
                filled_grid_img = np.zeros((8 * 60, 8 * 60, 4), dtype=np.float32)

                if is_game_over:
                    result_img_folder = './resources/results/'
                    if game_result == '1-0':
                        result_img = cv2.imread(result_img_folder + 'white_won.jpg')
                    elif game_result == '0-1':
                        result_img = cv2.imread(result_img_folder + 'black_won.jpg')
                    else:
                        result_img = cv2.imread(result_img_folder + 'tie.jpg')
                    # img = result_img
                    for i in range(10):
                        cv2.imshow(WINDOW_NAME, (img // 10 * (10 - i) + result_img // 10 * i))
                        cv2.waitKey(100)

                    cv2.imshow(WINDOW_NAME, result_img)
                    cv2.waitKey()
                    return

                if selection_flag:
                    filled_grid_img = utils_3d.fill_grid(filled_grid_img, selected_x, selected_y, contours_corners_2d,
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
                                                         utils_chess.get_grid_coord(utils_chess.king_position(board))[
                                                             0],
                                                         utils_chess.get_grid_coord(utils_chess.king_position(board))[
                                                             1], contours_corners_2d,
                                                         (255, 0, 255))

                if mouse_x is not None:
                    filled_grid_img = utils_3d.fill_grid(filled_grid_img, x, y, contours_corners_2d, (0, 255, 0))

                if len(filled_grid_img != 0) != 0:
                    mask = filled_grid_img[:, :, 3] != 0

                    for i in range(3):
                        filled_grid_img[:, :, i][mask] /= filled_grid_img[:, :, 3][mask]

                    filled_grid_img[:, :, 3][mask] = 255
                    filled_grid_img = filled_grid_img.astype(np.uint8)

                piece_type = '3d' if args.d3_piece else '2d'

                img = utils_3d.draw_3d_board(img, board, rvecs, tvecs, A, contours_corners, piece_type, filled_grid_img)

                cv2.imshow(WINDOW_NAME, img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            else:
                cv2.imshow(WINDOW_NAME, img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ARchess parameters')
    parser.add_argument('--camera-id', default=1, type=int,
                        help='id of the camera you want to use. Start from 0. Try different value if you have multiple cameras')
    parser.add_argument('--d3-piece', action='store_true', help='Display 3d pieces')  # to be finished
    main(parser.parse_args())
