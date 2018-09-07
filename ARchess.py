import numpy as np
import cv2
import json
import chess
import argparse

mouse_x, mouse_y = 0, 0
click_x, click_y, click_flag = 0,0,False
selection_flag = False #If true, currently a grid is seleted.
last_starting  = np.array([0, 0]) #store the last starting point of the chessboard.

SELECT_A_PIECE = 0
SELECT_INVALID_GRID = 1
RESELECT_SAME_GRID = 2
VALID_MOVE = 3

##################### chess utils ########################

def get_grid_id(x, y):#get the grid id from x and y
    return 8*y + x

def get_grid_coord(id):#get x and y from grid id defined in python-chess
    return(id%8, id//8)

def peices_position(board):#get the positions where pieces exist 

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

def king_position(board):#get the position of the king belongs to current player

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

def make_move(board, x, y, last_x, last_y, promotion_type):#make a move. Move must be valid
    if promotion_type != 0:
        board.push(chess.Move(get_grid_id(last_x, last_y), get_grid_id(x, y), promotion_type))
    else:
        board.push(chess.Move(get_grid_id(last_x, last_y), get_grid_id(x, y)))

    return board.is_game_over(), board.result()

def bottom_line(board):#the row id of bottom row
    return 7 if board.turn == chess.WHITE else 0

##########################################################

######################## 3d utils ########################

def affine_transform(img, texture, corners):#transform a 2d image to the plane z=0 in 3d space 
    rows,cols,_ = img.shape
    #contours = np.mgrid[0:9,0:9].T.reshape(-1,2) - 1
    w, h, _ = texture.shape

    mask = np.zeros_like(img)

    idx = np.array([72, 0, 8, 80])

    '''

    if trun:
        idx = np.array([(x) + (y)*9, (x) + (y+1)*9, (x+1) + (y+1)*9, (x+1) + y*9])
    else:
        idx = np.array([(x+1) + (y+1)*9, (x+1) + y*9, (x) + (y)*9, (x) + (y+1)*9])

    '''
  
    pts1 = np.float32([[0,0],[0,h],[w,h],[w,0]])
    pts2 = np.int32(corners).reshape(-1,2)[idx, :]
    
    cv2.fillConvexPoly(mask, pts2, (255,255,255))
    
    M = cv2.getPerspectiveTransform(pts1,np.float32(pts2))
    dst = cv2.warpPerspective(texture,M,(cols, rows))

    mask = mask!=0

    img[mask] = dst[mask]

    return img

def draw_2d_piece(img, board, textures):#draw 2d piece on a board
    current_state = board.__str__().split('\n')
    M = cv2.getRotationMatrix2D((30, 30),180,1)
    for i in range(8):
        pieces = current_state[i].split(' ')
        for j in range(8):
            piece = pieces[j]              
            if piece != '.':
                texture = textures[piece]
                if piece.islower():
                    texture = cv2.warpAffine(texture,M,(60,60))
                
                transparent_texture = img[i*60:(i+1)*60, j*60:(j+1)*60]
                transparent_texture[texture[:,:,3] != 0] = texture[:,:,0:3][texture[:,:,3] != 0]
                img[i*60:(i+1)*60, j*60:(j+1)*60] = transparent_texture

    return img

def draw_3d_piece():
    pass

def draw_2d_board():#draw a 2d board with 2d pieces
    img = np.zeros((8*60, 8*60, 3), dtype = np.uint8)

    for i in range(8):
        for j in range(8):
            if (i + j)%2 == 1:
                img[i*60:(i+1)*60, j*60:(j+1)*60] = np.array([24,141,158])
            else:
                img[i*60:(i+1)*60, j*60:(j+1)*60] = np.array([132,223,236])

    return img

def darw_3d_board(img, board, rvec, tvec, A, corners, textures, mode = '2d', fill_grid_img = None):#draw a board in the 3d space
    if mode == 'letter':
        fontFace = cv2.FONT_HERSHEY_PLAIN

        objp = np.zeros((8*8,3), np.float32)
        objp[:,:2] = np.mgrid[0:8,0:8].T.reshape(-1,2) - 0.5

        #grid_centers = np.zeros((8*8,2), np.float32)

        grid_centers, _ =  cv2.projectPoints(objp, rvec, tvec, A, None)

        grid_centers = np.squeeze(grid_centers)

        current_state = board.__str__().split('\n')

        for i in range(8):
            pieces = current_state[i].split(' ')
            for j in range(len(pieces)):
                piece = pieces[j]
                if piece != '.':
                    text_place = tuple(grid_centers[8*(7-i) + j].astype(int).tolist())
                    cv2.putText(img = img, text = piece, org = text_place, fontFace = fontFace, fontScale = 2, color = (255, 0, 255))  

    if mode == '2d':                    
        d2_board = draw_2d_board()
        d2_board = draw_2d_piece(d2_board, board, textures)

        if fill_grid_img is not None and len(fill_grid_img[:,:,3]!=0) != 0:
            mask = fill_grid_img[:,:,3]!=0
            d2_board[mask] = d2_board[mask]//3*2 + fill_grid_img[:,:,0:3][mask]//3

        img = affine_transform(img, d2_board, corners)

    if mode == '3d':
        d2_board = draw_2d_board()
        if fill_grid_img is not None and len(fill_grid_img[:,:,3]!=0) != 0:
            mask = fill_grid_img[:,:,3]!=0
            d2_board[mask] = d2_board[mask]//3*2 + fill_grid_img[:,:,0:3][mask]//3
        img = affine_transform(img, d2_board, corners)
        draw_3d_board() #TODO: finish this

    return img

def project_to_3d(u,v,z,rvec,tvec,A):#get the 3-d coordinate of a point given the z coordinate of the point 
    R = np.zeros((3, 3))
    cv2.Rodrigues(rvec, R)
    R = np.matrix(R)
    t = tvec
    uv = np.matrix([u,v,1]).transpose()
    left_hand_side = R.getI()*A.getI()*uv
    s = float(((R.getI()*t)[2] + z)/left_hand_side[2])
    x = float((s*left_hand_side - R.getI()*t)[0])
    y = float((s*left_hand_side - R.getI()*t)[1])

    return x,y,z

def corners_reorder(img, corners, num_x, num_y):#reorder corners detected by get_extrinsic_parameters() so that the order is consistent among frames
    global last_starting

    sample1 = (corners[0] + corners[8])/2
    sample2 = (corners[48] + corners[40])/2
    
    if img[int(sample1[1]), int(sample1[0])] > 128 and img[int(sample2[1]), int(sample2[0])] > 128:#white
        corners = np.reshape(corners, (num_x, num_y, 2))
        for x in range(num_x):
            corners[x, :, :] = corners[x, ::-1, :]
        corners = np.reshape(corners, (num_x*num_y, 2))

    # if corners[0,0] > corners[-1,0]: #use the left black square as first square
    #     corners = corners[::-1,:]

    if np.linalg.norm(corners[0] - last_starting) > np.linalg.norm(corners[-1] - last_starting):
        corners = corners[::-1,:]

    last_starting[:] = corners[0]

    if np.cross(corners[0] - corners[-1], corners[0] - corners[num_x - 1]) < 0: #scanning every row then column
        corners = np.reshape(corners, (num_x, num_y, 2), order = 'F')
        corners = np.reshape(corners, (num_x*num_y, 2)) 

    return corners

def get_extrinsic_parameters(img, num_x, num_y, A):#caliberate camera
    #img should be in bgr format
 
    objp = np.zeros((num_x*num_y,3), np.float32)
    objp[:,:2] = np.mgrid[0:num_x,0:num_y].T.reshape(-1,2)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (num_x, num_y), None)

    rvecs = None
    tvecs = None

    if ret:
        corners = np.squeeze(corners)
        corners = corners_reorder(gray, corners, num_x, num_y)
        ret, rvecs, tvecs = cv2.solvePnP(objp, corners, A, None)

    if not ret:
        print('fail to get camera extrinsic parameters!')

    return ret, rvecs, tvecs, corners

def within_boundary(mouse_x,mouse_y,z,rvecs,tvecs,A):#check wherther a mouse position is within the boundary of the borad and return the x,y the mouse is selecting
    x, y, _ = project_to_3d(mouse_x,mouse_y,0,rvecs,tvecs,A)

    x, y = x+0.5, y+0.5

    if x > -0.5 and x < 7.5 and y > -0.5 and y < 7.5:
        ret = True
    else:
        ret = False

    return ret, round(x), round(y)

def fill_grid(img, x, y, contours_corners, color, alpha = 0.5):#fill a gird in 2d board
    assert img.shape[2] == 4 #an RGBA image

    #idx = np.array([(x) + (y)*9, (x) + (y+1)*9, (x+1) + (y+1)*9, (x+1) + (y)*9])
    x = x
    y = 8 - y

    idx = np.array([(x) + (y)*9, (x+1) + (y-1)*9])
    imgpts = np.int32(contours_corners).reshape(-1,2)[idx, :]

    img[imgpts[-1,1]:imgpts[0,1], imgpts[0,0]:imgpts[-1,0]] += np.array([color[0], color[1], color[2], 1])

    #img = np.add(img>>1, cv2.fillConvexPoly(img, imgpts, color)>>1)

    return img

##########################################################

##################### miscellaneous ######################

def event_processor(event,x,y,flags,param):#mouse event
    global mouse_x, mouse_y, click_x, click_y, click_flag
    if event == cv2.EVENT_MOUSEMOVE:
        mouse_x = x
        mouse_y = y

    if event == cv2.EVENT_LBUTTONDOWN:
        click_x, click_y, click_flag = x, y, True



class frame_reader(object):#read frame from camera. 
    def __init__(self, width = 1280, height = 720, camera_id = 0):
        self.width = width
        self.height = height
        self.capture = cv2.VideoCapture(camera_id)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

    def read(self):
        ret, img = self.capture.read()
        if not ret:
            print("fail to read frame!")
        return ret, img #img in gbr format

##########################################################

def main(args):
    global mouse_x, mouse_y, click_x, click_y, click_flag, selection_flag

    texture_folder = './resources/chess_piece_2d/'
    textures = {'K': cv2.imread(texture_folder + 'Chess_klt60.png', cv2.IMREAD_UNCHANGED),
    'k':cv2.imread(texture_folder + 'Chess_kdt60.png', cv2.IMREAD_UNCHANGED),
    'R':cv2.imread(texture_folder + 'Chess_rlt60.png', cv2.IMREAD_UNCHANGED),
    'r':cv2.imread(texture_folder + 'Chess_rdt60.png', cv2.IMREAD_UNCHANGED),
    'N':cv2.imread(texture_folder + 'Chess_nlt60.png', cv2.IMREAD_UNCHANGED),
    'n':cv2.imread(texture_folder + 'Chess_ndt60.png', cv2.IMREAD_UNCHANGED),
    'B':cv2.imread(texture_folder + 'Chess_blt60.png', cv2.IMREAD_UNCHANGED),
    'b':cv2.imread(texture_folder + 'Chess_bdt60.png', cv2.IMREAD_UNCHANGED),
    'Q':cv2.imread(texture_folder + 'Chess_qlt60.png', cv2.IMREAD_UNCHANGED),
    'q':cv2.imread(texture_folder + 'Chess_qdt60.png', cv2.IMREAD_UNCHANGED),
    'P':cv2.imread(texture_folder + 'Chess_plt60.png', cv2.IMREAD_UNCHANGED),
    'p':cv2.imread(texture_folder + 'Chess_pdt60.png', cv2.IMREAD_UNCHANGED)}

    selected_x, selected_y = -1, -1

    board = chess.Board()

    '''

    with open("./camera_parameter.json", 'r') as f:
        camera_parameter = json.load(f)['video']

    A = np.matrix([[camera_parameter['FocalLength'][0], 0 , camera_parameter['PrincipalPoint'][0]],
    [0, camera_parameter['FocalLength'][1], camera_parameter['PrincipalPoint'][1]],
    [0,0,1]])
    '''

    with open("./camera_parameters.json", 'r') as f:
        A = np.matrix(json.load(f))

    reader = frame_reader(camera_id = args.camera_id)

    cv2.namedWindow('img')
    cv2.setMouseCallback('img', event_processor)
    
    while True:
        ret = True
        ret, img = reader.read()
        #img = cv2.imread('./test_img/test2.jpg')
        img = cv2.resize(img, (1280,720))

        if ret:
            ret, rvecs, tvecs, corners = get_extrinsic_parameters(img, 7, 7, A)

            ''' 
            print(rvecs, tvecs)
            R = np.zeros((3, 3))
            cv2.Rodrigues(rvecs, R)
            print(R)
            cv2.imwrite('sample.jpg', img)
            return
            '''

            if ret:
                contours = np.zeros((9*9,3), np.float32)
                contours[:,:2] = np.mgrid[0:9,0:9].T.reshape(-1,2) - 1

                contours_corners, _ = cv2.projectPoints(contours, rvecs, tvecs, A, None)
                contours_corners = contours_corners.squeeze()

                contours_corners_2d = (contours[:,0:2] + 1)*60
                filled_grid_img = np.zeros((8*60, 8*60, 4), dtype = np.float32)

                #cv2.drawChessboardCorners(img, (7,7), corners, ret)

                if click_flag:
                    click_flag = False
                    ret, x, y = within_boundary(click_x,click_y,0,rvecs,tvecs,A)

                    if ret:
                        move = move_type(board, x, y, selected_x, selected_y)

                        if not selection_flag:
                            if move == SELECT_A_PIECE:
                                selection_flag = True
                                selected_x ,selected_y = x, y

                            elif move == SELECT_INVALID_GRID:
                                pass        

                        else:#selected something
                            if move == SELECT_A_PIECE:
                                selection_flag = True
                                selected_x ,selected_y = x, y

                            elif move == SELECT_INVALID_GRID:
                                selection_flag = False

                            elif move == RESELECT_SAME_GRID:
                                selection_flag = False

                            elif move == VALID_MOVE:
                                selection_flag = False
                                if board.piece_type_at(get_grid_id(selected_x, selected_y)) == chess.PAWN and y == bottom_line(board):
                                    promotion_type = chess.QUEEN
                                else:
                                    promotion_type = 0
                                    
                                is_game_over, game_result = make_move(board, x, y, selected_x, selected_y, promotion_type)

                                if is_game_over:
                                    result_img_folder = './resources/results/'
                                    if game_result == '1-0':
                                        result_img = cv2.imread(result_img_folder+'white_won.jpg')
                                    elif game_result == '0-1':
                                        result_img = cv2.imread(result_img_folder+'black_won.jpg')
                                    else:
                                        result_img = cv2.imread(result_img_folder+'tie.jpg')
                                    #img = result_img
                                    for i in range(10):
                                        cv2.imshow('img', (img//10*(10-i) + result_img//10*i))
                                        cv2.waitKey(100)

                                    cv2.imshow('img', result_img)
                                    cv2.waitKey()
                                    
                                    return 

                if selection_flag:
                    filled_grid_img = fill_grid(filled_grid_img, selected_x, selected_y, contours_corners_2d, (0,0,255))
                    for i in range(64):
                        if chess.Move(get_grid_id(selected_x, selected_y), i) in board.legal_moves or chess.Move(get_grid_id(selected_x, selected_y), i, promotion = chess.QUEEN) in board.legal_moves:
                            filled_grid_img = fill_grid(filled_grid_img, get_grid_coord(i)[0], get_grid_coord(i)[1], contours_corners_2d, (255,0,0))

                if board.is_check():
                    filled_grid_img = fill_grid(filled_grid_img, get_grid_coord(king_position(board))[0], get_grid_coord(king_position(board))[1], contours_corners_2d, (255,0,255))

                bd_flag, x, y = within_boundary(mouse_x,mouse_y,0,rvecs,tvecs,A)

                if bd_flag:
                    filled_grid_img = fill_grid(filled_grid_img, x, y, contours_corners_2d, (0,255,0))

                if len(filled_grid_img!=0) != 0:
                    mask = filled_grid_img[:,:,3]!=0

                    for i in range(3):
                        filled_grid_img[:,:,i][mask] /= filled_grid_img[:,:,3][mask]

                    filled_grid_img[:,:,3][mask] = 255
                    filled_grid_img = filled_grid_img.astype(np.uint8)

                piece_type = '3d' if args.d3_piece else '2d'

                img = darw_3d_board(img, board, rvecs, tvecs, A, contours_corners, textures, piece_type, filled_grid_img)

                cv2.imshow('img', img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
            else:
                cv2.imshow('img', img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ARchess parameters')
    parser.add_argument('--camera-id', default=0, type=int, help='id of the camera you want to use. Start from 0. Try different value if you have multiple cameras')
    parser.add_argument('--d3-piece', action='store_true', help='Display 3d pieces') #to be finished
    main(parser.parse_args())
        






