import numpy as np
import cv2


def affine_transform(img, texture, corners):  # transform a 2d image to the plane z=0 in 3d space
    rows, cols, _ = img.shape
    # contours = np.mgrid[0:9,0:9].T.reshape(-1,2) - 1
    w, h, _ = texture.shape

    mask = np.zeros_like(img)

    idx = np.array([72, 0, 8, 80])

    '''

    if trun:
        idx = np.array([(x) + (y)*9, (x) + (y+1)*9, (x+1) + (y+1)*9, (x+1) + y*9])
    else:
        idx = np.array([(x+1) + (y+1)*9, (x+1) + y*9, (x) + (y)*9, (x) + (y+1)*9])

    '''

    pts1 = np.float32([[0, 0], [0, h], [w, h], [w, 0]])
    pts2 = np.int32(corners).reshape(-1, 2)[idx, :]

    cv2.fillConvexPoly(mask, pts2, (255, 255, 255))

    M = cv2.getPerspectiveTransform(pts1, np.float32(pts2))
    dst = cv2.warpPerspective(texture, M, (cols, rows))

    mask = mask != 0

    img[mask] = dst[mask]

    return img


def draw_2d_piece(img, board, textures):  # draw 2d piece on a board
    current_state = board.__str__().split('\n')
    M = cv2.getRotationMatrix2D((30, 30), 180, 1)
    for i in range(8):
        pieces = current_state[i].split(' ')
        for j in range(8):
            piece = pieces[j]
            if piece != '.':
                texture = textures[piece]
                if piece.islower():
                    texture = cv2.warpAffine(texture, M, (60, 60))

                transparent_texture = img[i * 60:(i + 1) * 60, j * 60:(j + 1) * 60]
                transparent_texture[texture[:, :, 3] != 0] = texture[:, :, 0:3][texture[:, :, 3] != 0]
                img[i * 60:(i + 1) * 60, j * 60:(j + 1) * 60] = transparent_texture

    return img


def draw_3d_piece(rvec, tvec, A):
    pass


def draw_2d_board():  # draw a 2d board with 2d pieces
    img = np.zeros((8 * 60, 8 * 60, 3), dtype=np.uint8)

    for i in range(8):
        for j in range(8):
            if (i + j) % 2 == 1:
                img[i * 60:(i + 1) * 60, j * 60:(j + 1) * 60] = np.array([24, 141, 158])
            else:
                img[i * 60:(i + 1) * 60, j * 60:(j + 1) * 60] = np.array([132, 223, 236])

    return img


texture_folder = './resources/chess_piece_2d/'
textures = {'K': cv2.imread(texture_folder + 'Chess_klt60.png', cv2.IMREAD_UNCHANGED),
            'k': cv2.imread(texture_folder + 'Chess_kdt60.png', cv2.IMREAD_UNCHANGED),
            'R': cv2.imread(texture_folder + 'Chess_rlt60.png', cv2.IMREAD_UNCHANGED),
            'r': cv2.imread(texture_folder + 'Chess_rdt60.png', cv2.IMREAD_UNCHANGED),
            'N': cv2.imread(texture_folder + 'Chess_nlt60.png', cv2.IMREAD_UNCHANGED),
            'n': cv2.imread(texture_folder + 'Chess_ndt60.png', cv2.IMREAD_UNCHANGED),
            'B': cv2.imread(texture_folder + 'Chess_blt60.png', cv2.IMREAD_UNCHANGED),
            'b': cv2.imread(texture_folder + 'Chess_bdt60.png', cv2.IMREAD_UNCHANGED),
            'Q': cv2.imread(texture_folder + 'Chess_qlt60.png', cv2.IMREAD_UNCHANGED),
            'q': cv2.imread(texture_folder + 'Chess_qdt60.png', cv2.IMREAD_UNCHANGED),
            'P': cv2.imread(texture_folder + 'Chess_plt60.png', cv2.IMREAD_UNCHANGED),
            'p': cv2.imread(texture_folder + 'Chess_pdt60.png', cv2.IMREAD_UNCHANGED)}


def draw_3d_board(img, board, rvec, tvec, A, corners, mode='2d',
                  fill_grid_img=None):  # draw a board in the 3d space
    if mode == 'letter':
        fontFace = cv2.FONT_HERSHEY_PLAIN

        objp = np.zeros((8 * 8, 3), np.float32)
        objp[:, :2] = np.mgrid[0:8, 0:8].T.reshape(-1, 2) - 0.5

        # grid_centers = np.zeros((8*8,2), np.float32)

        grid_centers, _ = cv2.projectPoints(objp, rvec, tvec, A, None)

        grid_centers = np.squeeze(grid_centers)

        current_state = board.__str__().split('\n')

        for i in range(8):
            pieces = current_state[i].split(' ')
            for j in range(len(pieces)):
                piece = pieces[j]
                if piece != '.':
                    text_place = tuple(grid_centers[8 * (7 - i) + j].astype(int).tolist())
                    cv2.putText(img=img, text=piece, org=text_place, fontFace=fontFace, fontScale=2,
                                color=(255, 0, 255))

    if mode == '2d':
        d2_board = draw_2d_board()
        d2_board = draw_2d_piece(d2_board, board, textures)

        if fill_grid_img is not None and len(fill_grid_img[:, :, 3] != 0) != 0:
            mask = fill_grid_img[:, :, 3] != 0
            d2_board[mask] = d2_board[mask] // 3 * 2 + fill_grid_img[:, :, 0:3][mask] // 3

        img = affine_transform(img, d2_board, corners)

    if mode == '3d':
        d2_board = draw_2d_board()
        if fill_grid_img is not None and len(fill_grid_img[:, :, 3] != 0) != 0:
            mask = fill_grid_img[:, :, 3] != 0
            d2_board[mask] = d2_board[mask] // 3 * 2 + fill_grid_img[:, :, 0:3][mask] // 3

        img = affine_transform(img, d2_board, corners)

    return img


def project_to_3d(u, v, z, rvec, tvec, A):  # get the 3-d coordinate of a point given the z coordinate of the point
    R = np.zeros((3, 3))
    cv2.Rodrigues(rvec, R)
    R = np.matrix(R)
    t = tvec
    uv = np.matrix([u, v, 1]).transpose()
    left_hand_side = R.getI() * A.getI() * uv
    s = float(((R.getI() * t)[2] + z) / left_hand_side[2])
    x = float((s * left_hand_side - R.getI() * t)[0])
    y = float((s * left_hand_side - R.getI() * t)[1])

    return x, y, z


last_starting = np.array([0, 0])  # store the last starting point of the chessboard.


def corners_reorder(img, corners, num_x, num_y):  # reorder corners detected by get_extrinsic_parameters() so that the order is consistent among frames
    global last_starting

    mask = np.zeros_like(img, dtype = np.uint8)
    cv2.fillConvexPoly(mask, np.int32(corners[[0, 6, 48, 42],:]), (255, ))
    avg_color = np.average(img[mask!=0]) #take the average color of chessborad as threshold

    sample1 = (corners[0] + corners[8]) / 2
    sample2 = (corners[48] + corners[40]) / 2

    if img[int(sample1[1]), int(sample1[0])] > avg_color and img[int(sample2[1]), int(sample2[0])] > avg_color:  # white
        corners = np.reshape(corners, (num_x, num_y, 2))
        for x in range(num_x):
            corners[x, :, :] = corners[x, ::-1, :]
        corners = np.reshape(corners, (num_x * num_y, 2))

    # if corners[0,0] > corners[-1,0]: #use the left black square as first square
    #     corners = corners[::-1,:]

    if np.linalg.norm(corners[0] - last_starting) > np.linalg.norm(corners[-1] - last_starting):
        corners = corners[::-1, :]

    last_starting[:] = corners[0]

    if np.cross(corners[0] - corners[-1], corners[0] - corners[num_x - 1]) < 0:  # scanning every row then column
        corners = np.reshape(corners, (num_x, num_y, 2), order='F')
        corners = np.reshape(corners, (num_x * num_y, 2))

    return corners


def get_extrinsic_parameters(img, num_x, num_y, A):  # caliberate camera
    # img should be in bgr format

    objp = np.zeros((num_x * num_y, 3), np.float32)
    objp[:, :2] = np.mgrid[0:num_x, 0:num_y].T.reshape(-1, 2)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (num_x, num_y), None)
    if not ret:
        raise RuntimeError('fail to find chessboard!')

    corners = np.squeeze(corners)
    corners = corners_reorder(gray, corners, num_x, num_y)
    ret, rvecs, tvecs = cv2.solvePnP(objp, corners, A, None)

    if not ret:
        raise RuntimeError('fail to get camera extrinsic parameters!')

    return rvecs, tvecs, corners


def within_boundary(mouse_x, mouse_y, z, rvecs, tvecs,
                    A):  # check wherther a mouse position is within the boundary of the borad and return the x,y the mouse is selecting
    x, y, _ = project_to_3d(mouse_x, mouse_y, 0, rvecs, tvecs, A)

    x, y = x + 0.5, y + 0.5

    if x > -0.5 and x < 7.5 and y > -0.5 and y < 7.5:
        ret = True
    else:
        ret = False

    return ret, round(x), round(y)


def fill_grid(img, x, y, contours_corners, color, alpha=0.5):  # fill a gird in 2d board
    assert img.shape[2] == 4  # an RGBA image

    # idx = np.array([(x) + (y)*9, (x) + (y+1)*9, (x+1) + (y+1)*9, (x+1) + (y)*9])
    x = x
    y = 8 - y

    idx = np.array([(x) + (y) * 9, (x + 1) + (y - 1) * 9])
    imgpts = np.int32(contours_corners).reshape(-1, 2)[idx, :]

    img[imgpts[-1, 1]:imgpts[0, 1], imgpts[0, 0]:imgpts[-1, 0]] += np.array([color[0], color[1], color[2], 1])

    # img = np.add(img>>1, cv2.fillConvexPoly(img, imgpts, color)>>1)

    return img
