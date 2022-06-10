'''
Fonte: https://opencv24-python-tutorials.readthedocs.io/en/stable/py_tutorials/py_calib3d/py_calibration/py_calibration.html
'''

import cv2 as cv
import numpy as np
import glob
import os
import json

class CameraCalibration:

    def __init__(self, fpath, viewCalibration=True):
        self.fpath = fpath
        self.viewCalibration = viewCalibration
        self.K = []
        self.D = []

    def dump_param(self, fpath, K, D):
        if not os.path.exists(fpath):
            directory = fpath.split('/')[0]
            os.makedirs(directory)
        self.K = K
        self.D = D
        data = dict()
        K = K.tolist()
        D = D.tolist()
        data['K'] = K
        data['D'] = D
        to_json = json.dumps(data, indent=4)
        with open(fpath, 'w') as f:
            f.write(to_json)

    def calibracao(self, imagepath):
        # Características de imagem e de padrão de calibração
        board_size = (10, 7) # ajustar conforme o padrão de calibração

        # criterio de parada
        test1 = cv.TERM_CRITERIA_EPS
        test2 = cv.TERM_CRITERIA_MAX_ITER
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # Preparando pontos de objeto
        objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)

        # Arrays para armazenar pontos 2D e pontos 3D em todas as imagens
        obj_points = [] # 3D (mundo real)
        img_points = [] # 2D (plano imagem)

        # Chamada das imagens na pasta
        images = glob.glob(os.path.join(imagepath, '*'))

        for fname in images:
            img = cv.imread(fname)
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            ret, corners = cv.findChessboardCorners(gray, board_size, None)
            if ret == True:
                obj_points.append(objp)

                corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria) # Parâmetros default
                img_points.append(corners2)

                # Desenho e reprodução dos corners
                if self.viewCalibration:
                    orig = img.copy()
                    img = cv.drawChessboardCorners(img, board_size, corners2, ret)
                    cv.putText(img, 'Pressione ESC para abortar.', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
                    cv.namedWindow('img',cv.WINDOW_NORMAL)
                    cv.imshow('img', img)
                    k = cv.waitKey(500)
                    if k == 27:
                        exit()

        cv.destroyAllWindows()

        # Calibração
        ret, camera_matrix, distort, rvecs, tvecs = cv.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)

        print(f"Matriz da câmera:\n {camera_matrix}\n")
        print(f"Parâmetros de distorção:\n {distort}\n")

        # Correção de distorção
        self.dump_param(self.fpath, camera_matrix, distort)
        h, w = orig.shape[:2]
        new_camera_matrix, roi = cv.getOptimalNewCameraMatrix(self.K, self.D, (w, h), 1, (w, h))
        distort = cv.undistort(orig, self.K, self.D, None, new_camera_matrix)

        # Cortando a imagem
        x1, y1, w1, h1 = roi
        distort = distort[y1:(y1 + h1), x1:(x1 + w1)]
        distort = cv.resize(distort, (w, h))

        orig = cv.drawChessboardCorners(orig, board_size, corners2, True)
        
        if self.viewCalibration:
            im_h = cv.hconcat([orig, distort])
            cv.putText(im_h, 'Press Any Key to Continue', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
            cv.imshow('result', im_h)
            cv.waitKey(0)

        # Correção de distorção com Remapping
        # mapx, mapy = cv.initUndistortRectifyMap(camera_matrix, dist, None, new_camera_matrix, (w, h), 5)
        # distort = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)

        # Cortando a imagem
        # x, y, w, h = roi
        # distort = distort[y:(y + h), x:(x + w)]
        # cv.imwrite(..., distort)

        # Erro de reprojeção
        mean_error = 0

        for i in range(len(obj_points)):
            img_points2, _ = cv.projectPoints(obj_points[i], rvecs[i], tvecs[i], camera_matrix, distort)
            error = cv.norm(img_points[i], img_points2, cv.NORM_L2)/len(img_points2)
            mean_error += error

        print(f"Total error: {mean_error/len(obj_points)}")

    def calib_param(self):
        fpath = self.fpath
        with open(fpath) as f:
            data = json.load(f)
        self.K = np.array(data['K'])
        self.D = np.array(data['D'])
        return self.K, self.D