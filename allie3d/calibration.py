'''
Fonte: https://opencv24-python-tutorials.readthedocs.io/en/stable/py_tutorials/py_calib3d/py_calibration/py_calibration.html
'''

import cv2 as cv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pathlib as Path
import glob

p = Path()

board_size = (20, 20) # ajustar conforme a calibracao
frame_size = (1440, 1440) # ajustar conforme a calibracao

# criterio de parada
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Preparando pontos de objeto
objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)

# Arrays para armazenar pontos 2D e pontos 3D em todas as imagens
obj_points = [] # 3D (mundo real)
img_points = [] # 2D (plano imagem)

images = glob.glob('*.png')

for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    ret, corners = cv.findChessboardCorners(gray, board_size, None)

    if ret == True:

        obj_points.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria) # Parâmetros default
        img_points.append(corners2) # Há duas maneiras de fazer essa parte - corners e corners2. Testar diferenças.

        # Desenho e reprodução dos corners
        cv.drawChessboardCorners(img, board_size, corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(1000)

cv.destroyAllWindows()