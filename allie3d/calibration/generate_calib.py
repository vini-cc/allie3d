import argparse
import cv2 as cv
import ast
from xml.etree import ElementTree as et

from CameraCalibration.CameraCalibration import CameraCalibration

parser = argparse.ArgumentParser(description='Camera Calibration Procedure')
parser.add_argument('--xml', help='Path to XML parameters for model excecution')
args = parser.parse_args()

tree = et.parse(args.xml)
root = tree.getroot()

xmlfile = root.findall('camera')

for params in xmlfile:
    phone_brand = params.find('phoneBrand').text
    phone_model = params.find('phoneModel').text
    calibration_file = params.find('calibrationFile').text
    images_folder = params.find('imagesFolder').text
    view_calibration = ast.literal_eval(params.find('viewCalibration').text)

cam_cal = CameraCalibration(calibration_file, view_calibration)
cam_cal.calibracao(images_folder)