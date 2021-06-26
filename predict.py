'''
predict.py有几个注意点
1、无法进行批量预测，如果想要批量预测，可以利用os.listdir()遍历文件夹，利用Image.open打开图片文件进行预测。
2、如果想要保存，利用r_image.save("img.jpg")即可保存。
3、如果想要获得框的坐标，可以进入detect_image函数，读取top,left,bottom,right这四个值。
4、如果想要截取下目标，可以利用获取到的top,left,bottom,right这四个值在原图上利用矩阵的方式进行截取。
'''
from PIL import Image
from yolo import YOLO
import cv2
import numpy as np
import torch

from models.defaults import _C as cfg
from models.hrnet import HighResolutionNet
from utils.hr_utils.transform import keypoints_from_heatmaps, get_transform
from utils.hr_utils.utils import show_image

yolo = YOLO()
cam = cv2.VideoCapture('hands.mp4')
while True:
    ret, frame = cam.read()
    if ret:
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        r_image, box = yolo.detect_image(image)
        img = cv2.cvtColor(np.asarray(r_image), cv2.COLOR_RGB2BGR)
        cv2.imshow('img', img)
        cv2.waitKey(1)
    else:
        print('Video Open Failed')
        break
