import cgitb
import os  # 可以用来打开文件
import sys  #
from datetime import datetime
import time as t1
from time import time  # 用于计算时间差 可以用来计算一个模块运行的时间
import torch
from yolo import YOLO
from models.defaults import _C as cfg
from models.hrnet import HighResolutionNet
from utils.hr_utils.transform import keypoints_from_heatmaps, get_transform
from utils.hr_utils.utils import show_image

import cv2
import numpy as np
from PIL import Image
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import *
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox, QFileDialog, QInputDialog

from qt_gui import ui_main_window  # 主窗体ui代码

cgitb.enable(format='text')

t = time()


# 显示成功消息框
def push_msg(success_text):
    success_msg = QMessageBox(QMessageBox.Question, '提示', success_text)
    success_msg.setStandardButtons(QMessageBox.Ok)
    success_msg.setIcon(1)
    success_msg.setGeometry(800, 500, 0, 0)
    success_msg.exec_()


class MyMainWindow(QMainWindow, ui_main_window):
    signal = pyqtSignal()  # 初始化信号  为了实现双重界面

    def __init__(self, parent=None):
        super(MyMainWindow, self).__init__(parent)
        self.setupUi(self)
        self.source = 0
        # self.timer_camera = QTimer()  # 需要定时器刷新摄像头界面
        self.cap = cv2.VideoCapture()
        self.success_text = ''
        self.pic_path = ''
        self.video_path = ''
        # 信号槽设置  ------------------------------
        self.button_gesture_detection.clicked.connect(self.gesture_detection)  # 手势识别
        self.button_gesture_recognition.clicked.connect(self.gesture_recognition)  # 手势识别
        self.button_gesture_score.clicked.connect(self.hand_put_right)  # 口罩切割 连接函数
        self.show()

    def open_pic(self):
        print('图片')
        pic_path = QFileDialog.getOpenFileName(self, "选择图片", "./",
                                               "All Files (*);;Image Files (*.jpg);;Image Files (*.png)")
        self.pic_path = pic_path
        print(f'pic_path: {pic_path}')

    def open_video(self):
        print('视频')
        video_path = QFileDialog.getOpenFileName(self, "选择视频", "./",
                                                 "All Files (*);;Video Files (*.mp4);;Image Files (*.avi)")
        self.video_path = memoryview
        print(f'pic_path: {video_path}')

    # 手势检测
    def gesture_detection(self):
        print('手势检测')
        flag = self.msg_box()

        if flag != 0:
            yolo = YOLO()
            if flag == 3:
                cap = self.predict()
            else:
                cap = cv2.VideoCapture('./video/hands.mp4')
            count = 0
            while True:
                # if (count % 10) == 0:
                # print(f'Frame: {count}')
                ret, frame = cap.read()
                if ret:
                    if flag == 1:
                        print(f'pic path: {self.pic_path[0]}')
                        frame = cv2.imread(self.pic_path[0])
                        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    else:
                        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    r_image, boxes = yolo.detect_image(image)
                    hand_img = cv2.cvtColor(np.asarray(r_image), cv2.COLOR_RGB2BGR)
                    image = cv2.resize(hand_img, (960, 544))
                    show = np.asarray(image)
                    show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)  # 这里指的是显示原图
                    # opencv 读取图片的样式，不能通过Qlabel进行显示，需要转换为Qimage QImage(uchar * data, int width,
                    self.showImage = QImage(show.data, show.shape[1], show.shape[0], QImage.Format_RGB888)
                    self.label_5.setPixmap(QPixmap.fromImage(self.showImage))
                    cv2.waitKey(1)
                    cv2.imwrite('img/test.jpg', hand_img)
                    if flag == 1:
                        self.success_text = '恭喜您，图片检测成功！'
                        push_msg(self.success_text)
                        break
                else:
                    print('video failed')
                    if flag == 2:
                        self.success_text = '恭喜您，视频检测成功！'
                        push_msg(self.success_text)
                    break
            count += 1

    # 手势识别
    def gesture_recognition(self):
        print('手势识别')
        flag = self.msg_box()
        if flag != 0:
            yolo = YOLO()
            if flag == 3:

                cap = self.predict()
            else:
                cap = cv2.VideoCapture('./video/hands.mp4')
            # checkpoint = torch.load("./logs/coco_checkpoint.pth")
            # checkpoint = torch.load("./logs/coco_checkpoint.pth", map_location="cpu")
            checkpoint = torch.load("./logs/coco_checkpoint.pth", map_location=lambda storage, loc: storage.cuda(1))
            # checkpoint = torch.load("./logs/coco_checkpoint.pth", map_location={'cuda:1': 'cuda:0'})
            model = HighResolutionNet(cfg).cuda()
            model.eval()
            model.load_state_dict(checkpoint["state_dict"])
            with torch.no_grad():
                count = 0
                while True:
                    # if (count % 10) == 0:
                    # print(f'Frame: {count}')
                    ret, frame = cap.read()
                    if ret:
                        if flag == 1:
                            print(f'pic path: {self.pic_path[0]}')
                            frame = cv2.imread(self.pic_path[0])
                            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        else:
                            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        r_image, boxes = yolo.detect_image(image)
                        img = cv2.cvtColor(np.asarray(r_image), cv2.COLOR_RGB2BGR)

                        image = img[:, :, ::-1]
                        for box in boxes:
                            x, y, w, h = box
                            scale = max(w, h) / 200. * 1.25
                            center = (x + w / 2., y + h / 2.)
                            rot = 0
                            meta = get_transform(center, scale, (256, 256), rot)
                            warp_img = cv2.warpAffine(image,
                                                      meta[:2], (256, 256),
                                                      flags=cv2.INTER_LINEAR)
                            input_tensor = np.transpose(warp_img / 255., (2, 0, 1))
                            input_tensor = torch.from_numpy(input_tensor).float().unsqueeze(0).cuda()
                            out = model(input_tensor)
                            pred, maxvals = keypoints_from_heatmaps(out.cpu().numpy(),
                                                                    meta[None])
                            pred = np.concatenate((pred, maxvals), axis=2)
                            show_image(image, pred[0], "img/test.jpg")
                            hand_img = cv2.imread('img/test.jpg')
                            image = cv2.imread('img/test.jpg')
                            image = cv2.resize(image, (960, 544))
                            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        image = cv2.resize(hand_img, (960, 544))
                        show = np.asarray(image)
                        show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)  # 这里指的是显示原图
                        # opencv 读取图片的样式，不能通过Qlabel进行显示，需要转换为Qimage QImage(uchar * data, int width,
                        self.showImage = QImage(show.data, show.shape[1], show.shape[0], QImage.Format_RGB888)
                        self.label_5.setPixmap(QPixmap.fromImage(self.showImage))
                        cv2.waitKey(1)
                        cv2.imwrite('img/test.jpg', hand_img)
                        if flag == 1:
                            self.success_text = '恭喜您，图片识别成功！'
                            push_msg(self.success_text)
                            break
                    else:
                        print('video failed')
                        if flag == 2:
                            self.success_text = '恭喜您，视频识别成功！'
                            push_msg(self.success_text)
                        break
                count += 1

    #  手势评分
    def hand_put_right(self):
        print('手势评分')

        flag = self.msg_box(1)
        if flag != 0:
            yolo = YOLO()
            if flag == 3:

                cap = self.predict()
            else:
                cap = cv2.VideoCapture('./video/hands.mp4')
            checkpoint = torch.load("./logs/coco_checkpoint.pth", map_location="cpu")
            model = HighResolutionNet(cfg).cuda()
            model.eval()
            model.load_state_dict(checkpoint["state_dict"])
            with torch.no_grad():
                count = 0
                sum_hand_ture = 0
                sum_hand_wrong = 0
                while True:
                    # if (count % 10) == 0:
                    # print(f'Frame: {count}')
                    ret, frame = cap.read()
                    if ret:
                        if flag == 1:
                            print(f'pic path: {self.pic_path[0]}')
                            frame = cv2.imread(self.pic_path[0])
                            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        else:
                            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        r_image, boxes, hand_ture, hand_wrong = yolo.detect_image(image, 1)
                        img = cv2.cvtColor(np.asarray(r_image), cv2.COLOR_RGB2BGR)

                        image = img[:, :, ::-1]
                        for box in boxes:
                            x, y, w, h = box
                            scale = max(w, h) / 200. * 1.25
                            center = (x + w / 2., y + h / 2.)
                            rot = 0
                            meta = get_transform(center, scale, (256, 256), rot)
                            warp_img = cv2.warpAffine(image,
                                                      meta[:2], (256, 256),
                                                      flags=cv2.INTER_LINEAR)
                            input_tensor = np.transpose(warp_img / 255., (2, 0, 1))
                            input_tensor = torch.from_numpy(input_tensor).float().unsqueeze(0).cuda()
                            out = model(input_tensor)
                            pred, maxvals = keypoints_from_heatmaps(out.cpu().numpy(),
                                                                    meta[None])
                            pred = np.concatenate((pred, maxvals), axis=2)
                            show_image(image, pred[0], "img/test.jpg")
                            hand_img = cv2.imread('img/test.jpg')
                            image = cv2.imread('img/test.jpg')
                            image = cv2.resize(image, (960, 544))
                            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        image = cv2.resize(hand_img, (960, 544))
                        show = np.asarray(image)
                        show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)  # 这里指的是显示原图
                        # opencv 读取图片的样式，不能通过Qlabel进行显示，需要转换为Qimage QImage(uchar * data, int width,
                        self.showImage = QImage(show.data, show.shape[1], show.shape[0], QImage.Format_RGB888)
                        self.label_5.setPixmap(QPixmap.fromImage(self.showImage))
                        cv2.waitKey(1)
                        cv2.imwrite('img/test.jpg', hand_img)

                        sum_hand_ture += hand_ture
                        sum_hand_wrong += hand_wrong

                    else:
                        print('video failed')
                        if flag == 2:
                            print(f'sum_frame: {count}')
                            print(f'sum_ture: {sum_hand_ture}')
                            print(f'sum_wrong: {sum_hand_wrong}')
                            self.success_text = f'恭喜你，你的手势分数为：{sum_hand_ture}！'
                            push_msg(self.success_text)
                        break
                count += 1
        pass

    # 选择 图片 视频 摄像头
    def msg_box(self, only_camera=0):
        flag = 0

        if only_camera:
            # 创建一个问答框，注意是Question
            self.box = QMessageBox(QMessageBox.Question, '选择', '手势评分仅支持视频评分')
            # 添加按钮，可用中文
            self.box.setStandardButtons(QMessageBox.Yes)
            video = self.box.button(QMessageBox.Yes)
            video.setText('选择视频')
            # 设置消息框中内容前面的图标
            self.box.setIcon(1)
            # 设置消息框的位置，大小无法设置
            self.box.setGeometry(800, 500, 1000, 1000)
            # 显示该问答框
            self.box.exec_()
            if self.box.clickedButton() == video:
                print('选择视频')
                self.open_video()

            return 2

        # 创建一个问答框，注意是Question
        self.box = QMessageBox(QMessageBox.Question, '选择', '请选择下方使用方式')
        # 添加按钮，可用中文
        self.box.setStandardButtons(QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel)
        pic = self.box.button(QMessageBox.Yes)
        pic.setText("使用图片")
        video = self.box.button(QMessageBox.No)
        video.setText("使用视频")
        camera = self.box.button(QMessageBox.Cancel)
        camera.setText("使用摄像头")
        # 设置消息框中内容前面的图标
        self.box.setIcon(1)
        # 设置消息框的位置，大小无法设置
        self.box.setGeometry(800, 500, 1000, 1000)
        # 显示该问答框
        # self.box.show()
        self.box.exec_()
        if self.box.clickedButton() == pic:
            print('选择图片')
            pic_path = QFileDialog.getOpenFileName(self, "选择图片", "./",
                                                   "All Files (*);;Image Files (*.jpg);;Image Files (*.png)")
            self.pic_path = pic_path
            flag = 1
        elif self.box.clickedButton() == video:
            print('选择视频')
            self.open_video()
            flag = 2
        elif self.box.clickedButton() == camera:
            print('选择摄像头')
            flag = 3

        return flag


if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWin = MyMainWindow()
    print('欢迎使用肖帮！')
    sys.exit(app.exec_())
