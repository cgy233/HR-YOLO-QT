import torch
from PIL import Image
import cv2
import numpy as np
from yolo import YOLO
from models.defaults import _C as cfg
from models.hrnet import HighResolutionNet
from utils.hr_utils.transform import keypoints_from_heatmaps, get_transform
from utils.hr_utils.utils import show_image


yolo = YOLO()
cap = cv2.VideoCapture('./video/hands.mp4')
# checkpoint = torch.load("./checkpoint.pth", map_location="cpu")
checkpoint = torch.load("./logs/coco_checkpoint.pth", map_location="cpu")
model = HighResolutionNet(cfg).cuda()
model.eval()
model.load_state_dict(checkpoint["state_dict"])

# fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
# fps = 30
# frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
# frame_size = (960, 544)
# outCamera = cv2.VideoWriter('./hands_test4.avi', fourcc, fps, frame_size)

with torch.no_grad():
    count = 0
    while True:
        # if (count % 10) == 0:
        # print(f'Frame: {count}')
        ret, frame = cap.read()
        if ret:
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            r_image, boxes = yolo.detect_image(image)
            img = cv2.cvtColor(np.asarray(r_image), cv2.COLOR_RGB2BGR)

            image = img[:, :, ::-1]
            # w, h = image.shape[:-1]
            for box in boxes:
                # box = boxs[i]
                # box = [490, 202, w, h]
                # print(f'BOX: {box}')
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
                # maxvals[maxvals<0.15] = 0
                pred = np.concatenate((pred, maxvals), axis=2)
                # show_image(image, pred[0], "tmp/" + imgnames[i][:-4] + "_coco.jpg")
                show_image(image, pred[0], "img/test.jpg")
                hand_img = cv2.imread('img/test.jpg')
                image = cv2.imread('img/test.jpg')
                image = cv2.resize(image, (960, 544))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # cv2.imshow('plt1', image)
                # cv2.waitKey(1)
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            cv2.namedWindow('hand', cv2.WINDOW_NORMAL)
            cv2.imshow('hand', hand_img)
            cv2.waitKey(1)
            cv2.imwrite('img/test.jpg', hand_img)
            # cv2.imwrite(f'./hands_demo/{str(count)}.jpg', hand_img)
            # outCamera.write(hand_img)
        else:
            print('video failed')
            break
        count += 1
