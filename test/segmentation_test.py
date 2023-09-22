from ultralytics import YOLO
import torch
import numpy as np
import cv2
import time
import os
model_YOLO = YOLO(r"C:\Users\mirok\PycharmProjects\API\app\model\YOLOv8segmentation.pt")

img = cv2.imread(r'C:\Users\mirok\PycharmProjects\API\app\Images\DSC04248.JPG')
H,W, _ = img.shape
img = cv2.resize(img,(384,640))
results = model_YOLO.predict(img)
# model_YOLO.predict(r'C:\Users\mirok\PycharmProjects\API\app\Images\DSC04248.JPG', save=True, conf=0.5)
# print(results)
# for result in results:
#     for mask in result.masks.data:
#         mask = mask.numpy()*255
#         mask = cv2.resize(mask,(W,H))
#         cv2.imshow("result", mask)
#         cv2.waitKey(0)
for result in results:
    for mask in result.masks:
        print(mask)
        print(mask.data.shape)
        m = torch.squeeze(mask.data)
        print(m.shape)
        composite = torch.stack((m,m,m), 2)
        print(composite.shape)
        tmp = img * composite.numpy().astype(np.uint8)
        print(tmp.shape)
        tmp = cv2.resize(tmp,(W, H))
        cv2.imshow("result", tmp)
        cv2.waitKey(0)