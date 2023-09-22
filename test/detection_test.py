from ultralytics import YOLO
from PIL import Image
import cv2

model_YOLO = YOLO(r"../model/YOLOv8detection.pt")
# img = Image.open(r'C:\Users\mirok\PycharmProjects\API\app\Images\DSC04248.JPG')
img =cv2.imread(r'/app/Images/DSC04248.JPG')
results =model_YOLO(img)
color = (0, 255, 0)
thickness = 2
image = cv2.imread(r'/app/Images/DSC04248.JPG')

for r in results:
    print(r.probs)
    boxes = r.boxes
    for box in boxes:  # iterate boxes
        r = box.xyxy[0].numpy() # get corner points as int
        print(r.shape)
        print(list(r))
        # print(int(r[0]))
        # # print(r)  # print boxes
        # cv2.rectangle(image,(int(r[0]), int(r[1])), (int(r[2]), int(r[3])), color,thickness)
        # cv2.imshow('Rectangle Image', image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
names = model_YOLO.names

