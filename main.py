from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import torch
from Architecture.Architecture import MyArchitecture
from torchvision import transforms
from PIL import Image
from ultralytics import YOLO
import io
import cv2
import numpy as np

app = FastAPI()

model_classify = MyArchitecture(3, 5)
# model_classify.load_state_dict(torch.load(r"model/classfication_model.pth", map_location=torch.device('cpu')))

transform_classify = transforms.Compose([
    transforms.Resize((150, 150)),  # Resize to a fixed size
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize image
])


model_YOLO_detection = YOLO(r"YOLOv8detection.pt")
model_YOLO_segmentation = YOLO(r"YOLOv8segmentation.pt")

def classify_image(image_bytes):

    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    input_tensor = transform_classify(image)
    input_batch = input_tensor.unsqueeze(0)
    class_to_idx = {'spino': 0, 'trex': 1, 'stego': 2, 'velo': 3, 'para': 4}
    with torch.no_grad():
        output = model_classify(input_batch)

    return image, [i for i in class_to_idx if class_to_idx[i]==int(torch.argmax(output))]

def detect_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    results = model_YOLO_detection(image)
    names = model_YOLO_detection.names
    for r in results:
        boxes = r.boxes
        for box in boxes:  # iterate boxes
            r = box.xyxy[0].numpy()  # get corner points as int
        for c in boxes.cls:
            predicted_class = names[int(c)]
    return image,list(r),predicted_class

def segmentation_image(image_bytes):

    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    results = model_YOLO_segmentation(image)
    W, H  = image.size
    image = image.resize((640, 384))
    for result in results:
        for mask in result.masks:
            m = torch.squeeze(mask.data)
            composite = torch.stack((m, m, m), 2)
            tmp = image * composite.numpy().astype(np.uint8)
            tmp = cv2.resize(tmp, (W, H))

    return tmp

@app.post("/classify/")
async def  classify_endpoint(image: UploadFile = File(...)):

    image_bytes = await image.read()
    classified_image, predicted_class = classify_image(image_bytes)
    output_image_path = "classified_image.jpg"
    classified_image.save(output_image_path)
    position = (50, 50)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (255, 255, 255)  # Text color in BGR format (white in this case)
    font_thickness = 2
    img = cv2.imread(output_image_path)
    cv2.putText(img,str(predicted_class),position, font, font_scale, font_color, font_thickness)
    cv2.imwrite(output_image_path, img)

    return FileResponse(output_image_path, media_type="image/jpeg")

@app.post("/detect/")
async def detect_endpoint(image: UploadFile = File(...)):

    image_bytes = await image.read()
    detected_image,predicted_boxes,predicted_class = detect_image(image_bytes)
    output_image_path = "detected_image.jpg"
    detected_image.save(output_image_path)
    color = (0, 255, 0)
    thickness = 2
    position = (50, 50)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (255, 255, 255)
    font_thickness = 2
    img = cv2.imread(output_image_path)
    cv2.rectangle(img,(int(predicted_boxes[0]), int(predicted_boxes[1])), (int(predicted_boxes[2]), int(predicted_boxes[3])), color,thickness)
    cv2.putText(img,str(predicted_class),position, font, font_scale, font_color, font_thickness)
    cv2.imwrite(output_image_path, img)
    return FileResponse(output_image_path, media_type="image/jpeg")

@app.post("/segmentation/")
async def segmentation_endpoint(image: UploadFile = File(...)):
    image_bytes = await image.read()
    img = segmentation_image(image_bytes)
    output_image_path = "segmentation_image.jpg"
    cv2.imwrite(output_image_path, img)
    return FileResponse(output_image_path, media_type="image/jpeg")

@app.get("/")
def root():
    return "Miro je boh"

