
from torchvision import transforms
import torch
from app.Architecture import Architecture as A
from PIL import Image

# onnx_model_path = r"C:\Users\mirok\PycharmProjects\API\model\model.onnx"
# model = onnx.load(onnx_model_path)
# onnx.checker.check_model(model)
# print(onnx.helper.printable_graph(model.graph))

model = A.MyArchitecture(3,5)  # Replace YourModelClass with the actual class name of your model architecture
model.load_state_dict(torch.load(r"/app/model\trained_model.pth", map_location=torch.device('cpu')))
class_to_idx ={'spino': 0, 'trex': 1, 'stego': 2, 'velo': 3, 'para': 4}
transform = transforms.Compose([
    transforms.Resize((150,150)),  # Resize to a fixed size
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize image
])

image = Image.open(r"C:\Users\mirok\PycharmProjects\API\Images\DSC05576.JPG").convert('RGB')
input_tensor = transform(image)
input_batch = input_tensor.unsqueeze(0)
 # Use the model to make a prediction
with torch.no_grad():
        output = model(input_batch)


print(output)
print({i for i in class_to_idx if class_to_idx[i]==int(torch.argmax(output))})