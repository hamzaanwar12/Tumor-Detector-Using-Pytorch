import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import os

class BrainTumorCNN(nn.Module):
    def __init__(self):
        super(BrainTumorCNN,self).__init__()
        self.cnn_model = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5),
        nn.Tanh(),
        nn.AvgPool2d(kernel_size=2, stride=5),
        nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
        nn.Tanh(),
        nn.AvgPool2d(kernel_size=2, stride=5))

        self.fc_model = nn.Sequential(
        nn.Linear(in_features=256, out_features=120),
        nn.Tanh(),
        nn.Linear(in_features=120, out_features=84),
        nn.Tanh(),
        nn.Linear(in_features=84, out_features=1))

    def forward(self, x):
        x = self.cnn_model(x)
        x = x.view(x.size(0), -1)
        x = self.fc_model(x)
        x = F.sigmoid(x)

        return x

# Define the function to predict tumor presence
def predict_tumor(image_path, model_path):
    
    # Load the saved model
    model = BrainTumorCNN()  # Assuming BrainTumorCNN is defined as in your previous code
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode

    # Define image transformations
    image_transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Resize the image to 128x128
        transforms.ToTensor(),           # Convert PIL image to tensor
        transforms.Normalize((0, 0, 0), (1/255, 1/255, 1/255))  # Normalize the image by dividing by 255
    ])

    # Load and preprocess the image
    image = Image.open(image_path)
    image = image_transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    # Perform prediction
    with torch.no_grad():
        output = model(image)
        predicted_class = torch.round(output).item()  # Round the output to get binary prediction (0 or 1)

    # Interpret the prediction
    if predicted_class == 0:
        return False
    else:
        return True


