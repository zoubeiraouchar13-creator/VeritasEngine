## Architecture Pytorch (LSTM et CNN)
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from transformers import AutoTokenizer
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Architecture NLP : Bi-LSTM
class FakeNewsLSTM(nn.Module):
    def __init__(self, vocab_size=30522, embedding_dim=128, hidden_dim=64):
        super(FakeNewsLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=2, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        out = torch.cat((_[0][-2,:,:], _[0][-1,:,:]), dim=1) # Concaténation des deux directions
        return self.sigmoid(self.fc(out))

# 2. Architecture Computer Vision : CNN Transfer Learning
class ImageManipulationCNN(nn.Module):
    def __init__(self):
        super(ImageManipulationCNN, self).__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_ftrs, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.backbone(x)

# 3. Fonctions d'inférence réelles
def inference_text(text: str) -> float:
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    inputs = tokenizer(text, max_length=50, padding="max_length", truncation=True, return_tensors="pt")
    
    model = FakeNewsLSTM().to(device)
    model.eval()
    with torch.no_grad():
        output = model(inputs["input_ids"].to(device))
    return float(output.item())

def inference_image(image_path: str) -> float:
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    try:
        image = Image.open(image_path).convert('RGB')
        tensor = transform(image).unsqueeze(0).to(device)
        
        model = ImageManipulationCNN().to(device)
        model.eval()
        with torch.no_grad():
            output = model(tensor)
        return float(output.item())
    except Exception:
        return 0.0