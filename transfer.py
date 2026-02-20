import os, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np
from sklearn.metrics import classification_report
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
EPOCHS = 25
IMG_SIZE = 224
LR = 0.001
CLASS_NAMES = ["Healthy", "Coccidiosis", "Salmonella", "Newcastle", "InfectiousBronchitis"]
NUM_CLASSES = len(CLASS_NAMES)

class BroilerDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.images, self.labels = [], []
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        for cls in self.classes:
            cls_dir = os.path.join(root_dir, cls)
            for img in os.listdir(cls_dir):
                if img.lower().endswith(('.png','.jpg','.jpeg')):
                    self.images.append(os.path.join(cls_dir, img))
                    self.labels.append(self.class_to_idx[cls])
        print(f"Loaded {len(self.images)} images from {root_dir}")

    def __len__(self): return len(self.images)
    
    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('RGB')
        if self.transform: img = self.transform(img)
        return img, self.labels[idx]

mean, std = [0.485,0.456,0.406], [0.229,0.224,0.225]
train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])
val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

class BroilerClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet50(weights='IMAGENET1K_V1')
        for p in self.backbone.parameters():
            p.requires_grad = False
        for p in self.backbone.layer4.parameters():
            p.requires_grad = True
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, NUM_CLASSES)
        )
    
    def forward(self, x): return self.backbone(x)

def train():
    print("="*50)
    print("BROILER DISEASE CLASSIFICATION")
    print("="*50)
    print(f"Device: {DEVICE}")
    
    train_ds = BroilerDataset("train", train_tf)
    val_ds = BroilerDataset("valid", val_tf)
    test_ds = BroilerDataset("test", val_tf)
    
    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, BATCH_SIZE)
    test_loader = DataLoader(test_ds, BATCH_SIZE)
    
    model = BroilerClassifier().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    
    best_acc = 0
    history = {'train_loss': [], 'val_acc': []}
    
    for epoch in range(1, EPOCHS+1):
        print(f"\nEpoch {epoch}/{EPOCHS}")
        
        model.train()
        train_loss = 0
        for x, y in tqdm(train_loader, desc="Train"):
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc="Val"):
                x, y = x.to(DEVICE), y.to(DEVICE)
                pred = model(x).argmax(1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        
        val_acc = 100 * correct / total
        avg_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Train Loss: {avg_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            print(f"✓ Saved best model ({val_acc:.2f}%)")
    
    print("\n" + "="*50)
    print("TESTING BEST MODEL")
    print("="*50)
    model.load_state_dict(torch.load("best_model.pth"))
    model.eval()
    
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="Test"):
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = model(x).argmax(1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    
    test_acc = 100 * np.mean(np.array(all_preds) == np.array(all_labels))
    print(f"\nTest Accuracy: {test_acc:.2f}%")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES, digits=4))
    
    torch.save({
        'state_dict': model.state_dict(),
        'class_names': CLASS_NAMES,
        'test_acc': test_acc
    }, "final_model.pth")
    
    print(f"\nModels saved: best_model.pth, final_model.pth")

if __name__ == "__main__":
    for folder in ['train', 'valid', 'test']:
        if not os.path.exists(folder):
            print(f"Error: '{folder}' folder not found!")
            exit()
    train()
    
   