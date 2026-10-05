import os
import random
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from tqdm import tqdm

TRAIN_DIR = r"E:\大学作业\计算机视觉\cowface-verification-U\train"
TEST_DIR  = r"E:\大学作业\计算机视觉\cowface-verification-U\test-new"
TEST_CSV  = r"E:\大学作业\计算机视觉\total_set\test-1118.csv"

MODEL_DIR = r"E:\大学作业\计算机视觉\cowface-verification-U\models"
os.makedirs(MODEL_DIR, exist_ok=True)  

#数据集
class TrainPairDataset(Dataset):
    def __init__(self, train_root, transform=None):
        self.transform = transform
        self.cow_dict = {}
        for cow_id in os.listdir(train_root):
            folder = os.path.join(train_root, cow_id)
            if os.path.isdir(folder):
                self.cow_dict[cow_id] = [
                    os.path.join(folder, img) for img in os.listdir(folder)
                ]
        self.cow_ids = list(self.cow_dict.keys())
        self.pairs = []
        for cow in self.cow_ids:
            imgs = self.cow_dict[cow]
            #正样本
            if len(imgs) > 1:
                for i in range(len(imgs)-1):
                    self.pairs.append((imgs[i], imgs[i+1], 1))
            #负样本
            for _ in range(3):
                img1 = random.choice(imgs)
                other = random.choice([c for c in self.cow_ids if c != cow])
                img2 = random.choice(self.cow_dict[other])
                self.pairs.append((img1, img2, 0))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img1_path, img2_path, label = self.pairs[idx]
        img1 = Image.open(img1_path).convert("RGB")
        img2 = Image.open(img2_path).convert("RGB")
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return img1, img2, torch.tensor(label, dtype=torch.float32)

class TestPairDataset(Dataset):
    def __init__(self, test_dir, csv_file, transform=None):
        self.df = pd.read_csv(csv_file)
        self.root = test_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        pair = self.df.iloc[i, 0]
        img1, img2 = pair.split("_")
        img1 = Image.open(os.path.join(self.root, img1+".jpg")).convert("RGB")
        img2 = Image.open(os.path.join(self.root, img2+".jpg")).convert("RGB")
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return img1, img2, pair

#模型
class SiameseNet(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.resnet18(pretrained=True)
        base.fc = nn.Identity()
        self.encoder = base
        self.classifier = nn.Sequential(
            nn.Linear(512*2, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        f1 = self.encoder(x1)
        f2 = self.encoder(x2)
        x = torch.cat([f1, f2], dim=1)
        return self.classifier(x)

#训练
def train_model(epochs=5, batch_size=16, device="cpu"):
    transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
    dataset = TrainPairDataset(TRAIN_DIR, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = SiameseNet().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(epochs):
        total_loss = 0
        for img1, img2, labels in tqdm(loader):
            img1, img2, labels = img1.to(device), img2.to(device), labels.unsqueeze(1).to(device)
            preds = model(img1, img2)
            loss = criterion(preds, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss={total_loss/len(loader):.4f}")

    model_path = os.path.join(MODEL_DIR, "model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"模型已保存到 {model_path}")

#预测
def predict(device="cpu"):
    transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
    model = SiameseNet().to(device)
    model_path = os.path.join(MODEL_DIR, "model.pth")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    dataset = TestPairDataset(TEST_DIR, TEST_CSV, transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    results = []
    with torch.no_grad():
        for img1, img2, pair_ids in tqdm(loader):
            img1, img2 = img1.to(device), img2.to(device)
            preds = model(img1, img2).cpu().numpy().reshape(-1)
            for pid, p in zip(pair_ids, preds):
                results.append([pid, int(p>0.5)])

    df = pd.DataFrame(results, columns=["ID","target"])
    submission_path = os.path.join(MODEL_DIR, "submission.csv")
    df.to_csv(submission_path, index=False)
    print(f"已生成 {submission_path}")

#主程序
if __name__ == "__main__":
    MODE = "train"  # "train" 或 "predict"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    if MODE=="train":
        train_model(epochs=5, batch_size=16, device=DEVICE)
    elif MODE=="predict":
        predict(device=DEVICE)
    else:
        print("请设置 MODE 为 'train' 或 'predict'")
