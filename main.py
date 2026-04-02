import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
from dotenv import load_dotenv
from lmdb_dataset import LMDBDataset
import subprocess
import shutil

load_dotenv()


def get_device():
    """Определяет доступное устройство (CUDA/CPU)."""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        return torch.device("cuda"), f"{gpu_name} (CUDA)"
    return torch.device("cpu"), "CPU"


def train_model():
    # Загрузка настроек из .env
    input_db = os.getenv("DATASET_PATH")
    model_save_path = os.getenv("MODEL_PATH")
    batch_size = int(os.getenv("BATCH_SIZE", 32))
    epochs = int(os.getenv("EPOCHS", 5))
    learning_rate = float(os.getenv("LEARNING_RATE", 0.001))
    resume_flag = os.getenv("RESUME_TRAINING", "False").lower() == "true"

    device, device_info = get_device()
    print(f"DATASET_PATH = {input_db}")
    print(f"MODEL_PATH = {model_save_path}")
    print(f"--- Работаем на: {device} ({device_info}) ---")

    # Трансформации
    train_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Датасет
    dataset = LMDBDataset(input_db, transform=train_transforms)
    num_classes = len(dataset.classes)

    # ОПТИМИЗАЦИЯ: num_workers=0 экономит RAM и предотвращает SegFault
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )

    # Модель ResNet50
    model = models.resnet50(weights='DEFAULT' if not resume_flag else None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # Загрузка весов, если продолжаем обучение
    start_epoch = 0
    if resume_flag and os.path.exists(model_save_path):
        print(f"🔄 Загружаем сохраненную модель для продолжения...")
        checkpoint = torch.load(model_save_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print("Старт обучения...")
    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Эпоха [{epoch + 1}/{epochs}] | Ошибка (Loss): {avg_loss:.4f}")

        # ОБНОВЛЕНИЕ МОДЕЛИ НА ДИСКЕ ПОСЛЕ КАЖДОЙ ЭПОХИ
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'classes': dataset.classes,
            'class_to_idx': dataset.class_to_idx,
            'loss': avg_loss
        }, model_save_path)
        print(f"--- Чекпоинт сохранен: {model_save_path} ---")

        # Принудительная очистка кэша видеопамяти
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"✅ Обучение окончено.")


if __name__ == "__main__":
    train_model()