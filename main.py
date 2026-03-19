import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from dotenv import load_dotenv
from lmdb_dataset import LMDBDataset
import torch
import subprocess
import shutil

load_dotenv()

def get_device():
    """
    Определяет доступное устройство для вычислений.
    Возвращает: torch.device и строку с информацией об устройстве
    Поддерживает: NVIDIA CUDA, AMD ROCm, CPU
    """

    # Проверяем NVIDIA GPU через nvidia-smi (самый надёжный способ)
    if shutil.which('nvidia-smi') is not None:
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                                    capture_output=True, text=True, timeout=2)
            if result.returncode == 0 and result.stdout.strip():
                gpu_name = result.stdout.strip().split('\n')[0]
                # Дополнительно проверяем через torch
                if torch.cuda.is_available():
                    return torch.device("cuda"), f"NVIDIA {gpu_name} (CUDA)"
                else:
                    return torch.device("cpu"), f"NVIDIA {gpu_name} (CUDA не доступна в PyTorch)"
        except:
            pass

    # Проверяем AMD GPU (ROCm)
    if torch.cuda.is_available() and hasattr(torch.version, 'hip') and torch.version.hip is not None:
        try:
            # Проверяем, что это действительно AMD через rocm-smi
            if shutil.which('rocm-smi') is not None:
                result = subprocess.run(['rocm-smi', '--showproductname'],
                                        capture_output=True, text=True, timeout=2)
                if result.returncode == 0:
                    return torch.device("cuda"), f"AMD GPU (ROCm {torch.version.hip})"

            # Альтернативная проверка через имя устройства
            gpu_name = torch.cuda.get_device_name(0)
            if 'amd' in gpu_name.lower() or 'radeon' in gpu_name.lower() or 'instinct' in gpu_name.lower():
                return torch.device("cuda"), f"AMD {gpu_name} (ROCm {torch.version.hip})"
        except:
            pass

    # Если CUDA доступна через torch, но не определена как NVIDIA или AMD
    if torch.cuda.is_available():
        try:
            gpu_name = torch.cuda.get_device_name(0)
            return torch.device("cuda"), f"{gpu_name} (CUDA)"
        except:
            return torch.device("cuda"), "Неизвестный GPU (CUDA)"

    # Если ничего не найдено - используем CPU
    return torch.device("cpu"), "CPU (GPU не обнаружены)"


def train_model():
    input_db = os.getenv("DATASET_PATH")
    print("DATASET_PATH =", input_db)
    model_save_path = os.getenv("MODEL_PATH")
    print("MODEL_PATH =", model_save_path)
    batch_size = int(os.getenv("BATCH_SIZE", 32))
    epochs = int(os.getenv("EPOCHS", 5))
    learning_rate = float(os.getenv("LEARNING_RATE", 0.001))
    resume_flag = os.getenv("RESUME_TRAINING", "False").lower() == "true"

    # Выбор устройства: GPU (cuda) или процессор (cpu)
    device, device_info = get_device()
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Работаем на: {device} ({device_info}) ---")

    # Трансформации: преобразуем картинку в понятный нейронке вид
    train_transforms = transforms.Compose([
        transforms.ToPILImage(),  # Из массива в формат PIL (нужно для Resize)
        transforms.Resize((224, 224)),  # Размер
        transforms.ToTensor(),  # В тензор и диапазон [0, 1]
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Стандарт ImageNet пока не трогаю
    ])

    # Инициализируем Dataset LMDB
    dataset = LMDBDataset(input_db, transform=train_transforms)
    num_classes = len(dataset.classes)

    # DataLoader берет данные из Dataset и подает их пачками
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)

    # Берем готовую архитектуру ResNet50 (условие с продолжением обучения)
    model = models.resnet50(weights='DEFAULT' if not resume_flag else None)

    # fc последний слой изначальной нейронки - количество выходов
    in_features = model.fc.in_features
    # Заменяем на свои
    model.fc = nn.Linear(in_features, num_classes)

    # Если в .env указано продолжение — загружаем старый файл модели
    if resume_flag and os.path.exists(model_save_path):
        print(f"🔄 Загружаем сохраненную модель...")
        checkpoint = torch.load(model_save_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

    model = model.to(device)

    # Функция потерь
    criterion = nn.CrossEntropyLoss()
    # Алгоритм обновления весов (Adam — самый универсальный)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # --- ЦИКЛ ОБУЧЕНИЯ ---
    print("Старт обучения...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for images, labels in train_loader:
            # Перекидываем данные на видеокарту/процессор
            images, labels = images.to(device), labels.to(device)

            # Сбрасываем старые градиенты, чтобы не накапливались
            optimizer.zero_grad()

            # Прямой проход
            outputs = model(images)

            # Считаем ошибку
            loss = criterion(outputs, labels)

            # Обратный проход
            loss.backward()

            # Шаг обновления весов
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Эпоха [{epoch + 1}/{epochs}] | Ошибка (Loss): {avg_loss:.4f}")

    # Сохраняем веса и список классов
    torch.save({
        'model_state_dict': model.state_dict(),
        'classes': dataset.classes,
        'class_to_idx': dataset.class_to_idx
    }, model_save_path)

    print(f"✅ Обучение окончено. Модель сохранена в: {model_save_path}")


if __name__ == "__main__":
    train_model()

