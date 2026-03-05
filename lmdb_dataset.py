import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
import lmdb

class LMDBDataset(Dataset):

    # Из байтов делает картинку и превращает текст класса в ID
    def __init__(self, db_path, transform=None):
        self.db_path = db_path
        self.transform = transform

        # LMDB (self.env != .env)
        self.env = lmdb.open(
            db_path,
            readonly=True, # True запрет писать в базу
            lock=False, # False отключена блокировка многопоточного чтения
            readahead=False,  # False экономит память
            meminit=False # False ускоряет запуск.
        )

        # Открываем (txn) в режиме "только чтение" (write=False).
        with self.env.begin(write=False) as txn:
            # Извлекаем значение по ключу "num_samples".
            self.length = int(txn.get(b"num_samples").decode())

            # Список для сбора всех текстовых меток
            all_labels = []

            # Сcursor - итератор, который позволяет перебирать базу, ключ за ключом
            cursor = txn.cursor()

            for key, value in cursor:
                # Декодируем ключ в строку
                # Проверить, начинается ли он с "label_"
                if key.decode().startswith("label_"):
                    # Добавляем в список
                    all_labels.append(value.decode())

            # set() убирает дубликаты
            # list() превращает это обратно в список.
            # sorted(...) сортирует названия по алфавиту
            self.classes = sorted(list(set(all_labels)))

            # Словарь [(0, 'Апельсин'), (1, 'Манго'), (2, 'Яблоко')]
            self.class_to_idx = {name: i for i, name in enumerate(self.classes)}

            print(f"Найдено уникальных классов: {len(self.classes)}")

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # Формируем строковый ключ (например, image_00000005)
        # Нам нужно строго 8 знаков с нулями, как в упаковщике
        string_id = f"{index:08d}"
        img_key = f"image_{string_id}".encode()
        label_key = f"label_{string_id}".encode()

        with self.env.begin(write=False) as txn:
            # Достаем сырые байты из LMDB
            img_bytes = txn.get(img_key)
            label_text = txn.get(label_key).decode()

            # Превращаем байты в массив чисел (картинку) через OpenCV
            # numpy.frombuffer создает массив из байтов без копирования памяти (быстро)
            img_buffer = np.frombuffer(img_bytes, dtype=np.uint8)
            image = cv2.imdecode(img_buffer, cv2.IMREAD_COLOR)

            # Важно: OpenCV выдает BGR, а нейросети учились на RGB. Меняем каналы.
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Превращаем текст "Добрый..." в цифру 0, 1 или 2...
            target_idx = self.class_to_idx[label_text]

        # Применяем трансформации (изменение размера, нормализация и т.д.)
        if self.transform:
            image = self.transform(image)

        return image, target_idx