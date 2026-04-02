import cv2
import numpy as np
from torch.utils.data import Dataset
import lmdb


class LMDBDataset(Dataset):
    def __init__(self, db_path, transform=None):
        self.db_path = db_path
        self.transform = transform
        self.env = None  # Не открываем базу сразу

        # Один раз открываем базу, чтобы только прочитать метаданные (классы)
        temp_env = lmdb.open(db_path, readonly=True, lock=False, readahead=False, meminit=False)
        with temp_env.begin(write=False) as txn:
            self.length = int(txn.get(b"num_samples").decode())
            all_labels = []
            cursor = txn.cursor()
            for key, value in cursor:
                if key.decode().startswith("label_"):
                    all_labels.append(value.decode())
            self.classes = sorted(list(set(all_labels)))
            self.class_to_idx = {name: i for i, name in enumerate(self.classes)}
        temp_env.close()  # Сразу закрываем, чтобы не держать ресурсы
        print(f"Найдено уникальных классов: {len(self.classes)}")

    def _init_db(self):
        # Инициализируем соединение только один раз внутри воркера
        if self.env is None:
            self.env = lmdb.open(
                self.db_path,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False
            )

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        self._init_db()  # Гарантируем, что база открыта в текущем процессе

        string_id = f"{index:08d}"
        img_key = f"image_{string_id}".encode()
        label_key = f"label_{string_id}".encode()

        with self.env.begin(write=False) as txn:
            img_bytes = txn.get(img_key)
            label_text = txn.get(label_key).decode()
            img_buffer = np.frombuffer(img_bytes, dtype=np.uint8)
            image = cv2.imdecode(img_buffer, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            target_idx = self.class_to_idx[label_text]

        if self.transform:
            image = self.transform(image)

        return image, target_idx