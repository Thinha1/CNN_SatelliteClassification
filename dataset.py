from lib import *
from imagetransform import ImageTransform
# Cấu hình GPU nếu có
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Dataset class đọc file .tif
class SatelliteDataset(Dataset):
    def __init__(self, base_dir, transform=None):
        self.files = []
        self.labels = []
        self.transform = transform

        # Duyệt qua 3 class: mưa và không mưa
        for label, folder in enumerate(['not_rain', 'medium_rain', 'heavy_rain']):
            folder_path = os.path.join(base_dir, folder)
            for file in os.listdir(folder_path):
                if file.lower().endswith('.tif'):
                    self.files.append(os.path.join(folder_path, file))
                    self.labels.append(label)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        label = self.labels[idx]

        try:
            img = tiff.imread(path).astype(np.float32)

            if img.ndim == 3 and img.shape[2] >= 3:
                img = img[..., [2, 1, 0]]  # chuyển sang RGB nếu cần
            elif img.ndim == 2:  # ảnh xám
                img = np.stack([img] * 3, axis=-1)
            else:
                # Nếu đọc lỗi hoặc sai kích thước, tạo ảnh đen
                img = np.zeros((224, 224, 3), dtype=np.float32)

        except Exception as e:
            print(f"Lỗi đọc ảnh {path}: {e}")
            img = np.zeros((224, 224, 3), dtype=np.float32)

        img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)

        # Scale dữ liệu Sentinel-2 (giá trị 0–10000 → 0–1)
        img = np.clip(img / 10000.0, 0, 1)

        # Đưa về kiểu uint8 (0–255) để dùng được với PIL
        img = (img * 255).astype(np.uint8)

        # Chuyển sang ảnh PIL
        img = Image.fromarray(img)

        # Áp dụng transform (nếu có)
        if self.transform:
            img = self.transform(img)

        return img, label