import torch
import numpy as np
from PIL import Image
import tifffile as tiff
from torchvision import transforms
from imagetransform import ImageTransform

def predict_image(model, image_path, transform, class_names):
    model.eval()  # Đưa model về chế độ đánh giá
    device = next(model.parameters()).device  # Lấy device của model (CPU/GPU)

    try:
        # 1️⃣ Đọc ảnh .tif hoặc .jpg
        if image_path.lower().endswith(".tif") or image_path.lower().endswith(".tiff"):
            img = tiff.imread(image_path).astype(np.float32)

            # Đảm bảo có 3 kênh
            if img.ndim == 3 and img.shape[2] >= 3:
                img = img[..., [2, 1, 0]]  # Chuyển sang RGB nếu cần
            elif img.ndim == 2:
                img = np.stack([img] * 3, axis=-1)
            else:
                img = np.zeros((224, 224, 3), dtype=np.float32)

            img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
            img = np.clip(img / 10000.0, 0, 1)  # Scale Sentinel-2
            img = (img * 255).astype(np.uint8)
            pil_img = Image.fromarray(img)

        else:
            # Nếu là ảnh thường (jpg, png,…)
            pil_img = Image.open(image_path).convert("RGB")

    except Exception as e:
        print(f"Lỗi đọc ảnh {image_path}: {e}")
        return None


    if transform:
        img_tensor = transform(pil_img)
    else:
        img_tensor = transforms.ToTensor()(pil_img)

    #Thêm batch dimension
    img_tensor = img_tensor.unsqueeze(0).to(device)

    #Dự đoán
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()

    # 5️⃣ Trả kết quả
    predicted_class = class_names[pred_idx]
    print(f"Ảnh: {image_path}")
    print(f"Dự đoán: {predicted_class}")

    return predicted_class

class Predictor:
    def __init__(self, model_path, transform, class_names):
        self.model = model_path
        self.transform = transform
        self.class_names = class_names

    def predict(self, image_path):
        return predict_image(self.model, image_path, self.transform, self.class_names)


    
model = torch.load('satelite_model.pth', map_location='cuda')
model.eval()
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
class_names = ['not_rain', 'medium_rain', 'heavy_rain']
image_path = 'Dataset_split/test/medium_rain/Sentinel2_HCMC_20250705.tif'  # Thay bằng đường dẫn tới ảnh của bạn
predict = Predictor(model, transform, class_names)
predict.predict(image_path)