from flask import Flask, request, render_template, send_from_directory
import io
import torch
from torchvision import transforms
from flask_cors import CORS
import os, base64
from predict import Predictor
import tifffile as tiff
import numpy as np
from PIL import Image
import pandas as pd

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
CORS(app)

# Đọc CSV
df = pd.read_csv("rainfall.csv")

# Chuyển cột date sang datetime
df['date'] = pd.to_datetime(df['date'], format='mixed') # dayfirst=True vì định dạng d/m/yyyy

# Tạo cột year
df['year'] = df['date'].dt.year

model = torch.load('satelite_model.pth', map_location='cuda')
model.eval()
class_names = ['not_rain', 'medium_rain', 'heavy_rain']
transform = transforms.Compose([transforms.Resize((224,224)),
                                transforms.ToTensor()])
predictor = Predictor(model, transform, class_names)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    label = ""
    img_data = None  # để truyền ảnh Pillow dạng Base64
    if request.method == 'POST':
        if 'file' not in request.files:
            label = "No file part"
        else:
            file = request.files['file']
            if file.filename == "":
                label = "No selected file"
            else:
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(file_path)

                try:
                    if file_path.lower().endswith((".tif", ".tiff")):
                        img = tiff.imread(file_path).astype(np.float32)
                        img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)

                        if img.ndim == 3 and img.shape[2] >= 3:
                            img_rgb = img[..., [2,1,0]]  # BGR -> RGB
                        elif img.ndim == 2:
                            img_rgb = np.stack([img]*3, axis=-1)
                        else:
                            img_rgb = np.zeros((224,224,3), dtype=np.float32)

                        img_rgb = np.clip(img_rgb / 10000.0, 0, 1)
                        pil_img = Image.fromarray((img_rgb*255).astype(np.uint8))
                    else:
                        pil_img = Image.open(file_path).convert("RGB")

                    # Convert Pillow image sang Base64
                    buffered = io.BytesIO()
                    pil_img.save(buffered, format="PNG")
                    img_data = base64.b64encode(buffered.getvalue()).decode()

                except Exception as e:
                    print("Lỗi xử lý ảnh:", e)

                # Dự đoán nhãn
                label = predictor.predict(file_path)

    return render_template("index.html", label=label, img_data=img_data)

@app.route('/rainfall', methods=['GET', 'POST'])
def getRain():
    years = sorted(df['year'].unique())
    selected_year = None
    year_data = None

    if request.method == "POST":
        selected_year = int(request.form.get("year"))
        year_data = df[df['year'] == selected_year].sort_values('date')

    return render_template("rainfall.html", years=years, selected_year=selected_year, year_data=year_data)

app.run()

    