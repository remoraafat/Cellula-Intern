from flask import Flask, render_template, request, jsonify
import os, uuid
import torch
import numpy as np
from PIL import Image
import rasterio
from rasterio.enums import Resampling

# === Configuration ===
app = Flask(__name__)
UPLOAD_FOLDER = "static/results"
MODEL_PATH = "Models/best_model_unetpp_full.pth"
IMG_HEIGHT, IMG_WIDTH = 128, 128
SELECTED_BANDS = [2, 3, 4, 5, 6, 7, 10, 11]

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# === Load model ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
model.eval()

# === Helpers ===
def preprocess_image(path):
    with rasterio.open(path) as src:
        img = src.read(out_shape=(src.count, IMG_HEIGHT, IMG_WIDTH), resampling=Resampling.nearest)
        img = img.astype(np.float32)
        img = img[SELECTED_BANDS, :, :]
        return torch.tensor(img).unsqueeze(0)

# === Routes ===
@app.route("/")
def index():
    return render_template("upload.html")

@app.route("/upload_preview", methods=["POST"])
def upload_preview():
    file = request.files["file"]
    uid = uuid.uuid4().hex
    tif_path = os.path.join(UPLOAD_FOLDER, f"{uid}.tif")
    file.save(tif_path)

    # Save RGB preview
    with rasterio.open(tif_path) as src:
        rgb = src.read([4, 3, 2], out_shape=(3, IMG_HEIGHT, IMG_WIDTH), resampling=Resampling.nearest)
        rgb = np.transpose(rgb, (1, 2, 0))
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8) * 255
        rgb_img = Image.fromarray(rgb.astype(np.uint8))
        rgb_file = f"{uid}_rgb.png"
        rgb_img.save(os.path.join(UPLOAD_FOLDER, rgb_file))

    return jsonify({"id": uid, "rgb_file": rgb_file})

@app.route("/predict_all", methods=["POST"])
def predict_all():
    file_ids = request.form.get("file_ids", "").split(",")
    results = []

    for uid in file_ids:
        tif_path = os.path.join(UPLOAD_FOLDER, f"{uid}.tif")
        rgb_file = f"{uid}_rgb.png"
        mask_file = f"{uid}_mask.png"

        img_tensor = preprocess_image(tif_path).to(DEVICE)
        with torch.no_grad():
            output = model(img_tensor)
            prob = torch.sigmoid(output)
            mask = (prob > 0.5).float().squeeze().cpu().numpy()
            inverted = (1.0 - mask) * 255
            Image.fromarray(inverted.astype(np.uint8)).save(os.path.join(UPLOAD_FOLDER, mask_file))

        results.append({
            "rgb_file": rgb_file,
            "mask_file": mask_file
        })

    return render_template("results.html", results=results)

# === Run ===
if __name__ == "__main__":
    app.run(debug=True)
