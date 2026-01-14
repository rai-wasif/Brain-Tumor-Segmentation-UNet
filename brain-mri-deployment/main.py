import torch
import numpy as np
import cv2
import io
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image
from model import UNET  # Imports the class from model.py

app = FastAPI(title="Brain Tumor Segmentation API")

# --- LOAD MODEL ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNET(in_channels=3, out_channels=1).to(device)

# Load the weights (The Muscle)
try:
    model.load_state_dict(torch.load("unet_brain_mri_best.pth", map_location=device))
    model.eval()
    print("✅ Model Loaded Successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")

# --- HELPER FUNCTION ---
def process_image(image_bytes):
    # 1. Read Image
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_np = np.array(image)

    # 2. Resize to 256x256 (Same as your training)
    image_resized = cv2.resize(image_np, (256, 256))

    # 3. Normalize (Same as Albumentations used in validation)
    # Your code used: A.Normalize(mean=(0.0, ...), std=(1.0, ...)) which just divides by 255 implicitly if using float
    img_tensor = image_resized.astype(np.float32) / 255.0  
    
    # 4. Transpose (H, W, C) -> (C, H, W)
    img_tensor = np.transpose(img_tensor, (2, 0, 1))
    
    # 5. Add Batch Dimension (1, C, H, W)
    img_tensor = torch.tensor(img_tensor).unsqueeze(0).to(device)
    
    return img_tensor, image_resized

@app.get("/")
def read_root():
    return {"message": "Brain Tumor API is Live! POST image to /predict"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read file
    contents = await file.read()
    input_tensor, original_img = process_image(contents)

    # Inference
    with torch.no_grad():
        output = model(input_tensor)
        output = torch.sigmoid(output) # Probability map
        
    # Process Output (Threshold > 0.5)
    mask = output.squeeze().cpu().numpy()
    mask = (mask > 0.5).astype(np.uint8) * 255

    # Create Red Heatmap Overlay
    heatmap = np.zeros_like(original_img)
    heatmap[:, :, 0] = mask  # Red channel (RGB) - If using PIL/FastAPI, it's usually RGB
    
    # In main.py, OpenCV usually reads as BGR, but PIL reads as RGB.
    # Since we used PIL, Channel 0 is Red.
    
    # Blend: Original 70%, Heatmap 30%
    overlay = cv2.addWeighted(original_img, 0.7, heatmap, 0.3, 0)

    # Convert to PNG for response
    res_image = Image.fromarray(overlay)
    io_buf = io.BytesIO()
    res_image.save(io_buf, format="PNG")
    io_buf.seek(0)

    return StreamingResponse(io_buf, media_type="image/png")