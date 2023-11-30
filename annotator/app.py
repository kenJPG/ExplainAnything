from fastapi import FastAPI, Request, File, UploadFile, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import cv2
import os
import numpy as np

app = FastAPI()

app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

UPLOAD_FOLDER = 'uploads/'
MASKS_FOLDER = 'masks/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MASKS_FOLDER, exist_ok=True)

templates = Jinja2Templates(directory="templates")

@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(file_path, "wb+") as f:
        f.write(file.file.read())
    return {"filename": file.filename}

@app.post("/save_mask/")
async def save_mask(mask_data: str = Form(...), filename: str = Form(...), width: int = Form(...), height: int = Form(...)):
    mask = np.array(mask_data.split(','), dtype=np.uint8).reshape(height, width)
    cv2.imwrite(os.path.join(MASKS_FOLDER, filename), mask)
    return {"message": "Saved"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)