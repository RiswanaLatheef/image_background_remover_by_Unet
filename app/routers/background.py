from fastapi import APIRouter, File, UploadFile,Response
from fastapi.responses import StreamingResponse,FileResponse
from io import BytesIO
from PIL import Image
from app.utils.image_preprocessing import preprocess_image, postprocess_foreground
from app.utils.model import load_trained_model
from starlette.background import BackgroundTask
from PIL import Image
import os


def cleanup():
    os.remove('tempfile.png')

# Load the model once when the app starts
model = load_trained_model("saved_model/unet_background_removal.keras")

router = APIRouter()


@router.post("/upload")
async def create_upload_file(files: list[UploadFile]):
    return {"filename": [file.filename for file in files]}

@router.post("/remove-background/")
async def remove_background(file: UploadFile = File(...)):
    # Read the uploaded image file
    image_data = await file.read()
    image = Image.open(BytesIO(image_data))

    # Preprocess image for prediction
    input_image = preprocess_image(image)

    # Predict background removal mask
    mask = model.predict(input_image)

    # Postprocess to extract the foreground
    foreground_image = postprocess_foreground(image, mask)
    foreground_image.save("tempfile.png")




    return FileResponse('tempfile.png', media_type="image/png",background=BackgroundTask(cleanup))