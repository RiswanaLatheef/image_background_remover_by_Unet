from fastapi import APIRouter, File, UploadFile
from fastapi.responses import StreamingResponse
from io import BytesIO
from PIL import Image
from app.utils.image_processing import preprocess_image, postprocess_foreground
from app.utils.model import load_trained_model

# Load the model once when the app starts
model = load_trained_model("saved_model\\unet_background_removal.keras")

router = APIRouter()

@router.post("/remove_background/")
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

    # Convert the foreground image to a file-like object
    foreground_image_io = BytesIO()
    foreground_image.save(foreground_image_io, format="PNG")
    foreground_image_io.seek(0)

    return StreamingResponse(foreground_image_io, media_type="image/png")