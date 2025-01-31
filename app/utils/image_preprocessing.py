from PIL import Image
import numpy as np

def preprocess_image(image: Image.Image) -> np.ndarray:
    # Resize and normalize the image as required by your model
    image = image.resize((128, 128))  # Assuming the model input size is 128x128
    image = np.array(image) / 255.0
    return np.expand_dims(image, axis=0)

def postprocess_foreground(image: Image.Image, mask: np.ndarray) -> Image.Image:
    # Resize the mask to match the original image size
    mask_resized = np.array(Image.fromarray(mask.squeeze()).resize(image.size, Image.NEAREST))
    
    # Convert the mask to a binary mask (values of 0 and 1)
    mask_resized = mask_resized > 0.5  # If your mask values are between 0 and 1, adjust this threshold
    mask_resized = np.expand_dims(mask_resized, axis=-1)  # Add the channel dimension for image processing

    # Convert the image to a numpy array
    image_array = np.array(image)

    # Repeat the mask across the three channels (RGB)
    mask_resized_rgb = np.repeat(mask_resized, 3, axis=-1)

    # Apply the mask to the image (foreground will remain, background will be set to transparent)
    foreground = np.zeros_like(image_array)
    foreground[mask_resized_rgb] = image_array[mask_resized_rgb]  # Only keep the foreground where the mask is 1

    return Image.fromarray(foreground)