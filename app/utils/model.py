from tensorflow.keras.models import load_model # type: ignore

def load_trained_model(model_path: str):
    # Load the trained model
    return load_model("saved_model\\unet_background_removal.keras")