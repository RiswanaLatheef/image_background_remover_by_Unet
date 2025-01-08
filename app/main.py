from fastapi import FastAPI
from app.routers import background

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "API is working"}
# Include the background removal router
app.include_router(background.router)