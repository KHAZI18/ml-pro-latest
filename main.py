# from fastapi import FastAPI, UploadFile, Form
# from pydantic import BaseModel
# import uvicorn
# import torch
# import librosa
# from transformers import BertTokenizer
# import io
# import numpy as np

# from model import AudioTextEmotionModel, predict_emotion, le  # import from saved module

# app = FastAPI()

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
# model = AudioTextEmotionModel(num_classes=len(le.classes_))
# model.load_state_dict(torch.load("audio_text_emotion_model.pt", map_location=device))
# model.to(device)
# model.eval()


# class TextOnly(BaseModel):
#     text: str


# @app.post("/predict")
# async def predict(text: str = Form(None), mode: str = Form(...), audio: UploadFile = None):
#     # Save and load audio if uploaded
#     audio_path = None
#     if audio:
#         contents = await audio.read()
#         audio_path = f"temp_audio.wav"
#         with open(audio_path, "wb") as f:
#             f.write(contents)

#     prediction = predict_emotion(model, tokenizer, le, text=text, audio_path=audio_path, mode=mode)
#     return {"emotion": prediction}


# if __name__ == "__main__":
#     uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)


from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware  # Add this import
from pydantic import BaseModel
import uvicorn
import torch
import librosa
from transformers import BertTokenizer
import io
import numpy as np

from model import AudioTextEmotionModel, predict_emotion, le  # import from saved module

app = FastAPI()

# Add CORS middleware to allow requests from your React app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development only! In production specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = AudioTextEmotionModel(num_classes=len(le.classes_))
model.load_state_dict(torch.load("audio_text_emotion_model.pt", map_location=device))
model.to(device)
model.eval()


class TextOnly(BaseModel):
    text: str


@app.post("/predict")
async def predict(text: str = Form(None), mode: str = Form(...), audio: UploadFile = None):
    # Save and load audio if uploaded
    audio_path = None
    if audio:
        contents = await audio.read()
        audio_path = f"temp_audio.wav"
        with open(audio_path, "wb") as f:
            f.write(contents)

    prediction = predict_emotion(model, tokenizer, le, text=text, audio_path=audio_path, mode=mode)
    return {"emotion": prediction}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)


# from fastapi import FastAPI, UploadFile, Form, File
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# import uvicorn
# import os
# from typing import Optional

# # Import the demo prediction function instead of loading the actual model
# from model import predict_emotion

# app = FastAPI()

# # Add CORS middleware to allow requests from your React app
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # For development only! In production specify your frontend domain
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# class TextOnly(BaseModel):
#     text: str

# @app.get("/")
# async def root():
#     return {"message": "Emotion Recognition API is running"}

# @app.post("/predict")
# async def predict(
#     mode: str = Form(...),
#     text: Optional[str] = Form(None),
#     audio: Optional[UploadFile] = File(None)
# ):
#     # Validate mode
#     if mode not in ["text", "audio", "both"]:
#         return {"error": "Invalid mode. Must be one of: text, audio, both"}
    
#     # Check required inputs based on mode
#     if mode in ["text", "both"] and not text:
#         return {"error": "Text input is required for text or both modes"}
    
#     if mode in ["audio", "both"] and not audio:
#         return {"error": "Audio file is required for audio or both modes"}
    
#     # Create temp directory if it doesn't exist
#     if not os.path.exists("temp"):
#         os.makedirs("temp")
    
#     # Save and load audio if uploaded
#     audio_path = None
#     if audio:
#         contents = await audio.read()
#         audio_path = f"temp/{audio.filename}"
#         with open(audio_path, "wb") as f:
#             f.write(contents)
    
#     # Perform prediction using the demo model
#     prediction = predict_emotion(text=text, audio_path=audio_path, mode=mode)
    
#     # Optional: clean up the temporary audio file
#     if audio_path and os.path.exists(audio_path):
#         os.remove(audio_path)
    
#     return {"emotion": prediction}

# if __name__ == "__main__":
#     uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)