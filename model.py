# # model.py
# import torch
# import torch.nn as nn
# import librosa
# import numpy as np
# from transformers import BertModel, BertTokenizer
# import joblib

# # Load Label Encoder (must be saved from training)
# le = joblib.load("label_encoder.pkl")
# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# class AudioTextEmotionModel(nn.Module):
#     def __init__(self, num_classes):
#         super(AudioTextEmotionModel, self).__init__()

#         self.bert = BertModel.from_pretrained("bert-base-uncased")
#         self.text_fc = nn.Linear(self.bert.config.hidden_size, 256)

#         self.audio_cnn = nn.Conv1d(in_channels=13, out_channels=64, kernel_size=3, stride=1)
#         self.audio_fc = nn.Linear(64, 256)

#         self.classifier = nn.Linear(512, num_classes)
#         self.text_only_classifier = nn.Linear(256, num_classes)
#         self.audio_only_classifier = nn.Linear(256, num_classes)

#     def forward(self, input_ids=None, attention_mask=None, audio=None, mode="both"):
#         if mode == "text":
#             text_output = self.bert(input_ids=input_ids, attention_mask=attention_mask).pooler_output
#             text_feat = self.text_fc(text_output)
#             return self.text_only_classifier(text_feat)

#         elif mode == "audio":
#             audio_feat = self.audio_cnn(audio)
#             audio_feat = torch.mean(audio_feat, dim=2)
#             audio_feat = self.audio_fc(audio_feat)
#             return self.audio_only_classifier(audio_feat)

#         elif mode == "both":
#             text_output = self.bert(input_ids=input_ids, attention_mask=attention_mask).pooler_output
#             text_feat = self.text_fc(text_output)

#             audio_feat = self.audio_cnn(audio)
#             audio_feat = torch.mean(audio_feat, dim=2)
#             audio_feat = self.audio_fc(audio_feat)

#             combined = torch.cat((text_feat, audio_feat), dim=1)
#             return self.classifier(combined)

# def predict_emotion(model, text=None, audio_path=None, mode="both"):
#     model.eval()
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     if mode not in ["text", "audio", "both"]:
#         raise ValueError("Invalid mode")

#     input_ids = attention_mask = audio_tensor = None

#     if mode in ["text", "both"]:
#         tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
#         input_ids = tokens["input_ids"].to(device)
#         attention_mask = tokens["attention_mask"].to(device)

#     if mode in ["audio", "both"]:
#         audio, _ = librosa.load(audio_path, sr=16000)
#         mfcc = librosa.feature.mfcc(y=audio, sr=16000, n_mfcc=13).T
#         mfcc_tensor = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0).permute(0, 2, 1).to(device)
#         audio_tensor = mfcc_tensor

#     with torch.no_grad():
#         outputs = model(input_ids=input_ids, attention_mask=attention_mask, audio=audio_tensor, mode=mode)
#         _, predicted = torch.max(outputs, dim=1)

#     return le.inverse_transform(predicted.cpu().numpy())[0]


# model.py
import joblib
import numpy as np
import re

# Load Label Encoder if available
try:
    le = joblib.load("label_encoder.pkl")
except:
    # Create a mock label encoder if the file doesn't exist
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    le.classes_ = np.array(["anger", "happy", "sad", "neutral", "fear"])

class DemoEmotionModel:
    """
    Demo model that simulates emotion prediction without requiring actual PyTorch models
    """
    def __init__(self):
        self.emotions = ["anger", "happy", "sad", "neutral", "fear"]
        self.emotion_keywords = {
            "anger": ["angry", "furious", "mad", "rage", "irritated", "annoyed", "enraged", "upset"],
            "happy": ["happy", "joy", "excited", "great", "wonderful", "excellent", "good", "positive", "elated", "cheerful", "thrilled", "love", "appreciat", "fond", "adorable"],
            "sad": ["sad", "unhappy", "depressed", "down", "heartbroken", "mournful", "grief", "sorry", "miserable", "unfortunate"],
            "fear": ["fear", "afraid", "scared", "terrified", "nervous", "anxious", "worried", "frighten", "dread", "panic"],
            "neutral": ["okay", "fine", "neutral", "meh", "whatever", "indifferent", "normal", "ambivalent", "confused", "uncertain"],
        }
    
    def predict_emotion(self, text=None, audio_path=None, mode="both"):
        """
        Predict emotion based on text input, audio file path, or both
        """
        if mode == "text" or mode == "both":
            if text and len(text.strip()) > 0:
                return self._predict_from_text(text)
            
        if mode == "audio" or mode == "both":
            if audio_path:
                # For demo purposes, we'll use a simple heuristic based on filename
                # In a real implementation, you'd analyze the audio features
                return self._predict_from_audio(audio_path)
        
        # Default to neutral if we can't determine
        return "neutral"
    
    def _predict_from_text(self, text):
        """
        Predict emotion from text using keyword matching
        """
        text = text.lower()
        scores = {}
        
        # Calculate score for each emotion based on keyword matches
        for emotion, keywords in self.emotion_keywords.items():
            score = 0
            for keyword in keywords:
                # Count occurrences of each keyword
                matches = len(re.findall(r'\b' + keyword + r'\w*\b', text))
                score += matches
            
            scores[emotion] = score
        
        # If no keywords were matched, use a simple heuristic
        if sum(scores.values()) == 0:
            # Check for exclamation marks (might indicate excitement or anger)
            if "!" in text:
                if "?" in text:
                    return "fear"  # "!?" might indicate confusion or fear
                return "happy" if len(text) < 50 else "anger"
            
            # Check for question marks (might indicate confusion)
            if "?" in text:
                return "neutral"
            
            # Check sentence length (longer sentences might indicate more complex emotions)
            if len(text) > 100:
                return "sad"  # Longer texts tend to be more explanatory, often for negative emotions
            
            return "neutral"
        
        # Return the emotion with the highest score
        return max(scores, key=scores.get)
    
    def _predict_from_audio(self, audio_path):
        """
        For demo purposes, predict based on filename
        In a real implementation, you would extract audio features and use a model
        """
        audio_file = audio_path.lower()
        
        for emotion in self.emotions:
            if emotion in audio_file:
                return emotion
        
        # Simple fallback based on file characteristics
        import os
        file_size = os.path.getsize(audio_path)
        
        # Very arbitrary rules for demo purposes only
        if file_size < 10000:  # Small file
            return "neutral"
        elif file_size < 50000:  # Medium file
            return "happy"
        else:  # Large file
            return "fear"  # Assuming longer audio might contain more emotional content

# Create a singleton instance of the demo model
demo_model = DemoEmotionModel()

def predict_emotion(model=None, tokenizer=None, le=None, text=None, audio_path=None, mode="both"):
    """
    Wrapper function that provides the same interface as the original predict_emotion function
    but uses the demo model instead
    """
    return demo_model.predict_emotion(text=text, audio_path=audio_path, mode=mode)