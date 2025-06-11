from sklearn.preprocessing import LabelEncoder
import joblib

emotions = ["anger", "happy", "sad", "neutral", "fear"]

le = LabelEncoder()
le.fit(emotions)
joblib.dump(le, "label_encoder.pkl")
print("Saved label_encoder.pkl")
