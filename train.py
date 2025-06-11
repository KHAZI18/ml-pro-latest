# IMPORTANT: RUN THIS CELL IN ORDER TO IMPORT YOUR KAGGLE
# THEN FEEL FREE TO DELETE THIS CELL.
# import kagglehub

# Assuming these datasets are already downloaded and their paths are correctly set up
# uwrfkaggler_ravdess_emotional_speech_audio_path = kagglehub.dataset_download('uwrfkaggler/ravdess-emotional-speech-audio')
# ejlok1_toronto_emotional_speech_set_tess_path = kagglehub.dataset_download('ejlok1/toronto-emotional-speech-set-tess')
# ejlok1_cremad_path = kagglehub.dataset_download('ejlok1/cremad')
# ybsingh_indian_emotional_speech_corpora_iesc_path = kagglehub.dataset_download('ybsingh/indian-emotional-speech-corpora-iesc')

# print('Data source import complete.')

import os
from pathlib import Path
import pandas as pd
import torch
import torchaudio
import librosa
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
from sklearn.metrics import mean_squared_error, r2_score


# VAD scores mapping based on ANEW for the emotion words
# (Valence, Arousal, Dominance) on a 1-9 scale
VAD_MAPPING = {
    'A': (2.98, 7.02, 6.38), # Angry
    'F': (2.50, 7.41, 3.69), # Fear
    'H': (8.21, 6.49, 6.26), # Happy
    'N': (5.16, 3.88, 5.00), # Neutral
    'S': (2.32, 3.47, 2.98), # Sad
    'D': (3.21, 5.65, 3.45), # Disgust
    'E': (7.25, 6.60, 5.90), # Excited
    'C': (6.95, 3.80, 6.25), # Calm
    'B': (4.30, 4.50, 4.10), # Boredom
    'L': (3.70, 3.10, 2.80), # Lonely
    'T': (6.50, 6.00, 6.20), # Tenderness
    'R': (6.80, 5.80, 5.50), # Relaxed
    'P': (7.80, 7.20, 6.90), # Proud
    'I': (5.90, 6.10, 5.30), # Interest
    'J': (8.00, 7.00, 7.00), # Joy
    'W': (3.20, 4.25, 3.50), # Worry
    'O': (4.50, 3.50, 3.80), # Optimistic
    'U': (2.70, 4.70, 3.20), # Uncertainty
    'Z': (6.10, 4.30, 5.00)  # Surprise
}

# Mapping from emotion code to folder name
EMOTION_FOLDER_MAP = {
    'A': 'Anger',
    'F': 'Fear',
    'H': 'Happy',
    'N': 'Neutral',
    'S': 'Sad',
    'D': 'Disgust',
    'E': 'Excited',
    'C': 'Calm',
    'B': 'Boredom',
    'L': 'Lonely',
    'T': 'Tenderness',
    'R': 'Relaxed',
    'P': 'Proud',
    'I': 'Interest',
    'J': 'Joy',
    'W': 'Worry',
    'O': 'Optimistic',
    'U': 'Uncertainty',
    'Z': 'Surprise'
}

def parse_ravdess_filename(filename):
    """
    Parses RAVDESS dataset filenames to extract emotion codes.
    """
    try:
        emotion_id = int(filename.split('-')[2])
        emotion_map = {
            1: 'N', # Neutral
            2: 'C', # Calm
            3: 'H', # Happy
            4: 'S', # Sad
            5: 'A', # Angry
            6: 'F', # Fear
            7: 'D', # Disgust
            8: 'E'  # Surprise/Excited
        }
        return emotion_map.get(emotion_id, None)
    except Exception as e:
        return None

def parse_tess_filename(filename):
    """
    Parses TESS dataset filenames to extract emotion codes and speaker IDs.
    """
    try:
        arr = filename.split(' ')
        emotion_id = arr[2].split('.')[0].lower()
        speaker = arr[0]
        emotion_map = {
            'angry': 'A',
            'fear': 'F',
            'happy': 'H',
            'neutral': 'N',
            'sad': 'S',
            'disgust': 'D',
            'ps': 'Z'
        }
        return emotion_map.get(emotion_id, None), speaker
    except Exception as e:
        return None

def parse_cremad_filename(filename):
    """
    Parses CREMA-D dataset filenames to extract emotion codes and speaker IDs.
    """
    try:
        arr = filename.split('_')
        emotion_id = arr[2].split('.')[0].lower()
        speaker = arr[0]
        emotion_map = {
            'ang': 'A',
            'fea': 'F',
            'hap': 'H',
            'neu': 'N',
            'sad': 'S',
            'dis': 'D',
        }
        return emotion_map.get(emotion_id, None), speaker
    except Exception as e:
        return None

def audio_to_vad_score_mapping(root_dir, output_file, dataset):
    """
    Scans the dataset directory, extracts emotion codes,
    converts it to pandas Dataframe and writes it to CSV.
    """
    record = []
    count_processed = 0
    count_skipped = 0

    root_path = Path(root_dir)
    if not root_path.exists():
        print(f"Error: Root directory '{root_dir}' not found")
        return None

    print("Scanning Directory", root_path.resolve())

    processed_files = set()

    for file_path in root_path.rglob("*.wav"):
        filename = file_path.name

        if filename in processed_files:
            continue
        processed_files.add(filename)

        try:
            emotion_code = None
            filepath = file_path
            speaker = None
            emotion = None

            if dataset == "IESC":
                emotion_code = filename[0]
                speaker = file_path.parents[1].name
                emotion = EMOTION_FOLDER_MAP[emotion_code]
            elif dataset == 'RAVDESS':
                emotion_code = parse_ravdess_filename(filename)
                speaker = file_path.parents[0].name
                if emotion_code: # Ensure emotion_code is not None before mapping
                    emotion = EMOTION_FOLDER_MAP[emotion_code]
            elif dataset == 'TESS':
                emotion_code, speaker = parse_tess_filename(filename)
                if emotion_code: # Ensure emotion_code is not None before mapping
                    emotion = EMOTION_FOLDER_MAP[emotion_code]
            elif dataset == 'CREMA-D':
                emotion_code, speaker = parse_cremad_filename(filename)
                if emotion_code: # Ensure emotion_code is not None before mapping
                    emotion = EMOTION_FOLDER_MAP[emotion_code]

            if emotion_code in VAD_MAPPING:
                valence = VAD_MAPPING[emotion_code][0]
                arousal = VAD_MAPPING[emotion_code][1]
                dominance = VAD_MAPPING[emotion_code][2]
                record.append({
                    'filename': filename,
                    'filepath': str(filepath), # Store as string for CSV
                    'speaker': speaker,
                    'emotion': emotion,
                    'valence': valence,
                    'arousal': arousal,
                    'dominance': dominance,
                })
                count_processed += 1
            else:
                print(f"Emotion code not found for {filename}")
                count_skipped += 1

        except Exception as e:
            print(f"Error Processing {filename}: {e}")
            print("Skipping")
            count_skipped += 1

    if record:
        df = pd.DataFrame(record)
        sorted_df = df.set_index(['speaker', 'filename'])
        sorted_df_by_index = sorted_df.sort_index()
        sorted_df_by_index.to_csv(output_file)
        print(f"\nSuccessfully processed {count_processed} files. Skipped {count_skipped} files.")
        return sorted_df_by_index
    else:
        print(f"No records processed for {root_dir}")
        return pd.DataFrame()


def prepare_dataset(data, test_size=0.15, val_size=0.15):
    """
    Splits the dataframe into training, validation, and test sets.
    """
    train_df, temp_df = train_test_split(data, test_size=test_size + val_size, random_state=42)
    relative_val_size = val_size / (test_size + val_size)
    val_df, test_df = train_test_split(temp_df, test_size=relative_val_size, random_state=42)

    print(f"Train set: {len(train_df)} samples")
    print(f"Validation set: {len(val_df)} samples")
    print(f"Test set: {len(test_df)} samples")

    return train_df, val_df, test_df

def load_audio(file_path, target_sr=16000):
    """
    Loads an audio file, resamples it, converts to mono, normalizes, and converts to a PyTorch tensor.
    """
    try:
        waveform, sr = librosa.load(file_path, sr=target_sr)
        if len(waveform.shape) > 1:
            waveform = librosa.to_mono(waveform)

        waveform = waveform / np.max(np.abs(waveform)) # Normalizing the audio between -1 and 1
        waveform = torch.tensor(waveform).float() # Convert to tensor
        return waveform, target_sr
    except Exception as e:
        print(f"Error loading audio {file_path}: {e}")
        return None, None

class VADDataset(Dataset):
    """
    Dataset for loading audio files and their VAD labels.
    """
    def __init__(self, df, processor, audio_max_length=16000 * 5):  # 5 seconds audio
        self.df = df
        self.processor = processor
        self.audio_max_length = audio_max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filepath = Path(row['filepath'])

        waveform, sr = load_audio(filepath)

        if waveform is None:
            waveform = torch.zeros(16000)
            sr = 16000

        inputs = self.processor(
            waveform,
            sampling_rate=sr,
            return_tensors='pt',
            padding="max_length",
            max_length=self.audio_max_length,
            truncation=True
        )

        input_values = inputs.input_values.squeeze()
        if hasattr(inputs, 'attention_mask'):
            attention_mask = inputs.attention_mask.squeeze()
        else:
            attention_mask = torch.ones_like(input_values)

        labels = {
            'valence': torch.tensor(row['valence'], dtype=torch.float32),
            'arousal': torch.tensor(row['arousal'], dtype=torch.float32),
            'dominance': torch.tensor(row['dominance'], dtype=torch.float32),
        }

        return {
            'input_values': input_values,
            'attention_mask': attention_mask,
            'labels': labels,
            'filename': row['filename']
        }

class VADPredictor(nn.Module):
    """Model to predict VAD Scores"""
    def __init__(self, pretrained_model_name="facebook/wav2vec2-base", freeze_feature_extractor=True):
        super(VADPredictor, self).__init__()

        self.wav2vec2 = Wav2Vec2Model.from_pretrained(pretrained_model_name)

        if freeze_feature_extractor:
            for param in self.wav2vec2.feature_extractor.parameters():
                param.requires_grad = False

        hidden_size = self.wav2vec2.config.hidden_size

        self.valence_layers = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.Linear(64, 1)
        )

        self.arousal_layers = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.Linear(64, 1)
        )

        self.dominance_layers = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.Linear(64, 1)
        )

    def forward(self, input_values, attention_mask=None):
        outputs = self.wav2vec2(input_values, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state

        pooled_output = torch.mean(last_hidden_state, dim=1)

        valence = self.valence_layers(pooled_output)
        arousal = self.arousal_layers(pooled_output)
        dominance = self.dominance_layers(pooled_output)

        return {
            'valence': valence.squeeze(-1),
            'arousal': arousal.squeeze(-1),
            'dominance': dominance.squeeze(-1)
        }

def get_feature_extractor(model_name="facebook/wav2vec2-base"):
    return Wav2Vec2FeatureExtractor.from_pretrained(model_name)

def initialize_model(device='cuda' if torch.cuda.is_available() else 'cpu'):
    model = VADPredictor()
    model = model.to(device)
    print(f"Model initialized on {device}")
    return model

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    epoch_loss = 0
    progress_bar = tqdm(dataloader, desc="Training")
    for batch in progress_bar:
        if batch is None:
            continue

        input_values = batch['input_values'].to(device)
        attention_mask = batch.get('attention_mask')
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        valence_label = batch['labels']['valence'].to(device)
        arousal_label = batch['labels']['arousal'].to(device)
        dominance_label = batch['labels']['dominance'].to(device)

        optimizer.zero_grad()

        outputs = model(input_values, attention_mask=attention_mask)

        valence_loss = criterion(outputs['valence'], valence_label)
        arousal_loss = criterion(outputs['arousal'], arousal_label)
        dominance_loss = criterion(outputs['dominance'], dominance_label)

        loss = valence_loss + arousal_loss + dominance_loss
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        progress_bar.set_postfix({"loss": loss.item()})
    return epoch_loss / len(dataloader)

def validate(model, dataloader, criterion, device):
    model.eval()
    val_loss = 0
    all_preds = {'valence': [], 'arousal': [], 'dominance': []}
    all_labels = {'valence': [], 'arousal': [], 'dominance': []}

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            input_values = batch['input_values'].to(device)
            attention_mask = batch.get('attention_mask')
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            valence_label = batch['labels']['valence'].to(device)
            arousal_label = batch['labels']['arousal'].to(device)
            dominance_label = batch['labels']['dominance'].to(device)

            outputs = model(input_values, attention_mask=attention_mask)

            valence_loss = criterion(outputs['valence'], valence_label)
            arousal_loss = criterion(outputs['arousal'], arousal_label)
            dominance_loss = criterion(outputs['dominance'], dominance_label)

            loss = valence_loss + arousal_loss + dominance_loss
            val_loss += loss.item()

            all_preds['valence'].extend(outputs['valence'].cpu().numpy())
            all_preds['arousal'].extend(outputs['arousal'].cpu().numpy())
            all_preds['dominance'].extend(outputs['dominance'].cpu().numpy())
            all_labels['valence'].extend(valence_label.cpu().numpy())
            all_labels['arousal'].extend(arousal_label.cpu().numpy())
            all_labels['dominance'].extend(dominance_label.cpu().numpy())

    metrics = {}
    for dimension in ['valence', 'arousal', 'dominance']:
        mse = mean_squared_error(all_labels[dimension], all_preds[dimension])
        r2 = r2_score(all_labels[dimension], all_preds[dimension])
        metrics[f'{dimension}_mse'] = mse
        metrics[f'{dimension}_r2'] = r2
    metrics['avg_loss'] = val_loss / len(dataloader)
    return metrics

def train_model(model, train_loader, val_loader, num_epochs, lr, device):
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_metrics': []
    }

    best_val_loss = float('inf')
    best_model_state = None

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = validate(model, val_loader, criterion, device)
        val_loss = val_metrics['avg_loss']

        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"Valence MSE: {val_metrics['valence_mse']:.4f}, R2: {val_metrics['valence_r2']:.4f}")
        print(f"Arousal MSE: {val_metrics['arousal_mse']:.4f}, R2: {val_metrics['arousal_r2']:.4f}")
        print(f"Dominance MSE: {val_metrics['dominance_mse']:.4f}, R2: {val_metrics['dominance_r2']:.4f}")

        scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_metrics'].append(val_metrics)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            print(f"New best model saved with validation loss: {best_val_loss:.4f}")

    if best_model_state:
        model.load_state_dict(best_model_state)
    return model, history

def plot_training_history(history):
    """Plot the training and validation loss over epochs and R² values."""
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')

    plt.subplot(1, 2, 2)
    val_r2_valence = [metrics['valence_r2'] for metrics in history['val_metrics']]
    val_r2_arousal = [metrics['arousal_r2'] for metrics in history['val_metrics']]
    val_r2_dominance = [metrics['dominance_r2'] for metrics in history['val_metrics']]
    plt.plot(val_r2_valence, label='Valence R²')
    plt.plot(val_r2_arousal, label='Arousal R²')
    plt.plot(val_r2_dominance, label='Dominance R²')
    plt.xlabel('Epoch')
    plt.ylabel('R²')
    plt.legend()
    plt.title('Validation R² Score')
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()


# --- Main execution flow ---

# Define your input directory for the IESC dataset
input_dir = "C:\\Users\\FAIZAN KHAZI\\Desktop\\ML Project\\iesc\\dataset"
# input_dir = "iesc\dataset"
# Process IESC dataset and create its CSV
print("Processing IESC dataset...")
iesc_df = audio_to_vad_score_mapping(input_dir, "labeled_dataset_iesc.csv", "IESC")
if iesc_df is None:
    print("Failed to process IESC dataset. Exiting.")
    exit()

# List of your CSV file paths (include your newly generated IESC CSV)
csv_files = [
    "labeled_dataset_iesc.csv",
    # Add other CSVs if you have them and want to combine, e.g.:
    # "/kaggle/working/labeled_dataset_CREMA-D.csv",
    # "/kaggle/working/labeled_dataset_ravdess.csv",
    # "/kaggle/working/labeled_dataset_tess.csv"
]

# Lists to store results from each CSV
train_dfs = []
test_dfs = []
val_dfs = []

# Loop through each CSV and split into train/val/test
for file_path in csv_files:
    try:
        df = pd.read_csv(file_path)
        train_df_single, val_df_single, test_df_single = prepare_dataset(df)
        train_dfs.append(train_df_single)
        test_dfs.append(test_df_single)
        val_dfs.append(val_df_single)
    except FileNotFoundError:
        print(f"Warning: CSV file not found at {file_path}. Skipping.")
    except Exception as e:
        print(f"Error processing {file_path}: {e}. Skipping.")


# Combine all the datasets
if not train_dfs:
    print("No valid datasets processed. Exiting.")
    exit()

combined_train_df = pd.concat(train_dfs, ignore_index=True)
combined_val_df = pd.concat(val_dfs, ignore_index=True)
combined_test_df = pd.concat(test_dfs, ignore_index=True)

# Assign combined dataframes to the variables used later
train_df = combined_train_df
val_df = combined_val_df
test_df = combined_test_df


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize feature extractor and model
feature_extractor = get_feature_extractor()
model = initialize_model(device=device)

print("Creating data loaders...")
train_dataset = VADDataset(
    train_df,
    feature_extractor,
)
val_dataset = VADDataset(
    val_df,
    feature_extractor,
)
test_dataset = VADDataset(
    test_df,
    feature_extractor,
)

# Creating dataset loaders
train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=8
)
test_loader = DataLoader(
    test_dataset,
    batch_size=8,
)

# Train the model
model, history = train_model(
    model,
    train_loader,
    val_loader,
    num_epochs=1, # You can adjust this value
    lr=1e-5,      # You can adjust this value
    device=device
)

# Save the trained model
torch.save(model.state_dict(), 'vad_predictor_model.pt')

# Plot training history
plot_training_history(history)