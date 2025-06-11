import torch
import torch.nn as nn
# Assuming model, test_loader, and device are already defined from your training script
# If not, you'd need to re-initialize model, load its state_dict, and create test_loader
# For simplicity, here's how you'd typically do it after running the full script:

# Re-initialize the model structure
# model = initialize_model(device=device) # Assuming initialize_model is available
model.load_state_dict(torch.load('vad_predictor_model.pt'))
model.eval() # Set the model to evaluation mode

print("\nEvaluating model on the test set...")
# Assuming validate function is defined as in the script
test_metrics = validate(model, test_loader, nn.MSELoss(), device)

print(f"Test Set Evaluation Results:")
print(f"Valence MSE: {test_metrics['valence_mse']:.4f}, R2: {test_metrics['valence_r2']:.4f}")
print(f"Arousal MSE: {test_metrics['arousal_mse']:.4f}, R2: {test_metrics['arousal_r2']:.4f}")
print(f"Dominance MSE: {test_metrics['dominance_mse']:.4f}, R2: {test_metrics['dominance_r2']:.4f}")