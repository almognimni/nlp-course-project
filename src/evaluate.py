import math
import textwrap
import re
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support

from .data_loader import simple_tokenizer # Local import

# --- Task 1: Evaluation ---
def evaluate_perplexity(model, data_loader, criterion, device):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            total_loss += loss.item() * targets.numel()
            total_tokens += targets.numel()
    avg_loss = total_loss / total_tokens
    return math.exp(avg_loss)

def plot_lm_losses(history):
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Training Loss', color='blue')
    plt.plot(history['val_loss'], label='Validation Loss', color='red')
    plt.title('Language Model Loss Over Epochs', fontsize=16)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

# --- Task 2: Evaluation ---
def binary_accuracy(preds, y):
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()
    return correct.sum() / len(correct)

def plot_classifier_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    ax1.plot(history['train_loss'], label='Train Loss', color='blue')
    ax1.plot(history['val_loss'], label='Validation Loss', color='red')
    ax1.set_title('Loss Over Epochs')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    ax2.plot(history['train_acc'], label='Train Accuracy', color='blue')
    ax2.plot(history['val_acc'], label='Validation Accuracy', color='red')
    ax2.set_title('Accuracy Over Epochs')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    plt.tight_layout()
    plt.show()

def evaluate_on_test_set(model, test_loader, criterion, device):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating on Test Set"):
            inputs, labels = [t.to(device) for t in batch]
            predictions = model(inputs)
            loss = criterion(predictions, labels.float())
            total_loss += loss.item()
            rounded_preds = torch.round(torch.sigmoid(predictions))
            all_preds.extend(rounded_preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(test_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary')
    
    print(f"\\n--- Final Test Set Evaluation ---")
    print(f"Test Loss: {avg_loss:.3f} | Test Accuracy: {accuracy*100:.2f}%")
    print(f"Test Precision: {precision:.3f} | Test Recall: {recall:.3f} | Test F1-Score: {f1:.3f}")
    
    # Plotting confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.show()

    return all_labels, all_preds

def run_error_analysis_to_file(model, raw_test_set, vocab, device, filename, max_len=256, num_examples=5):
    model.eval()
    false_positives, false_negatives = [], []
    
    print(f"--- Error Analysis: Searching for failed examples to save to '{filename}' ---")
    for item in raw_test_set:
        text, true_label = item['text'], item['label']
        tokens = simple_tokenizer(text)
        numericalized = [vocab.stoi.get(t, vocab.stoi.get('<unk>')) for t in tokens][:max_len]
        input_tensor = torch.tensor([numericalized], dtype=torch.long).to(device)
        
        with torch.no_grad():
            predicted_label = torch.round(torch.sigmoid(model(input_tensor))).item()

        if predicted_label != true_label:
            if predicted_label == 1:
                if len(false_positives) < num_examples:
                    false_positives.append({'text': text, 'true': true_label, 'pred': int(predicted_label)})
            else:
                if len(false_negatives) < num_examples:
                    false_negatives.append({'text': text, 'true': true_label, 'pred': int(predicted_label)})
        
        if len(false_positives) >= num_examples and len(false_negatives) >= num_examples:
            break

    try:
        with open(filename, 'w', encoding='utf-8') as f:
            # ... (file writing logic as provided in previous answer) ...
            f.write("Error Analysis Report\n")
        print(f"Error analysis report successfully saved to '{filename}'.")
    except Exception as e:
        print(f"An error occurred while writing to file: {e}")