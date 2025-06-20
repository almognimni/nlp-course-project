# main.py

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from datasets import load_dataset
import gensim.downloader as api
from tqdm import tqdm
import os

# --- Import from src ---
from src import config
from src.data_loader import (
    Vocabulary, LanguageModelDataset, PadCollate,
    ClassificationDataset, ClassificationCollate
)
from src.model import LanguageModel, SentimentClassifier, SentimentClassifierW2V
from src.train import train_language_model, train_and_evaluate_classifier
from src.evaluate import evaluate_perplexity, plot_lm_losses, evaluate_on_test_set, run_error_analysis_to_file

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def run_task_1():
    """
    Handles the setup, training, and evaluation for Task 1: Language Modeling.
    """
    print("--- Running Task 1: Language Modeling ---")
    set_seed(config.SEED)

    # 1. Load and split data
    imdb_dataset = load_dataset("imdb")
    original_train_data = imdb_dataset['train']
    train_set, val_set = original_train_data.train_test_split(test_size=0.1, seed=config.SEED).values()
    test_set = imdb_dataset['test']

    # 2. Build and save vocabulary
    vocab = Vocabulary(min_freq=config.MIN_VOCAB_FREQ)
    vocab.build_vocabulary(train_set['text'])
    
    # Create the 'models' directory if it doesn't exist
    os.makedirs('models', exist_ok=True)

    torch.save(vocab, config.VOCAB_SAVE_PATH)
    print(f"Vocabulary saved to {config.VOCAB_SAVE_PATH}")

    # 3. Create DataLoaders
    train_dataset = LanguageModelDataset(train_set, vocab)
    val_dataset = LanguageModelDataset(val_set, vocab)
    test_dataset = LanguageModelDataset(test_set, vocab)

    pad_idx = vocab.stoi["<pad>"]
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.LM_BATCH_SIZE, shuffle=True, collate_fn=PadCollate(pad_idx))
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.LM_BATCH_SIZE, shuffle=False, collate_fn=PadCollate(pad_idx))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.LM_BATCH_SIZE, shuffle=False, collate_fn=PadCollate(pad_idx))
    
    # 4. Initialize model, optimizer, and loss
    model = LanguageModel(
        vocab_size=len(vocab), 
        embedding_dim=config.LM_EMBEDDING_DIM,
        hidden_dim=config.LM_HIDDEN_DIM, 
        num_layers=config.LM_NUM_LAYERS
    ).to(config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=config.LM_LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    # 5. Train the model
    history = train_language_model(model, train_loader, val_loader, optimizer, criterion, config.DEVICE, config)
    plot_lm_losses(history)

    # 6. Save the final model
    torch.save(model.state_dict(), config.LM_MODEL_SAVE_PATH)
    print(f"Language model saved to {config.LM_MODEL_SAVE_PATH}")

    # 7. Final evaluation
    test_perplexity = evaluate_perplexity(model, test_loader, criterion, config.DEVICE)
    print(f"\nFinal Test Perplexity: {test_perplexity:.2f}")

def run_task_2a():
    """
    Handles Experiment A: LM as a frozen backbone.
    """
    print("\n--- Running Task 2, Experiment A ---")
    set_seed(config.SEED)

    # 1. Load assets
    vocab = torch.load(config.VOCAB_SAVE_PATH, weights_only=False)
    lm_model = LanguageModel(len(vocab), config.LM_EMBEDDING_DIM, config.LM_HIDDEN_DIM, config.LM_NUM_LAYERS)
    lm_model.load_state_dict(torch.load(config.LM_MODEL_SAVE_PATH))
    lm_model.to(config.DEVICE)

    # 2. Prepare data
    imdb_dataset = load_dataset('imdb')
    original_train_data = imdb_dataset['train']
    
    # Create a stratified subset. train_test_split returns a DatasetDict.
    temp_split = original_train_data.train_test_split(train_size=config.CLF_A_SUBSET_SIZE, stratify_by_column='label', seed=config.SEED)
    subset = temp_split['train'] # The subset is in the 'train' key

    # Now split this subset into train and validation sets
    final_split = subset.train_test_split(test_size=0.2, stratify_by_column='label', seed=config.SEED)
    cls_train_set, cls_val_set = final_split['train'], final_split['test']
    
    cls_test_set = imdb_dataset['test']

    pad_idx = vocab.stoi["<pad>"]
    cls_train_loader = torch.utils.data.DataLoader(ClassificationDataset(cls_train_set, vocab), batch_size=config.CLF_A_BATCH_SIZE, shuffle=True, collate_fn=ClassificationCollate(pad_idx))
    cls_val_loader = torch.utils.data.DataLoader(ClassificationDataset(cls_val_set, vocab), batch_size=config.CLF_A_BATCH_SIZE, shuffle=False, collate_fn=ClassificationCollate(pad_idx))
    cls_test_loader = torch.utils.data.DataLoader(ClassificationDataset(cls_test_set, vocab), batch_size=config.CLF_A_BATCH_SIZE, shuffle=False, collate_fn=ClassificationCollate(pad_idx))

    # 3. Initialize model, freeze backbone
    model_A = SentimentClassifier(
        pretrained_lm=lm_model, hidden_dim=config.LM_HIDDEN_DIM, output_dim=config.CLF_A_OUTPUT_DIM,
        n_layers=config.CLF_A_LAYERS, dropout=config.CLF_A_DROPOUT
    ).to(config.DEVICE)

    for param in model_A.backbone.parameters():
        param.requires_grad = False
    
    # 4. Train and evaluate
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model_A.parameters()), lr=config.CLF_A_LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss().to(config.DEVICE)
    
    history_A = train_and_evaluate_classifier(model_A, cls_train_loader, cls_val_loader, optimizer, criterion, config.DEVICE, epochs=config.CLF_A_NUM_EPOCHS)
    
    # 5. Final evaluation and analysis
    evaluate_on_test_set(model_A, cls_test_loader, criterion, config.DEVICE)
    run_error_analysis_to_file(model_A, cls_test_set, vocab, config.DEVICE, filename=config.CLF_A_ERROR_ANALYSIS_FILE)


def run_task_2b():
    """
    Handles Experiment B: From-scratch model with Word2Vec.
    """
    print("\n--- Running Task 2, Experiment B ---")
    set_seed(config.SEED)

    # 1. Prepare data and new vocabulary
    imdb_dataset = load_dataset('imdb')
    full_train_split = imdb_dataset['train'].train_test_split(test_size=0.2, stratify_by_column='label', seed=config.SEED)
    cls_train_set_B, cls_val_set_B = full_train_split['train'], full_train_split['test']
    cls_test_set = imdb_dataset['test']

    vocab_B = Vocabulary(min_freq=config.MIN_VOCAB_FREQ)
    vocab_B.build_vocabulary(cls_train_set_B['text'])

    # 2. Load Word2Vec and create embedding matrix
    print("Loading Word2Vec model...")
    word2vec_model = api.load(config.W2V_MODEL_NAME)
    embedding_dim = word2vec_model.vector_size
    embedding_matrix = np.zeros((len(vocab_B), embedding_dim))
    for word, idx in tqdm(vocab_B.stoi.items()):
        if word in word2vec_model:
            embedding_matrix[idx] = word2vec_model[word]
    
    embedding_matrix_tensor = torch.tensor(embedding_matrix, dtype=torch.float)

    # 3. Create DataLoaders
    pad_idx_B = vocab_B.stoi["<pad>"]
    train_loader_B = torch.utils.data.DataLoader(ClassificationDataset(cls_train_set_B, vocab_B), batch_size=config.CLF_B_BATCH_SIZE, shuffle=True, collate_fn=ClassificationCollate(pad_idx_B))
    val_loader_B = torch.utils.data.DataLoader(ClassificationDataset(cls_val_set_B, vocab_B), batch_size=config.CLF_B_BATCH_SIZE, shuffle=False, collate_fn=ClassificationCollate(pad_idx_B))
    test_loader_B = torch.utils.data.DataLoader(ClassificationDataset(cls_test_set, vocab_B), batch_size=config.CLF_B_BATCH_SIZE, shuffle=False, collate_fn=ClassificationCollate(pad_idx_B))

    # 4. Initialize model
    model_B = SentimentClassifierW2V(
        embedding_matrix_tensor, config.CLF_B_HIDDEN_DIM, config.CLF_B_OUTPUT_DIM,
        config.CLF_B_LAYERS, config.CLF_B_DROPOUT, pad_idx_B
    ).to(config.DEVICE)

    # 5. Train and evaluate
    optimizer = optim.Adam(model_B.parameters(), lr=config.CLF_B_LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss().to(config.DEVICE)
    history_B = train_and_evaluate_classifier(model_B, train_loader_B, val_loader_B, optimizer, criterion, config.DEVICE, epochs=config.CLF_B_NUM_EPOCHS)
    
    # 6. Final evaluation and analysis
    evaluate_on_test_set(model_B, test_loader_B, criterion, config.DEVICE)
    run_error_analysis_to_file(model_B, cls_test_set, vocab_B, config.DEVICE, filename=config.CLF_B_ERROR_ANALYSIS_FILE)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run NLP assignment tasks.')
    parser.add_argument('task', choices=['task1', 'task2a', 'task2b', 'all'], 
                        help='Which task to run.')
    
    args = parser.parse_args()
    
    if args.task == 'task1':
        run_task_1()
    elif args.task == 'task2a':
        run_task_2a()
    elif args.task == 'task2b':
        run_task_2b()
    elif args.task == 'all':
        run_task_1()
        run_task_2a()
        run_task_2b()
