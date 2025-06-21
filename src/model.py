import torch
import torch.nn as nn

# --- Task 1: Language Model ---
class LanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        out = self.fc(lstm_out)
        return out


# --- Task 2, Experiment A: LM Backbone Classifier ---
class SentimentClassifier(nn.Module):
    def __init__(self, pretrained_lm, hidden_dim, output_dim, dropout):
        super().__init__()
        self.backbone = pretrained_lm

        # The embedding layer will remain trainable (fine-tuned).
        for param in self.backbone.lstm.parameters():
            param.requires_grad = False
        for param in self.backbone.fc.parameters():
            param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        embedded = self.backbone.embedding(x)
        lstm_out, (hidden, cell) = self.backbone.lstm(embedded)
        sentence_embedding = hidden[-1, :, :]
        prediction = self.classifier(sentence_embedding)
        return prediction.squeeze(1)


# --- Task 2, Experiment B: Word2Vec Classifier ---
class SentimentClassifierW2V(nn.Module):
    def __init__(self, embedding_matrix, hidden_dim, output_dim, n_layers, dropout, pad_idx):
        super().__init__()
        vocab_size, embedding_dim = embedding_matrix.shape
        self.embedding = nn.Embedding.from_pretrained(
            embedding_matrix,
            freeze=True,
            padding_idx=pad_idx
        )
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        embedded = self.embedding(text)
        embedded = self.dropout(embedded)
        packed_output, (hidden, cell) = self.lstm(embedded)
        hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        return self.fc(hidden).squeeze(1)