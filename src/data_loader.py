import re
from collections import Counter
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

# --- Tokenizer ---
def simple_tokenizer(text):
    """An improved tokenizer that handles HTML line breaks and cleans the text."""
    text = re.sub(r'<br\\s*/?>', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\\\']', ' ', text)
    tokens = text.lower().split()
    return tokens


# --- Vocabulary Class ---
class Vocabulary:
    """Vocabulary class to map tokens to indices."""
    def __init__(self, min_freq=5):
        self.itos = {0: "<pad>", 1: "<unk>"}
        self.stoi = {"<pad>": 0, "<unk>": 1}
        self.min_freq = min_freq

    def __len__(self):
        return len(self.itos)

    def build_vocabulary(self, sentence_list):
        print("Building vocabulary...")
        token_counts = Counter()
        for sentence in tqdm(sentence_list):
            token_counts.update(simple_tokenizer(sentence))

        idx = 2
        for token, count in token_counts.items():
            if count >= self.min_freq:
                self.stoi[token] = idx
                self.itos[idx] = token
                idx += 1
        print(f"Built vocabulary with {len(self)} words.")


# --- Task 1: Language Model Dataset & Collate ---
class LanguageModelDataset(Dataset):
    def __init__(self, dataset, vocab, max_len=256):
        self.texts = [example['text'] for example in dataset]
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        tokens = simple_tokenizer(text)
        numericalized = [self.vocab.stoi.get(token, self.vocab.stoi["<unk>"]) for token in tokens]
        if len(numericalized) > self.max_len:
            numericalized = numericalized[:self.max_len]
        
        input_seq = torch.tensor(numericalized[:-1], dtype=torch.long)
        target_seq = torch.tensor(numericalized[1:], dtype=torch.long)
        return input_seq, target_seq

class PadCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        inputs = [item[0] for item in batch]
        targets = [item[1] for item in batch]
        padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=self.pad_idx)
        padded_targets = pad_sequence(targets, batch_first=True, padding_value=self.pad_idx)
        return padded_inputs, padded_targets


# --- Task 2: Classification Dataset & Collate ---
class ClassificationDataset(Dataset):
    def __init__(self, dataset, vocab, max_len=256):
        self.dataset = dataset
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        text, label = item['text'], item['label']
        tokens = simple_tokenizer(text)
        numericalized = [self.vocab.stoi.get(token, self.vocab.stoi["<unk>"]) for token in tokens]
        if len(numericalized) > self.max_len:
            numericalized = numericalized[:self.max_len]
        return torch.tensor(numericalized, dtype=torch.long), torch.tensor(label, dtype=torch.long)

class ClassificationCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        inputs = [item[0] for item in batch]
        labels = [item[1] for item in batch]
        padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=self.pad_idx)
        return padded_inputs, torch.stack(labels)