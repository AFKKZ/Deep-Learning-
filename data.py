import pandas as pd
import numpy as np
import re
import nltk
from sklearn.model_selection import train_test_split
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import Dataset, DataLoader
import torch

#nltk.download('punkt')
from nltk.tokenize import word_tokenize

MAX_TEXT_LEN = 50
MAX_CONTEXT_LEN = 20
MAX_VOCAB_SIZE = 5000

SPECIAL_TOKENS = {'<PAD>': 0, '<UNK>': 1}

# Preprocessing function
def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return word_tokenize(text)

# Custom dataset class
class SentimentDataset(Dataset):
    def __init__(self, df, vocab, label_map):
        self.df = df
        self.vocab = vocab
        self.label_map = label_map

    def __len__(self):
        return len(self.df)
    
    def encode(self, tokens, max_len):
        ids = [self.vocab[token] for token in tokens]
        if len(ids) < max_len:
            ids += [SPECIAL_TOKENS['<PAD>']] * (max_len - len(ids))
        return ids[:max_len]

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text_tokens = preprocess(row['text'])
        context_tokens = preprocess(row['context'])
        text_ids = self.encode(text_tokens, MAX_TEXT_LEN)
        context_ids = self.encode(context_tokens, MAX_CONTEXT_LEN)
        label = self.label_map[row['label']]
        return torch.tensor(text_ids), torch.tensor(context_ids), torch.tensor(label)

# Function to load data
def load_data(path='sentiment_data.csv', batch_size=32):
    df = pd.read_csv(path)
    df.dropna(inplace=True)
    df = df[df['label'].isin(['Positive', 'Negative', 'Neutral'])]

    # Preprocess all tokens to build vocab
    def yield_tokens():
        for _, row in df.iterrows():
            yield preprocess(row['text'])
            yield preprocess(row['context'])

    vocab = build_vocab_from_iterator(yield_tokens(), specials=['<PAD>', '<UNK>'], max_tokens=MAX_VOCAB_SIZE)
    vocab.set_default_index(SPECIAL_TOKENS['<UNK>'])

    glove_path = 'glove.6B.100d.txt'  
    pretrained_embeddings = load_glove_embeddings(glove_path, vocab, embed_dim=100)

    label_map = {'Negative': 0, 'Neutral': 1, 'Positive': 2}

    # First split: train 80%, temp 20%
    train_df, temp_df = train_test_split(df, test_size=0.2, shuffle=True, random_state=42)
    # Second split: val 70%, test 30%
    val_df, test_df = train_test_split(temp_df, test_size=0.3, shuffle=True, random_state=42)

    train_ds = SentimentDataset(train_df, vocab, label_map)
    val_ds = SentimentDataset(val_df, vocab, label_map)
    test_ds = SentimentDataset(test_df, vocab, label_map)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    return train_loader, val_loader, test_loader, vocab, label_map, pretrained_embeddings

def load_glove_embeddings(glove_path, vocab, embed_dim=100):
    embeddings = np.random.uniform(-0.25, 0.25, (len(vocab), embed_dim)).astype(np.float32)
    embeddings[vocab['<PAD>']] = np.zeros(embed_dim)

    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            vector = np.array(parts[1:], dtype=np.float32)
            if word in vocab:
                embeddings[vocab[word]] = vector
    return torch.tensor(embeddings)