from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer
import torch
import requests


class TextDataset(Dataset):
    def __init__(self, token_ids, block_size=512):
        self.token_ids = token_ids
        self.block_size = block_size
    
    def __len__(self):
        return len(self.token_ids) - self.block_size
    
    def __getitem__(self, idx):
        chunk = self.token_ids[idx:idx + self.block_size + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


def get_tiny_shakespeare_loaders(batch_size=32, block_size=512, train_split=0.9):
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    response = requests.get(url)
    text = response.text
    
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    token_ids = tokenizer.encode(text)
    
    split_idx = int(len(token_ids) * train_split)
    train_ids = token_ids[:split_idx]
    val_ids = token_ids[split_idx:]
    
    train_dataset = TextDataset(train_ids, block_size)
    val_dataset = TextDataset(val_ids, block_size)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, tokenizer
