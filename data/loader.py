from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer
import torch
import requests


class ChunkedTextDataset(Dataset):
    def __init__(self, token_ids, block_size=128, stride=128):
        self.block_size = block_size
        self.chunks = []
        
        for i in range(0, len(token_ids) - block_size, stride):
            chunk = token_ids[i:i + block_size + 1]
            if len(chunk) == block_size + 1:
                self.chunks.append(chunk)
    
    def __len__(self):
        return len(self.chunks)
    
    def __getitem__(self, idx):
        chunk = self.chunks[idx]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


def get_tiny_shakespeare_loaders(batch_size=8, block_size=128, max_tokens=100000, train_split=0.9):
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    response = requests.get(url)
    text = response.text
    
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    token_ids = tokenizer.encode(text)
    
    if len(token_ids) > max_tokens:
        token_ids = token_ids[:max_tokens]
    
    split_idx = int(len(token_ids) * train_split)
    train_ids = token_ids[:split_idx]
    val_ids = token_ids[split_idx:]
    
    train_dataset = ChunkedTextDataset(train_ids, block_size=block_size, stride=block_size)
    val_dataset = ChunkedTextDataset(val_ids, block_size=block_size, stride=block_size)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, tokenizer
