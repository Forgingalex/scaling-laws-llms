import torch
import torch.nn.functional as F
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.tiny_transformer import TinyTransformer
from data.loader import get_tiny_shakespeare_loaders


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def train_epoch(model, train_loader, optimizer, device, val_loader=None, eval_interval=100, log_interval=50):
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        if (batch_idx + 1) % log_interval == 0:
            print(f"  Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        if val_loader is not None and (batch_idx + 1) % eval_interval == 0:
            val_loss = validate(model, val_loader, device)
            print(f"  Val Loss at batch {batch_idx + 1}: {val_loss:.4f}")
        
        del x, y, logits, loss
    
    return total_loss / num_batches


def validate(model, val_loader, device):
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            total_loss += loss.item()
            num_batches += 1
            
            del x, y, logits, loss
    
    return total_loss / num_batches


def train_model(num_layers=4, hidden_dim=256, num_epochs=1, learning_rate=3e-4, batch_size=8, 
                block_size=128, save_results=True, eval_interval=100, log_interval=50):
    device = torch.device('cpu')
    
    train_loader, val_loader, tokenizer = get_tiny_shakespeare_loaders(
        batch_size=batch_size, 
        block_size=block_size
    )
    vocab_size = tokenizer.vocab_size
    
    num_heads = min(8, hidden_dim)
    while hidden_dim % num_heads != 0:
        num_heads -= 1
    
    model = TinyTransformer(
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        max_seq_len=block_size
    ).to(device)
    
    param_count = count_parameters(model)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        train_loss = train_epoch(model, train_loader, optimizer, device, val_loader, eval_interval, log_interval)
        val_loss = validate(model, val_loader, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"Epoch {epoch+1}/{num_epochs} - Train: {train_loss:.4f}, Val: {val_loss:.4f}")
    
    results = {
        'config': {
            'num_layers': num_layers,
            'hidden_dim': hidden_dim,
            'num_epochs': num_epochs,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'block_size': block_size
        },
        'parameter_count': param_count,
        'train_loss': train_losses,
        'val_loss': val_losses
    }
    
    if save_results:
        os.makedirs('plots', exist_ok=True)
        with open('plots/results.json', 'w') as f:
            json.dump(results, f, indent=2)
    
    return results


if __name__ == '__main__':
    train_model(num_layers=4, hidden_dim=256)
