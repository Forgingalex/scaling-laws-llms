import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train.train_model import train_model


configs = [
    {"layers": 2, "hidden_dim": 64},
    {"layers": 4, "hidden_dim": 128},
    {"layers": 6, "hidden_dim": 256}
]


def load_results():
    results_path = 'plots/results.json'
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            else:
                return [data]
    return []


def save_results(all_results):
    os.makedirs('plots', exist_ok=True)
    with open('plots/results.json', 'w') as f:
        json.dump(all_results, f, indent=2)


def main():
    existing_results = load_results()
    new_results = []
    
    for i, config in enumerate(configs):
        print(f"\nTraining model {i+1}/{len(configs)}: {config['layers']} layers, {config['hidden_dim']} hidden_dim")
        results = train_model(
            num_layers=config['layers'],
            hidden_dim=config['hidden_dim'],
            num_epochs=3,
            save_results=False
        )
        new_results.append(results)
    
    all_results = existing_results + new_results
    save_results(all_results)
    
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print(f"{'Model Size':<20} {'Parameters':<20} {'Final Loss':<15}")
    print("-"*60)
    
    for result in new_results:
        model_size = f"{result['config']['num_layers']}L/{result['config']['hidden_dim']}D"
        params = f"{result['parameter_count']:,}"
        final_loss = f"{result['val_loss'][-1]:.4f}"
        print(f"{model_size:<20} {params:<20} {final_loss:<15}")


if __name__ == '__main__':
    main()

