import json
import matplotlib.pyplot as plt


def plot_scaling_law():
    with open('plots/results.json', 'r') as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        data = [data]
    
    param_counts = []
    val_losses = []
    
    for result in data:
        param_counts.append(result['parameter_count'])
        val_losses.append(result['val_loss'][-1])
    
    sorted_pairs = sorted(zip(param_counts, val_losses))
    param_counts, val_losses = zip(*sorted_pairs)
    
    plt.figure(figsize=(8, 6))
    plt.plot(param_counts, val_losses, marker='o')
    plt.xlabel('Parameter Count')
    plt.ylabel('Validation Loss')
    plt.title('Scaling Law: Model Size vs Validation Loss')
    plt.savefig('plots/scaling_law.png')
    plt.close()


if __name__ == '__main__':
    plot_scaling_law()

