# scaling-laws-llms

This repository explores empirical scaling laws in small transformer language models to observe how model size affects learning dynamics.

## Background

Scaling laws describe predictable relationships between model capacity and performance. Understanding these relationships helps estimate the compute and model size required to achieve target performance levels. This project examines scaling behavior in the small model regime to observe power-law relationships between parameters and validation loss.

## Experiment Setup

The experiment trains transformer language models of increasing size on the tiny_shakespeare dataset:

- **Models**: Three transformer configurations with 2/4/6 layers and 64/128/256 hidden dimensions
- **Dataset**: tiny_shakespeare, tokenized with GPT-2 tokenizer
- **Training**: AdamW optimizer, 3 epochs per model
- **Metrics**: Parameter count and validation loss are recorded for each configuration

Results are saved to `plots/results.json` and can be visualized with the plotting script.

## Usage

Run the scaling experiment:

```bash
python experiments/run_scaling.py
```

Generate a visualization of the scaling relationship:

```bash
python plots/plot_results.py
```

## Expected Observations

As model size increases, validation loss typically decreases following a power-law relationship. Larger models generally achieve lower loss due to increased capacity, though the rate of improvement may diminish at larger scales. The relationship between log parameters and log loss is often approximately linear in the power-law regime.
