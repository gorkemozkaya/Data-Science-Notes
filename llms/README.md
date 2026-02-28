Large Language Models

## Micro GPT Visualizers

### 1) Vanilla JS autograd demo

Open `micro_gpt_visualizer.html` in a browser (or serve this folder with `python -m http.server`) to run an interactive, in-browser Micro GPT-style training demo.

Features:
- Configure embedding size, attention heads, block size, learning rate, and total steps.
- Train step-by-step or continuously.
- See all trainable parameter matrices (`wte`, `wpe`, attention, MLP, and `lm_head`) as live-updating heatmaps during training.

### 2) TensorFlow.js names demo

Open `micro_gpt_tfjs_visualizer.html` to run a more efficient Micro GPT-style visualization powered by TensorFlow.js.

Features:
- Downloads Karpathy's `names.txt` dataset and trains a tiny character-level GPT-like model in the browser.
- Displays train/validation loss and generated name predictions every configurable number of steps.
- Visualizes parameter matrices as heatmaps with mouseover tooltips (row/column/value plus matrix statistics).
