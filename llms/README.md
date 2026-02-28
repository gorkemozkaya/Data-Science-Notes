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

### 3) JS PyTorch-style micro char LM (2D embeddings)

Open `micro_char_lm_torchjs_visualizer.html` for a very small, educational next-character language model implemented in plain JavaScript with PyTorch-style tensor math.

Features:
- Downloads natural English text from a public repo (GitHub Tiny Shakespeare, with CDN fallback) and normalizes it to the constrained vocabulary.
- Learns **2D character embeddings** and displays them live during training.
- Predicts next-character probabilities for a selected input character.
- Shows training loss and sampled representative text as the model improves.
