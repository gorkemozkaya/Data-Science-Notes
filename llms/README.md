Large Language Models

## Micro GPT Visualizer

Open `micro_gpt_visualizer.html` in a browser (or serve this folder with `python -m http.server`) to run an interactive, in-browser Micro GPT-style training demo.

Features:
- Configure embedding size, attention heads, block size, learning rate, and total steps.
- Train step-by-step or continuously.
- See all trainable parameter matrices (`wte`, `wpe`, attention, MLP, and `lm_head`) as live-updating heatmaps during training.
