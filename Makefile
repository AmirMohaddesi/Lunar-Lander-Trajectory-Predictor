.PHONY: install install-gym test collect-frames train-demo

install:
	pip install -e ".[dev]"

install-gym:
	pip install -e ".[gym,dev]"

test:
	pytest -q

# Example: make collect-frames OUT=Train EPISODES=5
collect-frames:
	python scripts/collect_episode_frames.py --out $(OUT) --episodes $(EPISODES)

# After collecting frames into Train/: short CPU smoke train
train-demo:
	python scripts/train_autoencoder.py --data Train --epochs 2 --device cpu
