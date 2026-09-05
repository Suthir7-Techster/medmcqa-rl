# MedMCQA RL Environment

A reinforcement learning environment for training AI models on medical multiple-choice questions using the MedMCQA dataset.

## Overview

This environment enables RL training on medical exam questions from AIIMS & NEET PG entrance exams, covering 21 medical subjects with 194k+ questions.

## Features

- **194k+ medical questions** from real medical entrance exams
- **21 medical subjects** (Anatomy, Physiology, Pharmacology, etc.)
- **Multi-criteria reward scoring**:
  - Exact match reward (1.0 weight)
  - Explanation quality reward (0.3 weight)
  - Reasoning steps reward (0.2 weight)
- **Native v1 implementation** with v0 compatibility wrapper
- **Training configs** for various GPT models

## Quick Start

### Installation

```bash
pip install verifiers datasets
```

### Validate Environment

```bash
python validate_medmcqa.py
```

### Run Training Simulation

```bash
python train_medmcqa.py
```

### Launch Real Training

```bash
# Quick test with GPT-4o-mini
uv run vf-gepa configs/rl/medmcqa-quick.toml

# Full training with GPT-OSS-20B
uv run vf-gepa configs/rl/medmcqa-gpt-oss.toml

# Alternative with GPT-4o
uv run vf-gepa configs/rl/medmcqa-gpt4o.toml
```

## Project Structure

```
medmcqa-rl/
├── environments/
│   └── medmcqa/
│       ├── medmcqa.py          # Main RL environment
│       └── pyproject.toml      # Package configuration
├── configs/
│   └── rl/
│       ├── medmcqa-gpt-oss.toml    # GPT-OSS-20B config
│       ├── medmcqa-gpt4o.toml      # GPT-4o config
│       └── medmcqa-quick.toml      # Quick test config
├── validate_medmcqa.py         # Validation script
├── train_medmcqa.py            # Training simulation
├── MEDMCQA_TRAINING.md         # Documentation
└── README.md                   # This file
```

## Environment Details

### Dataset

- **Source**: openlifescienceai/medmcqa (HuggingFace)
- **Questions**: 194k+ medical MCQs
- **Subjects**: 21 medical specialties
- **Format**: 4 options (A, B, C, D) with correct answer

### Reward System

| Reward Type | Weight | Description |
|-------------|--------|-------------|
| Exact Match | 1.0 | Binary reward for correct answer |
| Explanation Quality | 0.3 | Rewards detailed explanations |
| Reasoning Steps | 0.2 | Rewards step-by-step reasoning |

### Metrics

- `accuracy`: Running accuracy across episodes
- `questions_answered`: Number of questions answered
- `subject_name`: Medical subject (for grouping)

## Training Configuration

### GPT-OSS-20B (Recommended)

```toml
model = "openai/gpt-oss-20b"
max_steps = 100
batch_size = 64
rollouts_per_example = 8
```

### GPT-4o (Faster)

```toml
model = "openai/gpt-4o"
max_steps = 150
batch_size = 32
rollouts_per_example = 4
```

### Quick Test

```toml
model = "openai/gpt-4o-mini"
max_steps = 10
batch_size = 8
rollouts_per_example = 2
```

## Requirements

- Python ≥ 3.10
- verifiers ≥ 0.2.0
- datasets ≥ 2.0.0
- OpenRouter API key (for GPT models)

## Documentation

See [MEDMCQA_TRAINING.md](MEDMCQA_TRAINING.md) for complete training guide.

## License

MIT License
