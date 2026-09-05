# MedMCQA RL Environment Documentation

## Overview

This is an RL environment for training models on medical multiple-choice questions using the MedMCQA dataset. The environment includes:

- **194k+ medical questions** from AIIMS & NEET PG entrance exams
- **21 medical subjects** (Anatomy, Physiology, Pharmacology, etc.)
- **Multi-criteria reward scoring** (accuracy, explanation quality, reasoning steps)

## Environment Structure

```
environments/medmcqa/
├── medmcqa.py          # Main environment (v0 + v1 compatible)
├── pyproject.toml      # Package configuration
└── servers/            # Tool servers (if needed)
```

## Quick Start

### 1. Validate Environment

```bash
python validate_medmcqa.py
```

### 2. Run Training Simulation

```bash
python train_medmcqa.py
```

### 3. Launch Real Training

```bash
# Quick test with GPT-4o-mini (small, fast)
uv run vf-gepa configs/rl/medmcqa-quick.toml

# Full training with GPT-OSS-20B
uv run vf-gepa configs/rl/medmcqa-gpt-oss.toml

# Alternative with GPT-4o
uv run vf-gepa configs/rl/medmcqa-gpt4o.toml
```

## Training Configuration

### GPT-OSS-20B (Recommended)

```toml
# configs/rl/medmcqa-gpt-oss.toml
model = "openai/gpt-oss-20b"
max_steps = 100
batch_size = 64
rollouts_per_example = 8
```

### GPT-4o (Faster, Smaller)

```toml
# configs/rl/medmcqa-gpt4o.toml
model = "openai/gpt-4o"
max_steps = 150
batch_size = 32
rollouts_per_example = 4
```

### Quick Test

```toml
# configs/rl/medmcqa-quick.toml
model = "openai/gpt-4o-mini"
max_steps = 10
batch_size = 8
rollouts_per_example = 2
```

## Reward System

The environment uses multi-criteria reward scoring:

| Reward Type | Weight | Description |
|-------------|--------|-------------|
| Exact Match | 1.0 | Binary reward for correct answer |
| Explanation Quality | 0.3 | Rewards detailed explanations |
| Reasoning Steps | 0.2 | Rewards step-by-step reasoning |

### Total Reward Calculation

```
total_reward = (1.0 × exact_match) + (0.3 × explanation_quality) + (0.2 × reasoning_steps)
```

## Metrics

| Metric | Description |
|--------|-------------|
| accuracy | Running accuracy across episodes |
| questions_answered | Number of questions answered |
| subject_name | Medical subject (for grouping) |

## Data Format

Each question contains:
- `question`: Medical question text
- `opa`, `opb`, `opc`, `opd`: Four answer options
- `cop`: Correct option (1=A, 2=B, 3=C, 4=D)
- `exp`: Expert explanation
- `subject_name`: Medical subject
- `topic_name`: Medical topic

## Example Output

```
MedMCQA RL Training Simulation
============================================================
Loaded 10 questions

Episode 1: Pathology - Correct
Episode 2: Pharmacology - Wrong
Episode 3: Physiology - Correct
Episode 4: Biochemistry - Correct
Episode 5: Microbiology - Wrong

Results: Avg Reward: 0.600, Accuracy: 60.0%

✓ Training simulation complete
```

## Environment Features

1. **Native v1 Implementation**: Full v1 Taskset/Task/TaskData pattern
2. **v0 Compatibility Wrapper**: Works with existing `vf-eval` CLI
3. **Sample Data Fallback**: Generates test data if HuggingFace unavailable
4. **Subject Filtering**: Filter questions by medical subject
5. **Configurable Scoring**: Adjustable reward weights

## Requirements

- Python ≥ 3.10
- verifiers ≥ 0.2.0
- datasets ≥ 2.0.0
- OpenRouter API key (for GPT models)

## Files Created

- `environments/medmcqa/medmcqa.py` - Main environment
- `environments/medmcqa/pyproject.toml` - Package config
- `configs/rl/medmcqa-gpt-oss.toml` - GPT-OSS-20B training config
- `configs/rl/medmcqa-gpt4o.toml` - GPT-4o training config
- `configs/rl/medmcqa-quick.toml` - Quick test config
- `validate_medmcqa.py` - Validation script
- `train_medmcqa.py` - Training simulation script
