#!/usr/bin/env python3
"""Simulate training on MedMCQA RL environment."""

import asyncio
import random
from pathlib import Path
import sys

def simulate_training():
    """Simulate training on MedMCQA environment."""
    print("=" * 60)
    print("MedMCQA RL Training Simulation")
    print("=" * 60)

    # Add environments directory to path
    env_dir = Path(__file__).parent / "environments"
    sys.path.insert(0, str(env_dir))

    from medmcqa.medmcqa import MedMCQATaskset, MedMCQATasksetConfig, MedQAState

    config = MedMCQATasksetConfig(num_scenarios=10)
    taskset = MedMCQATaskset(config)
    tasks = taskset.load()

    print(f"\nLoaded {len(tasks)} questions")

    # Simulate 5 episodes
    total_reward = 0
    correct = 0
    for episode in range(5):
        task = random.choice(tasks)
        state = MedQAState()
        state.reasoning_chain = ["Analyzing question", "Considering options"]

        # Simulate random answer
        state.selected_answer = random.choice(["A", "B", "C", "D"])
        correct_letter = chr(ord("A") + task.data.correct_option - 1)
        is_correct = state.selected_answer == correct_letter

        reward = 1.0 if is_correct else 0.0
        total_reward += reward
        if is_correct:
            correct += 1

        print(f"Episode {episode + 1}: {task.data.subject_name} - {'Correct' if is_correct else 'Wrong'}")

    avg_reward = total_reward / 5
    accuracy = correct / 5
    print(f"\nResults: Avg Reward: {avg_reward:.3f}, Accuracy: {accuracy:.1%}")

    print("\n[OK] Training simulation complete")
    return True

if __name__ == "__main__":
    success = simulate_training()
    sys.exit(0 if success else 1)
