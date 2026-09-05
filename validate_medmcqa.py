#!/usr/bin/env python3
"""Validate MedMCQA RL environment setup and functionality."""

import sys
import os
from pathlib import Path

def validate_environment():
    """Validate the MedMCQA environment."""
    print("=" * 60)
    print("MedMCQA RL Environment Validation")
    print("=" * 60)

    # Add venv site-packages to path
    venv_path = Path(__file__).parent / ".venv" / "Lib" / "site-packages"
    if venv_path.exists():
        sys.path.insert(0, str(venv_path))

    # Add environments directory to path
    env_dir = Path(__file__).parent / "environments"
    sys.path.insert(0, str(env_dir))

    # 1. Check imports
    print("\n1. Testing imports...")
    try:
        from medmcqa.medmcqa import (
            MedMCQATaskset, MedMCQAData, MedQAState,
            MedMCQAConfig, MedMCQATask, MedMCQATasksetConfig,
            load_environment, ChoiceType, AnswerLetter
        )
        print("   [OK] All imports successful")
    except Exception as e:
        print(f"   [FAIL] Import failed: {e}")
        return False

    # 2. Test taskset loading
    print("\n2. Testing taskset loading...")
    try:
        config = MedMCQATasksetConfig(num_scenarios=5)
        taskset = MedMCQATaskset(config)
        tasks = taskset.load()
        print(f"   [OK] Loaded {len(tasks)} tasks")
        if tasks:
            task = tasks[0]
            print(f"   [OK] Sample task data:")
            print(f"     - Question ID: {task.data.question_id}")
            print(f"     - Subject: {task.data.subject_name}")
            print(f"     - Topic: {task.data.topic_name}")
            print(f"     - Correct option: {task.data.correct_option}")
    except Exception as e:
        print(f"   [FAIL] Taskset loading failed: {e}")
        return False

    # 3. Test reward calculation
    print("\n3. Testing reward calculation...")
    try:
        import asyncio

        async def test_rewards():
            task = tasks[0]
            state = MedQAState()
            # Select a wrong answer (not the correct one)
            correct_letter = chr(ord('A') + task.data.correct_option - 1)
            wrong_answers = [l for l in ['A', 'B', 'C', 'D'] if l != correct_letter]
            state.selected_answer = wrong_answers[0]  # Wrong answer
            state.reasoning_chain = ["Step 1", "Step 2"]

            trace = type('Trace', (), {'state': state, 'messages': []})()

            exact_reward = await task.exact_match_reward(trace)
            explanation_reward = await task.explanation_quality_reward(trace)
            reasoning_reward = await task.reasoning_reward(trace)

            print(f"   [OK] Exact match reward (wrong): {exact_reward}")
            print(f"   [OK] Explanation reward: {explanation_reward}")
            print(f"   [OK] Reasoning reward: {reasoning_reward}")

            # Test correct answer
            correct_letter = chr(ord('A') + task.data.correct_option - 1)
            state.selected_answer = correct_letter
            exact_reward = await task.exact_match_reward(trace)
            print(f"   [OK] Exact match reward (correct): {exact_reward}")

            return True

        asyncio.run(test_rewards())
    except Exception as e:
        print(f"   [FAIL] Reward calculation failed: {e}")
        return False

    # 4. Test v0 compatibility wrapper
    print("\n4. Testing v0 compatibility wrapper...")
    try:
        env = load_environment(num_scenarios=5)
        print(f"   [OK] v0 Environment created: {type(env).__name__}")
        print(f"   [OK] Dataset size: {len(env.dataset)}")
    except Exception as e:
        print(f"   [FAIL] v0 wrapper failed: {e}")
        return False

    print("\n" + "=" * 60)
    print("[OK] All validation tests passed!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = validate_environment()
    sys.exit(0 if success else 1)
