#!/usr/bin/env python3
"""
Inference Script — Closed-Loop Life Support OpenEnv
===================================================
MANDATORY:
- Before submitting, ensure the following variables are defined:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
"""

import os
import json
import textwrap
import requests
import time
from typing import List, Optional, Dict, Any

from openai import OpenAI

HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
HOST = os.getenv("HOST", "http://127.0.0.1:7860")
BENCHMARK = "closed-loop-life-support"
MAX_STEPS = 20
TEMPERATURE = 0.2
MAX_TOKENS = 150

# Submission Configuration (OpenAI Client Only)
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.1-8b-instant")
client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

SYSTEM_PROMPT = textwrap.dedent("""
    You are an AI agent controlling a space habitat life support system.
    You receive sensor readings and must output control actions to keep the crew alive.

    CRITICAL THRESHOLDS:
    - O2 must stay between 19.5% and 23.5%
    - CO2 must stay below 1000 ppm
    - Water must stay above 5 liters
    - Food must stay above 0 kg
    - Crew health is your primary objective (keep above 0.8)

    ACTIONS (all floats in given ranges):
    - increase_plant_growth [0-1]
    - recycle_water [0-1]
    - adjust_oxygen [-1 to +1]
    - ration_food [0-1]
    - crew_activity [0-1]

    Respond ONLY with a valid JSON object. No explanation:
    {"increase_plant_growth": 0.7, "recycle_water": 0.6, "adjust_oxygen": 0.1, "ration_food": 1.0, "crew_activity": 0.8}
""").strip()


def call_env(endpoint: str, method: str = "POST", data: Dict = None) -> Dict:
    url = f"{HOST}{endpoint}"
    if method == "GET":
        resp = requests.get(url, timeout=30)
    else:
        resp = requests.post(url, json=data, timeout=30)
    resp.raise_for_status()
    return resp.json()


def run_episode(task_name: str, seed: int = 42) -> float:
    print(f"[START] task={task_name} env={BENCHMARK} model={MODEL_NAME}", flush=True)

    rewards = []
    success = False
    score = 0.0
    steps_taken = 0

    try:
        # Reset environment
        reset_resp = call_env("/reset", data={"task_id": task_name, "seed": seed})
        session_id = reset_resp["session_id"]
        obs = reset_resp["observation"]
        done = False

        for step in range(1, MAX_STEPS + 1):
            if done:
                break

            # Build observation text for model
            obs_text = (
                f"Step {step}/{MAX_STEPS}\n"
                f"O2: {obs['o2_percent']:.2f}% | CO2: {obs['co2_ppm']:.0f}ppm | "
                f"Water: {obs['water_liters']:.1f}L | Food: {obs['food_kg']:.2f}kg\n"
                f"Crew health: {obs['crew_health']:.3f}"
            )

            try:
                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": obs_text},
                    ],
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                )
                action_text = (completion.choices[0].message.content or "").strip()
                action_data = json.loads(action_text)
            except Exception as e:
                action_data = {"increase_plant_growth": 0.5, "recycle_water": 0.5, "adjust_oxygen": 0.0, "ration_food": 0.9, "crew_activity": 0.7}
                action_text = "default_fallback"

            # Step environment
            step_resp = call_env("/step", data={"session_id": session_id, "action": action_data})
            obs = step_resp["observation"]
            reward = step_resp["reward"]
            done = step_resp["done"]
            error = step_resp["info"].get("failure_reason", "null")
            if error is None: error = "null"

            rewards.append(reward)
            steps_taken = step

            print(f"[STEP] step={step} action={json.dumps(action_data)} reward={reward:.2f} done={str(done).lower()} error={error}", flush=True)

        # Grade completion
        grade_resp = call_env("/grade", data={"session_id": session_id, "task_id": task_name})
        score = grade_resp["score"]
        success = grade_resp["passed"]

    except Exception as e:
        print(f"[DEBUG] Runtime error: {e}", flush=True)
    finally:
        log_rewards = ",".join(f"{r:.2f}" for r in rewards)
        print(f"[END] success={str(success).lower()} steps={steps_taken} score={score:.3f} rewards={log_rewards}", flush=True)

    return score


def main():
    # Verify server is up
    try:
        requests.get(f"{HOST}/health", timeout=5).raise_for_status()
    except Exception as e:
        print(f"✗ Server not reachable at {HOST}: {e}")
        return

    for task_name in ["task_easy", "task_medium", "task_hard"]:
        run_episode(task_name)
        time.sleep(1)


if __name__ == "__main__":
    main()
