#!/usr/bin/env python3
"""
Baseline inference script for the Closed-Loop Life Support OpenEnv environment.
Compliant with Hackathon Pre-Submission Checklist.
"""
import json
import os
import sys
import time
import requests
from typing import Dict, Any, List

from openai import OpenAI

# Required Environment Variables
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.1-8b-instant")
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

HOST = os.getenv("HOST", "http://localhost:7860") # Your space URL

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

SYSTEM_PROMPT = """You are an AI agent controlling a space habitat life support system.
You receive sensor readings and must output control actions to keep the crew alive.

CRITICAL THRESHOLDS:
- O2 must stay between 19.5% and 23.5% (below 19.5% = crew suffocates)
- CO2 must stay below 1000 ppm (above 3000 = incapacitation)
- Water must stay above 5 liters
- Food must stay above 0 kg
- Crew health is your primary objective (keep above 0.8)

ACTIONS (all floats in given ranges):
- increase_plant_growth [0-1]: Boost photosynthesis (produces O2, consumes CO2 and water, grows food)
- recycle_water [0-1]: Water reclamation intensity (recovers crew/plant waste water)
- adjust_oxygen [-1 to +1]: Positive = release stored O2; Negative = activate CO2 scrubber
- ration_food [0-1]: 1.0 = full rations; 0.3 = emergency rations (extends food supply)
- crew_activity [0-1]: Lower activity reduces O2/food consumption but hurts morale

STRATEGY:
- If O2 is low: increase_plant_growth AND adjust_oxygen positive
- If CO2 is high: increase_plant_growth AND adjust_oxygen negative
- If water is low: recycle_water high, reduce plant growth temporarily
- If food is low: ration_food down, grow plants for future harvest

Respond ONLY with a valid JSON object. No explanation, no markdown:
{"increase_plant_growth": 0.7, "recycle_water": 0.6, "adjust_oxygen": 0.1, "ration_food": 1.0, "crew_activity": 0.8}"""


def call_env(host: str, endpoint: str, method: str = "POST", data: Dict = None) -> Dict:
    url = f"{host}{endpoint}"
    if method == "GET":
        resp = requests.get(url, timeout=30)
    else:
        resp = requests.post(url, json=data, timeout=30)
    resp.raise_for_status()
    return resp.json()


def run_episode(host: str, client: OpenAI, model: str, task_id: str, seed: int) -> Dict[str, Any]:
    print("[START]")
    print(f"Task: {task_id} | Model: {model} | Seed: {seed}")

    reset_resp = call_env(host, "/reset", data={"task_id": task_id, "seed": seed})
    session_id = reset_resp["session_id"]
    max_steps = reset_resp["info"]["max_steps"]
    obs = reset_resp["observation"]

    step_count = 0
    total_reward = 0.0
    conversation: List[Dict] = []

    while step_count < max_steps:
        print("[STEP]")
        
        # Build user message with current observation
        obs_text = (
            f"Step {step_count + 1}/{max_steps}\n"
            f"O2: {obs['o2_percent']:.2f}% | CO2: {obs['co2_ppm']:.0f}ppm | "
            f"Water: {obs['water_liters']:.1f}L | Food: {obs['food_kg']:.2f}kg\n"
            f"Crew health: {obs['crew_health']:.3f} | Crew size: {obs['crew_size']}\n"
            f"Plant growth rate: {obs['plant_growth_rate']:.2f} | "
            f"Water recycling: {obs['water_recycling_rate']:.2f}\n"
            f"Power budget: {obs['power_budget']:.2f} | Day: {obs['day']}"
        )

        conversation.append({"role": "user", "content": obs_text})

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": SYSTEM_PROMPT}] + conversation[-6:],
                temperature=0.3,
                max_tokens=100,
            )
            action_text = response.choices[0].message.content.strip()
            conversation.append({"role": "assistant", "content": action_text})

            # Parse action
            action_data = json.loads(action_text)
            action = {
                "increase_plant_growth": float(action_data.get("increase_plant_growth", 0.5)),
                "recycle_water": float(action_data.get("recycle_water", 0.5)),
                "adjust_oxygen": float(action_data.get("adjust_oxygen", 0.0)),
                "ration_food": float(action_data.get("ration_food", 1.0)),
                "crew_activity": float(action_data.get("crew_activity", 0.7)),
            }
        except Exception as e:
            print(f"  ⚠ LLM/Action parse error at step {step_count}: {e}. Using defaults.")
            action = {
                "increase_plant_growth": 0.6,
                "recycle_water": 0.6,
                "adjust_oxygen": 0.0,
                "ration_food": 0.9,
                "crew_activity": 0.7,
            }

        # Step environment
        step_resp = call_env(host, "/step", data={"session_id": session_id, "action": action})
        obs = step_resp["observation"]
        reward = step_resp["reward"]
        done = step_resp["done"]
        total_reward += reward
        step_count += 1

        if done:
            if step_resp["info"].get("failure_reason"):
                print(f"  ✗ FAILED: {step_resp['info']['failure_reason']}")
            else:
                print(f"  ✓ Episode complete")
            break

    # Grade the episode
    grade_resp = call_env(host, "/grade", data={"session_id": session_id, "task_id": task_id})
    print(f"  Score: {grade_resp['score']:.4f} | Passed: {grade_resp['passed']}")
    print("[END]")

    return {
        "task_id": task_id,
        "score": grade_resp["score"],
        "passed": grade_resp["passed"],
        "steps": step_count,
        "total_reward": round(total_reward, 4),
        "breakdown": grade_resp["breakdown"],
        "feedback": grade_resp["feedback"],
    }


def main():
    seed = 42
    tasks = ["task_easy", "task_medium", "task_hard"]

    try:
        health = requests.get(f"{HOST}/health", timeout=5)
        health.raise_for_status()
    except Exception as e:
        print(f"✗ Server not reachable at {HOST}: {e}")
        sys.exit(1)

    results = []
    for task_id in tasks:
        result = run_episode(HOST, client, MODEL_NAME, task_id, seed)
        results.append(result)
        time.sleep(1)

    avg_score = sum(r["score"] for r in results) / len(results)

    output = {
        "model": MODEL_NAME,
        "seed": seed,
        "host": HOST,
        "results": results,
        "average_score": round(avg_score, 4),
    }
    with open("baseline_results.json", "w") as f:
        json.dump(output, f, indent=2)

if __name__ == "__main__":
    main()
