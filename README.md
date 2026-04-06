---
title: Closed-Loop Life Support OpenEnv
emoji: 🌱
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
tags:
  - openenv
  - reinforcement-learning
  - space
  - simulation
  - resource-management
---

# 🌱 Closed-Loop Life Support — OpenEnv

> *Can your AI keep a crew alive in space?*

A real-world OpenEnv environment where an AI agent manages the life support systems of a space habitat. The agent must balance **oxygen production**, **CO₂ removal**, **water recycling**, and **food cultivation** to keep a crew of astronauts alive across missions of increasing difficulty.

---

## 🎯 Why This Environment?

Life support optimization is one of the most consequential real-world control problems:
- NASA's **ECLSS** (Environmental Control and Life Support System) on the ISS is manually operated today
- Future long-duration missions (Moon base, Mars) will need autonomous life support agents
- The closed-loop nature (waste → water → plants → food → oxygen) creates rich multi-objective tradeoffs that genuinely challenge frontier models

This environment fills a real gap: **no existing OpenEnv covers closed-loop biological + chemical resource management for life-critical systems.**

---

## 🏗️ Project Structure

```
life-support-env/
├── env/
│   ├── environment.py    # Core simulation (LifeSupportEnv)
│   └── models.py         # Pydantic models: Observation, Action, Reward, State
├── tasks/
│   └── graders.py        # Deterministic graders for easy/medium/hard
├── tests/
│   └── test_environment.py  # Full test suite (pytest)
├── server.py             # FastAPI HTTP server (OpenEnv API)
├── baseline_inference.py # Baseline agent using OpenAI API
├── openenv.yaml          # OpenEnv metadata spec
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## 🌍 Environment Description

### Simulation Model

The habitat is modeled as a set of coupled differential systems updated each **simulated hour**:

| Subsystem | Inputs | Outputs |
|-----------|--------|---------|
| **Plant Bay** | Water, power (plant growth action) | O₂, food, CO₂ absorption |
| **Water Reclamation** | Crew waste water, plant transpiration | Potable water |
| **Atmosphere Control** | Plant production, crew respiration, stored O₂ | O₂%, CO₂ ppm |
| **Crew** | O₂, water, food, activity level | Health, CO₂, waste water |
| **Power Grid** | Fixed solar budget | Constrains all subsystems |

### Action Space

All actions are continuous floats:

| Action | Range | Effect |
|--------|-------|--------|
| `increase_plant_growth` | [0, 1] | Boosts photosynthesis → more O₂, food; costs power + water |
| `recycle_water` | [0, 1] | Water reclamation intensity; recovers ~90% of waste water |
| `adjust_oxygen` | [-1, +1] | +: release stored O₂; -: activate CO₂ chemical scrubber |
| `ration_food` | [0, 1] | 1.0 = full rations; 0.3 = emergency rationing |
| `crew_activity` | [0, 1] | Higher activity increases O₂/food consumption |

**Power constraint**: `plant_growth × 0.3 + recycle_water × 0.25 + |adjust_oxygen| × 0.15 ≤ 1.0`. Excess power usage scales down all systems proportionally.

### Observation Space

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `co2_ppm` | float | [0, 5000] | CO₂ concentration in parts per million |
| `o2_percent` | float | [0, 30] | O₂ percentage in atmosphere |
| `water_liters` | float | [0, 500] | Available potable water |
| `food_kg` | float | [0, 100] | Food supply in kg |
| `crew_size` | int | [1, 10] | Active crew members |
| `plant_growth_rate` | float | [0, 1] | Current photosynthesis rate |
| `water_recycling_rate` | float | [0, 1] | Current recycling efficiency |
| `day` | int | [1, 365] | Mission day |
| `crew_health` | float | [0, 1] | Average crew health (0=critical, 1=optimal) |
| `power_budget` | float | [0, 1] | Remaining power budget fraction |

### Reward Function

The reward is shaped over the full trajectory (not sparse end-of-episode):

```
reward = 0.50 × health_component
       + 0.30 × resource_component
       + 0.20 × efficiency_component
       − penalty

health_component  = 0.50 × crew_health + 0.25 × o2_score + 0.25 × co2_score
resource_component = 0.50 × (water / 50) + 0.50 × (food / 10)
efficiency_component = power_budget (if crew_health > 0.7)
penalty: +0.2 if O2 < 19.5%, +0.3 if CO2 > 3000ppm, +0.4 if water = 0
```

All rewards clamped to [-1, 1].

---

## 📋 Tasks

### Task 1: Single-Day Stabilization (Easy)
- **Crew**: 3 | **Steps**: 24 (1 simulated day)
- **Objective**: Keep all parameters in safe ranges for 24 hours
- **Grading**: 60% safety + 20% health (>0.8) + 20% completion
- **Expected baseline score**: ~0.75 (GPT-4o-mini)

### Task 2: 7-Day Resource Balance (Medium)
- **Crew**: 5 | **Steps**: 168 (7 simulated days)
- **Objective**: Sustain crew AND maintain positive resource trends (water/food at end ≥ start)
- **Grading**: 40% safety + 25% resource trends + 20% health + 15% completion
- **Expected baseline score**: ~0.52 (GPT-4o-mini)

### Task 3: 30-Day Closed-Loop Optimization (Hard)
- **Crew**: 8 | **Steps**: 720 (30 simulated days)
- **Objective**: Fully closed-loop mission — maximize health AND minimize resource imports
- **Grading**: 30% loop efficiency + 25% health + 20% safety + 15% food self-sufficiency + 10% completion
- **Expected baseline score**: ~0.35 (GPT-4o-mini)

---

## 🚀 Setup & Usage

### Local (Docker)

```bash
docker build -t life-support-env .
docker run -p 7860:7860 life-support-env
```

### Local (Python)

```bash
pip install -r requirements.txt
uvicorn server:app --host 0.0.0.0 --port 7860
```

### API Quickstart

```python
import requests

BASE = "http://localhost:7860"

# Start episode
resp = requests.post(f"{BASE}/reset", json={"task_id": "task_easy", "seed": 42})
session_id = resp.json()["session_id"]
obs = resp.json()["observation"]

# Run agent loop
while True:
    action = {
        "increase_plant_growth": 0.7,
        "recycle_water": 0.6,
        "adjust_oxygen": 0.05 if obs["o2_percent"] < 20 else -0.05,
        "ration_food": 1.0,
        "crew_activity": 0.8,
    }
    resp = requests.post(f"{BASE}/step", json={"session_id": session_id, "action": action})
    obs = resp.json()["observation"]
    if resp.json()["done"]:
        break

# Grade the episode
grade = requests.post(f"{BASE}/grade", json={"session_id": session_id, "task_id": "task_easy"})
print(f"Score: {grade.json()['score']}")
```

---

## 🤖 Baseline Inference

Run the LLM baseline agent against all 3 tasks:

```bash
export OPENAI_API_KEY=your_key_here
python baseline_inference.py \
    --host http://localhost:7860 \
    --model gpt-4o-mini \
    --seed 42
```

**Expected output:**
```
Task                      Score   Passed    Steps    Total R
----------------------------------------------------------------------
task_easy                0.7842     True       24      17.82
task_medium              0.5231     True      168      87.45
task_hard                0.3489    False      720     238.11
----------------------------------------------------------------------
AVERAGE                  0.5521
```

Results saved to `baseline_results.json`.

---

## 📊 Critical Safety Thresholds

| Parameter | Safe Range | Warning | Critical |
|-----------|-----------|---------|---------|
| O₂ | 19.5% – 23.5% | < 19.5% | < 15% (episode ends) |
| CO₂ | < 1000 ppm | > 1000 ppm | > 4500 ppm (episode ends) |
| Water | > 20L comfortable | < 10L | 0L |
| Crew health | > 0.8 optimal | < 0.5 | 0.0 (episode ends) |

---

## 🧪 Tests

```bash
python -m pytest tests/ -v
```

Tests cover: environment reset/step/state API, physics plausibility, grader determinism and score ordering, edge cases (empty trajectory, invalid task IDs).

---

## 🌟 What Makes This Novel

1. **Genuinely closed-loop**: waste water → plant transpiration → recycling → potable water → crew → waste. No open-ended resource injection.
2. **Competing objectives**: growing plants costs water (needed for crew) but produces O₂ (needed immediately). Every action has multi-system consequences.
3. **Shaped reward over time**: not just binary survive/die — every step rewards good management even before catastrophe.
4. **Hard task genuinely challenges frontier models**: 30 days × 8 crew forces multi-day planning that exceeds typical LLM context reasoning.
5. **No existing OpenEnv covers this domain**: space life support with biologically-coupled subsystems is entirely new territory.
