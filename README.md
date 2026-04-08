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

**Scaler School of Technology Hackathon | Team: BigByte**

An AI-driven life support simulation for autonomous space habitat management.

---

## 📖 Overview

This repository contains **Team BigByte's** submission for the **Scaler School of Technology Hackathon**. We have developed a high-fidelity, real-world OpenEnv environment where an AI agent manages the critical life support systems of a space habitat. 

The environment models a **genuinely closed-loop system** where every resource is interconnected. The agent must balance complex trade-offs: every action (like boosting plant growth) has secondary consequences in other subsystems (water usage and power drain), making it a true test of long-term planning and multi-objective optimization.

---

## 🎯 Why This Environment?

Autonomous life support is critical for the future of humanity in space.
- **Real-World Impact**: NASA's ECLSS on the ISS is still largely manual. Future Moon and Mars bases will require autonomous agents.
- **Complexity**: The coupling of waste-to-water, water-to-plants, and plants-to-oxygen creates one of the most challenging control problems in robotics.
- **Gap in OpenEnv**: No existing environment in the OpenEnv ecosystem covers biological + chemical resource management for life-critical systems.

---

## 🏗️ Project Structure

```text
Scaler/
├── env/                  # Core Physics & Simulation
│   ├── environment.py    # Main LifeSupportEnv class
│   └── models.py         # Pydantic Schemas (Observation, Action, State)
├── server/               # HTTP API Interface
│   └── app.py            # FastAPI implementation (OpenEnv compliant)
├── tasks/                # Challenge Configurations
│   ├── easy.py           # 24-hour stabilization
│   ├── medium.py         # 7-day resource balance
│   ├── hard.py           # 30-day closed-loop mission
│   └── graders.py        # Automated grading logic
├── tests/                # Validation Suite
├── env.py                # OpenEnv Root Entry Point
├── inference.py          # Baseline AI Agent (Submission Version)
├── openenv.yaml          # OpenEnv Metadata
├── pyproject.toml        # Build & Dependency Config
├── Dockerfile            # Multi-mode Deployment
└── README.md             # This Documentation
```

---

## 🌍 Environment Description

### Subsystem Coupling
| Subsystem | Inputs | Outputs |
|-----------|--------|---------|
| **Plant Bay** | Water, Power | Oxygen, Food, CO₂ removal |
| **Water Reclamation**| Wastewater, Transpiration | Potable Water |
| **Atmo Control** | Stored O₂, Plant O₂ | Balanced O₂/CO₂ % |
| **Power Grid** | Solar Budget | (Constraint for all systems) |

### Action Space (Continuous)
- `increase_plant_growth` [0, 1]: Boosts O₂/Food; consumes water + power.
- `recycle_water` [0, 1]: Intensity of reclamation.
- `adjust_oxygen` [-1, 1]: Manage O₂ concentration and CO₂ scrubbers.
- `ration_food` [0, 1]: Manage consumption vs. supply.
- `crew_activity` [0, 1]: Higher activity uses more O₂ but improves health scores.

---

## 📋 Tasks

| Task | Steps | Objective | Difficulty |
|------|-------|-----------|------------|
| **Stabilization** | 24 | Keep params in safe ranges for 1 day. | 🟢 Easy |
| **Balancing** | 168 | Sustain crew and maintain resource trends. | 🟡 Medium |
| **Optimization** | 720 | Fully closed-loop 30-day survival. | 🔴 Hard |

---

## 🚀 Setup & Usage

### 1. Local Installation
```bash
pip install -r requirements.txt
```

### 2. Start the Environment Server
Enable the OpenEnv-compliant API:
```bash
python -m server.app
```
*Server will start on `http://127.0.0.1:7860`*

### 3. Run the AI Agent
Run the baseline agent using your API key:
```bash
export HF_TOKEN="your_token"
python inference.py
```

---

## 📬 Submission — Team BigByte

This repository adheres to all **Scaler School of Technology Hackathon** requirements.

### Reproducibility
- **Validator**: Passed `openenv validate` suite.
- **Seed**: All results reproducible using `seed=42`.
- **Logs**: Strictly follows the `[START]`, `[STEP]`, `[END]` logging protocol for automated grading.

### Performance
The baseline agent demonstrates robust performance on stabilization tasks and serves as a foundation for deep reinforcement learning agents.

---

*Team BigByte · Scaler School of Technology Hackathon · 2026*