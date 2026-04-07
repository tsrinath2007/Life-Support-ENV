"""
FastAPI server — exposes DepUpgradeEnv over HTTP for HF Spaces.
Endpoints: POST /reset  POST /step  GET /state  GET /health  GET /tasks
"""

from fastapi import FastAPI, HTTPException
from env import DepUpgradeEnv, Action, Observation, Reward
from pydantic import BaseModel

app = FastAPI(title="DepUpgradeEnv", version="1.0.0")

_envs: dict[str, DepUpgradeEnv] = {}


def _get_env(task_id: str) -> DepUpgradeEnv:
    if task_id not in _envs:
        _envs[task_id] = DepUpgradeEnv(task_id=task_id)
    return _envs[task_id]


class ResetRequest(BaseModel):
    task_id: str = "easy"


class StepRequest(BaseModel):
    task_id: str = "easy"
    action: Action


class StepResponse(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: dict


@app.get("/health")
def health():
    return {"status": "ok", "env": "DepUpgradeEnv"}


@app.post("/reset", response_model=Observation)
def reset(req: ResetRequest):
    env = _get_env(req.task_id)
    return env.reset()


@app.post("/step", response_model=StepResponse)
def step(req: StepRequest):
    env = _get_env(req.task_id)
    try:
        obs, reward, done, info = env.step(req.action)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return StepResponse(observation=obs, reward=reward, done=done, info=info)


@app.get("/state")
def state(task_id: str = "easy"):
    return _get_env(task_id).state()


@app.get("/tasks")
def list_tasks():
    return {"tasks": [
        {"id": "easy",   "difficulty": "easy",   "description": "2 CVEs + 1 outdated package. Safe upgrades, no conflicts."},
        {"id": "medium", "difficulty": "medium",  "description": "2 CVEs + version conflicts. Upgrade order matters."},
        {"id": "hard",   "difficulty": "hard",   "description": "2 CVEs + diamond dependency conflict + locked package + breaking API change."},
    ]}
