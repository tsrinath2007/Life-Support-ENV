"""
Inference Script — DepUpgradeEnv
==================================
MANDATORY env vars:
  API_BASE_URL  — The API endpoint for the LLM
  MODEL_NAME    — The model identifier
  HF_TOKEN      — Your Hugging Face / API key
"""

import os
import json
import textwrap
from typing import List

from openai import OpenAI
from dep_upgrade_env import DepUpgradeEnv, Action

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME   = os.getenv("MODEL_NAME")
MAX_STEPS    = 20
TEMPERATURE  = 0.2
MAX_TOKENS   = 400
FALLBACK_ACTION = '{"action_type": "validate"}'
DEBUG = True

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

SYSTEM_PROMPT = textwrap.dedent("""
    You are a dependency upgrade agent. You manage a Python project's requirements.txt.
    At each step reply with EXACTLY one JSON action and nothing else.

    Available actions:
      {"action_type": "upgrade",   "package": "<name>", "version": "<target>"}
      {"action_type": "pin",       "package": "<name>", "version": "<version>"}
      {"action_type": "remove",    "package": "<name>"}
      {"action_type": "run_tests"}
      {"action_type": "validate"}
      {"action_type": "skip"}

    Strategy:
    - Fix CVEs first, especially critical ones.
    - Resolve conflicts before upgrading conflicting packages.
    - Never upgrade locked packages.
    - Run tests after major upgrades to verify.
    - Output only valid JSON. No explanations.
""").strip()


def build_history(history: List[str]) -> str:
    return "\n".join(history[-5:]) if history else "None"


def build_prompt(step: int, obs, history: List[str]) -> str:
    pkgs = [
        f"  {p.name} {p.current_version} → {p.latest_version}"
        f"{' [CVE:'+p.cve_severity+']' if p.has_cve else ''}"
        f"{' [OUTDATED]' if p.is_outdated else ''}"
        f"{' [CONFLICT: '+p.conflict_reason+']' if p.is_conflicting else ''}"
        f"{' [LOCKED]' if p.locked else ''}"
        for p in obs.packages
    ]
    tests = [f"  {k}: {'PASS' if v else 'FAIL'}" for k, v in obs.test_results.items()]

    return textwrap.dedent(f"""
        Step: {step}  |  Task: {obs.task_id}  |  Score: {obs.score_so_far:.2f}
        
        Packages:
        {chr(10).join(pkgs)}
        
        Tests:
        {chr(10).join(tests)}
        
        Issues remaining: {obs.issues_remaining}
        
        History:
        {build_history(history)}
        
        Reply with exactly one JSON action.
    """).strip()


def parse_action(text: str) -> Action:
    text = text.strip()
    if "```" in text:
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    try:
        return Action(**json.loads(text))
    except Exception:
        if DEBUG:
            print(f"  [parse error] {text!r}")
        return Action(**json.loads(FALLBACK_ACTION))


def run_task(task_id: str) -> float:
    env = DepUpgradeEnv(task_id=task_id)
    obs = env.reset()
    history: List[str] = []
    final_score = 0.0

    print(f"\n{'='*55}")
    print(f"Task: {task_id.upper()}")
    print(f"{'='*55}")
    print(f"Issues: {obs.issues_remaining}")

    for step in range(1, MAX_STEPS + 1):
        prompt = build_prompt(step, obs, history)

        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                stream=False,
            )
            response_text = completion.choices[0].message.content or ""
        except Exception as exc:
            print(f"  Model error: {exc}. Using fallback.")
            response_text = FALLBACK_ACTION

        action = parse_action(response_text)
        if DEBUG:
            print(f"  Step {step}: {action.model_dump(exclude_none=True)}")

        obs, reward, done, info = env.step(action)
        final_score = reward.score

        history.append(
            f"Step {step}: {action.action_type}"
            f"({action.package or ''}) → score {reward.score:.2f}"
        )

        print(f"  Reward: {reward.score:.4f} | Done: {done} | {reward.breakdown}")

        if done:
            print(f"  Episode complete at step {step}.")
            break
    else:
        print(f"  Reached max steps ({MAX_STEPS}).")

    print(f"  Final [{task_id}]: {final_score:.4f}")
    return final_score


def main():
    scores = {}
    for task_id in ["easy", "medium", "hard"]:
        scores[task_id] = run_task(task_id)

    print(f"\n{'='*55}")
    print("BASELINE SCORES")
    print("="*55)
    for t, s in scores.items():
        print(f"  {t:<8}: {s:.4f}")
    avg = sum(scores.values()) / len(scores)
    print(f"  {'average':<8}: {avg:.4f}")


if __name__ == "__main__":
    main()
