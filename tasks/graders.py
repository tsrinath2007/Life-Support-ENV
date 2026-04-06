"""
Agent graders for the Closed-Loop Life Support environment.
Each grader evaluates a complete episode trajectory and returns a score in [0.0, 1.0].
Graders are deterministic given the same trajectory.
"""
from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class GradeResult:
    score: float          # 0.0 – 1.0
    passed: bool
    breakdown: Dict[str, float]
    feedback: str


def grade_easy(trajectory: List[Dict[str, Any]]) -> GradeResult:
    """
    Task: Single-Day Stabilization (24 steps, 3 crew)

    Grading criteria:
    - 50% — All critical parameters in safe ranges every step (weighted by violation severity)
    - 20% — Crew health above 0.8 throughout
    - 15% — Resource sustainability: water and food above comfortable margins at end
    - 15% — Episode completion
    """
    if not trajectory:
        return GradeResult(0.0, False, {}, "Empty trajectory")

    total_steps = len(trajectory)
    max_steps = 24

    safe_steps = 0
    high_health_steps = 0
    violation_penalty = 0.0

    first_obs = trajectory[0]["observation"]
    last_obs = trajectory[-1]["observation"]

    for step in trajectory:
        obs = step["observation"]
        o2_safe = 19.5 <= obs["o2_percent"] <= 23.5
        co2_safe = obs["co2_ppm"] <= 1000
        water_safe = obs["water_liters"] > 5.0

        step_violation = 0.0
        if not o2_safe:
            step_violation += abs(obs["o2_percent"] - 21.0) * 0.05
        if not co2_safe:
            step_violation += (obs["co2_ppm"] - 1000) / 1000 * 0.3
        if not water_safe:
            step_violation += 0.2

        if step_violation == 0.0:
            safe_steps += 1
        violation_penalty += step_violation

        if obs["crew_health"] >= 0.8:
            high_health_steps += 1

    safe_fraction = safe_steps / total_steps
    avg_violation = violation_penalty / total_steps
    safety_score = max(0.0, 1.0 - avg_violation) * 0.50

    health_fraction = high_health_steps / total_steps
    health_score = health_fraction * 0.20

    # Resource sustainability: reward keeping water + food above comfortable margins
    water_end_score = min(1.0, last_obs["water_liters"] / max(first_obs["water_liters"], 1.0))
    food_end_score = min(1.0, last_obs["food_kg"] / max(first_obs["food_kg"], 1.0))
    resource_score = (0.5 * water_end_score + 0.5 * food_end_score) * 0.15

    completion_score = (total_steps / max_steps) * 0.15

    total = min(1.0, safety_score + health_score + resource_score + completion_score)

    return GradeResult(
        score=round(total, 4),
        passed=total >= 0.6,
        breakdown={
            "safety_score": round(safety_score, 4),
            "health_score": round(health_score, 4),
            "resource_score": round(resource_score, 4),
            "completion_score": round(completion_score, 4),
            "safe_fraction": round(safe_fraction, 4),
            "water_end_score": round(water_end_score, 4),
            "food_end_score": round(food_end_score, 4),
            "steps_completed": total_steps,
        },
        feedback=(
            f"Completed {total_steps}/{max_steps} steps. "
            f"Safe parameters {safe_steps}/{total_steps} steps. "
            f"High health {high_health_steps}/{total_steps} steps. "
            f"Water end: {water_end_score:.2f}x, Food end: {food_end_score:.2f}x."
        )
    )


def grade_medium(trajectory: List[Dict[str, Any]]) -> GradeResult:
    """
    Task: 7-Day Resource Balance (168 steps, 5 crew)

    Grading criteria:
    - 40% — All parameters safe (weighted by severity of violations)
    - 25% — Positive resource trends (water/food at end >= start)
    - 20% — Crew health average above 0.7
    - 15% — Episode completion
    """
    if not trajectory:
        return GradeResult(0.0, False, {}, "Empty trajectory")

    total_steps = len(trajectory)
    max_steps = 168

    safe_steps = 0
    health_sum = 0.0
    total_penalty = 0.0

    first_obs = trajectory[0]["observation"]
    last_obs = trajectory[-1]["observation"]

    for step in trajectory:
        obs = step["observation"]
        step_penalty = 0.0

        if obs["o2_percent"] < 19.5:
            step_penalty += (19.5 - obs["o2_percent"]) * 0.1
        if obs["co2_ppm"] > 1000:
            step_penalty += (obs["co2_ppm"] - 1000) / 2000 * 0.5
        if obs["water_liters"] < 10.0:
            step_penalty += 0.3

        if step_penalty == 0:
            safe_steps += 1
        total_penalty += step_penalty
        health_sum += obs["crew_health"]

    # Safety score (penalized)
    avg_penalty = total_penalty / total_steps
    safety_score = max(0.0, 1.0 - avg_penalty) * 0.40

    # Resource trend score
    water_trend = min(1.0, max(0.0, last_obs["water_liters"] / max(first_obs["water_liters"], 1.0)))
    food_trend = min(1.0, max(0.0, last_obs["food_kg"] / max(first_obs["food_kg"], 1.0)))
    resource_score = (0.5 * water_trend + 0.5 * food_trend) * 0.25

    # Health score
    avg_health = health_sum / total_steps
    health_score = max(0.0, (avg_health - 0.5) / 0.5) * 0.20

    # Completion
    completion = (total_steps / max_steps) * 0.15

    total = min(1.0, safety_score + resource_score + health_score + completion)

    return GradeResult(
        score=round(total, 4),
        passed=total >= 0.5,
        breakdown={
            "safety_score": round(safety_score, 4),
            "resource_score": round(resource_score, 4),
            "health_score": round(health_score, 4),
            "completion": round(completion, 4),
            "avg_health": round(avg_health, 4),
            "water_trend": round(water_trend, 4),
            "food_trend": round(food_trend, 4),
        },
        feedback=(
            f"7-day mission: {total_steps}/{max_steps} steps. "
            f"Avg health: {avg_health:.2f}. "
            f"Water trend: {water_trend:.2f}x, Food trend: {food_trend:.2f}x."
        )
    )


def grade_hard(trajectory: List[Dict[str, Any]]) -> GradeResult:
    """
    Task: 30-Day Closed-Loop Optimization (720 steps, 8 crew)

    Grading criteria:
    - 30% — Closed-loop efficiency: fraction of water needs met by recycling
    - 25% — Crew health sustained (mean > 0.8, variance penalty)
    - 20% — Zero catastrophic events (O2 < 19%, CO2 > 2000ppm, water=0)
    - 15% — Food self-sufficiency (plant harvests vs consumption)
    - 10% — Mission completion
    """
    if not trajectory:
        return GradeResult(0.0, False, {}, "Empty trajectory")

    total_steps = len(trajectory)
    max_steps = 720

    health_vals = []
    catastrophic_events = 0
    recycling_rates = []
    plant_growth_rates = []

    first_food = trajectory[0]["observation"]["food_kg"]
    last_food = trajectory[-1]["observation"]["food_kg"]

    for step in trajectory:
        obs = step["observation"]
        health_vals.append(obs["crew_health"])
        recycling_rates.append(obs["water_recycling_rate"])
        plant_growth_rates.append(obs["plant_growth_rate"])

        # Count catastrophic moments
        if obs["o2_percent"] < 19.0:
            catastrophic_events += 1
        if obs["co2_ppm"] > 2000:
            catastrophic_events += 1
        if obs["water_liters"] < 1.0:
            catastrophic_events += 1

    # Closed-loop efficiency score
    avg_recycling = sum(recycling_rates) / len(recycling_rates)
    avg_plant_growth = sum(plant_growth_rates) / len(plant_growth_rates)
    loop_efficiency = (avg_recycling * 0.6 + avg_plant_growth * 0.4)
    efficiency_score = loop_efficiency * 0.30

    # Health score with variance penalty
    import statistics
    avg_health = statistics.mean(health_vals)
    health_variance = statistics.variance(health_vals) if len(health_vals) > 1 else 0
    health_score = max(0.0, avg_health - health_variance * 2) * 0.25

    # Catastrophic event penalty
    cat_rate = catastrophic_events / (total_steps * 3)  # 3 possible events per step
    safety_score = max(0.0, 1.0 - cat_rate * 10) * 0.20

    # Food self-sufficiency
    food_delta = last_food - first_food
    food_score = min(1.0, max(0.0, 0.5 + food_delta / 20.0)) * 0.15

    # Completion
    completion = (total_steps / max_steps) * 0.10

    total = min(1.0, efficiency_score + health_score + safety_score + food_score + completion)

    return GradeResult(
        score=round(total, 4),
        passed=total >= 0.4,
        breakdown={
            "efficiency_score": round(efficiency_score, 4),
            "health_score": round(health_score, 4),
            "safety_score": round(safety_score, 4),
            "food_score": round(food_score, 4),
            "completion": round(completion, 4),
            "avg_health": round(avg_health, 4),
            "catastrophic_events": catastrophic_events,
            "avg_recycling": round(avg_recycling, 4),
            "loop_efficiency": round(loop_efficiency, 4),
        },
        feedback=(
            f"30-day mission: {total_steps}/{max_steps} steps. "
            f"Loop efficiency: {loop_efficiency:.2f}. "
            f"Avg health: {avg_health:.2f}. "
            f"Catastrophic events: {catastrophic_events}. "
            f"Food delta: {food_delta:+.1f}kg."
        )
    )


GRADERS = {
    "task_easy": grade_easy,
    "task_medium": grade_medium,
    "task_hard": grade_hard,
}


def grade_episode(task_id: str, trajectory: List[Dict[str, Any]]) -> GradeResult:
    """Grade a complete episode trajectory for the given task."""
    if task_id not in GRADERS:
        raise ValueError(f"Unknown task_id: {task_id}")
    return GRADERS[task_id](trajectory)
