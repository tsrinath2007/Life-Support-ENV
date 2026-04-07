"""
Closed-Loop Life Support Environment
OpenEnv-compliant simulation of space habitat resource management.
"""
import random
import math
from typing import Tuple, Dict, Any, Optional

from env.models import Observation, Action, Reward, EnvironmentState


# ── Physical constants ──────────────────────────────────────────────────────
O2_SAFE_MIN = 19.5          # % — below this crew suffocates
O2_SAFE_MAX = 23.5          # % — above this fire risk
CO2_SAFE_MAX = 1000         # ppm — above this cognitive impairment
CO2_CRITICAL = 3000         # ppm — above this incapacitation
WATER_PER_CREW_PER_STEP = 0.25   # liters per crew member per hour
FOOD_PER_CREW_PER_DAY = 0.7     # kg per crew member per day
O2_PER_CREW_PER_STEP = 0.04     # % consumed per crew per hour (activity-scaled)
CO2_PER_CREW_PER_STEP = 0.035   # % produced per crew per hour (activity-scaled)
PLANT_O2_PER_KG_PER_STEP = 0.008  # O2 % produced per kg of plants per hour
PLANT_CO2_CONSUME = 0.006       # CO2 ppm removed per kg plants per hour (×1000 scale)
WATER_FROM_PLANTS = 0.02        # liters transpired per kg plants per hour (recyclable)
MAX_PLANT_BIOMASS = 50.0        # kg
POWER_COST_PLANT = 0.3          # power fraction for max plant growth
POWER_COST_RECYCLE = 0.25       # power fraction for max water recycling
POWER_COST_O2_ADJUST = 0.15     # power fraction for O2 adjustment


class LifeSupportEnv:
    """
    Closed-loop life support simulation environment.

    Simulates a space habitat where an AI agent must balance:
    - Oxygen production (via plants + stored O2)
    - CO2 removal (via plants + chemical scrubbers)
    - Water recycling (closed-loop urine/transpiration recovery)
    - Food cultivation (plant biomass → crew food)
    - Power budget management

    API:
        env = LifeSupportEnv(task_id="task_easy")
        obs = env.reset()
        obs, reward, done, info = env.step(action)
        state = env.state()
    """

    TASK_CONFIGS = {
        "task_easy": {
            "crew_size": 3,
            "max_steps": 24,          # 24 hours
            "initial_water": 200.0,
            "initial_food": 30.0,
            "initial_o2": 21.0,
            "initial_co2": 400.0,
            "initial_plant_biomass": 15.0,
            "description": "Keep all parameters safe for 24h with 3-person crew.",
        },
        "task_medium": {
            "crew_size": 5,
            "max_steps": 168,         # 7 days × 24 hours
            "initial_water": 250.0,
            "initial_food": 50.0,
            "initial_o2": 21.0,
            "initial_co2": 450.0,
            "initial_plant_biomass": 20.0,
            "description": "Sustain 5-person crew for 7 days with positive resource trends.",
        },
        "task_hard": {
            "crew_size": 8,
            "max_steps": 720,         # 30 days × 24 hours
            "initial_water": 300.0,
            "initial_food": 60.0,
            "initial_o2": 21.0,
            "initial_co2": 400.0,
            "initial_plant_biomass": 25.0,
            "description": "30-day fully closed-loop mission with 8 crew. Maximize health, minimize imports.",
        },
    }

    def __init__(self, task_id: str = "task_easy", seed: Optional[int] = None):
        if task_id not in self.TASK_CONFIGS:
            raise ValueError(f"Unknown task_id: {task_id}. Choose from {list(self.TASK_CONFIGS.keys())}")
        self.task_id = task_id
        self.config = self.TASK_CONFIGS[task_id]
        self.rng = random.Random(seed)
        self._step_count = 0
        self._done = False
        self._total_reward = 0.0
        self._failure_reason: Optional[str] = None

        # Internal state
        self._co2_ppm: float = 0
        self._o2_percent: float = 0
        self._water: float = 0
        self._food: float = 0
        self._plant_biomass: float = 0
        self._waste_water_buffer: float = 0
        self._crew_health: float = 1.0
        self._water_recycling_rate: float = 0.5
        self._plant_growth_rate: float = 0.5
        self._co2_scrubber_efficiency: float = 0.85
        self._cumulative_health: float = 0.0

    # ── OpenEnv API ──────────────────────────────────────────────────────────

    def reset(self) -> Observation:
        """Reset to initial state and return first observation."""
        cfg = self.config
        self._step_count = 0
        self._done = False
        self._total_reward = 0.0
        self._failure_reason = None
        self._cumulative_health = 0.0

        self._co2_ppm = cfg["initial_co2"] + self.rng.uniform(-50, 50)
        self._o2_percent = cfg["initial_o2"] + self.rng.uniform(-0.3, 0.3)
        self._water = cfg["initial_water"]
        self._food = cfg["initial_food"]
        self._plant_biomass = cfg["initial_plant_biomass"]
        self._waste_water_buffer = 40.0
        self._crew_health = 1.0
        self._water_recycling_rate = 0.5
        self._plant_growth_rate = 0.5
        self._co2_scrubber_efficiency = 0.85

        return self._make_observation()

    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        """
        Apply action and advance simulation by one hour.

        Returns:
            observation: Updated sensor readings
            reward: Float in [-1, 1]
            done: Whether episode is over
            info: Debug information dict
        """
        if self._done:
            raise RuntimeError("Episode is done. Call reset() first.")

        self._step_count += 1

        # Clamp actions
        pg = max(0.0, min(1.0, action.increase_plant_growth))
        rw = max(0.0, min(1.0, action.recycle_water))
        ao = max(-1.0, min(1.0, action.adjust_oxygen))
        rf = max(0.0, min(1.0, action.ration_food))
        ca = max(0.0, min(1.0, action.crew_activity))

        crew = self.config["crew_size"]

        # ── Power budget ────────────────────────────────────────────────────
        power_used = (pg * POWER_COST_PLANT +
                      rw * POWER_COST_RECYCLE +
                      abs(ao) * POWER_COST_O2_ADJUST)
        power_budget = max(0.0, 1.0 - power_used)

        # If over budget, scale down all subsystems proportionally
        if power_used > 1.0:
            scale = 1.0 / power_used
            pg *= scale
            rw *= scale
            ao *= scale
            power_budget = 0.0

        # ── Plant simulation ─────────────────────────────────────────────────
        # Plants grow if they have water and light (power)
        water_for_plants = min(self._water * 0.05, pg * 2.0)
        growth_delta = pg * 0.5 * (water_for_plants / max(water_for_plants, 0.01))
        self._plant_biomass = min(MAX_PLANT_BIOMASS,
                                  self._plant_biomass + growth_delta - 0.1)
        self._plant_biomass = max(0.0, self._plant_biomass)
        self._plant_growth_rate = pg

        # Harvest mature plant mass for food (10% of excess over 30kg)
        if self._plant_biomass > 30.0:
            harvest = (self._plant_biomass - 30.0) * 0.1
            self._food = min(100.0, self._food + harvest * 0.8)  # 80% edible
            self._plant_biomass -= harvest

        # ── O2 / CO2 dynamics ────────────────────────────────────────────────
        crew_o2_consumed = crew * O2_PER_CREW_PER_STEP * ca
        crew_co2_produced = crew * CO2_PER_CREW_PER_STEP * ca * 1000  # to ppm scale

        plant_o2_produced = self._plant_biomass * PLANT_O2_PER_KG_PER_STEP
        plant_co2_consumed = self._plant_biomass * PLANT_CO2_CONSUME * 1000

        # Manual O2 adjustment (stored cylinders / electrolysis)
        o2_release = ao * 0.5  # max ±0.5% per step

        # CO2 chemical scrubber
        co2_scrubbed = self._co2_ppm * self._co2_scrubber_efficiency * abs(min(0.0, ao)) * 0.3

        self._o2_percent += plant_o2_produced - crew_o2_consumed + o2_release
        self._co2_ppm += crew_co2_produced - plant_co2_consumed - co2_scrubbed

        self._o2_percent = max(0.0, min(30.0, self._o2_percent))
        self._co2_ppm = max(0.0, min(5000.0, self._co2_ppm))

        # ── Water dynamics ───────────────────────────────────────────────────
        crew_water_used = crew * WATER_PER_CREW_PER_STEP
        plant_transpiration = self._plant_biomass * WATER_FROM_PLANTS
        self._waste_water_buffer += crew_water_used * 0.7 + plant_transpiration

        recycled = self._waste_water_buffer * rw * 0.9  # 90% purity
        self._water -= crew_water_used
        self._water = min(500.0, self._water + recycled)
        self._waste_water_buffer -= recycled
        self._waste_water_buffer = max(0.0, self._waste_water_buffer)
        self._water = max(0.0, self._water)
        self._water_recycling_rate = rw

        # ── Food dynamics ────────────────────────────────────────────────────
        food_consumed = crew * (FOOD_PER_CREW_PER_DAY / 24.0) * rf * ca
        self._food = max(0.0, self._food - food_consumed)

        # ── Crew health dynamics ─────────────────────────────────────────────
        health_delta = 0.0

        # O2 effects
        if self._o2_percent < O2_SAFE_MIN:
            health_delta -= 0.05 * (O2_SAFE_MIN - self._o2_percent)
        elif self._o2_percent > O2_SAFE_MAX:
            health_delta -= 0.01 * (self._o2_percent - O2_SAFE_MAX)

        # CO2 effects
        if self._co2_ppm > CO2_SAFE_MAX:
            health_delta -= 0.03 * (self._co2_ppm - CO2_SAFE_MAX) / 1000
        if self._co2_ppm > CO2_CRITICAL:
            health_delta -= 0.1

        # Water effects
        if self._water < 10.0:
            health_delta -= 0.04 * (10.0 - self._water) / 10.0

        # Food effects
        if self._food <= 0.0:
            health_delta -= 0.03

        # Natural recovery if all systems good
        if (O2_SAFE_MIN <= self._o2_percent <= O2_SAFE_MAX and
                self._co2_ppm < CO2_SAFE_MAX and
                self._water > 20.0 and self._food > 0.5):
            health_delta += 0.005

        self._crew_health = max(0.0, min(1.0, self._crew_health + health_delta))
        self._cumulative_health += self._crew_health

        # ── Reward calculation ───────────────────────────────────────────────
        reward_breakdown = self._compute_reward(power_budget, rf)
        reward = reward_breakdown.total
        self._total_reward += reward

        # ── Termination conditions ───────────────────────────────────────────
        done = False
        failure_reason = None

        if self._crew_health <= 0.0:
            done = True
            failure_reason = "Crew health reached critical zero"
        elif self._o2_percent < 15.0:
            done = True
            failure_reason = "O2 level below survivable threshold (15%)"
        elif self._co2_ppm > 4500:
            done = True
            failure_reason = "CO2 reached lethal concentration"
        elif self._step_count >= self.config["max_steps"]:
            done = True

        self._done = done
        self._failure_reason = failure_reason

        obs = self._make_observation(power_budget)
        info = {
            "step": self._step_count,
            "total_reward": self._total_reward,
            "failure_reason": failure_reason,
            "plant_biomass": self._plant_biomass,
            "waste_water_buffer": self._waste_water_buffer,
            "power_used": power_used,
            "reward_breakdown": reward_breakdown.dict(),
        }

        return obs, reward, done, info

    def state(self) -> EnvironmentState:
        """Return full internal state for debugging and checkpointing."""
        return EnvironmentState(
            observation=self._make_observation(),
            step_count=self._step_count,
            episode_done=self._done,
            task_id=self.task_id,
            total_reward=self._total_reward,
            failure_reason=self._failure_reason,
            co2_scrubber_efficiency=self._co2_scrubber_efficiency,
            plant_biomass=self._plant_biomass,
            waste_water_buffer=self._waste_water_buffer,
        )

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _make_observation(self, power_budget: float = 1.0) -> Observation:
        return Observation(
            co2_ppm=round(self._co2_ppm, 2),
            o2_percent=round(self._o2_percent, 3),
            water_liters=round(self._water, 2),
            food_kg=round(self._food, 3),
            crew_size=self.config["crew_size"],
            plant_growth_rate=round(self._plant_growth_rate, 3),
            water_recycling_rate=round(self._water_recycling_rate, 3),
            day=max(1, (self._step_count // 24) + 1),
            crew_health=round(self._crew_health, 4),
            power_budget=round(power_budget, 3),
        )

    def _compute_reward(self, power_budget: float, ration_level: float) -> Reward:
        penalty = 0.0

        # O2 component — most critical
        if O2_SAFE_MIN <= self._o2_percent <= O2_SAFE_MAX:
            o2_score = 1.0
        elif self._o2_percent < O2_SAFE_MIN:
            o2_score = max(-1.0, (self._o2_percent - O2_SAFE_MIN) / O2_SAFE_MIN)
            penalty += 0.2
        else:
            o2_score = max(0.0, 1.0 - (self._o2_percent - O2_SAFE_MAX) / 5.0)

        # CO2 component
        if self._co2_ppm <= CO2_SAFE_MAX:
            co2_score = 1.0 - (self._co2_ppm / CO2_SAFE_MAX) * 0.2
        else:
            co2_score = max(-1.0, 1.0 - (self._co2_ppm - CO2_SAFE_MAX) / 2000.0)
            if self._co2_ppm > CO2_CRITICAL:
                penalty += 0.3

        health_component = 0.5 * self._crew_health + 0.25 * o2_score + 0.25 * co2_score

        # Resource sustainability component
        water_score = min(1.0, self._water / 50.0)  # 50L is comfortable baseline
        food_score = min(1.0, self._food / 10.0)    # 10kg margin is good
        resource_component = 0.5 * water_score + 0.5 * food_score

        # Efficiency — reward using minimal power for adequate life support
        efficiency_component = power_budget * 0.3 if self._crew_health > 0.7 else 0.0

        # Critical resource failure penalties
        if self._water <= 0:
            penalty += 0.4
        if self._food <= 0:
            penalty += 0.2

        raw = (0.5 * health_component +
               0.3 * resource_component +
               0.2 * efficiency_component -
               penalty)

        total = max(-1.0, min(1.0, raw))

        return Reward(
            total=round(total, 4),
            health_component=round(health_component, 4),
            resource_component=round(resource_component, 4),
            efficiency_component=round(efficiency_component, 4),
            penalty=round(penalty, 4),
        )
