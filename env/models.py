"""
Pydantic models for the Closed-Loop Life Support OpenEnv environment.
"""
from pydantic import BaseModel, Field
from typing import Optional


class Observation(BaseModel):
    """Sensor readings from the habitat at each timestep."""
    co2_ppm: float = Field(..., ge=0, le=5000, description="CO2 in parts per million")
    o2_percent: float = Field(..., ge=0, le=30, description="O2 percentage in atmosphere")
    water_liters: float = Field(..., ge=0, le=500, description="Potable water available (L)")
    food_kg: float = Field(..., ge=0, le=100, description="Food supply (kg)")
    crew_size: int = Field(..., ge=1, le=10, description="Active crew members")
    plant_growth_rate: float = Field(..., ge=0, le=1, description="Normalized plant growth rate")
    water_recycling_rate: float = Field(..., ge=0, le=1, description="Water recycling efficiency")
    day: int = Field(..., ge=1, le=365, description="Mission day")
    crew_health: float = Field(..., ge=0, le=1, description="Crew health score (0=critical, 1=optimal)")
    power_budget: float = Field(..., ge=0, le=1, description="Remaining power budget fraction")


class Action(BaseModel):
    """Control inputs to the life support subsystems."""
    increase_plant_growth: float = Field(0.5, ge=0, le=1, description="Photosynthesis boost (uses power + water)")
    recycle_water: float = Field(0.5, ge=0, le=1, description="Water reclamation intensity")
    adjust_oxygen: float = Field(0.0, ge=-1, le=1, description="O2 release (+) or CO2 scrub (-)")
    ration_food: float = Field(1.0, ge=0, le=1, description="Food ration level")
    crew_activity: float = Field(0.7, ge=0, le=1, description="Permitted crew activity level")


class Reward(BaseModel):
    """Reward signal with breakdown for interpretability."""
    total: float = Field(..., ge=-1, le=1, description="Total shaped reward")
    health_component: float = Field(..., description="Reward from crew health")
    resource_component: float = Field(..., description="Reward from resource sustainability")
    efficiency_component: float = Field(..., description="Reward from power efficiency")
    penalty: float = Field(..., description="Penalty for critical failures")


class EnvironmentState(BaseModel):
    """Full internal state for reproducibility and debugging."""
    observation: Observation
    step_count: int
    episode_done: bool
    task_id: str
    total_reward: float
    failure_reason: Optional[str] = None
    # Internal simulation state
    co2_scrubber_efficiency: float = 0.85
    plant_biomass: float = 10.0  # kg of growing plants
    waste_water_buffer: float = 50.0  # liters of recoverable waste
