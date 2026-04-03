# environment/models.py
from pydantic import BaseModel, Field

class EnvironmentState(BaseModel):
    """Typed state model for OpenEnv"""
    timestamp: int = Field(default=0)
    energy_demand: float = Field(default=50.0, ge=0)
    solar_generation: float = Field(default=20.0, ge=0)
    battery_level: float = Field(default=50.0, ge=0, le=100)
    grid_price: float = Field(default=0.15, ge=0)
    grid_stability: float = Field(default=1.0, ge=0, le=1)
    co2_emissions: float = Field(default=0.0, ge=0)
    total_cost: float = Field(default=0.0, ge=0)
    hour_of_day: int = Field(default=8, ge=0, le=23)
    day: int = Field(default=0, ge=0)

class Action(BaseModel):
    """Action model for agent"""
    battery_charge_rate: float = Field(default=0, ge=-1, le=1)
    solar_usage_ratio: float = Field(default=0.5, ge=0, le=1)
    grid_draw_ratio: float = Field(default=0.5, ge=0, le=1)