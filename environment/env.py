# environment/env.py
from typing import Any, Dict, Tuple
import numpy as np
from .models import EnvironmentState, Action

class SmartGridEnvironment:
    """OpenEnv-compliant Smart Grid Environment"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.task_difficulty = self.config.get('difficulty', 'easy')
        self.max_steps = self._get_max_steps()
        self.current_step = 0
        self._state = None
        self.reset()
    
    def reset(self) -> Dict[str, Any]:
        """Reset environment"""
        base_demand = 50.0
        base_solar = 20.0
        
        if self.task_difficulty == 'medium':
            base_demand += np.random.normal(0, 10)
            base_solar += np.random.normal(0, 5)
        elif self.task_difficulty == 'hard':
            base_demand += np.random.normal(0, 20)
            base_solar += np.random.normal(0, 10)
        
        self._state = EnvironmentState(
            timestamp=0,
            energy_demand=max(0, base_demand),
            solar_generation=max(0, base_solar),
            battery_level=50.0,
            grid_price=0.15,
            grid_stability=1.0,
            co2_emissions=0.0,
            total_cost=0.0,
            hour_of_day=8,
            day=0
        )
        
        self.current_step = 0
        return self._get_observation()
    
    def step(self, action: Dict[str, float]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Execute one step"""
        action_obj = Action(
            battery_charge_rate=action.get('battery_charge_rate', 0),
            solar_usage_ratio=action.get('solar_usage_ratio', 0.5),
            grid_draw_ratio=action.get('grid_draw_ratio', 0.5)
        )
        
        self._update_demand_and_supply()
        self._process_action(action_obj)
        reward = self._calculate_reward(action_obj)
        
        self.current_step += 1
        self._state.timestamp = self.current_step
        done = self.current_step >= self.max_steps or self._state.grid_stability <= 0
        
        info = {
            'task_difficulty': self.task_difficulty,
            'step': self.current_step,
            'grid_stability': float(self._state.grid_stability),
            'battery_level': float(self._state.battery_level),
            'total_cost': float(self._state.total_cost)
        }
        
        return self._get_observation(), reward, done, info
    
    @property
    def state(self) -> EnvironmentState:
        return self._state
    
    def _update_demand_and_supply(self):
        """Update demand and solar"""
        self._state.hour_of_day = (self._state.hour_of_day + 1) % 24
        if self._state.hour_of_day == 0:
            self._state.day += 1
        
        hour = self._state.hour_of_day
        
        # Demand pattern
        if 6 <= hour <= 9:
            demand_factor = 1.5
        elif 17 <= hour <= 20:
            demand_factor = 1.8
        elif hour <= 5 or hour >= 22:
            demand_factor = 0.6
        else:
            demand_factor = 1.0
        
        base_demand = 50 * demand_factor
        
        # Difficulty-based variability
        if self.task_difficulty == 'easy':
            noise = np.random.normal(0, 2)
        elif self.task_difficulty == 'medium':
            noise = np.random.normal(0, 10)
            if np.random.random() < 0.05:
                base_demand *= 1.5
        else:
            noise = np.random.normal(0, 20)
            if np.random.random() < 0.1:
                base_demand *= np.random.uniform(1.5, 2.5)
            self._state.grid_price *= np.random.uniform(0.8, 1.2)
            self._state.grid_price = np.clip(self._state.grid_price, 0.05, 0.50)
        
        self._state.energy_demand = max(0, base_demand + noise)
        
        # Solar generation
        if 6 <= hour <= 18:
            solar_factor = np.sin(np.pi * (hour - 6) / 12)
            base_solar = 80 * solar_factor
        else:
            base_solar = 0
        
        if self.task_difficulty == 'easy':
            solar_noise = np.random.normal(0, 5)
            cloud_prob = 0.02
        elif self.task_difficulty == 'medium':
            solar_noise = np.random.normal(0, 15)
            cloud_prob = 0.05
        else:
            solar_noise = np.random.normal(0, 25)
            cloud_prob = 0.1
        
        if np.random.random() < cloud_prob:
            base_solar *= np.random.uniform(0.1, 0.5)
        
        self._state.solar_generation = max(0, base_solar + solar_noise)
    
    def _process_action(self, action: Action):
        """Process action"""
        solar_used = self._state.solar_generation * action.solar_usage_ratio
        grid_used = self._state.energy_demand * action.grid_draw_ratio
        battery_power = action.battery_charge_rate * 20
        
        self._state.battery_level += battery_power / 100
        self._state.battery_level = np.clip(self._state.battery_level, 0, 100)
        
        if battery_power < 0:
            solar_used += abs(battery_power)
        
        total_supplied = solar_used + grid_used
        unmet_demand = max(0, self._state.energy_demand - total_supplied)
        
        self._state.total_cost += grid_used * self._state.grid_price
        self._state.co2_emissions += grid_used * 0.4
        
        if unmet_demand > 0:
            stability_loss = (unmet_demand / self._state.energy_demand) * 0.1
            self._state.grid_stability -= stability_loss
        elif total_supplied > 0:
            self._state.grid_stability += 0.01
        
        self._state.grid_stability = np.clip(self._state.grid_stability, 0, 1)
    
    def _calculate_reward(self, action: Action) -> float:
        """Calculate reward"""
        # Stability reward (most important)
        if self._state.grid_stability > 0.9:
            stability_reward = 10
        elif self._state.grid_stability > 0.7:
            stability_reward = 5
        elif self._state.grid_stability > 0.5:
            stability_reward = 0
        else:
            stability_reward = -20
        
        # Cost penalty
        cost_penalty = -self._state.total_cost / 100
        
        # Solar bonus
        solar_bonus = action.solar_usage_ratio * 5
        
        # Demand satisfaction
        solar_used = self._state.solar_generation * action.solar_usage_ratio
        grid_used = self._state.energy_demand * action.grid_draw_ratio
        battery_power = action.battery_charge_rate * 20
        
        if battery_power < 0:
            solar_used += abs(battery_power)
        
        total_supplied = solar_used + grid_used
        satisfaction = min(1.0, total_supplied / max(0.001, self._state.energy_demand))
        demand_reward = satisfaction * 8
        
        # Penalties
        penalty = 0
        if self._state.battery_level < 20:
            penalty -= 3
        if self._state.grid_stability < 0.3:
            penalty -= 15
        
        total = stability_reward + cost_penalty + solar_bonus + demand_reward + penalty
        return float(total)
    
    def _get_max_steps(self) -> int:
        if self.task_difficulty == 'easy':
            return 500
        elif self.task_difficulty == 'medium':
            return 1000
        return 1500
    
    def _get_observation(self) -> Dict[str, Any]:
        return {
            'energy_demand': float(self._state.energy_demand),
            'solar_generation': float(self._state.solar_generation),
            'battery_level': float(self._state.battery_level),
            'grid_price': float(self._state.grid_price),
            'grid_stability': float(self._state.grid_stability),
            'hour_of_day': int(self._state.hour_of_day),
            'day': int(self._state.day),
            'timestamp': int(self._state.timestamp)
        }
    
    def render(self, mode='human'):
        if mode == 'human':
            print(f"Step {self.current_step}: Demand={self._state.energy_demand:.1f}kW, "
                  f"Solar={self._state.solar_generation:.1f}kW, "
                  f"Battery={self._state.battery_level:.1f}%, "
                  f"Stability={self._state.grid_stability:.2f}")