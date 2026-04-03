# agents/baseline_agent.py - FIXED
import numpy as np

class BaselineAgent:
    """Improved Baseline Agent - with optimal grid draw 0.85"""
    
    def __init__(self, env):
        self.env = env
    
    def get_action(self, observation):
        if hasattr(observation, 'get'):
            stability = observation.get('grid_stability', 1.0)
            battery = observation.get('battery_level', 50.0)
            hour = observation.get('hour_of_day', 12)
        else:
            stability = getattr(observation, 'grid_stability', 1.0)
            battery = getattr(observation, 'battery_level', 50.0)
            hour = getattr(observation, 'hour_of_day', 12)
        
        # FIXED: Minimum grid draw 0.85 for stability
        if stability < 0.4:
            grid_draw = 0.95  # Emergency - max grid
        elif stability < 0.6:
            grid_draw = 0.90  # Critical - high grid
        elif stability < 0.8:
            grid_draw = 0.88  # Unstable - high grid
        else:
            grid_draw = 0.85  # Stable - minimum 0.85 (CRITICAL!)
        
        # Battery management
        if battery < 20:
            battery_charge = 0.6  # Emergency charge
        elif battery > 85:
            battery_charge = -0.3  # Discharge
        elif 10 <= hour <= 16:
            battery_charge = 0.2   # Daytime charge
        else:
            battery_charge = -0.15 # Night discharge
        
        return {
            'battery_charge_rate': battery_charge,
            'solar_usage_ratio': 1.0,
            'grid_draw_ratio': grid_draw
        }