# tasks/medium_task.py
from typing import Dict, Any, Tuple
import numpy as np

class MediumTask:
    def __init__(self):
        self.name = "Peak Load Management"
        self.description = "Handle demand spikes while maintaining stability"
        self.max_steps = 1000
        self.target_stability = 0.8
    
    def evaluate(self, agent, env) -> Tuple[float, Dict[str, Any]]:
        # Set difficulty - use string directly
        env.task_difficulty = 'medium'
        obs = env.reset()
        
        total_reward = 0
        stability_history = []
        spike_handling = []
        
        for step in range(self.max_steps):
            action = agent.get_action(obs)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            stability_history.append(info['grid_stability'])
            
            # Check if spike was handled (stability maintained)
            if info['grid_stability'] > 0.7:
                spike_handling.append(1)
            else:
                spike_handling.append(0)
            
            if done:
                break
        
        avg_stability = np.mean(stability_history) if stability_history else 0
        spike_rate = np.mean(spike_handling) if spike_handling else 0
        stability_score = min(1.0, avg_stability / self.target_stability)
        
        score = (stability_score * 0.6 + spike_rate * 0.4)
        
        return score, {
            'score': score,
            'avg_stability': float(avg_stability),
            'spike_handling_rate': float(spike_rate),
            'total_reward': float(total_reward),
            'steps': step + 1
        }