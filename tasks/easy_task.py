# tasks/easy_task.py
from typing import Dict, Any, Tuple
import numpy as np

class EasyTask:
    def __init__(self):
        self.name = "Basic Load Balancing"
        self.description = "Maintain grid stability above 0.9"
        self.max_steps = 500
        self.target_stability = 0.9
    
    def evaluate(self, agent, env) -> Tuple[float, Dict[str, Any]]:
        env.task_difficulty = 'easy'
        obs = env.reset()
        
        total_reward = 0
        stability_history = []
        
        for step in range(self.max_steps):
            action = agent.get_action(obs)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            stability_history.append(info['grid_stability'])
            
            if done:
                break
        
        avg_stability = np.mean(stability_history) if stability_history else 0
        score = min(1.0, avg_stability / self.target_stability)
        
        return score, {
            'score': score,
            'avg_stability': float(avg_stability),
            'total_reward': float(total_reward),
            'steps': step + 1
        }