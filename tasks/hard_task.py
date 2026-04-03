# tasks/hard_task.py
from typing import Dict, Any, Tuple
import numpy as np

class HardTask:
    def __init__(self):
        self.name = "Emergency Grid Recovery"
        self.description = "Recover from grid failures"
        self.max_steps = 1500
        self.target_stability = 0.7
    
    def evaluate(self, agent, env) -> Tuple[float, Dict[str, Any]]:
        # Set difficulty - use string directly
        env.task_difficulty = 'hard'
        obs = env.reset()
        
        total_reward = 0
        stability_history = []
        recovery_events = []
        
        for step in range(self.max_steps):
            action = agent.get_action(obs)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            stability_history.append(info['grid_stability'])
            
            # Check for recovery from low stability
            if len(stability_history) > 10:
                recent_min = min(stability_history[-10:])
                if info['grid_stability'] > 0.6 and recent_min < 0.3:
                    recovery_events.append(1)
                else:
                    recovery_events.append(0)
            
            if done:
                break
        
        avg_stability = np.mean(stability_history) if stability_history else 0
        recovery_rate = np.mean(recovery_events) if recovery_events else 0
        stability_score = min(1.0, avg_stability / self.target_stability)
        
        score = (stability_score * 0.5 + recovery_rate * 0.5)
        
        return score, {
            'score': score,
            'avg_stability': float(avg_stability),
            'recovery_rate': float(recovery_rate),
            'total_reward': float(total_reward),
            'steps': step + 1
        }