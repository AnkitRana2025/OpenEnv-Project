# graders/agent_graders.py
from typing import Dict, Any, Tuple
from tasks.easy_task import EasyTask
from tasks.medium_task import MediumTask
from tasks.hard_task import HardTask

class AgentGrader:
    def __init__(self):
        self.tasks = {
            'easy': EasyTask(),
            'medium': MediumTask(),
            'hard': HardTask()
        }
    
    def grade(self, agent, task_difficulty: str) -> Tuple[float, Dict]:
        if task_difficulty not in self.tasks:
            raise ValueError(f"Unknown difficulty: {task_difficulty}")
        
        task = self.tasks[task_difficulty]
        return task.evaluate(agent, agent.env)
    
    def full_evaluation(self, agent) -> Dict[str, Any]:
        results = {}
        total_score = 0
        
        for difficulty in ['easy', 'medium', 'hard']:
            score, details = self.grade(agent, difficulty)
            results[difficulty] = {'score': score, 'details': details}
            total_score += score
        
        results['overall_score'] = total_score / 3
        results['grade'] = self._get_grade(results['overall_score'])
        
        return results
    
    def _get_grade(self, score: float) -> str:
        if score >= 0.9:
            return 'A+'
        elif score >= 0.8:
            return 'A'
        elif score >= 0.7:
            return 'B'
        elif score >= 0.6:
            return 'C'
        elif score >= 0.5:
            return 'D'
        else:
            return 'F'