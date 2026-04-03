# optimal_agent.py
from environment import SmartGridEnvironment
from graders.agent_graders import AgentGrader
import numpy as np

class OptimalAgent:
    """
    OPTIMAL AGENT - Based on grid draw analysis
    Key insight: Need MINIMUM 0.6 grid draw to maintain stability
    """
    
    def __init__(self, env):
        self.env = env
        self.step_count = 0
    
    def get_action(self, observation):
        self.step_count += 1
        
        # Extract observations
        if hasattr(observation, 'get'):
            stability = observation.get('grid_stability', 1.0)
            battery = observation.get('battery_level', 50.0)
            solar = observation.get('solar_generation', 0)
            demand = observation.get('energy_demand', 50)
            hour = observation.get('hour_of_day', 12)
        else:
            stability = getattr(observation, 'grid_stability', 1.0)
            battery = getattr(observation, 'battery_level', 50.0)
            solar = getattr(observation, 'solar_generation', 0)
            demand = getattr(observation, 'energy_demand', 50)
            hour = getattr(observation, 'hour_of_day', 12)
        
        # ============================================
        # OPTIMAL STRATEGY (Based on test results)
        # ============================================
        
        # RULE 1: Critical stability - max everything
        if stability < 0.4:
            return {
                'battery_charge_rate': -0.7,
                'solar_usage_ratio': 1.0,
                'grid_draw_ratio': 0.8
            }
        
        # RULE 2: Low stability - high grid draw
        if stability < 0.6:
            return {
                'battery_charge_rate': -0.4,
                'solar_usage_ratio': 1.0,
                'grid_draw_ratio': 0.75
            }
        
        # RULE 3: Moderate stability - maintain
        if stability < 0.8:
            return {
                'battery_charge_rate': -0.2,
                'solar_usage_ratio': 1.0,
                'grid_draw_ratio': 0.7
            }
        
        # RULE 4: Good stability - optimize
        if stability >= 0.8:
            # Battery management
            if battery < 25:
                # Low battery - charge
                return {
                    'battery_charge_rate': 0.4,
                    'solar_usage_ratio': 1.0,
                    'grid_draw_ratio': 0.65
                }
            elif battery > 85:
                # High battery - discharge
                return {
                    'battery_charge_rate': -0.25,
                    'solar_usage_ratio': 1.0,
                    'grid_draw_ratio': 0.65
                }
            else:
                # Normal - maintain high grid draw
                return {
                    'battery_charge_rate': 0,
                    'solar_usage_ratio': 1.0,
                    'grid_draw_ratio': 0.65
                }
        
        # Default safe action (grid draw at least 0.6)
        return {
            'battery_charge_rate': 0,
            'solar_usage_ratio': 1.0,
            'grid_draw_ratio': 0.65
        }


class HighGridAgent:
    """
    Always use high grid draw (0.7-0.8) for maximum stability
    """
    
    def __init__(self, env):
        self.env = env
    
    def get_action(self, observation):
        if hasattr(observation, 'get'):
            stability = observation.get('grid_stability', 1.0)
            battery = observation.get('battery_level', 50.0)
        else:
            stability = getattr(observation, 'grid_stability', 1.0)
            battery = getattr(observation, 'battery_level', 50.0)
        
        # Always keep grid draw high
        grid_draw = 0.75
        
        # Battery strategy
        if stability < 0.6:
            battery_rate = -0.5
        elif battery < 30:
            battery_rate = 0.3
        elif battery > 85:
            battery_rate = -0.2
        else:
            battery_rate = 0
        
        return {
            'battery_charge_rate': battery_rate,
            'solar_usage_ratio': 1.0,
            'grid_draw_ratio': grid_draw
        }


def test_optimal_agents():
    """Test all optimized agents"""
    print("=" * 60)
    print("Testing OPTIMIZED Agents")
    print("=" * 60)
    
    grader = AgentGrader()
    
    agents = {
        'Optimal': OptimalAgent,
        'HighGrid': HighGridAgent
    }
    
    results = {}
    
    for name, AgentClass in agents.items():
        print(f"\n{'='*50}")
        print(f"Testing {name} Agent")
        print(f"{'='*50}")
        
        total_score = 0
        
        for difficulty in ['easy', 'medium', 'hard']:
            env = SmartGridEnvironment({'difficulty': difficulty})
            agent = AgentClass(env)
            
            score, details = grader.grade(agent, difficulty)
            total_score += score
            
            steps = details.get('steps', 0)
            stability = details.get('avg_stability', 0)
            
            # Check max steps for this difficulty
            if difficulty == 'easy':
                max_steps = 500
            elif difficulty == 'medium':
                max_steps = 1000
            else:
                max_steps = 1500
            
            status = "✅" if steps >= max_steps * 0.9 else "⚠️"
            
            print(f"{status} {difficulty.upper()}: Score={score:.3f}, Steps={steps}/{max_steps}, Stability={stability:.3f}")
        
        avg_score = total_score / 3
        results[name] = avg_score
        print(f"\n📊 {name} Average Score: {avg_score:.3f}")
    
    print("\n" + "=" * 60)
    print(f"🏆 WINNER: {max(results, key=results.get)}")
    print("=" * 60)
    
    return results


def test_easy_task_only():
    """Focus on easy task first"""
    print("\n" + "=" * 60)
    print("FOCUSED TEST: Easy Task Only")
    print("=" * 60)
    
    grader = AgentGrader()
    env = SmartGridEnvironment({'difficulty': 'easy'})
    
    # Test Optimal Agent
    print("\n1. Testing Optimal Agent...")
    agent = OptimalAgent(env)
    score, details = grader.grade(agent, 'easy')
    
    print(f"   Score: {score:.3f}/1.0")
    print(f"   Grade: {grader._get_grade(score)}")
    print(f"   Avg Stability: {details.get('avg_stability', 0):.3f}")
    print(f"   Steps: {details.get('steps', 0)}/500")
    print(f"   Total Reward: {details.get('total_reward', 0):.2f}")
    
    if details.get('steps', 0) >= 450:
        print("\n   ✅ SUCCESS! Agent completed Easy task!")
    else:
        print(f"\n   ⚠️ Agent failed at step {details.get('steps', 0)}")
    
    # Test HighGrid Agent
    print("\n2. Testing HighGrid Agent...")
    env = SmartGridEnvironment({'difficulty': 'easy'})
    agent = HighGridAgent(env)
    score, details = grader.grade(agent, 'easy')
    
    print(f"   Score: {score:.3f}/1.0")
    print(f"   Grade: {grader._get_grade(score)}")
    print(f"   Avg Stability: {details.get('avg_stability', 0):.3f}")
    print(f"   Steps: {details.get('steps', 0)}/500")
    print(f"   Total Reward: {details.get('total_reward', 0):.2f}")
    
    if details.get('steps', 0) >= 450:
        print("\n   ✅ SUCCESS! Agent completed Easy task!")
    else:
        print(f"\n   ⚠️ Agent failed at step {details.get('steps', 0)}")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("OPTIMAL AGENT BASED ON GRID DRAW ANALYSIS")
    print("Key Finding: Need MINIMUM 0.6 grid draw for stability")
    print("=" * 60)
    
    # First test easy task only
    test_easy_task_only()
    
    # Then test all tasks
    print("\n" + "=" * 60)
    test_optimal_agents()