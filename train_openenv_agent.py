# train_openenv_agent.py - UPDATED with Grid Draw Fix (0.85)
import numpy as np
import json
import os
import random
from environment import SmartGridEnvironment
from graders.agent_graders import AgentGrader
import time
import matplotlib.pyplot as plt

# ============================================================
# OPTIMAL AGENT (UPDATED with Grid Draw 0.85)
# ============================================================

class OptimalAgent:
    """Optimal agent - ALWAYS maintains grid draw >= 0.85"""
    
    def __init__(self, env):
        self.env = env
        self.step_count = 0
    
    def get_action(self, observation):
        self.step_count += 1
        
        if hasattr(observation, 'get'):
            stability = observation.get('grid_stability', 1.0)
            battery = observation.get('battery_level', 50.0)
            solar = observation.get('solar_generation', 0)
            hour = observation.get('hour_of_day', 12)
        else:
            stability = getattr(observation, 'grid_stability', 1.0)
            battery = getattr(observation, 'battery_level', 50.0)
            solar = getattr(observation, 'solar_generation', 0)
            hour = getattr(observation, 'hour_of_day', 12)
        
        # CRITICAL FIX: Minimum grid draw = 0.85 (from testing)
        if stability < 0.3:
            return {'battery_charge_rate': -0.8, 'solar_usage_ratio': 1.0, 'grid_draw_ratio': 0.95}
        if stability < 0.5:
            return {'battery_charge_rate': -0.5, 'solar_usage_ratio': 1.0, 'grid_draw_ratio': 0.90}
        if stability < 0.7:
            return {'battery_charge_rate': -0.3, 'solar_usage_ratio': 1.0, 'grid_draw_ratio': 0.88}
        if stability < 0.85:
            return {'battery_charge_rate': -0.1, 'solar_usage_ratio': 1.0, 'grid_draw_ratio': 0.87}
        
        # Stable zone - minimum 0.85 grid draw
        if battery < 20:
            return {'battery_charge_rate': 0.5, 'solar_usage_ratio': 1.0, 'grid_draw_ratio': 0.85}
        if battery > 85:
            return {'battery_charge_rate': -0.3, 'solar_usage_ratio': 1.0, 'grid_draw_ratio': 0.85}
        if 7 <= hour <= 17 and solar > 20:
            return {'battery_charge_rate': 0.1 if battery < 70 else -0.05, 'solar_usage_ratio': 1.0, 'grid_draw_ratio': 0.85}
        
        return {'battery_charge_rate': 0, 'solar_usage_ratio': 1.0, 'grid_draw_ratio': 0.85}


# ============================================================
# BASELINE AGENT (UPDATED with Grid Draw 0.85)
# ============================================================

class BaselineAgent:
    """Baseline agent - UPDATED with minimum grid draw 0.85"""
    
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
        
        # FIXED: Minimum grid draw 0.85 (from testing)
        if stability < 0.4:
            grid_draw = 0.95
        elif stability < 0.6:
            grid_draw = 0.90
        elif stability < 0.8:
            grid_draw = 0.88
        else:
            grid_draw = 0.85  # NEVER below 0.85!
        
        # Battery management
        if battery < 20:
            battery_charge = 0.6
        elif battery > 85:
            battery_charge = -0.3
        elif 10 <= hour <= 16:
            battery_charge = 0.2
        else:
            battery_charge = -0.15
        
        return {
            'battery_charge_rate': battery_charge,
            'solar_usage_ratio': 1.0,
            'grid_draw_ratio': grid_draw
        }


# ============================================================
# Q-LEARNING AGENT (UPDATED with Grid Draw Fix)
# ============================================================

class OpenEnvQLearningAgent:
    """Q-Learning Agent - Learns but maintains minimum grid draw"""
    
    def __init__(self, env, learning_rate=0.1, discount=0.95, epsilon=0.2):
        self.env = env
        self.learning_rate = learning_rate
        self.discount = discount
        self.epsilon = epsilon
        self.q_table = {}
        self.training_history = []
        
    def _get_state_key(self, observation):
        """Convert observation to discrete state"""
        stability = int(observation.get('grid_stability', 1.0) * 10)
        battery = int(observation.get('battery_level', 50) / 10)
        return f"{stability}_{battery}"
    
    def _get_action_key(self, action):
        """Convert action to discrete action"""
        grid_draw = int(action['grid_draw_ratio'] * 4)  # 0-4 (0.0, 0.25, 0.5, 0.75, 1.0)
        battery_rate = int((action['battery_charge_rate'] + 1) * 2)
        return f"{battery_rate}_{grid_draw}"
    
    def _action_from_key(self, action_key):
        """Convert discrete action back to continuous with MINIMUM 0.85"""
        parts = action_key.split('_')
        battery_rate = (int(parts[0]) / 2) - 1
        
        # CRITICAL FIX: Minimum grid draw options (0.85 to 1.0)
        grid_options = [0.85, 0.88, 0.90, 0.92, 0.95, 0.98, 1.0]
        grid_idx = min(int(parts[1]), len(grid_options) - 1)
        grid_draw = grid_options[grid_idx]
        
        return {
            'battery_charge_rate': np.clip(battery_rate, -1, 1),
            'solar_usage_ratio': 1.0,
            'grid_draw_ratio': grid_draw
        }
    
    def _get_all_actions(self):
        """Generate all possible discrete actions (only valid grid draws)"""
        actions = []
        for battery in range(5):  # -1 to 1
            for grid in range(7):  # 0.85 to 1.0
                actions.append(f"{battery}_{grid}")
        return actions
    
    def get_action(self, observation, training=True):
        """Choose action using epsilon-greedy policy"""
        state_key = self._get_state_key(observation)
        
        if training and random.random() < self.epsilon:
            # Explore - but always valid grid draws
            grid_draw = random.choice([0.85, 0.88, 0.90, 0.92, 0.95, 0.98, 1.0])
            battery_rate = random.uniform(-0.5, 0.5)
            return {
                'battery_charge_rate': battery_rate,
                'solar_usage_ratio': 1.0,
                'grid_draw_ratio': grid_draw
            }
        else:
            # Exploit
            if state_key in self.q_table and self.q_table[state_key]:
                best_action = max(self.q_table[state_key], key=self.q_table[state_key].get)
                return self._action_from_key(best_action)
            # Default safe action
            return {
                'battery_charge_rate': 0,
                'solar_usage_ratio': 1.0,
                'grid_draw_ratio': 0.85
            }
    
    def learn(self, observation, action, reward, next_observation, done):
        """Update Q-table using Q-learning formula"""
        state_key = self._get_state_key(observation)
        action_key = self._get_action_key(action)
        next_state_key = self._get_state_key(next_observation)
        
        if state_key not in self.q_table:
            self.q_table[state_key] = {}
        if action_key not in self.q_table[state_key]:
            self.q_table[state_key][action_key] = 0
        
        max_next_q = 0
        if not done and next_state_key in self.q_table and self.q_table[next_state_key]:
            max_next_q = max(self.q_table[next_state_key].values())
        
        old_q = self.q_table[state_key][action_key]
        new_q = old_q + self.learning_rate * (reward + self.discount * max_next_q - old_q)
        self.q_table[state_key][action_key] = new_q
    
    def save(self, filepath):
        """Save trained Q-table"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            q_table_serializable = {}
            for state, actions in self.q_table.items():
                q_table_serializable[state] = {}
                for action, value in actions.items():
                    q_table_serializable[state][action] = float(value)
            json.dump(q_table_serializable, f)
        print(f"✅ Model saved to {filepath}")
    
    def load(self, filepath):
        """Load trained Q-table"""
        with open(filepath, 'r') as f:
            self.q_table = json.load(f)
        print(f"✅ Model loaded from {filepath}")
    
    def is_trained(self):
        return len(self.q_table) > 0


# ============================================================
# TRAINING FUNCTION
# ============================================================

def train_agent(difficulty='easy', episodes=100):
    """Train Q-Learning agent on OpenEnv environment"""
    
    max_steps = {'easy': 500, 'medium': 1000, 'hard': 1500}[difficulty]
    
    print("=" * 60)
    print(f"🤖 Training OpenEnv Q-Learning Agent")
    print(f"📈 Task: {difficulty.upper()}")
    print(f"🎯 Episodes: {episodes}")
    print(f"🔄 Max Steps per Episode: {max_steps}")
    print("=" * 60)
    
    env = SmartGridEnvironment({'difficulty': difficulty})
    agent = OpenEnvQLearningAgent(env, learning_rate=0.1, discount=0.95, epsilon=0.3)
    
    training_stats = []
    best_steps = 0
    
    for episode in range(episodes):
        obs = env.reset()
        total_reward = 0
        steps = 0
        done = False
        
        # Decay epsilon
        agent.epsilon = max(0.05, 0.3 * (1 - episode / episodes))
        
        while not done and steps < max_steps:
            action = agent.get_action(obs, training=True)
            next_obs, reward, done, info = env.step(action)
            agent.learn(obs, action, reward, next_obs, done)
            total_reward += reward
            obs = next_obs
            steps += 1
        
        training_stats.append({'episode': episode + 1, 'reward': total_reward, 'steps': steps})
        
        if steps > best_steps:
            best_steps = steps
            # Save best model
            agent.save(f"trained_models/openenv_qlearning_{difficulty}_best.json")
        
        if (episode + 1) % 20 == 0:
            avg_reward = np.mean([s['reward'] for s in training_stats[-20:]])
            avg_steps = np.mean([s['steps'] for s in training_stats[-20:]])
            print(f"Episode {episode+1}/{episodes}: Reward={avg_reward:.2f}, Steps={avg_steps:.1f}, Best={best_steps}")
    
    # Save final model
    model_path = f"trained_models/openenv_qlearning_{difficulty}.json"
    agent.save(model_path)
    
    # Plot training progress
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot([s['episode'] for s in training_stats], [s['steps'] for s in training_stats])
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.title(f'Training Progress - {difficulty.upper()} Task')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot([s['episode'] for s in training_stats], [s['reward'] for s in training_stats])
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Reward Progress')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'training_progress_{difficulty}.png')
    plt.close()
    
    print(f"\n✅ Training Complete!")
    print(f"📊 Best Steps: {best_steps}/{max_steps}")
    print(f"📈 Training chart saved to training_progress_{difficulty}.png")
    
    return agent, training_stats


# ============================================================
# EVALUATE FUNCTION
# ============================================================

def evaluate_agent(difficulty='easy', model_path=None, episodes=10):
    """Evaluate trained Q-Learning agent"""
    
    max_steps = {'easy': 500, 'medium': 1000, 'hard': 1500}[difficulty]
    
    print("\n" + "=" * 60)
    print(f"📊 Evaluating Trained Agent on {difficulty.upper()} Task")
    print("=" * 60)
    
    env = SmartGridEnvironment({'difficulty': difficulty})
    agent = OpenEnvQLearningAgent(env)
    
    if model_path and os.path.exists(model_path):
        agent.load(model_path)
    else:
        print("⚠️ No trained model found!")
        return 0, {'avg_steps': 0}
    
    agent.epsilon = 0
    
    total_rewards = []
    total_steps = []
    stability_history_all = []
    
    for episode in range(episodes):
        obs = env.reset()
        episode_reward = 0
        steps = 0
        done = False
        stability_history = []
        
        while not done and steps < max_steps:
            action = agent.get_action(obs, training=False)
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            steps += 1
            stability_history.append(info['grid_stability'])
        
        total_rewards.append(episode_reward)
        total_steps.append(steps)
        stability_history_all.append(np.mean(stability_history))
        
        status = "✅" if steps >= max_steps - 10 else "⚠️"
        print(f"{status} Episode {episode+1}: Reward={episode_reward:.2f}, Steps={steps}, Stability={np.mean(stability_history):.3f}")
    
    avg_reward = np.mean(total_rewards)
    avg_steps = np.mean(total_steps)
    avg_stability = np.mean(stability_history_all)
    
    print(f"\n📊 Average Results:")
    print(f"   Reward: {avg_reward:.2f}")
    print(f"   Steps: {avg_steps:.1f}")
    print(f"   Stability: {avg_stability:.3f}")
    
    score = min(1.0, avg_steps / max_steps)
    print(f"\n🎯 Score: {score:.3f}/1.0")
    
    return score, {'avg_steps': avg_steps, 'avg_reward': avg_reward, 'avg_stability': avg_stability}


# ============================================================
# COMPARE ALL AGENTS
# ============================================================

def compare_all_agents(difficulty='easy'):
    """Compare Baseline, Optimal, and Trained Q-Learning Agent"""
    
    grader = AgentGrader()
    max_steps = {'easy': 500, 'medium': 1000, 'hard': 1500}[difficulty]
    
    print("\n" + "=" * 70)
    print(f"🏆 AGENT COMPARISON - {difficulty.upper()} TASK")
    print("=" * 70)
    
    results = {}
    
    # 1. Baseline Agent (UPDATED)
    print("\n1️⃣ Baseline Agent (UPDATED - Grid Draw 0.85):")
    env = SmartGridEnvironment({'difficulty': difficulty})
    baseline_agent = BaselineAgent(env)
    baseline_score, baseline_details = grader.grade(baseline_agent, difficulty)
    baseline_steps = baseline_details.get('steps', 0)
    print(f"   Score: {baseline_score:.3f}, Steps: {baseline_steps}/{max_steps}, Stability: {baseline_details.get('avg_stability', 0):.3f}")
    
    # 2. Optimal Agent (UPDATED)
    print("\n2️⃣ Optimal Agent (UPDATED - Grid Draw 0.85):")
    env = SmartGridEnvironment({'difficulty': difficulty})
    optimal_agent = OptimalAgent(env)
    optimal_score, optimal_details = grader.grade(optimal_agent, difficulty)
    optimal_steps = optimal_details.get('steps', 0)
    print(f"   Score: {optimal_score:.3f}, Steps: {optimal_steps}/{max_steps}, Stability: {optimal_details.get('avg_stability', 0):.3f}")
    
    # 3. Trained Q-Learning Agent
    print("\n3️⃣ Trained Q-Learning Agent:")
    model_path = f"trained_models/openenv_qlearning_{difficulty}_best.json"
    if os.path.exists(model_path):
        rl_score, rl_details = evaluate_agent(difficulty, model_path, episodes=5)
        rl_steps = rl_details.get('avg_steps', 0)
    else:
        # Try without _best
        model_path = f"trained_models/openenv_qlearning_{difficulty}.json"
        if os.path.exists(model_path):
            rl_score, rl_details = evaluate_agent(difficulty, model_path, episodes=5)
            rl_steps = rl_details.get('avg_steps', 0)
        else:
            rl_score = 0
            rl_steps = 0
            print("   ⚠️ Trained model not found! Run training first.")
    
    # Summary Table
    print("\n" + "=" * 70)
    print("📊 FINAL COMPARISON RESULTS")
    print("=" * 70)
    print(f"{'Agent Type':<30} {'Score':<10} {'Steps':<12} {'Status'}")
    print("-" * 70)
    
    baseline_status = "✅ COMPLETE" if baseline_steps >= max_steps - 10 else "❌ FAILED"
    optimal_status = "✅ COMPLETE" if optimal_steps >= max_steps - 10 else "❌ FAILED"
    rl_status = "✅ COMPLETE" if rl_steps >= max_steps - 10 else "❌ FAILED"
    
    print(f"{'Baseline Agent (Updated)':<30} {baseline_score:<10.3f} {baseline_steps:<10}/{max_steps} {baseline_status}")
    print(f"{'Optimal Agent (Updated)':<30} {optimal_score:<10.3f} {optimal_steps:<10}/{max_steps} {optimal_status}")
    print(f"{'Q-Learning (Trained)':<30} {rl_score:<10.3f} {rl_steps:<10.0f}/{max_steps} {rl_status}")
    print("=" * 70)
    
    # Recommendation
    print("\n💡 RECOMMENDATION:")
    if optimal_steps >= max_steps - 10:
        print("   ✅ Use OPTIMAL AGENT for best results - it consistently completes all steps!")
    elif baseline_steps >= max_steps - 10:
        print("   ✅ Use BASELINE AGENT - it now completes all steps!")
    else:
        print("   ⚠️ Train Q-Learning agent with more episodes: --episodes 200")
    
    return {
        'baseline': baseline_score,
        'optimal': optimal_score,
        'qlearning': rl_score
    }


# ============================================================
# QUICK TEST FUNCTION
# ============================================================

def quick_test():
    """Quick test to verify grid draw fix"""
    print("\n" + "=" * 60)
    print("🔍 Quick Test: Grid Draw 0.85 Fix")
    print("=" * 60)
    
    env = SmartGridEnvironment({'difficulty': 'easy'})
    
    # Test with optimal grid draw
    print("\nTesting with Grid Draw = 0.85:")
    obs = env.reset()
    steps = 0
    
    for step in range(500):
        action = {
            'battery_charge_rate': 0,
            'solar_usage_ratio': 1.0,
            'grid_draw_ratio': 0.85
        }
        obs, reward, done, info = env.step(action)
        steps += 1
        if done:
            break
    
    if steps >= 500:
        print(f"   ✅ SUCCESS! Completed {steps} steps")
    else:
        print(f"   ❌ FAILED at {steps} steps")
    
    return steps


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='test',
                        choices=['train', 'evaluate', 'compare', 'test'],
                        help='Mode to run')
    parser.add_argument('--difficulty', type=str, default='easy',
                        choices=['easy', 'medium', 'hard'],
                        help='Task difficulty')
    parser.add_argument('--episodes', type=int, default=150,
                        help='Number of training episodes')
    
    args = parser.parse_args()
    
    if args.mode == 'test':
        quick_test()
    
    elif args.mode == 'train':
        train_agent(difficulty=args.difficulty, episodes=args.episodes)
    
    elif args.mode == 'evaluate':
        model_path = f"trained_models/openenv_qlearning_{args.difficulty}_best.json"
        if not os.path.exists(model_path):
            model_path = f"trained_models/openenv_qlearning_{args.difficulty}.json"
        evaluate_agent(args.difficulty, model_path, episodes=10)
    
    elif args.mode == 'compare':
        compare_all_agents(difficulty=args.difficulty)