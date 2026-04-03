# app.py - FULLY UPDATED with Working Q-Learning Agent
import gradio as gr
import numpy as np
from environment import SmartGridEnvironment
from agents.baseline_agent import BaselineAgent
from graders.agent_graders import AgentGrader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import os
import json
import random
import warnings
warnings.filterwarnings('ignore')
from dotenv import load_dotenv

load_dotenv()

grader = AgentGrader()

# ============================================================
# OPTIMAL AGENT (Heuristic) - UPDATED with Grid Draw 0.85
# ============================================================

class OptimalAgent:
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
        
        # UPDATED: Minimum grid draw 0.85
        if stability < 0.3:
            return {'battery_charge_rate': -0.8, 'solar_usage_ratio': 1.0, 'grid_draw_ratio': 0.95}
        if stability < 0.5:
            return {'battery_charge_rate': -0.5, 'solar_usage_ratio': 1.0, 'grid_draw_ratio': 0.90}
        if stability < 0.7:
            return {'battery_charge_rate': -0.3, 'solar_usage_ratio': 1.0, 'grid_draw_ratio': 0.88}
        if stability < 0.85:
            return {'battery_charge_rate': -0.1, 'solar_usage_ratio': 1.0, 'grid_draw_ratio': 0.87}
        
        if battery < 20:
            return {'battery_charge_rate': 0.5, 'solar_usage_ratio': 1.0, 'grid_draw_ratio': 0.85}
        if battery > 85:
            return {'battery_charge_rate': -0.3, 'solar_usage_ratio': 1.0, 'grid_draw_ratio': 0.85}
        if 7 <= hour <= 17 and solar > 20:
            return {'battery_charge_rate': 0.1 if battery < 70 else -0.05, 'solar_usage_ratio': 1.0, 'grid_draw_ratio': 0.85}
        
        return {'battery_charge_rate': 0, 'solar_usage_ratio': 1.0, 'grid_draw_ratio': 0.85}


# ============================================================
# Q-LEARNING AGENT (Trained) - FIXED VERSION
# ============================================================

class QLearningAgent:
    """Q-Learning Agent - Loads trained model correctly"""
    
    def __init__(self, env, model_path=None):
        self.env = env
        self.q_table = {}
        self.epsilon = 0
        
        # Grid options (must match training)
        self.grid_options = [0.85, 0.88, 0.90, 0.92, 0.95, 0.98, 1.0]
        
        # Try multiple possible model paths
        if model_path is None:
            difficulty = env.task_difficulty if hasattr(env, 'task_difficulty') else 'easy'
            possible_paths = [
                f"trained_models/openenv_qlearning_{difficulty}_best.json",
                f"trained_models/openenv_qlearning_{difficulty}.json",
                f"trained_models/qlearning_{difficulty}.json"
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    model_path = path
                    break
        
        if model_path and os.path.exists(model_path):
            self.load(model_path)
            print(f"✅ Loaded trained Q-Learning agent from {model_path}")
            print(f"📊 Q-Table size: {len(self.q_table)} states")
        else:
            print(f"⚠️ No trained model found at: {model_path}")
    
    def _get_state_key(self, observation):
        """Convert observation to discrete state - MUST MATCH TRAINING"""
        if hasattr(observation, 'get'):
            stability = int(observation.get('grid_stability', 1.0) * 10)
            battery = int(observation.get('battery_level', 50) / 10)
        else:
            stability = int(getattr(observation, 'grid_stability', 1.0) * 10)
            battery = int(getattr(observation, 'battery_level', 50) / 10)
        return f"{stability}_{battery}"
    
    def _action_from_key(self, action_key):
        """Convert discrete action back to continuous - MUST MATCH TRAINING"""
        try:
            parts = action_key.split('_')
            battery_rate = (int(parts[0]) / 2) - 1
            grid_idx = min(int(parts[1]), len(self.grid_options) - 1)
            grid_draw = self.grid_options[grid_idx]
            
            return {
                'battery_charge_rate': np.clip(battery_rate, -1, 1),
                'solar_usage_ratio': 1.0,
                'grid_draw_ratio': grid_draw
            }
        except:
            # Default action if parsing fails
            return {'battery_charge_rate': 0, 'solar_usage_ratio': 1.0, 'grid_draw_ratio': 0.85}
    
    def get_action(self, observation):
        """Get best action from trained Q-table"""
        state_key = self._get_state_key(observation)
        
        if state_key in self.q_table and self.q_table[state_key]:
            best_action = max(self.q_table[state_key], key=self.q_table[state_key].get)
            return self._action_from_key(best_action)
        
        # Default safe action if state not seen
        return {'battery_charge_rate': 0, 'solar_usage_ratio': 1.0, 'grid_draw_ratio': 0.85}
    
    def load(self, path):
        """Load trained Q-table"""
        try:
            with open(path, 'r') as f:
                self.q_table = json.load(f)
        except Exception as e:
            print(f"Error loading model: {e}")
            self.q_table = {}
    
    def is_trained(self):
        return len(self.q_table) > 0


# ============================================================
# EVALUATION FUNCTIONS
# ============================================================

def run_optimal_agent(difficulty):
    """Run optimal agent evaluation"""
    env = SmartGridEnvironment({'difficulty': difficulty})
    agent = OptimalAgent(env)
    return _run_evaluation(env, agent, difficulty, "Optimal Agent (Heuristic)")


def run_qlearning_agent(difficulty):
    """Run Q-Learning agent evaluation"""
    env = SmartGridEnvironment({'difficulty': difficulty})
    agent = QLearningAgent(env)
    
    if not agent.is_trained():
        return _get_no_model_message(difficulty, "Q-Learning")
    
    return _run_evaluation(env, agent, difficulty, "Q-Learning Agent (Trained)")


def run_baseline_agent(difficulty):
    """Run baseline agent evaluation"""
    env = SmartGridEnvironment({'difficulty': difficulty})
    agent = BaselineAgent(env)
    return _run_evaluation(env, agent, difficulty, "Baseline Agent")


def _run_evaluation(env, agent, difficulty, agent_name):
    """Common evaluation function"""
    
    score, details = grader.grade(agent, difficulty)
    
    # Collect data for visualization
    stability_history = []
    battery_history = []
    reward_history = []
    action_history = {'battery': [], 'solar': [], 'grid': []}
    
    obs = env.reset()
    max_steps = {'easy': 500, 'medium': 1000, 'hard': 1500}[difficulty]
    
    for step in range(max_steps):
        action = agent.get_action(obs)
        obs, reward, done, info = env.step(action)
        stability_history.append(float(info['grid_stability']))
        battery_history.append(float(info['battery_level']))
        reward_history.append(float(reward))
        action_history['battery'].append(float(action['battery_charge_rate']))
        action_history['solar'].append(float(action['solar_usage_ratio']))
        action_history['grid'].append(float(action['grid_draw_ratio']))
        if done:
            break
    
    steps_completed = len(stability_history)
    avg_stability = np.mean(stability_history)
    
    # Create visualization
    fig = plt.figure(figsize=(14, 8), facecolor='white')
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # Plot 1: Grid Stability
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(stability_history, '#1a73e8', linewidth=2)
    ax1.fill_between(range(len(stability_history)), stability_history, 0.7, 
                      where=np.array(stability_history)>=0.7, color='#34a853', alpha=0.3)
    ax1.fill_between(range(len(stability_history)), stability_history, 0.7,
                      where=np.array(stability_history)<0.7, color='#ea4335', alpha=0.3)
    ax1.axhline(y=0.9, color='#1e8e3e', linestyle='--', alpha=0.8, label='Excellent')
    ax1.axhline(y=0.7, color='#f9ab00', linestyle='--', alpha=0.8, label='Good')
    ax1.axhline(y=0.5, color='#ea4335', linestyle='--', alpha=0.8, label='Poor')
    ax1.set_title('Grid Stability', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Stability')
    ax1.legend(loc='lower right', fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1.1])
    
    # Plot 2: Battery Level
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(battery_history, '#e8710a', linewidth=2)
    ax2.axhline(y=20, color='#ea4335', linestyle='--', alpha=0.8, label='Critical')
    ax2.axhline(y=50, color='#34a853', linestyle='--', alpha=0.8, label='Healthy')
    ax2.set_title('Battery Level', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Battery (%)')
    ax2.legend(loc='lower right', fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 105])
    
    # Plot 3: Rewards
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(reward_history, '#9334e6', linewidth=2)
    ax3.axhline(y=0, color='#ea4335', linestyle='--', alpha=0.5)
    ax3.set_title('Rewards', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Reward')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Actions
    ax4 = fig.add_subplot(gs[1, 0])
    if len(action_history['battery']) > 0:
        x = ['Battery', 'Solar', 'Grid']
        y = [np.mean(action_history['battery']), np.mean(action_history['solar']), np.mean(action_history['grid'])]
        colors_bar = ['#1a73e8', '#34a853', '#ea4335']
        bars = ax4.bar(x, y, color=colors_bar, edgecolor='black', linewidth=1)
        for bar, val in zip(bars, y):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
                    f'{val:.2f}', ha='center', va='bottom', fontweight='bold')
    ax4.set_title('Average Actions', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Value')
    ax4.set_ylim([-1, 1])
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Plot 5: Score Card
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.axis('off')
    
    grade = grader._get_grade(score)
    if score >= 0.7:
        grade_color = '#34a853'
        grade_emoji = '✅'
    elif score >= 0.5:
        grade_color = '#f9ab00'
        grade_emoji = '📈'
    else:
        grade_color = '#ea4335'
        grade_emoji = '⚠️'
    
    score_text = f"""
    ╔════════════════════════════════════╗
    ║       PERFORMANCE SCORE            ║
    ╠════════════════════════════════════╣
    ║  Score: {score:.3f} / 1.0            ║
    ║  Grade: {grade_emoji} {grade}                         ║
    ╠════════════════════════════════════╣
    ║        METRICS                     ║
    ╠════════════════════════════════════╣
    ║  Stability: {avg_stability:.3f}                 ║
    ║  Steps: {steps_completed}/{max_steps}                   ║
    ║  Reward: {details.get('total_reward', 0):.2f}          ║
    ╚════════════════════════════════════╝
    """
    
    ax5.text(0.5, 0.5, score_text, transform=ax5.transAxes,
             fontsize=10, verticalalignment='center', horizontalalignment='center',
             fontfamily='monospace', fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='white', edgecolor='#1a73e8', linewidth=2))
    
    # Plot 6: Agent Info
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    
    is_trained = "YES (Trained)" if "Trained" in agent_name else "NO (Rule-based)"
    
    info_text = f"""
    ╔════════════════════════════════════╗
    ║        AGENT INFO                  ║
    ╠════════════════════════════════════╣
    ║  Name: {agent_name[:20]}       ║
    ║  Trained: {is_trained}            ║
    ╠════════════════════════════════════╣
    ║      KEY INSIGHT                   ║
    ╠════════════════════════════════════╣
    ║  Maintain grid draw >= 0.85        ║
    ║  for stability maintenance!        ║
    ╚════════════════════════════════════╝
    """
    
    ax6.text(0.5, 0.5, info_text, transform=ax6.transAxes,
             fontsize=9, verticalalignment='center', horizontalalignment='center',
             fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#e8f0fe', edgecolor='#34a853', linewidth=2))
    
    plt.suptitle(f'{agent_name} - {difficulty.upper()} Task', 
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close('all')
    
    # HTML Results
    if steps_completed >= max_steps - 10:
        completion_status = "COMPLETE"
        completion_color = "#34a853"
    else:
        completion_status = "EARLY STOP"
        completion_color = "#ea4335"
    
    results_html = f"""
    <div style="font-family: 'Segoe UI', Arial, sans-serif; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px;">
        <div style="background: white; border-radius: 15px; padding: 20px; text-align: center; margin-bottom: 20px;">
            <h2 style="color: #1a73e8; margin: 0;">🤖 {agent_name}</h2>
            <h3 style="color: #5f6368; margin: 10px 0 0 0;">Task: {difficulty.upper()}</h3>
        </div>
        
        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
            <div style="background: white; border-radius: 15px; padding: 15px; text-align: center;">
                <h3 style="color: #1a73e8; margin: 0;">Score</h3>
                <p style="font-size: 36px; font-weight: bold; color: #1a73e8; margin: 10px 0;">{score:.3f}<span style="font-size: 18px;">/1.0</span></p>
                <p style="font-size: 24px; font-weight: bold; color: {grade_color};">Grade: {grade_emoji} {grade}</p>
            </div>
            
            <div style="background: white; border-radius: 15px; padding: 15px; text-align: center;">
                <h3 style="color: #1a73e8; margin: 0;">Stability</h3>
                <p style="font-size: 36px; font-weight: bold; color: #34a853; margin: 10px 0;">{avg_stability:.3f}</p>
                <p>Trained: {is_trained}</p>
            </div>
            
            <div style="background: white; border-radius: 15px; padding: 15px; text-align: center;">
                <h3 style="color: #1a73e8; margin: 0;">Steps</h3>
                <p style="font-size: 36px; font-weight: bold; color: {completion_color}; margin: 10px 0;">{steps_completed}<span style="font-size: 18px;">/{max_steps}</span></p>
                <p>Status: {completion_status}</p>
            </div>
            
            <div style="background: white; border-radius: 15px; padding: 15px; text-align: center;">
                <h3 style="color: #1a73e8; margin: 0;">Reward</h3>
                <p style="font-size: 36px; font-weight: bold; color: #ea4335; margin: 10px 0;">{details.get('total_reward', 0):.2f}</p>
                <p>Total accumulated</p>
            </div>
        </div>
    </div>
    """
    
    return f"<img src='data:image/png;base64,{img_base64}' style='width:100%; border-radius:15px; margin-bottom:20px;'/>", results_html


def _get_no_model_message(difficulty, agent_type):
    """Show message when trained model not found"""
    return f"""
    <div style="text-align: center; padding: 40px; background: #fef7e0; border-radius: 15px;">
        <h2 style="color: #e37400;">⚠️ Trained Model Not Found</h2>
        <p>The {agent_type} agent needs to be trained first!</p>
        <p>Run the following command in terminal:</p>
        <code style="background: #f1f3f4; padding: 10px; display: inline-block; border-radius: 8px;">
        python train_openenv_agent.py --mode train --difficulty {difficulty}
        </code>
        <p style="margin-top: 20px;">After training, come back and run this agent again!</p>
    </div>
    """, ""


# ============================================================
# GRADIO INTERFACE
# ============================================================

with gr.Blocks(title="OpenEnv Smart Grid", theme=gr.themes.Soft()) as demo:
    
    gr.HTML("""
    <div style="text-align: center; padding: 25px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; margin-bottom: 25px;">
        <h1 style="color: white; margin: 0;">🌱 OpenEnv Smart Grid</h1>
        <p style="color: white; margin: 10px 0 0 0;">AI Agent for Energy Management | Hackathon Assessment</p>
    </div>
    """)
    
    with gr.Tabs():
        
        # Tab 1: Optimal Agent
       # --- TAB 1: OPTIMAL AGENT (HEURISTIC) ---
        with gr.TabItem("Optimal Agent (Heuristic)"):
            with gr.Row():
                with gr.Column(scale=1):
                    difficulty_optimal = gr.Radio(
                        choices=['easy', 'medium', 'hard'],
                        label="Task Difficulty",
                        value='easy'
                    )
                    run_optimal_btn = gr.Button("Run Optimal Agent", variant="primary", size="lg")
                with gr.Column(scale=1):
                    # Professional Expert-System themed About Card
                    gr.HTML("""
                    <div style="
                        background: #ffffff; 
                        padding: 20px; 
                        border-radius: 12px; 
                        border-left: 6px solid #1a73e8; 
                        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
                        color: #2c3e50;
                        font-family: 'Segoe UI', sans-serif;
                    ">
                        <h4 style="margin: 0 0 10px 0; color: #1a73e8; font-weight: 700; display: flex; align-items: center;">
                            <span style="margin-right: 10px;">🛡️</span> About Optimal Agent
                        </h4>
                        <p style="margin: 0; line-height: 1.6; font-size: 14px; color: #34495e;">
                            Uses expert-tuned heuristics for maximum stability.<br>
                            <span style="display: block; margin-top: 5px; color: #7f8c8d; font-size: 13px;">
                                Not trained via ML—instead uses manually optimized rules
                                to maintain grid draw ≥ 0.85 and handle critical stability dips.
                            </span>
                        </p>
                    </div>
                    """)
            
            with gr.Row():
                optimal_plot = gr.HTML()
                optimal_results = gr.HTML()
            
            run_optimal_btn.click(
                fn=run_optimal_agent,
                inputs=[difficulty_optimal],
                outputs=[optimal_plot, optimal_results]
            )
        
        # Tab 2: Q-Learning Agent (Trained)
        # --- TAB 2: Q-LEARNING AGENT (TRAINED) ---
        with gr.TabItem("Q-Learning Agent (Trained)"):
            with gr.Row():
                with gr.Column(scale=1):
                    difficulty_q = gr.Radio(
                        choices=['easy', 'medium', 'hard'],
                        label="Task Difficulty",
                        value='easy'
                    )
                    run_q_btn = gr.Button("Run Q-Learning Agent", variant="primary", size="lg")
                with gr.Column(scale=1):
                    # Professional AI/Machine Learning themed About Card
                    gr.HTML("""
                    <div style="
                        background: #ffffff; 
                        padding: 20px; 
                        border-radius: 12px; 
                        border-left: 6px solid #27ae60; 
                        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
                        color: #2c3e50;
                        font-family: 'Segoe UI', sans-serif;
                    ">
                        <h4 style="margin: 0 0 10px 0; color: #219150; font-weight: 700; display: flex; align-items: center;">
                            <span style="margin-right: 10px;">🧠</span> About Q-Learning
                        </h4>
                        <p style="margin: 0; line-height: 1.6; font-size: 14px; color: #34495e;">
                            Advanced Reinforcement Learning agent.<br>
                            <span style="display: block; margin-top: 5px; color: #7f8c8d; font-size: 13px;">
                                Learns optimal strategies through trial and error.
                                Automatically improves energy distribution based on grid stability rewards.
                            </span>
                        </p>
                    </div>
                    """)
            
            with gr.Row():
                q_plot = gr.HTML()
                q_results = gr.HTML()
            
            run_q_btn.click(
                fn=run_qlearning_agent,
                inputs=[difficulty_q],
                outputs=[q_plot, q_results]
            )
        
        # Tab 3: Baseline Agent
        # --- TAB 3: BASELINE AGENT ---
        with gr.TabItem("Baseline Agent"):
            with gr.Row():
                with gr.Column(scale=1):
                    difficulty_baseline = gr.Radio(
                        choices=['easy', 'medium', 'hard'],
                        label="Task Difficulty",
                        value='easy'
                    )
                    run_baseline_btn = gr.Button("Run Baseline Agent", variant="secondary", size="lg")
                with gr.Column(scale=1):
                    # Refined UI for the About Section
                    gr.HTML("""
                    <div style="
                        background: #ffffff; 
                        padding: 20px; 
                        border-radius: 12px; 
                        border-left: 6px solid #f39c12; 
                        box-shadow: 0 4px 10px rgba(0,0,0,0.05);
                        color: #2c3e50;
                        font-family: 'Segoe UI', sans-serif;
                    ">
                        <h4 style="margin: 0 0 8px 0; color: #d35400; font-weight: 700;">
                            ℹ️ About Baseline
                        </h4>
                        <p style="margin: 0; line-height: 1.5; font-size: 14px; color: #34495e;">
                            Simple rule-based baseline.<br>
                            <span style="opacity: 1.0; color: #34495e;">Not trained — uses basic if-else logic for performance benchmarking.</span>
                        </p>
                    </div>
                    """)
            
            with gr.Row():
                baseline_plot = gr.HTML()
                baseline_results = gr.HTML()
            
            run_baseline_btn.click(
                fn=run_baseline_agent,
                inputs=[difficulty_baseline],
                outputs=[baseline_plot, baseline_results]
            )
        
        # Tab 4: Training Guide
        with gr.TabItem("Training Guide"):
            gr.Markdown("""
            ## How to Train Q-Learning Agent
            
            ### Step 1: Train the agent
            Open terminal and run:
            
            ```bash
            # Train on Easy task
            python train_openenv_agent.py --mode train --difficulty easy --episodes 150
            
            # Train on Medium task  
            python train_openenv_agent.py --mode train --difficulty medium --episodes 150
            
            # Train on Hard task
            python train_openenv_agent.py --mode train --difficulty hard --episodes 150
                        """)

if __name__ == "__main__":
 port = int(os.environ.get("PORT", 7860))
demo.launch(server_name="127.0.0.1", server_port=port)

