# inference.py
import sys
import json
import numpy as np
from environment import SmartGridEnvironment
from agents.baseline_agent import BaselineAgent
from agents.deepseek_agent import DeepSeekAgent
from graders.agent_graders import AgentGrader
import os
from dotenv import load_dotenv

load_dotenv()

def run_inference(agent_type='baseline'):
    """Run inference with specified agent type"""
    
    print("=" * 60)
    print("OpenEnv Smart Grid - DeepSeek AI Agent")
    print("=" * 60)
    
    # Set seed for reproducibility
    np.random.seed(42)
    
    # Create grader
    grader = AgentGrader()
    results = {}
    
    for difficulty in ['easy', 'medium', 'hard']:
        print(f"\n{'='*60}")
        print(f"Evaluating on {difficulty.upper()} Task")
        print(f"{'='*60}")
        
        # Create environment
        env = SmartGridEnvironment({'difficulty': difficulty})
        
        # Create agent
        if agent_type == 'baseline':
            agent = BaselineAgent(env)
            agent_name = "Baseline Agent"
        else:
            try:
                agent = DeepSeekAgent(env)
                agent_name = "DeepSeek AI Agent"
            except Exception as e:
                print(f"⚠️ DeepSeek initialization failed: {e}")
                print("Falling back to Baseline Agent")
                agent = BaselineAgent(env)
                agent_name = "Baseline Agent (Fallback)"
        
        print(f"\n🤖 Agent: {agent_name}")
        print(f"🎯 Task: {difficulty.upper()}")
        print("-" * 40)
        
        # Run evaluation
        score, details = grader.grade(agent, difficulty)
        results[difficulty] = {'score': score, 'details': details}
        
        # Print detailed results
        print(f"✅ Score: {score:.3f}/1.0")
        print(f"📊 Grade: {grader._get_grade(score)}")
        print(f"📈 Avg Stability: {details.get('avg_stability', 0):.3f}")
        print(f"💰 Total Reward: {details.get('total_reward', 0):.2f}")
        print(f"🔄 Steps: {details.get('steps', 0)}")
        
        if 'spike_handling_rate' in details:
            print(f"⚡ Spike Handling: {details['spike_handling_rate']:.3f}")
        if 'recovery_rate' in details:
            print(f"🔄 Recovery Rate: {details['recovery_rate']:.3f}")
    
    # Calculate overall score
    overall_score = np.mean([results[d]['score'] for d in ['easy', 'medium', 'hard']])
    results['overall_score'] = overall_score
    results['grade'] = grader._get_grade(overall_score)
    
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Easy Task:   {results['easy']['score']:.3f}/1.0")
    print(f"Medium Task: {results['medium']['score']:.3f}/1.0")
    print(f"Hard Task:   {results['hard']['score']:.3f}/1.0")
    print(f"\n🎯 OVERALL SCORE: {overall_score:.3f}/1.0")
    print(f"🏆 FINAL GRADE: {results['grade']}")
    print("=" * 60)
    
    # Save results
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.float32) or isinstance(obj, np.float64):
            return float(obj)
        return str(obj)
    
    filename = f"deepseek_results.json"
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2, default=convert_to_serializable)
    
    print(f"\n✅ Results saved to {filename}")
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', type=str, default='deepseek', 
                        choices=['baseline', 'deepseek'],
                        help='Type of agent to use')
    
    args = parser.parse_args()
    
    run_inference(agent_type=args.agent)