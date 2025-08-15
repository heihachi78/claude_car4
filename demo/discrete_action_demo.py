"""
Random action demonstration.

This demo runs the car with random actions within proper ranges.
"""

import sys
import os
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.car_env import CarEnv



def main():
    print("=" * 50)
    
    env = CarEnv(render_mode="human", 
                 track_file="tracks/nascar.track", 
                 reset_on_lap=True,
                 discrete_action_space=True,
                 enable_fps_limit=False)
    
    try:
        # Reset environment first
        obs, info = env.reset()
        print("\n🚗 Running simulation with discrete random actions...")
        print("   0: Do nothing (coast)")
        print("   1: Accelerate")
        print("   2: Brake")
        print("   3: Turn left")
        print("   4: Turn right")
        total_reward = 0.0
        
        for step in range(100000):
            if env.check_quit_requested():
                print(f"   User requested quit at step {step}")
                break

            action = env.action_space.sample()

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            env.render()

            if terminated or truncated:
                print(f"   Episode terminated at step {step}, total reward: {total_reward:.2f}")
                obs, info = env.reset()
                total_reward = 0.0
        
        if env.render_mode == "human":
            env.render()
        
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        env.close()
        print("🔒 Environment closed")



if __name__ == "__main__":
    main()
