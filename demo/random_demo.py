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
                 enable_fps_limit=False)
    
    print("Action space:", env.action_space)
    print("Low:", env.action_space.low)
    print("High:", env.action_space.high)
    
    try:
        # Reset environment first
        obs, info = env.reset()
        print("\n🚗 Running simulation with random actions...")
        print("   Throttle: [0.0, 1.0]")
        print("   Brake: [0.0, 1.0]")
        print("   Steering: [-1.0, 1.0]")
        total_reward = 0.0
        
        for step in range(100000):
            if env.check_quit_requested():
                print(f"   User requested quit at step {step}")
                break

            action = np.array(env.action_space.sample(), dtype=np.float32)

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
