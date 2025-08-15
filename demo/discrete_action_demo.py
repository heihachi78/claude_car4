#!/usr/bin/env python3
"""Demo script showing discrete action space control of the car."""

import sys
import os
import pygame
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.car_env import CarEnv

def run_discrete_demo():
    """Run interactive demo with discrete keyboard controls."""
    
    # Create environment with discrete action space
    env = CarEnv(
        render_mode="human",
        track_file="tracks/nascar.track",
        discrete_action_space=True,
        enable_fps_limit=True
    )
    
    print("=== Discrete Action Car Control Demo ===")
    print("Controls:")
    print("  UP Arrow    - Accelerate (Action 1)")
    print("  DOWN Arrow  - Brake (Action 2)")
    print("  LEFT Arrow  - Turn Left (Action 3)")
    print("  RIGHT Arrow - Turn Right (Action 4)")
    print("  No key      - Do Nothing (Action 0)")
    print("  R           - Reset environment")
    print("  ESC         - Quit")
    print("")
    print("Action Space:", env.action_space)
    print("Number of discrete actions:", env.action_space.n)
    
    # Reset environment
    obs, info = env.reset()
    done = False
    total_reward = 0
    
    # Initial render to ensure pygame is initialized
    env.render()
    
    clock = pygame.time.Clock()
    
    while not done:
        # Check for quit
        if env.check_quit_requested():
            break
            
        # Get keyboard input and map to discrete action
        keys = pygame.key.get_pressed()
        
        # Default action is 0 (do nothing)
        action = 0
        
        # Map keys to discrete actions
        if keys[pygame.K_UP]:
            action = 1  # Accelerate
        elif keys[pygame.K_DOWN]:
            action = 2  # Brake
        elif keys[pygame.K_LEFT]:
            action = 3  # Turn left
        elif keys[pygame.K_RIGHT]:
            action = 4  # Turn right
        
        # Reset on R key
        if keys[pygame.K_r]:
            obs, info = env.reset()
            total_reward = 0
            print("Environment reset!")
            continue
            
        # Quit on ESC
        if keys[pygame.K_ESCAPE]:
            break
        
        # Take step with discrete action
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render
        env.render()
        
        # Display current state
        action_names = ["Nothing", "Accelerate", "Brake", "Turn Left", "Turn Right"]
        print(f"\rAction: {action_names[action]:12} | Speed: {info.get('car_speed_kmh', 0):6.1f} km/h | "
              f"Reward: {reward:7.3f} | Total: {total_reward:8.2f}", end="")
        
        # Check if episode ended
        done = terminated or truncated
        if done:
            print(f"\nEpisode ended! Total reward: {total_reward:.2f}")
            print("Press R to reset or ESC to quit")
            
            # Wait for reset or quit
            waiting = True
            while waiting:
                if env.check_quit_requested():
                    break
                    
                keys = pygame.key.get_pressed()
                if keys[pygame.K_r]:
                    obs, info = env.reset()
                    total_reward = 0
                    done = False
                    waiting = False
                    print("Environment reset!")
                elif keys[pygame.K_ESCAPE]:
                    break
                    
                env.render()
                clock.tick(30)
        
        # Control frame rate
        clock.tick(60)
    
    env.close()
    print("\nDemo ended.")

if __name__ == "__main__":
    run_discrete_demo()