"""
TD3 AI-powered car control function for the competition.

This module loads a trained TD3 (Twin Delayed Deep Deterministic) model
to control a single car autonomously. Falls back to rule-based control
if the TD3 model cannot be loaded.

Each car's state is managed independently by the competition framework.
"""

import numpy as np
import os

try:
    from stable_baselines3 import TD3
    TD3_AVAILABLE = True
except ImportError:
    print("Warning: stable_baselines3 not available. Using rule-based control.")
    TD3_AVAILABLE = False


# Global state for TD3 model and fallback control variables
model_state = {
    'td3_model': None,
    'model_loaded': False,
    'use_fallback': False
}

# Fallback control state (persists between calls)
# Used if TD3 model cannot be loaded
control_state = {
    'throttle': 0.0,
    'brake': 0.0,
    'steering': 0.0
}

def load_td3_model():
    """
    Load the trained TD3 model from checkpoint.
    
    Returns:
        bool: True if model loaded successfully, False otherwise
    """
    if not TD3_AVAILABLE:
        print("TD3 not available, using rule-based control")
        return False
        
    model_path = "./learn/checkpoints/model_1500000_steps.zip"
    
    # Check if running from demo/ directory, adjust path accordingly
    if not os.path.exists(model_path):
        model_path = "../learn/checkpoints/model_1500000_steps.zip"
    
    try:
        model_state['td3_model'] = TD3.load(model_path)
        model_state['model_loaded'] = True
        print(f"Successfully loaded TD3 model from {model_path}")
        return True
    except Exception as e:
        print(f"Failed to load TD3 model: {e}")
        print("Falling back to rule-based control")
        model_state['use_fallback'] = True
        return False


def car_control(observation):
    """
    Calculate control actions based on observation using trained TD3 model.
    
    This function uses a trained TD3 (Twin Delayed Deep Deterministic) model
    to predict optimal racing actions. If the model cannot be loaded,
    it falls back to rule-based control logic.
    
    Args:
        observation: numpy array of shape (29,) containing:
            - Position (x, y): indices 0-1
            - Velocity (x, y, magnitude): indices 2-4
            - Orientation and angular velocity: indices 5-6
            - Tire loads (4): indices 7-10
            - Tire temperatures (4): indices 11-14
            - Tire wear (4): indices 15-18
            - Collision data (impulse, angle): indices 19-20
            - Distance sensor readings (8 directions): indices 21-28
    
    Returns:
        numpy array of shape (3,) containing [throttle, brake, steering]
        - throttle: 0.0 to 1.0
        - brake: 0.0 to 1.0
        - steering: -1.0 to 1.0
    """
    # Try to load TD3 model on first call
    if not model_state['model_loaded'] and not model_state['use_fallback']:
        load_td3_model()
    
    # Use TD3 model if available
    if model_state['model_loaded'] and model_state['td3_model'] is not None:
        try:
            # Use TD3 model for prediction (deterministic=True for consistent racing)
            action, _ = model_state['td3_model'].predict(observation, deterministic=True)
            return action.astype(np.float32)
        except Exception as e:
            print(f"Error using TD3 model: {e}")
            print("Switching to fallback control")
            model_state['use_fallback'] = True
    
    # Fallback to rule-based control
    return _fallback_control(observation)

def _fallback_control(observation):
    """
    Rule-based fallback control logic.
    
    This is the original rule-based control algorithm used when
    the TD3 model is not available or fails.
    
    Args:
        observation: numpy array of shape (29,) containing car state
    
    Returns:
        numpy array of shape (3,) containing [throttle, brake, steering]
    """
    # Extract sensor data
    sensors = observation[21:29]  # All 8 sensor distances
    forward = sensors[0]          # Forward sensor
    front_left = sensors[1]       # Front-left sensor  
    front_right = sensors[7]      # Front-right sensor
    current_speed = observation[4]  # Speed from observation
    
    # Calculate speed limit based on forward distance (matching original logic)
    # Note: Original used car_idx for small speed variations, we use 0 here
    speed_limit = forward * 500 / 3.6
    
    # Throttle control - accumulate changes like original
    if current_speed * 200 < speed_limit:
        control_state['throttle'] += 0.1
    if current_speed * 200 > speed_limit:
        control_state['throttle'] -= 0.1
    
    # Brake control - accumulate changes like original
    if current_speed * 200 < speed_limit:
        control_state['brake'] -= 0.01
    if current_speed * 200 > speed_limit:
        control_state['brake'] += 0.01
    
    # Steering control based on sensor readings (matching original logic)
    if front_right > front_left:
        control_state['steering'] = front_right / front_left - forward
    elif front_right < front_left:
        control_state['steering'] = front_left / front_right - forward
        control_state['steering'] *= -1
    else:
        control_state['steering'] = 0
    
    # Apply limits and adjustments (matching original)
    control_state['brake'] = max(min(control_state['brake'], 1), 0)
    control_state['steering'] = max(min(control_state['steering'], 1), -1)
    
    # Reduce throttle when steering hard (matching original)
    if abs(control_state['steering']) > 0.1:
        control_state['throttle'] -= 0.05
    
    control_state['throttle'] = max(min(control_state['throttle'], 1), 0)
    
    return np.array([control_state['throttle'], control_state['brake'], control_state['steering']], dtype=np.float32)