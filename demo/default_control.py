"""
Default car control function for the competition.

This module contains the control logic for a single car.
Each car's state is managed independently by the competition framework.
"""

import numpy as np


# Global state for control variables (persists between calls)
# In competition, each car would need its own state
control_state = {
    'throttle': 0.0,
    'brake': 0.0,
    'steering': 0.0
}


def car_control(observation):
    """
    Calculate control actions based on observation for a single car.
    
    This function maintains state between calls to accumulate
    throttle/brake values like in the original car_demo.py.
    
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