# Car Racing Simulation & Reinforcement Learning Environment

A realistic car racing simulation and reinforcement learning environment built with Box2D physics, Gymnasium API, and TD3 deep reinforcement learning. Features multi-car racing, realistic car dynamics, tire physics, and comprehensive RL training capabilities.

## 🏎️ Features

### Core Simulation
- **Realistic Physics**: Box2D-based car dynamics with engine torque curves, tire heating, weight transfer, and aerodynamics
- **Multi-Car Racing**: Support for 1-10 cars simultaneously with collision handling and camera switching
- **Track System**: Custom track loading from `.track` files with collision boundaries and racing lines
- **Visual Rendering**: Pygame-based visualization with debug overlays and dynamic camera modes

### Reinforcement Learning
- **Gymnasium Interface**: Standard RL environment with 29-dimensional normalized observation space
- **TD3 Training**: Deep deterministic policy gradient learning with experience replay
- **Comprehensive Rewards**: Distance-based, speed-based, collision penalties, and lap completion bonuses
- **Multi-Car RL**: Independent agents or single agent controlling multiple cars

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Environment Details](#environment-details)
  - [Observation Space](#observation-space)
  - [Action Space](#action-space)
  - [Reward Structure](#reward-structure)
- [Multi-Car Environment](#multi-car-environment)
- [Competition Framework](#competition-framework)
  - [Competition Overview](#competition-overview)
  - [Developing Control Functions](#developing-control-functions)
  - [Competition Usage](#competition-usage)
- [Training with TD3](#training-with-td3)
- [Physics Systems](#physics-systems)
- [Usage Examples](#usage-examples)
- [Configuration](#configuration)
- [Performance](#performance)

## 🚀 Installation

### Prerequisites
- Python 3.8+
- Box2D physics engine
- Pygame for rendering
- NumPy for numerical operations

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Core Dependencies
```
gymnasium==0.29.1      # RL environment interface
box2d-py==2.3.8        # Physics simulation
pygame==2.5.2          # Rendering and visualization
numpy==1.26.3          # Numerical operations
pytest==7.4.4          # Testing framework
stable-baselines3      # For RL training (optional)
```

## 🎮 Quick Start

### Basic Car Demo
```bash
python demo/car_demo.py
```

### Random Action Demo
```bash
python demo/random_demo.py
```

### Competition Demo (External Control)
```bash
python demo/competition.py
```

### Train RL Agent
```bash
python learn/learn.py
```

### Controls (During Rendering)
- **0-9 Keys**: Switch between cars (multi-car mode)
- **R**: Toggle reward display
- **D**: Toggle debug physics display
- **I**: Toggle track info display  
- **C**: Toggle camera mode (track view / car follow)
- **F**: Toggle fullscreen
- **ESC**: Exit

## 🔬 Environment Details

### Observation Space (29 Dimensions)

The environment provides a comprehensive 29-dimensional observation vector, all values normalized to [-1,1] or [0,1]:

| Index | Component | Range | Description |
|-------|-----------|-------|-------------|
| 0-1 | Position (x, y) | [-1, 1] | Car position in world coordinates |
| 2-3 | Velocity (x, y) | [-1, 1] | Car velocity components |
| 4 | Speed Magnitude | [0, 1] | Total speed magnitude |
| 5 | Orientation | [-1, 1] | Car heading angle (normalized from [-π, π]) |
| 6 | Angular Velocity | [-1, 1] | Car rotation rate |
| 7-10 | Tire Loads | [0, 1] | Normal forces on each tire (FL, FR, RL, RR) |
| 11-14 | Tire Temperatures | [0, 1] | Heat from friction and slip |
| 15-18 | Tire Wear | [0, 1] | Cumulative tire degradation |
| 19 | Collision Impulse | [0, 1] | Magnitude of recent collision |
| 20 | Collision Angle | [-1, 1] | Direction of collision relative to car |
| 21-28 | Distance Sensors | [0, 1] | 8-direction obstacle detection |

#### Normalization Constants
```python
NORM_MAX_POSITION = 1000.0        # ±1000m world bounds
NORM_MAX_VELOCITY = 50.0          # Up to 180 km/h (50 m/s)
NORM_MAX_ANGULAR_VEL = 10.0       # Rotation speed limit
NORM_MAX_TYRE_LOAD = 8000.0       # Maximum tire force
NORM_MAX_TYRE_TEMP = 120.0        # Temperature in Celsius
NORM_MAX_TYRE_WEAR = 100.0        # Wear percentage
NORM_MAX_COLLISION_IMPULSE = 1000.0 # Collision severity
```

### Action Space

#### Continuous Mode (Default)
3-dimensional continuous control:
```python
action_space = Box(low=[0.0, 0.0, -1.0], high=[1.0, 1.0, 1.0])
```
- **action[0]**: Throttle [0.0, 1.0] - Engine power output
- **action[1]**: Brake [0.0, 1.0] - Braking force  
- **action[2]**: Steering [-1.0, 1.0] - Steering wheel angle

#### Discrete Mode
5 discrete actions:
```python
action_space = Discrete(5)
```
- **0**: Do nothing (coast)
- **1**: Accelerate (throttle=1.0)
- **2**: Brake (brake=1.0)
- **3**: Turn left (steering=-1.0)
- **4**: Turn right (steering=1.0)

### Reward Structure

The environment uses a comprehensive reward function that encourages fast, smooth driving with realistic penalties:

#### Speed Rewards (Time-Based)
```python
# Base speed reward - encourages forward motion (quadratic for aggressive acceleration)
speed_reward = speed**2 * 0.004 * dt  # REWARD_SPEED_MULTIPLIER = 0.004

# High speed bonus for speeds > 83 m/s (~300 km/h)
if speed > 83.0:  # REWARD_HIGH_SPEED_THRESHOLD
    speed_reward += 1.0 * dt  # REWARD_HIGH_SPEED_BONUS
```

#### Distance Rewards
```python
# Reward for distance traveled (+0.1 per meter)
distance_reward = distance_traveled * 0.1  # REWARD_DISTANCE_MULTIPLIER
```

#### Forward Sensor Reward (Time-Based)
```python
# Reward for having clear space ahead (encourages good positioning)
forward_sensor_reward = forward_sensor_distance * 5.0 * dt  # REWARD_FORWARD_SENSOR_MULTIPLIER
```

#### Penalties
```python
# Low speed penalty for speeds < 0.278 m/s (~1 km/h)
if speed < 0.278:  # PENALTY_LOW_SPEED_THRESHOLD
    penalty = -0.001 * dt  # PENALTY_LOW_SPEED_RATE per second

# Collision penalties (continuous, applied per timestep while colliding)
collision_penalties = {
    "minor": -25,     # < 500 impulse (COLLISION_SEVERITY_MINOR)
    "moderate": -50,   # 500-1000 impulse (COLLISION_SEVERITY_MODERATE)  
    "severe": -125,    # 1000-2000 impulse (COLLISION_SEVERITY_SEVERE)
    "critical": -250,  # 2000-5000 impulse (COLLISION_SEVERITY_EXTREME)
    "extreme": -1000   # > 5000 impulse (immediate disabling)
}

# Collision penalties are applied continuously during collisions:
# - Per-second rate for minor/moderate/severe collisions
# - Full penalty immediately for extreme collisions
```

#### Lap Completion Bonuses
```python
# Major bonus for completing laps
lap_bonus = 50.0 * laps_completed  # REWARD_LAP_COMPLETION

# Additional bonus for fast laps (< 40 seconds)
if lap_time < 40.0:  # REWARD_FAST_LAP_TIME
    fast_lap_bonus = 50.0  # REWARD_FAST_LAP_BONUS
```

#### Reward Function Summary

The complete reward calculation per timestep:
```python
reward = (
    speed**2 * 0.004 * dt +                    # Quadratic speed reward
    distance_traveled * 0.1 +                  # Distance progress
    forward_sensor_distance * 5.0 * dt +       # Clear path ahead
    (1.0 * dt if speed > 83.0 else 0) +       # High speed bonus
    (-0.001 * dt if speed < 0.278 else 0) +   # Low speed penalty
    lap_completion_bonus +                     # Lap bonuses
    collision_penalty                          # Collision penalties
)
```

## 🏁 Multi-Car Environment

The environment supports 1-10 cars racing simultaneously with independent physics and collision handling.

### Single Car Mode (Backward Compatible)
```python
env = CarEnv(
    render_mode="human",
    track_file="tracks/nascar.track",
    num_cars=1
)

observation, info = env.reset()
action = [throttle, brake, steering]  # Single action
obs, reward, terminated, truncated, info = env.step(action)
```

### Multi-Car Mode
```python
env = CarEnv(
    render_mode="human", 
    track_file="tracks/nascar.track",
    num_cars=3,
    car_names=["Lightning", "Thunder", "Blaze"]
)

observations, info = env.reset()  # Shape: (3, 29)
actions = [[throttle, brake, steering] for _ in range(3)]  # Shape: (3, 3)
obs, rewards, terminated, truncated, infos = env.step(actions)
```

### Key Differences: Single vs Multi-Car

#### Action and Observation Spaces

**Single Car Mode:**
```python
# Action: single 3D array
action = [throttle, brake, steering]  # Shape: (3,)

# Observation: single 29D array  
observation = np.array([...])  # Shape: (29,)

# Step returns
obs, reward, terminated, truncated, info = env.step(action)
```

**Multi-Car Mode:**
```python
# Actions: array of actions for each car
actions = [[throttle, brake, steering], ...]  # Shape: (num_cars, 3)

# Observations: array of observations for each car
observations = np.array([...])  # Shape: (num_cars, 29)

# Step returns arrays for each car
obs, rewards, terminated, truncated, infos = env.step(actions)
```

#### Termination Conditions

**Single Car Mode:**
- **Collision Termination**: Severe/critical collisions terminate episode immediately
- **Low Reward**: Cumulative reward below threshold terminates (training mode only)
- **Stuck Detection**: Low speed for extended time terminates episode
- **Time Limits**: Episodes terminate after max time (training mode)

**Multi-Car Mode:**
- **No Collision Termination**: Individual cars are disabled, but episode continues
- **Car Disabling**: Cars disabled after severe collisions or sustained damage
- **All Cars Disabled**: Episode terminates only when ALL cars are disabled
- **Followed Car**: Low reward/stuck detection only applies to currently followed car
- **Time Limits**: Same as single car mode

#### Car Disabling System (Multi-Car Only)

Cars are permanently disabled during an episode under these conditions:

```python
# Immediate disabling for extreme collisions
if collision_impulse > 5000:  # COLLISION_SEVERITY_EXTREME
    disable_car_immediately()

# Sustained severe collision disabling  
if collision_impulse > 2000 and duration > 0.8_seconds:  # COLLISION_SEVERITY_SEVERE
    disable_car_after_sustained_damage()
```

**Disabled Car Behavior:**
- Receives zero actions (no throttle, brake, or steering)
- Still provides observations (for consistency)
- Marked as `"disabled": True` in info dictionary
- Cannot be re-enabled during episode

#### Reward Calculation

**Single Car:** Single reward value based on followed car performance
**Multi-Car:** Array of rewards, one per car (disabled cars still receive rewards)

#### Info Dictionaries

**Single Car Info:**
```python
{
    "simulation_time": float,
    "episode_stats": dict,
    "termination_reason": str,
    "cumulative_reward": float,
    "num_cars": 1,
    "followed_car_index": 0,
    # ... car-specific data
}
```

**Multi-Car Info (per car):**
```python
{
    "simulation_time": float,
    "car_index": int,
    "num_cars": int,
    "followed_car_index": int,
    "termination_reason": str,
    "disabled": bool,  # Unique to multi-car
    # ... car-specific data
}
```

### Multi-Car Features
- **Individual Collision Handling**: Cars disabled instead of episode termination
- **Camera Switching**: Press 0-9 keys to follow different cars
- **Independent Rewards**: Each car receives its own reward signal
- **Independent Physics**: Each car has separate collision and physics state
- **Race Positions**: Real-time position tracking based on track progress
- **Lap Time Leaderboards**: Best lap times tracked per car
- **Selective Termination**: Episode continues until all cars disabled or time limit

### Camera and Following
- Environment tracks a "followed car" for camera and primary RL agent
- Switch followed car with keyboard (0-9 keys) or programmatically
- Legacy single-car references maintained for backward compatibility
- Termination conditions apply primarily to followed car in multi-car mode

### Environment Configuration Examples

#### Single Car Training Setup
```python
env = CarEnv(
    render_mode=None,              # Headless for speed
    track_file="tracks/nascar.track",
    num_cars=1,                    # Single car
    reset_on_lap=True,             # Auto-reset for training
    discrete_action_space=False    # Continuous actions
)
```

#### Multi-Car Competition Setup
```python
env = CarEnv(
    render_mode="human",           # Visual feedback
    track_file="tracks/nascar.track", 
    num_cars=5,                    # Up to 10 supported
    car_names=["AI-1", "AI-2", "AI-3", "AI-4", "AI-5"],
    reset_on_lap=False,            # Manual reset control
    enable_fps_limit=True          # Stable visualization
)
```

#### Single Car Demo/Testing
```python
env = CarEnv(
    render_mode="human",
    track_file="tracks/nascar.track",
    num_cars=1,
    reset_on_lap=False,            # No auto-reset
    enable_fps_limit=True
)
```

### Training Considerations

**Single Car Mode:**
- Ideal for initial RL training and algorithm development
- Faster execution (less computation per step)
- Simpler observation/action handling
- Direct reward signal without interference

**Multi-Car Mode:**
- Better for competition and comparative evaluation
- More realistic racing scenarios with traffic
- Complex multi-agent interactions
- Each car learns independently or can use shared policies

## 🏆 Competition Framework

The competition framework provides a structured way to develop and test custom car control algorithms. It supports both single-car and multi-car competitions with comprehensive performance tracking and real-time visualization.

### Competition Overview

The competition system consists of:
- **External Control Functions**: Python functions that take observations and return actions
- **Competition Runner**: Framework that manages multiple cars and tracks performance
- **Performance Metrics**: Lap times, rewards, collision tracking, and leaderboards
- **Multi-Car Support**: Up to 10 cars competing simultaneously

### Key Features
- **Real-time Performance Tracking**: Live lap times, best lap tracking, and gap analysis
- **Collision Monitoring**: Detailed collision statistics and severity tracking
- **Reward Analysis**: Per-car reward accumulation and lap-based reward tracking
- **Camera Switching**: Dynamic camera following with keyboard controls (0-9)
- **Leaderboards**: Overall best lap tracking across all competitors

### Developing Control Functions

#### Control Function Structure

All control functions must follow this signature:
```python
def car_control(observation):
    """
    Calculate control actions based on observation.
    
    Args:
        observation: numpy array of shape (29,) containing normalized sensor data
        
    Returns:
        numpy array of shape (3,) containing [throttle, brake, steering]
        - throttle: 0.0 to 1.0 (engine power)
        - brake: 0.0 to 1.0 (braking force)  
        - steering: -1.0 to 1.0 (left/right steering)
    """
    # Your control logic here
    return np.array([throttle, brake, steering], dtype=np.float32)
```

#### Observation Space for Control Functions

The observation array contains 29 normalized values:

```python
# Position and motion (indices 0-6)
pos_x = observation[0]           # Car X position [-1, 1]
pos_y = observation[1]           # Car Y position [-1, 1]  
vel_x = observation[2]           # Velocity X component [-1, 1]
vel_y = observation[3]           # Velocity Y component [-1, 1]
speed = observation[4]           # Speed magnitude [0, 1]
heading = observation[5]         # Car heading angle [-1, 1]
angular_vel = observation[6]     # Angular velocity [-1, 1]

# Tire physics (indices 7-18)
tire_loads = observation[7:11]   # Normal forces on tires [0, 1]
tire_temps = observation[11:15]  # Tire temperatures [0, 1]
tire_wear = observation[15:19]   # Tire wear levels [0, 1]

# Collision data (indices 19-20)
collision_impulse = observation[19]  # Recent collision magnitude [0, 1]
collision_angle = observation[20]    # Collision direction [-1, 1]

# Distance sensors (indices 21-28) - CRITICAL FOR NAVIGATION
sensors = observation[21:29]     # 8-direction distance sensors [0, 1]
forward = sensors[0]             # Forward (0°)
front_left = sensors[1]          # Front-left (45°)
left = sensors[2]                # Left (90°)
rear_left = sensors[3]           # Rear-left (135°)
rear = sensors[4]                # Rear (180°)
rear_right = sensors[5]          # Rear-right (225°)
right = sensors[6]               # Right (270°)
front_right = sensors[7]         # Front-right (315°)
```

#### Control Function Best Practices

1. **Use Distance Sensors**: The 8-direction sensors (indices 21-28) are essential for navigation
2. **Maintain State**: Use global variables to maintain control state between function calls
3. **Smooth Control**: Gradually adjust throttle/brake/steering for stable driving
4. **Speed Management**: Base speed on forward sensor distance to avoid collisions
5. **Steering Logic**: Use front-left and front-right sensors for cornering decisions

#### Example Control Function (Sensor-Based)

```python
import numpy as np

# Global state for smooth control transitions
control_state = {
    'throttle': 0.0,
    'brake': 0.0, 
    'steering': 0.0
}

def car_control(observation):
    """Example sensor-based control function."""
    # Extract critical sensor data
    sensors = observation[21:29]
    forward = sensors[0]        # Forward distance sensor
    front_left = sensors[1]     # Front-left sensor
    front_right = sensors[7]    # Front-right sensor  
    current_speed = observation[4]  # Current speed
    
    # Calculate dynamic speed limit based on forward clearance
    speed_limit = forward * 500 / 3.6  # Convert to appropriate units
    current_speed_scaled = current_speed * 200  # Scale for comparison
    
    # Smooth throttle control
    if current_speed_scaled < speed_limit:
        control_state['throttle'] += 0.1  # Accelerate
    else:
        control_state['throttle'] -= 0.1  # Reduce throttle
    
    # Smooth brake control  
    if current_speed_scaled > speed_limit:
        control_state['brake'] += 0.01   # Apply brakes
    else:
        control_state['brake'] -= 0.01   # Release brakes
    
    # Steering based on sensor differential
    if front_right > front_left:
        # More space on right, steer right
        control_state['steering'] = (front_right / front_left) - forward
    elif front_left > front_right:
        # More space on left, steer left  
        control_state['steering'] = -((front_left / front_right) - forward)
    else:
        # Equal space, go straight
        control_state['steering'] = 0.0
    
    # Apply control limits
    control_state['throttle'] = np.clip(control_state['throttle'], 0.0, 1.0)
    control_state['brake'] = np.clip(control_state['brake'], 0.0, 1.0)
    control_state['steering'] = np.clip(control_state['steering'], -1.0, 1.0)
    
    # Reduce throttle when steering hard (racing line optimization)
    if abs(control_state['steering']) > 0.1:
        control_state['throttle'] -= 0.05
        control_state['throttle'] = max(control_state['throttle'], 0.0)
    
    return np.array([
        control_state['throttle'], 
        control_state['brake'], 
        control_state['steering']
    ], dtype=np.float32)
```

#### Advanced Control Strategies

1. **Predictive Control**: Use multiple sensor readings to predict track curvature
2. **Racing Line Optimization**: Combine sensors to find optimal racing line
3. **Tire Management**: Monitor tire temperatures and wear for pit strategy
4. **Collision Avoidance**: Use rear sensors for defensive driving in multi-car races
5. **Adaptive Speed**: Adjust maximum speed based on tire condition and track section

### Competition Usage

#### Running Competitions

```bash
# Single car competition
python demo/competition.py

# Multi-car competition (edit competition.py line 20)
# Change: num_cars = 3  # Up to 10 cars supported
python demo/competition.py
```

#### Competition Configuration

Edit `demo/competition.py` to customize:

```python
# Competition settings
num_cars = 3  # Number of competing cars (1-10)
car_names = ["Lightning", "Thunder", "Blaze"]  # Custom car names

# Environment configuration
env = CarEnv(
    track_file="tracks/nascar.track",    # Track selection
    num_cars=num_cars,
    render_mode="human",                 # "human" for visualization, None for speed
    enable_fps_limit=True,               # Limit FPS for stable visualization
    reset_on_lap=False,                  # Manual reset control
    car_names=car_names
)
```

#### Multi-Car Competition Setup

For multi-car competitions, you can use different control functions per car:

```python
# Import multiple control functions
from control_function_1 import aggressive_control
from control_function_2 import conservative_control  
from control_function_3 import balanced_control

# Map cars to control functions
control_functions = [aggressive_control, conservative_control, balanced_control]

# In the competition loop
for car_idx in range(num_cars):
    car_obs = observations[car_idx]
    action = control_functions[car_idx](car_obs)
    car_actions.append(action)
```

#### Competition Controls During Runtime

- **0-9 Keys**: Switch camera between cars
- **R**: Toggle reward display overlay
- **D**: Toggle debug physics information
- **I**: Toggle track information display
- **C**: Change camera mode (track view / car follow)
- **F**: Toggle fullscreen mode
- **ESC**: Exit competition

#### Performance Metrics Tracked

The competition framework automatically tracks:

1. **Lap Times**: Individual and best lap times per car
2. **Gap Analysis**: Time gaps between cars and to overall best
3. **Reward Tracking**: Cumulative rewards and per-lap rewards
4. **Collision Statistics**: Total collisions and maximum impact severity
5. **Overall Leaderboard**: Best lap time across all competitors
6. **Consistency Analysis**: Average lap times and improvement tracking

#### Competition Output Example

```
🏁 Lightning NEW BEST LAP! Time: 1:23.456 | Reward: 1250.3
   ⚡ Improved by 0.234 seconds!
   🌟 NEW OVERALL BEST LAP! Lightning set the pace: 1:23.456

🏁 Thunder Lap 2 completed: 1:24.123 | Reward: 1180.7  
   🏆 Gap to overall best (Lightning): +0.667s

📊 MULTI-CAR LAP TIME SUMMARY
🏆 Overall best lap time: 1:23.456 (Lightning)
📋 PER-CAR RESULTS:
   🚗 Lightning: Best: 1:23.456⭐ | Average: 1:23.891
   🚗 Thunder: Best: 1:24.123 | Average: 1:24.567
```

## 🧠 Training with TD3

### TD3 Configuration
The included training script uses Twin Delayed DDPG with optimized hyperparameters:

```python
model = TD3(
    "MlpPolicy", 
    env,
    learning_rate=3e-5,           # Conservative learning rate
    buffer_size=1_000_000,        # Large replay buffer
    batch_size=256,               # Stable batch size
    tau=0.005,                    # Soft update rate
    gamma=0.99,                   # Discount factor
    train_freq=1,                 # Update every step
    gradient_steps=4,             # Multiple gradient steps
    action_noise=NormalActionNoise(sigma=0.1),  # Exploration
    target_policy_noise=0.15,     # Target smoothing
    target_noise_clip=0.3,        # Noise clipping
    learning_starts=500_000,      # Warm-up period
    policy_kwargs=dict(net_arch=[1024, 1024])  # Large networks
)
```

### Training Process
```bash
# Start training
python learn/learn.py

# Monitor with TensorBoard
tensorboard --logdir=./tensorboard/

# View training logs
tail -f logs/monitor.csv
```

### Model Checkpoints
Models are automatically saved every 250,000 steps:
```
learn/checkpoints/
├── model_250000_steps.zip
├── model_500000_steps.zip
├── replay_buffer_250000.pkl
└── replay_buffer_500000.pkl
```

### Resume Training
Edit `learn/learn.py` line 120 to load a checkpoint:
```python
model = TD3.load("./learn/checkpoints/model_500000_steps")
```

## ⚙️ Physics Systems

### Car Dynamics
- **Engine Model**: Realistic torque curves with power delivery based on RPM
- **Transmission**: Automatic gear selection with realistic ratios
- **Aerodynamics**: Drag force proportional to velocity squared
- **Mass Distribution**: Weight transfer during acceleration, braking, and cornering

### Tire Physics
```python
class TyreManager:
    """Advanced tire simulation with heating, wear, and pressure"""
    
    def update_tire_physics(self, slip_angle, lateral_force, load, dt):
        # Heat generation from slip and friction
        heat = abs(slip_angle) * abs(lateral_force) * 0.01
        self.temperature += heat * dt
        
        # Wear accumulation
        wear_rate = (abs(slip_angle) + abs(lateral_force) * 0.0001) * dt
        self.wear += wear_rate
        
        # Grip degradation with temperature and wear
        grip_factor = self.calculate_grip_factor()
        return grip_factor
```

### Collision Detection
- **Box2D Integration**: Precise collision detection and response
- **Damage Calculation**: Impulse-based damage with severity classification
- **Multi-Car Handling**: Independent collision tracking per car
- **Termination Logic**: Severe collisions can disable cars in multi-car mode

### Distance Sensors
8-directional raycasting for obstacle detection:
```python
sensor_angles = [0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°]  # Relative to car
max_distance = 100.0  # meters
```

## 💻 Usage Examples

### Basic Environment Usage
```python
import numpy as np
from src.car_env import CarEnv

# Create environment
env = CarEnv(
    render_mode="human",
    track_file="tracks/nascar.track",
    reset_on_lap=False,
    discrete_action_space=False
)

# Reset and run
observation, info = env.reset()
for step in range(1000):
    # Random action
    action = env.action_space.sample()
    
    # Step environment
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Render
    env.render()
    
    # Check termination
    if terminated or truncated:
        print(f"Episode ended: {info['termination_reason']}")
        observation, info = env.reset()

env.close()
```

### Multi-Car Racing
```python
# Create 3-car environment
env = CarEnv(
    render_mode="human",
    track_file="tracks/nascar.track", 
    num_cars=3,
    car_names=["Racer1", "Racer2", "Racer3"]
)

observations, info = env.reset()

for step in range(1000):
    # Generate actions for all cars
    actions = []
    for car_idx in range(3):
        car_obs = observations[car_idx]
        # Simple AI: accelerate and steer based on sensors
        sensors = car_obs[21:29]
        throttle = 0.8
        brake = 0.0
        steering = (sensors[1] - sensors[7]) * 0.5  # Simple steering
        actions.append([throttle, brake, steering])
    
    # Step all cars
    observations, rewards, terminated, truncated, infos = env.step(actions)
    env.render()
    
    # Check individual car status
    for i, info in enumerate(infos):
        if info.get('disabled', False):
            print(f"Car {i} disabled due to collision")

env.close()
```

### External Control Functions
```python
# Example: Using external control function for competition
from demo.default_control import car_control

env = CarEnv(
    render_mode="human",
    track_file="tracks/nascar.track",
    num_cars=1,
    reset_on_lap=False
)

observation, info = env.reset()

for step in range(1000):
    # Use external control function
    action = car_control(observation)  # Returns [throttle, brake, steering]
    
    # Step environment
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    
    if terminated or truncated:
        observation, info = env.reset()

env.close()
```

The `default_control.py` file provides an example control function that:
- Uses distance sensors for obstacle detection
- Implements simple steering logic based on front sensors
- Maintains control state between calls for smooth driving
- Can be easily modified for custom AI implementations

### Loading and Testing Trained Models
```python
from stable_baselines3 import TD3

# Load trained model
model = TD3.load("learn/checkpoints/model_500000_steps")

# Test in environment
env = CarEnv(render_mode="human", track_file="tracks/nascar.track")
obs, info = env.reset()

for step in range(1000):
    # Get model prediction
    action, _states = model.predict(obs, deterministic=True)
    
    # Step environment
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    
    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

## 🎛️ Configuration

### Environment Parameters
```python
env = CarEnv(
    render_mode="human",           # "human" or None
    track_file="tracks/nascar.track",  # Track definition file
    start_position=(0, 0),         # Starting position (auto-detected)
    start_angle=0.0,               # Starting angle in radians
    enable_fps_limit=True,         # Limit rendering FPS
    reset_on_lap=False,            # Auto-reset on lap completion
    discrete_action_space=False,   # Use discrete vs continuous actions
    num_cars=1,                    # Number of cars (1-10)
    car_names=None                 # Custom car names (optional)
)
```

### Track Files
Tracks are stored in `tracks/` directory:
- `nascar.track` - Oval racing circuit
- `nascar2.track` - Alternative oval layout

Track files define racing circuits with:
- Start/finish lines
- Track boundaries  
- Racing lines
- Corner definitions

### Constants (src/constants.py)
Key physics and reward parameters:
```python
# Car specifications
CAR_MASS = 1200.0                 # kg
CAR_MAX_POWER = 350000.0          # Watts
CAR_MAX_SPEED = 300.0             # km/h

# Physics parameters  
GRAVITY_MS2 = 9.81                # Earth gravity
BOX2D_TIME_STEP = 1.0/120.0       # 120Hz physics
DRAG_COEFFICIENT = 0.3            # Aerodynamic drag

# Reward parameters
REWARD_SPEED_MULTIPLIER = 2.0     # Speed reward scaling
REWARD_DISTANCE_MULTIPLIER = 0.1  # Distance reward scaling
REWARD_LAP_COMPLETION = 1000.0    # Lap completion bonus
PENALTY_COLLISION_SEVERE = -50.0  # Severe collision penalty
```

## 📊 Performance

### Physics Performance
- **Simulation Rate**: 120Hz fixed timestep for deterministic behavior
- **Rendering**: Configurable FPS limiting (default 60 FPS)
- **Multi-Car**: Supports up to 10 cars with minimal performance impact

### Training Performance
- **Environment Steps**: ~1000-2000 FPS (headless mode)
- **Memory Usage**: ~2-4 GB for TD3 with 1M replay buffer
- **Training Time**: ~6-12 hours for 1M steps (depends on hardware)

### Optimization Tips
```python
# Fast training (headless)
env = CarEnv(
    render_mode=None,           # Disable rendering
    enable_fps_limit=False,     # Remove FPS limits
    reset_on_lap=True          # Quick episode resets
)

# Stable training (with visualization)
env = CarEnv(
    render_mode="human",
    enable_fps_limit=True,      # Stable 60 FPS
    reset_on_lap=False         # Manual resets
)
```

## 🔧 Development

### Running Tests
```bash
# Run all tests
pytest

# Run specific test category
pytest test/test_car.py
pytest test/test_car_env.py

# Verbose output
pytest -v
```

### Code Structure
```
src/
├── car_env.py              # Main environment class
├── base_env.py             # Base Gymnasium environment
├── car_physics.py          # Multi-car physics world
├── car.py                  # Individual car physics
├── tyre_manager.py         # Tire physics simulation
├── collision.py            # Collision detection & reporting
├── renderer.py             # Pygame visualization
├── camera.py               # Dynamic camera system
├── track_generator.py      # Track loading & parsing
├── distance_sensor.py      # Obstacle detection sensors
├── lap_timer.py            # Race timing system
└── constants.py            # Physics & reward constants

demo/
├── car_demo.py             # Interactive car demonstration
├── random_demo.py          # Random action testing
├── discrete_action_demo.py # Discrete action testing
├── competition.py          # Multi-car competition demo
└── default_control.py      # Example external control function

learn/
├── learn.py                # TD3 training script
└── checkpoints/            # Model saves & replay buffers

tracks/
├── nascar.track            # Oval racing circuit
└── nascar2.track           # Alternative layout
```

### Adding New Tracks
1. Create `.track` file in `tracks/` directory
2. Define track segments with start/end points
3. Include GRID or STARTLINE segment for car positioning
4. Test with `python demo/car_demo.py`

### Custom Reward Functions
Override `_calculate_reward()` method in `CarEnv`:
```python
class CustomCarEnv(CarEnv):
    def _calculate_reward(self):
        # Custom reward logic
        reward = 0.0
        # ... implement custom rewards
        return reward
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/new-feature`)
3. Run tests (`pytest`)
4. Commit changes (`git commit -am 'Add new feature'`)
5. Push to branch (`git push origin feature/new-feature`)
6. Create Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Box2D**: Physics simulation engine
- **Gymnasium**: RL environment standard
- **Stable-Baselines3**: RL algorithm implementations
- **Pygame**: Graphics and rendering

---

For detailed API documentation and advanced usage, see the code comments and docstrings in the source files.