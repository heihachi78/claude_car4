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

The environment uses a comprehensive reward function that encourages fast, smooth driving:

#### Speed Rewards (Time-Based)
```python
# Base speed reward - encourages forward motion
speed_reward = speed_ms * 2.0 * dt

# High speed bonus for speeds > 30 m/s
if speed > 30.0:
    speed_reward += 5.0 * dt
```

#### Distance Rewards
```python
# Reward for distance traveled (+0.1 per meter)
distance_reward = distance_traveled * 0.1
```

#### Penalties
```python
# Low speed penalty for speeds < 5 m/s
if speed < 5.0:
    penalty = -1.0 * dt

# Collision penalties (severity-based, applied once per collision)
collision_penalties = {
    "minor": -10.0,      # < 100 impulse
    "moderate": -25.0,   # 100-500 impulse  
    "severe": -50.0,     # 500-1000 impulse
    "critical": -100.0   # > 1000 impulse
}

# Off-track penalty
if not on_track:
    penalty = -2.0 * dt
```

#### Lap Completion Bonuses
```python
# Major bonus for completing laps
lap_bonus = 1000.0 * laps_completed

# Additional bonus for fast laps (< 60 seconds)
if lap_time < 60.0:
    fast_lap_bonus = 200.0
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

### Multi-Car Features
- **Individual Collision Handling**: Cars can be disabled due to severe collisions
- **Camera Switching**: Press 0-9 keys to follow different cars
- **Independent Rewards**: Each car receives its own reward signal
- **Race Positions**: Real-time position tracking based on track progress
- **Lap Time Leaderboards**: Best lap times tracked per car

### Camera and Following
- Environment tracks a "followed car" for camera and primary RL agent
- Switch followed car with keyboard (0-9 keys) or programmatically
- Legacy single-car references maintained for backward compatibility

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
└── discrete_action_demo.py # Discrete action testing

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