# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a realistic car racing simulation and reinforcement learning environment built with:
- **Box2D Physics**: Realistic car dynamics, collision detection, and tire physics
- **Gymnasium API**: Standard RL environment interface with multi-car support
- **TD3 Learning**: Deep reinforcement learning for autonomous driving
- **Multi-Car Racing**: Support for up to 10 cars with collision handling and camera switching

## Architecture

### Core Components

**Environment Layer** (`src/`):
- `car_env.py` - Main Gymnasium environment with single/multi-car support
- `base_env.py` - Base environment class with action/observation space definitions
- `car_physics.py` - Box2D integration and realistic physics simulation
- `car.py` - Individual car physics with engine, tires, and aerodynamics

**Physics Systems**:
- `tyre_manager.py` / `tyre.py` - Tire wear, temperature, pressure, and grip modeling
- `collision.py` - Collision detection and damage reporting
- `car_physics.py` - Multi-car physics world management

**Rendering & Visualization**:
- `renderer.py` - Pygame-based visualization with debug overlays
- `camera.py` - Dynamic camera system (track view / car follow modes)
- `debug_info_renderer.py` - Real-time physics debug visualization

**Track System**:
- `track_generator.py` - Track loading from `.track` files
- `track_boundary.py` - Track collision boundaries
- `track_polygon_renderer.py` - Track rendering and camera integration
- `centerline_generator.py` - Racing line generation

**Sensors & Timing**:
- `distance_sensor.py` - 8-direction distance sensors for RL observations
- `lap_timer.py` - Lap timing and race progress tracking

### Key Features

**Multi-Car Support**: Environment supports 1-10 cars simultaneously with:
- Individual car names and colors
- Per-car collision handling and disabling
- Camera switching between cars (keys 0-9)
- Independent reward calculation per car

**Realistic Physics**:
- Engine torque curves and power delivery
- Tire heating from slip angle and lateral forces
- Weight transfer during acceleration/braking/cornering
- Aerodynamic drag and rolling resistance
- Friction circle modeling for combined grip

**RL Integration**:
- Normalized observation space (29 dimensions)
- Continuous action space [throttle, brake, steering]
- Distance-based and speed-based rewards
- Collision penalties with severity levels
- Lap completion bonuses

## Development Commands

### Basic Testing
```bash
# Run basic environment test
python demo/car_demo.py

# Run simplified physics demo
python demo/simplified_physics.py

# Run random action demo
python demo/random_demo.py

# Run discrete action demo
python demo/discrete_action_demo.py
```

### Training
```bash
# Train RL agent with TD3
python learn/learn.py

# Resume training from checkpoint (edit learn.py line 79)
# Uncomment and update model path to resume training
```

### Testing
```bash
# Run all tests
pytest

# Run specific test file
pytest test/test_car.py

# Run with verbose output
pytest -v
```

### Installation
```bash
# Install dependencies
pip install -r requirements.txt
```

## Important Implementation Details

### Multi-Car Environment Usage
```python
# Single car (backward compatible)
env = CarEnv(render_mode="human", track_file="tracks/nascar.track")

# Multi-car with custom names
car_names = ["Lightning", "Thunder", "Blaze"]
env = CarEnv(render_mode="human", num_cars=3, car_names=car_names)

# Runtime car switching (keys 0-9 in render mode)
# followed_car_index tracks which car camera follows
```

### Observation Space (29 dimensions)
Normalized to [-1,1] or [0,1]:
- Position (x, y) 
- Velocity (x, y, magnitude)
- Orientation and angular velocity
- Tire loads (4), temperatures (4), wear (4)
- Collision data (impulse, angle)
- Distance sensor readings (8 directions)

### Action Space
- **Continuous**: [throttle, brake, steering] in ranges [0,1], [0,1], [-1,1]
- **Discrete**: 5 actions (idle, accelerate, brake, left, right)

### Track Files
Tracks stored in `tracks/` directory:
- `nascar.track` - Oval racing circuit
- `nascar2.track` - Alternative oval layout

### Constants and Tuning
All physics constants in `src/constants.py`:
- Car specifications (mass, power, dimensions)
- Physics parameters (drag, friction, tire grip)
- Reward function parameters
- Multi-car limits and collision thresholds

### Rendering Controls
- `R` - Toggle reward display
- `C` - Toggle camera mode (track view / car follow)  
- `I` - Toggle track info display
- `F` - Toggle fullscreen
- `0-9` - Switch between cars (multi-car mode)

## Performance Notes

- Physics runs at 120Hz fixed timestep for deterministic behavior
- Rendering FPS can be limited or unlimited via `enable_fps_limit` parameter
- Use `reset_on_lap=True` for training, `False` for demo/testing
- Disable rendering (`render_mode=None`) for faster training