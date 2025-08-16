"""
Main car racing environment.

This module provides the complete car racing simulation environment with
realistic physics, track integration, and comprehensive observation space.
"""

import logging
import math
import numpy as np
import pygame
import time
from typing import Optional, Tuple, Dict, Any
from .base_env import BaseEnv
from .car_physics import CarPhysics
from .collision import CollisionReporter
from .track_generator import TrackLoader
from .renderer import Renderer
from .distance_sensor import DistanceSensor
from .lap_timer import LapTimer
from .constants import (
    DEFAULT_WINDOW_SIZE,
    DEFAULT_RENDER_FPS,
    RENDER_MODE_HUMAN,
    TYRE_START_TEMPERATURE,
    CAR_MASS,
    GRAVITY_MS2,
    BOX2D_TIME_STEP,
    SENSOR_MAX_DISTANCE,
    SENSOR_NUM_DIRECTIONS,
    STUCK_SPEED_THRESHOLD,
    STUCK_TIME_THRESHOLD,
    STUCK_INPUT_THRESHOLD,
    STUCK_RECENT_INPUT_TIME,
    # Reward constants
    REWARD_SPEED_MULTIPLIER,
    REWARD_DISTANCE_MULTIPLIER,
    REWARD_HIGH_SPEED_THRESHOLD,
    REWARD_HIGH_SPEED_BONUS,
    REWARD_LAP_COMPLETION,
    REWARD_FAST_LAP_TIME,
    REWARD_FAST_LAP_BONUS,
    REWARD_FORWARD_SENSOR_MULTIPLIER,
    PENALTY_LOW_SPEED_THRESHOLD,
    PENALTY_LOW_SPEED_RATE,
    PENALTY_COLLISION_MINOR,
    PENALTY_COLLISION_MODERATE,
    PENALTY_COLLISION_SEVERE,
    PENALTY_COLLISION_CRITICAL,
    PENALTY_COLLISION_EXTREME,
    # Collision severity constants
    COLLISION_SEVERITY_MINOR,
    COLLISION_SEVERITY_MODERATE,
    COLLISION_SEVERITY_SEVERE,
    COLLISION_SEVERITY_EXTREME,
    # Termination constants
    TERMINATION_MIN_REWARD,
    TERMINATION_MAX_TIME,
    TRUNCATION_MAX_TIME,
    TERMINATION_COLLISION_WINDOW,
    # Normalization constants
    NORM_MAX_POSITION,
    NORM_MAX_VELOCITY,
    NORM_MAX_ANGULAR_VEL,
    NORM_MAX_TYRE_LOAD,
    NORM_MAX_TYRE_TEMP,
    NORM_MAX_TYRE_WEAR,
    NORM_MAX_COLLISION_IMPULSE,
    # Multi-car constants
    MAX_CARS,
    MULTI_CAR_COLORS,
    CAR_SELECT_KEYS
)

# Setup module logger
logger = logging.getLogger(__name__)


class CarEnv(BaseEnv):
    """Complete car racing environment with realistic physics"""
    
    def __init__(self, 
                 render_mode: Optional[str] = None,
                 track_file: Optional[str] = None,
                 start_position: Optional[Tuple[float, float]] = None,
                 start_angle: float = 0.0,
                 enable_fps_limit: bool = True,
                 reset_on_lap: bool = False,
                 discrete_action_space: bool = False,
                 num_cars: int = 1,
                 car_names: Optional[list] = None):
        """
        Initialize car racing environment.
        
        Args:
            render_mode: Rendering mode ("human" or None)
            track_file: Path to track definition file
            start_position: Car starting position (auto-detected if None)
            start_angle: Car starting angle in radians
            enable_fps_limit: Whether to limit render FPS (True for normal use, False for benchmarking)
            reset_on_lap: If True, reset environment automatically when a lap is completed
            discrete_action_space: If True, use discrete action space (5 actions) instead of continuous
            num_cars: Number of cars to create (1-10)
            car_names: List of names for each car (optional, defaults to "Car 0", "Car 1", etc.)
        """
        super().__init__(discrete_action_space=discrete_action_space, num_cars=num_cars)
        
        # Validate num_cars parameter
        if num_cars < 1 or num_cars > MAX_CARS:
            raise ValueError(f"Number of cars must be between 1 and {MAX_CARS}")
        
        self.render_mode = render_mode
        self.track_file = track_file
        self.track = None
        self.start_position = start_position or (0.0, 0.0)
        self.start_angle = start_angle
        self.reset_on_lap = reset_on_lap
        
        # Multi-car attributes
        self.num_cars = num_cars
        self.followed_car_index = 0  # Index of car being followed by camera and RL agent
        self.car_colors = MULTI_CAR_COLORS[:num_cars]  # Assign colors to each car
        self.car_lap_timers = []  # One lap timer per car
        
        # Initialize car names
        if car_names is None:
            self.car_names = [f"Car {i}" for i in range(num_cars)]
        else:
            if len(car_names) != num_cars:
                raise ValueError(f"Number of car names ({len(car_names)}) must match number of cars ({num_cars})")
            self.car_names = list(car_names)  # Make a copy
        
        # Load track if specified
        if track_file:
            self._load_track(track_file)
            
        # Create physics system
        self.car_physics = CarPhysics(self.track)
        self.car = None  # Legacy single car reference (for backward compatibility)
        self.cars = []   # List of all cars in the simulation
        
        # Collision system
        self.collision_reporter = CollisionReporter()
        
        # Distance sensor system
        self.distance_sensor = DistanceSensor()
        
        # Lap timing system (create one timer per car)
        self._initialize_lap_timers()
        
        # Rendering system
        self.renderer = None
        self.headless_clock = None  # Clock for timing in headless mode
        
        # Force headless mode when enable_fps_limit=False
        if not enable_fps_limit:
            render_mode = None
            
        if render_mode == RENDER_MODE_HUMAN:
            self.renderer = Renderer(
                window_size=DEFAULT_WINDOW_SIZE,
                render_fps=self.metadata["render_fps"],
                track=self.track,
                enable_fps_limit=enable_fps_limit  # Use the parameter from constructor
            )
        else:
            # Initialize pygame clock for headless mode to maintain proper physics timing
            import pygame
            if not pygame.get_init():
                pygame.init()
            self.headless_clock = pygame.time.Clock()
            self.headless_enable_fps_limit = enable_fps_limit
            
        # Environment state
        self.simulation_time = 0.0
        self.actual_dt = BOX2D_TIME_STEP  # Will be updated with each physics step
        
        # Store latest action for physics updates
        self._current_physics_action = None
        
        # Performance tracking
        self.episode_stats = {}
        
        # Input tracking for stuck detection
        self.input_history = []  # List of (time, throttle, brake, steering) tuples
        self.last_meaningful_input_time = 0.0
        
        # Current action for rendering
        self.current_action = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # [throttle, brake, steering]
        self.all_actions = None  # Will store all actions for multi-car scenarios
        
        # Track disabled cars (for multi-car collision handling)
        self.disabled_cars = set()  # Set of car indices that are disabled due to collisions
        
        # Lap reset control - prevent immediate reset on first lap
        self._lap_reset_pending = False
        
        # Track previous lap count for lap completion rewards
        self._previous_lap_count = 0
        
        # Track car position for distance-based rewards
        self._previous_car_position = None
        self._total_distance_traveled = 0.0
        
        # Track last penalized collision to avoid double penalties
        self._last_penalized_collision_time = -float('inf')
        
        # Reward display control
        self._show_reward = False
        self._current_reward = 0.0
        self._cumulative_reward = 0.0
        
        # Termination reason tracking
        self.termination_reason = None
        
        logger.info(f"CarEnv initialized with {num_cars} car(s) and track: {track_file}")
    
    def _initialize_lap_timers(self) -> None:
        """Initialize lap timers for all cars"""
        self.car_lap_timers = []
        for i in range(self.num_cars):
            car_name = self.car_names[i] if i < len(self.car_names) else f"Car {i}"
            lap_timer = LapTimer(self.track, car_id=car_name)
            self.car_lap_timers.append(lap_timer)
        
        # Set main lap timer to first car's timer for backward compatibility
        self.lap_timer = self.car_lap_timers[0] if self.car_lap_timers else LapTimer(self.track, car_id="Legacy Car")
        
    def _load_track(self, track_file: str) -> None:
        """Load track from file"""
        try:
            track_loader = TrackLoader()
            self.track = track_loader.load_track(track_file)
            logger.info(f"Loaded track: {track_file}")
            
            # Set start position to track start if not specified
            if self.start_position == (0.0, 0.0) and self.track.segments:
                # Find GRID or STARTLINE segment for starting position
                for segment in self.track.segments:
                    if segment.segment_type in ["GRID", "STARTLINE"]:
                        self.start_position = segment.start_position
                        break
                        
        except Exception as e:
            logger.error(f"Failed to load track {track_file}: {e}")
            self.track = None
            
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset environment to initial state.
        
        Args:
            seed: Random seed (optional)
            options: Additional options (optional)
            
        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed=seed)
        
        # Create or reset cars
        if not self.cars:
            # Create multiple cars
            self.cars = self.car_physics.create_cars(self.num_cars, self.start_position, self.start_angle)
            self.car = self.cars[0] if self.cars else None  # Legacy reference
        else:
            # Reset existing cars
            self.car_physics.reset_cars(self.start_position, self.start_angle)
            
        # Reset systems
        self.collision_reporter.reset()
        
        # Reset all lap timers
        for lap_timer in self.car_lap_timers:
            lap_timer.reset()
        self.lap_timer.reset()  # Legacy timer
        
        self.simulation_time = 0.0
        
        # Reset physics action tracking
        self._current_physics_action = None
        
        # Reset input tracking
        self.input_history = []
        self.last_meaningful_input_time = 0.0
        
        # Reset current action
        self.current_action = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.all_actions = None
        self._last_rewards = None
        
        # Reset disabled cars
        self.disabled_cars.clear()
        
        # Reset lap reset control
        self._lap_reset_pending = False
        
        # Reset lap count tracking for rewards
        self._previous_lap_count = 0
        
        # Reset distance tracking for rewards (use followed car)
        if self.cars and self.followed_car_index < len(self.cars):
            car_state = self.car_physics.get_car_state(self.followed_car_index)
            self._previous_car_position = (car_state[0], car_state[1]) if car_state else None
        else:
            self._previous_car_position = None
        
        # Reset collision penalty tracking
        self._last_penalized_collision_time = -float('inf')
        self._total_distance_traveled = 0.0
        
        # Reset reward tracking
        self._current_reward = 0.0
        self._cumulative_reward = 0.0
        
        # Reset multi-car cumulative rewards and per-car tracking
        if self.num_cars > 1:
            self._cumulative_rewards = [0.0] * self.num_cars
            
            # Reset per-car collision tracking and lap count tracking
            for car_index in range(self.num_cars):
                setattr(self, f'_last_penalized_collision_time_{car_index}', -float('inf'))
                setattr(self, f'_previous_car_position_{car_index}', None)
                setattr(self, f'_previous_lap_count_{car_index}', 0)
        
        # Reset termination reason
        self.termination_reason = None
        
        # Reset episode statistics
        self.episode_stats = {
            "max_speed": 0.0,
            "distance_traveled": 0.0,
            "collisions": 0,
            "time_on_track": 0.0,
            "performance_valid": False
        }
        
        # Get initial observation (use appropriate method for single vs multi-car)
        if self.num_cars == 1:
            observation = self._get_obs()
            info = self._get_info()
        else:
            observation = self._get_multi_obs()
            info = self._get_multi_info()
        
        logger.debug("Environment reset complete")
        return observation, info
        
    def update_physics(self, actions) -> None:
        """
        Update physics simulation with proper timing modes.
        
        Two modes:
        1. enable_fps_limit=True: One physics step per frame, simulating delta time from frame rate
        2. enable_fps_limit=False: Run physics as fast as possible with fixed 1/60 timesteps
        
        Args:
            actions: For single car: [throttle, brake, steering] for followed car
                    For multi-car: array of actions for all cars or single action for followed car only
        """
        if not self.cars:
            return
            
        # Store the latest action(s)
        self._current_physics_action = actions
        
        # Prepare actions for physics step
        if self.num_cars == 1:
            # Single car mode - handle followed car only
            if self.followed_car_index >= len(self.cars):
                return
            throttle, brake, steering = float(actions[0]), float(actions[1]), float(actions[2])
            physics_actions = (throttle, brake, steering)
        else:
            # Multi-car mode - handle all cars
            if hasattr(actions, 'shape') and len(actions.shape) == 2:
                # Multi-car actions: (num_cars, 3) array
                physics_actions = actions
            elif len(actions) == 3 and isinstance(actions[0], (int, float)):
                # Single action for followed car - create multi-car action array with idle for others
                physics_actions = np.zeros((self.num_cars, 3), dtype=np.float32)
                if self.followed_car_index < self.num_cars:
                    physics_actions[self.followed_car_index] = actions
            else:
                # Assume it's already a list/array of actions for each car
                physics_actions = actions

        # Check if fps limit is enabled
        enable_fps_limit = getattr(self, 'headless_enable_fps_limit', True)
        if self.renderer:
            enable_fps_limit = self.renderer.enable_fps_limit
        
        if enable_fps_limit:
            # MODE 1: enable_fps_limit=True
            # One physics step per frame, using the expected frame rate as timestep
            # This ensures consistent physics regardless of actual frame rate variations
            
            # Get target fps for this environment
            target_fps = self.metadata.get("render_fps", DEFAULT_RENDER_FPS)
            expected_dt = 1.0 / target_fps
            
            # Run exactly one physics step with expected delta time
            self._run_single_physics_step(physics_actions, expected_dt)
            
        else:
            # MODE 2: enable_fps_limit=False  
            # Run physics as fast as possible with fixed 1/60 timesteps
            
            # Always use fixed timestep for consistent simulation speed
            fixed_dt = 1.0/60.0
            
            # Run exactly one physics step with fixed timestep
            self._run_single_physics_step(physics_actions, fixed_dt)

    def _run_single_physics_step(self, physics_actions, dt):
        """Run a single physics step with the given timestep"""
        
        # Store the actual timestep used (for reward calculations)
        self.actual_dt = dt
        
        # Update collision reporter time
        self.collision_reporter.update_time(self.simulation_time)
        
        # Process collision events from car physics
        self._process_physics_collisions()
        
        # Perform physics step with given timestep
        self.car_physics.step(physics_actions, dt)
        
        # Update simulation time
        self.simulation_time += dt
        
        # Update lap timers for all cars
        for car_index in range(len(self.cars)):
            if car_index < len(self.car_lap_timers):
                car_state = self.car_physics.get_car_state(car_index)
                if car_state:
                    car_position = (car_state[0], car_state[1])  # x, y position
                    
                    # Update this car's lap timer with simulation time
                    lap_timer = self.car_lap_timers[car_index]
                    lap_completed = lap_timer.update(car_position, self.simulation_time)
                    
                    if lap_completed:
                        logger.info(f"Car {car_index} lap completed! Time: {lap_timer.format_time(lap_timer.get_last_lap_time())}")
                        
                        # Mark reset as pending if reset_on_lap is enabled and this is the followed car
                        if self.reset_on_lap and car_index == self.followed_car_index:
                            self._lap_reset_pending = True
        
    def step(self, action):
        """
        Execute one environment step.
        
        Args:
            action: Control inputs - for single car: [throttle, brake, steering]
                   for multi-car: [[throttle, brake, steering], ...] or (num_cars, 3) array
            
        Returns:
            For single car: (observation, reward, terminated, truncated, info)
            For multi-car: (observations, rewards, terminated, truncated, infos)
        """
        # Validate action
        assert self.action_space.contains(action), f"Invalid action {action}"
        
        if not self.cars:
            raise RuntimeError("Environment not properly initialized. Call reset() first.")
        
        if self.num_cars == 1:
            # Single car mode - backward compatibility
            return self._step_single_car(action)
        else:
            # Multi-car mode
            return self._step_multi_car(action)
    
    def _step_single_car(self, action):
        """Handle single car step for backward compatibility"""
        # Convert discrete action to continuous if needed
        if self.discrete_action_space:
            continuous_action = self._discrete_to_continuous(action)
            throttle, brake, steering = continuous_action[0], continuous_action[1], continuous_action[2]
        else:
            # Convert action to tuple for car physics
            throttle, brake, steering = float(action[0]), float(action[1]), float(action[2])
        
        # Store current action for rendering
        self.current_action = np.array([throttle, brake, steering], dtype=np.float32)
        
        # Update physics simulation
        continuous_action_array = np.array([throttle, brake, steering], dtype=np.float32)
        self.update_physics(continuous_action_array)
        
        # Maintain timing in headless mode
        if self.headless_clock:
            if self.headless_enable_fps_limit:
                self.headless_clock.tick(self.metadata["render_fps"])
            else:
                from .constants import UNLIMITED_FPS_CAP
                self.headless_clock.tick(UNLIMITED_FPS_CAP)
        
        # Track input for stuck detection
        self._track_input(throttle, brake, steering)
        
        # Update episode statistics
        self._update_episode_stats()
        
        # Get observation
        observation = self._get_obs()
        
        # Calculate reward
        reward = self._calculate_reward()
        self._current_reward = reward
        self._cumulative_reward += reward
        
        # Check termination conditions
        terminated, truncated = self._check_termination()
        
        # Check if lap reset is pending
        if self._lap_reset_pending:
            self._lap_reset_pending = False
            terminated = True
            
        # Get info
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def _step_multi_car(self, actions):
        """Handle multi-car step"""
        # Convert actions to proper format
        if self.discrete_action_space:
            # Convert each discrete action to continuous
            continuous_actions = []
            for discrete_action in actions:
                continuous_action = self._discrete_to_continuous(discrete_action)
                continuous_actions.append(continuous_action)
            actions = np.array(continuous_actions, dtype=np.float32)
        
        # Store all actions for rendering
        self.all_actions = np.array(actions, dtype=np.float32)
        
        # Store current action for the followed car for backward compatibility
        if hasattr(self, 'followed_car_index') and 0 <= self.followed_car_index < len(actions):
            self.current_action = np.array(actions[self.followed_car_index], dtype=np.float32)
        else:
            self.current_action = np.array(actions[0], dtype=np.float32) if len(actions) > 0 else np.array([0.0, 0.0, 0.0], dtype=np.float32)
        
        # Filter actions for disabled cars (set to zero action)
        filtered_actions = np.copy(actions)
        for car_idx in self.disabled_cars:
            if car_idx < len(filtered_actions):
                filtered_actions[car_idx] = [0.0, 0.0, 0.0]  # No throttle, no brake, no steering
        
        # Update physics with filtered actions
        self.update_physics(filtered_actions)
        
        # Maintain timing in headless mode
        if self.headless_clock:
            if self.headless_enable_fps_limit:
                self.headless_clock.tick(self.metadata["render_fps"])
            else:
                from .constants import UNLIMITED_FPS_CAP
                self.headless_clock.tick(UNLIMITED_FPS_CAP)
        
        # Check for new collisions and disable cars if necessary
        self._check_and_disable_cars()
        
        # Update physics system with disabled cars info (to suppress collision messages)
        self.car_physics.set_disabled_cars(self.disabled_cars)
        
        # Get observations, rewards, and info for all cars
        observations = self._get_multi_obs()
        rewards = self._calculate_multi_rewards()
        terminated, truncated = self._check_multi_termination()
        infos = self._get_multi_info()
        
        # Store last rewards for display purposes
        self._last_rewards = rewards
        
        # Update cumulative rewards
        if not hasattr(self, '_cumulative_rewards'):
            self._cumulative_rewards = [0.0] * self.num_cars
        for i, reward in enumerate(rewards):
            self._cumulative_rewards[i] += reward
        
        return observations, rewards, terminated, truncated, infos
    
    def _check_and_disable_cars(self):
        """Check for sustained severe collisions and disable cars in multi-car mode"""
        if self.num_cars <= 1:
            return  # Only apply in multi-car mode
        
        # Track cars disabled in this timestep for final penalty application
        if not hasattr(self, '_just_disabled_cars'):
            self._just_disabled_cars = set()
        else:
            self._just_disabled_cars.clear()
            
        for car_idx in range(self.num_cars):
            if car_idx in self.disabled_cars:
                continue  # Car is already disabled
                
            # Check collision impulse for this car using continuous collision data
            collision_impulse = self.car_physics.get_continuous_collision_impulse(car_idx)
            
            # Initialize collision duration tracking if needed
            collision_duration_attr = f'_collision_duration_{car_idx}'
            if not hasattr(self, collision_duration_attr):
                setattr(self, collision_duration_attr, 0.0)
            
            if collision_impulse > COLLISION_SEVERITY_EXTREME:
                # Immediate disabling for extreme collisions (>5000N)
                if car_idx not in self.disabled_cars:
                    self.disabled_cars.add(car_idx)
                    self._just_disabled_cars.add(car_idx)  # Track for final penalty
                    car_name = self.car_names[car_idx] if car_idx < len(self.car_names) else f"Car {car_idx}"
                    print(f"🚫 {car_name} IMMEDIATELY DISABLED due to extreme collision (impulse: {collision_impulse:.0f}N)")
            elif collision_impulse > COLLISION_SEVERITY_SEVERE:
                # Accumulate collision duration for severe/critical collisions
                current_duration = getattr(self, collision_duration_attr)
                current_duration += self.actual_dt
                setattr(self, collision_duration_attr, current_duration)
                
                # Disable after sustained severe collision (reduced from 2.0s to 0.8s)
                if current_duration > 0.8:
                    self.disabled_cars.add(car_idx)
                    self._just_disabled_cars.add(car_idx)  # Track for final penalty
                    car_name = self.car_names[car_idx] if car_idx < len(self.car_names) else f"Car {car_idx}"
                    print(f"🚫 {car_name} disabled due to sustained severe collision (impulse: {collision_impulse:.0f}N, duration: {current_duration:.1f}s)")
            else:
                # Reset collision duration when not colliding severely
                setattr(self, collision_duration_attr, 0.0)
    
    def _get_multi_obs(self):
        """Get observations for all cars"""
        observations = []
        for car_index in range(self.num_cars):
            if car_index < len(self.cars) and self.cars[car_index]:
                # Get car state
                car_state = self.car_physics.get_car_state(car_index)
                if car_state:
                    pos_x, pos_y, vel_x, vel_y, orientation, angular_vel = car_state
                    
                    # Normalize position, velocity, etc. (same as single car)
                    norm_pos_x = np.clip(pos_x / NORM_MAX_POSITION, -1.0, 1.0)
                    norm_pos_y = np.clip(pos_y / NORM_MAX_POSITION, -1.0, 1.0)
                    norm_vel_x = np.clip(vel_x / NORM_MAX_VELOCITY, -1.0, 1.0)
                    norm_vel_y = np.clip(vel_y / NORM_MAX_VELOCITY, -1.0, 1.0)
                    speed_magnitude_ms = (vel_x**2 + vel_y**2)**0.5
                    norm_speed_magnitude = np.clip(speed_magnitude_ms / NORM_MAX_VELOCITY, 0.0, 1.0)
                    norm_orientation = orientation / np.pi
                    norm_angular_vel = np.clip(angular_vel / NORM_MAX_ANGULAR_VEL, -1.0, 1.0)
                    
                    # Get tyre data
                    tyre_data = self.car_physics.get_tyre_data(car_index)
                    tyre_loads, tyre_temps, tyre_wear = tyre_data
                    norm_tyre_loads = [np.clip(load / NORM_MAX_TYRE_LOAD, 0.0, 1.0) for load in tyre_loads]
                    norm_tyre_temps = [np.clip(temp / NORM_MAX_TYRE_TEMP, 0.0, 1.0) for temp in tyre_temps]
                    norm_tyre_wear = [np.clip(wear / NORM_MAX_TYRE_WEAR, 0.0, 1.0) for wear in tyre_wear]
                    
                    # Get collision data (simplified for multi-car)
                    collision_impulse, collision_angle = self.car_physics.get_collision_data(car_index)
                    norm_collision_impulse = np.clip(collision_impulse / NORM_MAX_COLLISION_IMPULSE, 0.0, 1.0)
                    norm_collision_angle = collision_angle / np.pi
                    
                    # Get sensor distances
                    world = self.car_physics.world if self.car_physics else None
                    sensor_distances = self.distance_sensor.get_sensor_distances(
                        world, (pos_x, pos_y), orientation
                    )
                    normalized_sensor_distances = [np.clip(d / SENSOR_MAX_DISTANCE, 0.0, 1.0) for d in sensor_distances]
                    
                    # Construct observation
                    observation = np.array([
                        norm_pos_x, norm_pos_y,
                        norm_vel_x, norm_vel_y,
                        norm_speed_magnitude,
                        norm_orientation, norm_angular_vel,
                        norm_tyre_loads[0], norm_tyre_loads[1], norm_tyre_loads[2], norm_tyre_loads[3],
                        norm_tyre_temps[0], norm_tyre_temps[1], norm_tyre_temps[2], norm_tyre_temps[3],
                        norm_tyre_wear[0], norm_tyre_wear[1], norm_tyre_wear[2], norm_tyre_wear[3],
                        norm_collision_impulse, norm_collision_angle,
                        normalized_sensor_distances[0], normalized_sensor_distances[1], 
                        normalized_sensor_distances[2], normalized_sensor_distances[3],
                        normalized_sensor_distances[4], normalized_sensor_distances[5], 
                        normalized_sensor_distances[6], normalized_sensor_distances[7]
                    ], dtype=np.float32)
                    
                    observations.append(observation)
                else:
                    # Car state not available - add default observation
                    observations.append(self._get_default_observation())
            else:
                # Car doesn't exist - add default observation
                observations.append(self._get_default_observation())
        
        return np.array(observations, dtype=np.float32)
    
    def _get_default_observation(self):
        """Get default normalized observation for non-existent or invalid cars"""
        static_load = CAR_MASS * GRAVITY_MS2 / 4.0
        normalized_static_load = static_load / NORM_MAX_TYRE_LOAD
        normalized_start_temp = TYRE_START_TEMPERATURE / NORM_MAX_TYRE_TEMP
        default_normalized_sensor_distances = [1.0] * SENSOR_NUM_DIRECTIONS
        
        return np.array([
            0.0, 0.0,  # position
            0.0, 0.0,  # velocity
            0.0,       # speed magnitude
            0.0, 0.0,  # orientation, angular velocity
            normalized_static_load, normalized_static_load, 
            normalized_static_load, normalized_static_load,  # tyre loads
            normalized_start_temp, normalized_start_temp,
            normalized_start_temp, normalized_start_temp,  # tyre temperatures
            0.0, 0.0, 0.0, 0.0,  # tyre wear
            0.0, 0.0,  # collision data
            *default_normalized_sensor_distances  # sensor distances
        ], dtype=np.float32)
    
    def _calculate_multi_rewards(self):
        """Calculate rewards for all cars"""
        rewards = []
        
        for car_index in range(self.num_cars):
            if car_index < len(self.cars) and self.cars[car_index]:
                # Calculate reward even for disabled cars (to apply final collision penalty)
                # but only for the timestep they were disabled
                car_just_disabled = hasattr(self, '_just_disabled_cars') and car_index in getattr(self, '_just_disabled_cars', set())
                
                if car_index in self.disabled_cars and not car_just_disabled:
                    rewards.append(0.0)
                    continue
                    
                car = self.cars[car_index]
                reward = 0.0
                
                # Speed reward - time-based
                speed = car.get_velocity_magnitude()
                reward += speed**2 * REWARD_SPEED_MULTIPLIER * self.actual_dt
                
                # Penalty for being too slow
                if speed < PENALTY_LOW_SPEED_THRESHOLD:
                    reward -= PENALTY_LOW_SPEED_RATE * self.actual_dt
                
                # Distance reward (track per car if needed)
                car_state = self.car_physics.get_car_state(car_index)
                if car_state:
                    current_position = (car_state[0], car_state[1])
                    
                    # For simplicity, calculate distance from last position if we had one
                    # (In full implementation, you'd track previous positions per car)
                    if hasattr(self, f'_previous_car_position_{car_index}'):
                        prev_pos = getattr(self, f'_previous_car_position_{car_index}')
                        if prev_pos:
                            dx = current_position[0] - prev_pos[0]
                            dy = current_position[1] - prev_pos[1]
                            distance = (dx**2 + dy**2)**0.5
                            reward += distance * REWARD_DISTANCE_MULTIPLIER
                    
                    # Update previous position for this car
                    setattr(self, f'_previous_car_position_{car_index}', current_position)
                
                # CONTINUOUS collision penalty - applied every timestep while colliding
                # Get current collision impulse from physics system
                collision_impulse = self.car_physics.get_continuous_collision_impulse(car_index)
                
                if collision_impulse > 0:
                    # Determine severity based on impulse magnitude
                    if collision_impulse < COLLISION_SEVERITY_MINOR:
                        collision_penalty = PENALTY_COLLISION_MINOR
                    elif collision_impulse < COLLISION_SEVERITY_MODERATE:
                        collision_penalty = PENALTY_COLLISION_MODERATE
                    elif collision_impulse < COLLISION_SEVERITY_SEVERE:
                        collision_penalty = PENALTY_COLLISION_SEVERE
                    elif collision_impulse < COLLISION_SEVERITY_EXTREME:
                        collision_penalty = PENALTY_COLLISION_CRITICAL
                    else:
                        collision_penalty = PENALTY_COLLISION_EXTREME
                    
                    # For extreme collisions that cause immediate disabling, apply full penalty
                    # For other collisions, apply penalty per second
                    if collision_impulse >= COLLISION_SEVERITY_EXTREME and car_just_disabled:
                        penalty_applied = collision_penalty  # Full penalty for car destruction
                        #print(f"💰 EXTREME PENALTY: car={car_index} impulse={collision_impulse:.1f} penalty={penalty_applied:.1f} (FULL DESTRUCTION PENALTY)")
                    else:
                        penalty_applied = collision_penalty * self.actual_dt  # Normal per-second penalty
                        #print(f"💰 REWARD PENALTY: car={car_index} impulse={collision_impulse:.1f} penalty={penalty_applied:.3f} (rate={collision_penalty:.1f}/s)")
                    
                    reward -= penalty_applied
                
                
                # High speed bonus - time-based  
                if speed > REWARD_HIGH_SPEED_THRESHOLD:
                    reward += REWARD_HIGH_SPEED_BONUS * self.actual_dt
                
                # Forward sensor reward - reward for having clear space ahead (time-based)
                # Get sensor distances for this car
                world = self.car_physics.world if self.car_physics else None
                car_position = (car_state[0], car_state[1])
                car_orientation = car_state[4] if len(car_state) > 4 else 0.0
                sensor_distances = self.distance_sensor.get_sensor_distances(
                    world, car_position, car_orientation
                )
                # Normalize forward sensor distance (index 0)
                normalized_forward_distance = np.clip(sensor_distances[0] / SENSOR_MAX_DISTANCE, 0.0, 1.0)
                # Apply reward per second of having forward space
                reward += REWARD_FORWARD_SENSOR_MULTIPLIER * normalized_forward_distance * self.actual_dt
                
                # Lap completion bonus
                if car_index < len(self.car_lap_timers):
                    lap_timer = self.car_lap_timers[car_index]
                    lap_info = lap_timer.get_timing_info()
                    current_lap_count = lap_info.get("lap_count", 0)
                    
                    # Track previous lap count per car
                    prev_lap_attr = f'_previous_lap_count_{car_index}'
                    prev_lap_count = getattr(self, prev_lap_attr, 0)
                    
                    if current_lap_count > prev_lap_count:
                        laps_completed = current_lap_count - prev_lap_count
                        reward += REWARD_LAP_COMPLETION * laps_completed
                        
                        # Fast lap bonus
                        last_lap_time = lap_info.get("last_lap_time", None)
                        if last_lap_time and last_lap_time < REWARD_FAST_LAP_TIME:
                            reward += REWARD_FAST_LAP_BONUS
                        
                        setattr(self, prev_lap_attr, current_lap_count)
                
                rewards.append(reward)
            else:
                # Car doesn't exist - no reward
                rewards.append(0.0)
        
        return np.array(rewards, dtype=np.float32)
    
    def _check_multi_termination(self):
        """Check termination conditions for multi-car environment"""
        # In multi-car mode, we use a different strategy:
        # - Don't terminate on individual car collisions (cars get disabled instead)
        # - Only terminate on time limits or if ALL cars are disabled
        
        # Check if all cars are disabled
        if len(self.disabled_cars) >= self.num_cars:
            self.termination_reason = "all_cars_disabled"
            return True, False
        
        # Only check non-collision termination conditions for followed car
        if self.followed_car_index < len(self.cars) and self.cars[self.followed_car_index]:
            followed_car = self.cars[self.followed_car_index]
            
            # Only check reward termination if followed car is not disabled 
            if self.followed_car_index not in self.disabled_cars:
                # Check cumulative reward for followed car
                if hasattr(self, '_cumulative_rewards') and self.followed_car_index < len(self._cumulative_rewards):
                    followed_car_reward = self._cumulative_rewards[self.followed_car_index]
                    if followed_car_reward < TERMINATION_MIN_REWARD:
                        if self.reset_on_lap:
                            # In training mode, terminate on low rewards
                            self.termination_reason = f"low_reward_car_{self.followed_car_index}"
                            return True, False
                        else:
                            # In demo mode, just log the event but don't terminate
                            logger.debug(f"Low reward detected for car {self.followed_car_index} (cumulative: {followed_car_reward:.2f}) but demo mode active - not terminating")
                
                # Check if followed car is stuck (but don't terminate on collisions)
                speed = followed_car.get_velocity_magnitude()
                if speed < STUCK_SPEED_THRESHOLD and self.simulation_time > STUCK_TIME_THRESHOLD:
                    self.termination_reason = f"stuck_car_{self.followed_car_index}"
                    return True, False
        
        # Time-based termination
        if self.reset_on_lap and self.simulation_time > TERMINATION_MAX_TIME:
            self.termination_reason = "time_limit"
            return True, False
            
        if self.simulation_time > TRUNCATION_MAX_TIME:
            self.termination_reason = "truncated"
            return False, True
        
        return False, False
    
    def _get_multi_info(self):
        """Get info dictionaries for all cars"""
        infos = []
        
        for car_index in range(self.num_cars):
            info = {
                "simulation_time": self.simulation_time,
                "car_index": car_index,
                "num_cars": self.num_cars,
                "followed_car_index": self.followed_car_index,
                "termination_reason": self.termination_reason,
                "disabled": car_index in self.disabled_cars,
            }
            
            if car_index < len(self.cars) and self.cars[car_index]:
                car = self.cars[car_index]
                car_state = self.car_physics.get_car_state(car_index)
                
                if car_state:
                    info.update({
                        "car_position": (car_state[0], car_state[1]),
                        "car_speed_kmh": car.get_velocity_kmh(),
                        "car_speed_ms": car.get_velocity_magnitude(),
                        "on_track": self.car_physics.is_car_on_track(car_index),
                    })
                
                # Add performance info
                performance = car.validate_performance()
                info["performance"] = performance
                
                # Add lap timing info for this car
                if car_index < len(self.car_lap_timers):
                    lap_timer = self.car_lap_timers[car_index]
                    lap_timing = lap_timer.get_timing_info()
                    info["lap_timing"] = lap_timing
                
                # Add cumulative reward for this car
                if hasattr(self, '_cumulative_rewards') and car_index < len(self._cumulative_rewards):
                    info["cumulative_reward"] = self._cumulative_rewards[car_index]
            else:
                # Car doesn't exist
                info.update({
                    "car_position": (0.0, 0.0),
                    "car_speed_kmh": 0.0,
                    "car_speed_ms": 0.0,
                    "on_track": True,
                    "performance": {"performance_valid": False},
                    "lap_timing": {},
                    "cumulative_reward": 0.0
                })
            
            infos.append(info)
        
        # Add shared collision and physics info to all infos
        collision_stats = self.collision_reporter.get_collision_statistics()
        physics_stats = self.car_physics.get_performance_stats()
        
        for info in infos:
            info["collisions"] = collision_stats
            info["physics"] = physics_stats
        
        return infos
        
    def _track_input(self, throttle: float, brake: float, steering: float) -> None:
        """Track control inputs for stuck detection"""
        current_time = self.simulation_time
        
        # Add current input to history
        self.input_history.append((current_time, throttle, brake, steering))
        
        # Remove old inputs (keep only recent history)
        cutoff_time = current_time - STUCK_RECENT_INPUT_TIME
        self.input_history = [(t, th, br, st) for t, th, br, st in self.input_history if t >= cutoff_time]
        
        # Check if current input is meaningful
        input_magnitude = abs(throttle) + abs(brake) + abs(steering)
        if input_magnitude >= STUCK_INPUT_THRESHOLD:
            self.last_meaningful_input_time = current_time
        
    def _process_physics_collisions(self) -> None:
        """Process collision events from physics system"""
        # Get all collision data from physics system directly
        recent_collisions = self.car_physics.recent_collisions
        
        # Process each collision that hasn't been reported yet
        for collision in recent_collisions:
            if (hasattr(collision, 'car_id') and collision.car_id and 
                hasattr(collision, 'reported_to_environment') and 
                not collision.reported_to_environment):
                
                # Report collision with car identification
                self.collision_reporter.report_collision(
                    position=collision.position,
                    impulse=collision.impulse,
                    normal=collision.normal,
                    car_angle=0.0,  # We don't need car angle for this reporting
                    car_id=collision.car_id
                )
                
                # Mark as reported to avoid re-processing
                collision.reported_to_environment = True
                
    def _get_obs(self) -> np.ndarray:
        """Get current environment observation with normalized values (for followed car)"""
        if not self.cars or self.followed_car_index >= len(self.cars):
            # Return default normalized observation if no car
            static_load = CAR_MASS * GRAVITY_MS2 / 4.0
            normalized_static_load = static_load / NORM_MAX_TYRE_LOAD
            normalized_start_temp = TYRE_START_TEMPERATURE / NORM_MAX_TYRE_TEMP
            
            # Default normalized sensor distances (1.0 means max distance)
            default_normalized_sensor_distances = [1.0] * SENSOR_NUM_DIRECTIONS
            return np.array([
                0.0, 0.0,  # position (normalized)
                0.0, 0.0,  # velocity (normalized)
                0.0,       # speed magnitude (normalized)
                0.0, 0.0,  # orientation, angular velocity (normalized)
                normalized_static_load, normalized_static_load, 
                normalized_static_load, normalized_static_load,  # tyre loads (normalized)
                normalized_start_temp, normalized_start_temp,
                normalized_start_temp, normalized_start_temp,  # tyre temperatures (normalized)
                0.0, 0.0, 0.0, 0.0,  # tyre wear (normalized)
                0.0, 0.0,  # collision data (normalized)
                *default_normalized_sensor_distances  # 8 normalized sensor distances
            ], dtype=np.float32)
            
        # Get followed car state
        car_state = self.car_physics.get_car_state(self.followed_car_index)
        pos_x, pos_y, vel_x, vel_y, orientation, angular_vel = car_state
        
        # Normalize position to [-1, 1]
        norm_pos_x = np.clip(pos_x / NORM_MAX_POSITION, -1.0, 1.0)
        norm_pos_y = np.clip(pos_y / NORM_MAX_POSITION, -1.0, 1.0)
        
        # Normalize velocity to [-1, 1]
        norm_vel_x = np.clip(vel_x / NORM_MAX_VELOCITY, -1.0, 1.0)
        norm_vel_y = np.clip(vel_y / NORM_MAX_VELOCITY, -1.0, 1.0)
        
        # Calculate and normalize speed magnitude to [0, 1]
        speed_magnitude_ms = (vel_x**2 + vel_y**2)**0.5
        norm_speed_magnitude = np.clip(speed_magnitude_ms / NORM_MAX_VELOCITY, 0.0, 1.0)
        
        # Normalize orientation from [-π, π] to [-1, 1]
        norm_orientation = orientation / np.pi
        
        # Normalize angular velocity to [-1, 1]
        norm_angular_vel = np.clip(angular_vel / NORM_MAX_ANGULAR_VEL, -1.0, 1.0)
        
        # Get tyre data for followed car
        tyre_data = self.car_physics.get_tyre_data(self.followed_car_index)
        tyre_loads, tyre_temps, tyre_wear = tyre_data
        
        # Normalize tyre loads to [0, 1]
        norm_tyre_loads = [np.clip(load / NORM_MAX_TYRE_LOAD, 0.0, 1.0) for load in tyre_loads]
        
        # Normalize tyre temperatures to [0, 1]
        norm_tyre_temps = [np.clip(temp / NORM_MAX_TYRE_TEMP, 0.0, 1.0) for temp in tyre_temps]
        
        # Normalize tyre wear to [0, 1] (already 0-100, so divide by 100)
        norm_tyre_wear = [np.clip(wear / NORM_MAX_TYRE_WEAR, 0.0, 1.0) for wear in tyre_wear]
        
        # Get collision data (use main collision reporter for now)
        collision_impulse, collision_angle = self.collision_reporter.get_collision_for_observation()
        
        # Normalize collision impulse to [0, 1]
        norm_collision_impulse = np.clip(collision_impulse / NORM_MAX_COLLISION_IMPULSE, 0.0, 1.0)
        
        # Normalize collision angle from [-π, π] to [-1, 1]
        norm_collision_angle = collision_angle / np.pi
        
        # Get sensor distances
        world = self.car_physics.world if self.car_physics else None
        sensor_distances = self.distance_sensor.get_sensor_distances(
            world, (pos_x, pos_y), orientation
        )
        
        # Normalize sensor distances to range [0, 1] based on SENSOR_MAX_DISTANCE
        normalized_sensor_distances = [np.clip(d / SENSOR_MAX_DISTANCE, 0.0, 1.0) for d in sensor_distances]
        
        # Construct normalized observation vector
        observation = np.array([
            norm_pos_x, norm_pos_y,  # Normalized car position
            norm_vel_x, norm_vel_y,  # Normalized car velocity
            norm_speed_magnitude,  # Normalized car speed magnitude
            norm_orientation, norm_angular_vel,  # Normalized orientation and rotation
            norm_tyre_loads[0], norm_tyre_loads[1], norm_tyre_loads[2], norm_tyre_loads[3],  # Normalized tyre loads
            norm_tyre_temps[0], norm_tyre_temps[1], norm_tyre_temps[2], norm_tyre_temps[3],  # Normalized tyre temperatures
            norm_tyre_wear[0], norm_tyre_wear[1], norm_tyre_wear[2], norm_tyre_wear[3],  # Normalized tyre wear
            norm_collision_impulse, norm_collision_angle,  # Normalized collision data
            normalized_sensor_distances[0], normalized_sensor_distances[1], normalized_sensor_distances[2], normalized_sensor_distances[3],  # Normalized sensor distances 0-3
            normalized_sensor_distances[4], normalized_sensor_distances[5], normalized_sensor_distances[6], normalized_sensor_distances[7]   # Normalized sensor distances 4-7
        ], dtype=np.float32)
        
        return observation
        
    def _calculate_reward(self) -> float:
        """Calculate step reward (for followed car)"""
        if not self.cars or self.followed_car_index >= len(self.cars):
            return 0.0
            
        followed_car = self.cars[self.followed_car_index]
        if not followed_car:
            return 0.0
            
        reward = 0.0
        
        # Base reward for staying alive (penalty to encourage action)
        reward -= 0.0
        
        # Speed reward (encourage forward progress) - time-based
        speed = followed_car.get_velocity_magnitude()
        reward += speed**2 * REWARD_SPEED_MULTIPLIER * self.actual_dt  # Reward per second of speed
        
        # NEW: Penalty for being too slow
        # Penalty per second when speed is less than threshold
        if speed < PENALTY_LOW_SPEED_THRESHOLD:
            # Apply penalty scaled by physics timestep (1/60 second)
            reward -= PENALTY_LOW_SPEED_RATE * self.actual_dt
        
        # NEW: Reward for distance traveled (+0.1 per meter)
        if self._previous_car_position is not None:
            car_state = self.car_physics.get_car_state(self.followed_car_index)
            if car_state:
                current_position = (car_state[0], car_state[1])
                
                # Calculate distance traveled since last step
                dx = current_position[0] - self._previous_car_position[0]
                dy = current_position[1] - self._previous_car_position[1]
                distance = (dx**2 + dy**2)**0.5
                
                # Add reward for distance traveled
                reward += distance * REWARD_DISTANCE_MULTIPLIER
                
                # Update tracking
                self._total_distance_traveled += distance
                self._previous_car_position = current_position
        
        # CONTINUOUS collision penalty - applied every timestep while colliding
        # Get current collision impulse from physics system
        collision_impulse = self.car_physics.get_continuous_collision_impulse(self.followed_car_index)
        
        if collision_impulse > 0:
            # Determine severity based on impulse magnitude
            if collision_impulse < COLLISION_SEVERITY_MINOR:
                collision_penalty = PENALTY_COLLISION_MINOR
            elif collision_impulse < COLLISION_SEVERITY_MODERATE:
                collision_penalty = PENALTY_COLLISION_MODERATE
            elif collision_impulse < COLLISION_SEVERITY_SEVERE:
                collision_penalty = PENALTY_COLLISION_SEVERE
            elif collision_impulse < COLLISION_SEVERITY_EXTREME:
                collision_penalty = PENALTY_COLLISION_CRITICAL
            else:
                collision_penalty = PENALTY_COLLISION_EXTREME
            
            # For extreme collisions, apply full penalty (since this is single-car mode, car would terminate)
            # For other collisions, apply penalty per second
            if collision_impulse >= COLLISION_SEVERITY_EXTREME:
                penalty_applied = collision_penalty  # Full penalty for extreme collision
                #print(f"💰 EXTREME PENALTY: car={self.followed_car_index} impulse={collision_impulse:.1f} penalty={penalty_applied:.1f} (FULL EXTREME PENALTY)")
            else:
                penalty_applied = collision_penalty * self.actual_dt  # Normal per-second penalty
                #print(f"💰 REWARD PENALTY: car={self.followed_car_index} impulse={collision_impulse:.1f} penalty={penalty_applied:.3f} (rate={collision_penalty:.1f}/s)")
            
            reward -= penalty_applied
            
            
        # Performance bonus - time-based
        if speed > REWARD_HIGH_SPEED_THRESHOLD:
            reward += REWARD_HIGH_SPEED_BONUS * self.actual_dt  # Bonus per second at high speed
        
        # Forward sensor reward - reward for having clear space ahead (time-based)
        # Get normalized forward sensor distance from observation
        observation = self._get_obs()
        # Forward sensor is at index 22 (after 2 pos, 2 vel, 1 speed, 2 orient/angular, 
        # 4 tyre loads, 4 tyre temps, 4 tyre wear, 2 collision = 21 values, then sensor 0)
        forward_sensor_distance = observation[22]  # Already normalized to [0, 1]
        # Apply reward per second of having forward space
        reward += REWARD_FORWARD_SENSOR_MULTIPLIER * forward_sensor_distance * self.actual_dt
        
        # Lap completion bonus - major reward for completing laps (for followed car)
        followed_lap_timer = self.car_lap_timers[self.followed_car_index] if self.followed_car_index < len(self.car_lap_timers) else self.lap_timer
        lap_info = followed_lap_timer.get_timing_info()
        current_lap_count = lap_info.get("lap_count", 0)
        
        if current_lap_count > self._previous_lap_count:
            # Lap completed! Give substantial bonus
            laps_completed_this_step = current_lap_count - self._previous_lap_count
            lap_bonus = REWARD_LAP_COMPLETION * laps_completed_this_step
            reward += lap_bonus
            
            # Optional: Additional bonus for faster laps
            last_lap_time = lap_info.get("last_lap_time", None)
            if last_lap_time and last_lap_time < REWARD_FAST_LAP_TIME:
                reward += REWARD_FAST_LAP_BONUS
            
            self._previous_lap_count = current_lap_count
            
        return reward
        
    def _check_termination(self) -> Tuple[bool, bool]:
        """Check if episode should terminate"""
        terminated = False
        truncated = False
        
        if not self.cars or self.followed_car_index >= len(self.cars):
            self.termination_reason = "no_car"
            return True, False
            
        followed_car = self.cars[self.followed_car_index]
        if not followed_car:
            self.termination_reason = "no_followed_car"
            return True, False
            
        # Terminate if cumulative reward goes below threshold (only in training mode)
        # In demo mode (reset_on_lap=False), don't terminate on low rewards
        if self._cumulative_reward < TERMINATION_MIN_REWARD:
            if self.reset_on_lap:
                terminated = True
                self.termination_reason = f"low_reward (cumulative reward: {self._cumulative_reward:.2f})"
            else:
                # In demo mode, just log the event but don't terminate
                logger.debug(f"Low reward detected (cumulative: {self._cumulative_reward:.2f}) but demo mode active - not terminating")
            
        # Terminate on severe collision
        recent_collision = self.collision_reporter.get_recent_collision(TERMINATION_COLLISION_WINDOW)
        if recent_collision and recent_collision.severity in ["severe", "critical"]:
            terminated = True
            self.termination_reason = f"collision_{recent_collision.severity} (impulse: {recent_collision.impulse:.0f})"
            
        # Intelligent stuck detection (for followed car)
        speed = followed_car.get_velocity_magnitude()
        if speed < STUCK_SPEED_THRESHOLD and self.simulation_time > STUCK_TIME_THRESHOLD:
            # Check if there have been any meaningful inputs recently
            time_since_input = self.simulation_time - self.last_meaningful_input_time
            
            # Only consider stuck if no meaningful inputs for extended time
            # AND not currently braking (brake input is intentional stopping)
            current_input = self.input_history[-1] if self.input_history else (0, 0, 0, 0)
            _, _, current_brake, _ = current_input
            
            # Don't terminate if:
            # 1. Currently braking (intentional stop)
            # 2. Had meaningful input recently (within time window)
            # 3. Just started simulation (give some grace period)
            if (current_brake < STUCK_INPUT_THRESHOLD and 
                time_since_input > STUCK_RECENT_INPUT_TIME and
                self.simulation_time > STUCK_TIME_THRESHOLD):
                terminated = True
                self.termination_reason = f"stuck (speed: {speed:.2f} m/s for {time_since_input:.1f}s)"
            
        # Terminate if episode exceeds time limit (only when reset_on_lap=True for training)
        # When reset_on_lap=False (demo mode), allow unlimited time
        if self.reset_on_lap and self.simulation_time > TERMINATION_MAX_TIME:
            terminated = True
            if not self.termination_reason:  # Only set if no other reason already
                self.termination_reason = f"time_limit ({self.simulation_time:.1f}s > {TERMINATION_MAX_TIME}s)"
            
        # Truncate on hard time limit
        if self.simulation_time > TRUNCATION_MAX_TIME:
            truncated = True
            self.termination_reason = f"truncated (hard time limit {TRUNCATION_MAX_TIME}s)"
            
        return terminated, truncated
        
    def _update_episode_stats(self) -> None:
        """Update episode statistics (for followed car)"""
        if not self.cars or self.followed_car_index >= len(self.cars):
            return
            
        followed_car = self.cars[self.followed_car_index]
        if not followed_car:
            return
            
        speed = followed_car.get_velocity_magnitude()
        self.episode_stats["max_speed"] = max(self.episode_stats["max_speed"], speed)
        
        # Simple distance approximation
        self.episode_stats["distance_traveled"] += speed * self.actual_dt
        
        # Count collisions
        if self.collision_reporter.has_recent_collision(self.actual_dt):
            self.episode_stats["collisions"] += 1
            
        # Track time on track
        if self.car_physics.is_car_on_track(self.followed_car_index):
            self.episode_stats["time_on_track"] += self.actual_dt
            
    def _get_info(self) -> Dict[str, Any]:
        """Get environment info dictionary (for followed car)"""
        info = {
            "simulation_time": self.simulation_time,
            "episode_stats": self.episode_stats.copy(),
            "termination_reason": self.termination_reason,
            "cumulative_reward": self._cumulative_reward,
            "num_cars": self.num_cars,
            "followed_car_index": self.followed_car_index,
        }
        
        if self.cars and self.followed_car_index < len(self.cars):
            followed_car = self.cars[self.followed_car_index]
            if followed_car:
                # Add followed car-specific info
                car_state = self.car_physics.get_car_state(self.followed_car_index)
                if car_state:
                    info.update({
                        "car_position": (car_state[0], car_state[1]),
                        "car_speed_kmh": followed_car.get_velocity_kmh(),
                        "car_speed_ms": followed_car.get_velocity_magnitude(),
                        "on_track": self.car_physics.is_car_on_track(self.followed_car_index),
                    })
                
                # Add performance validation
                performance = followed_car.validate_performance()
                info["performance"] = performance
                self.episode_stats["performance_valid"] = performance["performance_valid"]
            
        # Add collision info
        collision_stats = self.collision_reporter.get_collision_statistics()
        info["collisions"] = collision_stats
        
        # Add physics performance
        physics_stats = self.car_physics.get_performance_stats()
        info["physics"] = physics_stats
        
        # Add lap timing info (for followed car)
        followed_lap_timer = self.car_lap_timers[self.followed_car_index] if self.followed_car_index < len(self.car_lap_timers) else self.lap_timer
        lap_timing = followed_lap_timer.get_timing_info()
        info["lap_timing"] = lap_timing
        
        return info
        
    def check_quit_requested(self) -> bool:
        """Check if user has requested to quit (e.g., by clicking window close button)"""
        if self.render_mode != RENDER_MODE_HUMAN:
            return False
            
        # Check if pygame is initialized before trying to get events
        if not pygame.get_init():
            return False
            
        # Check for pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    # Toggle reward display
                    self._show_reward = not self._show_reward
                    logger.info(f"Reward display {'enabled' if self._show_reward else 'disabled'}")
                elif event.key in CAR_SELECT_KEYS:
                    # Handle car switching (keys 0-9)
                    key_index = CAR_SELECT_KEYS.index(event.key)
                    if key_index < self.num_cars:
                        old_car_index = self.followed_car_index
                        self.followed_car_index = key_index
                        logger.info(f"Switched from Car {old_car_index} to Car {key_index}")
                        
                        # Update legacy car reference for backward compatibility
                        if self.cars and key_index < len(self.cars):
                            self.car = self.cars[key_index]
                    else:
                        logger.warning(f"Car {key_index} does not exist (only {self.num_cars} cars available)")
                else:
                    # Re-post other key events
                    pygame.event.post(event)
            else:
                # Re-post non-quit events so they can be handled by renderer
                pygame.event.post(event)
                
        return False
        
    def is_lap_reset_pending(self) -> bool:
        """Check if a lap reset is pending (useful for external code)"""
        return self._lap_reset_pending
        
    def render(self) -> None:
        """Render the environment"""
        if self.render_mode == RENDER_MODE_HUMAN and self.renderer:
            # Prepare multi-car data for rendering
            cars_data = []
            if self.cars:
                for i, car in enumerate(self.cars):
                    if car:
                        car_state = self.car_physics.get_car_state(i)
                        if car_state:
                            car_data = {
                                'position': (car_state[0], car_state[1]),
                                'angle': car_state[4],
                                'color': self.car_colors[i] if i < len(self.car_colors) else (255, 255, 255),
                                'name': self.car_names[i] if i < len(self.car_names) else f"Car {i}"
                            }
                            cars_data.append(car_data)
            
            # Get debug data for followed car only
            debug_data = None
            if self.cars and self.followed_car_index < len(self.cars):
                debug_data = self.get_debug_data()
                
            # Get lap timing info for followed car
            followed_lap_timer = self.car_lap_timers[self.followed_car_index] if self.followed_car_index < len(self.car_lap_timers) else self.lap_timer
            lap_timing_info = followed_lap_timer.get_timing_info()
            # Add car name to timing info
            lap_timing_info['car_name'] = self.car_names[self.followed_car_index] if self.followed_car_index < len(self.car_names) else f"Car {self.followed_car_index}"
            
            # Prepare reward info if display is enabled
            reward_info = None
            if self._show_reward:
                # Get reward for the currently followed car
                if self.num_cars == 1:
                    # Single car mode
                    current_reward = self._current_reward
                    cumulative_reward = self._cumulative_reward
                else:
                    # Multi-car mode - get followed car's rewards
                    current_reward = 0.0
                    cumulative_reward = 0.0
                    
                    # Get current reward from last step if available
                    if hasattr(self, '_last_rewards') and self._last_rewards is not None and self.followed_car_index < len(self._last_rewards):
                        current_reward = self._last_rewards[self.followed_car_index]
                    
                    # Get cumulative reward for followed car
                    if hasattr(self, '_cumulative_rewards') and self._cumulative_rewards is not None and self.followed_car_index < len(self._cumulative_rewards):
                        cumulative_reward = self._cumulative_rewards[self.followed_car_index]
                
                reward_info = {
                    'current_reward': current_reward,
                    'cumulative_reward': cumulative_reward,
                    'show': True
                }
            
            # For backward compatibility, also provide single car data
            followed_car_position = None
            followed_car_angle = None
            if cars_data and self.followed_car_index < len(cars_data):
                followed_car_data = cars_data[self.followed_car_index]
                followed_car_position = followed_car_data['position']
                followed_car_angle = followed_car_data['angle']
            
            # Get current action for followed car
            followed_car_action = None
            if hasattr(self, 'all_actions') and self.all_actions is not None and self.followed_car_index < len(self.all_actions):
                followed_car_action = self.all_actions[self.followed_car_index]
            elif hasattr(self, 'current_action'):
                followed_car_action = self.current_action
            
            # Calculate race data for tables display
            race_positions_data = self._calculate_race_positions()
            best_lap_times_data = self._get_best_lap_times_data()
            
            self.renderer.render_frame(
                car_position=followed_car_position, 
                car_angle=followed_car_angle, 
                debug_data=debug_data, 
                current_action=followed_car_action, 
                lap_timing_info=lap_timing_info,
                reward_info=reward_info,
                cars_data=cars_data,
                followed_car_index=self.followed_car_index,
                race_positions_data=race_positions_data,
                best_lap_times_data=best_lap_times_data
            )
            
    def get_track(self):
        """Get the loaded track (for compatibility)"""
        return self.track
        
    def is_position_on_track(self, position: Tuple[float, float]) -> bool:
        """Check if position is on track (for compatibility)"""
        return self.car_physics._is_position_on_track(position) if self.car_physics else True
        
    def close(self) -> None:
        """Clean up environment resources"""
        if self.renderer:
            self.renderer.close()
            
        if self.car_physics:
            self.car_physics.cleanup()
            
        # Clean up headless clock
        if self.headless_clock:
            self.headless_clock = None
            
        logger.info("CarEnv closed")
    
    def _calculate_race_positions(self) -> list:
        """
        Calculate current race positions based on completed laps and track progress.
        
        Returns:
            List of tuples (car_index, car_name, total_progress, completed_laps) sorted by race position (1st, 2nd, etc.)
        """
        if not self.cars or not self.track or not self.track.segments:
            return []
        
        car_positions = []
        track_length = self.track.get_total_track_length()
        
        for car_index, car in enumerate(self.cars):
            if not car or car_index in self.disabled_cars:
                continue
                
            car_state = self.car_physics.get_car_state(car_index)
            if not car_state:
                continue
                
            car_pos = (car_state[0], car_state[1])  # x, y position
            car_name = self.car_names[car_index] if car_index < len(self.car_names) else f"Car {car_index}"
            
            # Get completed laps for this car
            completed_laps = 0
            lap_timer = None
            if car_index < len(self.car_lap_timers):
                lap_timer = self.car_lap_timers[car_index]
                completed_laps = lap_timer.get_lap_count()
            
            # Calculate current track progress (distance along track from start)
            current_progress = self._calculate_track_progress(car_pos)
            
            # Virtual lap increment for position calculation: if car is near start line 
            # but has traveled significant distance, they likely just crossed finish
            virtual_laps = completed_laps
            if (lap_timer and 
                lap_timer.is_timing() and 
                lap_timer.has_crossed_startline and
                current_progress < (track_length * 0.15) and  # Within 15% of start
                lap_timer.total_distance_traveled > (track_length * 0.8)):  # Traveled 80%+ of track
                # Car likely just crossed finish line but validation pending
                virtual_laps = completed_laps + 1
            
            # Total progress = (virtual_laps * track_length) + current_progress
            # Use virtual_laps for sorting to handle finish line crossings
            total_progress = virtual_laps * track_length + current_progress
            
            # Store virtual lap count and current progress for proper sorting
            car_positions.append((car_index, car_name, total_progress, virtual_laps, current_progress))
        
        # Sort by completed laps first (descending), then by current progress (descending)
        # This ensures cars with more laps always rank higher, and among cars on the same lap,
        # those further along rank higher
        car_positions.sort(key=lambda x: (x[3], x[4]), reverse=True)
        
        return car_positions
    
    def _calculate_track_progress(self, car_position: Tuple[float, float]) -> float:
        """
        Calculate how far along the track a car is (distance from start).
        
        Args:
            car_position: Car position (x, y) in world coordinates
            
        Returns:
            Distance traveled along track from start in meters
        """
        if not self.track or not self.track.segments:
            return 0.0
        
        car_x, car_y = car_position
        min_distance = float('inf')
        closest_segment_index = 0
        closest_point_on_segment = (0.0, 0.0)
        
        # Find closest point on any track segment
        for i, segment in enumerate(self.track.segments):
            # Calculate closest point on this segment
            start_x, start_y = segment.start_position
            end_x, end_y = segment.end_position
            
            # Vector from start to end of segment
            dx = end_x - start_x
            dy = end_y - start_y
            segment_length_sq = dx * dx + dy * dy
            
            if segment_length_sq < 1e-6:  # Very short segment
                closest_x, closest_y = start_x, start_y
                t = 0.0
            else:
                # Project car position onto line segment
                t = max(0, min(1, ((car_x - start_x) * dx + (car_y - start_y) * dy) / segment_length_sq))
                closest_x = start_x + t * dx
                closest_y = start_y + t * dy
            
            # Calculate distance from car to this closest point
            dist_sq = (car_x - closest_x) ** 2 + (car_y - closest_y) ** 2
            
            if dist_sq < min_distance:
                min_distance = dist_sq
                closest_segment_index = i
                closest_point_on_segment = (closest_x, closest_y)
        
        # Calculate total distance from start to closest point
        total_progress = 0.0
        
        # Add lengths of all complete segments before the closest one
        for i in range(closest_segment_index):
            segment = self.track.segments[i]
            segment_length = math.sqrt(
                (segment.end_position[0] - segment.start_position[0]) ** 2 +
                (segment.end_position[1] - segment.start_position[1]) ** 2
            )
            total_progress += segment_length
        
        # Add partial distance within the closest segment
        if closest_segment_index < len(self.track.segments):
            closest_segment = self.track.segments[closest_segment_index]
            partial_distance = math.sqrt(
                (closest_point_on_segment[0] - closest_segment.start_position[0]) ** 2 +
                (closest_point_on_segment[1] - closest_segment.start_position[1]) ** 2
            )
            total_progress += partial_distance
        
        return total_progress
    
    def _get_best_lap_times_data(self) -> list:
        """
        Get best lap times for all cars sorted by fastest time.
        
        Returns:
            List of tuples (car_index, car_name, best_lap_time) sorted by fastest time
        """
        if not self.car_lap_timers:
            return []
        
        lap_times_data = []
        
        for car_index, lap_timer in enumerate(self.car_lap_timers):
            if car_index in self.disabled_cars:
                continue
                
            car_name = self.car_names[car_index] if car_index < len(self.car_names) else f"Car {car_index}"
            best_time = lap_timer.get_best_lap_time()
            
            if best_time is not None:
                lap_times_data.append((car_index, car_name, best_time))
        
        # Sort by best lap time (fastest first)
        lap_times_data.sort(key=lambda x: x[2])
        
        return lap_times_data
        
    def get_debug_info(self) -> str:
        """Get debug information string"""
        if not self.car:
            return "No car initialized"
            
        # Use car's debug string which includes engine info
        car_info = str(self.car)
        collision_info = self.collision_reporter.get_debug_info()
        
        return f"{car_info}, {collision_info}"
        
    def get_debug_data(self) -> dict:
        """Get structured debug data for enhanced debug visualization (for followed car)"""
        if not self.cars or self.followed_car_index >= len(self.cars):
            return {}
            
        followed_car = self.cars[self.followed_car_index]
        if not followed_car:
            return {}
            
        # Car physics data
        car_position = followed_car.body.position
        car_angle = followed_car.body.angle
        drag_force = followed_car.get_drag_force()
        velocity_vector = followed_car.get_velocity_vector()
        acceleration_vector = followed_car.get_acceleration_vector()
        input_steering_vector, actual_steering_vector = followed_car.get_steering_vectors()
        
        # Tyre data
        tyre_loads = followed_car.tyre_manager.get_tyre_loads()
        tyre_temperatures = followed_car.tyre_manager.get_tyre_temperatures()
        tyre_wear = followed_car.tyre_manager.get_tyre_wear()
        tyre_pressures = followed_car.tyre_manager.get_tyre_pressures()
        
        # Sensor data
        world = self.car_physics.world if self.car_physics else None
        sensor_distances = self.distance_sensor.get_sensor_distances(
            world, (car_position.x, car_position.y), car_angle
        )
        sensor_angles = self.distance_sensor.get_sensor_angles(car_angle)
        
        return {
            'car_position': (car_position.x, car_position.y),
            'car_angle': car_angle,
            'drag_force': drag_force,
            'velocity_vector': velocity_vector,
            'acceleration_vector': acceleration_vector,
            'steering_vectors': {
                'input': input_steering_vector,
                'actual': actual_steering_vector
            },
            'tyre_data': {
                'loads': tyre_loads,
                'temperatures': tyre_temperatures,
                'wear': tyre_wear,
                'pressures': tyre_pressures
            },
            'sensor_data': {
                'distances': sensor_distances,
                'angles': sensor_angles
            }
        }
        
    def __str__(self) -> str:
        """String representation of environment"""
        track_name = self.track_file if self.track_file else "No track"
        return (f"CarEnv: {track_name}, "
                f"Time: {self.simulation_time:.1f}s, "
                f"Car: {'Yes' if self.car else 'No'}")