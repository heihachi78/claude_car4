import numpy as np
import math


# Environment Constants  
INITIAL_ELAPSED_TIME = 0.0
DEFAULT_REWARD = 0.0
DEFAULT_TERMINATED = False
DEFAULT_TRUNCATED = False

# Collision Filtering Constants
COLLISION_CATEGORY_TRACK_WALLS = 0x0001   # Category bit for track walls
COLLISION_CATEGORY_CARS = 0x0002          # Category bit for cars
COLLISION_MASK_TRACK_WALLS = 0xFFFF       # Track walls collide with everything
COLLISION_MASK_CARS = 0x0001              # Cars only collide with track walls (not other cars)

# Rendering Constants
DEFAULT_RENDER_FPS = 60
UNLIMITED_FPS_CAP = 240  # FPS cap to use when enable_fps_limit=False (prevents physics timing issues)
DEFAULT_WINDOW_WIDTH = 1024
DEFAULT_WINDOW_HEIGHT = 768
DEFAULT_WINDOW_SIZE = (DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT)
RENDER_MODE_HUMAN = "human"

# Colors (RGB)
BACKGROUND_COLOR = (0, 100, 0)  # Green
FPS_COLOR_LOW = (255, 0, 0)     # Red
FPS_COLOR_NORMAL = (0, 255, 0)  # Green

# UI Constants
FONT_SIZE = 36
FPS_TEXT_TOP_MARGIN = 10
FPS_THRESHOLD = 60
WINDOW_CAPTION = "Simple Environment"

# Action Bar UI Constants
ACTION_BAR_WIDTH = 200  # pixels width of each action bar
ACTION_BAR_HEIGHT = 20  # pixels height of each action bar
ACTION_BAR_SPACING = 8  # pixels spacing between bars
ACTION_BAR_TOP_MARGIN = 50  # pixels from top of screen (below FPS)
ACTION_BAR_PADDING = 4  # pixels padding inside bars
ACTION_BAR_LABEL_FONT_SIZE = 18  # font size for action labels

# Action Bar Colors (RGB)
THROTTLE_BAR_COLOR = (0, 200, 0)  # Green for throttle
THROTTLE_BAR_BG_COLOR = (50, 100, 50)  # Dark green background
BRAKE_BAR_COLOR = (200, 0, 0)  # Red for brake  
BRAKE_BAR_BG_COLOR = (100, 50, 50)  # Dark red background
STEERING_BAR_COLOR = (0, 100, 200)  # Blue for steering
STEERING_BAR_BG_COLOR = (50, 50, 100)  # Dark blue background
ACTION_BAR_BORDER_COLOR = (255, 255, 255)  # White borders
ACTION_BAR_TEXT_COLOR = (255, 255, 255)  # White text

# Race Tables UI Constants
RACE_TABLES_FONT_SIZE = 20  # Font size for race tables
RACE_TABLES_WIDTH = 200  # Width of each table in pixels
RACE_TABLES_HEIGHT = 300  # Height of each table in pixels
RACE_TABLES_PADDING = 10  # Padding inside tables in pixels
RACE_TABLES_SPACING = 20  # Space between left and right tables in pixels
RACE_TABLES_LINE_HEIGHT = 22  # Height per text line in pixels

# Race Tables Colors (RGB)
RACE_TABLES_BG_COLOR = (0, 0, 0)  # Black background
RACE_TABLES_BG_ALPHA = 180  # Semi-transparent background
RACE_TABLES_TEXT_COLOR = (255, 255, 255)  # White text
RACE_TABLES_HEADER_COLOR = (255, 255, 0)  # Yellow headers
RACE_TABLES_POSITION_COLOR = (100, 255, 100)  # Green for positions
RACE_TABLES_LAP_TIME_COLOR = (255, 200, 100)  # Orange for lap times

# Track Constants
DEFAULT_TRACK_WIDTH = 20.0  # meters
DEFAULT_GRID_LENGTH = 100.0  # meters
STARTLINE_LENGTH = 5.0  # meters
FINISHLINE_LENGTH = 5.0  # meters

# Track Colors (RGB)
GRID_COLOR = (192, 192, 192)  # Light gray
TRACK_COLOR = (128, 128, 128)  # Dark gray
STARTLINE_COLOR = (255, 255, 0)  # Yellow
FINISHLINE_COLOR = (255, 255, 0)  # Yellow

# Physics Constants
PIXELS_PER_METER = 3  # Default conversion factor for rendering (now dynamic)
TRACK_WALL_THICKNESS = 1.0  # meters

# Box2D Physics Constants
BOX2D_WALL_DENSITY = 0.0  # Density for track wall fixtures
BOX2D_WALL_FRICTION = 0.8  # Friction for track wall fixtures  
BOX2D_WALL_RESTITUTION = 0.2  # Restitution (bounciness) for track wall fixtures
BOX2D_TIME_STEP = 1.0/DEFAULT_RENDER_FPS  # Physics simulation time step (auto-calculated)
BOX2D_VELOCITY_ITERATIONS = 6  # Box2D velocity constraint solver iterations
BOX2D_POSITION_ITERATIONS = 2  # Box2D position constraint solver iterations

# Collision Constants
COLLISION_FORCE_SCALE_FACTOR = 0.1  # Scale factor for collision force estimation (realistic car collision forces)
COLLISION_MIN_FORCE = 10.0  # Minimum collision force in Newtons
COLLISION_MAX_FORCE = 50000.0  # Maximum realistic collision force in Newtons for validation

# Performance Validation Constants
MIN_REALISTIC_FPS = 5.0  # Minimum realistic FPS value
MAX_REALISTIC_FPS = 300.0  # Maximum realistic FPS value

# Camera Constants
CAMERA_MARGIN_FACTOR = 0.1  # 10% margin around track when auto-fitting
MIN_ZOOM_FACTOR = 0.01  # Minimum pixels per meter (maximum zoom out) - allows very long tracks
MAX_ZOOM_FACTOR = 50.0  # Maximum pixels per meter (maximum zoom in)
FULLSCREEN_TOGGLE_KEY = 'f'  # Key to toggle fullscreen mode
TRACK_INFO_TOGGLE_KEY = 'i'  # Key to toggle track info display

# Camera Mode Constants
CAMERA_MODE_TOGGLE_KEY = 'c'  # Key to toggle camera mode (track view vs car follow)
CAR_FOLLOW_ZOOM_FACTOR = 6.0  # Pixels per meter when following car
CAMERA_MODE_TRACK_VIEW = "track_view"  # Show entire track (default mode)
CAMERA_MODE_CAR_FOLLOW = "car_follow"  # Follow car with zoom

# Track Info Display Constants
TRACK_INFO_TEXT_COLOR = (255, 255, 255)  # White text
TRACK_INFO_BG_COLOR = (0, 0, 0)  # Black background
TRACK_INFO_BG_ALPHA = 128  # Semi-transparent background
TRACK_INFO_PADDING = 20  # Padding around text in info box
TRACK_INFO_MARGIN = 20  # Margin from screen edge
TRACK_INFO_FONT_SIZE = 24  # Font size for track info (smaller than main FONT_SIZE)
DEFAULT_ZOOM = 1.0  # Default camera zoom level
DEFAULT_TRACK_WIDTH_FALLBACK = 100.0  # Default width when track has zero width
DEFAULT_TRACK_HEIGHT_FALLBACK = 100.0  # Default height when track has zero height

# Rendering/Display Constants  
DISPLAY_RESET_DELAY = 0.1  # Seconds to wait during display reset
WINDOW_CREATION_MAX_ATTEMPTS = 3  # Maximum attempts for robust window creation
WINDOW_CREATION_STEP_DELAY = 0.05  # Seconds between multi-step window creation
TEMPORARY_WINDOW_SIZE = (100, 100)  # Small window size for intermediate steps
MINIMUM_LINE_WIDTH = 1  # Minimum line width for drawing
SEGMENT_RECT_HEIGHT_DIVISOR = 10  # Divisor for zero-length segment height

# Centerline Generation Constants
CENTERLINE_DEFAULT_SPACING = 2.0  # Default target spacing between centerline points (meters)
CENTERLINE_MIN_CURVE_POINTS = 8  # Minimum number of points for curved segments
CENTERLINE_TIGHT_CURVE_THRESHOLD = 50.0  # Radius threshold for tight curve detection (meters)
CENTERLINE_ADAPTIVE_MIN_FACTOR = 1.0  # Minimum adaptive sampling factor
CENTERLINE_SMOOTHING_FACTOR = 0.1  # Default smoothing factor for centerline smoothing

# Track Boundary Constants  
BOUNDARY_SMOOTHING_MAX_ANGLE = 120.0  # Maximum angle before corner smoothing kicks in (degrees)
BOUNDARY_SMOOTHING_MAX_FACTOR = 0.3  # Maximum smoothing amount (30%)
BOUNDARY_POINTS_EQUAL_TOLERANCE = 1e-6  # Tolerance for point equality checks
BOUNDARY_MIN_POLYGON_AREA = 1.0  # Minimum polygon area for validation (square units)

# Track Polygon Rendering Constants
POLYGON_CENTERLINE_DEFAULT_SPACING = 3.5  # Default centerline spacing for polygon rendering (meters) - optimized for performance
POLYGON_POSITION_MATCHING_TOLERANCE = 5.0  # Tolerance for position matching (world units)
POLYGON_MIN_CENTERLINE_SPACING = 0.5  # Minimum allowed centerline spacing (meters)

# Quality Preset Constants
QUALITY_LOW_CENTERLINE_SPACING = 5.0  # Centerline spacing for low quality mode (meters)
QUALITY_MEDIUM_CENTERLINE_SPACING = 3.5  # Centerline spacing for medium quality mode (meters)
QUALITY_HIGH_CENTERLINE_SPACING = 1.5  # Centerline spacing for high quality mode (meters)
QUALITY_LOW_ANTIALIASING = False  # Anti-aliasing setting for low quality mode
QUALITY_HIGH_ANTIALIASING = True  # Anti-aliasing setting for high quality mode

# Debug Visualization Constants
DEBUG_TOGGLE_KEY = 'd'  # Key to toggle debug visualization
DEBUG_CENTERLINE_COLOR = (255, 0, 0)  # Red color for centerline
DEBUG_CENTERLINE_WIDTH = 1  # Thin line width for centerline

# Debug Info Panel Constants
DEBUG_INFO_PANEL_X = 20  # Left margin for debug info panel (pixels)
DEBUG_INFO_PANEL_Y = 120  # Top margin for debug info panel (pixels)
DEBUG_INFO_PANEL_WIDTH = 400  # Width of debug info panel (pixels)
DEBUG_INFO_PANEL_HEIGHT = 800  # Height of debug info panel (pixels)
DEBUG_INFO_PANEL_BG_COLOR = (0, 0, 0)  # Black background for debug panel
DEBUG_INFO_PANEL_BG_ALPHA = 200  # Semi-transparent background (0-255)
DEBUG_INFO_PANEL_TEXT_COLOR = (255, 255, 255)  # White text color
DEBUG_INFO_PANEL_PADDING = 12  # Inner padding for debug panel (pixels)
DEBUG_INFO_TEXT_FONT_SIZE = 11  # Font size for debug text
DEBUG_INFO_LINE_HEIGHT = 22  # Line spacing for debug text (pixels)

# Debug Vector Rendering Constants
DEBUG_VELOCITY_VECTOR_COLOR = (0, 255, 0)  # Green for velocity vector
DEBUG_ACCELERATION_VECTOR_COLOR = (255, 255, 0)  # Yellow for acceleration vector
DEBUG_STEERING_INPUT_VECTOR_COLOR = (0, 0, 255)  # Blue for steering input vector
DEBUG_STEERING_ACTUAL_VECTOR_COLOR = (255, 0, 255)  # Magenta for actual steering vector
DEBUG_VECTOR_VELOCITY_SCALE_FACTOR = 2.0  # Scale factor for velocity vectors (pixels per m/s)
DEBUG_VECTOR_ACCELERATION_SCALE_FACTOR = 10.0  # Scale factor for acceleration vectors (pixels per m/s²) - increased for visibility
DEBUG_VECTOR_STEERING_SCALE_FACTOR = 50.0  # Scale factor for steering vectors (pixels per unit) - increased for visibility
DEBUG_VECTOR_MAX_LENGTH = 80  # Maximum vector length to prevent screen overflow (pixels)
DEBUG_VECTOR_ARROW_WIDTH = 3  # Width of vector arrows (pixels)
DEBUG_VECTOR_ARROW_HEAD_SIZE = 8  # Size of vector arrow heads (pixels)
DEBUG_VECTOR_MIN_LENGTH = 5  # Minimum vector length for rendering (pixels)

# Distance Sensor Constants
SENSOR_NUM_DIRECTIONS = 8  # Number of sensor directions around the car
SENSOR_MAX_DISTANCE = 250.0  # Maximum sensor range in meters
SENSOR_ANGLE_STEP = 45.0  # Degrees between sensors (360/8)

# Distance Sensor Debug Constants  
DEBUG_SENSOR_VECTOR_COLOR = (255, 0, 255)  # Magenta color for distance sensor vectors
DEBUG_SENSOR_VECTOR_SCALE = 1.0  # Scale factor for sensor vector display (pixels per meter)
DEBUG_SENSOR_TEXT_COLOR = (255, 0, 255)  # Magenta color for sensor distance text

# Car rendering constants
CAR_COLOR = (255, 0, 0)  # Red color for car (legacy single-car)
CAR_OUTLINE_COLOR = (0, 0, 0)  # Black outline for car
CAR_VISUAL_LENGTH = 5.0  # Visual length in meters (same as CAR_LENGTH)
CAR_VISUAL_WIDTH = 2.0   # Visual width in meters (same as CAR_WIDTH)

# Multi-car constants
MAX_CARS = 10  # Maximum number of cars allowed
MULTI_CAR_COLORS = [
    (255, 0, 0),    # Red
    (0, 0, 255),    # Blue  
    (0, 255, 0),    # Green
    (255, 255, 0),  # Yellow
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Cyan
    (255, 128, 0),  # Orange
    (128, 0, 255),  # Purple
    (255, 192, 203), # Pink
    (128, 128, 128) # Gray
]

# Car switching key mappings (pygame keys)
import pygame
CAR_SELECT_KEYS = [
    pygame.K_0, pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4,
    pygame.K_5, pygame.K_6, pygame.K_7, pygame.K_8, pygame.K_9
]

# Logging Constants
DEFAULT_LOG_LEVEL = "INFO"  # Default logging level
DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"  # Default log format

# =============================================================================
# CAR PHYSICS CONSTANTS
# =============================================================================

# Car Specifications
CAR_MASS = 1500.0  # kg
CAR_HORSEPOWER = 670.0  # hp
CAR_MAX_TORQUE = 820.0  # Nm (increased to maintain performance after smooth transition fix)
CAR_MAX_SPEED_MPH = 200.0  # mph
CAR_ACCELERATION_0_100_KMH = 3.0  # seconds
CAR_WEIGHT_DISTRIBUTION_FRONT = 0.50  # 50% front
CAR_WEIGHT_DISTRIBUTION_REAR = 0.50  # 50% rear
CAR_DRAG_COEFFICIENT = 0.38  # dimensionless (high-performance aerodynamics)
CAR_DRIVE_TYPE = "RWD"  # Rear Wheel Drive

# Car Dimensions (in meters, converted from mm)
CAR_LENGTH = 5.042  # meters (5042 mm)
CAR_WIDTH = 1.996  # meters (1996 mm)
CAR_WHEELBASE = 2.794  # meters (2794 mm)

# Physics Conversion Factors
HP_TO_WATTS = 745.7  # 1 HP = 745.7 Watts
MPH_TO_MS = 0.44704  # 1 MPH = 0.44704 m/s
KMH_TO_MS = 0.277778  # 1 km/h = 0.277778 m/s
MS_TO_KMH = 3.6  # 1 m/s = 3.6 km/h

# Derived Car Constants
CAR_MAX_POWER = CAR_HORSEPOWER * HP_TO_WATTS  # Watts
CAR_MAX_SPEED_MS = CAR_MAX_SPEED_MPH * MPH_TO_MS  # m/s
CAR_TARGET_100KMH_MS = 100.0 * KMH_TO_MS  # 27.78 m/s (100 km/h in m/s)

# Car Physics Properties
CAR_FRONTAL_AREA = 2.5  # m² (estimated frontal area for drag calculation)
AIR_DENSITY = 1.225  # kg/m³ (air density at sea level)
CAR_ROLLING_RESISTANCE_COEFFICIENT = 0.015  # typical for racing tires on asphalt

# Car Box2D Physics Properties  
CAR_DENSITY = CAR_MASS / (CAR_LENGTH * CAR_WIDTH)  # kg/m²
CAR_FRICTION = 0.7  # friction coefficient with track
CAR_RESTITUTION = 0.1  # low restitution for realistic collisions
CAR_MOMENT_OF_INERTIA_FACTOR = 0.5  # factor for calculating rotational inertia (increased for realistic car body rotation)

# =============================================================================
# TYRE SYSTEM CONSTANTS
# =============================================================================

# Tyre Temperature Constants
TYRE_START_TEMPERATURE = 80.0  # °C
TYRE_IDEAL_TEMPERATURE_MIN = 85.0  # °C
TYRE_IDEAL_TEMPERATURE_MAX = 105.0  # °C
TYRE_OPTIMAL_TEMPERATURE = (TYRE_IDEAL_TEMPERATURE_MIN + TYRE_IDEAL_TEMPERATURE_MAX) / 2.0

# Tyre Grip Constants
TYRE_MAX_GRIP_COEFFICIENT = 1.4  # 1.4 maximum grip coefficient at ideal temperature (optimal baseline)
                                    # NOTE: This is the single source of truth for tire grip in the simulation
TYRE_MIN_GRIP_COEFFICIENT = 0.8  # 0.8 minimum grip coefficient when far from ideal (realistic degraded performance)
TYRE_GRIP_FALLOFF_RATE = 0.02  # grip reduction per degree outside ideal range

# Tyre Wear Constants
TYRE_MAX_WEAR = 100.0  # maximum wear percentage (100% = completely worn)
TYRE_IDEAL_WEAR_RATE = 0.012  # wear rate per second at ideal temperature (increased 6x for realistic racing wear)
TYRE_EXCESSIVE_WEAR_MULTIPLIER = 2.5  # wear multiplier when outside ideal temperature
TYRE_GRIP_WEAR_FACTOR = 0.5  # grip reduction factor when tyre is worn

# Speed-Dependent Wear Constants
TYRE_SPEED_WEAR_BASE_SPEED = 100.0  # km/h - speed above which wear increases
TYRE_SPEED_WEAR_REFERENCE_SPEED = 400.0  # km/h - reference speed for calculation  
TYRE_SPEED_WEAR_MULTIPLIER = 3.0  # multiplier factor for speed-based wear
TYRE_SPEED_WEAR_MAX_MULTIPLIER = 3.0  # maximum speed wear multiplier cap

# Cornering-Intensity Wear Constants
TYRE_CORNERING_WEAR_BASE_G = 2.0  # m/s² - lateral G above which cornering wear increases  
TYRE_CORNERING_WEAR_MULTIPLIER = 1.5  # multiplier for cornering-based wear
TYRE_CORNERING_WEAR_MAX_MULTIPLIER = 2.5  # maximum cornering wear multiplier

# Tyre Thermal Constants
TYRE_HEATING_RATE_FRICTION = 0.040  # temperature increase per unit of friction work (moderate increase)
TYRE_COOLING_RATE_AMBIENT = 0.01  # temperature decrease rate in ambient air at low speeds
TYRE_HIGH_SPEED_COOLING_REDUCTION = 0.5  # cooling reduction factor at high speeds (aerodynamic heating effect)
TYRE_MIN_COOLING_SPEED_THRESHOLD = 50.0  # m/s - speed above which cooling is reduced
TYRE_AERODYNAMIC_HEATING_FACTOR = 0.0002  # additional heating from aerodynamic friction (reduced)
AMBIENT_TEMPERATURE = 25.0  # °C ambient air temperature
TYRE_THERMAL_MASS = 125.0  # thermal mass factor for temperature changes

# Tyre Pressure Constants
TYRE_OPTIMAL_PRESSURE_PSI = 32.0  # PSI (pounds per square inch)
TYRE_MIN_PRESSURE_PSI = 20.0  # PSI minimum safe pressure
TYRE_MAX_PRESSURE_PSI = 50.0  # PSI maximum safe pressure
TYRE_PRESSURE_TEMPERATURE_FACTOR = 0.015  # Pressure change per degree Celsius (reduced)
TYRE_PRESSURE_LOAD_FACTOR = 0.00005  # Pressure change per Newton of load (halved)
MAX_TYRE_PRESSURE_INCREASE = 8.0  # Maximum pressure increase from base (PSI)

# =============================================================================
# CAR CONTROL CONSTANTS  
# =============================================================================

# Action Space Constants (for continuous control)
THROTTLE_MIN = 0.0  # minimum throttle input
THROTTLE_MAX = 1.0  # maximum throttle input
BRAKE_MIN = 0.0  # minimum brake input  
BRAKE_MAX = 1.0  # maximum brake input
STEERING_MIN = -1.0  # full left steering
STEERING_MAX = 1.0  # full right steering

# Control Response Constants
MAX_STEERING_ANGLE = 30.0  # degrees maximum wheel turn angle (increased for tighter turns)
ENGINE_RESPONSE_TIME = 0.1  # seconds for engine to respond to throttle
BRAKE_RESPONSE_TIME = 0.05  # seconds for brakes to respond

# Car Action Space Constants (Continuous Control)
CAR_ACTION_SHAPE = (3,)  # [throttle, brake, steering]
CAR_ACTION_LOW = np.array([THROTTLE_MIN, BRAKE_MIN, STEERING_MIN], dtype=np.float32)
CAR_ACTION_HIGH = np.array([THROTTLE_MAX, BRAKE_MAX, STEERING_MAX], dtype=np.float32)

# Car Observation Space Constants (Comprehensive Car State)
# Observation vector: [pos_x, pos_y, vel_x, vel_y, speed_magnitude, orientation, angular_vel, 
#                      tyre_load_fl, tyre_load_fr, tyre_load_rl, tyre_load_rr,
#                      tyre_temp_fl, tyre_temp_fr, tyre_temp_rl, tyre_temp_rr,
#                      tyre_wear_fl, tyre_wear_fr, tyre_wear_rl, tyre_wear_rr,
#                      collision_impulse, collision_angle_relative,
#                      sensor_dist_0, sensor_dist_1, ..., sensor_dist_7]
CAR_OBSERVATION_SHAPE = (29,)  # 29 total observation elements (21 + 8 sensor distances)

# Observation bounds
MAX_POSITION_VALUE = 10000.0  # maximum world position coordinate
MAX_VELOCITY_VALUE = 200.0    # maximum velocity (m/s, well above max car speed)  
MAX_ANGULAR_VELOCITY = 10.0   # maximum angular velocity (rad/s)
MAX_COLLISION_FORCE = 100000.0  # maximum collision force (N)

# =============================================================================
# OBSERVATION NORMALIZATION CONSTANTS
# =============================================================================

# Normalization factors for observation space
# These values are used to normalize observations to [-1, 1] or [0, 1] ranges
NORM_MAX_POSITION = MAX_POSITION_VALUE  # Use same as max position
NORM_MAX_VELOCITY = MAX_VELOCITY_VALUE  # Use same as max velocity
NORM_MAX_ANGULAR_VEL = MAX_ANGULAR_VELOCITY  # Use same as max angular velocity
NORM_MAX_TYRE_TEMP = 200.0  # Maximum realistic tyre temperature in Celsius
NORM_MAX_TYRE_WEAR = TYRE_MAX_WEAR  # 100.0 - maximum wear percentage
NORM_MAX_COLLISION_IMPULSE = MAX_COLLISION_FORCE  # Use same as max collision force
# Note: NORM_MAX_TYRE_LOAD will be defined after MAX_TYRE_LOAD is calculated below


# =============================================================================
# WEIGHT TRANSFER CONSTANTS
# =============================================================================

# Weight Transfer Physics
CAR_CENTER_OF_GRAVITY_HEIGHT = 0.35  # meters above ground (realistic for sports car, balanced between 0.15m and 0.50m)
WHEELBASE_FRONT = CAR_WHEELBASE * CAR_WEIGHT_DISTRIBUTION_FRONT  # distance from CG to front axle
WHEELBASE_REAR = CAR_WHEELBASE * CAR_WEIGHT_DISTRIBUTION_REAR  # distance from CG to rear axle
TRACK_WIDTH = CAR_WIDTH * 0.8  # effective track width for weight transfer calculation

# Effective Weight Transfer Factors (combines physics and rigidity)
# These combine the geometric factors (CoG height / distance) with suspension rigidity
EFFECTIVE_LONGITUDINAL_TRANSFER_FACTOR = 0.02  # (CoG_height / wheelbase) * longitudinal_rigidity (original)
EFFECTIVE_LATERAL_TRANSFER_FACTOR = 0.01      # (CoG_height / track_width) * lateral_rigidity (original)

# Weight Transfer Limits
MIN_TYRE_LOAD = 200.0  # Minimum load per tyre in Newtons (prevents complete unloading)
MAX_LONGITUDINAL_TRANSFER_RATIO = 0.95  # Maximum fraction of axle load that can transfer (95%)
MAX_LATERAL_TRANSFER_RATIO = 0.6  # Maximum fraction of axle load that can transfer laterally (60%)

# Aerodynamic downforce constants for high-speed physics
AERODYNAMIC_DOWNFORCE_COEFFICIENT = 0.12  # downforce coefficient for high speeds (realistic sports car)
AERODYNAMIC_DOWNFORCE_REAR_BIAS = 0.6  # 60% of downforce goes to rear axle (more balanced)
AERODYNAMIC_DOWNFORCE_SPEED_THRESHOLD = 50.0  # m/s - speed above which downforce becomes significant
MAX_DOWNFORCE_MULTIPLIER = 1.5  # Maximum downforce as multiple of car weight (realistic limit)

# Cornering Heating Constants
CORNERING_OUTER_TYRE_HEATING_FACTOR = 1.5  # Outer tyres heat more during cornering
CORNERING_INNER_TYRE_HEATING_FACTOR = 0.5  # Inner tyres heat less during cornering

# Acceleration Limits (realistic vehicle dynamics)
MAX_LONGITUDINAL_ACCELERATION = 12.0  # m/s² maximum acceleration/braking (0.8g - realistic road car limit)
MAX_LATERAL_ACCELERATION = 12.0  # m/s² maximum cornering acceleration (1.2g - original limit)
ACCELERATION_SANITY_CHECK_THRESHOLD = 14.0  # m/s² threshold for detecting unrealistic accelerations
ACCELERATION_SANITY_DAMPENING = 0.5  # Dampening factor applied when acceleration exceeds sanity threshold
ACCELERATION_HISTORY_SIZE = 10  # number of acceleration samples to average for smoothing

# =============================================================================
# COLLISION DETECTION CONSTANTS
# =============================================================================

# Car Collision Properties
CAR_COLLISION_SHAPES = 4  # number of collision shapes for car (corners)
CAR_COLLISION_RADIUS = 0.3  # radius of corner collision shapes in meters
COLLISION_FORCE_THRESHOLD = 50.0  # minimum force to register as significant collision (increased to ignore light touches)
COLLISION_DAMAGE_FACTOR = 0.001  # damage per unit collision force

# Collision Severity Thresholds (impulse in N·s)
COLLISION_SEVERITY_MINOR = 500.0  # Below this is minor collision
COLLISION_SEVERITY_MODERATE = 1000.0  # Below this is moderate collision
COLLISION_SEVERITY_SEVERE = 2000.0  # Below this is severe collision
COLLISION_SEVERITY_EXTREME = 5000.0  # Above this is extreme collision (immediate disabling)
# Above SEVERE but below EXTREME is critical collision

# Collision Reporting
MAX_COLLISION_HISTORY = 10  # maximum number of recent collisions to track
COLLISION_COOLDOWN_TIME = 0.1  # seconds between collision reports for same contact

# =============================================================================
# ENGINE AND DRIVETRAIN CONSTANTS
# =============================================================================

# Engine RPM Constants
ENGINE_IDLE_RPM = 1000.0  # RPM at idle
ENGINE_PEAK_TORQUE_RPM = 5500.0  # RPM at peak torque output
ENGINE_MAX_RPM = 9000.0  # Maximum RPM (redline)
ENGINE_MIN_RPM = 600.0  # Minimum stable RPM
ENGINE_MAX_RPM_LIMIT = 9500.0  # Hard RPM limit
ENGINE_REDLINE_RPM_RANGE = 800.0  # RPM range from idle to redline
ENGINE_RPM_RESPONSE_RATE = 3000.0  # RPM per second response rate
ENGINE_RPM_RESPONSE_EPSILON = 0.1  # Small value to prevent division by zero

# Torque Curve Constants
ENGINE_TORQUE_CURVE_LOW_FACTOR = 0.7  # Torque factor at idle RPM
ENGINE_TORQUE_CURVE_HIGH_FACTOR = 0.3  # Additional torque factor at peak RPM
ENGINE_TORQUE_CURVE_FALLOFF_FACTOR = 0.6  # Torque falloff factor after peak RPM

# Drivetrain Constants
FINAL_DRIVE_RATIO = 7.5  # Final drive ratio (gearbox + differential)
WHEEL_RADIUS = 0.35  # Wheel radius in meters
WHEEL_CIRCUMFERENCE_TO_RPM = 60.0  # Conversion factor for wheel RPM calculation

# Power Limiting Constants  
POWER_LIMIT_TRANSITION_START = 12.0  # Speed (m/s) where power limiting begins to blend (delayed transition)
POWER_LIMIT_TRANSITION_END = 25.0  # Speed (m/s) where power limiting fully active (much higher threshold)
POWER_LIMIT_TRANSITION_RANGE = POWER_LIMIT_TRANSITION_END - POWER_LIMIT_TRANSITION_START

# Smooth Transition Curve Constants
POWER_TRANSITION_CURVE_SHARPNESS = 2.0  # Exponential curve sharpness (gentler transition)
POWER_TRANSITION_MIN_BLEND = 0.05  # Minimum blend factor to avoid division issues
POWER_TRANSITION_MAX_BLEND = 0.75  # Allow more torque contribution at high speeds

# =============================================================================
# FORCE AND PHYSICS CONSTANTS
# =============================================================================

# Force Application Constants
# RWD_GRIP_FACTOR removed - rely directly on tire physics model for grip calculation

# Friction Circle Constants (grip sharing between longitudinal and lateral forces)
FRICTION_CIRCLE_STEERING_REDUCTION_MAX = 0.4  # Maximum grip reduction when steering (40% reduction at full lock - racing tires)
FRICTION_CIRCLE_STEERING_FACTOR = 1.5  # Multiplier for steering angle effect on grip (reduced for racing tires)
FRICTION_CIRCLE_BRAKE_REDUCTION_MAX = 0.3  # Maximum brake force reduction when steering (30% reduction - racing tires)
FRICTION_CIRCLE_BRAKE_FACTOR = 1.5  # Multiplier for steering angle effect on braking
GRAVITY_MS2 = 9.81  # Gravitational acceleration (m/s²)
MAX_TYRE_LOAD = CAR_MASS * GRAVITY_MS2 * 2.0  # maximum load on single tyre (2x static)
STATIC_LOAD_PER_TYRE = CAR_MASS * GRAVITY_MS2 / 4.0  # static load per tyre with equal distribution (N)
ROLLING_RESISTANCE_FORCE = CAR_ROLLING_RESISTANCE_COEFFICIENT * CAR_MASS * GRAVITY_MS2  # total rolling resistance force (N)

# Add normalization constant for tyre load now that MAX_TYRE_LOAD is defined
NORM_MAX_TYRE_LOAD = MAX_TYRE_LOAD  # Use same as max tyre load

# Normalized Observation Space Arrays
# All values are normalized to [-1, 1] or [0, 1] ranges for better neural network training
CAR_OBSERVATION_LOW = np.array([
    -1.0, -1.0,         # pos_x, pos_y (normalized to [-1, 1])
    -1.0, -1.0,         # vel_x, vel_y (normalized to [-1, 1])
    0.0,                # speed_magnitude (normalized to [0, 1])
    -1.0, -1.0,         # orientation, angular_vel (normalized to [-1, 1])
    0.0, 0.0, 0.0, 0.0, # tyre loads (normalized to [0, 1])
    0.0, 0.0, 0.0, 0.0, # tyre temperatures (normalized to [0, 1])
    0.0, 0.0, 0.0, 0.0, # tyre wear (normalized to [0, 1])
    0.0, -1.0,          # collision impulse (normalized to [0, 1]), angle (normalized to [-1, 1])
    0.0, 0.0, 0.0, 0.0, # sensor distances (normalized to [0, 1])
    0.0, 0.0, 0.0, 0.0  # sensor distances (normalized to [0, 1])
], dtype=np.float32)

CAR_OBSERVATION_HIGH = np.array([
    1.0, 1.0,           # pos_x, pos_y (normalized to [-1, 1])
    1.0, 1.0,           # vel_x, vel_y (normalized to [-1, 1])
    1.0,                # speed_magnitude (normalized to [0, 1])
    1.0, 1.0,           # orientation, angular_vel (normalized to [-1, 1])
    1.0, 1.0, 1.0, 1.0, # tyre loads (normalized to [0, 1])
    1.0, 1.0, 1.0, 1.0, # tyre temperatures (normalized to [0, 1])
    1.0, 1.0, 1.0, 1.0, # tyre wear (normalized to [0, 1])
    1.0, 1.0,           # collision impulse (normalized to [0, 1]), angle (normalized to [-1, 1])
    1.0, 1.0, 1.0, 1.0, # sensor distances (normalized to [0, 1])
    1.0, 1.0, 1.0, 1.0  # sensor distances (normalized to [0, 1])
], dtype=np.float32)

# Braking Constants
MAX_BRAKE_DECELERATION_G = 14.0  # Maximum braking deceleration in m/s² (~1.4g)
BRAKE_FORCE_DISTRIBUTION_WHEELS = 4.0  # Number of wheels for force distribution
BRAKE_FRICTION_SPEED_THRESHOLD = 1.0  # Speed (m/s) below which brake friction for heating is reduced
BRAKE_FRICTION_MIN_SPEED_FACTOR = 0.05  # Minimum fraction of brake friction when stationary (for residual heating)

# Speed and Control Thresholds
MINIMUM_SPEED_FOR_DRAG = 0.1  # Minimum speed (m/s) to apply aerodynamic drag
MINIMUM_SPEED_FOR_BRAKE = 0.1  # Minimum speed (m/s) to apply braking force
MINIMUM_SPEED_FOR_STEERING = 0.5  # Minimum speed (m/s) to apply steering forces
MINIMUM_THROTTLE_THRESHOLD = 0.01  # Minimum throttle to consider "on throttle"
MINIMUM_BRAKE_THRESHOLD = 0.01  # Minimum brake to consider "braking"
MINIMUM_STEERING_THRESHOLD = 0.01  # Minimum steering angle to consider "steering"

# Steering Force Constants
STEERING_TORQUE_MULTIPLIER = 0.8  # Steering torque multiplier factor (reduced for smoother turns)
STEERING_ANGULAR_DAMPING = 4.0  # Damping factor for angular velocity (increased to reduce over-rotation)
LATERAL_FORCE_SPEED_THRESHOLD = 0.2  # Minimum speed (m/s) for lateral tyre forces (reduced for low-speed maneuverability)
MAX_LATERAL_FORCE = 15000.0  # Maximum lateral force from steering (N) - reduced for realistic grip
LATERAL_FORCE_SPEED_MULTIPLIER = 40.0  # Multiplier for speed-dependent lateral force (increased multiplier)
LATERAL_FORCE_STEERING_MULTIPLIER = 10.0  # Additional multiplier for steering force calculation (increased multiplier)
VELOCITY_ALIGNMENT_FORCE_FACTOR = 2.5  # Force multiplier for aligning velocity with car orientation (reduced for realistic slip and stability)
FRONT_TYRE_LATERAL_FACTOR = 0.3  # Front tyre lateral force factor
REAR_TYRE_LATERAL_FACTOR = 0.2  # Rear tyre lateral force factor

# Friction Force Constants (for tyre heating)
MAX_FRICTION_FORCE_CAP = 2000.0  # Maximum friction force per tyre (N)
REAR_WHEEL_COUNT = 2.0  # Number of rear wheels for RWD force distribution

# =============================================================================
# PERFORMANCE VALIDATION CONSTANTS  
# =============================================================================

# Performance History
VELOCITY_HISTORY_SECONDS = 10  # Seconds of velocity history to keep
VELOCITY_HISTORY_SIZE = 600  # History size at 60 FPS (10 seconds * 60 FPS)
PERFORMANCE_VALIDATION_MIN_SAMPLES = 10  # Minimum samples for performance validation
PERFORMANCE_SPEED_TOLERANCE = 0.95  # Speed tolerance (95% of target)
PERFORMANCE_TIME_TOLERANCE = 1.1  # Time tolerance (110% of target)

# Coordinate System Constants
WORLD_FORWARD_VECTOR = (1.0, 0.0)  # Forward direction in world coordinates  
WORLD_RIGHT_VECTOR = (0.0, 1.0)  # Right direction in world coordinates
COORDINATE_ZERO = (0.0, 0.0)  # Zero coordinate pair
BODY_CENTER_OFFSET = (0.0, 0.0)  # Center of mass offset from body center
RESET_ANGULAR_VELOCITY = 0.0  # Angular velocity value for reset
AERODYNAMIC_DRAG_FACTOR = 0.5  # Aerodynamic drag equation factor (½ in ½ρCdAv²)

# =============================================================================
# SLIP ANGLE HEATING CONSTANTS
# =============================================================================

# Slip Angle Detection and Heating
SLIP_ANGLE_THRESHOLD_DEGREES = 5.0  # Degrees - minimum slip angle to consider for heating
SLIP_ANGLE_HEATING_BASE_MULTIPLIER = 2.2  # Base heating multiplier for slip angle
SLIP_ANGLE_HEATING_EXPONENTIAL_FACTOR = 0.04  # Exponential factor for slip angle heating (per degree)
SLIP_ANGLE_MAX_HEATING_MULTIPLIER = 8.0  # Maximum heating multiplier from slip angle
SLIP_ANGLE_SPEED_THRESHOLD = 2.0  # m/s - minimum speed for slip angle heating

# Lateral Force Heating Constants
LATERAL_FORCE_HEATING_FACTOR = 0.05  # Heating factor per Newton of lateral force (reduced for realistic temps)
LATERAL_FORCE_DISTRIBUTION_FRONT = 0.5  # 50% of lateral heating goes to front tyres (balanced)
LATERAL_FORCE_DISTRIBUTION_REAR = 0.5  # 50% of lateral heating goes to rear tyres (balanced)
MAX_LATERAL_HEATING_PER_TYRE = 25.0  # Maximum additional heating per tyre from lateral forces (°C/s) - increased

# Cornering Load Transfer Heating
CORNERING_LOAD_HEATING_FACTOR = 0.001  # Additional heating factor for loaded outer tyres during cornering (increased 25x)
CORNERING_SPEED_THRESHOLD = 3.0  # m/s - minimum speed for cornering load heating effects

# =============================================================================
# STUCK DETECTION CONSTANTS
# =============================================================================

# Stuck Detection Thresholds
STUCK_SPEED_THRESHOLD = 0.5  # m/s - speed below which car might be considered stuck
STUCK_TIME_THRESHOLD = 10.0  # seconds - time with low speed and no inputs before stuck
STUCK_INPUT_THRESHOLD = 0.1  # minimum input magnitude to consider as "active input"
STUCK_RECENT_INPUT_TIME = 3.0  # seconds - time window to check for recent meaningful inputs

# =============================================================================
# PRE-COMPUTED MATHEMATICAL CONSTANTS (eliminates redundant runtime calculations)
# =============================================================================

# Pre-computed drag calculation constant (eliminates 4 multiplications per physics step)
DRAG_CONSTANT = AERODYNAMIC_DRAG_FACTOR * AIR_DENSITY * CAR_DRAG_COEFFICIENT * CAR_FRONTAL_AREA

# Pre-computed wheel circumference (eliminates trigonometric calculation)
WHEEL_CIRCUMFERENCE = 2 * math.pi * WHEEL_RADIUS

# Pre-computed moment of inertia (eliminates complex calculation)
CAR_MOMENT_OF_INERTIA = CAR_MASS * (CAR_LENGTH**2 + CAR_WIDTH**2) * CAR_MOMENT_OF_INERTIA_FACTOR / 12.0

# Pre-computed angle constants (eliminates repeated trigonometric calculations)
PERPENDICULAR_ANGLE_OFFSET = math.pi / 2  # 90 degrees in radians
TWO_PI = 2 * math.pi  # Full circle in radians

# =============================================================================
# REWARD STRUCTURE CONSTANTS
# =============================================================================

# Positive Rewards
REWARD_SPEED_MULTIPLIER = 0.004  # Bonus per m/s of speed
REWARD_DISTANCE_MULTIPLIER = 0.1  # Bonus per meter traveled
REWARD_HIGH_SPEED_THRESHOLD = 83.0  # Speed threshold (m/s) for performance bonus (~300 km/h)
REWARD_HIGH_SPEED_BONUS = 1.0  # Bonus when exceeding high speed threshold
REWARD_LAP_COMPLETION = 50.0  # Bonus per completed lap
REWARD_FAST_LAP_TIME = 40.0  # Time threshold (seconds) for fast lap bonus
REWARD_FAST_LAP_BONUS = 50.0  # Bonus for completing lap under threshold time
REWARD_FORWARD_SENSOR_MULTIPLIER = 5  # Bonus per normalized forward sensor distance

# Negative Rewards (Penalties)
PENALTY_LOW_SPEED_THRESHOLD = 0.278  # Speed threshold (m/s) for low speed penalty (1 km/h)
PENALTY_LOW_SPEED_RATE = 0.05  # Penalty per second when below low speed threshold
# Scaled collision penalties based on severity
PENALTY_COLLISION_MINOR = 25  # Light scrape penalty
PENALTY_COLLISION_MODERATE = 50  # Moderate hit penalty
PENALTY_COLLISION_SEVERE = 125  # Severe crash penalty
PENALTY_COLLISION_CRITICAL = 250.0  # Critical crash penalty
PENALTY_COLLISION_EXTREME = 1000.0  # Extreme crash penalty (immediate disabling)

# =============================================================================
# TERMINATION CONDITION CONSTANTS
# =============================================================================

# Termination Thresholds
TERMINATION_MIN_REWARD = -100.0  # Terminate if cumulative reward drops below this
TERMINATION_MAX_TIME = 60.0  # Terminate after this many seconds
TRUNCATION_MAX_TIME = 600.0  # Truncate (hard limit) after this many seconds
TERMINATION_COLLISION_WINDOW = 1.0  # Time window (seconds) to check for recent severe collision

# Physical constants (eliminates magic numbers)
TYRES_PER_AXLE = 2.0  # Number of tyres per axle
MIN_TYRE_LOAD_CONSTRAINT = 50.0  # Minimum load to prevent complete tyre unloading (Newtons)

# =============================================================================
# LAP TIMING CONSTANTS
# =============================================================================

# Lap Timer Display Constants
LAP_TIMER_FONT_SIZE = 28  # Font size for lap time display
LAP_TIMER_BOTTOM_MARGIN = 15  # Pixels from bottom of screen
LAP_TIMER_LINE_SPACING = 25  # Pixels between timer lines
LAP_TIMER_LABEL_WIDTH = 80  # Pixels for label width (Current:, Last:, Best:)

# Lap Timer Colors (RGB)
LAP_TIMER_CURRENT_COLOR = (255, 255, 255)  # White for current lap time
LAP_TIMER_LAST_COLOR = (255, 255, 100)     # Yellow for last lap time
LAP_TIMER_BEST_COLOR = (100, 255, 100)     # Green for best lap time
LAP_TIMER_BG_COLOR = (0, 0, 0)             # Black background
LAP_TIMER_BG_ALPHA = 180                   # Semi-transparent background

# Lap Detection Constants
LAP_DETECTION_POSITION_TOLERANCE = 2.0  # Meters tolerance for position-based detection