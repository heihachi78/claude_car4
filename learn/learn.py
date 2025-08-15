import sys
import os
import numpy as np

from stable_baselines3 import TD3
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import NormalActionNoise

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.car_env import CarEnv


class SaveModelCallback(BaseCallback):
    """
    Callback for saving the model every n steps.
    """
    def __init__(self, save_freq: int = 250000, save_path: str = "./learn/checkpoints", verbose: int = 0):
        super(SaveModelCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.save_count = 0
        
        # Create save directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)
        
    def _on_step(self) -> bool:
        # Check if we should save the model
        if self.n_calls % self.save_freq == 0:
            self.save_count += 1
            model_path = os.path.join(self.save_path, f"model_{self.num_timesteps}_steps")
            self.model.save(model_path)
            self.model.save_replay_buffer(f"./learn/checkpoints/replay_buffer_{self.num_timesteps}.pkl")
            
            if self.verbose > 0:
                print(f"Saved model checkpoint to {model_path} at step {self.num_timesteps}")
                
        return True  # Continue training


# Noise parameters for TD3 exploration
# These values are tuned for car control where actions are in range [-1, 1] or [0, 1]
EXPLORATION_NOISE_STD = 0.1  # Standard deviation for exploration noise (adds randomness to actions)
TARGET_POLICY_NOISE_STD = 0.15  # Standard deviation for target policy noise (smooths Q-learning targets)
TARGET_POLICY_NOISE_CLIP = 0.25  # Clip value for target policy noise (prevents extreme noise values)

# Create log dir for Monitor
log_dir = "./logs/"
os.makedirs(log_dir, exist_ok=True)

# Create environment factory function for vectorization
def make_env():
    env = CarEnv(
        render_mode=None,
        track_file="tracks/nascar.track",
        discrete_action_space=False,
        enable_fps_limit=False,
        reset_on_lap=True
    )
    env = Monitor(env, filename=os.path.join(log_dir, "monitor.csv"))  # tracks rewards
    return env

# Create vectorized environment
env = DummyVecEnv([make_env])

# Get action space dimensions for noise creation
n_actions = env.action_space.shape[-1]

# Create action noise for exploration
action_noise = NormalActionNoise(
    mean=np.zeros(n_actions), 
    sigma=EXPLORATION_NOISE_STD * np.ones(n_actions)
)

# Try to load existing model
try:
    model = TD3.load("./learn/checkpoints/XXXXXXXXXXXXX")
    model.set_env(env)
    if os.path.exists("./learn/checkpoints/replay_buffer.pkl"):
        model.load_replay_buffer("./learn/checkpoints/replay_buffer.pkl")
    print("Loaded existing model for continued training")
except:
    model = TD3(
        "MlpPolicy", env,
        learning_rate=3e-5,
        buffer_size=1_000_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=4,
        action_noise=action_noise,  # Add exploration noise
        target_policy_noise=TARGET_POLICY_NOISE_STD,  # Target policy smoothing noise
        target_noise_clip=TARGET_POLICY_NOISE_CLIP,  # Clip target policy noise
        verbose=1,
        tensorboard_log='./tensorboard/',
        learning_starts = 500_000,
        policy_kwargs = dict(net_arch=[1024, 1024])
    )
    print("Created new model")

# Create callback for saving model every 250,000 steps
save_callback = SaveModelCallback(save_freq=250000, save_path="./learn/checkpoints", verbose=1)

# Training loop with callback
# Train for 25 million steps total (100 x 250,000)
model.learn(
    total_timesteps=25_000_000, 
    tb_log_name="run_1",
    log_interval=1,
    callback=save_callback
)

# Save final model
model.save("./learn/final_model")
