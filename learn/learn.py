import sys
import os
import numpy as np
import math

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
        if self.n_calls % self.save_freq == 0:
            self.save_count += 1
            model_path = os.path.join(self.save_path, f"model_{self.num_timesteps}_steps")
            self.model.save(model_path)
            self.model.save_replay_buffer(f"./learn/checkpoints/replay_buffer_{self.num_timesteps}.pkl")
            
            if self.verbose > 0:
                print(f"Saved model checkpoint to {model_path} at step {self.num_timesteps}")
                
        return True


class DecayNoiseCallback(BaseCallback):
    """
    Callback to decay both action noise and target policy noise every fixed number of steps.
    """
    def __init__(
        self,
        decay_freq: int,
        decay_factor: float,
        min_action_sigma: float,
        min_target_policy_noise: float,
        min_target_noise_clip: float,
        verbose: int = 0
    ):
        super().__init__(verbose)
        self.decay_freq = decay_freq
        self.decay_factor = decay_factor
        self.min_action_sigma = min_action_sigma
        self.min_target_policy_noise = min_target_policy_noise
        self.min_target_noise_clip = min_target_noise_clip

    def _on_step(self) -> bool:
        if self.n_calls % self.decay_freq == 0:
            # Decay exploration noise
            current_sigma = self.model.action_noise.sigma
            new_sigma = np.maximum(current_sigma * self.decay_factor, self.min_action_sigma)
            self.model.action_noise.sigma = new_sigma

            # Decay target policy noise
            current_target_noise = self.model.target_policy_noise
            new_target_noise = max(current_target_noise * self.decay_factor, self.min_target_policy_noise)
            self.model.target_policy_noise = new_target_noise

            # Decay target noise clip
            current_clip = self.model.target_noise_clip
            new_clip = max(current_clip * self.decay_factor, self.min_target_noise_clip)
            self.model.target_noise_clip = new_clip

            if self.verbose > 0:
                print(f"[Noise Decay] Step {self.num_timesteps}: "
                      f"action_noise.sigma={new_sigma}, "
                      f"target_policy_noise={new_target_noise}, "
                      f"target_noise_clip={new_clip}")
        return True


# Noise parameters
EXPLORATION_NOISE_STD = 0.5
TARGET_POLICY_NOISE_STD = 0.3
TARGET_POLICY_NOISE_CLIP = 0.5

# Create log dir
log_dir = "./logs/"
os.makedirs(log_dir, exist_ok=True)

def make_env(rank):
    def _init():
        env = CarEnv(
            render_mode=None,
            track_file="tracks/nascar.track",
            discrete_action_space=False,
            enable_fps_limit=False,
            reset_on_lap=True
        )
        return Monitor(env, filename=os.path.join(log_dir, f"monitor_{rank}.csv"))
    return _init

# Create 8 environments
env = DummyVecEnv([make_env(i) for i in range(8)])

# Action noise needs shape according to number of actions in a single env
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(
    mean=np.zeros(n_actions),
    sigma=EXPLORATION_NOISE_STD * np.ones(n_actions)
)

def linear_schedule(initial_value, final_value=1e-5):
    initial_value = float(initial_value)
    final_value = float(final_value)
    def schedule(progress_remaining: float) -> float:
        # progress_remaining goes from 1.0 (start) to 0.0 (end)
        return final_value + (initial_value - final_value) * progress_remaining
    return schedule

def cosine_schedule(initial_value, final_value=1e-5):
    initial_value = float(initial_value)
    final_value = float(final_value)
    def schedule(progress_remaining: float) -> float:
        # 1.0 -> 0.0; map to cosine(0..pi)
        cos_term = 0.5 * (1 + math.cos(math.pi * (1 - progress_remaining)))
        return final_value + (initial_value - final_value) * cos_term
    return schedule

# Try to load existing model
try:
    model = TD3.load("./learn/checkpoints/model_XXXXXXXXXXXXX_steps.zip")
    model.set_env(env)
    if os.path.exists("./learn/checkpoints/replay_buffer_XXXXXXXXXXXXX.pkl"):
        model.load_replay_buffer("./learn/checkpoints/replay_buffer_XXXXXXXXXXXXX.pkl")
    print("Loaded existing model for continued training")
except:
    model = TD3(
        "MlpPolicy", env,
        learning_rate=cosine_schedule(3e-4, 1e-5),
        buffer_size=1_000_000,
        batch_size=128,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=4,
        action_noise=action_noise,
        target_policy_noise=TARGET_POLICY_NOISE_STD,
        target_noise_clip=TARGET_POLICY_NOISE_CLIP,
        verbose=1,
        tensorboard_log='./tensorboard/',
        learning_starts=125_000,
        policy_kwargs=dict(net_arch=[1024, 1024])
    )
    print("Created new model")

# Callbacks
save_callback = SaveModelCallback(save_freq=250000, save_path="./learn/checkpoints", verbose=1)
decay_noise_callback = DecayNoiseCallback(
    decay_freq=2_500_000,
    decay_factor=0.8,
    min_action_sigma=0.02,
    min_target_policy_noise=0.05,
    min_target_noise_clip=0.1,
    verbose=1
)

# Train
model.learn(
    total_timesteps=25_000_000,
    tb_log_name="run_1",
    log_interval=1,
    callback=[save_callback, decay_noise_callback]
)

# Save final model
model.save("./learn/final_model")
