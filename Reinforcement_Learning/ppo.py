import argparse
import gc
import os

os.environ.setdefault("MUJOCO_GL", "egl")

from collections import deque
from datetime import datetime
from pathlib import Path
from time import perf_counter
import subprocess
import sys

import gymnasium as gym
import imageio
import keras
import numpy as np
import tensorflow as tf
import tqdm

from Reinforcement_Learning.utils import MetricLog


# Runs in a fresh Python subprocess WITHOUT TensorFlow to avoid the
# TF-LLVM vs Mesa-LLVM (EGL software rendering) conflict that segfaults
# when TF and MuJoCo rendering coexist in one process
_MUJOCO_DEMO_CODE = r"""
import sys
import numpy as np
import gymnasium as gym
import imageio

env_id, npz_path, gif_path, max_steps = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
data = np.load(npz_path)
W1, b1 = data["W1"], data["b1"]
W2, b2 = data["W2"], data["b2"]
Wmu, bmu = data["Wmu"], data["bmu"]
action_mean, action_scale = data["action_mean"], data["action_scale"]
action_low, action_high = data["action_low"], data["action_high"]

env = gym.make(env_id, render_mode="rgb_array")
obs, _ = env.reset()
frames = []
total = 0.0
for _ in range(max_steps):
    frames.append(env.render())
    x = obs.astype(np.float32) @ W1 + b1
    x = np.maximum(x, 0)
    x = x @ W2 + b2
    x = np.maximum(x, 0)
    mu = action_mean + action_scale * np.tanh(x @ Wmu + bmu)
    action = np.clip(mu, action_low, action_high).astype(np.float32)
    obs, reward, terminated, truncated, _ = env.step(action)
    total += float(reward)
    if terminated or truncated:
        break
env.close()
imageio.mimsave(gif_path, frames, fps=30)
print(f"demo done reward={total:.1f} frames={len(frames)} -> {gif_path}")
"""


def save_mujoco_demo(
    env_id, model, action_mean, action_scale, action_low, action_high,
    max_steps, npz_path, gif_path,
):
    W1, b1 = model.shared_network.layers[0].get_weights()
    W2, b2 = model.shared_network.layers[1].get_weights()
    Wmu, bmu = model.mu_head.get_weights()
    np.savez(
        npz_path,
        W1=W1, b1=b1, W2=W2, b2=b2, Wmu=Wmu, bmu=bmu,
        action_mean=np.asarray(action_mean, dtype=np.float32),
        action_scale=np.asarray(action_scale, dtype=np.float32),
        action_low=np.asarray(action_low, dtype=np.float32),
        action_high=np.asarray(action_high, dtype=np.float32),
    )
    cmd = [
        sys.executable, "-c", _MUJOCO_DEMO_CODE,
        env_id, str(npz_path), str(gif_path), str(max_steps),
    ]
    subprocess.run(cmd, check=True)
    try:
        Path(npz_path).unlink()
    except OSError:
        pass


def configure_hardware(device="auto", mixed_precision="auto", threads=0):
    import os as _os

    if threads and threads > 0:
        n_threads = int(threads)
    else:
        n_threads = _os.cpu_count() or 8
    try:
        tf.config.threading.set_inter_op_parallelism_threads(n_threads)
        tf.config.threading.set_intra_op_parallelism_threads(n_threads)
    except RuntimeError:
        pass  # must be set before any TF runtime init

    gpus = tf.config.list_physical_devices("GPU")
    want_gpu = device in ("auto", "gpu")
    gpu_available = bool(gpus) and want_gpu
    if device == "gpu" and not gpus:
        print("WARNING: --device=gpu requested but no GPU found, using CPU")
    if gpu_available:
        try:
            for g in gpus:
                tf.config.experimental.set_memory_growth(g, True)
        except Exception as e:  # pragma: no cover
            print(f"Could not set GPU memory growth: {e}")
    use_mp = (
        (mixed_precision == "on")
        or (mixed_precision == "auto" and gpu_available)
    )
    if use_mp:
        try:
            from keras import mixed_precision as _mp

            _mp.set_global_policy("mixed_float16")
            print("Mixed precision enabled (mixed_float16)")
        except Exception as e:  # pragma: no cover
            print(f"Could not enable mixed precision: {e}")
            use_mp = False

    if gpu_available:
        device_str = f"GPU ({len(gpus)}x {[g.name for g in gpus]})"
    else:
        device_str = f"CPU ({n_threads} threads)"
    print(f"Hardware: {device_str} | mixed_precision={use_mp}")
    return device_str, gpu_available, use_mp


def build_arg_parser():
    # Defaults are tuned for HumanoidStandup-v5 (348-dim obs, 17-dim
    # continuous actions, ~1k reward scale, 1000-step episodes), so the
    # script runs with good values and NO flags -- e.g. on Google Colab:
    #   !python Reinforcement_Learning/ppo.py
    # or from a notebook cell:  main([])
    # Any flag still overrides its default (local CLI usage unchanged).
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--env-id", default="HumanoidStandup-v5", type=str)
    arg_parser.add_argument("--num-envs", default=16, type=int)
    arg_parser.add_argument("--num-test-envs", default=10, type=int)
    arg_parser.add_argument("--gamma", default=0.99, type=float)
    arg_parser.add_argument("--gae-lambda", default=0.95, type=float)
    arg_parser.add_argument("--clip-eps", default=0.2, type=float)
    arg_parser.add_argument("--value-coef", default=0.5, type=float)
    arg_parser.add_argument("--entropy-beta", default=0.01, type=float)
    arg_parser.add_argument("--alpha", default=0.0003, type=float)
    arg_parser.add_argument("--num-steps", default=512, type=int)
    arg_parser.add_argument("--epochs", default=10, type=int)
    arg_parser.add_argument("--minibatch-size", default=512, type=int)
    arg_parser.add_argument("--max-grad-norm", default=0.5, type=float)
    arg_parser.add_argument("--log-std-init", default=-0.5, type=float)
    arg_parser.add_argument("--train-iteration", default=2_500, type=int)
    arg_parser.add_argument("--log-interval", default=500, type=int)
    # Gymnasium defines no solved threshold for HumanoidStandup
    # (spec.reward_threshold is None); 100000 effectively disables the
    # "Problem Solved" early trigger while best-checkpointing still works.
    arg_parser.add_argument("--reward-threshold", default=100000.0, type=float)
    arg_parser.add_argument("--test-max-steps", default=1000, type=int)
    arg_parser.add_argument("--seed", default=0, type=int)
    arg_parser.add_argument(
        "--device", default="auto", choices=["auto", "gpu", "cpu"], type=str,
        help="auto uses NVIDIA GPU if TF detects one, else CPU",
    )
    arg_parser.add_argument(
        "--mixed-precision", default="off", choices=["auto", "on", "off"],
        type=str,
        help="float16 trunk compute. Default off: this 2-layer MLP is "
        "env-step bound, not matmul bound, and float16 trunk activations "
        "can overflow (obs magnitudes x wide dot products -> inf -> NaN "
        "advantages). Opt in with 'on' (guarded by loss scaling + "
        "non-finite grad/batch checks). 'auto' = on iff GPU present.",
    )
    arg_parser.add_argument(
        "--xla", action="store_true",
        help="compile train steps with XLA (auto-enabled when a GPU is "
        "detected; on CPU-only machines it is ~neutral/slightly slower)",
    )
    arg_parser.add_argument(
        "--threads", default=0, type=int,
        help="TF inter/intra threads (0 = cpu_count, for CPU env stepping)",
    )
    return arg_parser


@keras.saving.register_keras_serializable(package="ppomodel", name="PPO_Model")
class ActorCritic(tf.keras.Model):
    def __init__(self, observation_space: tuple[int, ...], action_space: int, **kwargs):
        super().__init__(**kwargs)
        self.observation_space = observation_space
        self.action_space = action_space
        self.shared_network = tf.keras.models.Sequential(
            [
                tf.keras.layers.InputLayer(shape=self.observation_space),
                tf.keras.layers.Dense(
                    128, activation="relu", kernel_initializer="orthogonal"
                ),
                tf.keras.layers.Dense(
                    128, activation="relu", kernel_initializer="orthogonal"
                ),
            ]
        )
        self.actor = tf.keras.layers.Dense(
            self.action_space, kernel_initializer="orthogonal"
        )
        self.critic = tf.keras.layers.Dense(1, kernel_initializer="orthogonal")

    def call(self, obs: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        x = self.shared_network(obs)
        # Cast to float32: under mixed_float16 the trunk computes in
        # float16, but logits/values must stay float32 for
        # categorical sampling, log_softmax and loss stability.
        return tf.cast(self.actor(x), tf.float32), tf.cast(
            self.critic(x), tf.float32
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "observation_space": self.observation_space,
                "action_space": self.action_space,
            }
        )
        return config


@keras.saving.register_keras_serializable(package="ppomodel", name="PPO_Gaussian")
class GaussianActorCritic(tf.keras.Model):
    def __init__(
        self,
        observation_space: tuple[int, ...],
        action_dim: int,
        action_mean: list[float] | None = None,
        action_scale: list[float] | None = None,
        log_std_init: float = -0.5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.observation_space = observation_space
        self.action_dim = action_dim
        self.action_mean_list = (
            action_mean if action_mean is not None else [0.0] * action_dim
        )
        self.action_scale_list = (
            action_scale if action_scale is not None else [1.0] * action_dim
        )
        self.log_std_init = log_std_init
        self.action_mean_tf = tf.constant(
            self.action_mean_list, dtype=tf.float32
        )
        self.action_scale_tf = tf.constant(
            self.action_scale_list, dtype=tf.float32
        )
        self.shared_network = tf.keras.models.Sequential(
            [
                tf.keras.layers.InputLayer(shape=self.observation_space),
                tf.keras.layers.Dense(
                    256, activation="relu", kernel_initializer="orthogonal"
                ),
                tf.keras.layers.Dense(
                    256, activation="relu", kernel_initializer="orthogonal"
                ),
            ]
        )
        self.mu_head = tf.keras.layers.Dense(
            self.action_dim,
            activation="tanh",
            kernel_initializer="orthogonal",
        )
        self.critic = tf.keras.layers.Dense(1, kernel_initializer="orthogonal")
        self.log_std = tf.Variable(
            tf.fill([self.action_dim], float(self.log_std_init)),
            trainable=True,
            dtype=tf.float32,
            name="log_std",
        )

    def call(
        self, obs: tf.Tensor
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        x = self.shared_network(obs)
        # Head math in float32: under mixed_float16 mu_raw is float16 and
        # would clash with the float32 action constants / log_std, and the
        # Gaussian log-prob/entropy need float32 precision anyway.
        mu_raw = tf.cast(self.mu_head(x), tf.float32)
        mu = self.action_mean_tf + self.action_scale_tf * mu_raw
        clipped_log_std = tf.clip_by_value(self.log_std, -20.0, 2.0)
        std = tf.exp(clipped_log_std)  # [D], broadcasts to [B, D]
        value = tf.cast(self.critic(x), tf.float32)
        return mu, std, value

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "observation_space": self.observation_space,
                "action_dim": self.action_dim,
                "action_mean": self.action_mean_list,
                "action_scale": self.action_scale_list,
                "log_std_init": self.log_std_init,
            }
        )
        return config


def gaussian_log_prob(
    actions: tf.Tensor, mu: tf.Tensor, std: tf.Tensor
) -> tf.Tensor:
    var = tf.square(std) + 1e-8
    log_std = tf.math.log(std + 1e-8)
    log_prob = -0.5 * (
        tf.square(actions - mu) / var
        + 2.0 * log_std
        + tf.math.log(2.0 * np.pi)
    )
    return tf.reduce_sum(log_prob, axis=-1)


def gaussian_entropy(std: tf.Tensor) -> tf.Tensor:
    ent = 0.5 * (tf.math.log(2.0 * np.pi * np.e) + 2.0 * tf.math.log(std + 1e-8))
    return tf.reduce_sum(ent, axis=-1)


class Agent:
    def __init__(
        self,
        env,
        test_env,
        demo_env,
        model,
        optimizer,
        gamma,
        gae_lambda,
        clip_eps,
        value_coef,
        entropy_beta,
        n_steps,
        n_epochs,
        minibatch_size,
        test_max_steps,
        xla=False,
    ):
        self.env = env
        self.test_env = test_env
        self.demo_env = demo_env
        self.model = model
        self.optimizer = optimizer
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_beta = entropy_beta
        self.n_steps = n_steps
        self.n_epochs = n_epochs
        self.minibatch_size = minibatch_size
        self.test_max_steps = test_max_steps
        self.xla = bool(xla)
        # Duck-typed: a LossScaleOptimizer wrapper (used with mixed_float16)
        # needs scaled-loss / unscaled-gradients handling in train steps.
        self._loss_scaling = hasattr(
            optimizer, "scale_loss"
        ) and hasattr(optimizer, "get_unscaled_gradients")
        self.train_step_discrete = tf.function(
            self._train_step_discrete_impl,
            input_signature=[
                tf.TensorSpec(shape=(None, None), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.int32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
            ],
            jit_compile=self.xla,
        )
        self.train_step_continuous = tf.function(
            self._train_step_continuous_impl,
            input_signature=[
                tf.TensorSpec(shape=(None, None), dtype=tf.float32),
                tf.TensorSpec(shape=(None, None), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.float32),
            ],
            jit_compile=self.xla,
        )

        self.num_envs = self.env.num_envs
        self.obs_dim = int(self.env.single_observation_space.shape[0])
        self.is_continuous = isinstance(
            self.env.single_action_space, gym.spaces.Box
        )
        if self.is_continuous:
            self.action_dim = int(np.prod(self.env.single_action_space.shape))
            low = np.asarray(
                self.env.single_action_space.low, dtype=np.float32
            ).reshape(-1)
            high = np.asarray(
                self.env.single_action_space.high, dtype=np.float32
            ).reshape(-1)
            self.action_low = low
            self.action_high = high
            self.action_low_tf = tf.convert_to_tensor(low, dtype=tf.float32)
            self.action_high_tf = tf.convert_to_tensor(high, dtype=tf.float32)
        else:
            self.action_dim = int(self.env.single_action_space.n)
            self.action_low = None
            self.action_high = None
            self.action_low_tf = None
            self.action_high_tf = None
        self.current_states, _ = self.env.reset()

    @tf.function(input_signature=[tf.TensorSpec(shape=(None, None), dtype=tf.float32)])
    def predict(self, state):
        if self.is_continuous:
            mu, std, state_value = self.model(state)
            eps = tf.random.normal(tf.shape(mu), dtype=mu.dtype)
            raw_actions = mu + std * eps
            log_prob = gaussian_log_prob(raw_actions, mu, std)
            return raw_actions, log_prob, tf.squeeze(state_value, axis=-1)
        else:
            action_logits, state_value = self.model(state)
            action = tf.random.categorical(action_logits, num_samples=1, dtype=tf.int32)
            action = tf.squeeze(action, axis=-1)
            log_softmax = tf.nn.log_softmax(action_logits)
            log_prob = tf.gather(log_softmax, action, batch_dims=1)
            return action, log_prob, tf.squeeze(state_value, axis=-1)

    @tf.function(input_signature=[tf.TensorSpec(shape=(None, None), dtype=tf.float32)])
    def predict_deterministic(self, state):
        if self.is_continuous:
            mu, _, _ = self.model(state, training=False)
            return mu
        action_logits, _ = self.model(state, training=False)
        return tf.argmax(action_logits, axis=-1)

    def collect_rollout(self, states: np.ndarray):
        states = states.astype(np.float32)
        obs_buf = np.zeros(
            (self.n_steps, self.num_envs, self.obs_dim), dtype=np.float32
        )
        if self.is_continuous:
            actions_buf = np.zeros(
                (self.n_steps, self.num_envs, self.action_dim), dtype=np.float32
            )
        else:
            actions_buf = np.zeros(
                (self.n_steps, self.num_envs), dtype=np.int32
            )
        rewards_buf = np.zeros((self.n_steps, self.num_envs), dtype=np.float32)
        dones_buf = np.zeros((self.n_steps, self.num_envs), dtype=np.bool_)
        values_buf = np.zeros((self.n_steps, self.num_envs), dtype=np.float32)
        logprobs_buf = np.zeros((self.n_steps, self.num_envs), dtype=np.float32)

        for step in range(self.n_steps):
            states_t = tf.convert_to_tensor(states)
            actions, log_probs, values = self.predict(states_t)
            actions_np = actions.numpy()
            if self.is_continuous:
                env_actions = np.clip(
                    actions_np, self.action_low, self.action_high
                ).astype(np.float32)
            else:
                env_actions = actions_np
            next_states, rewards, terminated, truncated, _ = self.env.step(env_actions)
            dones = np.logical_or(terminated, truncated)

            obs_buf[step] = states
            actions_buf[step] = actions_np
            rewards_buf[step] = rewards.astype(np.float32)
            dones_buf[step] = dones
            values_buf[step] = values.numpy()
            logprobs_buf[step] = log_probs.numpy()

            states = next_states.astype(np.float32)

        self.current_states = states

        if self.is_continuous:
            _, _, last_value = self.model(tf.convert_to_tensor(states))
        else:
            _, last_value = self.model(tf.convert_to_tensor(states))
        last_value = last_value.numpy().flatten()

        advantages = np.zeros_like(rewards_buf)
        last_gae = np.zeros((self.num_envs,), dtype=np.float32)
        last_next_value = last_value
        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                next_value = last_next_value
            else:
                next_value = values_buf[t + 1]
            next_not_done = 1.0 - dones_buf[t].astype(np.float32)
            delta = (
                rewards_buf[t]
                + self.gamma * next_value * next_not_done
                - values_buf[t]
            )
            advantages[t] = (
                delta + self.gamma * self.gae_lambda * next_not_done * last_gae
            )
            last_gae = advantages[t]
            last_next_value = values_buf[t]

        returns = advantages + values_buf

        flat_states = obs_buf.reshape(-1, self.obs_dim)
        if self.is_continuous:
            flat_actions = actions_buf.reshape(-1, self.action_dim)
        else:
            flat_actions = actions_buf.reshape(-1)
        flat_logprobs = logprobs_buf.reshape(-1)
        flat_advantages = advantages.reshape(-1)
        flat_returns = returns.reshape(-1)
        flat_values = values_buf.reshape(-1)

        rollout_reward = float(np.mean(np.sum(rewards_buf, axis=0)))

        if self.is_continuous:
            actions_t = tf.convert_to_tensor(flat_actions, dtype=tf.float32)
        else:
            actions_t = tf.convert_to_tensor(flat_actions, dtype=tf.int32)
        return (
            tf.convert_to_tensor(flat_states, dtype=tf.float32),
            actions_t,
            tf.convert_to_tensor(flat_logprobs, dtype=tf.float32),
            tf.convert_to_tensor(flat_advantages, dtype=tf.float32),
            tf.convert_to_tensor(flat_returns, dtype=tf.float32),
            tf.convert_to_tensor(flat_values, dtype=tf.float32),
            rollout_reward,
        )

    def _compute_and_apply_grads(self, tape, loss):
        """Gradient step with optional loss scaling.

        With mixed_float16 the trunk runs in float16, so gradients must be
        computed from a scaled loss and unscaled before the update --
        otherwise small grads underflow to zero. Plain-Adam path unchanged.
        Logged losses always use the unscaled `loss`.

        Non-finite grads (overflow -> inf/NaN, e.g. from a blown-up batch)
        are zeroed so one bad minibatch can never poison the weights;
        returns 1.0 if every grad was finite else 0.0 so the caller can warn.
        XLA-safe: no Python control flow, only elementwise ops.
        """
        variables = self.model.trainable_variables
        if self._loss_scaling:
            grads = tape.gradient(
                self.optimizer.scale_loss(loss), variables
            )
            grads = self.optimizer.get_unscaled_gradients(grads)
        else:
            grads = tape.gradient(loss, variables)
        finite_flags = []
        clean_grads = []
        for g in grads:
            if g is None:
                clean_grads.append(None)
                continue
            is_finite = tf.math.is_finite(g)
            finite_flags.append(tf.reduce_all(is_finite))
            clean_grads.append(tf.where(is_finite, g, tf.zeros_like(g)))
        self.optimizer.apply_gradients(zip(clean_grads, variables))
        if not finite_flags:
            return tf.constant(1.0)
        return tf.cast(
            tf.reduce_all(tf.stack(finite_flags)), tf.float32
        )

    def _train_step_discrete_impl(
        self,
        states: tf.Tensor,
        actions: tf.Tensor,
        old_logprobs: tf.Tensor,
        advantages: tf.Tensor,
        returns: tf.Tensor,
        old_values: tf.Tensor,
    ):
        with tf.GradientTape() as tape:
            action_logits, state_values = self.model(states, training=True)
            state_values = tf.squeeze(state_values, axis=-1)

            log_softmax = tf.nn.log_softmax(action_logits)
            new_logprobs = tf.gather(log_softmax, actions, batch_dims=1)

            ratio = tf.exp(new_logprobs - old_logprobs)
            pg1 = ratio * advantages
            pg2 = (
                tf.clip_by_value(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps)
                * advantages
            )
            policy_loss = -tf.reduce_mean(tf.minimum(pg1, pg2))

            v_clipped = old_values + tf.clip_by_value(
                state_values - old_values, -self.clip_eps, self.clip_eps
            )
            v_loss_unclipped = tf.square(state_values - returns)
            v_loss_clipped = tf.square(v_clipped - returns)
            value_loss = 0.5 * tf.reduce_mean(tf.maximum(v_loss_unclipped, v_loss_clipped))

            policy = tf.nn.softmax(action_logits)
            entropy = -tf.reduce_mean(
                tf.reduce_sum(policy * log_softmax, axis=-1)
            )

            loss = (
                policy_loss
                + self.value_coef * value_loss
                - self.entropy_beta * entropy
            )

            approx_kl = tf.reduce_mean((ratio - 1.0) - (new_logprobs - old_logprobs))
            clipfrac = tf.reduce_mean(
                tf.cast(tf.abs(ratio - 1.0) > self.clip_eps, tf.float32)
            )

        update_ok = self._compute_and_apply_grads(tape, loss)
        return (
            loss,
            policy_loss,
            value_loss,
            entropy,
            approx_kl,
            clipfrac,
            update_ok,
        )

    def _train_step_continuous_impl(
        self,
        states: tf.Tensor,
        actions: tf.Tensor,
        old_logprobs: tf.Tensor,
        advantages: tf.Tensor,
        returns: tf.Tensor,
        old_values: tf.Tensor,
    ):
        with tf.GradientTape() as tape:
            mu, std, state_values = self.model(states, training=True)
            state_values = tf.squeeze(state_values, axis=-1)

            new_logprobs = gaussian_log_prob(actions, mu, std)

            ratio = tf.exp(new_logprobs - old_logprobs)
            pg1 = ratio * advantages
            pg2 = (
                tf.clip_by_value(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps)
                * advantages
            )
            policy_loss = -tf.reduce_mean(tf.minimum(pg1, pg2))

            v_clipped = old_values + tf.clip_by_value(
                state_values - old_values, -self.clip_eps, self.clip_eps
            )
            v_loss_unclipped = tf.square(state_values - returns)
            v_loss_clipped = tf.square(v_clipped - returns)
            value_loss = 0.5 * tf.reduce_mean(tf.maximum(v_loss_unclipped, v_loss_clipped))

            entropy = tf.reduce_mean(gaussian_entropy(std))

            loss = (
                policy_loss
                + self.value_coef * value_loss
                - self.entropy_beta * entropy
            )

            approx_kl = tf.reduce_mean((ratio - 1.0) - (new_logprobs - old_logprobs))
            clipfrac = tf.reduce_mean(
                tf.cast(tf.abs(ratio - 1.0) > self.clip_eps, tf.float32)
            )

        update_ok = self._compute_and_apply_grads(tape, loss)
        return (
            loss,
            policy_loss,
            value_loss,
            entropy,
            approx_kl,
            clipfrac,
            update_ok,
        )

    def learn_from_episode(self):
        start = perf_counter()
        current_states, _ = (
            self.env.reset()
            if self.current_states is None
            else (self.current_states, None)
        )
        t_roll = perf_counter()
        (
            states,
            actions,
            old_logprobs,
            advantages,
            returns,
            old_values,
            rollout_reward,
        ) = self.collect_rollout(current_states)
        rollout_time = perf_counter() - t_roll

        # Firewall: a blown-up batch (sim inf/NaN or overflowed values)
        # must never reach the optimizer -- skip updates and reset the
        # envs so the next rollout starts from a clean state. This runs
        # eagerly (a few scalar reduces per iteration: negligible cost).
        # Note: is_finite is float-only; discrete int32 actions cannot
        # carry NaN/inf by construction, so check them only when floating.
        actions_finite = (
            tf.reduce_all(tf.math.is_finite(actions))
            if actions.dtype.is_floating
            else tf.constant(True)
        )
        batch_finite = bool(
            tf.reduce_all(tf.math.is_finite(states))
            and actions_finite
            and tf.reduce_all(tf.math.is_finite(old_logprobs))
            and tf.reduce_all(tf.math.is_finite(returns))
            and tf.reduce_all(tf.math.is_finite(old_values))
        )
        if not batch_finite:
            print(
                "WARNING: non-finite rollout batch (obs/actions/returns "
                "contain inf or NaN) -- skipping gradient updates and "
                "resetting envs."
            )
            self.current_states, _ = self.env.reset()
            end = perf_counter()
            zero = tf.constant(0.0)
            self.last_rollout_time = rollout_time
            self.last_train_time = 0.0
            return (
                rollout_reward,
                zero,
                zero,
                zero,
                zero,
                zero,
                zero,
                end - start,
            )

        adv_mean, adv_var = tf.nn.moments(advantages, axes=[0])
        advantages = (advantages - adv_mean) / tf.sqrt(adv_var + 1e-8)

        t_train = perf_counter()
        loss, policy_loss, value_loss, entropy, approx_kl, clipfrac = (
            tf.constant(0.0),
            tf.constant(0.0),
            tf.constant(0.0),
            tf.constant(0.0),
            tf.constant(0.0),
            tf.constant(0.0),
        )
        finite_min = tf.constant(1.0)
        n_updates = 0
        train_step = (
            self.train_step_continuous
            if self.is_continuous
            else self.train_step_discrete
        )
        dataset = tf.data.Dataset.from_tensor_slices(
            (states, actions, old_logprobs, advantages, returns, old_values)
        )
        for _ in range(self.n_epochs):
            ds = (
                dataset.shuffle(buffer_size=int(states.shape[0]))
                .batch(self.minibatch_size, drop_remainder=False)
                .prefetch(tf.data.AUTOTUNE)
            )
            for mb in ds:
                (
                    mb_states,
                    mb_actions,
                    mb_old_logprobs,
                    mb_advantages,
                    mb_returns,
                    mb_old_values,
                ) = mb
                (
                    l,
                    pl,
                    vl,
                    ent,
                    kl,
                    cf,
                    ok,
                ) = train_step(
                    mb_states,
                    mb_actions,
                    mb_old_logprobs,
                    mb_advantages,
                    mb_returns,
                    mb_old_values,
                )
                loss += l
                policy_loss += pl
                value_loss += vl
                entropy += ent
                approx_kl += kl
                clipfrac += cf
                finite_min = tf.minimum(finite_min, ok)
                n_updates += 1

        n_updates = max(n_updates, 1)
        if float(finite_min) < 1.0:
            print(
                "WARNING: non-finite gradients were sanitized this "
                "iteration (updates zeroed where needed) -- if this "
                "repeats, the run is diverging: lower --alpha, check "
                "reward scale, or disable --mixed-precision."
            )

        n_updates = max(n_updates, 1)
        train_time = perf_counter() - t_train
        end = perf_counter()
        self.last_rollout_time = rollout_time
        self.last_train_time = train_time
        return (
            rollout_reward,
            loss / n_updates,
            policy_loss / n_updates,
            value_loss / n_updates,
            entropy / n_updates,
            approx_kl / n_updates,
            clipfrac / n_updates,
            end - start,
        )

    def test(self, demo=False):
        if demo:
            states, _ = self.demo_env.reset()
            frames = []
            total_reward = 0.0
            for _ in range(self.test_max_steps):
                frame = self.demo_env.render()
                frames.append(frame)
                if self.is_continuous:
                    s = states[np.newaxis].astype(np.float32)
                    mu, _, _ = self.model(s, training=False)
                    action = np.clip(
                        mu.numpy()[0], self.action_low, self.action_high
                    )
                    states, reward, terminated, truncated, _ = self.demo_env.step(
                        action.astype(np.float32)
                    )
                else:
                    s = states[np.newaxis].astype(np.float32)
                    action_logits, _ = self.model(s, training=False)
                    action = int(tf.argmax(action_logits, axis=-1).numpy()[0])
                    states, reward, terminated, truncated, _ = self.demo_env.step(action)
                total_reward += float(reward)
                if terminated or truncated:
                    break
            return total_reward, frames

        states, _ = self.test_env.reset()
        states = states.astype(np.float32)
        num_test = self.test_env.num_envs
        active = np.ones(num_test, dtype=bool)
        total_reward = np.zeros(num_test, dtype=np.float32)
        for _ in range(self.test_max_steps):
            if not np.any(active):
                break
            s = tf.convert_to_tensor(states.astype(np.float32))
            det = self.predict_deterministic(s).numpy()
            if self.is_continuous:
                actions = np.clip(
                    det, self.action_low, self.action_high
                ).astype(np.float32)
            else:
                actions = det
            states, rewards, terminated, truncated, _ = self.test_env.step(actions)
            states = states.astype(np.float32)
            done = np.logical_or(terminated, truncated)
            total_reward += rewards.astype(np.float32) * active.astype(np.float32)
            active = np.logical_and(active, np.logical_not(done))
        return float(np.mean(total_reward)), []


def main(argv=None):
    """Entry point. `argv` lets notebook/Colab cells call main([...]) or
    main([]) for baked-in defaults without touching argparse / sys.argv."""
    metrics = MetricLog()

    args = build_arg_parser().parse_args(argv)
    ENV_ID = args.env_id
    NUM_ENVS = args.num_envs
    NUM_TEST_ENVS = args.num_test_envs
    GAMMA = args.gamma
    GAE_LAMBDA = args.gae_lambda
    CLIP_EPS = args.clip_eps
    VALUE_COEF = args.value_coef
    ENTROPY_BETA = args.entropy_beta
    ALPHA = args.alpha
    NUM_STEPS = args.num_steps
    EPOCHS = args.epochs
    MINIBATCH_SIZE = args.minibatch_size
    MAX_GRAD_NORM = args.max_grad_norm
    LOG_STD_INIT = args.log_std_init
    TRAIN_ITERATION = args.train_iteration
    LOG_INTERVAL = args.log_interval
    REWARD_THRESHOLD = args.reward_threshold
    TEST_MAX_STEPS = args.test_max_steps
    SEED = args.seed

    device_str, gpu_available, use_mp = configure_hardware(
        device=args.device,
        mixed_precision=args.mixed_precision,
        threads=args.threads,
    )

    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    _probe = gym.make(ENV_ID)
    _is_box = isinstance(_probe.action_space, gym.spaces.Box)
    _probe.close()
    _vec_mode = "sync" if _is_box else "async"

    envs = gym.make_vec(
        ENV_ID, num_envs=NUM_ENVS, vectorization_mode=_vec_mode
    )
    test_env = gym.make_vec(
        ENV_ID, num_envs=NUM_TEST_ENVS, vectorization_mode=_vec_mode
    )
    demo_env = gym.make(ENV_ID, render_mode="rgb_array")

    is_continuous = isinstance(envs.single_action_space, gym.spaces.Box)
    obs_dim = int(envs.single_observation_space.shape[0])

    safe_env = ENV_ID.replace("/", "_")
    weights_path = f"ppo_{safe_env}.weights.h5"
    best_path = f"ppo_{safe_env}_best.keras"
    solved_path = f"ppo_{safe_env}.keras"

    if is_continuous:
        action_dim = int(np.prod(envs.single_action_space.shape))
        low = np.asarray(envs.single_action_space.low, dtype=np.float32).reshape(-1)
        high = np.asarray(envs.single_action_space.high, dtype=np.float32).reshape(-1)
        action_mean = ((high + low) / 2.0).tolist()
        action_scale = ((high - low) / 2.0).tolist()
        model = GaussianActorCritic(
            (obs_dim,), action_dim, action_mean, action_scale, LOG_STD_INIT
        )
    else:
        action_dim = int(envs.single_action_space.n)
        model = ActorCritic((obs_dim,), action_dim)
    dummy_input = tf.zeros((1, obs_dim), dtype=tf.float32)
    _ = model(dummy_input)

    if Path(weights_path).exists():
        try:
            model.load_weights(weights_path)
            print(f"Loaded weights from {weights_path}")
        except Exception as e:
            print(f"Could not load {weights_path} ({e}), training from scratch")

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=ALPHA, clipnorm=MAX_GRAD_NORM
    )
    if use_mp:
        # Custom GradientTape loop gets no automatic loss scaling from the
        # mixed_float16 policy, so wrap explicitly: without this, small
        # float16 gradients underflow to zero and training silently stalls.
        try:
            from keras.mixed_precision import LossScaleOptimizer

            optimizer = LossScaleOptimizer(optimizer)
            print("Loss scaling enabled (LossScaleOptimizer)")
        except Exception as e:  # pragma: no cover
            print(f"Could not enable loss scaling: {e}")

    # XLA pays off on NVIDIA GPUs (e.g. Colab T4/L4/A100) and is roughly
    # neutral on CPU, so auto-enable it whenever a GPU was detected.
    use_xla = bool(args.xla or gpu_available)
    agent = Agent(
        envs,
        test_env,
        demo_env,
        model,
        optimizer,
        GAMMA,
        GAE_LAMBDA,
        CLIP_EPS,
        VALUE_COEF,
        ENTROPY_BETA,
        NUM_STEPS,
        EPOCHS,
        MINIBATCH_SIZE,
        TEST_MAX_STEPS,
        xla=use_xla,
    )
    print(f"Device: {device_str} | xla={use_xla} | env={ENV_ID}")

    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    train_logs_dir = f"logs/PPO/{safe_env}/train/" + current_time
    test_logs_dir = f"logs/PPO/{safe_env}/test/" + current_time

    train_summary_writer = tf.summary.create_file_writer(train_logs_dir)
    test_summary_writer = tf.summary.create_file_writer(test_logs_dir)

    demo_dir = Path(f"./{safe_env}_PPO")
    demo_dir.mkdir(parents=True, exist_ok=True)

    moving_average_reward: deque = deque(maxlen=100)
    best_test_reward = float("-inf")
    try:
        t = tqdm.trange(TRAIN_ITERATION)
        for iteration in t:
            (
                rollout_reward,
                loss,
                policy_loss,
                value_loss,
                entropy,
                approx_kl,
                clipfrac,
                iteration_time,
            ) = agent.learn_from_episode()

            moving_average_reward.append(rollout_reward)
            running_reward = np.mean(moving_average_reward)
            t.set_postfix(
                rollout_reward=rollout_reward,
                running_reward=running_reward,
            )

            if iteration % LOG_INTERVAL == 0:
                test_reward, _ = agent.test()

                metrics.log(iteration, iteration_time)

                with train_summary_writer.as_default():
                    tf.summary.scalar("train rollout reward", running_reward, step=iteration)
                    tf.summary.scalar("policy loss", policy_loss, step=iteration)
                    tf.summary.scalar("value loss", value_loss, step=iteration)
                    tf.summary.scalar("entropy", entropy, step=iteration)
                    tf.summary.scalar("loss", loss, step=iteration)
                    tf.summary.scalar("approx kl", approx_kl, step=iteration)
                    tf.summary.scalar("clipfrac", clipfrac, step=iteration)
                    tf.summary.scalar("iteration_time", iteration_time, step=iteration)
                    tf.summary.scalar(
                        "rollout_time",
                        getattr(agent, "last_rollout_time", 0.0),
                        step=iteration,
                    )
                    tf.summary.scalar(
                        "train_time",
                        getattr(agent, "last_train_time", 0.0),
                        step=iteration,
                    )

                with test_summary_writer.as_default():
                    tf.summary.scalar("test reward", test_reward, step=iteration)

                print(
                    f"iter {iteration}: rollout={rollout_reward:.1f} "
                    f"running={running_reward:.1f} test={test_reward:.1f} "
                    f"loss={float(loss):.4f} kl={float(approx_kl):.4f} "
                    f"clipfrac={float(clipfrac):.3f} "
                    f"iter_t={iteration_time:.2f}s "
                    f"(rollout={getattr(agent, 'last_rollout_time', 0):.2f}s "
                    f"train={getattr(agent, 'last_train_time', 0):.2f}s) "
                    f"[{device_str}]"
                )

                model.save_weights(weights_path)

                if is_continuous:
                    # MuJoCo EGL rendering segfaults in-process with TF
                    # (LLVM conflict), so render demos in a TF-free subprocess.
                    gif_path = demo_dir / f"episode-{iteration}.gif"
                    npz_path = demo_dir / f"policy-{iteration}.npz"
                    save_mujoco_demo(
                        ENV_ID,
                        model,
                        action_mean,
                        action_scale,
                        low,
                        high,
                        TEST_MAX_STEPS,
                        npz_path,
                        gif_path,
                    )
                    gc.collect()
                else:
                    _, frames = agent.test(demo=True)
                    imageio.mimsave(
                        str(demo_dir / f"episode-{iteration}.gif"), frames, fps=30
                    )
                    del frames
                    gc.collect()

                if test_reward > best_test_reward:
                    best_test_reward = test_reward
                    model.save(best_path)

                if test_reward >= REWARD_THRESHOLD:
                    print(f"Problem Solved at iteration {iteration}!")
                    model.save(solved_path)

    except KeyboardInterrupt:
        pass
    finally:
        envs.close()
        test_env.close()
        demo_env.close()
        train_summary_writer.close()
        test_summary_writer.close()
        tf.keras.models.save_model(model, solved_path)


if __name__ == "__main__":
    main()
