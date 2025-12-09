import math
import time
import os
import torch
import random
import numpy as np
from tqdm import tqdm
import torch.multiprocessing as multiprocessing
import gymnasium as gym

from typing import Any, Optional, List

from myohuman.learning.memory import Memory
from myohuman.learning.trajbatch import TrajBatch
from myohuman.learning.logger_rl import LoggerRL
from myohuman.learning.learning_utils import to_test, to_cpu


random.seed(0)

os.environ["OMP_NUM_THREADS"] = "1"

done = multiprocessing.Event()


class Agent:

    def __init__(
        self,
        env: gym.Env,
        policy_net: torch.nn.Module,
        value_net: torch.nn.Module,
        dtype: torch.dtype,
        device: torch.device,
        gamma: float,
        mean_action: bool = False,
        headless: bool = False,
        num_threads: int = 1,
        clip_obs: bool = False,
        clip_actions: bool = False,
        clip_obs_range: Optional[List[float]] = None,
    ):
        self.env = env
        self.policy_net = policy_net
        self.value_net = value_net

        self.dtype = dtype
        self.device = device
        self.np_dtype = np.float32

        self.gamma = gamma
        self.noise_rate = 1.0

        self.num_steps = 0
        self.mean_action = mean_action
        self.headless = headless
        self.num_threads = num_threads
        
        self.clip_obs = clip_obs
        self.clip_actions = clip_actions
        self.obs_low = clip_obs_range[0]
        self.obs_high = clip_obs_range[1]

        self.actions_num = self.env.action_space.shape[0]
        self.actions_low = self.env.action_space.low.copy()
        self.actions_high = self.env.action_space.high.copy()

    def seed_worker(self, pid: int) -> None:
        if pid > 0:
            random.seed(self.epoch)
            seed = random.randint(0, 5000) * pid

            torch.manual_seed(seed)
            np.random.seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True)

    def sample_worker(self, pid: int, queue: Optional[multiprocessing.Queue], min_batch_size: int):
        """
        Worker function to sample trajectories from the environment.

        Args:
            pid (int): Process ID.
            queue (Optional[multiprocessing.Queue]): Queue to put the sampled data.
            min_batch_size (int): Minimum number of steps to sample.

        Returns:
            Optional[Tuple[Memory, LoggerRL]]: Memory and logger objects if queue is None.
        """
        self.seed_worker(pid)

        # Create memory and logger instances
        memory = Memory()
        logger = LoggerRL()

        # Initialize progress bar only for the main process (pid == 0)
        if pid == 0:
            pbar = tqdm(total=min_batch_size, desc="Sampling", unit="step")

        try:
            while logger.num_steps < min_batch_size:
                obs_dict, info = self.env.reset()
                state = self.preprocess_obs(
                    obs_dict
                )  # let's assume that the environment always return a np.ndarray (see https://gymnasium.farama.org/api/wrappers/observation_wrappers/#gymnasium.wrappers.FlattenObservation)
                logger.start_episode(self.env)
                for t in range(10000):
                    with torch.no_grad():
                        mean_action = self.mean_action or self.env.np_random.binomial(
                            1, 1 - self.noise_rate
                        )
                        actions = self.policy_net.select_action(
                            torch.from_numpy(state).to(self.dtype), mean_action
                        )[0].cpu().numpy()
                    # breakpoint()
                    next_obs, reward, terminated, truncated, info = self.env.step(
                        self.preprocess_actions(actions)
                    )  # action processing should not affect the recorded action
                    episode_done = terminated or truncated
                    next_state = self.preprocess_obs(next_obs)

                    logger.step(self.env, reward, info)

                    mask = 0 if episode_done else 1
                    exp = 1 - mean_action
                    self.push_memory(
                        memory,
                        state.squeeze(),
                        actions,
                        mask,
                        next_state.squeeze(),
                        reward,
                        exp,
                    )

                    # Update progress bar
                    if pid == 0:
                        pbar.update(1)
                        pbar.set_postfix({"current_steps": logger.num_steps})

                    if pid == 0 and not self.headless:
                        self.env.render()
                        
                    if episode_done:
                        break
                    state = next_state

                logger.end_episode(self.env)
        except Exception as e:
            logger.error(f"Sampling worker {pid} failed with exception: {e}")
        finally:
            logger.end_sampling()

            if pid == 0:
                pbar.close()

            if queue is not None:
                queue.put([pid, memory, logger])
                done.wait()
            else:
                return memory, logger

    def push_memory(
        self,
        memory: Memory,
        state: np.ndarray,
        action: np.ndarray,
        mask: int,
        next_state: np.ndarray,
        reward: float,
        exploration_flag: float,
    ) -> None:
        """
        Push a transition to the memory buffer.

        Args:
            memory (Memory): Memory buffer.
            state (np.ndarray): Current state.
            action (np.ndarray): Action taken.
            mask (int): Mask indicating if the episode is done (0) or not (1).
            next_state (np.ndarray): Next state.
            reward (float): Reward received.
            exploration_flag (float): Flag indicating exploration (1) or exploitation (0).
        """
        memory.push(state, action, mask, next_state, reward, exploration_flag)

    def sample(self, min_batch_size):

        # Record current time
        t_start = time.time()

        # Switch to test mode
        to_test(self.policy_net)

        # Run networks on CPU
        with to_cpu(self.policy_net):
            thread_batch_size = int(math.floor(min_batch_size / self.num_threads))
            queue = multiprocessing.Queue()
            memories = [None] * self.num_threads
            loggers = [None] * self.num_threads

            # Spawn workers with unique PIDs starting from 1
            for i in range(self.num_threads - 1):
                worker_args = (i + 1, queue, thread_batch_size)
                worker = multiprocessing.Process(
                    target=self.sample_worker, args=worker_args
                )
                worker.start()

            # Sample trajectories in the main process
            memories[0], loggers[0] = (
                self.sample_worker(0, None, thread_batch_size)
            )

            # Retrieve data from workers
            for i in range(self.num_threads - 1):
                pid, worker_memory, worker_logger = queue.get()
                memories[pid] = worker_memory
                loggers[pid] = worker_logger

            # Merge memories and loggers
            traj_batch = TrajBatch(memories)
            logger = LoggerRL.merge(loggers)

        logger.sample_time = time.time() - t_start

        # Signal sampling is done
        done.set()

        return traj_batch, logger

    def preprocess_obs(self, obs: Any) -> np.ndarray:
        """
        Preprocess observations by rehaping and clipping if necessary.

        Args:
            obs (Any): Observations from the environment.

        Returns:
            np.ndarray: Preprocessed observations.
        """
        if self.clip_obs:
            return np.clip(obs.reshape(1, -1), self.obs_low, self.obs_high)
        else:
            return obs.reshape(1, -1)

    def preprocess_actions(self, actions: np.ndarray) -> np.ndarray:
        actions = (
            int(actions)
            if self.policy_net.type == "discrete"
            else actions.astype(self.np_dtype)
        )
        if self.clip_actions:
            actions = self.rescale_actions(
                self.actions_low,
                self.actions_high,
                np.clip(actions, self.actions_low, self.actions_high),
            )
        return actions

    def rescale_actions(self, low, high, action):
        d = (high - low) / 2.0
        m = (high + low) / 2.0
        scaled_action = action * d + m
        return scaled_action
