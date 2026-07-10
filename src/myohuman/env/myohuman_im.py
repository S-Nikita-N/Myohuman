import os
import sys
import joblib
import numpy as np
import mujoco
import logging
import mujoco.viewer
import myohuman.utils.np_transform_utils as npt_utils

from pathlib import Path
from easydict import EasyDict
from collections import OrderedDict
from omegaconf import DictConfig
from collections.abc import Iterator
from scipy.spatial.transform import Rotation as sRot

from myohuman.env.myohuman_task import MyoHumanTask
from myohuman.utils.visual_capsule import add_visual_capsule


path_root = Path(__file__).resolve().parents[2]
sys.path.append(str(path_root))

logger = logging.getLogger(__name__)


MYOHUMAN_TRACKED_BODIES = [
    "root",  # Таз
    "tibia_l",  # Левая голень
    "tibia_r",  # Правая голень
    "talus_l",  # Левая лодыжка
    "talus_r",  # Правая лодыжка
    "toes_l",  # Левые пальцы стопы
    "toes_r",  # Правые пальцы стопы
    "head",  # Голова
    "humerus_r",  # Правая плечевая кость
    "radius_r",  # Правое предплечье
    "lunate_r",  # Правое запятье
    "humerus_l",  # Левая плечевая кость
    "radius_l",  # Левое предплечье
    "lunate_l",  # Левое запястье
]


class MyoHumanIm(MyoHumanTask):
    def __init__(self, cfg):
        self.initial_pose = None
        self.previous_pose = None
        self.ref_motion_cache = EasyDict()

        self.initialize_env_params(cfg)
        self.initialize_run_params(cfg)

        super().__init__(cfg)

        self.load_reference_data()
        self.setup_reference_model()
        self.initialize_biomechanical_recording()
        self.initialize_evaluation_metrics()

        # Heading randomization state (per-episode)
        self._randomize_heading = not self.test
        self._heading_rot = None  # sRot or None

    def initialize_env_params(self, cfg: DictConfig) -> None:
        """
        Initializes environment parameters from the configuration.

        Args:
            cfg (DictConfig): Configuration object.

        Sets:
            - `num_traj_samples`: Number of trajectory samples (default: 1).
            - `reward_specs`: Reward weight specifications.
            - `termination_distance`: Distance threshold for task termination.
        """
        self.num_traj_samples = 1  # parameter for number of future time steps
        self.reward_specs = cfg.env.reward_specs
        self.termination_distance = cfg.env.termination_distance
        self.valid_root_height = (0.86, 1.05)
        self.max_start_retries = 10

    def initialize_run_params(self, cfg: DictConfig) -> None:
        """
        Initializes run-specific parameters from the configuration.
        """
        self.motion_start_idx = cfg.run.motion_id
        self.im_eval = cfg.run.im_eval
        self.test = cfg.run.test
        self.num_motion_max = cfg.run.num_motions
        self.initial_pose_file = cfg.run.initial_pose_file
        self.random_sample = cfg.run.random_sample
        self.random_start = cfg.run.random_start
        self.recording_biomechanics = cfg.run.recording_biomechanics
        self.motion_ids_file = cfg.run.motion_ids_file

    def load_reference_data(self) -> None:
        """
        Loads pre-computed IK reference data produced by ``compute_ik.py``.

        Expected file format (new)::

            {"frames": {motion_id: {time: qpos, ...}, ...},
             "metadata": {motion_id: {"length", "dt", "fps", "num_frames"}, ...}}

        Also supports the legacy format ``{motion_id: {time: qpos, ...}}``.

        Sets:
            - ``self.initial_pos_data``  — frame lookup  {motion_id: {time: qpos}}
            - ``self._motion_metadata``  — per-motion metadata dict
            - ``self._all_motion_ids``   — sorted array of all motion IDs
            - ``self._active_motion_ids``— currently active subset (set by sample_motions)
            - ``self._num_total_motions``— total number of motions
        """
        if not os.path.exists(self.initial_pose_file):
            logger.warning(
                "!!! Reference data file not found: %s", self.initial_pose_file
            )
            self.initial_pos_data = {}
            self._motion_metadata = {}
            self._all_motion_ids = np.array([], dtype=int)
            self._active_motion_ids = np.array([], dtype=int)
            self._num_total_motions = 0
            return

        raw = joblib.load(self.initial_pose_file)

        # Detect format
        if isinstance(raw, dict) and "frames" in raw and "metadata" in raw:
            self.initial_pos_data = raw["frames"]
            self._motion_metadata = raw["metadata"]
        else:
            # Legacy format (flat dict of {motion_id: {time: qpos}})
            self.initial_pos_data = raw
            self._motion_metadata = {}

        all_ids = set(self.initial_pos_data.keys())

        # Filter by motion_ids_file (plain txt, one integer motion_id per line)
        if self.motion_ids_file:
            if not os.path.exists(self.motion_ids_file):
                raise FileNotFoundError(
                    f"motion_ids_file not found: {self.motion_ids_file}"
                )
            with open(self.motion_ids_file) as f:
                file_ids = {
                    int(line.strip())
                    for line in f
                    if line.strip() and not line.startswith("#")
                }
            n_before = len(all_ids)
            dropped = file_ids - all_ids
            if dropped:
                logger.warning(
                    "motion_ids_file: %d IDs not found in reference data (skipped)",
                    len(dropped),
                )
            all_ids = all_ids & file_ids
            logger.info(
                "Filtered by motion_ids_file: %d → %d motions (%s)",
                n_before,
                len(all_ids),
                self.motion_ids_file,
            )

        self._all_motion_ids = np.array(sorted(all_ids), dtype=int)
        self._num_total_motions = len(self._all_motion_ids)
        self._active_motion_ids = self._all_motion_ids.copy()

        logger.info(
            "Loaded reference data: %d motions from %s",
            self._num_total_motions,
            self.initial_pose_file,
        )

    def setup_reference_model(self) -> None:
        """
        Creates a secondary Mujoco data buffer used to compute reference muscle
        lengths/velocities from IK poses without touching the live simulation.
        """
        self.ref_data = mujoco.MjData(self.mj_model)

    def initialize_evaluation_metrics(self) -> None:
        """
        Initializes metrics used for evaluating motion performance.

        Sets:
            - `mpjpe` (list): Mean per-joint position error metric.
            - `frame_coverage` (float): Tracks the percentage of frames covered.
        """
        self.mpjpe = []
        self.frame_coverage = 0

    def initialize_biomechanical_recording(self):
        """
        Initializes storage for biomechanical data recording.

        If `self.recording_biomechanics` is True, prepares lists to store
        biomechanical data.

        Sets:
            - Various lists for biomechanical recording, including:
            - `self.feet`, `self.joint_pos`, `self.joint_vel`
            - `self.body_pos`, `self.body_rot`, `self.body_vel`
            - `self.ref_pos`, `self.ref_rot`, `self.ref_vel`
            - `self.motion_id`, `self.muscle_forces`, `self.muscle_controls`, `self.policy_outputs`
        """
        if self.recording_biomechanics:
            self.feet = []
            self.joint_pos = []
            self.joint_vel = []
            self.body_pos = []
            self.body_rot = []
            self.body_vel = []
            self.ref_pos = []
            self.ref_rot = []
            self.ref_vel = []
            self.motion_id = []
            self.muscle_forces = []
            self.muscle_controls = []
            self.policy_outputs = []

    def create_task_visualization(self) -> None:
        """
        Creates visual representations of tracked bodies in the task.

        Adds visual capsules to the viewer and renderer scenes for each tracked body.
        Capsules are color-coded for differentiation, with red and blue indicating different roles.
        """
        if self.viewer is not None:  # this implies that headless == False
            for _ in range(len(self.tracked_bodies)):
                add_visual_capsule(
                    self.viewer.user_scn,
                    np.zeros(3),
                    np.array([0.001, 0, 0]),
                    0.05,
                    np.array([1, 0, 0, 1]),
                )
                add_visual_capsule(
                    self.viewer.user_scn,
                    np.zeros(3),
                    np.array([0.001, 0, 0]),
                    0.05,
                    np.array([0, 0, 1, 1]),
                )

        if self.renderer is not None:
            for _ in range(len(self.tracked_bodies)):
                add_visual_capsule(
                    self.viewer.user_scn,
                    np.zeros(3),
                    np.array([0.001, 0, 0]),
                    0.05,
                    np.array([1, 0, 0, 1]),
                )

    def draw_task(self) -> None:
        """
        Updates the positions of visualized tracked bodies in the scene.

        Synchronizes visual objects in the viewer and renderer with the current
        task state, using positions from the motion library and simulation.
        """

        def draw_obj(scene):
            ref_dict = self.get_state_from_motionlib_cache(
                (self.cur_t) * self.dt + self._motion_start_time
            )
            ref_pos_subset = ref_dict.xpos[..., self.tracked_bodies_id, :]

            for i in range(len(self.tracked_bodies)):
                scene.geoms[2 * i].pos = ref_pos_subset[0, i]
                scene.geoms[2 * i + 1].pos = self.get_body_xpos()[
                    self.tracked_bodies_id[i]
                ]

        if self.viewer is not None:
            draw_obj(self.viewer.user_scn)
        if self.renderer is not None:
            draw_obj(self.renderer.scene)

    def setup_myohuman_params(self) -> None:
        """
        Configures body tracking and reset properties for MyoLeg.

        Initializes lists of tracked and resettable bodies, as well as their
        corresponding indices based on the original body names.

        Sets:
            - `self.full_tracked_bodies`: List of all original body names.
            - `self.tracked_bodies`: Names of tracked bodies specific to MyoLeg.
            - `self.reset_bodies`: Names of bodies to reset.
            - `self.tracked_bodies_id`: Indices of tracked bodies in `self.body_names`.
            - `self.reset_bodies_id`: Indices of reset bodies in `self.body_names`.
        """
        super().setup_myohuman_params()
        self.full_tracked_bodies = self.body_names
        self.tracked_bodies = MYOHUMAN_TRACKED_BODIES
        self.reset_bodies = self.tracked_bodies

        self.tracked_bodies_id = [
            self.body_names.index(j) for j in self.tracked_bodies
        ]

        raw_weights = self.cfg.env.get("body_reward_weights", None)
        if raw_weights is not None:
            if hasattr(raw_weights, "items"):
                w = np.ones(len(self.tracked_bodies), dtype=np.float64)
                for name, val in raw_weights.items():
                    assert name in self.tracked_bodies, (
                        f"body_reward_weights: unknown body '{name}'. "
                        f"Valid: {self.tracked_bodies}"
                    )
                    w[self.tracked_bodies.index(name)] = float(val)
            else:
                w = np.array(raw_weights, dtype=np.float64)
                assert len(w) == len(self.tracked_bodies), (
                    f"body_reward_weights length {len(w)} != tracked_bodies {len(self.tracked_bodies)}"
                )
            self.body_reward_weights = w / w.sum()
            logger.info(
                "Per-body reward weights: %s",
                {
                    b: f"{v:.3f}"
                    for b, v in zip(
                        self.tracked_bodies, self.body_reward_weights
                    )
                },
            )
        else:
            self.body_reward_weights = None

        raw_term = self.cfg.env.get("body_termination_distance", None)
        if raw_term is not None:
            arr = np.full(
                len(self.reset_bodies),
                self.termination_distance,
                dtype=np.float64,
            )
            for name, dist in raw_term.items():
                assert name in self.reset_bodies, (
                    f"body_termination_distance: unknown body '{name}'. "
                    f"Valid: {self.reset_bodies}"
                )
                arr[self.reset_bodies.index(name)] = float(dist)
            self.per_body_termination_distance = arr
            logger.info(
                "Per-body termination distances: %s",
                {b: f"{d:.3f}" for b, d in zip(self.reset_bodies, arr)},
            )
        else:
            self.per_body_termination_distance = None

        self.reset_bodies_id = [
            self.body_names.index(j) for j in self.reset_bodies
        ]

    def sample_motions(self) -> None:
        """
        Selects the active subset of motion IDs for the current training period.
        Called at init and periodically every ``resampling_interval`` epochs.
        """
        num_to_sample = min(self.num_motion_max, self._num_total_motions)
        if self.random_sample:
            self._active_motion_ids = np.random.choice(
                self._all_motion_ids,
                size=num_to_sample,
                replace=False,
            )
        else:
            start = self.motion_start_idx
            indices = (
                np.arange(start, start + num_to_sample)
                % self._num_total_motions
            )
            self._active_motion_ids = self._all_motion_ids[indices]

        # Initial values (will be updated in reset)
        self._sampled_motion_id = int(self._active_motion_ids[0])
        self._motion_start_time = 0

    def forward_motions(self) -> Iterator[int]:
        """
        Iterates through *all* motions one-by-one (used during evaluation).
        Yields the motion ID for each motion.
        """
        for motion_id in self._all_motion_ids:
            self._current_eval_motion_id = int(motion_id)
            self._active_motion_ids = np.array([motion_id])
            yield int(motion_id)

    def init_myohuman(self) -> None:
        """
        Initializes the MyoHuman environment state.

        This function sets up the initial state of the simulation, including position, velocity,
        and kinematics, using motion library data and cached or precomputed initial poses.
        It also initializes evaluation metrics and biomechanical recording if enabled.
        """
        super().init_myohuman()
        # Initialize motion states and poses
        self.initialize_motion_state()
        # Initialize evaluation metrics
        self.reset_evaluation_metrics()
        # Set up biomechanical recording if enabled
        if self.recording_biomechanics:
            self.delineate_biomechanical_recording()

    def initialize_motion_state(self) -> None:
        """
        Sets the simulation state (qpos / qvel) from pre-computed IK reference data,
        applying the current episode heading.
        """
        super().init_myohuman()

        motion_id = self._sampled_motion_id

        # Look up stored IK solution for this frame
        ref_qpos = self._lookup_reference_qpos(
            motion_id, self._motion_start_time
        )
        headed_qpos = self._apply_heading(ref_qpos.copy())

        self.mj_data.qpos[: len(headed_qpos)] = headed_qpos

        # Velocity from finite differences of two neighbouring frames
        if self._motion_start_time > 0:
            prev_qpos = self._lookup_reference_qpos(
                motion_id, self._motion_start_time - self.dt
            )
            prev_headed = self._apply_heading(prev_qpos.copy())
            mujoco.mj_differentiatePos(
                self.mj_model,
                self.mj_data.qvel,
                self.dt,
                prev_headed,
                headed_qpos,
            )
        else:
            self.mj_data.qvel[:] = 0

        # Zero out joint velocities — only root linear + angular are meaningful
        self.mj_data.qvel[6:] = 0

        mujoco.mj_kinematics(self.mj_model, self.mj_data)

    def get_termination_stats(self) -> dict[str, dict]:
        """Returns per-body termination stats (count + mean dist) and resets."""
        counts = getattr(self, "_termination_counts", {})
        dists = getattr(self, "_termination_dists", {})
        stats = {}
        for b in self.reset_bodies:
            c = counts.get(b, 0)
            d = dists.get(b, [])
            stats[b] = {
                "caused": c,
                "mean_dist": float(np.mean(d)) if d else 0.0,
            }
        self._termination_counts = dict.fromkeys(self.reset_bodies, 0)
        self._termination_dists = {b: [] for b in self.reset_bodies}
        return stats

    def reset_evaluation_metrics(self) -> None:
        """
        Resets evaluation metrics for motion imitation performance.
        """
        self.mpjpe = []

    def delineate_biomechanical_recording(self) -> None:
        """
        Adds a nan buffer to the biomechanical recording lists to indicate that a new episode has started.
        """
        self.feet.append(np.nan)
        self.joint_pos.append(np.full(self.get_qpos().shape, np.nan))
        self.joint_vel.append(np.full(self.get_qvel().shape, np.nan))
        self.body_pos.append(np.full(self.get_body_xpos()[None,].shape, np.nan))
        self.body_rot.append(
            np.full(
                self.get_body_xquat()[None,][
                    ..., self.tracked_bodies_id, :
                ].shape,
                np.nan,
            )
        )
        self.body_vel.append(
            np.full(
                self.get_body_linear_vel()[None,][
                    ..., self.tracked_bodies_id, :
                ].shape,
                np.nan,
            )
        )
        self.ref_pos.append(
            np.full(
                self.get_body_xpos()[None,][
                    ..., self.tracked_bodies_id, :
                ].shape,
                np.nan,
            )
        )
        self.ref_rot.append(
            np.full(
                self.get_body_xquat()[None,][
                    ..., self.tracked_bodies_id, :
                ].shape,
                np.nan,
            )
        )
        self.ref_vel.append(
            np.full(
                self.get_body_linear_vel()[None,][
                    ..., self.tracked_bodies_id, :
                ].shape,
                np.nan,
            )
        )
        self.motion_id.append(np.nan)
        self.muscle_forces.append(
            np.full(self.get_muscle_force().shape, np.nan)
        )
        self.muscle_controls.append(np.full(self.mj_data.ctrl.shape, np.nan))
        self.policy_outputs.append(np.full(self.mj_data.ctrl.shape, np.nan))

    def get_task_obs_size(self) -> int:
        """
        Calculates the size of the task observation vector based on configured inputs.

        This function sums up the dimensions of the observation components specified
        in `self.cfg.run.task_inputs`. Each component contributes to the observation size
        based on the number of tracked bodies and its dimensionality (e.g., 3 for position or velocity).

        Returns:
            int: The total size of the task observation vector.
        """
        inputs = self.cfg.run.task_inputs
        obs_size = 0
        if "diff_local_body_pos" in inputs:
            obs_size += 3 * len(self.tracked_bodies)
        if "diff_local_vel" in inputs:
            obs_size += 3 * len(self.tracked_bodies)
        if "local_ref_body_pos" in inputs:
            obs_size += 3 * len(self.tracked_bodies)
        if "diff_muscle_len" in inputs:
            obs_size += self.mj_model.nu
        if "diff_muscle_vel" in inputs:
            obs_size += self.mj_model.nu

        return obs_size

    def compute_reset(self) -> tuple[bool, bool]:
        """
        Determines whether the task should reset based on termination and truncation conditions.

        This function checks if the task should terminate early due to positional deviation
        from reference motions (termination) or if the task duration has exceeded the
        motion length (truncation). If either condition is met, evaluation metrics are computed
        and reset.

        Returns:
            Tuple containing
            - `terminated` (bool): True if the task exceeded the termination distance.
            - `truncated` (bool): True if the task exceeded the duration of the episode.

        Updates:
            - Calls `compute_evaluation_metrics` and `reset_evaluation_metrics` if reset conditions are met.
        """
        terminated, truncated = False, False
        sim_time = (self.cur_t) * self.dt + self._motion_start_time
        ref_dict = self.get_state_from_motionlib_cache(sim_time)
        body_pos = self.get_body_xpos()[None,]
        body_pos_subset = body_pos[..., self.reset_bodies_id, :]
        ref_pos_subset = ref_dict.xpos[..., self.reset_bodies_id, :]
        term_dist = (
            self.per_body_termination_distance
            if self.per_body_termination_distance is not None
            else self.termination_distance
        )
        terminated = compute_humanoid_im_reset(
            body_pos_subset,
            ref_pos_subset,
            termination_distance=term_dist,
            use_mean=True if self.im_eval else False,
        )[0]
        truncated = sim_time > self.get_motion_length(self._sampled_motion_id)

        if terminated:
            per_body_dist = np.linalg.norm(
                body_pos_subset[0] - ref_pos_subset[0], axis=-1
            )
            worst_idx = int(per_body_dist.argmax())
            self.last_termination_body = self.reset_bodies[worst_idx]
            self.last_termination_dist = per_body_dist
            if not hasattr(self, "_termination_counts"):
                self._termination_counts = dict.fromkeys(self.reset_bodies, 0)
                self._termination_dists = {b: [] for b in self.reset_bodies}
            self._termination_counts[self.reset_bodies[worst_idx]] += 1
            for i, b in enumerate(self.reset_bodies):
                self._termination_dists[b].append(float(per_body_dist[i]))

        if terminated or truncated:
            self.compute_evaluation_metrics(terminated, sim_time)
            self.reset_evaluation_metrics()

        return terminated, truncated

    def compute_evaluation_metrics(self, terminated, sim_time) -> None:
        """
        Computes evaluation metrics for the current simulation.

        This function calculates the mean per-joint position error (MPJPE) and frame
        coverage for the simulation. The frame coverage
        indicates the proportion of the motion completed before termination or completion.

        Args:
            terminated (bool): Indicates whether the simulation terminated early.
            sim_time (np.ndarray): Current time index for the simulation.

        Updates:
            - `self.mpjpe_value`: Average MPJPE across all frames.
            - `self.frame_coverage`: Ratio of completed frames to total motion length.
        """
        mpjpe_arr = np.array(self.mpjpe)
        self.mpjpe_value = mpjpe_arr.mean()
        self.max_mpjpe_value = mpjpe_arr.max() if len(mpjpe_arr) > 0 else 0.0
        if terminated:
            self.frame_coverage = sim_time / self.get_motion_length(
                self._sampled_motion_id
            )
        else:
            self.frame_coverage = 1.0
        # Kinesis-style: success = episode did not early-terminate (completed or got truncated)
        self.last_success = not terminated

    def reset_task(self, options: dict | None = None) -> None:
        """
        Resets the task: picks a motion, generates heading, sets start time.
        """
        # ── Pick a motion ────────────────────────────────────────────────
        if self.test and self.im_eval:
            # Evaluation mode: use the motion set by forward_motions()
            self._sampled_motion_id = getattr(
                self, "_current_eval_motion_id", int(self._active_motion_ids[0])
            )
        else:
            # Training or test-run: sample from the active pool
            self._sampled_motion_id = int(
                np.random.choice(self._active_motion_ids)
            )

        # ── Start time ───────────────────────────────────────────────────
        self._motion_start_time = 0
        if options is not None and "start_time" in options:
            self._motion_start_time = options["start_time"]
        elif self.random_start:
            motion_id = self._sampled_motion_id
            if motion_id in self.initial_pos_data:
                self._motion_start_time = self._sample_valid_start_time(
                    motion_id
                )

        # ── Heading randomization (per-episode) ─────────────────────────
        if self._randomize_heading:
            angle = np.random.uniform(-np.pi, np.pi)
            self._heading_rot = sRot.from_euler("z", angle)
        else:
            self._heading_rot = None

    def _sample_valid_start_time(self, motion_id: int) -> float:
        """
        Randomly picks a start time whose IK root height is within the valid range.
        Falls back to time 0 if no valid frame is found after several retries.
        """
        available_times = list(self.initial_pos_data[motion_id].keys())
        lo, hi = self.valid_root_height

        for _ in range(self.max_start_retries):
            t = np.random.choice(available_times)
            qpos = self._lookup_reference_qpos(motion_id, t)
            if qpos is not None and lo <= qpos[2] <= hi:
                return t

        # Fallback: return time 0 (standing pose is almost always valid)
        return 0

    def get_true_motion_id(self) -> int:
        """Returns the global motion ID (same as ``_sampled_motion_id``)."""
        return self._sampled_motion_id

    def _lookup_reference_qpos(
        self, motion_id: int, motion_time: float
    ) -> np.ndarray | None:
        """
        Returns the IK-computed qpos for the closest cached frame of the given motion.
        """
        motion_dict = self.initial_pos_data[motion_id]
        target_key = motion_time
        if target_key in motion_dict:
            return motion_dict[target_key]

        keys = np.array(list(motion_dict.keys()), dtype=float)
        nearest_key = keys[np.abs(keys - motion_time).argmin()]
        return motion_dict[nearest_key]

    def get_state_from_motionlib_cache(self, motion_time: np.ndarray) -> dict:
        """
        Retrieves (and caches) reference states using the Mujoco reference model.

        Returns:
            dict: Cached or newly computed reference state containing positions, rotations,
            velocities, and actuator readings.
        """
        if (
            "motion_id" not in self.ref_motion_cache
            or self.ref_motion_cache["motion_id"] != self._sampled_motion_id
            or abs(self.ref_motion_cache["motion_time"] - motion_time) > 1e-6
        ):
            self.ref_motion_cache["motion_id"] = self._sampled_motion_id
            self.ref_motion_cache["motion_time"] = motion_time

            ref_state = self._build_reference_state(motion_time)
            self.ref_motion_cache.update(ref_state)

        return self.ref_motion_cache

    def _build_reference_state(
        self, motion_time: np.ndarray
    ) -> dict[str, np.ndarray]:
        """
        Populates the reference Mujoco data buffer using pre-computed IK poses
        (with heading applied) and returns derived body/muscle states.
        """
        motion_id = self._sampled_motion_id

        ref_qpos = self._lookup_reference_qpos(motion_id, motion_time)
        headed = self._apply_heading(ref_qpos.copy())

        self.ref_data.qpos[: len(headed)] = headed
        self.ref_data.qvel[:] = 0

        if motion_time > 0:
            prev_ref_qpos = self._lookup_reference_qpos(
                motion_id, motion_time - self.dt
            )
            prev_headed = self._apply_heading(prev_ref_qpos.copy())
            mujoco.mj_differentiatePos(
                self.mj_model,
                self.ref_data.qvel,
                self.dt,
                prev_headed,
                headed,
            )
        mujoco.mj_forward(self.mj_model, self.ref_data)

        return {
            "xpos": self.get_body_xpos(self.ref_data)[None,],
            "xquat": self.get_body_xquat(self.ref_data)[None,],
            "body_vel": self.get_body_linear_vel(self.ref_data)[None,],
            "body_ang_vel": self.get_body_angular_vel(self.ref_data)[None,],
            "actuator_length": self.get_muscle_length(self.ref_data),
            "actuator_velocity": self.get_muscle_velocity(self.ref_data),
            "qpos": self.get_qpos(self.ref_data),
            "qvel": self.get_qvel(self.ref_data),
        }

    def compute_task_obs(self) -> np.ndarray:
        """
        Computes task-specific observations for the current simulation step.

        This function calculates and returns the observation vector based on the
        current and reference states of the simulated bodies. It includes positional
        and velocity differences as well as reference positions, tailored to the
        configured task inputs.

        Returns:
            np.ndarray: A concatenated array of task observations based on selected
            input features. The array is flattened for compatibility with downstream models.

        Updates:
            - Calls `record_evaluation_metrics` to update position and velocity metrics.
            - Calls `record_biomechanics` to store biomechanical data if enabled.

        Observation Features (if configured):
            - `diff_local_body_pos`: Differences in local body positions relative to references.
            - `diff_local_vel`: Differences in local body velocities relative to references.
            - `local_ref_body_pos`: Local reference body positions.
        """

        ref_dict = self.get_state_from_motionlib_cache(
            (self.cur_t + 1) * self.dt + self._motion_start_time
        )

        body_pos = self.get_body_xpos()[None,]
        body_rot = self.get_body_xquat()[None,]
        body_vel = self.get_body_linear_vel()[None,]
        muscle_len = self.get_muscle_length()
        muscle_vel = self.get_muscle_velocity()

        root_rot = body_rot[:, 0]
        root_pos = body_pos[:, 0]
        body_pos_subset = body_pos[..., self.tracked_bodies_id, :]
        body_rot_subset = body_rot[..., self.tracked_bodies_id, :]
        body_vel_subset = body_vel[..., self.tracked_bodies_id, :]

        ref_pos_subset = ref_dict.xpos[None,][..., self.tracked_bodies_id, :]
        ref_rot_subset = ref_dict.xquat[None,][..., self.tracked_bodies_id, :]
        ref_body_vel_subset = ref_dict.body_vel[None,][
            ..., self.tracked_bodies_id, :
        ]
        ref_muscle_len = ref_dict.actuator_length
        ref_muscle_vel = ref_dict.actuator_velocity

        if self.recording_biomechanics:
            self.record_biomechanics(
                body_pos_subset,
                body_rot_subset,
                body_vel_subset,
                ref_pos_subset,
                ref_rot_subset,
                ref_body_vel_subset,
            )

        full_task_obs = compute_imitation_observations(
            root_pos,
            root_rot,
            body_pos_subset,
            body_vel_subset,
            ref_pos_subset,
            ref_body_vel_subset,
            muscle_len,
            muscle_vel,
            ref_muscle_len,
            ref_muscle_vel,
            self.num_traj_samples,
        )

        task_obs = {}
        if "diff_local_body_pos" in self.cfg.run.task_inputs:
            task_obs["diff_local_body_pos"] = full_task_obs[
                "diff_local_body_pos"
            ]
        if "diff_local_vel" in self.cfg.run.task_inputs:
            task_obs["diff_local_vel"] = full_task_obs["diff_local_vel"]
        if "local_ref_body_pos" in self.cfg.run.task_inputs:
            task_obs["local_ref_body_pos"] = full_task_obs["local_ref_body_pos"]
        if "diff_muscle_len" in self.cfg.run.task_inputs:
            task_obs["diff_muscle_len"] = full_task_obs["diff_muscle_len"]
        if "diff_muscle_vel" in self.cfg.run.task_inputs:
            task_obs["diff_muscle_vel"] = full_task_obs["diff_muscle_vel"]

        return np.concatenate(
            [v.ravel() for v in task_obs.values()], axis=0, dtype=self.dtype
        )

    def record_biomechanics(
        self,
        body_pos: np.ndarray,
        body_rot: np.ndarray,
        body_vel: np.ndarray,
        ref_pos: np.ndarray,
        ref_rot: np.ndarray,
        ref_vel: np.ndarray,
    ) -> None:
        """
        Records biomechanical data for the current simulation step.

        Captures and stores data related to body states, reference states, joint
        positions and velocities, and muscle forces/controls for biomechanical analysis.

        Args:
            body_pos (np.ndarray): Current body positions.
            body_rot (np.ndarray): Current body rotations.
            body_vel (np.ndarray): Current body velocities.
            ref_pos (np.ndarray): Reference body positions.
            ref_rot (np.ndarray): Reference body rotations.
            ref_vel (np.ndarray): Reference body velocities.

        Updates:
            - `self.feet`: Tracks foot contact states (e.g., left, right, or both planted).
            - `self.joint_pos`, `self.joint_vel`: Joint positions and velocities.
            - `self.body_pos`, `self.body_rot`, `self.body_vel`: Current body states.
            - `self.ref_pos`, `self.ref_rot`, `self.ref_vel`: Reference body states.
            - `self.motion_id`: Current motion ID.
            - `self.muscle_forces`, `self.muscle_controls`: Muscle forces and control inputs.
        """
        feet_contacts = self.proprioception["feet_contacts"]
        planted_feet = -1
        if feet_contacts[0] > 0 or feet_contacts[1] > 0:
            planted_feet = 1
        if feet_contacts[2] > 0 or feet_contacts[3] > 0:
            planted_feet = 0
        if (feet_contacts[0] > 0 or feet_contacts[1] > 0) and (
            feet_contacts[2] > 0 or feet_contacts[3] > 0
        ):
            planted_feet = 2
        self.feet.append(planted_feet)
        self.joint_pos.append(self.get_qpos().copy())
        self.joint_vel.append(self.get_qvel().copy())
        self.body_pos.append(body_pos.copy())
        self.body_rot.append(body_rot.copy())
        self.body_vel.append(body_vel.copy())
        self.ref_pos.append(ref_pos.copy())
        self.ref_rot.append(ref_rot.copy())
        self.ref_vel.append(ref_vel.copy())
        self.motion_id.append(self._sampled_motion_id)
        self.muscle_forces.append(self.get_muscle_force().copy())
        self.muscle_controls.append(self.mj_data.ctrl.copy())

    def record_evaluation_metrics(
        self,
        body_pos: np.ndarray,
        ref_pos: np.ndarray,
        body_vel: np.ndarray,
        ref_vel: np.ndarray,
    ) -> None:
        """
        Records evaluation metrics (MPJPE) for the current simulation step.

        Args:
            body_pos (np.ndarray): Current body positions.
            ref_pos (np.ndarray): Reference body positions.
            body_vel (np.ndarray): Current body velocities.
            ref_vel (np.ndarray): Reference body velocities.

        Updates:
            - `self.mpjpe`: Appends the mean position error for the current step.
        """
        self.mpjpe.append(np.linalg.norm(body_pos - ref_pos, axis=-1).mean())

    def compute_muscle_imitation_reward(
        self, ref_state: dict
    ) -> tuple[float, dict[str, float]]:
        """
        Matches simulated muscle length/velocity to the IK reference.
        """
        ref_lengths = ref_state.actuator_length[0]
        ref_velocities = ref_state.actuator_velocity[0]
        curr_lengths = np.nan_to_num(self.mj_data.actuator_length.copy())
        curr_velocities = np.nan_to_num(self.mj_data.actuator_velocity.copy())

        k_len = self.reward_specs.get("k_muscle_len", 10.0)
        k_vel = self.reward_specs.get("k_muscle_vel", 1.0)
        w_len = self.reward_specs.get("w_muscle_len", 0.0)
        w_vel = self.reward_specs.get("w_muscle_vel", 0.0)

        r_len = np.exp(-k_len * ((curr_lengths - ref_lengths) ** 2).mean())
        r_vel = np.exp(
            -k_vel * ((curr_velocities - ref_velocities) ** 2).mean()
        )

        reward = w_len * r_len + w_vel * r_vel
        reward_raw = {"r_muscle_len": r_len, "r_muscle_vel": r_vel}
        return reward, reward_raw

    def compute_reward(self, action: np.ndarray | None = None) -> float:
        """
        Computes the reward for the current simulation step.

        The reward is a combination of imitation reward, upright posture reward,
        and energy efficiency. It is calculated by comparing the current body state
        to the reference motion and includes weighted contributions based on the
        reward specifications.

        Args:
            action (Optional[np.ndarray]): The action taken at the current step. Defaults to None.

        Returns:
            float: The total reward for the current simulation step.

        Updates:
            - `self.reward_info`: Stores raw reward components for analysis.
        """
        ref_dict = self.get_state_from_motionlib_cache(
            (self.cur_t) * self.dt + self._motion_start_time
        )
        muscle_reward, muscle_reward_raw = self.compute_muscle_imitation_reward(
            ref_dict
        )

        body_pos = self.get_body_xpos()[None,]

        body_pos_subset = body_pos[..., self.tracked_bodies_id, :]
        ref_pos_subset = ref_dict.xpos[..., self.tracked_bodies_id, :]

        body_vel = self.get_body_linear_vel()[None,]
        body_vel_subset = body_vel[..., self.tracked_bodies_id, :]
        ref_body_vel_subset = ref_dict.body_vel[..., self.tracked_bodies_id, :]

        reward, reward_raw = compute_imitation_reward(
            body_pos_subset,
            body_vel_subset,
            ref_pos_subset,
            ref_body_vel_subset,
            self.reward_specs,
            body_weights=self.body_reward_weights,
        )

        energy_cost = np.mean(self.curr_power_usage)
        self.curr_power_usage = []

        reward -= self.reward_specs["w_energy"] * energy_cost
        reward += muscle_reward

        self.reward_info = reward_raw
        self.reward_info["energy_cost"] = energy_cost
        self.reward_info.update(muscle_reward_raw)

        self.record_evaluation_metrics(
            body_pos_subset,
            ref_pos_subset,
            body_vel_subset,
            ref_body_vel_subset,
        )

        return reward

    def start_eval(self, im_eval=True):
        """
        Prepares the environment for evaluation (no heading randomization).
        """
        self._randomize_heading = False
        self._temp_random_start = self.random_start
        self.random_start = False
        self.im_eval = im_eval
        self.test = True

    def end_eval(self):
        """
        Concludes the evaluation process and restores training settings.
        """
        self._randomize_heading = True
        self.random_start = self._temp_random_start
        self.im_eval = False
        self.test = False
        self.sample_motions()

    def get_muscle_force(self) -> np.ndarray:
        """
        Retrieves the muscle forces from the simulation.
        """
        return self.mj_data.actuator_force

    # ──────────────────────────────────────────────────────────────────────
    # Heading helpers
    # ──────────────────────────────────────────────────────────────────────
    def _apply_heading(self, qpos: np.ndarray) -> np.ndarray:
        """
        Applies the current episode heading rotation to the root pose (first 7 qpos).
        Joint angles (qpos[7:]) are left unchanged.

        If ``self._heading_rot`` is None (heading disabled), returns qpos unmodified.
        """
        if self._heading_rot is None:
            return qpos

        # Rotate translation (X, Y change; Z = height stays)
        qpos[:3] = self._heading_rot.apply(qpos[:3])

        # Compose heading with root quaternion (Mujoco wxyz format)
        root_xyzw = qpos[np.array([4, 5, 6, 3])]  # wxyz → xyzw
        new_xyzw = (self._heading_rot * sRot.from_quat(root_xyzw)).as_quat()
        qpos[3:7] = np.roll(new_xyzw, 1)  # xyzw → wxyz

        return qpos

    def get_motion_length(self, motion_id: int) -> float:
        """
        Returns the duration (seconds) of the given motion.
        Falls back to counting frame keys if metadata is unavailable.
        """
        if motion_id in self._motion_metadata:
            return self._motion_metadata[motion_id]["length"]
        # Fallback: infer from stored frame times
        if motion_id in self.initial_pos_data:
            times = list(self.initial_pos_data[motion_id].keys())
            return max(times) if times else 0.0
        return 0.0

    def post_physics_step(self, action):
        """
        Adds imitation-eval metrics (mpjpe, frame_coverage, success) to info
        on the terminal step. Metrics are computed inside compute_reset() and
        remain on self until this point.
        """
        obs, reward, terminated, truncated, info = super().post_physics_step(
            action
        )
        if terminated or truncated:
            info["mpjpe"] = float(self.mpjpe_value)
            info["max_mpjpe"] = float(self.max_mpjpe_value)
            info["frame_coverage"] = float(self.frame_coverage)
            info["success"] = bool(self.last_success)
        if terminated and hasattr(self, "last_termination_body"):
            info["termination_body"] = self.last_termination_body
            info["termination_dists"] = {
                b: float(d)
                for b, d in zip(self.reset_bodies, self.last_termination_dist)
            }
        return obs, reward, terminated, truncated, info

    def step(self, action):
        """
        Executes a single step in the environment with the given action.

        Args:
            action: The action to apply at the current step.

        Returns:
            Tuple:
                - observation (np.ndarray): Current observations after the step.
                - reward (float): Reward for the applied action.
                - terminated (bool): Whether the task has terminated prematurely.
                - truncated (bool): Whether the task has exceeded its allowed time.
                - info (dict): Additional information about the step, including reward details.
        """
        if self.recording_biomechanics:
            self.policy_outputs.append(action)

        self.physics_step(action)
        observation, reward, terminated, truncated, info = (
            self.post_physics_step(action)
        )

        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, truncated, info


def compute_imitation_observations(
    root_pos: np.ndarray,
    root_rot: np.ndarray,
    body_pos: np.ndarray,
    body_vel: np.ndarray,
    ref_body_pos: np.ndarray,
    ref_body_vel: np.ndarray,
    muscle_len: np.ndarray,
    muscle_vel: np.ndarray,
    ref_muscle_len: np.ndarray,
    ref_muscle_vel: np.ndarray,
    time_steps: int,
) -> dict[str, np.ndarray]:
    """
    Computes imitation observations based on differences between current and reference states.

    Observations include local differences in body positions and velocities,
    as well as local reference positions relative to the root.

    Args:
        root_pos (np.ndarray): Root position of the current state.
        root_rot (np.ndarray): Root rotation of the current state.
        body_pos (np.ndarray): Current body positions.
        body_vel (np.ndarray): Current body velocities.
        ref_body_pos (np.ndarray): Reference body positions.
        ref_body_vel (np.ndarray): Reference body velocities.
        time_steps (int): Number of time steps for observation history.

    Returns:
        Dict[str, np.ndarray]: A dictionary containing:
            - `diff_local_body_pos`: Differences in local body positions.
            - `diff_local_vel`: Differences in local body velocities.
            - `local_ref_body_pos`: Local reference body positions.
    """
    obs = OrderedDict()
    B, J, _ = body_pos.shape

    heading_inv_rot = npt_utils.calc_heading_quat_inv(root_rot)

    heading_inv_rot_expand = np.tile(
        heading_inv_rot[..., None, :, :].repeat(body_pos.shape[1], axis=1),
        (time_steps, 1, 1),
    )

    diff_global_body_pos = ref_body_pos.reshape(
        B, time_steps, J, 3
    ) - body_pos.reshape(B, 1, J, 3)

    diff_local_body_pos_flat = npt_utils.quat_rotate(
        heading_inv_rot_expand.reshape(-1, 4),
        diff_global_body_pos.reshape(-1, 3),
    )

    obs["diff_local_body_pos"] = diff_local_body_pos_flat  # 1 * J * 3

    # Velocities
    diff_global_vel = ref_body_vel.reshape(
        B, time_steps, J, 3
    ) - body_vel.reshape(B, 1, J, 3)
    obs["diff_local_vel"] = npt_utils.quat_rotate(
        heading_inv_rot_expand.reshape(-1, 4), diff_global_vel.reshape(-1, 3)
    )

    local_ref_body_pos = ref_body_pos.reshape(
        B, time_steps, J, 3
    ) - root_pos.reshape(B, 1, 1, 3)  # preserves the body position
    obs["local_ref_body_pos"] = npt_utils.quat_rotate(
        heading_inv_rot_expand.reshape(-1, 4), local_ref_body_pos.reshape(-1, 3)
    )

    obs["diff_muscle_len"] = ref_muscle_len - muscle_len
    obs["diff_muscle_vel"] = ref_muscle_vel - muscle_vel

    return obs


def compute_imitation_reward(
    body_pos: np.ndarray,
    body_vel: np.ndarray,
    ref_body_pos: np.ndarray,
    ref_body_vel: np.ndarray,
    rwd_specs: dict,
    body_weights: np.ndarray | None = None,
) -> tuple[float, dict[str, np.ndarray]]:
    """
    Computes the imitation reward based on differences in positions and velocities
    between the current and reference states.

    Args:
        body_pos (np.ndarray): Current body positions.
        body_vel (np.ndarray): Current body velocities.
        ref_body_pos (np.ndarray): Reference body positions.
        ref_body_vel (np.ndarray): Reference body velocities.
        rwd_specs (dict): Reward specifications containing:
            - `"k_pos"`: Scaling factor for position reward.
            - `"k_vel"`: Scaling factor for velocity reward.
            - `"w_pos"`: Weight for position reward.
            - `"w_vel"`: Weight for velocity reward.
        body_weights (Optional[np.ndarray]): Per-body weights, shape (num_bodies,),
            must sum to 1. None = uniform mean (original behavior).

    Returns:
        Tuple:
            - reward (float): Weighted sum of position and velocity rewards.
            - reward_raw (Dict[str, np.ndarray]): Dictionary of raw reward components.
    """
    k_pos, k_vel = rwd_specs["k_pos"], rwd_specs["k_vel"]
    w_pos, w_vel = rwd_specs["w_pos"], rwd_specs["w_vel"]

    # body position reward — per-body squared error, then (weighted) mean over bodies
    diff_global_body_pos = ref_body_pos - body_pos
    per_body_pos_err = (diff_global_body_pos**2).mean(
        axis=-1
    )  # (..., num_bodies)
    if body_weights is not None:
        diff_body_pos_dist = (per_body_pos_err * body_weights).sum(axis=-1)
    else:
        diff_body_pos_dist = per_body_pos_err.mean(axis=-1)
    r_body_pos = np.exp(-k_pos * diff_body_pos_dist)

    # body linear velocity reward
    diff_global_vel = ref_body_vel - body_vel
    per_body_vel_err = (diff_global_vel**2).mean(axis=-1)
    if body_weights is not None:
        diff_global_vel_dist = (per_body_vel_err * body_weights).sum(axis=-1)
    else:
        diff_global_vel_dist = per_body_vel_err.mean(axis=-1)
    r_vel = np.exp(-k_vel * diff_global_vel_dist)

    reward = w_pos * r_body_pos + w_vel * r_vel
    reward_raw = {
        "r_body_pos": r_body_pos,
        "r_vel": r_vel,
    }

    return reward[0], reward_raw


def compute_humanoid_im_reset(
    rigid_body_pos, ref_body_pos, termination_distance, use_mean
) -> np.ndarray:
    """
    Determines whether the humanoid should reset based on deviations from reference positions.

    Args:
        rigid_body_pos (np.ndarray): Current positions of the humanoid's rigid bodies.
        ref_body_pos (np.ndarray): Reference positions of the humanoid's rigid bodies.
        termination_distance (float or np.ndarray): Threshold distance for termination.
            If ndarray, per-body thresholds with shape (num_bodies,).
        use_mean (bool): Whether to use the mean or maximum deviation for the reset condition.

    Returns:
        bool: Indicates whether the humanoid has exceeded the termination distance.
    """
    per_body_dist = np.linalg.norm(rigid_body_pos - ref_body_pos, axis=-1)

    if use_mean:
        has_fallen = np.any(
            per_body_dist.mean(axis=-1, keepdims=True)
            > (
                np.mean(termination_distance)
                if isinstance(termination_distance, np.ndarray)
                else termination_distance
            ),
            axis=-1,
        )
    else:
        # per_body_dist shape: (batch, num_bodies)
        # termination_distance: scalar or (num_bodies,) — broadcasts naturally
        has_fallen = np.any(
            per_body_dist > termination_distance,
            axis=-1,
        )

    return has_fallen
