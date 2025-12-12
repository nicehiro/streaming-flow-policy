"""
ReplayMotion.py - Motion replay script for recorded robot trajectories

Replays recorded motion data from JSON files with configurable control mode and playback speed.
"""

import os

cpath = os.path.dirname(os.path.abspath(__file__))
import numpy as np

np.set_printoptions(precision=5, suppress=True, linewidth=200)
import mujoco
import mujoco.viewer
import time
import json
import cv2
from copy import deepcopy
from scipy.spatial.transform import Rotation as R
from BaseMJEnv import Mj_Env


class ReplayMotion(Mj_Env):
    """
    Motion replay class that extends Mj_Env base class.

    Replays recorded robot motion and object trajectories from JSON data files.
    """

    def __init__(
        self,
        path=None,
        data_path=None,
        ctrl_mode="ctrl",
        speed_factor=1.0,
        save_video=False,
        output_path=None,
        video_fps=50,
        video_width=1920,
        video_height=1080,
        arm_trace_distance=0.05,
        arm_trace_count=10,
        object_trace_distance=0.05,
        object_trace_count=10,
        trace_start_delay=0.0,
        camera_lookat=None,
        camera_distance=None,
        camera_azimuth=None,
        camera_elevation=None,
    ):
        """
        Initialize the ReplayMotion environment.

        Args:
            path: Path to MuJoCo XML scene file
            data_path: Path to JSON data file containing recorded motion
            ctrl_mode: Control mode - 'qpos' for direct position, 'ctrl' for control signal
            speed_factor: Playback speed multiplier (1.0 = real-time, 2.0 = 2x speed)
            save_video: Whether to save the replay as a video file
            output_path: Path for output video file (default: auto-generated from data file)
            video_fps: Frames per second for output video
            video_width: Video width in pixels
            video_height: Video height in pixels
            arm_trace_distance: Min EEF distance in meters between arm traces (default: 0.05)
            arm_trace_count: Max arm trace snapshots (default: 10)
            object_trace_distance: Min object distance in meters between object traces (default: 0.05)
            object_trace_count: Max object trace snapshots (default: 10)
            trace_start_delay: Time in seconds to wait before starting trace recording (default: 0.0)
            camera_lookat: Camera lookat point [x, y, z] (default: [-1, 1, 0.6])
            camera_distance: Camera distance from lookat point (default: 3.3)
            camera_azimuth: Camera azimuth angle in degrees (default: 130)
            camera_elevation: Camera elevation angle in degrees (default: -30)
        """
        # Store save_video before calling super().__init__ so _set_viewer can access it
        self.save_video = save_video

        super(ReplayMotion, self).__init__(path=path)

        self.ctrl_mode = ctrl_mode
        self.speed_factor = speed_factor
        self.data_path = data_path
        self.output_path = output_path
        self.video_fps = video_fps
        self.video_width = video_width
        self.video_height = video_height
        self.video_writer = None
        self.video_renderer = None
        self.video_camera = None

        # Camera settings (None = use defaults)
        self.camera_lookat = camera_lookat
        self.camera_distance = camera_distance
        self.camera_azimuth = camera_azimuth
        self.camera_elevation = camera_elevation

        # Arm trace configuration (EEF-based, space-based sampling)
        self.arm_trace_distance = arm_trace_distance
        self.arm_trace_count = arm_trace_count
        self.arm_trace_snapshots = []
        self.arm_trace_last_pos = None
        self.arm_trace_complete = False
        self.arm_trace_body_ids = []
        self.arm_trace_geom_map = {}

        # Object trace configuration (object position-based, space-based sampling)
        self.object_trace_distance = object_trace_distance
        self.object_trace_count = object_trace_count
        self.object_trace_snapshots = []
        self.object_trace_last_pos = None
        self.object_trace_complete = False
        self.object_trace_body_ids = []
        self.object_trace_geom_map = {}

        # Trace start delay (wait before recording traces)
        self.trace_start_delay = trace_start_delay

        self._get_ids()

        # Set default camera for interactive viewer
        self._set_default_camera()

        # Data storage
        self.recorded_data = None
        self.episodes = None
        self.num_timesteps = 0
        self.object_type = None
        self.object_body_id = None

        if data_path:
            self._load_data(data_path)

    def _set_viewer(self):
        """Override to skip interactive viewer when saving video (avoids GLX conflicts)."""
        if self.save_video:
            self.viewer = None
            print("Skipping interactive viewer (video-only mode)")
        else:
            super()._set_viewer()

    def _get_ids(self):
        """Get MuJoCo body/joint IDs for robot and objects."""
        # Robot joint IDs (7 DOF arm)
        self.joint_ids = [
            mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, f"left_joint{i + 1}"
            )
            for i in range(7)
        ]
        print(f"Joint IDs: {self.joint_ids}")

        # Gripper actuator ID (controls gripper via tendon)
        self.gripper_actuator_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_gripper_act"
        )
        print(f"Gripper actuator ID: {self.gripper_actuator_id}")

        # Gripper joint names and their qpos addresses for direct kinematic control
        # The Robotiq 2F85 uses a complex linkage system - we need to set all joints
        gripper_joint_names = [
            # Driver joints (main control, range 0-0.8)
            "left_right_driver_joint",
            "left_left_driver_joint",
            # Coupler joints (coupled to drivers via constraints, range -1.57 to 0)
            "left_right_coupler_joint",
            "left_left_coupler_joint",
            # Spring link joints (range -0.297 to 0.8)
            "left_right_spring_link_joint",
            "left_left_spring_link_joint",
            # Follower joints (range -0.873 to 0.873)
            "left_right_follower_joint",
            "left_left_follower_joint",
        ]
        self.gripper_joint_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in gripper_joint_names
        ]
        self.gripper_joint_qpos_addrs = [
            self.model.jnt_qposadr[jid] for jid in self.gripper_joint_ids
        ]
        print(f"Gripper joint IDs: {self.gripper_joint_ids}")
        print(f"Gripper joint qpos addresses: {self.gripper_joint_qpos_addrs}")

        # Robot link IDs
        self.link_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"left_link{i + 1}")
            for i in range(7)
        ]
        self.link_ids.append(
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "left_end_effector")
        )
        print(f"Link IDs: {self.link_ids}")

        # End effector body ID for trace distance calculation
        self.eef_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "left_end_effector"
        )

        # Object body IDs and joint addresses (for free bodies)
        self.circle_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "circle2"
        )
        self.circle_center_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "circle2_center"
        )
        self.square_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "square"
        )
        self.triangle_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "triangle"
        )

        # Get qpos addresses for free bodies (to hide them by moving position)
        # Use body_jntadr to get joint index, then jnt_qposadr for qpos address
        self.circle_qpos_addr = self.model.jnt_qposadr[
            self.model.body_jntadr[self.circle_body_id]
        ]
        self.square_qpos_addr = self.model.jnt_qposadr[
            self.model.body_jntadr[self.square_body_id]
        ]
        self.triangle_qpos_addr = self.model.jnt_qposadr[
            self.model.body_jntadr[self.triangle_body_id]
        ]
        print(
            f"Object Body IDs - Circle: {self.circle_body_id}, Square: {self.square_body_id}, Triangle: {self.triangle_body_id}"
        )

        # Target column body IDs
        self.pillar_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "pillar1"
        )
        self.pillar_base_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "pillar_base1"
        )
        print(
            f"Target Column IDs - Pillar: {self.pillar_body_id}, Base: {self.pillar_base_body_id}"
        )

    def _set_default_camera(self):
        """Set default camera settings for both interactive viewer and video recording."""
        # Camera settings: use provided values or fall back to defaults
        self.default_camera_params = {
            "lookat": np.array(self.camera_lookat) if self.camera_lookat is not None else np.array([-1, 1, 0.6]),
            "distance": self.camera_distance if self.camera_distance is not None else 3.3,
            "azimuth": self.camera_azimuth if self.camera_azimuth is not None else 130,
            "elevation": self.camera_elevation if self.camera_elevation is not None else -30,
        }

        # Apply to interactive viewer
        if self.viewer is not None:
            self._set_camera_params(self.default_camera_params)

    def _load_data(self, data_path):
        """
        Load recorded motion data from JSON file.

        Args:
            data_path: Path to JSON data file
        """
        print(f"Loading data from: {data_path}")

        with open(data_path, "r") as f:
            self.recorded_data = json.load(f)

        # Extract metadata
        self.object_encoding = self.recorded_data["object"]
        self.target = self.recorded_data["target"]
        self.release_time = self.recorded_data["release_time"]
        self.episodes = self.recorded_data["episodes"]
        self.num_timesteps = len(self.episodes)

        # Determine object type from one-hot encoding
        # [1,0,0] = cirle, [0,1,0] = square, [0,0,1] = triangle
        if self.object_encoding[0] == 1:
            self.object_type = "circle"
            self.object_body_id = self.circle_body_id
        elif self.object_encoding[1] == 1:
            self.object_type = "square"
            self.object_body_id = self.square_body_id
        elif self.object_encoding[2] == 1:
            self.object_type = "triangle"
            self.object_body_id = self.triangle_body_id

        print(f"Loaded {self.num_timesteps} timesteps")
        print(f"Object type: {self.object_type}")
        print(f"Target: {self.target}")
        print(f"Release time: {self.release_time:.3f} seconds")

        # Set target column position
        self._set_target_position()

        # Hide objects not used in this replay
        self._hide_unused_objects()

    def _set_target_position(self):
        """Set target column position based on recorded target value."""
        if self.target is None:
            return

        target_pos = np.array(self.target)
        pillar_height = 0.180  # Half-height of the pillar cylinder
        base_height = 0.02  # Half-height of the base box

        # Set pillar position (target x, y, with z = pillar_height)
        self.model.body_pos[self.pillar_body_id] = [
            target_pos[0],
            target_pos[1],
            pillar_height,
        ]

        # Set pillar base position (target x, y, with z = base_height)
        self.model.body_pos[self.pillar_base_body_id] = [
            target_pos[0],
            target_pos[1],
            base_height,
        ]

        print(f"Target column moved to: ({target_pos[0]:.3f}, {target_pos[1]:.3f})")

    def _hide_unused_objects(self):
        """Hide objects that are not being used in this replay by moving them off-screen."""
        hidden_pos = [0, 0, -10]  # Move far below the scene

        # For free bodies, we need to modify data.qpos (first 3 values are x,y,z position)
        if self.object_type != "circle" and self.circle_body_id >= 0:
            self.data.qpos[self.circle_qpos_addr : self.circle_qpos_addr + 3] = (
                hidden_pos
            )
        if self.object_type != "square" and self.square_body_id >= 0:
            self.data.qpos[self.square_qpos_addr : self.square_qpos_addr + 3] = (
                hidden_pos
            )
        if self.object_type != "triangle" and self.triangle_body_id >= 0:
            self.data.qpos[self.triangle_qpos_addr : self.triangle_qpos_addr + 3] = (
                hidden_pos
            )

        mujoco.mj_forward(self.model, self.data)  # Update positions
        print(f"Hidden unused objects (keeping {self.object_type})")

    def _init_replay_trace(self):
        """
        Initialize motion trace for replay visualization.
        Sets up separate arm and object tracing with independent distance thresholds.
        - Arm traces: based on EEF position, blue→green gradient
        - Object traces: based on object position, solid orange
        """
        # Reset arm trace state
        self.arm_trace_snapshots = []
        self.arm_trace_last_pos = None
        self.arm_trace_complete = False

        # Reset object trace state
        self.object_trace_snapshots = []
        self.object_trace_last_pos = None
        self.object_trace_complete = False

        # Arm body IDs to trace (7 arm links + gripper, exclude end effector sphere)
        # Start with the 7 arm links
        self.arm_trace_body_ids = list(self.link_ids[:-1])

        # Add gripper bodies (all descendants of left_link7, excluding end effector)
        # left_link7 is the last item in link_ids[:-1], i.e., link_ids[-2]
        link7_body_id = self.link_ids[-2]  # left_link7
        end_effector_id = self.link_ids[-1]  # left_end_effector (to exclude)

        # BFS to find all descendants of link7 (gripper bodies)
        to_visit = []
        for body_id in range(self.model.nbody):
            if self.model.body_parentid[body_id] == link7_body_id:
                to_visit.append(body_id)

        while to_visit:
            current_body = to_visit.pop(0)
            # Skip the end effector (it's just a transparent sphere marker)
            if current_body == end_effector_id:
                continue
            self.arm_trace_body_ids.append(current_body)
            # Find all direct children of current body
            for body_id in range(self.model.nbody):
                if self.model.body_parentid[body_id] == current_body:
                    to_visit.append(body_id)

        # Object body IDs to trace (recursively find all descendants)
        self.object_trace_body_ids = []
        if self.object_body_id is not None:
            # Use BFS to find all descendants (object may have nested child bodies with geoms)
            to_visit = [self.object_body_id]
            while to_visit:
                current_body = to_visit.pop(0)
                self.object_trace_body_ids.append(current_body)
                # Find all direct children of current body
                for body_id in range(self.model.nbody):
                    if self.model.body_parentid[body_id] == current_body:
                        to_visit.append(body_id)

        # Build geom-to-body mapping for arm
        self.arm_trace_geom_map = {body_id: [] for body_id in self.arm_trace_body_ids}
        for gid in range(self.model.ngeom):
            body_id = self.model.geom_bodyid[gid]
            if body_id in self.arm_trace_geom_map:
                self.arm_trace_geom_map[body_id].append(gid)

        # Build geom-to-body mapping for object
        self.object_trace_geom_map = {
            body_id: [] for body_id in self.object_trace_body_ids
        }
        for gid in range(self.model.ngeom):
            body_id = self.model.geom_bodyid[gid]
            if body_id in self.object_trace_geom_map:
                self.object_trace_geom_map[body_id].append(gid)

        print(
            f"Arm trace: max {self.arm_trace_count} snapshots, distance {self.arm_trace_distance}m"
        )
        print(
            f"Object trace: max {self.object_trace_count} snapshots, distance {self.object_trace_distance}m"
        )

    def _record_arm_trace(self):
        """
        Record arm trace snapshot based on EEF distance traveled.

        Samples when the end effector has moved at least arm_trace_distance
        from the last recorded position, ensuring even spatial distribution.
        """
        # Stop if max trace count reached
        if len(self.arm_trace_snapshots) >= self.arm_trace_count:
            if not self.arm_trace_complete:
                self.arm_trace_complete = True
                print(f"Arm trace complete: {len(self.arm_trace_snapshots)} snapshots")
            return

        # Get current EEF position
        current_pos = self.data.xpos[self.eef_body_id].copy()

        # First snapshot: always record
        if self.arm_trace_last_pos is None:
            self.arm_trace_snapshots.append(self.data.qpos.copy())
            self.arm_trace_last_pos = current_pos
            return

        # Check distance threshold
        distance = np.linalg.norm(current_pos - self.arm_trace_last_pos)
        if distance >= self.arm_trace_distance:
            self.arm_trace_snapshots.append(self.data.qpos.copy())
            self.arm_trace_last_pos = current_pos

    def _record_object_trace(self):
        """
        Record object trace snapshot based on object position distance traveled.

        Samples when the object has moved at least object_trace_distance
        from the last recorded position, ensuring even spatial distribution.
        """
        # Skip if no object
        if self.object_body_id is None:
            return

        # Stop if max trace count reached
        if len(self.object_trace_snapshots) >= self.object_trace_count:
            if not self.object_trace_complete:
                self.object_trace_complete = True
                print(
                    f"Object trace complete: {len(self.object_trace_snapshots)} snapshots"
                )
            return

        # Get current object position
        current_pos = self.data.xpos[self.object_body_id].copy()

        # First snapshot: always record
        if self.object_trace_last_pos is None:
            self.object_trace_snapshots.append(self.data.qpos.copy())
            self.object_trace_last_pos = current_pos
            return

        # Check distance threshold
        distance = np.linalg.norm(current_pos - self.object_trace_last_pos)
        if distance >= self.object_trace_distance:
            self.object_trace_snapshots.append(self.data.qpos.copy())
            self.object_trace_last_pos = current_pos

    def _compute_arm_trace_color(self, t):
        """
        Compute RGBA color for an arm trace snapshot based on normalized time.

        Args:
            t: normalized time [0,1] where 0=oldest, 1=newest

        Returns:
            np.ndarray: [R, G, B, A] in range [0, 1]
        """
        # Soft blue: RGB ~(0.4, 0.6, 0.9) at t=0
        # Soft green: RGB ~(0.4, 0.8, 0.5) at t=1
        r = 0.4  # Constant low red
        g = 0.6 + 0.2 * t  # 0.6 → 0.8 (increasing green)
        b = 0.9 - 0.4 * t  # 0.9 → 0.5 (decreasing blue)
        alpha = 0.1 + 0.1 * t  # Fading: 0.1 → 0.2
        return np.array([r, g, b, alpha], dtype=np.float32)

    def _compute_object_trace_color(self):
        """
        Return solid orange color for object traces.

        Returns:
            np.ndarray: [R, G, B, A] in range [0, 1]
        """
        return np.array([1.0, 0.5, 0.0, 0.3], dtype=np.float32)

    def _add_ghost_geom_to_scene(self, scene, ghost_data, geom_id, rgba):
        """
        Add one ghost geometry to a scene.

        Args:
            scene: MjvScene to add geometry to
            ghost_data: MjData with ghost pose
            geom_id: geometry ID to render
            rgba: [R, G, B, A] color
        """
        if scene.ngeom >= scene.maxgeom:
            return  # Graceful degradation

        geom_type = self.model.geom_type[geom_id]

        # Use mjv_initGeom for all geometry types - it handles transformations correctly
        mujoco.mjv_initGeom(
            scene.geoms[scene.ngeom],
            geom_type,
            self.model.geom_size[geom_id],
            ghost_data.geom_xpos[geom_id],
            ghost_data.geom_xmat[geom_id],
            rgba=rgba,
        )

        # For mesh geometries, set the mesh data reference
        if geom_type == mujoco.mjtGeom.mjGEOM_MESH:
            g = scene.geoms[scene.ngeom]
            mesh_id = self.model.geom_dataid[geom_id]
            # MuJoCo internally uses dataid*2 for mesh (vs convex hull)
            g.dataid = mesh_id * 2
            g.objtype = mujoco.mjtObj.mjOBJ_GEOM
            g.objid = geom_id

        scene.ngeom += 1

    def _render_trace_to_scene(self, scene):
        """
        Render arm and object trace ghost geometries to a scene.

        Args:
            scene: MjvScene to render ghosts to
        """
        scene.ngeom = 0  # Clear previous frame's ghosts

        # Render arm traces (blue→green gradient)
        num_arm = len(self.arm_trace_snapshots)
        for i, qpos in enumerate(self.arm_trace_snapshots):
            # Create ghost data with this pose
            ghost_data = mujoco.MjData(self.model)
            ghost_data.qpos[:] = qpos
            ghost_data.qvel[:] = 0  # Zero velocities for static pose
            mujoco.mj_kinematics(self.model, ghost_data)  # Compute positions only

            # Time-based color and opacity
            t = i / (num_arm - 1) if num_arm > 1 else 0.5
            color = self._compute_arm_trace_color(t)

            # Render all geometries for arm bodies
            for body_id in self.arm_trace_body_ids:
                for geom_id in self.arm_trace_geom_map.get(body_id, []):
                    self._add_ghost_geom_to_scene(scene, ghost_data, geom_id, color)

        # Render object traces (solid orange)
        object_color = self._compute_object_trace_color()
        for qpos in self.object_trace_snapshots:
            # Create ghost data with this pose
            ghost_data = mujoco.MjData(self.model)
            ghost_data.qpos[:] = qpos
            ghost_data.qvel[:] = 0  # Zero velocities for static pose
            mujoco.mj_kinematics(self.model, ghost_data)  # Compute positions only

            # Render all geometries for object bodies
            for body_id in self.object_trace_body_ids:
                for geom_id in self.object_trace_geom_map.get(body_id, []):
                    self._add_ghost_geom_to_scene(
                        scene, ghost_data, geom_id, object_color
                    )

    def _set_object_pose(self, tf_matrix):
        """
        Set object pose from 4x4 transformation matrix.

        Args:
            tf_matrix: 4x4 homogeneous transformation matrix
        """
        tf = np.array(tf_matrix)
        position = tf[:3, 3]
        rotation_matrix = tf[:3, :3]

        # Convert rotation matrix to quaternion
        # scipy returns [x, y, z, w], MuJoCo uses [w, x, y, z]
        rot = R.from_matrix(rotation_matrix)
        quat_xyzw = rot.as_quat()
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])

        # Find the first joint belonging to this body (works for unnamed freejoints)
        body_jntadr = self.model.body_jntadr[self.object_body_id]
        if body_jntadr >= 0:
            qpos_addr = self.model.jnt_qposadr[body_jntadr]
            # Set position (3 DOF)
            self.data.qpos[qpos_addr : qpos_addr + 3] = position
            # Set quaternion (4 DOF) in [w, x, y, z] format
            self.data.qpos[qpos_addr + 3 : qpos_addr + 7] = quat_wxyz
            # Zero out velocities to prevent physics interference
            vel_addr = self.model.jnt_dofadr[body_jntadr]
            self.data.qvel[vel_addr : vel_addr + 6] = 0

    def _apply_joint_control(self, qpos_target, qvel_target=None, gripper_value=None):
        """
        Apply joint control based on current control mode.

        Args:
            qpos_target: Target joint positions (7 values, radians)
            qvel_target: Target joint velocities (optional, for 'qpos' mode)
            gripper_value: Gripper joint value (from gripper_history[0])
        """
        for i in range(7):
            if self.ctrl_mode == "ctrl":
                self.data.ctrl[self.joint_ids[i]] = qpos_target[i]
            elif self.ctrl_mode == "qpos":
                self.data.qpos[self.joint_ids[i]] = qpos_target[i]
                if qvel_target is not None:
                    self.data.qvel[self.joint_ids[i]] = qvel_target[i]

        # Apply gripper control via actuator (constraints will be solved by mj_forward)
        if gripper_value is not None:
            # Use recorded gripper value as driver joint position
            # gripper_value is in range 0-0.8 (driver joint range)
            # Convert to actuator control range (0-255)
            driver_value = float(gripper_value)
            ctrl_value = (driver_value / 0.8) * 255.0
            self.data.ctrl[self.gripper_actuator_id] = ctrl_value

    def _init_video_recording(self):
        """Initialize video recording if enabled."""
        if not self.save_video:
            return

        # Generate output path if not specified
        if self.output_path is None:
            if self.data_path:
                base_name = os.path.splitext(os.path.basename(self.data_path))[0]
                self.output_path = os.path.join(
                    os.path.dirname(self.data_path), f"{base_name}_replay.mp4"
                )
            else:
                self.output_path = "replay_output.mp4"

        # Update model's offscreen buffer size to match video resolution
        self.model.vis.global_.offwidth = self.video_width
        self.model.vis.global_.offheight = self.video_height

        # Create video renderer
        self.video_renderer = mujoco.Renderer(
            self.model, height=self.video_height, width=self.video_width
        )

        # Create camera for video recording using the same default settings
        self.video_camera = mujoco.MjvCamera()
        self.video_camera.lookat[:] = self.default_camera_params["lookat"]
        self.video_camera.distance = self.default_camera_params["distance"]
        self.video_camera.azimuth = self.default_camera_params["azimuth"]
        self.video_camera.elevation = self.default_camera_params["elevation"]

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.video_writer = cv2.VideoWriter(
            self.output_path,
            fourcc,
            self.video_fps,
            (self.video_width, self.video_height),
        )

        print(f"Video recording enabled: {self.output_path}")
        print(
            f"Video settings: {self.video_width}x{self.video_height} @ {self.video_fps} fps"
        )

    def _record_frame(self):
        """Capture and write a frame to the video."""
        if self.video_writer is None or self.video_renderer is None:
            return

        # Update renderer with current scene using the configured camera
        self.video_renderer.update_scene(self.data, camera=self.video_camera)

        # Render frame
        frame = self.video_renderer.render()

        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Write frame
        self.video_writer.write(frame_bgr)

    def _finalize_video(self):
        """Finalize and close the video file."""
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
            print(f"Video saved to: {self.output_path}")

    def replay(self):
        """Main replay loop - iterates through recorded timesteps and visualizes."""
        print(f"\n{'=' * 60}")
        print(f"Starting motion replay")
        print(f"Control mode: {self.ctrl_mode}")
        print(f"Speed factor: {self.speed_factor}x")
        print(f"Total timesteps: {self.num_timesteps}")
        print(f"{'=' * 60}\n")

        # Initialize motion trace
        self._init_replay_trace()

        # Initialize video recording if enabled
        self._init_video_recording()

        # Calculate delay between timesteps based on speed factor
        # When saving video, no real-time delay needed (render as fast as possible)
        base_delay = 0 if self.save_video else self.dt / self.speed_factor

        # Calculate frame skip for video recording
        # Record frames at video_fps rate, independent of speed_factor
        # Speed factor only affects interactive playback, not video recording
        sim_fps = 1.0 / self.dt  # Simulation frequency
        frame_interval = max(1, int(sim_fps / self.video_fps))

        current_timestep = 0
        replay_start_time = time.time()

        while self.sys_run_flag and current_timestep < self.num_timesteps:
            try:
                episode = self.episodes[current_timestep]

                # Extract data for this timestep
                qpos_target = np.array(episode["qpos"])
                qvel_target = np.array(episode["qvel"])
                center_tf = episode["center_history"]
                gripper_value = episode.get("gripper_history", [None])[0]

                # Apply robot joint control (including gripper)
                self._apply_joint_control(qpos_target, qvel_target, gripper_value)

                # Set object pose from recorded transformation matrix
                self._set_object_pose(center_tf)

                # Use mj_step() to solve gripper constraints properly
                # The arm qpos and object qpos are reset each frame, so physics won't accumulate
                mujoco.mj_step(self.model, self.data)

                # Record arm and object trace snapshots based on distance traveled
                # Only start recording after the delay period
                sim_time = current_timestep * self.dt
                if sim_time >= self.trace_start_delay:
                    self._record_arm_trace()
                    self._record_object_trace()

                # Render trace ghosts to interactive viewer
                if self.viewer is not None:
                    self._render_trace_to_scene(self.viewer.user_scn)
                    self.viewer.sync()

                # Record frame for video
                if self.save_video and current_timestep % frame_interval == 0:
                    self._record_frame()

                # Progress logging (every 500 timesteps)
                if current_timestep % 500 == 0:
                    sim_time = current_timestep * self.dt
                    real_time = time.time() - replay_start_time
                    print(
                        f"Timestep {current_timestep}/{self.num_timesteps} "
                        f"(sim: {sim_time:.2f}s, real: {real_time:.2f}s)"
                    )

                # Timing control for playback speed
                if base_delay > 0:
                    time.sleep(base_delay)

                current_timestep += 1

            except KeyboardInterrupt:
                self.sys_run_flag = False
                print("\nReplay interrupted by user")
                break

            except Exception as e:
                print(f"Error at timestep {current_timestep}: {e}")
                current_timestep += 1

        # Finalize video recording
        self._finalize_video()

        # Replay complete
        total_time = time.time() - replay_start_time
        print(f"\n{'=' * 60}")
        print(f"Replay complete!")
        print(f"Total timesteps: {current_timestep}")
        print(f"Simulation time: {current_timestep * self.dt:.2f}s")
        print(f"Real time: {total_time:.2f}s")
        print(f"Effective speed: {(current_timestep * self.dt) / total_time:.2f}x")
        print(f"{'=' * 60}")

        # Keep viewer open after replay (unless saving video)
        if not self.save_video:
            self._wait_for_close()

    def _wait_for_close(self):
        """Keep viewer open until user closes it."""
        print("\nViewer will remain open. Close the window or press Ctrl+C to exit.")
        while self.sys_run_flag and self.viewer is not None:
            try:
                if not self.viewer.is_running():
                    break
                self.viewer.sync()
                time.sleep(0.1)
            except KeyboardInterrupt:
                break

        self._close()


def main():
    """Main entry point with command-line argument parsing."""
    import argparse

    parser = argparse.ArgumentParser(description="Replay recorded robot motion")
    parser.add_argument(
        "--scene",
        type=str,
        default=os.path.join(
            cpath, "Environments", "xarm7_tossing_gripper_xml", "tossing.xml"
        ),
        help="Path to MuJoCo scene XML file",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=os.path.join(
            cpath, "..", "..", "..", "data", "demo", "data_save5.json"
        ),
        help="Path to recorded motion data JSON file",
    )
    parser.add_argument(
        "--ctrl-mode",
        type=str,
        choices=["qpos", "ctrl"],
        default="qpos",
        help="Control mode: qpos (direct position) or ctrl (control signal)",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Playback speed factor (1.0 = real-time)",
    )
    parser.add_argument(
        "--save-video", action="store_true", help="Save replay as video file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output video file path (default: auto-generated)",
    )
    parser.add_argument(
        "--video-fps",
        type=int,
        default=50,
        help="Video frames per second (default: 50)",
    )
    parser.add_argument(
        "--video-width",
        type=int,
        default=1920,
        help="Video width in pixels (default: 1920)",
    )
    parser.add_argument(
        "--video-height",
        type=int,
        default=1080,
        help="Video height in pixels (default: 1080)",
    )
    parser.add_argument(
        "--arm-trace-distance",
        type=float,
        default=0.05,
        help="Min EEF distance in meters between arm traces (default: 0.05)",
    )
    parser.add_argument(
        "--arm-trace-count",
        type=int,
        default=10,
        help="Max arm trace snapshots (default: 10)",
    )
    parser.add_argument(
        "--object-trace-distance",
        type=float,
        default=0.05,
        help="Min object distance in meters between object traces (default: 0.05)",
    )
    parser.add_argument(
        "--object-trace-count",
        type=int,
        default=10,
        help="Max object trace snapshots (default: 10)",
    )
    parser.add_argument(
        "--trace-start-delay",
        type=float,
        default=0.0,
        help="Time in seconds to wait before starting trace recording (default: 0.0)",
    )
    parser.add_argument(
        "--cam-lookat",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Camera lookat point [x, y, z] (default: -1, 1, 0.6)",
    )
    parser.add_argument(
        "--cam-distance",
        type=float,
        default=None,
        help="Camera distance from lookat point (default: 3.3)",
    )
    parser.add_argument(
        "--cam-azimuth",
        type=float,
        default=None,
        help="Camera azimuth angle in degrees (default: 130)",
    )
    parser.add_argument(
        "--cam-elevation",
        type=float,
        default=None,
        help="Camera elevation angle in degrees (default: -30)",
    )

    args = parser.parse_args()

    # Create replay instance
    replay = ReplayMotion(
        path=args.scene,
        data_path=args.data,
        ctrl_mode=args.ctrl_mode,
        speed_factor=args.speed,
        save_video=args.save_video,
        output_path=args.output,
        video_fps=args.video_fps,
        video_width=args.video_width,
        video_height=args.video_height,
        arm_trace_distance=args.arm_trace_distance,
        arm_trace_count=args.arm_trace_count,
        object_trace_distance=args.object_trace_distance,
        object_trace_count=args.object_trace_count,
        trace_start_delay=args.trace_start_delay,
        camera_lookat=args.cam_lookat,
        camera_distance=args.cam_distance,
        camera_azimuth=args.cam_azimuth,
        camera_elevation=args.cam_elevation,
    )

    # Start replay
    replay.replay()


if __name__ == "__main__":
    main()
