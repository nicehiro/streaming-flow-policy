import os
from typing import Dict
cpath = os.path.dirname(os.path.abspath(__file__))
ccpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
np.set_printoptions(precision=5, suppress=True, linewidth=200)
import mujoco
import mujoco.viewer
import time
import cv2
from copy import deepcopy
from xArm7PinoCtrlCls import *
from scipy.spatial.transform import Rotation as R


class Mj_Env(object):
    def __init__(self, path=None):
        # 初始化MuJoCo环境
        self.model = None
        self.data = None
        self.viewer = None

        self.sys_run_sta = 0
        self.sys_run_flag = True

        # Motion trace state
        self.trace_config = None
        self.trace_data = None

        self._create_scene(path=path)
        self._set_viewer()
        self._set_renderer()

    def _create_scene(self, path=None):
        # 创建模型
        self.model = mujoco.MjModel.from_xml_path(path)
        self.data = mujoco.MjData(self.model)

        # 读取仿真时间步长
        self.dt = self.model.opt.timestep
        print(f"仿真时间步长: {self.dt}秒")

    def _set_viewer(self):
        # 初始化可视化选项
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        """ 设置渲染选项(可选) """
        # self.viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 1        # 关闭阴影
        # self.viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 1    # 关闭反射
        # self.viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = 1        # 关闭天空盒
        # self.viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_WIREFRAME] = 0     # 开启线框模式
        # self.viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_FOG] = 1           # 关闭雾效
        # self.viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_SEGMENT] = 0       # 关闭分段渲染

    def _set_renderer(self):
        # 创建渲染器，用来采集相机图像
        self.camera_names = ['cam1', 'cam2', 'cam3', 'cam4']
        self.camera_width = 640
        self.camera_height = 480
        self.renderer = mujoco.Renderer(self.model, height=480, width=640)

    def _get_camera_params(self):
        """ 获取相机参数 """
        print('相机类型:', self.viewer.cam.type)
        print('相机位置 (xyz):', self.viewer.cam.lookat)
        print('相机距离:', self.viewer.cam.distance)
        print('相机方位角:', self.viewer.cam.azimuth)
        print('相机仰角:', self.viewer.cam.elevation)

        return {
            'lookat': self.viewer.cam.lookat,
            'distance': self.viewer.cam.distance,
            'azimuth': self.viewer.cam.azimuth,
            'elevation': self.viewer.cam.elevation
        }

    def _set_camera_params(self, dict_params=None):
        """ 设置相机参数 """
        if dict_params is not None:
            self.viewer.cam.lookat = dict_params['lookat']
            self.viewer.cam.distance = dict_params['distance']
            self.viewer.cam.azimuth = dict_params['azimuth']
            self.viewer.cam.elevation = dict_params['elevation']

    def _get_system_run_time(self, output=False):
        """
        获取系统运行时间（秒）
        第一个返回值：系统运行时间（秒）
        第二个返回值：系统运行次数（steps）
        """
        if self.data.time == 0.0:
            time_step = 0
        else:
            time_step = int(self.data.time / self.dt)

        if output:
            print("系统运行时间（秒）:", self.data.time, f', 系统运行次数: {time_step}')

        return self.data.time, time_step

    def _get_joint_states(self):
        """
        获取关节状态信息，包括位置和速度，返回的是弧度制
        """
        self.qpos = np.array([self.data.qpos[self.joint_ids[i]] for i in range(7)])
        self.qvel = np.array([self.data.qvel[self.joint_ids[i]] for i in range(7)])

        return self.qpos, self.qvel

    def _get_geom_states(self, geom_id=None):
        if geom_id >= 0:
            """获取指定body的状态信息，包括位置、四元数、旋转矩阵、线速度和角速度"""
            # 位置、四元数和旋转矩阵
            pos = deepcopy(self.data.geom_xpos[geom_id])
            # quat = deepcopy(self.data.geom_xquat[geom_id])
            rotMat = deepcopy(self.data.geom_xmat[geom_id].reshape(3, 3))

            # 线速度和角速度
            # linear_vel = deepcopy(self.data.geom_cvel[geom_id, :3])
            # angular_vel = deepcopy(self.data.geom_cvel[geom_id, 3:6])

            # 组合成齐次旋转矩阵
            transforMationMatrix = np.eye(4, dtype=np.float64)
            transforMationMatrix[:3, :3] = deepcopy(rotMat)
            transforMationMatrix[:3, 3] = deepcopy(pos)

            return {
                'position': pos,
                # 'quaternion': quat,
                'rotation_matrix': rotMat,
                # 'linear_velocity': linear_vel,
                # 'angular_velocity': angular_vel,
                'transformation_matrix': transforMationMatrix
            }

        else:
            print("请输入正确的geom_id!")
            return None

    def _get_body_states(self, body_id=None):
        if body_id >= 0:
            """获取指定body的状态信息，包括位置、四元数、旋转矩阵、线速度和角速度"""
            # 位置、四元数和旋转矩阵
            pos = deepcopy(self.data.xpos[body_id])
            quat = deepcopy(self.data.xquat[body_id])
            rotMat = deepcopy(self.data.xmat[body_id].reshape(3, 3))
            euler = R.from_matrix(rotMat).as_euler('xyz', degrees=True)

            # 线速度和角速度
            linear_vel = deepcopy(self.data.cvel[body_id, :3])
            angular_vel = deepcopy(self.data.cvel[body_id, 3:6])

            # 组合成齐次旋转矩阵
            transforMationMatrix = np.eye(4, dtype=np.float64)
            transforMationMatrix[:3, :3] = deepcopy(rotMat)
            transforMationMatrix[:3, 3] = deepcopy(pos)

            return {
                'position': pos,
                'quaternion': quat,
                'rotation_matrix': rotMat,
                'euler_angles': euler,
                'linear_velocity': linear_vel,
                'angular_velocity': angular_vel,
                'transformation_matrix': transforMationMatrix
            }

        else:
            print("请输入正确的body_id!")
            return None

    def _get_joint_force_info(self, joint_index=0):
        """ 读取各关节的各种力信息 """
        applied_torques = self.data.qfrc_applied[joint_index]      # 施加的力
        actuator_torques = self.data.qfrc_actuator[joint_index]    # 执行器力
        passive_forces = self.data.qfrc_passive[joint_index]       # 被动力
        inverse_dynamics = self.data.qfrc_inverse[joint_index]     # 逆动力学力

        return {
            'applied': applied_torques,
            'actuator': actuator_torques,
            'passive': passive_forces,
            'inverse': inverse_dynamics
        }

    def _get_link_force_info(self, link_index=0):
        """ 获取各连杆坐标系受到的外力和力矩 """
        force_torque = self.data.xfrc_applied[link_index]

        return force_torque

    def _reset_body_state(self, body_joint_id=None, pose=None):
        """
        重置指定body的位姿和速度:
        mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "square_freejoint")
        """
        if body_joint_id > 0:
            # 重置位姿
            body_addr = self.model.jnt_qposadr[body_joint_id]
            self.data.qpos[body_addr:(body_addr + 3)] = deepcopy(pose[0:3])  # x, y, z
            self.data.qpos[(body_addr + 3):(body_addr + 7)] = deepcopy(pose[3:7])  # w, x, y, z
            # 重置速度
            vel_addr = self.model.jnt_dofadr[body_joint_id]
            self.data.qvel[vel_addr:vel_addr+6] = 0

        else:
            print("请输入正确的body_joint_id!")

    def _get_images(self) -> Dict[str, np.ndarray]:
        """Capture images from all cameras"""
        images = {}

        # Check if we can initialize OpenGL renderer
        can_render = True
        if self.renderer is None:
            try:
                # Try to detect if we have a display
                import os
                if 'DISPLAY' not in os.environ:
                    can_render = False
                    # print("No DISPLAY environment variable, skipping image rendering")
                else:
                    self.renderer = mujoco.Renderer(self.model, height=self.camera_height, width=self.camera_width)
            except Exception as e:
                print(f"Warning: Cannot initialize renderer (likely no display): {e}")
                can_render = False
                self.renderer = None

        for i, cam_name in enumerate(['camera1', 'camera2', 'camera3', 'camera4']):
            # Default to black image
            img = np.zeros((self.camera_height, self.camera_width, 3), dtype=np.uint8)

            # Try to get camera image only if renderer is available
            if can_render and self.renderer is not None:
                try:
                    cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, self.camera_names[i])
                    self.renderer.update_scene(self.data, camera=cam_id)
                    img = self.renderer.render()
                except Exception as e:
                    print(f"Warning: Could not render camera {cam_name}: {e}")
                    pass

            images[cam_name] = img

        return images

    def _move_joint(self, goal, threshold=0.035, ctrl_mode='ctrl'):
        """
        控制机械臂关节角度
        1、如果用ctrl来控制，则机械臂是有运动的渲染功效，看起来真实一些，这个更推荐用在演示场景
        2、如果用qpos来控制，则是直接一步到位，机械臂不会有运动过程，看起来不真实，但更快一些，这个更推荐用在复位场景
        3、两种方式都可以实现机械臂的关节角度控制
        """
        for i in range(7):
            if ctrl_mode == 'ctrl':
                self.data.ctrl[self.joint_ids[i]] = goal[i]
            elif ctrl_mode == 'qpos':
                self.data.qpos[self.joint_ids[i]] = goal[i]

        return self._judge_joint_reached(goal, threshold=threshold)

    def _move_joint_velocity(self, goal):
        for i in range(7):
            self.data.qvel[self.joint_ids[i]] = goal[i]

    def _judge_joint_reached(self, goal, threshold=0.035):
        self._get_joint_states()

        if np.linalg.norm(self.qpos - goal, ord=2) <= threshold:
            return True
        else:
            return False

    def _step_simulation(self, delta=0.0):
        # 执行仿真步骤
        mujoco.mj_step(self.model, self.data)

        if self.viewer is not None:
            self.viewer.sync()

        if delta > 0.0:
            time.sleep(self.dt * delta)

    def _close(self):
        """
        关闭环境
        """
        if self.viewer is not None:
            self.viewer.close()
        print("程序结束")

    def _init_motion_trace(self, sample_interval=0.1, max_snapshots=50, link_ids=None):
        """
        Initialize motion trace visualization

        Args:
            sample_interval: Time between samples in seconds
            max_snapshots: Maximum number of snapshots to keep (sliding window)
            link_ids: List of body IDs to trace
        """
        if link_ids is None or len(link_ids) == 0:
            print("Warning: No link_ids provided for motion trace")
            return

        # Compute geom-to-link mapping
        geom_map = {link_id: [] for link_id in link_ids}
        for gid in range(self.model.ngeom):
            body_id = self.model.geom_bodyid[gid]
            if body_id in geom_map:
                geom_map[body_id].append(gid)

        self.trace_config = {
            'sample_interval': sample_interval,
            'max_snapshots': max_snapshots,
            'link_ids': link_ids,
            'geom_ids_per_link': geom_map
        }

        self.trace_data = {
            'last_sample_time': 0.0,
            'timestamps': [],
            'qpos_snapshots': []
        }

        print(f"Motion trace initialized: interval={sample_interval}s, max_snapshots={max_snapshots}, links={len(link_ids)}")

    def _update_motion_trace(self):
        """
        Update motion trace by capturing snapshots based on time intervals
        Call this in simulation loop after mj_step()
        """
        if self.trace_config is None:
            return

        current_time = self.data.time

        # Time-based sampling
        if current_time - self.trace_data['last_sample_time'] >= self.trace_config['sample_interval']:
            self.trace_data['timestamps'].append(current_time)
            self.trace_data['qpos_snapshots'].append(self.data.qpos.copy())
            self.trace_data['last_sample_time'] = current_time

            # Sliding window: remove oldest if exceeded limit
            if len(self.trace_data['qpos_snapshots']) > self.trace_config['max_snapshots']:
                self.trace_data['timestamps'].pop(0)
                self.trace_data['qpos_snapshots'].pop(0)

    def _compute_trace_color(self, t):
        """
        Compute RGBA color for a trace snapshot based on normalized time

        Args:
            t: normalized time [0,1] where 0=oldest, 1=newest

        Returns:
            np.ndarray: [R, G, B, A] in range [0, 1]
        """
        r = t           # 0 → 1 (blue to red)
        g = 0.0
        b = 1.0 - t     # 1 → 0 (blue to red)
        alpha = 0.1 + 0.1 * t  # Fading: 0.1 → 0.6
        return np.array([r, g, b, alpha], dtype=np.float32)

    def _add_ghost_geom(self, scene, ghost_data, geom_id, rgba):
        """
        Add one geometry to the ghost scene

        Args:
            scene: viewer.user_scn
            ghost_data: MjData with ghost pose
            geom_id: geometry ID to render
            rgba: [R, G, B, A] color
        """
        if scene.ngeom >= scene.maxgeom:
            return  # Graceful degradation

        geom_type = self.model.geom_type[geom_id]

        if geom_type == mujoco.mjtGeom.mjGEOM_MESH:
            # Manual mesh geometry setup
            g = scene.geoms[scene.ngeom]
            g.type = geom_type
            g.dataid = self.model.geom_dataid[geom_id]
            g.objtype = mujoco.mjtObj.mjOBJ_MESH
            g.objid = g.dataid
            g.pos[:] = ghost_data.geom_xpos[geom_id]
            g.mat[:, :] = ghost_data.geom_xmat[geom_id].reshape(3, 3)
            g.size[:] = self.model.geom_size[geom_id]
            g.rgba[:] = rgba
            scene.ngeom += 1
        else:
            # Primitive geometries (sphere, box, capsule, cylinder)
            mujoco.mjv_initGeom(
                scene.geoms[scene.ngeom],
                geom_type,
                self.model.geom_size[geom_id],
                ghost_data.geom_xpos[geom_id],
                ghost_data.geom_xmat[geom_id],
                rgba=rgba
            )
            scene.ngeom += 1

    def _render_motion_trace(self, enable=True):
        """
        Render motion trace ghosts to viewer
        Call this before viewer.sync() in simulation loop

        Args:
            enable: whether to render trace
        """
        if not enable or self.trace_config is None or self.viewer is None:
            return

        scene = self.viewer.user_scn
        scene.ngeom = 0  # Clear previous frame's ghosts

        num_snapshots = len(self.trace_data['qpos_snapshots'])
        if num_snapshots == 0:
            return

        # Render each snapshot as ghost
        for i, qpos in enumerate(self.trace_data['qpos_snapshots']):
            # Create ghost data with this pose
            ghost_data = mujoco.MjData(self.model)
            ghost_data.qpos[:] = qpos
            mujoco.mj_forward(self.model, ghost_data)  # Compute forward kinematics

            # Time-based color and opacity
            t = i / (num_snapshots - 1) if num_snapshots > 1 else 0.5
            color = self._compute_trace_color(t)

            # Render all geometries for all links
            for link_id in self.trace_config['link_ids']:
                for geom_id in self.trace_config['geom_ids_per_link'][link_id]:
                    self._add_ghost_geom(scene, ghost_data, geom_id, color)

    def _export_trace_image(self, filepath, width=1920, height=1080):
        """
        Export current trace visualization to image file

        Args:
            filepath: Output path (e.g., 'trace.png')
            width, height: Output resolution
        """
        if self.trace_config is None:
            print("Error: Motion trace not initialized")
            return

        try:
            # Create high-res renderer
            export_renderer = mujoco.Renderer(self.model, height=height, width=width)

            # Update with current scene and camera
            export_renderer.update_scene(self.data, camera=self.viewer.cam)

            # Render (includes user_scn geometries)
            img = export_renderer.render()

            # Save to file
            cv2.imwrite(filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            print(f"Motion trace exported to: {filepath}")
        except Exception as e:
            print(f"Error exporting trace image: {e}")

    def _simulation(self):
        while self.sys_run_flag:
            try:
                self._get_system_run_time(output=True)

                """ 这个目前不用，后续可以用来做图像采集 """
                """ 但目前问题是，都放在一个主线程里跑的话，图像采集会很卡顿 """
                imgs = self._get_images()
                for cam_name, img in imgs.items():
                    cv2.imshow(cam_name, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                cv2.waitKey(1)

                self._step_simulation(delta=0.0)


            except KeyboardInterrupt:
                self.sys_run_flag = False
                print("捕获到中断信号，停止仿真")
                self._close()

            except Exception as e:
                self.sys_run_flag = False
                print("仿真过程中出现异常，停止仿真")
                print("异常信息:", e)
                self._close()


if __name__ == "__main__":
    demo = Mj_Env(path=os.path.join(cpath, "Environments", "xarm7_official_xml", "scene.xml"))
    demo._simulation()
