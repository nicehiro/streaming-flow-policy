import os
cpath = os.path.dirname(os.path.abspath(__file__))
ccpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
np.set_printoptions(precision=5, suppress=True, linewidth=200)
import mujoco
import mujoco.viewer
import time
from copy import deepcopy
from xArm7PinoCtrlCls import *
from scipy.spatial.transform import Rotation as R
from BaseMJEnv import Mj_Env

class BoxForceDemo(Mj_Env):
    def __init__(self, path=None):
        super(BoxForceDemo, self).__init__(path=path)

        # 创建xArm7机械臂对象
        self.probot = PinocchioRobotCls(path=os.path.join(cpath, "Environments", "xarm7_without_gripper_urdf", "xarm7.urdf"), format='urdf')

        self._get_ids()

        # Initialize motion trace
        self._init_motion_trace(
            sample_interval=0.2,   # Sample every 0.1 seconds
            max_snapshots=10,      # Keep 5 seconds of history
            link_ids=self.link_ids  # Trace all arm links
        )

    def _get_ids(self):
        # 获取关节ID列表
        self.joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f'joint{i + 1}') for i in range(7)]
        print(f"Joint IDs: {self.joint_ids}")

        # 获取连杆ID列表
        self.link_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f'link{i + 1}') for i in range(7)]
        self.link_ids.append(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'end_effector'))
        print(f"Link IDs: {self.link_ids}")

        # 获取相机ID
        self.camera_idx = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, f'cam{i + 1}') for i in range(4)]
        print(f"Camera IDs: {self.camera_idx}")

    def display(self):
        while self.sys_run_flag:
            try:
                """ 获取末端执行器的受力信息 """
                # force_torque = self._get_link_force_info(link_index=self.link_ids[-1])
                # print(f"Applied Force on end effector: {force_torque[:3]}, Applied Torque on end effector: {force_torque[3:]}")

                """ 获取关节受力信息 """
                # force_torque = self._get_joint_force_info(joint_index=self.joint_ids[-1])
                # print(f"Applied Torque: {force_torque['applied']}", end=', ')
                # print(f"Actuator Torque: {force_torque['actuator']}", end=', ')
                # print(f"Passive Torque: {force_torque['passive']}", end=', ')
                # print(f"Inverse Torque: {force_torque['inverse']}")

                """ 获取关节状态信息 """
                self._get_joint_states()
                # print(f"Joint Angles (rad): {self.qpos}, Joint Velocities (rad/s): {self.qvel}")

                """ 利用MuJoCo获取末端执行器的变换矩阵 """
                # print('末端变换矩阵:\n', self._get_body_states(body_id=self.link_ids[-1])['transformation_matrix'])

                """ 利用Pinocchio进行正运动学计算末端执行器的变换矩阵 """
                # print('正运动学计算的末端变换矩阵:\n', self.probot.fkine(q=self.qpos, is_radian=True)[2])

                if self._get_system_run_time()[0] > 1.0 and self.sys_run_sta == 0:
                    joint_angle = np.deg2rad(np.array([10, 11, 12, 13, 14, 15, 16], dtype=np.float64) * 4)
                    if self._move_joint(goal=joint_angle, threshold=0.035, ctrl_mode='ctrl'):
                        print("机械臂已到达目标位置")
                        self._get_system_run_time(output=True)
                        self.sys_run_sta = 1

                elif self._get_system_run_time()[0] > 2.0 and self.sys_run_sta == 1:
                    joint_angle = np.deg2rad(np.array([80, 81, 82, 83, 84, 85, 86], dtype=np.float64))
                    if self._move_joint(goal=joint_angle, threshold=0.035, ctrl_mode='ctrl'):
                        print("机械臂已到达目标位置")
                        self._get_system_run_time(output=True)
                        self.sys_run_sta = 2

                elif self._get_system_run_time()[0] > 3.0 and self.sys_run_sta == 2:
                    targetMat = np.eye(4, dtype=np.float64)
                    targetMat[0:3, 3] = np.array([0.3, 0.3, 0.3], dtype=np.float64)
                    targetMat[0:3, 0:3] = R.from_euler('xyz', [180, 0, 0], degrees=True).as_matrix()
                    q_ik, solFlag = self.probot.ikine(target=targetMat, q0=self.qpos, tol=1e-3, max_iter=1000, dt=1e-2, damp=1e-12)
                    self.sys_run_sta = 3
                    print("逆运动学求解结果:", q_ik, f', 求解是否成功: {solFlag}')

                elif self._get_system_run_time()[0] > 4.0 and self.sys_run_sta == 3:
                    joint_angle = np.deg2rad(q_ik)
                    if self._move_joint(goal=joint_angle, threshold=0.035, ctrl_mode='ctrl'):
                        print("机械臂已到达目标位置")
                        self._get_system_run_time(output=True)
                        self.sys_run_sta = 4

                elif self._get_system_run_time()[0] > 5.0 and self.sys_run_sta == 4:
                    joint_velocity = np.deg2rad(np.array([10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64))
                    self._move_joint_velocity(goal=joint_velocity)
                    print("机械臂开始运动")
                    if self._get_system_run_time()[0] > 8.0:
                        self.sys_run_sta = 5

                elif self._get_system_run_time()[0] > 8.0 and self.sys_run_sta == 5:
                    joint_velocity = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                    self._move_joint_velocity(goal=joint_velocity)
                    print("机械臂停止运动")
                    self.sys_run_sta = 6

                elif self._get_system_run_time()[0] > 10.0 and self.sys_run_sta == 6:
                    joint_angle = np.array([0, 1.41, 0, 2.98, 0, 0, 0])
                    if self._move_joint(goal=joint_angle, threshold=0.035):
                        print("机械臂已到达目标位置")
                        self._get_system_run_time(output=True)
                        self.sys_run_sta = 7

                """
                需要注意的是，qvel是关节速度，如果只在一个时刻给定的话，机械臂只会动一下，需要持续不断地给定，才可以让机械臂持续运动
                另外，给定qvel后，机械臂的关节角度会不断变化，可以通过data.qpos来查看当前的关节角度
                这种方式更像是直接给电机下达速度指令，让电机以一定的速度运动
                这种方式更适合用在需要机械臂持续运动的场景，比如搬运、抛掷等场景
                当然，也可以通过给data.ctrl来控制机械臂的关节角度
                """

                # Update motion trace (capture snapshots)
                self._update_motion_trace()

                # Render motion trace (add ghosts to viewer)
                self._render_motion_trace(enable=True)

                self._step_simulation(delta=1.0)

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
    demo = BoxForceDemo("streaming_flow_policy/lasso/sim/Environments/xarm7_official_xml/scene.xml")
    demo.display()
