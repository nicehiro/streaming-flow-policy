#! /usr/bin/python
import sys
import os
from xml.parsers.expat import model
cpath = os.path.abspath(os.path.dirname(__file__))
ccpath = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
import numpy as np
np.set_printoptions(precision=4, suppress=True, linewidth=200)
import time
import pinocchio as pin
import matplotlib.pyplot as plt
from copy import deepcopy
from scipy.spatial.transform import Rotation as R



class PinocchioRobotCls:
    def __init__(self, path=None, format='urdf'):
        """ 初始化匹诺曹机器人类 """
        self.model = None
        self.data = None
        if path is not None:
            self.load_robot_example(path, format)
        else:
            print("请提供机器人模型路径")
            return

        # 设置基坐标系和工具中心点坐标系偏移
        self.sUcs_sTcp_set(sUcs=0, sTcp=0)

    def load_robot_example(self, path=None, format='urdf'):
        if format == 'urdf':
            self.model = pin.buildModelFromUrdf(path)
        elif format == 'xml':
            self.model = pin.buildModelFromXML(path)
        else:
            print("不支持的模型格式")
            return None, None

        self.data = self.model.createData()

    def sUcs_sTcp_set(self, sUcs=1, sTcp=1):
        """ 设计基坐标系偏移 """
        if sUcs == 1:
            base_offset = pin.SE3(R.from_euler('xyz', [0, 0, 0], degrees=True).as_matrix(), np.array([1.0, 1.0, 1.0], dtype=np.float64))
        else:
            base_offset = pin.SE3.Identity()

        """ 设计工具中心点 (TCP) 坐标系 """
        if sTcp == 1:
            tcp_offset = pin.SE3(R.from_euler('xyz', [0, 0, 0], degrees=True).as_matrix(), np.array([1.0, 2.0, 3.5], dtype=np.float64))
        else:
            tcp_offset = pin.SE3.Identity()

        """ 查找TCP帧ID """
        self.tcp_frame_id = self.model.getFrameId('link_eef')
        # self.tcp_frame_id = self.model.njoints - 1

        """ 重新生成数据容器 """
        self.data = self.model.createData()

    def fkine(self, q, is_radian=False):
        if is_radian is False:
            pin.forwardKinematics(self.model, self.data, np.deg2rad(q))
            pin.framesForwardKinematics(self.model, self.data, np.deg2rad(q))

        else:
            pin.forwardKinematics(self.model, self.data, q)
            pin.framesForwardKinematics(self.model, self.data, q)


        pin.updateFramePlacements(self.model, self.data)
        pin.updateFramePlacements(self.model, self.data)

        # 获取该帧的位姿（相对于世界坐标系）
        tcp_pose = self.data.oMf[self.tcp_frame_id]
        tcp_pose = self.data.oMf[self.tcp_frame_id]

        # tcp_pose = self.data.oMi[self.tcp_frame_id]
        # 利用齐次矩阵表示位姿
        transformationMatrix = np.eye(4, dtype=np.float64)
        transformationMatrix[0:3, 3] = deepcopy(tcp_pose.translation)
        transformationMatrix[0:3, 0:3] = deepcopy(tcp_pose.rotation)

        # 利用位置 + 欧拉角表示位姿
        position = deepcopy(transformationMatrix[0:3, 3])
        rotation_Euler = R.from_matrix(deepcopy(transformationMatrix[0:3, 0:3])).as_euler(seq='xyz', degrees=True)

        # 利用位置 + 旋转向量表示位姿
        rotation_vector = R.from_matrix(deepcopy(transformationMatrix[0:3, 0:3])).as_rotvec()

        results = []
        results.append(np.hstack((position, rotation_Euler)))
        results.append(np.hstack((position, rotation_vector)))
        results.append(transformationMatrix)

        return results

    def ikine(self, target=None, q0=None, tol = 1e-4, max_iter = 1000, dt = 1e-2, damp = 1e-12):
        if q0 is None:
            q0 = np.zeros(self.model.nv)

        if np.size(target, axis=0) == 4 and np.size(target, axis=1) == 4:
            oMdes = pin.SE3(target[0:3, 0:3], target[0:3, 3])

        elif np.size(target, axis=0) == 6:
            position = deepcopy(target[0:3])
            rotation_Euler = deepcopy(target[3:6])
            rotation_matrix = R.from_euler('xyz', rotation_Euler, degrees=True).as_matrix()
            oMdes = pin.SE3(rotation_matrix, position)
        else:
            print("目标位姿格式错误")
            return None, False

        solFlag = False
        end_effector_id = self.model.njoints - 1

        # 简单的逆向运动学（实际应用中可能需要更复杂的算法）
        q = deepcopy(q0)
        for i in range(max_iter):
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)

            # 计算目标位姿到当前位姿之间的变换
            # iMd = self.data.oMi[end_effector_id].actInv(oMdes)
            iMd = self.data.oMf[self.tcp_frame_id].actInv(oMdes)

            # 通过李群对数映射将变换矩阵转换为6维误差向量（包含位置误差和方向误差）
            # 用于量化当前位姿和目标位姿之间的误差关系
            err = pin.log(iMd).vector

            if np.linalg.norm(err) < tol:
                print(f"逆向运动学收敛于第 {i} 次迭代")
                solFlag = True
                break

            # 计算当前关节角度下的雅可比矩阵，关节速度与末端速度的映射关系
            J = pin.computeJointJacobian(self.model, self.data, q, end_effector_id)
            # J = pin.computeFrameJacobian(self.model, self.data, q, self.tcp_frame_id)

            # 对雅可比矩阵进行变换，转换到李代数空间，以匹配误差向量的坐标系，同时取反以调整误差方向
            J = -np.dot(pin.Jlog6(iMd.inverse()), J)

            # 使用阻尼最小二乘法求解关节速度
            v = -J.T.dot(np.linalg.solve(J.dot(J.T) + damp * np.eye(6), err))

            # 更新关节角度
            q = pin.integrate(self.model, q, v * dt)

        return np.rad2deg(q), solFlag



def Kinematics_example():
    """运动学示例"""
    print("=== 匹诺曹机器人库示例 ===")
    probot = PinocchioRobotCls(path=os.path.join(cpath, "Environments", "xarm7_without_gripper_urdf", "xarm7.urdf"), format='urdf')

    print("\n机器人关节数:", probot.model.nq)

    for name in probot.model.names:
        print(f"关节: {name}")

    print(np.rad2deg(probot.model.lowerPositionLimit))
    print(np.rad2deg(probot.model.upperPositionLimit))

    # 正向运动学
    print("\n1. 正向运动学:")
    q_random = np.array([10, 20, 30, 40, 50, 60, 90], dtype=np.float64)
    print("随机关节角度 (度):", q_random)
    fk_results = probot.fkine(q_random, is_radian=False)
    print("正向运动学结果:\n", fk_results[2])

    # 逆向运动学
    print("\n2. 逆向运动学:")
    q_ik, solFlag = probot.ikine(target=fk_results[2], q0=np.zeros(probot.model.nq), dt=0.03)
    print("逆向运动学解:", q_ik)
    print("逆向运动学收敛状态:", solFlag)
    fk_results = probot.fkine(q_ik, is_radian=False)
    print("正向运动学结果:\n", fk_results[2])

if __name__ == "__main__":
    Kinematics_example()
