import os
cpath = os.path.dirname(os.path.abspath(__file__))
ccpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
np.set_printoptions(precision=5, suppress=True, linewidth=200)
import mujoco
import mujoco.viewer
import time
import pygame as pyg
import json
from copy import deepcopy
from xArm7PinoCtrlCls import *
from scipy.spatial.transform import Rotation as R
from BaseMJEnv import Mj_Env

class BoxForceDemo(Mj_Env):    
    def __init__(self, path=None):
        super(BoxForceDemo, self).__init__(path=path)
        
        # 创建xArm7机械臂对象
        self.probot = PinocchioRobotCls(path=cpath + r"\Environments\xarm7_without_gripper_urdf\xarm7.urdf", format='urdf')

        self._get_ids()
        
        # mujoco.mj_resetDataKeyframe(self.model, self.data, 0)  # 0表示第一个keyframe        


    def _get_ids(self):
        # 获取关节ID列表
        self.joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f'left_joint{i + 1}') for i in range(7)]
        self.joint_ids.append(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'left_gripper_act'))
        print(f"Joint IDs: {self.joint_ids}")

        # 获取连杆ID列表
        self.link_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f'left_link{i + 1}') for i in range(7)]
        self.link_ids.append(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'left_end_effector'))
        print(f"Link IDs: {self.link_ids}")

        # 获取相机ID
        self.camera_idx = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, f'cam{i + 1}') for i in range(2)]
        print(f"Camera IDs: {self.camera_idx}")

        # 获取三种物体的ID
        self.circle_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'circle2')
        self.square_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'square')
        self.triangle_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'triangle')
        print(f"Ring ID: {self.circle_body_id}", f'Square ID: {self.square_body_id}', f'Triangle ID: {self.triangle_body_id}')

        # 获取焊接约束ID
        self.weld_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_EQUALITY, "weld_body1_body2")

    def get_circle_state(self):
        """获取圆环上各个小球的ID"""
        positions = []
        for i in range(100):
            sphere_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f'sphere_{i}')
            positions.append(deepcopy(self._get_body_states(sphere_id)['position']))
        
        positions = np.asarray(positions)
        
        return positions
    
    def get_square_state(self):
        positions = []

        for i in range(25):
            if i < 10:
                square_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f'bottom_0{i}')
            else:
                square_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f'bottom_{i}')
            
            positions.append(deepcopy(self._get_geom_states(square_id)['position']))

        for i in range(25):
            if i < 10:
                square_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f'right_0{i}')
            else:
                square_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f'right_{i}')
            
            positions.append(deepcopy(self._get_geom_states(square_id)['position']))

        for i in range(25):
            if i < 10:
                square_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f'top_0{i}')
            else:
                square_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f'top_{i}')
            
            positions.append(deepcopy(self._get_geom_states(square_id)['position']))
        
        for i in range(25):
            if i < 10:
                square_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f'left_0{i}')
            else:
                square_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f'left_{i}')
            
            positions.append(deepcopy(self._get_geom_states(square_id)['position']))
        
        positions = np.asarray(positions)

        return positions

    def get_triangle_state(self):
        positions = []
        for i in range(100):
            if i < 10:
                triangle_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f'sphere_00{i}')
            else:
                triangle_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f'sphere_0{i}') 
            
            positions.append(deepcopy(self._get_geom_states(triangle_id)['position']))
        
        positions = np.asarray(positions)

        return positions
    
    def compute_relative_pose(self, pos1, quat1, pos2, quat2):
        """计算两个位姿之间的相对变换"""
        # 将四元数转换为旋转矩阵
        rot1 = R.from_quat([quat1[1], quat1[2], quat1[3], quat1[0]])  # [x,y,z,w] -> [w,x,y,z]
        rot2 = R.from_quat([quat2[1], quat2[2], quat2[3], quat2[0]])
        
        # 计算相对旋转
        rel_rot = rot1.inv() * rot2
        rel_quat = rel_rot.as_quat()  # [x,y,z,w]
        rel_quat = [rel_quat[3], rel_quat[0], rel_quat[1], rel_quat[2]]  # [w,x,y,z]格式
        
        # 计算相对位置 (在body1坐标系下)
        rel_pos = rot1.inv().apply(pos2 - pos1)
        
        return rel_pos, rel_quat
    
    def get_current_relative_pose(self, body1, body2):
        """获取两个body当前的相对位姿"""
        body1_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body1)
        body2_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body2)
        
        if body1_id == -1 or body2_id == -1:
            print(f"错误: 未找到body {body1} 或 {body2}")
            return None, None
        
        # 获取世界坐标系下的位姿
        body1_pos = self.data.xpos[body1_id].copy()
        body1_quat = self.data.xquat[body1_id].copy()
        
        body2_pos = self.data.xpos[body2_id].copy()
        body2_quat = self.data.xquat[body2_id].copy()
        
        # 计算相对位姿 (body2相对于body1)
        rel_pos, rel_quat = self.compute_relative_pose(
            body1_pos, body1_quat, body2_pos, body2_quat
        )
        
        return rel_pos, rel_quat
    
    def weld_configuration(self, tf_center_to_end, case=1):
        if case == 1:
            self.model.eq_data[self.weld_id][0] = 0.0
            self.model.eq_data[self.weld_id][1] = 0.0
            self.model.eq_data[self.weld_id][2] = 0.0

            self.model.eq_data[self.weld_id][3] = deepcopy(tf_center_to_end[0, 3]) - 0.05
            self.model.eq_data[self.weld_id][4] = deepcopy(tf_center_to_end[1, 3]) - 0.05
            self.model.eq_data[self.weld_id][5] = deepcopy(tf_center_to_end[2, 3]) - deepcopy(tf_center_to_end[2, 3])
            
            rel_quat = R.from_matrix(tf_center_to_end[:3, :3].dot(R.from_euler('xyz', [90, 0, 0], degrees=True).as_matrix())).as_quat()
            self.model.eq_data[self.weld_id][6] = rel_quat[3]
            self.model.eq_data[self.weld_id][7] = rel_quat[0]
            self.model.eq_data[self.weld_id][8] = rel_quat[1]
            self.model.eq_data[self.weld_id][9] = rel_quat[2]

        elif case == 2:
            self.model.eq_data[self.weld_id][0] = 0.0
            self.model.eq_data[self.weld_id][1] = 0.0
            self.model.eq_data[self.weld_id][2] = 0.0

            self.model.eq_data[self.weld_id][3] = deepcopy(tf_center_to_end[0, 3]) - 0.00
            self.model.eq_data[self.weld_id][4] = deepcopy(tf_center_to_end[1, 3]) - 0.00
            self.model.eq_data[self.weld_id][5] = deepcopy(tf_center_to_end[2, 3]) - 0.05

            rel_quat = R.from_matrix(tf_center_to_end[:3, :3].dot(R.from_euler('xyz', [0, 0, 0], degrees=True).as_matrix())).as_quat()            
            self.model.eq_data[self.weld_id][6] = rel_quat[3]
            self.model.eq_data[self.weld_id][7] = rel_quat[0]
            self.model.eq_data[self.weld_id][8] = rel_quat[1]
            self.model.eq_data[self.weld_id][9] = rel_quat[2]
        
        elif case == 3:
            pass

        self.data.eq_active[self.weld_id] = 1
    
    def gripper_control(self, action=220):
        """
        控制夹爪开合
        action: 0    打开
        action: 255  闭合
        """
        self.data.ctrl[self.joint_ids[7]] = action

    def teach(self):
        # 初始化Pygame窗口
        pyg.init()
        self.screen = pyg.display.set_mode((640, 480), pyg.RESIZABLE)
        pyg.display.set_caption("固定尺寸窗口")

        self.viewer.cam.lookat = np.array([-0.0167, 0.5295, 0.234], dtype=np.float64)
        self.viewer.cam.distance = 3.1072314441448743
        self.viewer.cam.azimuth = -178.19530469530457
        self.viewer.cam.elevation = -23.391108891108896

        force_torque_history = []
        position_history = []
        transMat_history = []

        joint_init = np.array([0.377, -0.751, -2.200, 1.370, -0.251, -0.412, 1.190])

        while self.sys_run_flag:
            try:          
                """ 获取并打印圆环和中心的状态信息 """
                circle_state = self._get_body_states(self.circle2_id)
                center_state = self._get_body_states(self.circle2_center_id)
                square_state = self._get_body_states(self.square_id)
                triangle_state = self._get_body_states(self.triangle_id)
                if center_state['position'][2] >= 0.350:
                    print("中心位置:", center_state['position'], "中心速度:", center_state['linear_velocity'])

                """ 获取施加在物体上的力和力矩 """
                force_torque = np.zeros((8, 6), dtype=np.float64)
                for i in range(8): 
                    force_torque[i, :] = self.data.xfrc_applied[self.link_ids[i]]
                force_torque_history.append(deepcopy(force_torque.tolist()))

                """ 运动学算法测试 """
                self.qpos, self.qvel = self._get_joint_states()
                
                """ 获取末端执行器的变换矩阵，可以用来验证运动学算法的正确性 """
                tf_world_to_end = self._get_body_states(self.link_ids[-1])['transformation_matrix']
                tf_world_to_center = self._get_body_states(self.circle2_center_id)['transformation_matrix']
                tf_center_to_end = np.linalg.inv(tf_world_to_center).dot(tf_world_to_end)
                
                """ 获取系统运行时间 """
                self._get_system_run_time(output=False)

                if self.sys_run_sta == 0 and self._get_system_run_time()[0] >= 1.0:
                    print("计算圆环上方的位姿目标")
                    targetMat = np.eye(4, dtype=np.float64)
                    targetMat[0:3, 3] = center_state['position'] + np.array([0.0, 0.0, 0.3])
                    targetMat[0:3, 0:3] = center_state['rotation_matrix'].dot(R.from_euler('x', 180, degrees=True).as_matrix()).dot(R.from_euler('z', 45, degrees=True).as_matrix())
                    q_ik, solFlag = self.probot.ikine(target=targetMat, q0=self.qpos, tol=1e-3, max_iter=1000, dt=1e-2, damp=1e-12)
                    self.sys_run_sta = 1
                
                elif self.sys_run_sta == 1:
                    print("机械臂运动到圆环上方位置")
                    if self._move_joint(np.deg2rad(q_ik), threshold=0.035):
                        self.sys_run_sta = 2

                elif self.sys_run_sta == 2 and self._get_system_run_time()[0] >= 2.5:
                    print("执行抓取动作")
                    self.weld_configuration(tf_center_to_end, case=1)
                    self.gripper_control(220)
                    self.sys_run_sta = 3

                elif self.sys_run_sta == 3 and self._get_system_run_time()[0] >= 3.5:
                    if self._move_joint(joint_init, threshold=0.035):
                        print("机械臂运动到初始位置")
                        self.sys_run_sta = 4
                
                elif self.sys_run_sta == 4 and self._get_system_run_time()[0] >= 5.0:
                    print("末端执行器速度:", np.linalg.norm(self._get_body_states(self.link_ids[-1])['linear_velocity']))
                    if np.linalg.norm(self._get_body_states(self.link_ids[-1])['linear_velocity']) >= 6.5:
                        self.data.eq_active[self.weld_id] = 0
                        self.gripper_control(0)
                        self.sys_run_sta = 5
                
                if self._get_system_run_time()[0] >= 5.0 and square_state['position'][2] >= 0.100:
                    # position_history.append(deepcopy(self.get_circle_state().tolist()))
                    # transMat_history.append(deepcopy(circle_state['transformation_matrix'].tolist()))

                    position_history.append(deepcopy(self.get_square_state().tolist()))
                    transMat_history.append(deepcopy(square_state['transformation_matrix'].tolist()))

                    # position_history.append(deepcopy(self.get_triangle_state().tolist()))
                    # transMat_history.append(deepcopy(triangle_state['transformation_matrix'].tolist()))

                for event in pyg.event.get():
                    if event.type == pyg.QUIT:
                        print("exit")
                        exit()
                    elif event.type == pyg.KEYDOWN:      
                        if event.key == pyg.K_s:
                            with open(cpath + r"\data_history.json", "w") as f:
                                json.dump({"force_torque": force_torque_history,
                                           "position_trajectory": position_history,
                                           "transMat_trajectory": transMat_history}, f)
                            print("手动触发数据保存动作")

                        elif event.key == pyg.K_r:
                            self.sys_run_sta = 0
                            force_torque_history = []
                            position_history = []
                            transMat_history = []
                            print("手动触发放置动作")

                # 执行仿真步骤
                self._step_simulation(delta=0.0)

            except KeyboardInterrupt: 
                self.sys_run_flag = False
                print("捕获到中断信号，停止仿真")
                self.close()

            except Exception as e:
                print(e)

    def reproduce(self):
        with open(cpath + r"\force_torque_data.json", 'r') as json_file:
            loaded_list = json.load(json_file)
        
        force_torque_list = loaded_list["force_torque"]
        
        json_idx = 0
        joint_init = np.array([0.377, -0.751, -2.200, 1.370, -0.251, -0.412, 1.190 + np.deg2rad(0.0)])

        while self.sys_run_flag:
            try:
                # self.viewer.cam.lookat = np.array([-0.0167, 0.5295, 0.234])
                # self.viewer.cam.distance = 3.1072314441448743
                # self.viewer.cam.azimuth = -178.19530469530457
                # self.viewer.cam.elevation = -23.391108891108896

                print('相机类型:', self.viewer.cam.type)
                print('相机位置 (xyz):', self.viewer.cam.lookat)
                print('相机距离:', self.viewer.cam.distance)
                print('相机方位角:', self.viewer.cam.azimuth)
                print('相机仰角:', self.viewer.cam.elevation)

                """ 获取并打印圆环和中心的状态信息 """
                center_state = self._get_body_states(self.circle2_center_id)
                print("中心位置:", center_state['position'], "中心速度:", center_state['linear_velocity'])

                """ 运动学算法测试 """
                self._get_joint_states()

                """ 获取末端执行器的变换矩阵，可以用来验证运动学算法的正确性 """
                tf_world_to_end = self._get_body_states(self.link_ids[-1])['transformation_matrix']
                tf_world_to_center = self._get_body_states(self.circle2_center_id)['transformation_matrix']
                tf_center_to_end = np.linalg.inv(tf_world_to_center).dot(tf_world_to_end)
                
                """ 获取系统运行时间 """
                self._get_system_run_time(output=False)

                """ 自动执行抓取和投掷动作 """
                if self.sys_run_sta == 0 and self._get_system_run_time()[0] >= 1.0:
                    print("计算圆环上方的位姿目标")
                    targetMat = np.eye(4, dtype=np.float64)
                    targetMat[0:3, 3] = center_state['position'] + np.array([0.0, 0.0, 0.3])
                    targetMat[0:3, 0:3] = center_state['rotation_matrix'].dot(R.from_euler('x', 180, degrees=True).as_matrix()).dot(R.from_euler('z', 45, degrees=True).as_matrix())
                    q_ik, solFlag = self.probot.ikine(target=targetMat, q0=self.qpos, tol=1e-3, max_iter=1000, dt=1e-2, damp=1e-12)
                    self.sys_run_sta = 1

                    # id_x = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'bottom_00')

                    # self.model.geom_rgba[140][:3] = [1.0, 0.0, 0.0]  # 设置为红色，值范围为 [0, 1]
                    # self.model.geom_rgba[140][3] = 1.0  # 设置不透明

                    # for geom_id in range(self.model.ngeom):
                    #     if self.model.geom_bodyid[geom_id] == id_x:
                    #         self.model.geom_rgba[geom_id][:3] = [1.0, 0.0, 0.0]  # 设置为红色，值范围为 [0, 1]
                    #         self.model.geom_rgba[geom_id][3] = 1.0  # 设置不透明
                
                elif self.sys_run_sta == 1:
                    print("机械臂运动到圆环上方位置")
                    if self._move_joint(np.deg2rad(q_ik), threshold=0.035):
                        self.sys_run_sta = 2
                   
                elif self.sys_run_sta == 2 and self._get_system_run_time()[0] >= 2.5:
                    print("执行抓取动作")
                    self.weld_configuration(tf_center_to_end, case=1)
                    self.GripperControl(220)
                    self.sys_run_sta = 3
                    
                elif self.sys_run_sta == 3 and self._get_system_run_time()[0] >= 3.5:
                    if self._move_joint(joint_init, threshold=0.035):
                        print("机械臂运动到初始位置")
                        self.sys_run_sta = 4

                elif self.sys_run_sta == 4 and self._get_system_run_time()[0] >= 5.0:
                    if np.linalg.norm(self._get_body_states(self.link_ids[-1])['linear_velocity']) >= 6.5:
                        self.data.eq_active[self.weld_id] = 0
                        self.GripperControl(0)
                        self.sys_run_sta = 5    

                try:
                    for i in range(8):  
                        self.data.xfrc_applied[self.link_ids[i], :] = force_torque_list[json_idx][i]
                    json_idx += 1
                except Exception as e:
                    pass
                
                self._step_simulation(delta=0.0)

            except KeyboardInterrupt: 
                self.sys_run_flag = False
                print("捕获到中断信号，停止仿真")
                self._close()

            except Exception as e:
                pass
    
    def demo(self):        
                # pyg.init()
        # self.screen = pyg.display.set_mode((640, 480), pyg.RESIZABLE)
        # pyg.display.set_caption("固定尺寸窗口")

        while self.sys_run_flag:
            try:
                """ 获取圆环状态信息 """
                ring_state = self._get_body_states(self.circle2_id)
                print("圆环位置:", ring_state['position'], "圆环速度:", ring_state['linear_velocity'], end=',  ')

                center_state = self._get_body_states(self.circle2_center_id)
                # print("中心位置:", center_state['position'], "中心速度:", center_state['linear_velocity'], end='\n')

                """ 运动学算法测试 """
                
                """ 获取末端执行器的变换矩阵，可以用来验证运动学算法的正确性 """
                tf_world_to_center = self._get_body_states(self.circle2_center_id)['transformation_matrix']

                # tf_world_to_center = np.eye(4, dtype=np.float64)
                # tf_world_to_center[:3,:3] = deepcopy(self.data.xmat[self.circle2_center_id].reshape(3, 3))
                # tf_world_to_center[:3, 3] = deepcopy(self.data.xpos[self.circle2_center_id])
                # print('中心变换矩阵:\n', tf_world_to_center)

                tf_world_to_end = self._get_body_states(body_id=self.link_ids[-1])['transformation_matrix']
                tf_center_to_end = np.linalg.inv(tf_world_to_center).dot(tf_world_to_end)
                tf_end_to_center = np.linalg.inv(tf_center_to_end)
                # print('末端相对于中心的变换矩阵:\n', tf_center_to_end)

                """ 打印各关节的力矩信息 """
                for i in range(7):
                    np.set_printoptions(precision=5, suppress=True)
                    print(f"关节{i+1}力矩:",  self._get_joint_force_info(self.joint_ids[i])['actuator'], end=',  ')
                print()                

                """ 获取系统运行时间 """
                self._get_joint_states()

                """用来给定机械臂的关节角速度,然后在某一时刻松开焊接约束，让机械臂以一定的速度运动，从而实现抛掷箱子的效果"""
                # 初始化机械臂位置，即将机械臂末端移动到圆环上方        
                if self._get_system_run_time()[0] >= 1.0 and self.sys_run_sta == 0:
                    targetMat = np.eye(4, dtype=np.float64)
                    targetMat[0:3, 3] = np.array(center_state['position']) + np.array([0.0, 0.0, 0.30])
                    targetMat[0:3, 0:3] = R.from_euler('xyz', [180, 0, 0], degrees=True).as_matrix()
                    q_ik, solFlag = self.probot.ikine(target=targetMat, q0=self.qpos, tol=1e-4, max_iter=10000, dt=1e-2, damp=1e-12)

                    self.sys_run_sta = 1
                
                elif self._get_system_run_time()[0] >= 2.0 and self.sys_run_sta == 1:
                    
                    if self._move_joint(goal=np.deg2rad(q_ik), threshold=0.030, ctrl_mode='ctrl'):
                        print("机械臂已到达目标位置")
                        self._get_system_run_time(output=True)
                        self.sys_run_sta = 2

                    self.data.ctrl[self.joint_ids[-1]] = 0.0  # 控制末端执行器位置
                    

                elif self._get_system_run_time()[0] >= 3.0 and self.sys_run_sta == 2:                    
                    self.model.eq_data[self.weld_id][0] = 0.0
                    self.model.eq_data[self.weld_id][1] = 0.0
                    self.model.eq_data[self.weld_id][2] = 0.0

                    self.model.eq_data[self.weld_id][3] = deepcopy(tf_center_to_end[0, 3])
                    self.model.eq_data[self.weld_id][4] = deepcopy(tf_center_to_end[1, 3])
                    self.model.eq_data[self.weld_id][5] = deepcopy(tf_center_to_end[2, 3]) - 0.05
                    rel_quat = R.from_matrix(tf_center_to_end[:3, :3].dot(R.from_euler('xyz', [0, 0, 0], degrees=True).as_matrix())).as_quat()

                    self.model.eq_data[self.weld_id][6] = rel_quat[3]
                    self.model.eq_data[self.weld_id][7] = rel_quat[0]
                    self.model.eq_data[self.weld_id][8] = rel_quat[1]
                    self.model.eq_data[self.weld_id][9] = rel_quat[2]

                    self.data.eq_active[self.weld_id] = 1
                    print("焊接约束方向:", self.model.eq_data[self.weld_id])
                    self.sys_run_sta = 3

                    self.data.ctrl[self.joint_ids[-1]] = 255.0  # 控制末端执行器位置

                # 机械臂移动到预设位置准备抛掷箱子
                # elif self.sys_run_sta == 2 and self._get_system_run_time()[0] >= 3.0:
                #     joint_angle = np.array([0, -0.41, 0, 2.98, 0, 0, 0])
                #     for i in range(7):
                #         self.data.ctrl[self.joint_ids[i]] = joint_angle[i]
                    
                #     self.sys_run_sta = 3
                
                # 机械臂抛掷箱子
                # elif self.sys_run_sta == 3 and self._get_system_run_time()[0] >= 6.0:
                #     joint_velocity = np.array([0.0, 2 * np.pi, 0.0, 0.0, 0.0, 0.0, 0.0])
                #     for i in range(7):  
                #         self.data.qvel[self.joint_ids[i]] = joint_velocity[i]
                
                # 在6.3秒时松开焊接约束
                # if self._get_system_run_time()[0] >= 6.10:
                #     self.data.eq_active[weld_id] = 0
                #     self.data.ctrl[self.joint_ids[-1]] = 0.0  # 控制末端执行器位置
                
                # 机械臂结束轨迹停止
                # if self._get_system_run_time()[0] >= 6.20:
                #     joint_velocity = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                #     for i in range(7):  
                #         self.data.qvel[self.joint_ids[i]] = joint_velocity[i]
                
                          # for event in pyg.event.get():
                #     if event.type == pyg.QUIT:
                #         print("exit")
                #         exit()
                #     elif event.type == pyg.KEYDOWN:      
                #         if event.key == pyg.K_s:
                #             with open(cpath + r"\motion_trajectory.json", "w") as f:
                #                 json.dump({"position_trajectory": position_history,
                #                            "transMat_trajectory": transMat_history}, f)
                #             print("手动触发数据保存动作")

                #         elif event.key == pyg.K_r:
                #             self.sys_run_sta = 0
                #             position_history = []
                #             transMat_history = []
                #             print("手动触发放置动作")

                self._step_simulation(delta=0.0)

            except KeyboardInterrupt:                
                self.sys_run_flag = False
                print("捕获到中断信号，停止仿真")
                self._close()

            except Exception as e:
                print("仿真过程中出现异常，停止仿真")
                print("异常信息:", e)
                self._close()

    def display(self):        
        # with open(cpath + r"\data_history.json", 'r') as json_file:
        #     loaded_list = json.load(json_file)
        
        # circle_points = loaded_list["position_trajectory"][600]

        while self.sys_run_flag:
            try:
                self._get_camera_params()
                dict_params = {'lookat': np.array([0.17546, 1.06091, 0.47576]),
                               'distance': 2.0993417089679594,
                               'azimuth': -179.9115044247782,
                               'elevation': -21.991150442477824}
                self._set_camera_params(dict_params)  

                """ 关节状态更新 """
                self._get_joint_states()

                """ 在场景中添加自定义几何体 """
                # self.viewer.user_scn.ngeom = 0

                # for i in range(len(circle_points)):
                #     mujoco.mjv_initGeom(self.viewer.user_scn.geoms[i],
                #                         type=mujoco.mjtGeom.mjGEOM_SPHERE,
                #                         size=np.array([0.005, 0.005, 0.005]).reshape(3, 1),
                #                         pos=np.array(circle_points[i]).reshape(3, 1),
                #                         mat=np.eye(3, dtype=np.float64).flatten(),
                #                         rgba=np.array([1.0, 0.0, 0.0, 0.1], dtype=np.float32)
                #     )
                #     self.viewer.user_scn.ngeom += 1

                square_state = self._get_body_states(self.square_body_id)
                circle_state = self._get_body_states(self.circle_body_id)       
                triangle_state = self._get_body_states(self.triangle_body_id)

                if self._get_system_run_time()[0] > 1.0 and self.sys_run_sta == 0:
                   targetMat = np.eye(4, dtype=np.float64)
                   targetMat[0:3, 3] = square_state['position'] + np.array([0.0, -0.25, 0.20])
                   targetMat[0:3, 0:3] = square_state['rotation_matrix'].dot(R.from_euler('x', 180, degrees=True).as_matrix()).dot(R.from_euler('z', 90, degrees=True).as_matrix())
                   q_ik, solFlag = self.probot.ikine(target=targetMat, q0=self.qpos, tol=1e-3, max_iter=1000, dt=1e-2, damp=1e-12)
                   print("计算逆解结果:", np.deg2rad(q_ik), f', 求解状态: {solFlag}')
                   self.sys_run_sta = 1
                
                elif self._get_system_run_time()[0] > 2.0 and self.sys_run_sta == 1:
                    joint_angle = np.deg2rad(q_ik)                
                    if self._move_joint(goal=joint_angle, threshold=0.030):
                        print("机械臂已到达目标位置")
                        self._get_system_run_time(output=True)
                        self.sys_run_sta = 2
                    

      

                # 执行仿真步骤
                self._step_simulation(delta=0.0)

            except KeyboardInterrupt: 
                self.sys_run_flag = False
                print("捕获到中断信号，停止仿真")
                self._close()

            except Exception as e:
                print("异常信息:", e)
                pass    

if __name__ == "__main__":
    demo = BoxForceDemo(path=cpath + r"\Environments\xarm7_tossing_gripper_xml\tossing.xml")
    control_mode = 4  # 1: 手动控制模式，2: 自动控制模式

    if control_mode == 1:
        demo.teach()
    elif control_mode == 2:
        demo.reproduce()
    elif control_mode == 3:
        demo.demo()
    elif control_mode == 4:
        demo.display()

