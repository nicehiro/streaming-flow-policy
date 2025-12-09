#!/usr/bin/python
import sys
import os
cpath = os.path.abspath(os.path.dirname(__file__))
ccpath = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import roboticstoolbox as rtb
from roboticstoolbox import RevoluteDH, RevoluteMDH

class xArmRtbCls:
    def __init__(self):
        self.base = np.eye(4, dtype=np.float64)
        self.tool = np.eye(4, dtype=np.float64)
        self.tool[2, 3] = 250.0 / 1000

        self.rob_handle = rtb.DHRobot(
            [
                rtb.RevoluteDH(alpha=-np.pi / 2,    a=0,           offset=0,   d=267/1000,     qlim=[np.deg2rad(-350), np.deg2rad(350)]),
                rtb.RevoluteDH(alpha= np.pi / 2,    a=0,           offset=0,   d=0,            qlim=[np.deg2rad(-110), np.deg2rad(110)]),
                rtb.RevoluteDH(alpha= np.pi / 2,    a=52.5/1000,   offset=0,   d=293/1000,     qlim=[np.deg2rad(-350), np.deg2rad(350)]),
                rtb.RevoluteDH(alpha= np.pi / 2,    a=77.5/1000,   offset=0,   d=0,            qlim=[np.deg2rad(-10),  np.deg2rad(210)]),
                rtb.RevoluteDH(alpha= np.pi / 2,    a=0,           offset=0,   d=342.5/1000,   qlim=[np.deg2rad(-350), np.deg2rad(350)]),
                rtb.RevoluteDH(alpha=-np.pi / 2,    a=76/1000,     offset=0,   d=0,            qlim=[np.deg2rad(-90),  np.deg2rad(170)]),
                rtb.RevoluteDH(alpha=0,             a=0,           offset=0,   d=97/1000,      qlim=[np.deg2rad(-350), np.deg2rad(350)]),
            ],
            name='xArm7',
            base=self.base,
            tool=self.tool,
        )
        # print(self.rob_handle)

    def fkine(self, q, is_radian=False):
        if is_radian is False:
            transformationMatrix = self.rob_handle.fkine(np.deg2rad(q))
        else:
            transformationMatrix = self.rob_handle.fkine(q)

        results = []
        results.append(transformationMatrix)

        return results
    
    def ikine(self, Tep, q0, method='LM'):
        q_init = deepcopy(np.deg2rad(q0))

        if method == 'LM':
            res = self.rob_handle.ik_LM(Tep=Tep, q0=q_init, joint_limits=True, tol=1e-10, ilimit=100, slimit=500, method="chan")

        if method == "GN":
            res = self.rob_handle.ik_GN(Tep=Tep, q0=q_init, joint_limits=True, tol=1e-10, ilimit=100, slimit=500)
        
        if method == "NR":
            res = self.rob_handle.ik_NR(Tep=Tep, q0=q_init, joint_limits=True, tol=1e-10, ilimit=100, slimit=500)
        
        return np.rad2deg(res[0])
        

if __name__ == "__main__":
    try:
        xArm7 = xArmRtbCls()
        
        joint = np.rad2deg(np.random.rand(7))
        print("随机关节角度（角度制）：", joint)

        random_pose = xArm7.fkine(joint, is_radian=False)[0]
        print("对应的末端位姿：\n", random_pose)

        ik_joint = xArm7.ikine(random_pose, np.random.rand(7), method='LM')
        print("逆解得到的关节角度（角度制）：", ik_joint)

        ik_pose = xArm7.fkine(ik_joint, is_radian=False)[0]
        print("逆解得到的末端位姿：\n", ik_pose)

        xArm7.rob_handle.teach(np.random.rand(1, 7))

    except Exception as e:
        print("发生错误:", e)