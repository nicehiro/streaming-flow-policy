import mujoco
import mujoco.viewer
import numpy as np
import os
cpath = os.path.dirname(os.path.abspath(__file__))
ccpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 加载机械臂模型
model = mujoco.MjModel.from_xml_path(cpath + r"\Environments\xarm7_tossing_gripper_xml\tossing.xml")
data = mujoco.MjData(model)


# link_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f'link{i + 1}') for i in range(7)]

link6_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "geom_link4")
print(f"Link 6 ID: {link6_id}")


# 用于存储历史姿态（每个元素是 qpos 的 copy）
ghost_poses = []

# 设置残影数量（越大残影越长）
GHOST_COUNT = 12

# 每个残影的透明度（从弱到强）
def alpha(i):
    return 0.05 + 0.05 * i




    

# -----------------------
# 关键：生成一个“透明 ghost 场景”
# -----------------------
def build_ghost_scene(model, data, ghost_poses, viewer):
    ghost_scene = viewer.user_scn
    ghost_scene.ngeom = 0

    for i, q in enumerate(ghost_poses):

        # 创建一个临时 data，用来呈现 ghost 的位置
        ghost_data = mujoco.MjData(model)
        ghost_data.qpos[:] = q
        # mujoco.mj_forward(model, ghost_data)

        # 遍历所有 geoms，并生成透明版本
        for gid in range(model.ngeom):
            if gid == link6_id:
                # 原始 geom 的位置、方向
                pos = ghost_data.geom_xpos[gid]
                mat = ghost_data.geom_xmat[gid]
                size = model.geom_size[gid]

                # 残影半透明颜色
                rgba = model.geom_rgba[gid].copy()
                rgba[3] = alpha(i)            # 设置透明度

                mujoco.mjv_initGeom(
                    ghost_scene.geoms[ghost_scene.ngeom],
                    model.geom_type[gid],
                    size,
                    pos,
                    mat,
                    rgba=rgba
                )

                ghost_scene.ngeom += 1

    return ghost_scene


# -----------------------
# 主循环：仿真 + 渲染 + 残影
# -----------------------
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)

        # 保存当前姿态
        ghost_poses.append(data.qpos.copy())
        if len(ghost_poses) > GHOST_COUNT:
            ghost_poses.pop(0)

        # 构建透明残影
        build_ghost_scene(model, data, ghost_poses, viewer)

        viewer.sync()
