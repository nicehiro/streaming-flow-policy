import mujoco
import mujoco.viewer
import numpy as np
import os
cpath = os.path.dirname(os.path.abspath(__file__))
ccpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# load your model
model = mujoco.MjModel.from_xml_path(cpath + r"\Environments\xarm7_tossing_gripper_xml\tossing.xml")
data = mujoco.MjData(model)

link6_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'geom_link2')


ghost_poses = []
GHOST_NUM = 10

def alpha(i):
    return 0.1 + i * 0.06


def add_mesh_ghost(scene, model, d, gid, rgba):
    """Render one mesh geom as ghost"""
    g = scene.geoms[scene.ngeom]

    # --- Fill basic info ---
    g.type = mujoco.mjtGeom.mjGEOM_MESH
    g.pos[:] = d.geom_xpos[gid]
    g.mat[:, :] = d.geom_xmat[gid].reshape(3, 3)
    g.rgba[:] = rgba

    # --- bind mesh id ---
    g.dataid = model.geom_dataid[gid]        # key: mesh id
    g.objtype = mujoco.mjtObj.mjOBJ_MESH
    g.objid = g.dataid

    g.size[:] = model.geom_size[gid]         # usually ignored for mesh
    # g.userdata = None

    scene.ngeom += 1


def add_basic_geom(scene, model, d, gid, rgba):
    """Render sphere/box/capsule/plane etc."""
    pos = d.geom_xpos[gid]
    mat = d.geom_xmat[gid]
    size = model.geom_size[gid]

    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom],
        model.geom_type[gid],
        size,
        pos,
        mat,
        rgba=rgba
    )
    scene.ngeom += 1


def draw_ghosts(model, data, ghost_poses, viewer):
    scene = viewer.user_scn
    scene.ngeom = 0

    for i, qpos in enumerate(ghost_poses):
        d = mujoco.MjData(model)
        d.qpos[:] = qpos
        mujoco.mj_forward(model, d)

      

        for gid in range(model.ngeom):
            if gid == link6_id:
                rgba = model.geom_rgba[gid].copy()
                rgba[3] = alpha(i)
                add_mesh_ghost(scene, model, d, gid, rgba)

        #     rgba = model.geom_rgba[gid].copy()
        #     rgba[3] = alpha(i)

        #     if model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_MESH:
        #         add_mesh_ghost(scene, model, d, gid, rgba)
        #     else:
        #         add_basic_geom(scene, model, d, gid, rgba)


with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)

        ghost_poses.append(data.qpos.copy())
        if len(ghost_poses) > GHOST_NUM:
            ghost_poses.pop(0)

        draw_ghosts(model, data, ghost_poses, viewer)

        viewer.sync()
