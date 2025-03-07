import mujoco
import numpy as np

# 加载模型
model = mujoco.MjModel.from_xml_path("/home/xunyang/Desktop/Projects/legged-loco/assets/robots/g1/g1_minimal_terrain.xml")
data = mujoco.MjData(model)

# 仿真一步以初始化数据
mujoco.mj_step(model, data)

# 遍历所有关节
for joint_id in range(model.njnt):
    # 获取关节名称
    name_start = model.name_jntadr[joint_id]
    joint_name = bytes(model.names[name_start:]).split(b'\x00')[0].decode('utf-8')

    # 获取关节类型
    jnt_type = model.jnt_type[joint_id]

    # 获取关节在 qpos 和 qvel 中的索引
    qpos_adr = model.jnt_qposadr[joint_id]  # 在 qpos 中的起始索引
    dof_adr = model.jnt_dofadr[joint_id]    # 在 qvel 中的起始索引

    # 根据关节类型提取位置和速度
    if jnt_type == mujoco.mjtJoint.mjJNT_HINGE or jnt_type == mujoco.mjtJoint.mjJNT_SLIDE:
        # 单自由度关节（旋转或滑动）
        pos = data.qpos[qpos_adr]
        vel = data.qvel[dof_adr]
        print(f"Joint '{joint_name}': pos={pos:.3f}, vel={vel:.3f}")
    elif jnt_type == mujoco.mjtJoint.mjJNT_BALL:
        # 球关节（四元数表示姿态）
        quat = data.qpos[qpos_adr : qpos_adr + 4]  # 四元数 [w, x, y, z]
        ang_vel = data.qvel[dof_adr : dof_adr + 3] # 角速度 [wx, wy, wz]
        print(f"Joint '{joint_name}': quat={quat}, ang_vel={ang_vel}")
    elif jnt_type == mujoco.mjtJoint.mjJNT_FREE:
        # 自由关节（6自由度）
        pos = data.qpos[qpos_adr : qpos_adr + 3]  # 位置 [x, y, z]
        quat = data.qpos[qpos_adr + 3 : qpos_adr + 7]  # 四元数 [w, x, y, z]
        lin_vel = data.qvel[dof_adr : dof_adr + 3]  # 线速度 [vx, vy, vz]
        ang_vel = data.qvel[dof_adr + 3 : dof_adr + 6]  # 角速度 [wx, wy, wz]
        print(f"Joint '{joint_name}': pos={pos}, quat={quat}, lin_vel={lin_vel}, ang_vel={ang_vel}")