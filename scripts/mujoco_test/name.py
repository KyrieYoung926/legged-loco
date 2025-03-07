import mujoco

# 加载模型
model = mujoco.MjModel.from_xml_path("/home/xunyang/Desktop/Projects/legged-loco/assets/robots/g1/g1_minimal_terrain.xml")

# 遍历所有关节并打印名称
for joint_id in range(model.njnt):
    name_start = model.name_jntadr[joint_id]
    joint_name_bytes = bytes(model.names[name_start:]).split(b'\x00')[0]
    joint_name = joint_name_bytes.decode('utf-8')
    print(f"Joint {joint_id}: '{joint_name}'")