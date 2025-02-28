import time

import mujoco.viewer
import mujoco
import numpy as np
import torch
import yaml
import trimesh
import warp as wp

def get_gravity_orientation(quaternion):
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)

    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd

def get_lidar_data(d):
    mesh_path = "/home/xunyang/Desktop/Projects/legged-loco/meshes/terrain.obj"
    obstacle_mesh = trimesh.load(mesh_path)
    translation = trimesh.transformations.translation_matrix([x, y, z])
    obstacle_mesh.apply_transform(translation)
    vertices = obstacle_mesh.vertices  # (N, 3)
    triangles = obstacle_mesh.faces    # (M, 3)

    vertex_tensor = torch.tensor(vertices, device=self.device, dtype=torch.float32)
    faces_tensor = torch.tensor(triangles.flatten(), dtype=torch.int32, device=self.device)

    vertex_wp = wp.from_torch(vertex_tensor, dtype=wp.vec3)
    faces_wp = wp.from_tensor(faces_tensor, dtype=wp.int32)

    wp_mesh = wp.Mesh(points=vertex_wp, indices=faces_wp)
    combined_mesh = trimesh.util.concatenate([obstacle_mesh, terrain_mesh])
    ray_vectors = torch.zeros((num_scan_lines, num_points, 3))
    # 计算每个射线的方位角和俯仰角
    for i in range(num_scan_lines):
        for j in range(num_points):
            azimuth = horizontal_fov_min + j * (horizontal_fov / num_points)
            elevation = vertical_fov_min + i * (vertical_fov / num_scan_lines)
            ray_vectors[i,j] = [cos(azimuth)*cos(elevation), 
                            sin(azimuth)*cos(elevation), 
                            sin(elevation)]
    # 在Warp Kernel中计算
    wp.launch(
        kernel=lidar_raycast_kernel,
        dim=(num_envs, num_scan_lines, num_points),
        inputs=[wp_mesh.id, ray_origins, ray_vectors, max_distance],
        outputs=[distances]
    )
    self.local_dist = wp.to_torch(distances)  # 形状：(num_envs, num_scan_lines, num_points)
    lidar_obs = torch.log2(self.local_dist + 1) * self.obs_scales.lidar
    self.obs_buf = torch.cat([self.obs_buf, lidar_obs], dim=-1)

def load_joint_config(joint_config_path):
    with open(joint_config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        kps = np.array(config["kps"], dtype=np.float32)
        kds = np.array(config["kds"], dtype=np.float32)
        default_angles = np.array(config["default_angles"], dtype=np.float32)
    return kps, kds, default_angles

def get_obs(data, simcfg, proprio_obs_buf):

    # create observation
    qj = data.qpos[7:]
    dqj = data.qvel[6:]
    quat = data.qpos[3:7]
    vel = data.qvel[:3]
    omega = data.qvel[3:6]
    qj_rel = qj - simcfg.default_angles
    gravity_orientation = get_gravity_orientation(quat)
    lidar_measurement = np.zeros(648, dtype=np.float32)  

    # policy observation
    policy_obs[:3] = vel
    policy_obs[3:6] = omega
    policy_obs[6:9] = gravity_orientation
    policy_obs[9:12] = simcfg.cmd
    policy_obs[9 : 9 + simcfg.num_actions] = qj_rel
    policy_obs[9 + simcfg.num_actions : 9 + 2 * simcfg.num_actions] = dqj
    policy_obs[9 + 2 * simcfg.num_actions : 9 + 3 * simcfg.num_actions] = action 
    policy_obs[9 + 3 * simcfg.num_actions : 9 + 3 * simcfg.num_actions + 648] = lidar_measurement # 771
    policy_obs_tensor = torch.from_numpy(policy_obs).float()
    
    # proprio observation
    proprio_obs[:3] = vel
    proprio_obs[3:6] = omega
    proprio_obs[6:9] = gravity_orientation
    proprio_obs[9:12] = simcfg.cmd
    proprio_obs[9 : 9 + simcfg.num_actions] = qj_rel
    proprio_obs[9 + simcfg.num_actions : 9 + 2 * simcfg.num_actions] = dqj
    proprio_obs[9 + 2 * simcfg.num_actions : 9 + 3 * simcfg.num_actions] = action 
    proprio_obs_tensor = torch.from_numpy(proprio_obs).float()

    # Update proprio_obs buffer
    proprio_obs_buf = torch.roll(proprio_obs_buf, shifts=-1, dims=0)
    proprio_obs_buf[simcfg.history_length-1, :] = proprio_obs_tensor

    # Concatenate current observation
    proprio_obs_history = proprio_obs_buf.view(-1)
    obs = torch.cat([policy_obs_tensor, proprio_obs_history], dim=0)
    # history_proprio_obs = torch.roll(history_proprio_obs, shifts=-1, dims=0)
    # history_proprio_obs[-1, :] = torch.from_numpy(proprio_obs)

    # obs_buf = np.concatenate((policy_obs, history_proprio_obs.numpy().flatten()))
    # obs_tensor = torch.from_numpy(obs_buf).unsqueeze(0)
    # policy inference
    return obs, proprio_obs_buf

class SimConfig:
    # load config
    joint_config_path = "/home/xunyang/Desktop/Projects/legged-loco/scripts/g1.yaml"
    policy_path = "/home/xunyang/Desktop/Projects/legged-loco/logs/rsl_rl/g1_vision_rough/2025-02-25_21-49-44_g1-blind/exported/policy.jit"
    xml_path = "/home/xunyang/Desktop/Projects/legged-loco/assets/robots/g1/g1_minimal_terrain.xml"

    simulation_duration = 1600.0
    simulation_dt = 0.00125
    control_decimation = 16     # Controller update frequency (meets the requirement of simulation_dt * controll_decimation=0.02; 50Hz)
    num_actions = 37
    num_obs = 1878       
    history_length = 9
    cmd = np.array([0.8, 0.0, 0.0], dtype=np.float32)
    kps, kds, default_angles = load_joint_config(joint_config_path)
    policy_obs_dim = 771
    proprio_obs_dim = 123

if __name__ == "__main__":

    simcfg = SimConfig()
    # init buffer
    counter = 0
    action = np.zeros(simcfg.num_actions, dtype=np.float32)
    target_dof_pos = simcfg.default_angles.copy()
    policy_obs = np.zeros(simcfg.policy_obs_dim, dtype=np.float32)
    proprio_obs = np.zeros(simcfg.proprio_obs_dim, dtype=np.float32)
    proprio_obs_buf = torch.zeros(simcfg.history_length, simcfg.proprio_obs_dim, dtype=torch.float)
    history_proprio_obs = torch.zeros(simcfg.history_length, simcfg.proprio_obs_dim, dtype=torch.float)

    # Load robot model
    m = mujoco.MjModel.from_xml_path(simcfg.xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simcfg.simulation_dt

    obs, proprio_obs_buf = get_obs(d, simcfg, proprio_obs_buf)
    # load policy
    policy = torch.jit.load(simcfg.policy_path)

    with mujoco.viewer.launch_passive(m, d) as viewer:
        # Close the viewer automatically after simulation_duration wall-seconds.
        start = time.time()
        
        while viewer.is_running() and time.time() - start < simcfg.simulation_duration:
            step_start = time.time()
            tau = pd_control(target_dof_pos, d.qpos[7:], simcfg.kps, np.zeros_like(simcfg.kds), d.qvel[6:], simcfg.kds)
            d.ctrl[:] = tau
            # mj_step can be replaced with code that also evaluates
            # a policy and applies a control signal before stepping the physics.
            mujoco.mj_step(m, d)
            counter += 1

            if counter % simcfg.control_decimation == 0:
                # get observation
                obs, proprio_obs_buf = get_obs(d, simcfg, proprio_obs_buf)
                action = policy(obs)
                action = torch.clamp(action, -20, 20)
                action = action.detach().numpy().squeeze()
                # transform action to target_dof_pos
                target_dof_pos = action*0.5 + simcfg.default_angles

            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
