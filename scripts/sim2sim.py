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
    pp = (target_q - q) * kp
    dd = (target_dq - dq) * kd
    return pp + dd

def load_joint_config(joint_config_path):
    with open(joint_config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        kps = np.array(config["kps"], dtype=np.float32)
        kds = np.array(config["kds"], dtype=np.float32)
        default_angles = np.array(config["default_angles"], dtype=np.float32)
    return kps, kds, default_angles



def reindex_data(data):
    reindexed_data = [
        data[0],
        data[6],
        data[12],
        data[1],
        data[7],
        data[13],
        data[25],
        data[2],
        data[8],
        data[14],
        data[26],
        data[3],
        data[9],
        data[15],
        data[27],
        data[4],
        data[10],
        data[16],
        data[28],
        data[5],
        data[11],
        data[17],
        data[29],
        data[18],
        data[20],
        data[22],
        data[30],
        data[32],
        data[34],
        data[19],
        data[21],
        data[23],
        data[31],
        data[33],
        data[35],
        data[24],
        data[36],
        
    ]
    return reindexed_data

def get_obs(d, simcfg, proprio_obs_buf, history_proprio_obs, action, policy_obs, proprio_obs):
    # create observation
    base_vel, base_omega, joint_pos, joint_vel, base_quat = get_from_mjdata(d)

    joint_pos_rel = joint_pos - simcfg.default_angles

    gravity_orientation = get_gravity_orientation(base_quat)
    lidar_measurement = np.zeros(648, dtype=np.float32)  

    # policy observation
    policy_obs[:3] = base_vel
    policy_obs[3:6] = base_omega
    policy_obs[6:9] = gravity_orientation
    policy_obs[9:12] = simcfg.cmd
    policy_obs[12 : 12 + simcfg.num_actions] = joint_pos_rel
    policy_obs[12 + simcfg.num_actions : 12 + 2 * simcfg.num_actions] = joint_vel
    policy_obs[12 + 2 * simcfg.num_actions : 12 + 3 * simcfg.num_actions] = action 
    policy_obs[12 + 3 * simcfg.num_actions : 12 + 3 * simcfg.num_actions + 648] = lidar_measurement # 771
    policy_obs_tensor = torch.from_numpy(policy_obs).float()
    
    # proprio observation
    proprio_obs[:3] = base_vel
    proprio_obs[3:6] = base_omega
    proprio_obs[6:9] = gravity_orientation
    proprio_obs[9:12] = simcfg.cmd
    proprio_obs[12 : 12 + simcfg.num_actions] = joint_pos_rel
    proprio_obs[12 + simcfg.num_actions : 12 + 2 * simcfg.num_actions] = joint_vel
    proprio_obs[12 + 2 * simcfg.num_actions : 12 + 3 * simcfg.num_actions] = action 
    proprio_obs_tensor = torch.from_numpy(proprio_obs).float()

    # Update proprio_obs buffer
    proprio_obs_buf = torch.roll(proprio_obs_buf, shifts=-1, dims=0)
    proprio_obs_buf = torch.cat([proprio_obs_buf[:-1], proprio_obs_tensor.unsqueeze(0)], dim=0)

    # Concatenate current observation
    history_proprio_obs = proprio_obs_buf.view(-1)
    obs = torch.cat([policy_obs_tensor, history_proprio_obs], dim=0)

    return policy_obs_tensor, proprio_obs_buf, history_proprio_obs

def get_from_mjdata(data):
    base_vel = data.qvel[:3]
    base_omega = data.qvel[3:6]
    joint_pos = reindex_data(data.qpos[7:])
    joint_vel = reindex_data(data.qvel[6:])
    base_quat = data.qpos[3:7]
    return base_vel, base_omega, joint_pos, joint_vel, base_quat

def get_init_obs(d, simcfg, proprio_obs_buf, history_proprio_obs, action, policy_obs, proprio_obs):
    # create observation
    base_vel, base_omega, joint_pos, joint_vel, base_quat = get_from_mjdata(d)

    joint_pos_rel = joint_pos - simcfg.default_angles

    gravity_orientation = get_gravity_orientation(base_quat)
    lidar_measurement = np.zeros(648, dtype=np.float32)  

    # policy observation
    policy_obs[:3] = base_vel
    policy_obs[3:6] = base_omega
    policy_obs[6:9] = gravity_orientation
    policy_obs[9:12] = simcfg.cmd
    policy_obs[12 : 12 + simcfg.num_actions] = joint_pos_rel
    policy_obs[12 + simcfg.num_actions : 12 + 2 * simcfg.num_actions] = joint_vel
    policy_obs[12 + 2 * simcfg.num_actions : 12 + 3 * simcfg.num_actions] = action # 0 at first
    policy_obs[12 + 3 * simcfg.num_actions : 12 + 3 * simcfg.num_actions + 648] = lidar_measurement # 771
    policy_obs_tensor = torch.from_numpy(policy_obs).float()
    
    # proprio observation
    proprio_obs[:3] = base_vel
    proprio_obs[3:6] = base_omega
    proprio_obs[6:9] = gravity_orientation
    proprio_obs[9:12] = simcfg.cmd
    proprio_obs[12 : 12 + simcfg.num_actions] = joint_pos_rel
    proprio_obs[12 + simcfg.num_actions : 12 + 2 * simcfg.num_actions] = joint_vel
    proprio_obs[12 + 2 * simcfg.num_actions : 12 + 3 * simcfg.num_actions] = action 
    proprio_obs_tensor = torch.from_numpy(proprio_obs).float()

    # Update proprio_obs buffer
    proprio_obs_buf = torch.cat([proprio_obs_tensor.unsqueeze(0)] * simcfg.history_length, dim=0)

    # Concatenate current observation
    history_proprio_obs = proprio_obs_buf.view(-1)

    obs = torch.cat([policy_obs_tensor, history_proprio_obs], dim=0)

    return policy_obs_tensor, proprio_obs_buf, history_proprio_obs

class SimConfig:
    # load config
    joint_config_path = "./scripts/g1.yaml"
    # policy_path = "./logs/rsl_rl/g1_vision_rough/2025-02-25_21-49-44_g1-blind/exported/policy.jit"
    policy_path = "/home/xunyang/Desktop/Projects/legged-loco/logs/rsl_rl/g1_vision_rough/2025-03-01_21-16-40_XXX/exported/1policy.jit"
    xml_path = "./assets/robots/g1_hand/g1.xml"

    simulation_duration = 1600.0
    simulation_dt = 0.00125
    control_decimation = 16     # Controller update frequency (meets the requirement of simulation_dt * controll_decimation=0.02; 50Hz)
    num_actions = 37
    num_obs = 771       
    history_length = 9
    cmd = np.array([0.4, 0.0, 0.0], dtype=np.float32)
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
    history_proprio_obs = torch.zeros(simcfg.history_length*simcfg.proprio_obs_dim, dtype=torch.float)

    # Load robot model
    m = mujoco.MjModel.from_xml_path(simcfg.xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simcfg.simulation_dt
    # 仿真一步以初始化数据
    mujoco.mj_step(m, d)

    obs, proprio_obs_buf, history_proprio_obs = get_init_obs(d, simcfg, proprio_obs_buf, history_proprio_obs, action, policy_obs, proprio_obs)
    
    # load policy
    policy = torch.jit.load(simcfg.policy_path)

    with mujoco.viewer.launch_passive(m, d) as viewer:
        # Close the viewer automatically after simulation_duration wall-seconds.
        start = time.time()
        while viewer.is_running() and time.time() - start < simcfg.simulation_duration:
            with torch.inference_mode():    # 
                step_start = time.time()

                
                               
                if counter % simcfg.control_decimation == 0: 
                    obs, proprio_obs_buf, history_proprio_obs = get_obs(d, simcfg, proprio_obs_buf, history_proprio_obs, action, policy_obs, proprio_obs)
                    
                    action = policy(obs)
                    action = torch.clamp(action, -20, 20)
                    
                    # target_dof_pos = torch.clamp(action, -20, 20).detach().numpy().squeeze()
                    # transform action to target_dof_pos
                    _action = action.detach().numpy().squeeze()
                    target_dof_pos = _action*0.5 + simcfg.default_angles


                    
                tau_limit = 200. * np.ones(37, dtype=np.double)  
                tau = pd_control(target_dof_pos, reindex_data(d.qpos[7:]), simcfg.kps, np.zeros_like(simcfg.kds), reindex_data(d.qvel[6:]), simcfg.kds)
                tau = np.clip(tau, -tau_limit, tau_limit)  

                d.ctrl[:] = tau
                mujoco.mj_step(m, d)
                counter += 1
                

                # Pick up changes to the physics state, apply perturbations, update options from GUI.
                viewer.sync()

                # Rudimentary time keeping, will drift relative to wall clock.
                time_until_next_step = m.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
