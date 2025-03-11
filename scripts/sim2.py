import time
import mujoco
import mujoco.viewer
import numpy as np
import torch
import yaml
from typing import Tuple, Optional

class RobotConfig:
    """机器人仿真配置类"""
    def __init__(self):
        # 基本配置
        self.xml_path = "./assets/robots/g1/g1_minimal_terrain.xml"
        self.policy_path = "./logs/rsl_rl/g1_vision_rough/2025-03-01_21-16-40_XXX/exported/policy.jit"
        self.joint_config_path = "./scripts/g1.yaml"
        
        # 时间参数
        self.simulation_dt = 0.00125
        self.control_decimation = 16  # 50Hz 控制频率
        self.simulation_duration = 1600.0
        
        # 观测参数
        self.num_actions = 37
        self.policy_obs_dim = 771
        self.proprio_obs_dim = 123
        self.history_length = 0
        self.cmd = np.array([0.8, 0.0, 0.0], dtype=np.float32)
        
        # 加载关节配置
        self._load_joint_config()
        
        # 验证参数
        self._validate()

    def _load_joint_config(self):
        """加载关节配置文件"""
        with open(self.joint_config_path, "r") as f:
            config = yaml.safe_load(f)
            self.kps = np.array(config["kps"], dtype=np.float32)
            self.kds = np.array(config["kds"], dtype=np.float32)
            self.default_angles = np.array(config["default_angles"], dtype=np.float32)

    def _validate(self):
        """参数校验"""
        assert len(self.kps) == self.num_actions, "KP参数数量不匹配"
        assert len(self.kds) == self.num_actions, "KD参数数量不匹配"
        assert len(self.default_angles) == self.num_actions, "默认角度数量不匹配"

class RobotController:
    """机器人主控制类"""
    def __init__(self, config: RobotConfig):
        self.cfg = config
        
        # 初始化MuJoCo
        self.model = mujoco.MjModel.from_xml_path(self.cfg.xml_path)
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = self.cfg.simulation_dt
        
        # 初始化策略
        self._init_policy()
        
        # 初始化缓冲区
        self._init_buffers()
        
        # 控制状态
        self.counter = 0
        self.target_dof_pos = self.cfg.default_angles.copy()
        self.current_action = np.zeros(self.cfg.num_actions, dtype=np.float32)
        
    def _init_policy(self):
        """加载神经网络策略"""
        try:
            self.policy = torch.jit.load(self.cfg.policy_path)
            self.policy.eval()
        except Exception as e:
            raise RuntimeError(f"策略加载失败: {e}")

    def _init_buffers(self):
        """初始化观测缓冲区"""
        # Proprioceptive历史缓冲区
        self.proprio_hist_buf = torch.zeros(
            (self.cfg.history_length, self.cfg.proprio_obs_dim),
            dtype=torch.float32
        )
        
        # 当前观测缓存
        self._current_policy_obs = np.zeros(self.cfg.policy_obs_dim, dtype=np.float32)
        self._current_proprio_obs = np.zeros(self.cfg.proprio_obs_dim, dtype=np.float32)

    def reindex_joint_data(self, raw_data: np.ndarray) -> np.ndarray:
        """关节数据重索引"""
        REINDEX_MAP = [
            0,6,12,1,7,13,25,2,8,14,26,3,9,15,27,4,10,16,28,
            5,11,17,29,18,20,22,30,32,34,19,21,23,31,33,35,24,36
        ]
        return raw_data[REINDEX_MAP]

    @staticmethod
    def get_gravity_orientation(quat: np.ndarray) -> np.ndarray:
        """从四元数计算重力方向"""
        qw, qx, qy, qz = quat
        return np.array([
            2*(-qz*qx + qw*qy),
            -2*(qz*qy + qw*qx),
            1 - 2*(qw**2 + qz**2)
        ], dtype=np.float32)

    def compute_pd_control(self) -> np.ndarray:
        """计算PD控制力矩"""
        qpos = self.reindex_joint_data(self.data.qpos[7:])
        qvel = self.reindex_joint_data(self.data.qvel[6:])
        
        pos_error = self.target_dof_pos - qpos
        vel_error = np.zeros_like(qvel) - qvel  # 目标速度为0
        
        return pos_error * self.cfg.kps + vel_error * self.cfg.kds

    def _update_observations(self):
        """更新观测数据"""
        # 原始数据
        qpos = self.reindex_joint_data(self.data.qpos[7:])
        qvel = self.reindex_joint_data(self.data.qvel[6:])
        quat = self.data.qpos[3:7]
        lin_vel = self.data.qvel[:3]
        ang_vel = self.data.qvel[3:6]
        
        # 处理观测数据
        gravity = self.get_gravity_orientation(quat)
        qpos_rel = qpos - self.cfg.default_angles
        
        # 构建policy观测
        self._current_policy_obs[0:3] = lin_vel
        self._current_policy_obs[3:6] = ang_vel
        self._current_policy_obs[6:9] = gravity
        self._current_policy_obs[9:12] = self.cfg.cmd
        self._current_policy_obs[12:12+self.cfg.num_actions] = qpos_rel
        self._current_policy_obs[12+self.cfg.num_actions:12+2*self.cfg.num_actions] = qvel
        self._current_policy_obs[12+2*self.cfg.num_actions:12+3*self.cfg.num_actions] = self.current_action
        self._current_policy_obs[12+3*self.cfg.num_actions:] = 0  # Lidar占位

        # 构建proprio观测
        self._current_proprio_obs[0:12] = self._current_policy_obs[0:12]
        self._current_proprio_obs[12:] = self._current_policy_obs[12:12+3*self.cfg.num_actions]

    def _update_history(self):
        """更新历史缓冲区"""
        proprio_tensor = torch.from_numpy(self._current_proprio_obs)
        self.proprio_hist_buf = torch.roll(self.proprio_hist_buf, shifts=-1, dims=0)
        self.proprio_hist_buf[-1] = proprio_tensor

    def get_observation(self) -> torch.Tensor:
        """获取完整观测"""
        self._update_observations()
        # self._update_history()
        
        policy_obs_tensor = torch.from_numpy(self._current_policy_obs)
        hist_obs = self.proprio_hist_buf.view(-1)
        # return torch.cat([policy_obs_tensor, hist_obs], dim=0)
        return policy_obs_tensor

    def step(self):
        """执行单步控制"""
        # 应用PD控制
        self.data.ctrl[:] = self.compute_pd_control()
        
        # 执行物理仿真
        mujoco.mj_step(self.model, self.data)
        self.counter += 1
        
        # 策略更新
        if self.counter % self.cfg.control_decimation == 0:
            with torch.no_grad():
                obs = self.get_observation()
                action = self.policy(obs)
                self.current_action = torch.clamp(action, -20, 20).numpy().squeeze()
                self.target_dof_pos = self.current_action * 0.5 + self.cfg.default_angles

def main():
    # 初始化系统
    config = RobotConfig()
    controller = RobotController(config)
    
    # 运行仿真
    with mujoco.viewer.launch_passive(controller.model, controller.data) as viewer:
        start_time = time.time()
        
        while viewer.is_running() and (time.time() - start_time) < config.simulation_duration:
            step_start = time.perf_counter()
            
            # 执行控制步
            controller.step()
            
            # 同步可视化
            viewer.sync()
            
            # 精确计时
            elapsed = time.perf_counter() - step_start
            if (sleep_time := config.simulation_dt - elapsed) > 0:
                time.sleep(sleep_time)

if __name__ == "__main__":
    main()