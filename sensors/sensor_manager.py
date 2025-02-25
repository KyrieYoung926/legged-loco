from isaacgym import gymapi
from isaacgym import gymtorch
import os

from legged_gym.env_manager.base_env_manager import BaseManager
from aerial_gym.registry.robot_registry import robot_registry
import torch

# get all the sensor classes
from legged_gym.sensors.isaacgym_camera_sensor import IsaacGymCameraSensor
from legged_gym.sensors.warp.warp_sensor import WarpSensor
from legged_gym.sensors.imu_sensor import IMUSensor


from aerial_gym.utils.logging import CustomLogger
import pytorch3d.transforms as p3d_transforms
from legged_gym.envs.a1.a1_config import A1RoughCfg
logger = CustomLogger("robot_manager")


class SensorManagerIGE(BaseManager):
    def __init__(self, global_sim_dict, robot_name, controller_name, device):
        logger.debug("Initializing RobotManagerIGE")
        # self.gym = global_sim_dict["gym"]
        # self.sim = global_sim_dict["sim"]
        # self.env_config = global_sim_dict["env_cfg"]
        # self.use_warp = global_sim_dict["use_warp"]
        # self.cfg.env.num_envs = global_sim_dict["num_envs"]
        # create the robot from the name registry and use the configs from this created robot.
        # self.robot, robot_config = robot_registry.make_robot(
        #     robot_name, controller_name, self.env_config, device
        # )

        # Super-class is initialized when the robot registry tells us what the robot config is
        super().__init__(A1RoughCfg, device)

        self.robot_handles = []  # list of robot handles

        self.camera_sensor = None
        self.warp_sensor = None
        self.lidar_sensor = None
        self.imu_sensor = None
        self.has_IGE_sensors = False
        

        self.use_warp = self.cfg.sensor_config.use_warp
        self.enable_camera = True
        self.enable_lidar = False
        if self.use_warp == False:
            if self.cfg.sensor_config.enable_camera:
                logger.debug("Initializing Isaac Gym camera sensor")
                self.camera_sensor = IsaacGymCameraSensor(
                    self.cfg.sensor_config.camera_config,
                    self.cfg.env.num_envs,
                    self.gym,
                    self.sim,
                    self.device,
                )
                logger.debug("[DONE] Initializing Isaac Gym camera sensor")
            if self.cfg.sensor_config.enable_lidar:
                raise ValueError(
                    "Lidar sensors are not supported using Isaac Gym Rendering. Please enable warp."
                )
        elif self.use_warp == True and (
            self.cfg.enable_camera and self.enable_lidar
        ):
            logger.warning(
                "Warp is enabled. Appropriate camera sensors will be spawned using warp."
            )
            raise ValueError("Do not use both camera and lidar sensors together for now.")

            # Warp sensors are not instantiated here. They are prepared in the prepare_for_sim function as they require a ready environment to work with.

        logger.debug("[DONE] Initializing RobotManagerIGE")

        return


    def prepare_for_sim(self, global_tensor_dict):
        self.global_tensor_dict={}
        if not self.use_warp:
            logger.error("Not using warp. Initializing sensors")
            if self.cfg.sensor_config.enable_lidar:
                raise ValueError(
                    "Lidar sensors are not supported using Isaac Gym Rendering. Please enable warp."
                )
        
            if self.cfg.sensor_config.enable_camera:
                self.image_tensor = torch.zeros(
                    (
                        self.cfg.env.num_envs,
                        self.cfg.sensor_config.camera_config.num_sensors,
                        self.cfg.sensor_config.camera_config.height,
                        self.cfg.sensor_config.camera_config.width,
                    ),
                    device=self.device,
                    requires_grad=False,
                )
                self.global_tensor_dict["depth_range_pixels"] = self.image_tensor

                if self.cfg.sensor_config.camera_config.segmentation_camera:
                    self.segmentation_tensor = torch.zeros(
                        (
                            self.cfg.env.num_envs,
                            self.cfg.sensor_config.camera_config.num_sensors,
                            self.cfg.sensor_config.camera_config.height,
                            self.cfg.sensor_config.camera_config.width,
                        ),
                        dtype=torch.int32,
                        device=self.device,
                        requires_grad=False,
                    )
                    self.global_tensor_dict["segmentation_pixels"] = self.segmentation_tensor
                    logger.critical(
                        f"Segmentation pixels shape: {self.global_tensor_dict['segmentation_pixels'].shape}"
                    )
                logger.critical(
                    f"Depth range pixels shape: {self.global_tensor_dict['depth_range_pixels'].shape}"
                )

                self.camera_sensor.init_tensors(global_tensor_dict=self.global_tensor_dict)
        else:
            # assert that only one of camera or lidar is used at once
            assert not (
                self.enable_camera and self.enable_lidar
            ), "Do not use both camera and lidar sensors together for now."

            self.warp_sensor_config = None
            if self.enable_camera:
                self.warp_sensor_config = self.cfg.sensor_config.camera_config
                self.warp_sensor_class = WarpSensor
            elif self.enable_lidar:
                self.warp_sensor_config = self.cfg.sensor_config.lidar_config
                self.warp_sensor_class = WarpSensor

            if self.warp_sensor_config is not None:
                logger.debug("Initializing warp sensor")
                # prepare the tensors for simulation before preparing the tensors for the sensors
                image_tensor_dims = 3 * (self.warp_sensor_config.return_pointcloud == True)
                if self.global_tensor_dict["CONST_WARP_MESH_ID_LIST"] is None:
                    logger.critical(
                        "Warp camera is enabled but there is nothing in the environment. No rendering will take place and the camera tensor will not be populated."
                    )
                else:
                    if image_tensor_dims == 0:
                        self.image_tensor = torch.zeros(
                            (
                                self.cfg.env.num_envs,
                                self.warp_sensor_config.num_sensors,
                                self.warp_sensor_config.height,
                                self.warp_sensor_config.width,
                            ),
                            device=self.device,
                            requires_grad=False,
                        )
                    else:
                        self.image_tensor = torch.zeros(
                            (
                                self.cfg.num_envs,
                                self.warp_sensor_config.num_sensors,
                                self.warp_sensor_config.height,
                                self.warp_sensor_config.width,
                                image_tensor_dims,
                            ),
                            device=self.device,
                            requires_grad=False,
                        )
                    self.global_tensor_dict["depth_range_pixels"] = self.image_tensor

                    if self.warp_sensor_config.segmentation_camera:
                        self.segmentation_tensor = torch.zeros(
                            (
                                self.cfg.env.num_envs,
                                self.warp_sensor_config.num_sensors,
                                self.warp_sensor_config.height,
                                self.warp_sensor_config.width,
                            ),
                            dtype=torch.int32,
                            device=self.device,
                            requires_grad=False,
                        )
                        self.global_tensor_dict["segmentation_pixels"] = self.segmentation_tensor
                    self.warp_sensor = self.warp_sensor_class(
                        self.warp_sensor_config,
                        self.cfg.env.num_envs,
                        self.global_tensor_dict["CONST_WARP_MESH_ID_LIST"],
                        self.device,
                    )
                    self.warp_sensor.init_tensors(global_tensor_dict=self.global_tensor_dict)
                    logger.debug("[DONE] Initializing warp sensor")
                    logger.debug("Capturing warp sensor")
                    self.warp_sensor.update()
                    logger.debug("[DONE] Capturing warp sensor")

        if self.cfg.sensor_config.enable_imu:
            logger.debug("Initializing IMU sensor")
            # acquire force tensors for each of the assets
            self.force_sensor_tensor = gymtorch.wrap_tensor(
                self.gym.acquire_force_sensor_tensor(self.sim)
            )
            self.global_tensor_dict["force_sensor_tensor"] = self.force_sensor_tensor

            self.imu_sensor = IMUSensor(
                self.cfg.sensor_config.imu_config, self.cfg.env.num_envs, self.device
            )
            self.imu_sensor.init_tensors(global_tensor_dict=self.global_tensor_dict)
            logger.debug("[DONE] Initializing IMU sensor")

        elif self.use_warp == False and self.camera_sensor is not None:
            self.has_IGE_sensors = True
        return


    def post_physics_step(self):
        # have this sensor here rather than at capture_sensors
        # as this will still update the sensor without the user forgetting to call render()
        if self.imu_sensor is not None:
            self.imu_sensor.update()

    def capture_sensors(self):
        if self.warp_sensor is not None:
            self.warp_sensor.update()
        if self.camera_sensor is not None:
            self.camera_sensor.update()
