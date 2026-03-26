import os
unitree_rl_gym_path = os.path.abspath(__file__ + "../../../../")
import sys
sys.path.append(unitree_rl_gym_path)
from legged_gym import LEGGED_GYM_ROOT_DIR

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch
from isaacgym.torch_utils import quat_apply, quat_rotate_inverse

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import time
import atexit



def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 1)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_leg_mass = False
    env_cfg.domain_rand.randomize_gripper_mass = False
    env_cfg.domain_rand.randomize_motor = False
    env_cfg.domain_rand.randomize_base_com = False
    env_cfg.env.test = True

    if args.flat_terrain:
        env_cfg.terrain.height = [0.0, 0.0]
    
    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    # 开启遥控模式，禁止环境内部自动更新commands，使外部注入的画圆命令生效
    env.cfg.env.teleop_mode = True

    obs = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    
    
    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:

        import copy
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        os.makedirs(path, exist_ok=True)
        adaptation_module_path = os.path.join(path, 'adaptation_module.pt')
        model = copy.deepcopy(ppo_runner.alg.actor_critic.adaptation_encoder_module).to('cpu')
        traced_script_module = torch.jit.script(model)
        traced_script_module.save(adaptation_module_path)
        print('Exported policy as jit script to: ', adaptation_module_path)

        adaptation_decoder_path = os.path.join(path, 'adaptation_decoder.pt')
        model = copy.deepcopy(ppo_runner.alg.actor_critic.adaptation_decoder_module).to('cpu')
        traced_script_module = torch.jit.script(model)
        traced_script_module.save(adaptation_decoder_path)
        print('Exported policy as jit script to: ', adaptation_decoder_path)

        actor_body_path = os.path.join(path, 'actor_body.pt')
        model = copy.deepcopy(ppo_runner.alg.actor_critic.actor_body).to('cpu')
        traced_script_module = torch.jit.script(model)
        traced_script_module.save(actor_body_path)
        print('Exported policy as jit script to: ', actor_body_path)

    if VISUAL_PRED:
        fig_ee_force = plt.figure()
        ax_ee_force = fig_ee_force.add_subplot(111, projection='3d')
        vector1_ee_force = np.array([0, 0, 0])
        vector2_ee_force = np.array([0, 0, 0])
        ax_ee_force.set_xlim([-0.7, 0.7])
        ax_ee_force.set_ylim([-0.7, 0.7])
        ax_ee_force.set_zlim([-0.7, 0.7])
        ax_ee_force.set_xlabel('X axis')
        ax_ee_force.set_ylabel('Y axis')
        ax_ee_force.set_zlabel('Z axis')
        line1_ee_force, = ax_ee_force.plot([0, vector1_ee_force[0]], [0, vector1_ee_force[1]], [0, vector1_ee_force[2]], marker='o', label='ee_force_pred')
        line2_ee_force, = ax_ee_force.plot([0, vector2_ee_force[0]], [0, vector2_ee_force[1]], [0, vector2_ee_force[2]], marker='o', label='ee_force_gt')
        ax_ee_force.legend()

        fig_base_force = plt.figure()
        ax_base_force = fig_base_force.add_subplot(111, projection='3d')
        vector1_base_force = np.array([0, 0, 0])
        vector2_base_force = np.array([0, 0, 0])
        ax_base_force.set_xlim([-0.7, 0.7])
        ax_base_force.set_ylim([-0.7, 0.7])
        ax_base_force.set_zlim([-0.7, 0.7])
        ax_base_force.set_xlabel('X axis')
        ax_base_force.set_ylabel('Y axis')
        ax_base_force.set_zlabel('Z axis')
        line1_base_force, = ax_base_force.plot([0, vector1_base_force[0]], [0, vector1_base_force[1]], [0, vector1_base_force[2]], marker='o', label='base_force_pred')
        line2_base_force, = ax_base_force.plot([0, vector2_base_force[0]], [0, vector2_base_force[1]], [0, vector2_base_force[2]], marker='o', label='base_force_gt')
        ax_base_force.legend()

        fig_linvel = plt.figure()
        ax_linvel = fig_linvel.add_subplot(111, projection='3d')
        vector1_linvel = np.array([0, 0, 0])
        vector2_linvel = np.array([0, 0, 0])
        ax_linvel.set_xlim([-2, 2])
        ax_linvel.set_ylim([-2, 2])
        ax_linvel.set_zlim([-2, 2])
        ax_linvel.set_xlabel('X axis')
        ax_linvel.set_ylabel('Y axis')
        ax_linvel.set_zlabel('Z axis')
        line1_linvel, = ax_linvel.plot([0, vector1_linvel[0]], [0, vector1_linvel[1]], [0, vector1_linvel[2]], marker='o', label='linvel_pred')
        line2_linvel, = ax_linvel.plot([0, vector2_linvel[0]], [0, vector2_linvel[1]], [0, vector2_linvel[2]], marker='o', label='linvel_gt')
        ax_linvel.legend()

        fig_eepos = plt.figure()
        ax_eepos = fig_eepos.add_subplot(111, projection='3d')
        vector1_eepos = np.array([0, 0, 0])
        vector2_eepos = np.array([0, 0, 0])
        ax_eepos.set_xlim([-2, 2])
        ax_eepos.set_ylim([-2, 2])
        ax_eepos.set_zlim([-2, 2])
        ax_eepos.set_xlabel('X axis')
        ax_eepos.set_ylabel('Y axis')
        ax_eepos.set_zlabel('Z axis')
        line1_eepos, = ax_eepos.plot([0, vector1_eepos[0]], [0, vector1_eepos[1]], [0, vector1_eepos[2]], marker='o', label='eepos_pred')
        line2_eepos, = ax_eepos.plot([0, vector2_eepos[0]], [0, vector2_eepos[1]], [0, vector2_eepos[2]], marker='o', label='eepos_gt')
        ax_eepos.legend()
    

    env.play = True
    policy_info = {}
    DRAW_CIRCLE = True 
    
    # 数据记录容器
    log_data = []

    # [New] Capture nominal base height for Body-Relative correction
    nominal_base_height = None

    def save_log_data():
        if len(log_data) > 0:
            print("\nSaving tracking data on exit...")
            np_log_data = np.array(log_data)
            np.save("circle_tracking_data.npy", np_log_data)
            print(f"Data saved to circle_tracking_data.npy with shape {np_log_data.shape}")
    
    atexit.register(save_log_data)

    for i in range(100*int(env.max_episode_length)):
        actions = policy(obs, policy_info)
        
        # --- 新增画圆逻辑 ---
        if DRAW_CIRCLE:
            # 1. 机器人基座静止
            env.commands[:, 0] = 0.
            env.commands[:, 1] = 0.
            env.commands[:, 2] = 0.
            
            # 2. 定义圆周运动参数 (局部球坐标)
            target_radius = 0.65       
            target_pitch = 0.0         
            target_yaw = np.sin(i * 0.02) * 0.8
            
            # 3. 注入命令 
            env.commands[:, 3] = target_radius
            env.commands[:, 4] = target_pitch
            env.commands[:, 5] = target_yaw

            # [新增] 同步更新环境内部的目标变量
            env.curr_ee_goal_sphere[:, 0] = target_radius
            env.curr_ee_goal_sphere[:, 1] = target_pitch
            env.curr_ee_goal_sphere[:, 2] = target_yaw
            
            # 4. 力控指令清零
            env.commands[:, 9:12] = 0.0

            # --- 数据记录 (Data Logging for Local Frame Tracking) ---
            if nominal_base_height is None:
                nominal_base_height = env.root_states[0, 2].item()
                print(f"Captured Nominal Base Height: {nominal_base_height:.4f} m")

            # 目标位置 (已经在 Local Spherical 转换为 Local Cartesian 了)
            # 注意: 这里的 Local Frame 是相对于 "Sphere Center" 的，
            # 在 envs code 里: center = base_pos_xy_ground + rotated_offset
            # 我们简化计算，直接记录 
            # 1. 目标指令转换出的 Local Cartesian 指令 (相对于 Sphere Center)
            # 2. 实际末端位置转换回 Local Cartesian (相对于 Sphere Center)
            
            # Target (Local Cartesian derived from commands)
            t_x = target_radius * np.cos(target_pitch) * np.cos(target_yaw)
            t_y = target_radius * np.cos(target_pitch) * np.sin(target_yaw)
            t_z = target_radius * np.sin(target_pitch)
            
            # Actual (Convert World Position back to Local Cartesian relative to Sphere Center)
            # 参考 env._resample_ee_goal 中的逻辑
            # center calculation:
            # center = [base_x, base_y, 0] + quat_apply(base_yaw_quat, offset)
            # 我们需要获取 base_yaw_quat
            
            # env.root_states: [pos(3), quat(4), lin_vel(3), ang_vel(3)]
            base_quat = env.root_states[:, 3:7]
            
            # 提取 yaw rotation (project gravity vector logic equivalent or just extraction)
            # 简单起见，我们直接复用 env 中的 self.base_yaw_quat 如果它在 play loop 中更新了
            # env.post_physics_step() -> check_termination -> compute_observations
            # 在 compute_observations 之前通常会更新 base_quat 等
            # 但 env.base_yaw_quat 是在 compute_observations 里计算的吗？
            # 让我们手动计算一下以防万一
            from isaacgym.torch_utils import get_euler_xyz, quat_from_euler_xyz
            r, p, y = get_euler_xyz(base_quat)
            # base_yaw_quat just has the yaw rotation
            base_yaw_quat = quat_from_euler_xyz(torch.zeros_like(r), torch.zeros_like(p), y)
            
            # Recompute Sphere Center (Reference Point)
            # offset 应该是 env.ee_goal_center_offset
            # center = [base_x, base_y, 0] 
            # 注意: env code 里 center Z 是 0 (floor projection) + z_invariant_offset
            root_xy_ground = env.root_states[:, :3].clone()
            root_xy_ground[:, 2] = 0.0
            
            # [Fix] Need to ensure we use the same offset logic as during training
            # In update_ee_goal: 
            # self.ee_goal_center_offset = torch.tensor([x, y, z])
            # center = root_states + rotated_offset
            
            # Use env.ee_goal_center_offset if available, else fallback
            if hasattr(env, 'ee_goal_center_offset'):
                offset = env.ee_goal_center_offset.clone()
            else:
                # Fallbck: Assume mostly forward shoulder. 
                # Ideally you should check cfg, but let's try reading from env if possible or default to prev logic
                # For now let's hope env has it initialized
                 offset = torch.zeros((env.num_envs, 3), device=env.device)
            
            # [Correction] Apply Base Height Drift Compensation
            current_base_z = env.root_states[:, 2]
            z_drift = current_base_z - nominal_base_height
            
            sphere_center_ground = root_xy_ground + quat_apply(base_yaw_quat, offset)
            
            # Create a "Body-Attached" Sphere Center
            sphere_center_body = sphere_center_ground.clone()
            sphere_center_body[:, 2] += z_drift
            
            # Actual EE Position in World Frame
            ee_pos_world = env.ee_pos  # [num_envs, 3]
            
            # Compute actual position relative to BODY CORRECTED sphere center
            rel_pos_world = ee_pos_world - sphere_center_body
            
            # Rotate back to align with Local Frame (Heading)
            # valid_ee_local_cart = quat_rotate_inverse(base_yaw_quat, rel_pos_world)
            actual_local_pos = quat_rotate_inverse(base_yaw_quat, rel_pos_world)
            
            # Now we have Target Local and Actual Local
            # Target is: [t_x, t_y, t_z]
            # Actual is: actual_local_pos[0] (vector of 3)
            
            target_vec_local_np = np.array([t_x, t_y, t_z])
            actual_vec_local_np = actual_local_pos[0].detach().cpu().numpy()
            
            frame_data = np.concatenate([target_vec_local_np, actual_vec_local_np])
            log_data.append(frame_data)
            # -------------------------------
        # -------------------

        # breakpoint()
        if FIX_COMMAND and not DRAW_CIRCLE:
            env.commands[:, 0] = 0.    # 1.0
            env.commands[:, 1] = 0.
            env.commands[:, 2] = 0.0
            env.commands[:, 3] = 0.
            # env.gait_indices[:] = 0.
        obs, rews, dones, infos = env.step(actions.detach())
        if VISUAL_PRED:
            ee_force_pred = policy_info["latents"][0, 6:9]
            vector1_ee_force = ee_force_pred
            vector2_ee_force = env.forces_local[0, env.gripper_idx].detach().cpu().numpy() * env.obs_scales.ee_force
            print("ee_force_pred:", ee_force_pred*100)
            print("ee_force_ext:", vector2_ee_force*100)
            line1_ee_force.set_data([0, vector1_ee_force[0]], [0, vector1_ee_force[1]])
            line1_ee_force.set_3d_properties([0, vector1_ee_force[2]])

            line2_ee_force.set_data([0, vector2_ee_force[0]], [0, vector2_ee_force[1]])
            line2_ee_force.set_3d_properties([0, vector2_ee_force[2]])

            base_force_pred = policy_info["latents"][0, 9:12]
            vector1_base_force = base_force_pred
            vector2_base_force = env.forces_local[0,env.robot_base_idx].detach().cpu().numpy() * env.obs_scales.base_force
            print("ee_base_pred:", base_force_pred*100)
            print("ee_base_ext:", vector2_base_force*100)
            # line1_base_force.set_data([0, vector1_base_force[0]], [0, vector1_base_force[1]])
            # line1_base_force.set_3d_properties([0, vector1_base_force[2]])

            # line2_base_force.set_data([0, vector2_base_force[0]], [0, vector2_base_force[1]])
            # line2_base_force.set_3d_properties([0, vector2_base_force[2]])
            

            # linvel_pred = policy_info["latents"][0, 0:3]
            # vector1_linvel = linvel_pred
            # vector2_linvel = env.base_lin_vel[0].detach().cpu().numpy() * env.obs_scales.lin_vel
            # line1_linvel.set_data([0, vector1_linvel[0]], [0, vector1_linvel[1]])
            # line1_linvel.set_3d_properties([0, vector1_linvel[2]])

            # line2_linvel.set_data([0, vector2_linvel[0]], [0, vector2_linvel[1]])
            # line2_linvel.set_3d_properties([0, vector2_linvel[2]])

            # eepos_pred = policy_info["latents"][0, 3:6]
            # vector1_eepos = eepos_pred
            
            # vector2_eepos = np.array([env.ee_pos_sphe_arm[0,0].detach().cpu().numpy() * env.obs_scales.ee_sphe_radius_cmd,
            #                                 env.ee_pos_sphe_arm[0,1].detach().cpu().numpy() * env.obs_scales.ee_sphe_pitch_cmd,
            #                                 env.ee_pos_sphe_arm[0,2].detach().cpu().numpy() * env.obs_scales.ee_sphe_yaw_cmd])
            # line1_eepos.set_data([0, vector1_eepos[0]], [0, vector1_eepos[1]])
            # line1_eepos.set_3d_properties([0, vector1_eepos[2]])
            # line2_eepos.set_data([0, vector2_eepos[0]], [0, vector2_eepos[1]])
            # line2_eepos.set_3d_properties([0, vector2_eepos[2]])
            plt.draw()
            plt.pause(0.001)
    
    # 循环结束，atexit 会自动保存数据
    pass

if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    FIX_COMMAND = False
    VISUAL_PRED = True
    args = get_args()
    if args.task == "go2":
        args.task = "b2z1_pos_force"
    
    try:
        play(args)
    except KeyboardInterrupt:
        print("Simulation stopped.")

