#!/usr/bin/env python3
"""
ACT分布式推理客户端
在低配置电脑上运行仿真，通过网络连接服务器进行ACT推理
"""

import sys
sys.path.append('/home/jason/ws/catkin_ws/src/openRes/demonstration/act1/act')

import asyncio
import websockets
import json
import logging
import cv2
import numpy as np
import base64
from constants import SIM_TASK_CONFIGS
from sim_env import make_sim_env, BOX_POSE
import time
import argparse
from dm_env import StepType
from datetime import datetime
from collections import deque

# 导入仿真环境 - 使用当前目录下的ACT模块
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from sim_env import make_sim_env
from constants import SIM_TASK_CONFIGS

class SimulationClient:
    def __init__(self, server_url, task_name='sim_transfer_cube_scripted', 
                 max_timesteps=400, render=True):
        self.server_url = server_url
        self.task_name = task_name
        self.max_timesteps = max_timesteps
        self.render = render
        
        # 动作缓冲区
        self.action_buffer = deque(maxlen=200)  # 恢复原版大小
        self.action_request_threshold = 10    # 恢复原版阈值
        
        # 序列ID管理
        self.sequence_id = 0
        self.pending_requests = {}
        
        # 统计信息
        self.episode_count = 0
        self.total_steps = 0
        self.network_latencies = []
        self.inference_times = []
        
        # 设置日志
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # 创建仿真环境
        self.env = None
        self.init_environment()
        
        # 标志以确保viewer只启动一次
        self.viewer_launched = False
    
    def init_environment(self):
        """初始化仿真环境"""
        try:
            # 设置默认BOX_POSE以避免None错误
            BOX_POSE[0] = [0.2, 0.5, 0.05, 1, 0, 0, 0]  # [x, y, z, qw, qx, qy, qz]
            
            # 获取任务配置
            task_config = SIM_TASK_CONFIGS[self.task_name]
            dataset_dir = task_config['dataset_dir']
            num_episodes = task_config['num_episodes']
            episode_len = task_config['episode_len']
            camera_names = task_config['camera_names']
            
            # 创建环境
            self.env = make_sim_env(self.task_name)
            
            # 记录相关信息
            self.logger.info(f"Initialized task: {self.task_name}")
            self.logger.info(f"Dataset dir: {dataset_dir}")
            self.logger.info(f"Episode length: {episode_len}")
            self.logger.info(f"Camera names: {camera_names}")
            
            # 存储配置
            self.dataset_dir = dataset_dir
            self.episode_len = episode_len
            self.camera_names = camera_names
            
            self.logger.info("Environment initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize environment: {e}")
            raise
    
    def encode_image(self, image, quality=80):
        """编码图像为base64"""
        try:
            # 调整图像大小以减少传输量
            if image.shape[:2] != (240, 320):
                image = cv2.resize(image, (320, 240))
            
            # 转换BGR到RGB (如果需要)
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # JPEG编码
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            _, buffer = cv2.imencode('.jpg', image, encode_param)
            
            # Base64编码
            encoded = base64.b64encode(buffer).decode('utf-8')
            return encoded
            
        except Exception as e:
            self.logger.error(f"Failed to encode image: {e}")
            return None
    
    def prepare_request(self, obs):
        """准备推理请求数据"""
        try:
            # 打印观测结构以调试
            self.logger.debug(f"Observation type: {type(obs)}")
            self.logger.debug(f"Observation keys: {obs.observation.keys() if hasattr(obs, 'observation') else obs.keys()}")
            
            # 处理不同的观察数据格式
            if hasattr(obs, 'observation'):
                # TimeStep对象，通过observation属性访问
                observation = obs.observation
            else:
                # 直接返回观察字典
                observation = obs
            
            # 提取机器人状态
            qpos = observation['qpos']  # 关节位置
            qvel = observation['qvel']  # 关节速度
            
            # 提取图像
            camera_image = None
            if 'images' in observation and self.camera_names:
                camera_name = self.camera_names[0]
                if camera_name in observation['images']:
                    image = observation['images'][camera_name]
                    camera_image = self.encode_image(image)
            
            # 构造请求
            self.sequence_id += 1
            request = {
                'timestamp': time.time(),
                'sequence_id': self.sequence_id,
                'robot_state': {
                    'joint_positions': qpos.tolist(),
                    'joint_velocities': qvel.tolist()
                },
                'camera_image': camera_image
            }
            
            return request
            
        except Exception as e:
            self.logger.error(f"Failed to prepare request: {e}")
            return None
    
    async def request_actions(self, websocket, obs):
        """请求新的动作序列"""
        try:
            request = self.prepare_request(obs)
            if request is None:
                return False
            
            # 记录请求时间
            request_time = time.time()
            self.pending_requests[request['sequence_id']] = request_time
            
            # 发送请求
            await websocket.send(json.dumps(request))
            self.logger.info(f"✅ 发送动作请求 #{request['sequence_id']} - 关节位置: {len(request['robot_state']['joint_positions'])}个, 图像: {'有' if request['camera_image'] else '无'}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to request actions: {e}")
            return False
    
    def process_response(self, response_data):
        """处理服务器响应"""
        try:
            sequence_id = response_data['sequence_id']
            actions = np.array(response_data['actions'])
            
            # 计算网络延迟
            if sequence_id in self.pending_requests:
                request_time = self.pending_requests[sequence_id]
                latency = time.time() - request_time
                self.network_latencies.append(latency)
                del self.pending_requests[sequence_id]
                
                self.logger.info(f"📥 接收到服务器响应 #{sequence_id}: "
                               f"{len(actions)}个动作步骤, 延迟: {latency:.3f}s")
            
            # 添加动作到缓冲区
            for i, action in enumerate(actions):
                self.action_buffer.append(action)
                
            self.logger.info(f"✅ 已添加 {len(actions)} 个动作到缓冲区，缓冲区总数: {len(self.action_buffer)}")
            
            # 记录推理时间
            if 'inference_time' in response_data:
                self.inference_times.append(response_data['inference_time'])
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to process response: {e}")
            return False
    
    async def run_episode(self, websocket):
        """运行一个episode"""
        self.logger.info(f"Starting episode #{self.episode_count + 1}")
        
        try:
            # 重置环境
            obs = self.env.reset()
            episode_reward = 0
            step_count = 0
            
            # 初始动作请求
            await self.request_actions(websocket, obs)
            
            # 等待初始动作
            while len(self.action_buffer) == 0:
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                    response_data = json.loads(response)
                    
                    if 'error' in response_data:
                        self.logger.error(f"Server error: {response_data['error']}")
                        return False
                    
                    self.process_response(response_data)
                    
                except asyncio.TimeoutError:
                    self.logger.error("Timeout waiting for initial actions")
                    return False
            
            # 运行episode
            while step_count < self.max_timesteps:
                # 检查是否需要请求新动作
                if len(self.action_buffer) < self.action_request_threshold:
                    await self.request_actions(websocket, obs)
                
                # 处理服务器响应 (非阻塞)
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=0.001)
                    response_data = json.loads(response)
                    
                    if 'error' in response_data:
                        self.logger.warning(f"Server error: {response_data['error']}")
                    else:
                        self.process_response(response_data)
                        
                except asyncio.TimeoutError:
                    pass  # 没有新消息，继续执行
                
                # 执行动作
                if len(self.action_buffer) > 0:
                    action = self.action_buffer.popleft()
                    self.logger.info(f"🚀 执行动作步骤 {step_count+1}: 动作维度={len(action)}, 缓冲区剩余={len(self.action_buffer)}")
                    
                    # 详细分析动作值
                    action_analysis = f"动作值分析: min={np.min(action):.3f}, max={np.max(action):.3f}, mean={np.mean(action):.3f}"
                    self.logger.info(action_analysis)
                    
                    # 分析关节动作（前14个是双臂关节，后2个是夹爪）
                    if len(action) >= 16:
                        left_arm = action[:7]   # 左臂7个关节
                        right_arm = action[7:14] # 右臂7个关节  
                        grippers = action[14:16] # 双手夹爪
                        self.logger.info(f"左臂动作: {left_arm}")
                        self.logger.info(f"右臂动作: {right_arm}")
                        self.logger.info(f"夹爪动作: {grippers}")
                    
                    timestep = self.env.step(action)
                    obs = timestep.observation
                    reward = timestep.reward 
                    done = timestep.last()
                    
                    episode_reward += reward
                    step_count += 1
                    self.total_steps += 1
                    
                    # 显示当前奖励状态和关节位置
                    current_qpos = obs['qpos'][:16] if 'qpos' in obs else None
                    if current_qpos is not None:
                        self.logger.info(f"当前关节位置: 左臂={current_qpos[:7]}, 右臂={current_qpos[7:14]}")
                    
                    if reward > 0:
                        reward_status = {1: "接触右爪", 2: "右爪抓起", 3: "尝试转移", 4: "成功转移！"}
                        self.logger.info(f"📈 当前奖励: {reward} ({reward_status.get(reward, '未知状态')})")
                    
                    # 检查动作是否异常（全零或极值）
                    if np.allclose(action, 0, atol=1e-6):
                        self.logger.warning("⚠️  警告: 接收到全零动作，可能是模型推理问题")
                    elif np.any(np.abs(action) > 10):
                        self.logger.warning("⚠️  警告: 动作值过大，可能导致不稳定行为")
                    
                    # 渲染 - 重新启用viewer（已修复angle相机问题）
                    if self.render and not self.viewer_launched:
                        try:
                            from dm_control import viewer
                            self.logger.info("🎮 启动MuJoCo仿真窗口...")
                            viewer.launch(self.env)
                            self.viewer_launched = True
                            self.logger.info("✅ MuJoCo仿真窗口已启动")
                        except Exception as e:
                            self.logger.error(f"Failed to launch viewer: {e}")
                            # 继续运行，不因为viewer失败而停止仿真
                            self.render = False
                    
                    # 检查episode结束 - 使用原版ACT逻辑
                    if done or reward == 4:  # 原版逻辑：environment done或达到最高奖励4（成功转移）
                        success_msg = "成功完成抓取任务！" if reward == 4 else "Episode自然结束"
                        self.logger.info(f"Episode结束: {success_msg}, reward={episode_reward:.3f}, 最终奖励={reward}, steps={step_count}")
                        break
                        
                else:
                    # 没有可用动作，等待服务器响应
                    self.logger.warning("Action buffer empty, waiting for server response...")
                    await asyncio.sleep(0.01)  # 恢复原版较短等待时间
            
            self.episode_count += 1
            return True
            
        except Exception as e:
            import traceback
            self.logger.error(f"Episode error: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return False
    
    async def run_simulation(self):
        """运行仿真主循环"""
        self.logger.info(f"Connecting to server: {self.server_url}")
        
        try:
            async with websockets.connect(self.server_url) as websocket:
                self.logger.info("Connected to inference server")
                
                # 运行多个episodes
                success_count = 0
                max_episodes = 5  # 可配置
                
                for episode in range(max_episodes):
                    success = await self.run_episode(websocket)
                    if success:
                        success_count += 1
                    
                    # 清理缓冲区
                    self.action_buffer.clear()
                    self.pending_requests.clear()
                    
                    # 间隔
                    if episode < max_episodes - 1:
                        await asyncio.sleep(1.0)
                
                # 打印统计信息
                self.print_statistics(success_count, max_episodes)
                
        except Exception as e:
            import traceback
            self.logger.error(f"Connection error: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return False
        
        return True
    
    def print_statistics(self, success_count, total_episodes):
        """打印统计信息"""
        print("\n" + "="*50)
        print("📊 仿真统计信息")
        print("="*50)
        print(f"成功Episodes: {success_count}/{total_episodes}")
        print(f"总步数: {self.total_steps}")
        
        if self.network_latencies:
            avg_latency = np.mean(self.network_latencies)
            max_latency = np.max(self.network_latencies)
            print(f"网络延迟: 平均 {avg_latency:.3f}s, 最大 {max_latency:.3f}s")
        
        if self.inference_times:
            avg_inference = np.mean(self.inference_times)
            print(f"推理时间: 平均 {avg_inference:.3f}s")
        
        print("="*50 + "\n")

def main():
    parser = argparse.ArgumentParser(description='ACT分布式推理客户端')
    parser.add_argument('--server_url', type=str, required=True,
                       help='服务器WebSocket URL (如: ws://10.16.49.124:8765)')
    parser.add_argument('--task_name', type=str, 
                       default='sim_transfer_cube_scripted',
                       help='仿真任务名称')
    parser.add_argument('--max_timesteps', type=int, default=400,
                       help='每个episode最大步数')
    parser.add_argument('--no_render', action='store_true',
                       help='禁用渲染 (加快仿真速度)')
    
    args = parser.parse_args()
    
    # 创建客户端
    client = SimulationClient(
        server_url=args.server_url,
        task_name=args.task_name,
        max_timesteps=args.max_timesteps,
        render=not args.no_render
    )
    
    # 运行仿真
    try:
        asyncio.run(client.run_simulation())
    except KeyboardInterrupt:
        print("\n🛑 仿真已停止")
    except Exception as e:
        print(f"❌ 客户端错误: {e}")

if __name__ == "__main__":
    main()
