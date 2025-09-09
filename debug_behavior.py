#!/usr/bin/env python3
"""
调试机械臂行为的专用脚本
记录详细的动作序列、关节状态和奖励变化
"""

import sys
sys.path.append('/home/jason/ws/catkin_ws/src/openRes/demonstration/act1/act')

import asyncio
import websockets
import json
import logging
import numpy as np
import time
from simulation_client import SimulationClient

class BehaviorAnalyzer(SimulationClient):
    def __init__(self, server_url, task_name):
        super().__init__(server_url, task_name, render=False)
        self.action_log = []
        self.reward_log = []
        self.position_log = []
        
    async def analyze_episode(self):
        """分析一个episode的完整行为"""
        print("\n" + "="*60)
        print("🔍 开始行为分析")
        print("="*60)
        
        try:
            async with websockets.connect(self.server_url) as websocket:
                print(f"✅ 连接到推理服务器: {self.server_url}")
                
                # 重置环境
                timestep = self.env.reset()
                obs = timestep.observation
                print(f"🔄 环境已重置")
                
                # 调试观测结构
                print(f"🔍 观测数据类型: {type(obs)}")
                print(f"🔍 观测数据键: {list(obs.keys()) if isinstance(obs, dict) else 'Not a dict'}")
                
                # 初始位置
                initial_qpos = obs['qpos'][:16] if 'qpos' in obs else None
                if initial_qpos is not None:
                    print(f"🏠 初始关节位置:")
                    print(f"   左臂: {initial_qpos[:7]}")
                    print(f"   右臂: {initial_qpos[7:14]}")
                    print(f"   夹爪: {initial_qpos[14:16]}")
                else:
                    print(f"⚠️  无法获取qpos数据")
                
                step_count = 0
                max_steps = 50  # 只分析前50步
                
                # 初始动作请求
                await self.request_actions(websocket, obs)
                
                # 等待初始动作
                while len(self.action_buffer) == 0:
                    response = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                    response_data = json.loads(response)
                    self.process_response(response_data)
                
                print(f"📦 收到初始动作缓冲区: {len(self.action_buffer)} 个动作")
                
                while step_count < max_steps and len(self.action_buffer) > 0:
                    # 执行动作
                    action = self.action_buffer.popleft()
                    
                    print(f"\n--- 步骤 {step_count + 1} ---")
                    print(f"动作: min={np.min(action):.3f}, max={np.max(action):.3f}")
                    
                    # 分析动作模式
                    if len(action) >= 16:
                        left_arm = action[:7]
                        right_arm = action[7:14]
                        grippers = action[14:16]
                        
                        print(f"左臂变化幅度: {np.std(left_arm):.3f}")
                        print(f"右臂变化幅度: {np.std(right_arm):.3f}")
                        print(f"夹爪动作: {grippers}")
                    
                    # 执行并记录
                    timestep = self.env.step(action)
                    obs = timestep.observation
                    reward = timestep.reward
                    done = timestep.last()
                    
                    # 记录数据
                    self.action_log.append(action.copy())
                    self.reward_log.append(reward)
                    if 'qpos' in obs:
                        self.position_log.append(obs['qpos'][:16].copy())
                    else:
                        print("⚠️  当前步骤无法获取qpos")
                    
                    print(f"奖励: {reward}")
                    if reward > 0:
                        reward_status = {1: "接触右爪", 2: "右爪抓起", 3: "尝试转移", 4: "成功转移！"}
                        print(f"🎯 {reward_status.get(reward, '未知')}")
                    
                    # 检查是否停止变化
                    if step_count > 5:
                        recent_actions = self.action_log[-5:]
                        action_variance = np.var([np.std(a) for a in recent_actions])
                        if action_variance < 1e-6:
                            print("⚠️  检测到动作停止变化！")
                    
                    step_count += 1
                    
                    if done or reward == 4:
                        print(f"✅ Episode结束: reward={reward}")
                        break
                    
                    # 请求更多动作
                    if len(self.action_buffer) < 5:
                        await self.request_actions(websocket, obs)
                        
                        # 非阻塞接收
                        try:
                            response = await asyncio.wait_for(websocket.recv(), timeout=0.1)
                            response_data = json.loads(response)
                            self.process_response(response_data)
                        except asyncio.TimeoutError:
                            pass
                
                # 分析结果
                self.analyze_behavior_pattern()
                
        except Exception as e:
            print(f"❌ 分析过程出错: {e}")
            import traceback
            traceback.print_exc()
    
    def analyze_behavior_pattern(self):
        """分析行为模式"""
        print("\n" + "="*60)
        print("📊 行为模式分析")
        print("="*60)
        
        if not self.action_log:
            print("❌ 没有动作数据可分析")
            return
        
        # 分析动作变化趋势
        action_stds = [np.std(action) for action in self.action_log]
        position_changes = []
        
        for i in range(1, len(self.position_log)):
            change = np.linalg.norm(self.position_log[i] - self.position_log[i-1])
            position_changes.append(change)
        
        print(f"总步数: {len(self.action_log)}")
        print(f"最高奖励: {max(self.reward_log) if self.reward_log else 0}")
        print(f"平均动作幅度: {np.mean(action_stds):.4f}")
        print(f"动作幅度方差: {np.var(action_stds):.6f}")
        
        if position_changes:
            print(f"平均位置变化: {np.mean(position_changes):.4f}")
            print(f"位置变化趋势: {np.std(position_changes):.4f}")
        
        # 检测停滞
        if len(action_stds) > 10:
            recent_std = np.std(action_stds[-10:])
            if recent_std < 1e-4:
                print("🔍 检测到: 动作幅度在最后10步中基本不变 → 可能模型输出重复")
        
        # 检测奖励停滞
        reward_change_points = []
        for i, reward in enumerate(self.reward_log):
            if reward > 0:
                reward_change_points.append(i)
        
        if reward_change_points:
            print(f"🎯 奖励变化点: {reward_change_points}")
        else:
            print("⚠️  整个episode中没有获得任何奖励 → 可能模型训练不足")

async def main():
    analyzer = BehaviorAnalyzer(
        server_url="ws://10.16.49.124:8765",
        task_name="sim_transfer_cube_scripted"
    )
    
    await analyzer.analyze_episode()

if __name__ == "__main__":
    asyncio.run(main())
