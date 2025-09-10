#!/bin/bash

# ACT分布式推理客户端启动脚本
# 自动检测Python环境
if command -v conda &> /dev/null; then
    # 尝试使用conda环境
    if conda env list | grep -q "aloha_client"; then
        PYTHON_CMD="conda run -n aloha_client python"
        echo "✅ 使用conda环境: aloha_client"
    else
        PYTHON_CMD=$(which python3)
        echo "⚠️  未找到aloha_client环境，使用系统Python: $PYTHON_CMD"
    fi
else
    PYTHON_CMD=$(which python3)
    echo "⚠️  未安装conda，使用系统Python: $PYTHON_CMD"
fi

# 服务器配置
SERVER_IP="10.16.49.124"
SERVER_PORT="8765"
SERVER_URL="ws://${SERVER_IP}:${SERVER_PORT}"

echo "🚀 启动ACT分布式推理客户端"
echo "📡 连接服务器: ${SERVER_URL}"
echo "🐍 使用Python: $PYTHON_CMD"
echo "=================================="

# 检查Python可用性
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3未安装"
    exit 1
fi

# 检查网络连通性
echo "🔍 检查服务器连通性..."
if ping -c 1 -W 3 "$SERVER_IP" &> /dev/null; then
    echo "✅ 服务器可达: $SERVER_IP"
else
    echo "⚠️  警告: 无法ping通服务器 $SERVER_IP，但仍尝试连接"
fi

# 启动客户端
echo "🎯 启动仿真客户端..."
$PYTHON_CMD simulation_client.py \
    --server_url "$SERVER_URL" \
    --task_name "sim_transfer_cube_scripted" \
    --max_timesteps 400

echo "🛑 客户端已退出"
