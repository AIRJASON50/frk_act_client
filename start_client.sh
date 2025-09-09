#!/bin/bash
# ACT分布式推理客户端启动脚本

# 设置conda环境python路径
CONDA_ENV_PYTHON="~/miniconda/envs/aloha_client/bin/python"
PYTHON_CMD=$(eval echo $CONDA_ENV_PYTHON)

# 服务器配置
SERVER_IP="10.16.49.124"
SERVER_PORT="8765"
SERVER_URL="ws://${SERVER_IP}:${SERVER_PORT}"

echo "🚀 启动ACT分布式推理客户端"
echo "📡 连接服务器: ${SERVER_URL}"
echo "🔧 使用Python: $PYTHON_CMD"
echo "=================================="

# 检查网络连通性
echo "🔍 检查网络连通性..."
if ping -c 1 ${SERVER_IP} > /dev/null; then
    echo "✅ 网络连接正常"
else
    echo "❌ 无法连接到服务器 ${SERVER_IP}"
    exit 1
fi

# 检查依赖
echo "🔍 检查环境依赖..."
$PYTHON_CMD -c "import websockets, numpy, cv2, mujoco, dm_control; print('所有依赖包已安装')" 
if [ $? -eq 0 ]; then
    echo "✅ 依赖包检查通过"
else
    echo "❌ 缺少必要依赖包，请确保aloha_client环境正确安装"
    exit 1
fi

# 启动客户端
echo "🎮 启动仿真客户端..."
$PYTHON_CMD simulation_client.py --server_url ${SERVER_URL} --task_name sim_transfer_cube_scripted

echo "🛑 客户端已退出"
