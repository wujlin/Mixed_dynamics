## 远程工作站 vLLM 部署与本地调用（Qwen）

整理自先前已验证的流程，便于快速复用。

### 一、远程工作站（Server）端
目标：在工作站 `10.13.12.164` 上用 vLLM 启动 OpenAI 兼容服务。

1. 登录与会话
   - SSH 登录工作站，使用 `tmux` 保持会话：
     ```bash
     tmux new -s qwen_service
     # 若已有会话：tmux attach -t qwen_service
     ```
   - 激活包含 vllm 的环境：
     ```bash
     conda activate your_env_name
     # 若未安装：pip install vllm
     ```

2. 启动 vLLM API 服务（核心命令）
   ```bash
   python -m vllm.entrypoints.openai.api_server \
     --model Qwen/Qwen3-8B \
     --served-model-name Qwen/Qwen3-8B \
     --port 7890 \
     --host 0.0.0.0 \
     --api-key abc123 \
     --trust-remote-code
   ```
   - 如需其他模型，可改 `--model`（如 `Qwen/Qwen2-7B-Instruct`）。
   - `--host 0.0.0.0` 允许外部访问；`--api-key` 设置简易密钥。

3. 防火墙（如连接失败）
   ```bash
   sudo ufw allow 7890/tcp
   ```

### 二、本地客户端调用（Windows/Python，经 SOCKS5 代理）
目标：通过 Docker/EasyConnect 隧道访问远程 vLLM。

1. 配置代理与环境变量
   ```python
   import os
   from openai import OpenAI

   # 确保走隧道
   os.environ.pop("no_proxy", None)
   os.environ.pop("NO_PROXY", None)
   proxy_url = "socks5://127.0.0.1:1080"
   os.environ["http_proxy"] = proxy_url
   os.environ["https_proxy"] = proxy_url
   os.environ["ALL_PROXY"] = proxy_url
   ```

2. 初始化客户端并测试
   ```python
   client = OpenAI(
       base_url="http://10.13.12.164:7890/v1",
       api_key="abc123",
   )
   resp = client.chat.completions.create(
       model="Qwen/Qwen3-8B",
       messages=[{"role": "user", "content": "你好，能帮我写一个简短的自我介绍吗？"}]
   )
   print(resp.choices[0].message.content)
   ```

以上即可完成远程工作站的 vLLM 部署与本地调用。若端口/模型/密钥有变，按需调整对应参数。 

### 三、代理常见问题与处理
- 连接超时多因未走代理或被 no_proxy 绕过。解决：
  1) 清除环境变量中的 `no_proxy/NO_PROXY`，避免直连内网 IP。
  2) 明确设置代理：`http_proxy=https_proxy=ALL_PROXY=socks5://127.0.0.1:1080`（或你的隧道端口）。
  3) 在脚本内设置上述环境变量可防止每次命令行重复配置。
- 排查步骤：
  1) 用简单 GET 测试 `http://<server>:<port>/v1/models`（requests 或 curl）是否可达。
  2) 确认服务端监听 `0.0.0.0:<port>` 且防火墙已放行。
  3) 如需长超时，可在客户端增加 `timeout` 参数。 
