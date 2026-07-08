# LunarLander DDQN 强化学习课程设计

本项目基于 PyTorch 和 Gymnasium 实现 LunarLander-v3 离散控制任务的强化学习智能体训练。项目从基础 DQN 出发，逐步实现 Double DQN、Dueling Double DQN 和 Prioritized Experience Replay，并通过训练曲线、固定 seed 评估、成功率和推理视频对不同方法进行对比分析。

## 项目目标

- 搭建 LunarLander-v3 强化学习实验环境。
- 实现基础 DQN、Double DQN、Dueling Double DQN 和 Proposed 方法。
- 保存训练日志、模型权重、reward/loss/epsilon 曲线。
- 对训练后模型进行贪心策略评估，统计平均奖励、最高奖励和成功率。
- 录制 Dueling Double DQN 的成功着陆 GIF/MP4，作为报告展示材料。

## 方法说明

| 方法 | 说明 |
|---|---|
| DQN | 基础深度 Q 网络，包含经验回放、target network 和 epsilon-greedy 策略 |
| Double DQN | 使用 online network 选动作、target network 估值，缓解 Q 值过估计 |
| Dueling DDQN | 在 Double DQN 基础上引入 Value/Advantage 双分支网络 |
| Proposed | Dueling Double DQN + Prioritized Experience Replay |

当前实验结论：Proposed 方法的工程流程完整，但在当前训练规模和默认超参数下未超过 Dueling Double DQN。最终推荐展示模型为 Dueling Double DQN。

## 项目结构

```text
LunarLander_ddqn/
├── src/                      # 核心源码
│   ├── envs.py               # 环境创建
│   ├── networks.py           # DQN / Dueling DQN 网络
│   ├── replay_buffer.py      # 普通经验回放与优先经验回放
│   ├── agent.py              # 智能体动作选择与网络更新
│   ├── train.py              # 训练入口
│   ├── evaluate.py           # 模型评估
│   ├── record_video.py       # 推理视频、GIF 和关键帧生成
│   └── plot_results.py       # 实验曲线绘制
├── configs/                  # 实验配置
├── outputs/                  # 训练日志、模型、图表、评估结果和视频
├── screenshots/              # 报告截图素材
├── report/                   # Markdown 实验记录与报告素材
├── pyproject.toml            # uv 项目配置
├── uv.lock                   # 依赖锁文件
└── requirements.txt          # 提交用依赖清单
```

项目结构图位于：

```text
screenshots/project_structure.png
```

## 环境准备

本项目使用 `uv` 管理 Python 环境和依赖。

```powershell
uv sync
```

如需使用 `requirements.txt` 创建普通虚拟环境：

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## 运行方式

测试 LunarLander 环境：

```powershell
uv run python src/test_env.py
```

训练四组方法：

```powershell
uv run python src/train.py --method dqn --episodes 500
uv run python src/train.py --method double_dqn --episodes 500
uv run python src/train.py --method dueling_ddqn --episodes 500
uv run python src/train.py --method proposed --episodes 500
```

绘制训练曲线：

```powershell
uv run python src/plot_results.py
uv run python src/plot_last100_reward.py
```

评估训练后模型：

```powershell
uv run python src/evaluate.py --method dqn
uv run python src/evaluate.py --method double_dqn
uv run python src/evaluate.py --method dueling_ddqn
uv run python src/evaluate.py --method proposed
```

录制 Dueling Double DQN 推理展示：

```powershell
uv run python src/record_video.py --method dueling_ddqn
```

## 实验结果摘要

### 训练阶段

| 方法 | 最后 100 集均分 | 最高训练 reward | 最后一集 reward | 结论 |
|---|---:|---:|---:|---|
| DQN | 173.70 | 310.74 | 218.27 | 已学到基本着陆策略，但稳定性不足 |
| Double DQN | 58.53 | 302.55 | 251.87 | 收敛较慢，策略偏保守 |
| Dueling DDQN | 197.87 | 308.29 | -39.18 | 最接近 200 分解决标准 |
| Proposed | -63.69 | 313.75 | -119.36 | 后期性能退化 |

### 评估阶段

评估采用 epsilon=0 的贪心策略，每种方法测试 20 局，成功标准为 episode reward >= 200。

| 方法 | 平均 reward | 最高 reward | 成功率 | 平均步数 |
|---|---:|---:|---:|---:|
| DQN | 146.28 | 258.21 | 35% | 741.5 |
| Double DQN | 131.69 | 266.71 | 40% | 748.4 |
| Dueling DDQN | 131.46 | 282.42 | 40% | 681.5 |
| Proposed | -150.01 | 148.83 | 0% | 131.6 |

Dueling Double DQN 在训练曲线、最高推理 reward 和成功着陆展示方面综合表现最好，因此作为最终推荐模型。

## 主要输出

| 内容 | 路径 |
|---|---|
| 训练日志 | `outputs/*/train_log.csv` |
| 模型权重 | `outputs/*/*_model.pth` |
| reward 对比曲线 | `outputs/comparison/figures/` |
| Proposed loss/epsilon 曲线 | `outputs/proposed/figures/` |
| 评估结果 | `outputs/evaluation/` |
| 推理 GIF/MP4 | `outputs/videos/` |
| 推理关键帧 | `screenshots/inference/` |
| 项目结构图 | `screenshots/project_structure.png` |

## 报告材料说明

`report/` 目录保留 Markdown 实验记录和报告辅助材料。正式 Word 报告属于本地提交材料，已通过 `.gitignore` 排除，不会上传到 GitHub。

## 最终结论

本项目完成了 LunarLander-v3 控制任务的 DQN 系列算法实现、训练、评估和推理展示。实验表明，Dueling Double DQN 在本任务中具有最好的综合表现；优先经验回放虽然完成了工程实现，但在当前训练设置下没有带来稳定提升，适合作为消融实验和负结果分析保留。
