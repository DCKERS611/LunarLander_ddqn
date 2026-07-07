# LunarLander DDQN Course Project

强化学习课程设计项目：基于 DQN 及其改进方法训练 LunarLander 智能体。

## 项目内容

- LunarLander-v3 环境验证
- 随机智能体基线
- 基础 DQN
- 训练日志保存
- reward 曲线绘制
- 课程设计报告框架

## 环境

本项目使用 `uv` 管理 Python 环境和依赖。

```powershell
uv sync
```

## 运行

测试环境：

```powershell
uv run python src/test_env.py
```

训练基础 DQN：

```powershell
uv run python src/train.py
```

绘制 reward 曲线：

```powershell
uv run python src/plot_results.py
uv run python src/plot_last100_reward.py
```

## 当前结果

基础 DQN 已完成 500 个 episode 训练，训练日志和模型权重保存在 `outputs/dqn/`。
