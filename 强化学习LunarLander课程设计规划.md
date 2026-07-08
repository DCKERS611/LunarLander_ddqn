# 强化学习课程设计规划档案

## 一、课程设计基本信息

| 项目 | 内容 |
|---|---|
| 课程方向 | 人工智能课程设计 |
| 选题类别 | 强化学习 |
| 推荐题目 | 基于 Dueling Double DQN 与优先经验回放的 LunarLander 智能体训练研究 |
| 实验任务 | 训练一个智能体控制月球着陆器安全降落 |
| 主要算法 | DQN、Double DQN、Dueling DQN、Prioritized Experience Replay |
| 实验环境 | 本地 Windows + Python + PyTorch + Gymnasium |
| 本机显卡 | NVIDIA GeForce RTX 3050 Laptop GPU，4GB 显存 |
| 提交时间目标 | 2026 年 7 月 12 日前完成 |
| 项目定位 | 本地可完整完成、展示效果直观、具备一定算法改进和实验对比 |

---

## 二、选题说明

本课程设计选择强化学习方向中的 LunarLander 离散控制任务。该任务要求智能体根据月球着陆器的位置、速度、角度和支架接触状态等信息，选择合适的发动机控制动作，使着陆器安全降落到指定区域。

相比普通图像分类或手写数字识别任务，强化学习更能体现智能体在环境中的自主决策能力；相比大模型微调和 Stable Diffusion 等任务，LunarLander 对显存要求较低，适合在本地 RTX 3050 Laptop GPU 或 CPU 环境中完成。

本项目不是简单训练一个普通 DQN，而是在基础 DQN 上逐步引入 Double DQN、Dueling Network 和优先经验回放机制，通过对比实验和消融实验验证不同改进模块对训练稳定性、收敛速度和最终奖励的影响。

---

## 三、项目研究目标

### 3.1 总体目标

训练一个能够完成 LunarLander 着陆任务的强化学习智能体，使其在随机初始条件下能够通过控制发动机实现稳定降落。

### 3.2 具体目标

1. 搭建 LunarLander 强化学习实验环境。
2. 实现基础 DQN 算法。
3. 实现 Double DQN，缓解传统 DQN 的 Q 值过估计问题。
4. 实现 Dueling DQN，分离状态价值和动作优势。
5. 实现 Prioritized Experience Replay，使智能体优先学习 TD-error 较大的关键经验。
6. 对比不同算法的训练曲线、平均奖励、成功率和稳定性。
7. 保存训练过程、推理过程和最终着陆效果截图或视频。
8. 完成课程设计报告。

---

## 四、强化学习任务定义

### 4.1 训练内容

本项目训练的是一个能够自主决策的 LunarLander 智能体。智能体每一步根据当前状态选择一个动作，并根据环境反馈的奖励不断优化策略。

训练过程如下：

```text
观察状态 state
      ↓
选择动作 action
      ↓
环境反馈 reward 和 next_state
      ↓
存入经验回放池
      ↓
采样经验训练 Q 网络
      ↓
逐步学习更优策略
```

### 4.2 状态空间

LunarLander 的状态通常包含 8 个维度：

```text
1. 着陆器横向位置
2. 着陆器纵向位置
3. 横向速度
4. 纵向速度
5. 着陆器角度
6. 角速度
7. 左支架是否接触地面
8. 右支架是否接触地面
```

### 4.3 动作空间

动作空间为离散动作，通常包含 4 个动作：

```text
0：不启动发动机
1：启动左侧发动机
2：启动主发动机
3：启动右侧发动机
```

### 4.4 奖励目标

智能体需要学习：

```text
1. 尽量靠近指定着陆点
2. 降低下降速度
3. 保持机身角度稳定
4. 减少无效发动机喷射
5. 避免坠毁
6. 实现安全着陆
```

---

## 五、方法设计

## 5.1 基础 DQN

DQN 使用神经网络近似动作价值函数 Q(s, a)。

```text
输入：当前状态 s
输出：每个动作对应的 Q 值
策略：选择 Q 值最大的动作
```

基础 DQN 主要包含：

1. Q 网络。
2. Target Q 网络。
3. 经验回放池。
4. epsilon-greedy 探索策略。
5. Bellman 目标值更新。

---

## 5.2 Double DQN

传统 DQN 使用同一个目标网络选择和评估动作，容易导致 Q 值过估计。Double DQN 将动作选择和动作评估分离：

```text
使用 online network 选择下一状态的最优动作
使用 target network 计算该动作的 Q 值
```

预期效果：

```text
1. 降低 Q 值过估计
2. 提高训练稳定性
3. 改善策略收敛效果
```

---

## 5.3 Dueling DQN

Dueling DQN 将 Q 值分解为状态价值 V(s) 和动作优势 A(s, a)：

```text
Q(s, a) = V(s) + A(s, a) - mean(A(s, a))
```

网络结构：

```text
状态输入
   ↓
共享特征层
   ↓
┌───────────────┬────────────────┐
│ Value 分支    │ Advantage 分支  │
└───────────────┴────────────────┘
   ↓
合并得到 Q 值
```

预期效果：

```text
1. 更好地评估状态本身价值
2. 提高动作选择效率
3. 加快训练收敛
```

---

## 5.4 Prioritized Experience Replay

普通经验回放随机采样经验，而优先经验回放根据 TD-error 设置采样优先级。

核心思想：

```text
TD-error 越大，说明该经验当前学习误差越大
该经验越值得被优先学习
```

预期效果：

```text
1. 提高样本利用率
2. 加快关键经验学习
3. 提升训练效率
```

---

## 5.5 最终方法

最终方法定义为：

```text
Proposed = Double DQN + Dueling Network + Prioritized Experience Replay
```

实验将对比以下方法：

| 方法 | 说明 |
|---|---|
| DQN | 基础算法 |
| Double DQN | 在 DQN 基础上减少 Q 值过估计 |
| Dueling Double DQN | 在 Double DQN 基础上加入 Dueling 网络结构 |
| Proposed | Dueling Double DQN + 优先经验回放 |

当前实验后的推荐口径：

```text
工程实现层面：Proposed = Dueling Double DQN + Prioritized Experience Replay
最终推荐模型：Dueling Double DQN
```

说明：本项目已完成 Proposed 方法的代码实现和训练验证。实验结果显示，在当前 LunarLander-v3 训练规模与默认超参数下，PER 没有进一步提升 Dueling Double DQN，反而在后期出现性能退化。因此报告中将 Proposed 作为扩展方法和消融实验进行分析，最终展示效果以 Dueling Double DQN 为主。

---

## 六、工程结构规划

建议工程目录如下：

```text
rl-lunarlander-ddqn/
├── src/
│   ├── envs.py              # 环境创建与封装
│   ├── networks.py          # DQN / Dueling DQN 网络结构
│   ├── replay_buffer.py     # 普通经验回放与优先经验回放
│   ├── agent.py             # 智能体动作选择与网络更新
│   ├── train.py             # 训练入口
│   ├── evaluate.py          # 模型评估
│   ├── record_video.py      # 推理视频录制
│   └── plot_results.py      # 曲线绘制
├── configs/
│   ├── dqn.yaml
│   ├── double_dqn.yaml
│   ├── dueling_ddqn.yaml
│   └── proposed.yaml
├── outputs/
│   ├── logs/                # 训练日志
│   ├── checkpoints/         # 模型权重
│   ├── figures/             # reward/loss/epsilon 曲线
│   └── videos/              # 推理视频或 GIF
├── screenshots/             # 报告截图
├── report/                  # 课程报告
├── requirements.txt
└── README.md
```

---

## 七、Python 环境准备

本项目使用 `uv` 管理 Python 版本、虚拟环境、项目依赖和锁文件；同时保留 `requirements.txt`，方便课程提交和老师复现实验环境。

所有 `uv add`、`uv sync`、`uv export` 命令都在项目根目录执行，也就是包含 `pyproject.toml` 的目录：

```powershell
F:\AICourse\LunarLander_ddqn
```

不要在 `src` 目录下执行 `uv add`。`src` 只放源码，依赖应该写入根目录的 `pyproject.toml`，锁定结果写入根目录的 `uv.lock`。

### 7.1 创建 uv 虚拟环境

```powershell
cd F:\AICourse\LunarLander_ddqn

uv python install 3.12
uv python pin 3.12
uv venv --python 3.12
```

### 7.2 安装依赖

```powershell
uv add torch torchvision torchaudio
uv add numpy pandas matplotlib tqdm imageio opencv-python
uv add "gymnasium[box2d]" pygame swig
uv add pyyaml
```

如果是从已有 `pyproject.toml` 和 `uv.lock` 恢复环境，使用：

```powershell
uv sync
```

如需单独追加依赖，继续在项目根目录使用：

```powershell
uv add 包名
```

### 7.3 导出 requirements.txt

课程提交如果要求 `requirements.txt`，从 `uv.lock` 导出：

```powershell
uv export --format requirements.txt --no-hashes -o requirements.txt
```

如果只提交直接依赖，`requirements.txt` 可以保留为：

```text
torch
torchvision
torchaudio
numpy
pandas
matplotlib
tqdm
imageio
opencv-python
gymnasium[box2d]
pygame
swig
pyyaml
```

### 7.4 环境风险

如果 `gymnasium[box2d]` 或 `LunarLander-v3` 安装失败，立刻启用保底方案：

```text
保底环境：CartPole-v1
保底题目：基于改进 DQN 的离散动作控制任务智能体训练研究
```

不要在 Box2D 安装问题上消耗过多时间。

---

## 八、完成进度规划

## 阶段一：环境搭建与基础 DQN

### 目标

跑通强化学习环境，实现基础 DQN，并得到第一版训练曲线。

### 具体任务

1. 创建项目目录。
2. 创建 Python 虚拟环境。
3. 安装 PyTorch、Gymnasium、Box2D、Pygame 等依赖。
4. 测试 `LunarLander-v3` 环境是否可运行。
5. 如果 LunarLander 无法运行，切换到 `CartPole-v1`。
6. 实现基础 Q 网络。
7. 实现经验回放池。
8. 实现 epsilon-greedy 动作选择。
9. 实现 target network。
10. 实现训练主循环。
11. 跑 100-300 个 episode，确认 reward 能正常记录。

### 阶段产物

```text
1. 可运行的基础 DQN 代码
2. 初始训练日志
3. 初始 reward 曲线
4. 环境运行截图
```

### 截图清单

```text
1. 环境安装截图
2. LunarLander 环境运行截图
3. DQN 网络代码截图
4. 经验回放池代码截图
5. 训练开始截图
6. 初始 reward 曲线截图
```

---

## 阶段二：Double DQN 与 Dueling DQN

### 目标

在基础 DQN 上实现 Double DQN 和 Dueling Network。

### 具体任务

1. 修改 DQN 的目标值计算方式，实现 Double DQN。
2. 使用 online network 选择下一状态动作。
3. 使用 target network 评估该动作 Q 值。
4. 实现 Dueling Network：
   - 共享特征层
   - Value 分支
   - Advantage 分支
   - Q 值合并
5. 分别运行：
   - DQN
   - Double DQN
   - Dueling Double DQN
6. 保存三组训练日志。
7. 绘制三组初版 reward 曲线。

### 阶段产物

```text
1. Double DQN 代码
2. Dueling Network 代码
3. 三组实验初版结果
4. reward 曲线对比图
```

### 截图清单

```text
1. Double DQN 更新代码截图
2. Dueling 网络结构代码截图
3. 三组训练日志截图
4. 三组 reward 曲线截图
```

---

## 阶段三：优先经验回放与 Proposed 方法

### 目标

实现 Prioritized Experience Replay，并形成最终 Proposed 方法。

### 具体任务

1. 实现优先经验回放池。
2. 为每条经验维护 priority。
3. 根据 priority 进行采样。
4. 使用 TD-error 更新样本优先级。
5. 将 PER 接入 Dueling Double DQN。
6. 形成最终方法：

```text
Dueling Double DQN + Prioritized Experience Replay
```

7. 运行 Proposed 方法训练。
8. 保存训练日志、loss、reward、epsilon 和 checkpoint。

### 阶段产物

```text
1. PER 代码
2. Proposed 完整算法代码
3. Proposed 训练日志
4. checkpoint 文件
5. loss 曲线
```

### 截图清单

```text
1. PER 代码截图
2. TD-error 更新代码截图
3. Proposed 训练日志截图
4. checkpoint 文件截图
5. loss 曲线截图
```

### 阶段三实际完成记录

```text
1. 已实现 PrioritizedReplayBuffer：
   - 为每条经验维护 priority。
   - 使用 priority^alpha 计算采样概率。
   - 采样时返回 importance-sampling weight。
   - 每次训练后根据 abs(TD-error) 更新 priority。
2. 已将 PER 接入 Dueling Double DQN，新增 --method proposed。
3. 已完成 Proposed 默认 alpha=0.6 的 500 episode 训练：
   - outputs/proposed/train_log.csv
   - outputs/proposed/proposed_model.pth
   - outputs/proposed/proposed_checkpoint.pth
   - outputs/proposed/figures/proposed_reward_curve.png
   - outputs/proposed/figures/proposed_loss_curve.png
   - outputs/proposed/figures/proposed_epsilon_curve.png
4. 已完成 PER alpha=0.3 的 300 episode 对照实验：
   - outputs/proposed_alpha03/train_log.csv
   - outputs/proposed_alpha03/proposed_model.pth
   - outputs/proposed_alpha03/proposed_checkpoint.pth
   - outputs/proposed_alpha03/figures/proposed_reward_curve.png
   - outputs/proposed_alpha03/figures/proposed_reward_curve_last100.png
5. 已生成包含 Proposed 的四组 reward 对比曲线：
   - outputs/comparison/figures/reward_comparison_with_proposed.png
6. 7.9 验收记录已整理：
   - report/7.9_PER验收记录.md
```

### 阶段三实验结论

```text
PER 的工程验收通过，但实验效果没有达到预期。

默认 alpha=0.6 的 500 episode 实验中，Proposed 最高 reward 达到 313.75，
说明智能体曾经学到过有效着陆策略，但后期最后 100 集平均 reward 降至 -63.69，
出现明显性能退化。

降低优先级强度到 alpha=0.3 后，300 episode 最高 reward 只有 101.53，
最后 100 集平均 reward 为 -72.90，仍未展现稳定提升。

因此，后续报告将 PER 作为扩展模块和消融实验分析，而不是作为最终最优模型。
最终推荐模型保留为 Dueling Double DQN。
```

---

## 阶段四：完整训练与推理展示

### 目标

完成正式训练，保存最优模型，并录制智能体推理过程。

### 具体任务

1. 正式训练以下方法：
   - DQN
   - Double DQN
   - Dueling Double DQN
   - Proposed
2. 每种方法保存：
   - reward 日志
   - loss 日志
   - epsilon 日志
   - 最优 checkpoint
3. 使用最优模型进行测试。
4. 测试 10-20 个 episode。
5. 统计平均奖励、最高奖励和成功率。
6. 录制智能体着陆视频或 GIF。
7. 保存关键帧截图。

### 阶段产物

```text
1. 四组完整训练结果
2. reward 对比曲线
3. loss 曲线
4. epsilon 衰减曲线
5. 推理视频或 GIF
6. 成功着陆截图
```

### 截图清单

```text
1. 完整 reward 对比曲线截图
2. loss 曲线截图
3. epsilon 曲线截图
4. 推理运行截图
5. 成功着陆视频/GIF截图
```

### 阶段四实际完成记录

```text
1. 已复用前序完成的四组 500 episode 训练结果作为正式训练结果：
   - outputs/dqn/train_log.csv
   - outputs/double_dqn/train_log.csv
   - outputs/dueling_ddqn/train_log.csv
   - outputs/proposed/train_log.csv
2. 已新增 src/evaluate.py，用于加载训练后模型并进行固定 seed 的贪心策略评估。
3. 已新增 src/record_video.py，用于从候选 episode 中选择 reward 最高的一局并保存推理 GIF、MP4 和关键帧。
4. 已完成 DQN、Double DQN、Dueling Double DQN 和 Proposed 四组方法各 20 局评估，评估 seed 为 20260710-20260729，成功判定为 reward >= 200。
5. 四组测试结果如下：
   - DQN：平均 reward 146.28，最高 reward 258.21，成功率 35%。
   - Double DQN：平均 reward 131.69，最高 reward 266.71，成功率 40%。
   - Dueling Double DQN：平均 reward 131.46，最高 reward 282.42，成功率 40%。
   - Proposed：平均 reward -150.01，最高 reward 148.83，成功率 0%。
6. 已保存评估结果：
   - outputs/evaluation/dqn_eval_episodes.csv
   - outputs/evaluation/dqn_eval_summary.json
   - outputs/evaluation/double_dqn_eval_episodes.csv
   - outputs/evaluation/double_dqn_eval_summary.json
   - outputs/evaluation/dueling_ddqn_eval_episodes.csv
   - outputs/evaluation/dueling_ddqn_eval_summary.json
   - outputs/evaluation/proposed_eval_episodes.csv
   - outputs/evaluation/proposed_eval_summary.json
7. 已使用 Dueling Double DQN 生成推理展示，最佳案例 seed=20260722，reward=282.42，188 步成功着陆。
8. 已保存推理展示文件：
   - outputs/videos/dueling_ddqn_inference_best.gif
   - outputs/videos/dueling_ddqn_inference_best.mp4
   - outputs/videos/dueling_ddqn_inference_best_summary.json
9. 已保存随机策略和训练后策略关键帧：
   - screenshots/inference/random_policy_start.png
   - screenshots/inference/random_policy_middle.png
   - screenshots/inference/random_policy_end.png
   - screenshots/inference/dueling_ddqn_best_start.png
   - screenshots/inference/dueling_ddqn_best_middle.png
   - screenshots/inference/dueling_ddqn_best_end.png
10. 已整理 7.10 评估与推理记录：
   - report/7.10_评估与推理记录.md
```

---

## 阶段五：实验分析与报告撰写

### 目标

整理实验结果，完成课程报告主体内容。

### 具体任务

1. 汇总四组算法结果。
2. 计算每种方法的平均奖励。
3. 统计训练是否收敛。
4. 统计成功率。
5. 绘制最终曲线。
6. 编写实验结果分析。
7. 编写消融实验分析。
8. 整理全部截图。
9. 完成课程报告初稿。

### 主实验表格模板

| 方法 | 平均奖励 | 最高奖励 | 成功率 | 收敛轮数 | 训练耗时 |
|---|---:|---:|---:|---:|---:|
| DQN | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 |
| Double DQN | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 |
| Dueling DDQN | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 |
| Proposed | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 |

### 消融实验表格模板

| 方法 | 平均奖励 | 成功率 | 稳定性 | 说明 |
|---|---:|---:|---:|---|
| Proposed | 待填写 | 待填写 | 待填写 | 完整方法 |
| w/o Double DQN | 待填写 | 待填写 | 待填写 | 去掉 Double DQN |
| w/o Dueling | 待填写 | 待填写 | 待填写 | 去掉 Dueling Network |
| w/o PER | 待填写 | 待填写 | 待填写 | 去掉优先经验回放 |

### 阶段产物

```text
1. 实验结果表格
2. 消融实验表格
3. 曲线图
4. 报告初稿
5. 截图素材文件夹
```

---

## 阶段六：最终检查与提交

### 目标

完成报告和代码整理，不再进行大规模代码修改。

### 具体任务

1. 检查代码是否能运行。
2. 检查训练日志是否完整。
3. 检查截图是否覆盖代码、训练、推理和结果。
4. 检查报告图表编号。
5. 检查实验表格数据是否一致。
6. 检查报告格式。
7. 导出 Word 或 PDF。
8. 整理提交文件。

### 最终提交材料

```text
1. 课程报告 Word / PDF
2. 项目源代码
3. requirements.txt
4. 训练结果图
5. 推理视频或 GIF
6. 关键截图
7. 模型 checkpoint，可选
```

---

## 九、报告结构建议

```text
1. 绪论
   1.1 研究背景
   1.2 强化学习任务意义
   1.3 本文主要工作

2. 相关理论
   2.1 马尔可夫决策过程
   2.2 Q-learning
   2.3 DQN
   2.4 Double DQN
   2.5 Dueling DQN
   2.6 Prioritized Experience Replay

3. 方法设计
   3.1 LunarLander 任务建模
   3.2 DQN 智能体设计
   3.3 Double DQN 改进
   3.4 Dueling 网络结构
   3.5 优先经验回放机制
   3.6 完整算法流程

4. 实验设计
   4.1 实验环境
   4.2 参数设置
   4.3 对比方法
   4.4 评价指标

5. 实验结果与分析
   5.1 奖励曲线分析
   5.2 loss 曲线分析
   5.3 epsilon 衰减分析
   5.4 对比实验分析
   5.5 消融实验分析
   5.6 典型推理结果展示

6. 总结与展望
   6.1 本文总结
   6.2 不足之处
   6.3 后续改进方向
```

---

## 十、截图清单

### 10.1 代码截图

```text
1. Q 网络代码
2. Dueling 网络代码
3. 经验回放池代码
4. PER 代码
5. Double DQN 更新代码
6. 训练循环代码
7. 推理录制代码
```

### 10.2 训练截图

```text
1. 训练命令截图
2. 训练日志截图
3. reward 曲线截图
4. loss 曲线截图
5. epsilon 曲线截图
6. checkpoint 保存截图
```

### 10.3 推理截图

```text
1. 智能体随机策略运行截图
2. 训练后智能体推理截图
3. 成功着陆截图
4. 失败案例截图
5. 视频或 GIF 文件截图
```

### 10.4 实验结果截图

```text
1. 主实验表格截图
2. 消融实验表格截图
3. 四种算法 reward 对比图
4. 典型结果分析截图
```

---

## 十一、评价指标

| 指标 | 含义 |
|---|---|
| 平均奖励 | 测试若干轮后的平均 episode reward |
| 最高奖励 | 训练或测试过程中取得的最高奖励 |
| 成功率 | 测试 episode 中成功着陆的比例 |
| 收敛轮数 | 平均奖励达到设定阈值所需 episode 数 |
| 训练稳定性 | reward 曲线波动情况 |
| 平均 loss | Q 网络训练损失 |
| 推理效果 | 智能体是否能稳定控制着陆器 |

---

## 十二、创新点写法

报告中可以将创新点写为：

1. 针对传统 DQN 容易出现 Q 值过估计的问题，引入 Double DQN，将动作选择与动作价值评估分离，提高训练稳定性。
2. 针对不同状态下动作价值差异不明显的问题，引入 Dueling Network，将状态价值和动作优势分开建模，提高策略学习效率。
3. 针对普通经验回放随机采样导致关键经验利用不足的问题，引入优先经验回放机制，根据 TD-error 对经验样本赋予不同采样优先级。
4. 通过 DQN、Double DQN、Dueling Double DQN 和 Proposed 方法的对比实验，分析不同改进模块对 LunarLander 控制任务的影响。

---

## 十三、风险预案

### 13.1 LunarLander 环境安装失败

处理方式：

```text
优先尝试重新安装 gymnasium[box2d]
如果仍失败，切换到 CartPole-v1
```

报告处理方式：

```text
将题目改为：基于改进 DQN 的离散动作控制任务智能体训练研究
实验环境改为 CartPole-v1
算法结构保持不变
```

### 13.2 训练效果不稳定

处理方式：

```text
1. 增加训练 episode
2. 调整 learning rate
3. 调整 batch size
4. 调整 epsilon 衰减速度
5. 使用移动平均曲线展示趋势
```

### 13.3 时间不够

优先保留：

```text
1. DQN
2. Double DQN
3. Dueling DQN
4. reward 曲线
5. 推理截图
6. 报告
```

可以弱化：

```text
1. PER
2. 大量消融实验
3. 长时间训练
4. 复杂视频展示
```

---

## 十四、最低可交版本

```text
DQN
+ Double DQN
+ Dueling DQN
+ reward 曲线
+ loss 曲线
+ 推理截图
+ 实验表格
+ 课程报告
```

最低可交版本必须证明：

```text
1. 有代码实现
2. 有训练过程
3. 有训练曲线
4. 有推理结果
5. 有对比分析
```

---

## 十五、高分版本

```text
DQN
+ Double DQN
+ Dueling DQN
+ Prioritized Experience Replay
+ 完整消融实验
+ 成功着陆视频/GIF
+ 多指标实验对比
+ 完整课程报告
```

高分版本重点体现：

```text
1. 选题有挑战性
2. 方法有改进
3. 训练过程完整
4. 推理结果直观
5. 实验分析充分
```

---

## 十六、最终汇报口径

汇报时不要只说：

```text
我训练了一个 DQN 智能体玩 LunarLander。
```

建议说：

```text
本项目针对 LunarLander 离散动作控制任务，设计并实现了基于 DQN 的强化学习智能体。在基础 DQN 的基础上，引入 Double DQN 缓解 Q 值过估计问题，引入 Dueling Network 提高状态价值建模能力，并结合优先经验回放提升关键经验利用效率。通过多组对比实验和消融实验，验证了改进方法在平均奖励、收敛速度和训练稳定性方面的效果。
```

---

## 十七、最终检查清单

```text
[x] Python 虚拟环境创建完成
[x] 依赖安装完成
[x] LunarLander 或 CartPole 环境可运行
[x] DQN 代码完成
[x] Double DQN 代码完成
[x] Dueling DQN 代码完成
[x] PER 代码完成
[x] 训练日志保存完成
[x] reward 曲线保存完成
[x] loss 曲线保存完成
[x] epsilon 曲线保存完成
[x] checkpoint 保存完成
[x] 推理视频或截图保存完成
[ ] 主实验表格完成
[ ] 消融实验表格完成
[ ] 报告截图整理完成
[ ] 课程报告完成
[ ] Word / PDF 导出完成
[ ] 提交文件整理完成
```
