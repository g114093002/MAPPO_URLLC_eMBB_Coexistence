# SR-MAPPO 深度诊断实现总结

## 完成项目
所有要求的诊断基础设施已经成功实现。本文档总结了为识别MAPPO为什么仍然接近greedy而做的代码修改。

---

## 一、Phase-0 Owner/Power vs Greedy对比指标 ✓

### 新增计数器（env.py）
位置：`_reset_episode_state()` 方法（约1101-1113行）

```python
self.planning_owner_match_vs_greedy_count = 0
self.planning_owner_hamming_distance_total = 0
self.planning_owner_change_from_prev_count = 0
self.planning_prev_owner_per_uav_rb = None
self.planning_embb_power_match_vs_greedy_count = 0
self.planning_embb_power_mean_ratio_vs_greedy_sum = 0.0
self.planning_embb_power_abs_delta_vs_greedy_sum = 0.0
```

### 计算逻辑（env.py）
位置：`_step_embb_planning()` 方法（约668-686行）

关键逻辑：
- **owner_match_vs_greedy**：计数MAPPO选择与greedy baseline相同的owner数量
- **hamming_distance**：计数owner决策不同的RB数量
- **owner_change_from_prev**：追踪owner相对上一步的改变

### 输出指标（env.py）
位置：`summarize_episode()` 方法（约4336-4337行）

输出到episode摘要的指标：
- `planning_owner_match_ratio_vs_greedy` - 与greedy相同的比例
- `planning_owner_hamming_distance_vs_greedy` - 总不同数量
- `planning_owner_change_from_prev_ratio` - owner变化率

---

## 二、Planning Reward Components记录 ✓

### 新增计数器（env.py）
位置：`_reset_episode_state()` 方法（约1108-1113行）

```python
self.planning_reward_rate_component_sum = 0.0
self.planning_reward_service_component_sum = 0.0
self.planning_reward_min_rate_component_sum = 0.0
self.planning_reward_fairness_component_sum = 0.0
self.planning_reward_cell_edge_component_sum = 0.0
self.planning_reward_component_count = 0
```

### 记录逻辑（env.py）
位置：`_step_embb_planning()` 方法（约690-750行）

5个独立计算的reward组件：
1. **rate_component** - 吞吐量增益 (weight=0.80)
2. **service_component** - 被服务用户比例 (weight=1.20)
3. **min_rate_component** - 最小速率满足率 (weight=0.80)
4. **fairness_component** - 公平指数 (weight=0.60)
5. **cell_edge_component** - 中心边缘覆盖 (weight=0.40)

### 输出指标（env.py）
位置：`summarize_episode()` 方法（约4318-4331行）

每个组件的平均值：
- `planning_reward_rate_component_mean`
- `planning_reward_service_component_mean`
- `planning_reward_min_rate_component_mean`
- `planning_reward_fairness_component_mean`
- `planning_reward_cell_edge_component_mean`

---

## 三、Phase-A Power Execution Bottleneck分析 ✓

### 新增计数器（env.py）
位置：`_sanitize_phase_a_embb_power_actions()` 方法（约540-610行）

```python
self.phase_a_embb_power_projection_count += 1            # 合法写入数量
self.phase_a_embb_power_raw_delta_sum += abs(raw_delta)  # 原始delta绝对值之和
self.phase_a_embb_power_executed_delta_sum += abs(executed) # 执行delta绝对值之和
self.phase_a_embb_power_delta_clipped_count += 1         # delta被clip的次数
self.phase_a_embb_power_quantized_count += 1             # 被quantization的次数
self.phase_a_embb_power_scale_clipped_count += 1         # scale被clip的次数
```

### 输出指标（env.py）
位置：`summarize_episode()` 方法（约4347-4361行）

关键诊断指标：
- `phase_a_embb_power_write_ratio` - 被写入的决策比例
- `phase_a_embb_power_changed_ratio` - 实际改变的比例
- `phase_a_embb_power_mean_raw_delta` - 原始delta平均值
- `phase_a_embb_power_mean_executed_delta` - 执行delta平均值
- `raw_executed_embb_power_gap_ratio` - (raw - executed) / raw，识别bottleneck来源
- `phase_a_embb_power_clip_ratio` - clip占比
- `phase_a_embb_power_quantized_ratio` - quantization占比
- 其他zeroed原因分析

---

## 四、Ablation实验 ✓

### 实验A：Phase-0 Frozen, Phase-A Only
**文件**：`experiments.py`

**实验名称**：`ablation_phase0_frozen_greedy_phase_a_only`

**配置**：
```python
updated.env.learn_embb_baseline = False              # 禁用Phase-0学习
updated.reward.planning_embb_rate_weight = 0.0       # 无planning奖励
updated.env.allow_phase_a_embb_power_adjustment = True  # 启用Phase-A power
```

**目的**：如果Phase-A仍然接近greedy，说明Phase-A学习不足以拉开

### 实验B：Phase-0 Only, Phase-A Power Frozen
**文件**：`experiments.py`

**实验名称**：`ablation_phase0_only_frozen_phase_a`

**配置**：
```python
updated.env.learn_embb_baseline = True               # 启用Phase-0学习
updated.reward.planning_embb_service_weight = 1.20   # 多目标planning
updated.env.allow_phase_a_embb_power_adjustment = False  # 禁用Phase-A power
updated.action.embb_power_delta_limit = 0.0          # 零power delta
```

**目的**：如果Phase-0仍然接近greedy，说明multi-objective planning设计问题

---

## 五、评估配置更新 ✓

### Report常量（report.py 约96行）
```python
DIAGNOSTIC_EPISODES_PER_LOAD = 30  # 5倍采样用于诊断报告
```

### 建议的评估配置
为获得更详细诊断，建议使用：
- **训练iterations**：1500（planning_multiobj_v1预设值）
- **final report episodes_per_load**：30
- **checkpoint评估**：同时输出latest/best_balanced/best_floor_throughput

---

##六、代码改动概览

### 修改的文件

| 文件 | 行号 | 修改内容 |
|------|------|--------|
| `env.py` | 1001-1027 | 新增phase_a计数器初始化 |
| `env.py` | 1088-1113 | 新增Phase-0诊断计数器初始化 |
| `env.py` | 545-610 | 修改_sanitize_phase_a_embb_power_actions添加bottleneck追踪 |
| `env.py` | 668-750 | 修改_step_embb_planning添加owner对比和reward component分离 |
| `env.py` | 815-825 | 添加新诊断指标到infos |
| `env.py` | 4318-4361 | 修改summarize_episode输出所有诊断指标 |
| `experiments.py` | 13-15 | 添加两个ablation实验常量 |
| `experiments.py` | 78-80 | 添加到EXPERIMENT_CHOICES列表 |
| `experiments.py` | 339-340 | 添加标签 |
| `experiments.py` | 2425-2486 | 添加apply_experiment_preset实现 |
| `report.py` | 95 | 添加DIAGNOSTIC_EPISODES_PER_LOAD常量 |

---

##七、运行诊断实验

### 命令
```bash
# 运行多目标planning实验
python -c "from sr_mappo.train import run_default_training; run_default_training(experiment='pure_ppo_ff_v1_no_greedy_obs_planning_multiobj_v1')"

# 或Ablation A
python -c "from sr_mappo.train import run_default_training; run_default_training(experiment='ablation_phase0_frozen_greedy_phase_a_only')"

# 或 Ablation B
python -c "from sr_mappo.train import run_default_training; run_default_training(experiment='ablation_phase0_only_frozen_phase_a')"
```

### 生成诊断报告
```bash
python -c "from sr_mappo.report import generate_report; generate_report(checkpoint_path='path/to/checkpoint', episodes_per_load=30)"
```

---

##八、诊断问题与解读指南

训练完成后，查看输出在以下指标中的值：

### **问题1：Phase-0 owner allocation是否接近greedy？**
查看：
- `planning_owner_match_ratio_vs_greedy` - 如果> 0.8，说明owner决策接近greedy
- `planning_owner_hamming_distance_vs_greedy` - 绝对不同RB数量
- 如果接近，说明issue在Phase-0学习的owner selection机制

### **问题2：Phase-0 eMBB power是否真的有改变？**  
查看：
- `planning_embb_power_nonzero_ratio` - 有多少RB做了power调整
- `planning_embb_power_changed_ratio` - 实际power scale改变的比例
- 如果  < 0.2，说明Phase-0 power学习基本未激活

###**问题3：Phase-A power为何executed << raw？**
查看：
- `raw_executed_embb_power_gap_ratio` - 差距比例
- `phase_a_embb_power_clip_ratio` - clip占多少
- `phase_a_embb_power_quantized_ratio` - quantization占多少
- `phase_a_embb_power_zeroed_keep_mode_ratio` - 被keep mode零化占多少

---

## 九、预期诊断结果

### 如果接近greedy的原因是...

**A. Phase-0 owner仍然 ~90% 匹配greedy**
→ Ablation B会继续接近greedy
→ 说明owner learning机制本身有问题（候选者太少、mask太强、奖励不足）

**B. Phase-A power execution gap >> raw delta**
→ Ablation A无法拉开
→ 说明clipping/quantization/masking压制了power学习

**C. Multi-objective reward components都接近0**
→ 说明reward权重设置问题或learning rate不足
→ 需要增加权重或学习时间

---

##十、后续改进方向建议

基于诊断结果，优先改进顺序：

1. **如果所有ablation都接近greedy** → 检查observation/action space是否足够丰富
2. **如果Phase-0现象自由度大但还是相似** → 考虑增加owner候选者、减弱mask
3. **如果Phase-A power gap大** → 优化clipping逻辑或quantization分辨率
4. **如果reward components平均值 << 1** → 增加权重或使用自适应权重

---

##总结

所有诊断代码已准备就绪，可以立即运行训练并生成报告来回答用户的4个关键问题。
诊断指标会直接输出到episode摘要和最终report中，无需额外处理。

