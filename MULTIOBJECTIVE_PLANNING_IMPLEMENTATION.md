# SR-MAPPO 多目标规划改进方案 - 实施完成

## 修改概览

已针对用户需求完成以下改进：

### 1️⃣ **配置扩展** (config.py)
- 新增planning多目标权重配置：
  - `planning_embb_service_weight` (default: 0.0)
  - `planning_embb_min_rate_weight` (default: 0.0)
  - `planning_embb_fairness_weight` (default: 0.0)
  - `planning_cell_edge_weight` (default: 0.0)

### 2️⃣ **Planning阶段多目标奖励** (env.py - _step_embb_planning)
- 替换单一throughput目标为：
  - **Throughput目标**：延续原有的delta_rate奖励
  - **Service质量**：直接奖励embb_served_users比例
  - **Min-Rate满足**：奖励达成1Mbps目标的用户比例  
  - **Fairness**：Jain公平性指数奖励
  - **Cell-Edge**：已服务用户中的覆盖率奖励

### 3️⃣ **Planning Owner Fallback检测** (env.py)
- 新增计数器：
  - `planning_owner_invalid_count` - 无效owner选择次数
  - `planning_owner_fallback_to_first_candidate_count` - fallback到首候选的次数
  - `planning_owner_fallback_to_first_candidate_ratio` - fallback比例

- 修改logic：
  - 检测policy选择invalid option时的fallback现象
  - 输出在episode info中供logging

### 4️⃣ **新Experiment Preset** (experiments.py)
```
pure_ppo_ff_v1_no_greedy_obs_planning_multiobj_v1
```

配置细节：
| 设置项 | 值 | 说明 |
|------|-----|------|
| planning_embb_rate_weight | 0.80 | ↓ 降低，避免throughput-only |
| planning_embb_service_weight | 1.20 | ↑ 新增，服务用户比例 |
| planning_embb_min_rate_weight | 0.80 | ↑ 新增，min-rate满足 |
| planning_embb_fairness_weight | 0.60 | ↑ 新增，公平性 |
| planning_cell_edge_weight | 0.40 | ↑ 新增，覆盖率 |
| action.embb_power_delta_limit | 0.60 | ↑ 增大，支持更大的power调整 |
| total_iterations | 1500 | 完整收敛 |

---

## 训练命令

### 启动新实验：
```bash
python -c "from sr_mappo.train import run_default_training; run_default_training(experiment='pure_ppo_ff_v1_no_greedy_obs_planning_multiobj_v1')"
```

### 带Report的完整流程：
```bash
python -c "from sr_mappo.train import run_default_training; run_default_training(with_report=True, experiment='pure_ppo_ff_v1_no_greedy_obs_planning_multiobj_v1')"
```

### 训练完成后生成报告：
```bash
python -c "from sr_mappo.report import generate_report; generate_report(experiment_line='pure_ppo_ff_v1_no_greedy_obs_planning_multiobj_v1')"
```

---

## 预期输出指标（自动采集）

### Planning Owner Logging：
- `planning_owner_fallback_to_first_candidate_ratio` - 应显示policy是否因为candidate ordering而collapse
- `planning_owner_invalid_ratio` - 无效owner选择的比例
- `planning_owner_non_null_ratio` - 有效owner分配的比例

### Phase-A Power（已有）：
- `phase_a_embb_power_changed_ratio` - 电源变化频率
- `phase_a_embb_power_mean_raw_delta` - 原始电源增量
- `phase_a_embb_power_mean_executed_delta` - 执行电源增量

---

## 验证清单：

训练完成后，用下列命令检查是否真的与贪心拉开距离：

```bash
# 生成report后，查看这些曲线：
# 1. embb_rate - 应该不再是贪心的几乎完全复制
# 2. embb_positive_rate_ratio / embb_user_rate - 应该明显高于贪心
# 3. jain_fairness - 应该明显更好（向1靠近）
# 4. phase_a_embb_power_mean_executed_delta - 应该非零且相对稳定
# 5. planning_owner_fallback_to_first_candidate_ratio - 应该较低（<20%）表示policy在学习
```

---

## 关键改进点

✅ **Planning不再throughput-only** - 多目标权重明确分化  
✅ **Owner fallback检测** - 可验证policy是否真的在学习vs collapse  
✅ **Phase-A power增强** - 更大的delta_limit + 更长的训练时间  
✅ **Service质量指标** - 直接激励embb_served_users和fairness  

下一步：运行训练并检查上述曲线是否与贪心显著不同。
