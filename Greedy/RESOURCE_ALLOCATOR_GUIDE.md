# Resource Allocator 指南（中文版）

本文件說明 `resource_allocator.py` 的功能、SIC/功率控制邏輯的位置，
以及它如何被 Greedy 與 SR‑MAPPO 共同使用。

檔案位置：[resource_allocator.py](/d:/URLLC_eMBB_Coexisting/Greedy/resource_allocator.py)

---

## 1) 這個模組負責什麼

`ResourceAllocator` 是模擬器的 **PHY/MAC 執行核心**，主要負責：

- eMBB baseline 的 RB 擁有權與功率預算
- URLLC 在 minislot 級別的 admission 與排程
- overlay 與 puncture 的決策邏輯
- SIC 可行性檢查與 overlay 後的 eMBB 保留率
- 功率搜尋（URLLC 用二分法，eMBB 用本地 refinement）

SR‑MAPPO 的 policy 只負責「選動作」，
所有物理層計算都在這個檔案裡完成。

---

## 2) 主要入口函式

### `allocate_embb_greedy(...)`

建立 **eMBB baseline RB 配置**：

- 將每個 eMBB user 指派到 UAV
- 每個 RB 找出最好的一個 eMBB user（貪婪最大化速率）
- 依 RB 份額分配 per‑user 功率
- 計算 baseline rate 與 RB 擁有權

輸出：
- `rb_allocation`, `owner_per_rb`, `owner_per_uav_rb`
- `embb_power_allocation`, `base_rb_rates`, `rates`

### `allocate_urllc_power(...)`

在 minislot 內排 URLLC：

- 建立本 minislot 的 URLLC packet 集合
- 搜尋 overlay 或 puncture 的最佳可行動作
- 檢查可靠度與 admission 規則
- 更新 URLLC power / mode grid / 診斷資訊

輸出：
- `urllc_power_allocation`, `urllc_timefreq_grid`
- `noma_decisions`, `overlay/puncture diagnostics`

### `adjust_embb_after_urllc(...)`

URLLC 決策完成後：

- 套入 SIC residual 影響
- 重新計算 eMBB rate
- 視需要做 eMBB power refinement

---

## 3) SIC 與 Overlay 的邏輯在哪裡

SIC 相關檢查在：

- `_find_best_urllc_action(...)`
- `_compute_embb_state(...)`

關鍵內容：

- **URLLC 可靠度檢查（finite blocklength）**  
  `CapacityModels.decoding_error_probability(...)`

- **Post‑SIC eMBB SNIR**  
  在 `_find_best_urllc_action` 與 `_compute_embb_state` 中計算

- **Residual interference factor**  
  `algo_cfg.sic_residual_factor`（[config.py](/d:/URLLC_eMBB_Coexisting/Greedy/config.py)）

---

## 4) Power Control 在哪裡

### URLLC power

URLLC 最小功率透過二分法：

- `_bisection_search_urllc_power(...)`
- 由 `_find_best_urllc_action(...)` 呼叫

### eMBB power

eMBB 功率分兩階段：

1. `allocate_embb_greedy(...)` 設定初始 per‑user budget  
2. `_refine_embb_powers(...)` 做本地 refinement

---

## 5) 重要輔助函式

- `_compute_action_utility(...)`  
  計算 overlay/puncture 的效用（可靠度、eMBB 損失、功率成本）

- `_compute_intercell_interference(...)`  
  彙整跨 UAV 干擾（eMBB + URLLC）

- `_compute_embb_state(...)`  
  在 URLLC grid 與 SIC 條件下重新計算 eMBB rate

---

## 6) SR‑MAPPO 會用哪些

SR‑MAPPO 不會自己重寫 PHY/MAC，
而是直接呼叫：

- `allocate_embb_greedy`（baseline）
- `allocate_urllc_power`（可行動作評估）
- `adjust_embb_after_urllc`（共存後 rate）
