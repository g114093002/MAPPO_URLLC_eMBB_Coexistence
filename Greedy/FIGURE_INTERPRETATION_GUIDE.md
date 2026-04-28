# Figure Interpretation Guide

這份文件把目前主程式 [main.py](/d:/URLLC_eMBB_Coexisting/Greedy/main.py) 產生的主要圖，連同每張圖裡的小圖，一張一張說清楚：

- 這張圖在畫什麼
- 每個子圖代表什麼
- 為什麼會出現現在這樣的結果
- 哪些現象是合理的，哪些現象值得再追

本文的解讀都建立在目前這版 system model 上：

- 三台 UAV，各自有獨立的 `RB x mini-slot` 資源池
- uplink coexistence only:
  - `OMA`
  - `eMBB + URLLC overlay / NOMA`
  - `URLLC puncturing`
- 不允許 `eMBB-eMBB NOMA`
- 不允許 `URLLC-URLLC NOMA`
- URLLC reliability 是 hard constraint
- URLLC traffic 已改成 packet-level Poisson arrival

---

## A. 先回答一個最重要的問題

### 為什麼 `power_vs_density` 裡的 aggregate eMBB throughput 不是水平線？

你的直覺是：

- 資源總量固定
- RB 最後都會被分配出去
- 所以 aggregate eMBB throughput 好像應該接近常數

這個推論只在很特定的情況下成立，例如：

- 每個 RB 都永遠由 eMBB 使用
- 每個 RB 的有效 SINR 幾乎不變
- 沒有 URLLC 插入
- 沒有 puncturing
- overlay 不會改變 eMBB 的有效速率
- power per RB 不會因 user 數和 coexistence 模式而改變

但你現在的模型不是這樣。現在 aggregate eMBB throughput 會變動，原因至少有 6 個：

1. `URLLC puncturing` 會直接吃掉原本屬於 eMBB 的 minislot。
2. `overlay/NOMA` 雖然保留部分 eMBB rate，但 retention 小於 1，不是無損。
3. eMBB 的 rate 是 `log2(1+SINR)`，不是只看有沒有拿到 RB。
4. density 增加後，user 關聯、通道條件、inter-cell interference 都會變。
5. 雖然 RB 仍分出去，但不代表每個 RB 上的有效頻譜效率不變。
6. 某些 cell 在高負載下實際上會被 URLLC 主導，eMBB 只剩很少甚至接近 0 的有效貢獻。

所以：

- `RB 有被分配` 不等於 `eMBB throughput 不變`
- 你現在畫的是 **有效 aggregate eMBB throughput**
- 不是「單純把所有 RB 數量加總」

更精確地說，你現在這張 aggregate throughput 圖反映的是：

`在 coexistence、interference、power allocation、puncturing、overlay retention 全部作用後，系統還剩多少 eMBB 總速率`

因此它不是水平線，這是合理的。

如果你要畫一張接近你直覺的圖，那應該是：

- `baseline eMBB-only throughput with no URLLC`
- 或 `total assigned eMBB RB count`

那種圖才比較可能接近平線。

---

## 1. [power_vs_density.png](/d:/URLLC_eMBB_Coexisting/Greedy/results/power_vs_density.png)

這張是總覽圖，用來看系統在 user density 增加時，整體性能怎麼變。

### 左上：`Total Tx Power vs User Density`

**代表什麼**

- 所有 eMBB 與 URLLC 的總發射功率

**為什麼會這樣**

- density 增加後，URLLC offered load 也跟著拉高
- 共存動作變多，尤其 puncture 與 overlay 的數量增加
- 系統為了滿足 hard reliability，URLLC power 往往需要往上推

**怎麼看**

- 如果 power 增加但 throughput 沒有跟著增加，表示系統開始進入低效率區

### 右上：`Aggregate eMBB Throughput`

**代表什麼**

- 所有 eMBB user 的有效總速率加總

**為什麼不是水平線**

- 因為它不是資源數量圖，而是速率圖
- 受 puncturing、overlay retention、interference、power split 影響

**怎麼看**

- 如果高 density 時 aggregate throughput 下降，表示 coexistence 成本已經大到足以吞掉總 eMBB 產能

### 左下：`Per-user eMBB Rate Collapse`

**代表什麼**

- 平均每個 eMBB user 的 rate

**為什麼通常掉得比 aggregate 更快**

- aggregate throughput 可能還被少數 user 撐住
- 但 user 數變多時，平均到每個 user 的 rate 一定更容易崩

### 右下：`Admission / Service Stress`

**兩條線代表**

- `eMBB served ratio`
- `URLLC admission ratio`

**為什麼重要**

- 這張是系統是否開始 overload 的直接訊號
- eMBB served ratio 掉，表示越來越多 eMBB user 幾乎沒被服務
- URLLC admission ratio 掉，表示 offered URLLC packets 已經超過當下 capacity

### 最下左：`Admitted URLLC Reliability`

**代表什麼**

- 只有被 admission 的 URLLC packets，其可靠度平均值

**為什麼接近 1 是合理的**

- 因為 reliability 是 hard constraint
- 正常系統行為本來就是：
  - admission 掉
  - admitted reliability 保持高

### 最下右：文字框

**代表什麼**

- 這不是數據，是這張圖的口頭摘要

---

## 2. [mode_tradeoff_analysis.png](/d:/URLLC_eMBB_Coexisting/Greedy/results/mode_tradeoff_analysis.png)

這張是 mode-aware 圖，專門看 `overlay` 與 `puncture` 的使用方式。

### 左上：`Mode Selection Ratio vs User Density`

**兩條線**

- `Overlay ratio`
- `Puncture ratio`

**怎麼解讀**

- overlay 不是越高越好
- puncture 也不是越高就一定錯
- 重點是它們要隨負載與 feasibility 合理變化

### 右上：`Average eMBB Retention Under Overlay`

**代表什麼**

- overlay 發生時，eMBB 平均還保留多少速率

**怎麼解讀**

- 如果 retention 接近 1，表示 overlay 對 eMBB 很友善
- 如果 retention 很低，代表雖然用了 NOMA，但 eMBB 幾乎還是被打爛

### 左下：`Average eMBB Loss Per Puncture Action`

**代表什麼**

- 每次 puncturing 對 eMBB 帶來的平均損失

### 右下：`Average eMBB Loss Per Overlay Action`

**代表什麼**

- 每次 overlay 對 eMBB 帶來的平均損失

**這兩張一起看**

- 如果 overlay loss 顯著低於 puncture loss，表示 overlay 確實有保護 eMBB 的價值

---

## 3. [fairness_load_analysis.png](/d:/URLLC_eMBB_Coexisting/Greedy/results/fairness_load_analysis.png)

這張是公平性和 load balance 的觀察圖。

### 左上：`Jain's Fairness Index`

**代表什麼**

- eMBB user 間速率分配是否均衡

**怎麼解讀**

- 越接近 1 越公平
- 如果 aggregate throughput 還高但 fairness 很差，表示只有少數 user 在吃資源

### 右上：`Cell-edge eMBB Served Ratio`

**代表什麼**

- 邊緣 user 被服務到的比例

**為什麼重要**

- 這很能反映 greedy 是否太偏向好通道 user

### 左下：`Per-UAV Associated Load Imbalance`

**代表什麼**

- 三台 UAV 關聯 user 數的不平衡程度

### 右下：`Per-UAV Scheduled URLLC Imbalance`

**代表什麼**

- 三台 UAV 真正被排進去的 URLLC packet 數是否失衡

**這兩張一起看**

- 如果關聯本來就不平衡，那排程失衡通常也會跟著出現

---

## 4. [slot_mode_action_summary.png](/d:/URLLC_eMBB_Coexisting/Greedy/results/slot_mode_action_summary.png)

這張是 slot-level 動態摘要。

### 四組 bar

- `URLLC arrivals`
- `Admitted URLLC`
- `Overlay count`
- `Puncture count`

### 一條線

- `Slot eMBB throughput`

**這張圖最重要的用法**

直接回答：

- arrival 高時是不是 puncture 變多
- overlay 是否只在某些 slot 特別有用
- throughput 掉時，到底是 overlay 還是 puncture 在主導

---

## 5. [per_uav_performance_decomposition.png](/d:/URLLC_eMBB_Coexisting/Greedy/results/per_uav_performance_decomposition.png)

這張是 per-UAV 分解圖，通常選低、中、高三個代表 density。

每一列代表一個 density。

### 左邊子圖

每台 UAV 有兩根 bar：

- 左 bar：association load
  - associated eMBB
  - associated URLLC
- 右 bar：scheduled load
  - scheduled eMBB
  - admitted URLLC

**怎麼解讀**

- 可以看 association 和 actual scheduling 是否一致
- 如果某台 UAV 關聯很多 user，但實際排程能力有限，表示它是 bottleneck

### 右邊子圖

- overlay count
- puncture count
- eMBB throughput
- 平均 eMBB distance 折線

**怎麼解讀**

- 距離越遠，通常通道越差，排程壓力越大
- 如果某台 UAV 的 puncture 很多、throughput 卻很差，表示它的 coexistence 成本偏高

---

## 6. [overlay_feasibility_diagnostic.png](/d:/URLLC_eMBB_Coexisting/Greedy/results/overlay_feasibility_diagnostic.png)

這張是機制診斷圖。

### 三條線

- `Candidate overlay pairs`
- `Feasible overlay pairs`
- `Selected overlay pairs`

**這張圖的核心意義**

它區分兩件事：

1. overlay 本來就不 feasible
2. overlay feasible，但 greedy 沒有選

**判讀方式**

- `candidate` 很多，但 `feasible` 很少  
  代表物理層條件本來就限制 overlay

- `feasible` 很多，但 `selected` 很少  
  代表 scheduler/mode selection 可能太保守

---

## 7. [retention_loss_distribution.png](/d:/URLLC_eMBB_Coexisting/Greedy/results/retention_loss_distribution.png)

這張不是平均圖，是分布圖。

### 左圖：`eMBB Retention Under Overlay`

**代表什麼**

- overlay 時 eMBB retention 的分布

### 右圖：`eMBB Loss Under Puncturing`

**代表什麼**

- puncture 時 eMBB loss 的分布

**為什麼這張重要**

- 平均值可能太平滑
- boxplot 能看出：
  - 是否存在極端糟糕 case
  - retention 是不是被寫得太固定

---

## 8. [offered_load_curves.png](/d:/URLLC_eMBB_Coexisting/Greedy/results/offered_load_curves.png)

這張用 offered load 當橫軸，不再只看 density。

### 左上：`URLLC Admission vs Offered Load`

**代表什麼**

- offered load 上升時，系統還收得進多少 URLLC packets

### 右上：`Mode Ratio vs Offered Load`

- overlay ratio
- puncture ratio

### 左下：`eMBB Throughput vs Offered Load`

**代表什麼**

- URLLC traffic 壓力提高時，eMBB aggregate throughput 怎麼變

### 右下：`Fairness vs Offered Load`

**代表什麼**

- offered URLLC packet 壓力提高時，eMBB 公平性怎麼變

**為什麼這張比 density 更直接**

- density 同時混了：
  - eMBB user 數增加
  - URLLC user 數增加
  - offered traffic 增加
- offered load 圖則直接對準 URLLC traffic 本身

---

## 9. [resource_utilization_summary.png](/d:/URLLC_eMBB_Coexisting/Greedy/results/resource_utilization_summary.png)

這張是系統層 utilization 圖。

### 左上：`Resource Cell Composition`

stacked area 分成：

- eMBB only
- overlay
- puncture
- idle

### 右上：`Mini-slot Utilization`

**代表什麼**

- 真正非 idle 的 mini-slot cell 比例

### 左下：`Non-idle Resource Fraction`

**代表什麼**

- 所有 UAV-RB-minislot cell 中，有多少不是空的

### 右下：`Idle Resources and RB Utilization`

兩條線：

- idle fraction
- RB utilization

**怎麼看**

- 高 load 下如果 idle fraction 還很高，就很可疑
- 如果 idle 很低但 throughput 還在掉，表示問題不是資源沒用掉，而是 coexistence 成本太高

---

## 10. [urllc_arrival_timeline.png](/d:/URLLC_eMBB_Coexisting/Greedy/results/urllc_arrival_timeline.png)

### 長條

- 每個 slot 的 URLLC packet arrivals

### 折線

- 每個 slot 最後 scheduled 的 URLLC packets

**注意**

這張圖畫的是 packet，不是 user association。

所以：

- 有 8 個 URLLC users
- 不代表每個 slot 只會出現最多 8 個 packet

因為現在已經是 packet-level Poisson model，同一個 user 在同一個 slot 可以有多個 packets。

---

## 11. [urllc_minislot_arrival_map.png](/d:/URLLC_eMBB_Coexisting/Greedy/results/urllc_minislot_arrival_map.png)

### 橫軸

- mini-slot index

### 縱軸

- time slot

### 顏色與數字

- 該 `slot x mini-slot` 上實際被排進去的 URLLC packet 數

**怎麼看**

- 這張圖主要拿來檢查 mini-slot 使用是不是有固定偏好
- 如果某些 mini-slot 永遠都沒有 packet，通常是排程順序有偏差

---

## 12. [performance_timeline.png](/d:/URLLC_eMBB_Coexisting/Greedy/results/performance_timeline.png)

這張是 base case 的 slot-by-slot performance。

### 上圖

- slot-level eMBB throughput

### 中圖

- admitted URLLC reliability

### 下圖

- total Tx power
- eMBB Tx power
- URLLC Tx power

**怎麼看**

- 哪些 slot URLLC power 高
- 哪些 slot eMBB throughput 掉
- 這兩者是否同步

---

## 13. [spatial_grouping_slot9.png](/d:/URLLC_eMBB_Coexisting/Greedy/results/spatial_grouping_slot9.png)

這張是空間上的 association 圖。

### 符號

- circle = eMBB
- diamond = URLLC
- 方框 = UAV

**很重要**

這張只表示：

- 誰屬於哪台 UAV

它不表示：

- 這個 slot 內誰一定有 packet
- 這個 slot 內誰一定被排到資源

---

## 14. [per_uav_load_distribution_slot9.png](/d:/URLLC_eMBB_Coexisting/Greedy/results/per_uav_load_distribution_slot9.png)

### 左圖

- 每台 UAV 關聯了多少 eMBB / URLLC users

### 右圖

- 這個 slot 每台 UAV 真正 scheduled 了多少 URLLC packets

**這張圖的用法**

用來區分：

- 關聯 user 數
- 當下 active packet 數
- 當下實際 scheduled packet 數

---

## 15. [slot_timefreq_slot9.png](/d:/URLLC_eMBB_Coexisting/Greedy/results/slot_timefreq_slot9.png)

這張是最直觀的共存圖。

### 每個 panel

- 一台 UAV 自己的 time-frequency grid

### 背景

- eMBB 連續頻帶

### 棕色 patch

- URLLC
- 其中有些是 overlay
- 有些是 puncture

### 子圖標題

- `eMBB served`
- `URLLC packets`

**注意**

這裡的 `URLLC packets` 是該 slot 真正被排進去的 packet 數，不是關聯 URLLC user 總數。

---

## 16. [single_slot_heatmap_slot9.png](/d:/URLLC_eMBB_Coexisting/Greedy/results/single_slot_heatmap_slot9.png)

這張是 user-RB 佔用矩陣。

**用途**

- 比較像 debug 圖
- 幫你快速確認：
  - 哪些 eMBB user 拿到哪些 RB
  - 哪些 URLLC packets 佔到哪些 RB

---

## B. 目前這批圖共同傳達的事情

綜合來看，這批圖現在在說的是：

1. URLLC hard reliability constraint 有被守住。  
`admitted reliability` 維持高是合理結果。

2. 高負載下，先掉的是 `admission`，不是 `admitted reliability`。  
這和你的 formulation 一致。

3. eMBB aggregate throughput 不一定立刻崩，但 per-user rate、served ratio、fairness 會先惡化。  
這表示系統開始變成「少數 user 撐住總量」。

4. overlay 確實存在，但高 offered load 時 puncturing 通常變主導。  
這並不自動表示錯，而是表示高負載下可行 overlay pair 可能不足。

5. multi-UAV 架構有發揮作用，但不同 UAV 間仍可能負載不均。  
所以 per-UAV decomposition 很重要。

---

## C. 最值得先看的 6 張

如果你現在只想抓主線，建議優先看：

- [power_vs_density.png](/d:/URLLC_eMBB_Coexisting/Greedy/results/power_vs_density.png)
- [mode_tradeoff_analysis.png](/d:/URLLC_eMBB_Coexisting/Greedy/results/mode_tradeoff_analysis.png)
- [overlay_feasibility_diagnostic.png](/d:/URLLC_eMBB_Coexisting/Greedy/results/overlay_feasibility_diagnostic.png)
- [fairness_load_analysis.png](/d:/URLLC_eMBB_Coexisting/Greedy/results/fairness_load_analysis.png)
- [slot_mode_action_summary.png](/d:/URLLC_eMBB_Coexisting/Greedy/results/slot_mode_action_summary.png)
- [slot_timefreq_slot9.png](/d:/URLLC_eMBB_Coexisting/Greedy/results/slot_timefreq_slot9.png)

這 6 張已經足夠支撐一段完整的 results discussion。
