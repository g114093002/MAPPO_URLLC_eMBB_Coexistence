# SR-MAPPO Action Space Mapping

這份文件整理兩件事：

1. PDF/system model 中真正對應的完整決策變數
2. 目前 `sr_mappo` 已經實作到哪裡，還缺哪些 action

本文主要對照下列來源：

- `Greedy/鄭昀曜_20260323_r5_system_model - 複製.pdf`
- [system_model.txt](/d:/URLLC_eMBB_Coexisting/Greedy/system_model.txt#L290)
- [SYSTEM_MODEL_ALIGNMENT.md](/d:/URLLC_eMBB_Coexisting/Greedy/SYSTEM_MODEL_ALIGNMENT.md#L15)
- [SR_MAPPO_FRAMEWORK.md](/d:/URLLC_eMBB_Coexisting/sr_mappo/SR_MAPPO_FRAMEWORK.md#L79)

## 1. 目前 `sr_mappo` 的 action 到底是什麼

目前 `sr_mappo` 的 policy action 只有一個局部 hybrid action：

`a_j = (mode, packet_option, power_delta, embb_owner_option, embb_power_delta)`

對應實作：

- [HybridAction](/d:/URLLC_eMBB_Coexisting/sr_mappo/types.py#L20)
- [SR_MAPPO_FRAMEWORK.md](/d:/URLLC_eMBB_Coexisting/sr_mappo/SR_MAPPO_FRAMEWORK.md#L81)

其中：

- `mode ∈ {KEEP, NOMA, PUNCT}`
- `packet_option ∈ {0,1,...,6}`
- `power_delta ∈ [-1,1]`
- `embb_owner_option ∈ {0,1,...,M}`（在 `force_embb_owner_per_rb=True` 時，會強制選擇有效 owner）
- `embb_power_delta ∈ [-1,1]`（Phase-A planning 時用在每個 UAV-RB）

但要注意，這個 action 不是完整系統的 action，只是：

`在固定 UAV j、固定當前 cell = (k,s)、固定當前 eMBB owner = q 的條件下，替這個 cell 選一個局部 coexistence 行為`

這些「已經被固定、不由 policy 選」的東西是：

- `j`
  由 agent 身分決定
- `k, s`
  由 env 的 `_cell_schedule` 決定
  參考 [env.py](/d:/URLLC_eMBB_Coexisting/sr_mappo/env.py#L69)
- `q`
  在新的版本中，`q` 可由 `embb_owner_option` 控制
  參考 [env.py](/d:/URLLC_eMBB_Coexisting/sr_mappo/env.py#L420)
- 可選 packet 集合
  只來自目前 cell 可見、且被裁剪後的 top-k candidates
  參考 [env.py](/d:/URLLC_eMBB_Coexisting/sr_mappo/env.py#L425) 和 [env.py](/d:/URLLC_eMBB_Coexisting/sr_mappo/env.py#L624)

所以目前 policy 真正控制的，只是：

- 這一格要不要動
- 如果要動，是 `NOMA` 還是 `PUNCT`
- 在目前候選 packet 中選哪一個
- 針對最低可行功率做多大幅度的微調
- eMBB baseline owner 的選擇（每個 RB）
- eMBB baseline power scale 的選擇（每個 UAV-RB）

## 2. PDF 對應的完整 action space

如果按照 system model 的決策變數來看，完整 action space 應該不是單一 `HybridAction`，而是一組分層決策：

### Layer A. Association

- `\phi_{i,j}^t`
  user-UAV association indicator

意思是：

- 使用者 `i` 在 slot `t` 是否關聯到 UAV `j`

對應說明：

- [SYSTEM_MODEL_ALIGNMENT.md](/d:/URLLC_eMBB_Coexisting/Greedy/SYSTEM_MODEL_ALIGNMENT.md#L15)

### Layer B. eMBB baseline allocation

- `\alpha_{q,j,k}^{E,t}`
  eMBB RB allocation indicator

意思是：

- eMBB user `q` 在 slot `t` 是否使用 UAV `j` 的 RB `k`

對應說明：

- [system_model.txt](/d:/URLLC_eMBB_Coexisting/Greedy/system_model.txt#L574)
- [SYSTEM_MODEL_ALIGNMENT.md](/d:/URLLC_eMBB_Coexisting/Greedy/SYSTEM_MODEL_ALIGNMENT.md#L22)

### Layer C. Coexistence placement

- `\rho_{q,z,j,k,s}^t`
  superposition indicator
- `\varpi_{q,z,j,k,s}^t`
  puncturing indicator

意思是：

- URLLC user `z` 是否在 slot `t` 的 `(j,k,s)` 和 eMBB user `q` 做 superposition
- 或者是否在 slot `t` 的 `(j,k,s)` 對 eMBB user `q` 做 puncturing

對應說明：

- [system_model.txt](/d:/URLLC_eMBB_Coexisting/Greedy/system_model.txt#L290)
- [system_model.txt](/d:/URLLC_eMBB_Coexisting/Greedy/system_model.txt#L318)
- [SYSTEM_MODEL_ALIGNMENT.md](/d:/URLLC_eMBB_Coexisting/Greedy/SYSTEM_MODEL_ALIGNMENT.md#L23)
- [SYSTEM_MODEL_ALIGNMENT.md](/d:/URLLC_eMBB_Coexisting/Greedy/SYSTEM_MODEL_ALIGNMENT.md#L24)

### Layer D. Power control

- `p_q^t`
  eMBB transmit power
- `p_z^{s,t}`
  URLLC transmit power

意思是：

- eMBB user `q` 在 slot `t` 的發射功率
- URLLC user `z` 在 slot `t` minislot `s` 的發射功率

對應說明：

- [system_model.txt](/d:/URLLC_eMBB_Coexisting/Greedy/system_model.txt#L302)
- [system_model.txt](/d:/URLLC_eMBB_Coexisting/Greedy/system_model.txt#L306)
- [SYSTEM_MODEL_ALIGNMENT.md](/d:/URLLC_eMBB_Coexisting/Greedy/SYSTEM_MODEL_ALIGNMENT.md#L20)
- [SYSTEM_MODEL_ALIGNMENT.md](/d:/URLLC_eMBB_Coexisting/Greedy/SYSTEM_MODEL_ALIGNMENT.md#L21)

## 3. 對照表：完整 action vs 目前 `sr_mappo`

| 決策層 | PDF 變數 | 意義 | 目前 `sr_mappo` 有沒有直接控制 | 現在是怎麼被決定的 |
| --- | --- | --- | --- | --- |
| Association | `\phi_{i,j}^t` | UE 掛到哪台 UAV | 沒有 | env reset 時先固定 topology association |
| eMBB allocation | `\alpha_{q,j,k}^{E,t}` | eMBB user 在哪個 UAV/RB 上 | 有（新增） | `embb_owner_option` 決定 |
| eMBB power | `p_q^t` | eMBB 發射功率 | 有（新增） | `embb_power_delta` 先決定每個 UAV-RB 的 scale，再以使用者占用的 RB 平均成 `p_q^t` |
| Coexistence mode | `\rho / \varpi` | superposition 或 puncturing | 只有局部版本 | policy 只在固定 `(j,k,s,q)` 下選 `mode` |
| URLLC placement | `(z,j,k,s)` 的選擇 | packet 要放到哪個 UAV/RB/minislot | 只有局部版本 | `j` 由 agent 固定，`k,s` 由 env 掃描固定，`z` 只在候選內選 |
| URLLC power | `p_z^{s,t}` | URLLC 發射功率 | 只有局部版本 | policy 只能輸出 `power_delta` 微調 required power |

## 4. 目前最明確缺掉的 action

下面這些是「從完整系統模型角度看，現在 `sr_mappo` 還沒有真正控制到」的 action。

### 4.1 `association action`

缺少：

- 重新選擇 user-UAV 關聯的 action

目前狀態：

- 使用者先被綁到某台 UAV
- policy 不會改變 association

影響：

- policy 沒辦法主動把可能更適合 overlay 的 `eMBB-URLLC` 組合搬到同一台 UAV

### 4.2 `eMBB RB allocation action`

缺少：

- 選 `\alpha_{q,j,k}^{E,t}` 的 action

目前狀態：

- eMBB anchor 先由 greedy baseline 固定
- policy 只能在既有 eMBB anchor 上做 coexistence

影響：

- 如果原始 eMBB baseline 本來就不利於 overlay，policy 也只能在壞 anchor 上做局部修補

### 4.3 `eMBB power action`

缺少：

- 直接調整 eMBB 發射功率 `p_q^t`

目前狀態：

- policy 完全不控制 eMBB power

影響：

- policy 沒辦法主動創造更有利於 SIC / overlay 的 power separation

### 4.4 `URLLC placement action over (j,k,s)`

缺少：

- 直接決定 packet `z` 要放到哪個 UAV `j`
- 直接決定放在哪個 RB `k`
- 直接決定放在哪個 minislot `s`

目前狀態：

- `j` 被 agent 身分固定
- `k,s` 被 env 順序固定
- policy 只是在該 cell 的候選 packet 中選一個

影響：

- policy 不是在做完整 placement
- 它只是對每個 cell 做局部 accept / mode / power 決策

### 4.5 `absolute URLLC power action`

缺少：

- 直接輸出絕對功率 `p_z^{s,t}` 或 `p_{z,j,k,s}^t`

目前狀態：

- 只輸出 `power_delta`
- 這個 delta 只是繞著 feasibility-required power 做小幅調整

影響：

- 功率控制自由度比完整 system model 小很多

### 4.6 `explicit admit / defer / drop action`

這個不是 PDF 裡最核心的論文符號，但如果從 RL 設計角度看，它其實應該被顯式表示。

目前狀態：

- 只有 `KEEP`
- `KEEP` 比較像「這個 cell 不動」
- 不是「對這個 packet 明確 defer」
- 也不是「對這個 packet 明確 reject」

建議補成：

- `ADMIT_NOW`
- `DEFER`
- `DROP`

影響：

- 這樣 admission control 會變成 policy 真正可控的部分，而不只是被動地等 terminal penalty

## 5. 你現在最容易誤以為已經有、其實還沒有的 action

### `packet_option` 不是完整的 packet scheduling action

它只是在：

- 目前 cell
- 目前可見 packet
- 目前 top-k candidate 子集

裡面選一個 packet。

它不是：

- 對所有活躍 packet 做全域排序
- 也不是對所有 `(j,k,s)` 做全域 placement

### `mode` 不是完整的 `\rho / \varpi` tensor action

它只是在目前 cell 回答：

- 這一格如果要放 packet，是 `NOMA` 還是 `PUNCT`

但完整 system model 的 action 是：

- 對每個 `(q,z,j,k,s,t)` 決定 `rho` 或 `varpi`

### `power_delta` 不是完整的 power-control action

它不是直接決定：

- `p_q^t`
- `p_z^{s,t}`

而只是：

- 對當前已算好的最低可行功率做微調

## 6. 如果要定義「完整 RL action space」，建議怎麼寫

### 6.1 論文級完整 action

如果要完全對應 system model，可以寫成：

`a_t = { \phi^t, \alpha^{E,t}, \rho^t, \varpi^t, p_E^t, p_U^t }`

也就是：

- association
- eMBB RB allocation
- superposition tensor
- puncturing tensor
- eMBB power
- URLLC power

這是最完整，但也最難訓練。

### 6.2 對目前 `sr_mappo` 最合理的擴充版

如果你不想一次把問題做得太大，較合理的擴充順序是：

1. 先把 `URLLC placement` 從固定 `(j,k,s)` 改成可選
2. 再把 `absolute URLLC power` 加進來
3. 再加入 `explicit admit/defer/drop`
4. 最後才把 `eMBB RB allocation` 和 `eMBB power` 放進 RL

可以先定義成：

`a_t^{phaseA+} = (packet_id, j, k, s, mode, p_u, admit_state)`

其中：

- `packet_id`
- `j`
- `k`
- `s`
- `mode ∈ {overlay, puncture}`
- `p_u`
- `admit_state ∈ {now, defer, drop}`

這會比現在的

`(mode, packet_option, power_delta) @ fixed (j,k,s,q)`

完整很多，但還沒有把整個 eMBB baseline 一起交給 RL。

## 7. 我對目前 action 缺口的結論

如果只用一句話總結：

目前 `sr_mappo` 並不是在解 PDF 的完整資源配置問題，而是在解：

- 固定 association
- 固定 eMBB baseline
- 固定 cell 順序
- 固定 eMBB owner

之後的一個局部 coexistence 子問題。

所以現在真正缺掉的 action，核心就是 5 類：

- association
- eMBB RB allocation
- eMBB power
- 全域 URLLC placement `(j,k,s)`
- 絕對 URLLC power / admission-control

## 8. 最值得優先加的 action

如果目標是讓 `MAPPO` 有更大機會真正拉開 Greedy，而不是只把 action space 做大，我建議優先順序是：

1. `URLLC placement action`
   讓 policy 能選 `(j,k,s)`
2. `absolute URLLC power`
   不要只剩 `power_delta`
3. `explicit admit / defer / drop`
   讓 admission 變成真正的 policy decision
4. `eMBB RB allocation`
   讓 overlay 機會能從 anchor 層就被創造出來
5. `association`
   這是最完整但也最重的一層

如果你的目標是「先做最有感的一刀」，那第一個該加的不是更多 `mode`，而是：

`讓 policy 不再被固定在當前 cell 上，而能主動選擇 URLLC 要放去哪個 (UAV, RB, minislot)`
