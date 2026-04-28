# UE Capacity Estimate by Traffic Mix (eMBB:URLLC = 10:0 → 0:10)

This note provides a **simple, report-ready upper-bound estimate** of how many
UEs per UAV can be supported under different **traffic mix ratios**.

It is meant as a **sanity-check capacity table**, not as a precise simulator output.
The values below are **idealized upper bounds** based on the current system parameters.

---

## 1) Assumptions (Current Config)

From the current configuration:

- Total bandwidth per UAV: **5.76 MHz**
- eMBB target spectral efficiency: **2 bps/Hz**
- Minimum per‑eMBB target rate: **2 Mbps**

Thus, the **ideal eMBB-only throughput ceiling per UAV** is:

```
R_max ≈ 5.76e6 Hz × 2 bps/Hz = 11.52 Mbps
```

If each eMBB user requires **2 Mbps**, then the **maximum number of eMBB users per UAV** is:

```
N_eMBB,max ≈ 11.52 / 2 = 5.76 users
```

This is the **capacity anchor** used for the table below.

---

## 2) How the Table Is Computed

Let:

- `r_e` = eMBB ratio in the mix (e.g., 6:4 → r_e = 0.6)
- `N_eMBB,max` = 5.76 (from above)

Then the **total UE capacity per UAV** is approximated by:

```
N_total ≈ N_eMBB,max / r_e
```

This gives an **upper bound** for total UEs (eMBB + URLLC).

---

## 3) Capacity Table (10:0 → 0:10)

All values are **per UAV**, rounded down to a practical integer.

```
Mix (eMBB:URLLC)   eMBB ratio   Total UE capacity per UAV
---------------------------------------------------------
10 : 0             1.00         5
 9 : 1             0.90         6
 8 : 2             0.80         7
 7 : 3             0.70         8
 6 : 4             0.60         9
 5 : 5             0.50         11
 4 : 6             0.40         14
 3 : 7             0.30         19
 2 : 8             0.20         28
 1 : 9             0.10         57
 0 : 10            0.00         undefined
```

---

## 4) Important Caveats

1) **This is an upper bound**  
   It assumes full utilization and ignores feasibility losses from:
   - URLLC reliability constraints
   - overlay/puncture inefficiency
   - inter‑UAV interference
   - collision and scheduling contention

2) **When eMBB ratio is very small (e.g., 1:9)**  
   The formula yields large UE counts because it is anchored on the eMBB minimum‑rate constraint.
   In practice, URLLC reliability and latency constraints will dominate,
   so the actual feasible UE count will be much lower.

3) **The 0:10 case is undefined**  
   Because the formula is anchored on eMBB capacity.
   If you want a pure URLLC capacity upper bound,
   we need to derive it based on URLLC packet size and reliability constraints instead.

---

## 5) Recommended Practical Interpretation

- Use this table only as a **theoretical capacity ceiling**.
- Actual feasible loads should be measured via simulation under:
  - URLLC arrival process
  - reliability constraint
  - power limits
  - SIC constraints

If you want, I can add:

- a **conservative lower‑bound** table,
- a **URLLC-only capacity upper bound**, and
- a **simulation‑based empirical table** alongside this estimate.
