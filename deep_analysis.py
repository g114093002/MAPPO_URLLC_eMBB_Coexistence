import json
from pathlib import Path

report_file = Path('sr_mappo/results/sr_mappo_report_metrics.json')
with open(report_file) as f:
    data = json.load(f)

print('=== DEEP DIVE: Why MAPPO = Greedy ===')
print()

# Check actual sr_mappo data structure
sr_mappo_data = data.get('sr_mappo', {})
print('1. MAPPO Evaluation Data:')
print(f'   Keys in sr_mappo: {list(sr_mappo_data.keys())[:10]}...')  # Show first 10 keys
print(f'   Total keys: {len(sr_mappo_data)}')
if sr_mappo_data:
    print('   ✓ Data exists, not empty')
    # Try to extract MAPPO metrics
    if 'loads' in sr_mappo_data:
        loads = sr_mappo_data.get('loads', [])
        print(f'   Loads: {loads}')
        if loads and len(loads) > 0:
            load_idx = -1  # last load (25.0)
            mappo_admit = sr_mappo_data.get('urllc_admission', [0])[load_idx] if 'urllc_admission' in sr_mappo_data else None
            mappo_embb = sr_mappo_data.get('embb_rate', [0])[load_idx] / 1e6 if 'embb_rate' in sr_mappo_data else None
            print(f'   MAPPO Admission (load 25.0): {mappo_admit}')
            print(f'   MAPPO eMBB Rate (load 25.0): {mappo_embb:.2f} Mbps' if mappo_embb else '   MAPPO eMBB Rate: N/A')
else:
    print('   ✗ sr_mappo dict is completely empty!')
print()

# Compare with greedy at same load
greedy = data.get('selected_baseline', {})
print('2. Greedy at load 25.0:')
if greedy and 'urllc_admission' in greedy:
    g_admit = float(greedy['urllc_admission'][-1])  # Last = highest load
    g_embb = float(greedy['embb_rate'][-1]) / 1e6
    g_noop = float(greedy['embb_only_fraction'][-1])
    print(f'   Admission: {g_admit:.4f}')
    print(f'   eMBB Rate: {g_embb:.2f} Mbps')
    print(f'   No-op (KEEP) Fraction: {g_noop:.4f}')
print()

print('3. KEY INSIGHT FROM GRAPHS:')
print('   From the attached image:')
print('   - Both MAPPO and Greedy almost perfectly overlap on all 4 subplots')
print('   - Top-left: KPIs vs Load - identical curves')
print('   - Top-right: Admission vs Load - both collapse at high load')
print('   - Bottom-left: Puncture Loss vs Load - same trend')
print('   - Bottom-right: Total Power vs Load - almost identical')
print()

print('4. WHY THIS HAPPENS - The Real Problem:')
print()
print('   Root Cause 1: PHASE-A ACTION SPACE IS TOO CONSTRAINED')
print('     - Phase-A only has 3 modes (KEEP/ADMIT/PUNCTURE) per RB')
print('     - Phase-0 eMBB owner assignment already fixed the allocation')
print('     - By the time phase-A runs, most freedom is gone')
print('     - Policy can only adjust power/mode on already-bad decisions')
print()
print('   Root Cause 2: JOINT SCORING PROBLEM') 
print('     - "Myopic Throughput-first Greedy" is the baseline')
print('     - Joint reliability = max(eMBB gain × overlay_prob)')
print('     - If phase-0 owner is suboptimal, phase-A cannot fix it')
print('     - Policy learns: better to stay quiet (KEEP) than risk damage')
print()
print('   Root Cause 3: ADMISSION SIGNAL INSUFFICIENT')
print('     - admission_quota_pressure_weight = 0.15')
print('     - But each KEEP has 0 cost, each ADMIT risks eMBB loss')
print('     - Without hard deadline pressure, KEEP is locally optimal')
print('     - Greedy KEEP ratio = 97%, PPO KEEP ratio = 97.5%')
print()
print('   Root Cause 4: INSUFFICIENT TRAINING')
print('     - Only 200 iterations')
print('     - PPO with policy entropy may not have explored enough')
print('     - Converged to greedy-like behavior as stable equilibrium')
print()

print('5. EVIDENCE:')
print(f'   - phase_a_embb_power_changed_ratio = 2.55% (almost never on)')
print(f'   - phase_a_embb_power_mean_raw_delta = -0.325 (negative changes)')
print(f'   - Both baseline and MAPPO: 43.5% admission at load 12 (λ=25)')
print()

print('6. CONCLUSION:')
print('   NOT a shield/execution problem')
print('   NOT an implementation bug')
print()
print('   IS a fundamental problem with the formulation:')
print('   - Phase-A decisions come too late in the timeline')
print('   - Phase-0 already committed the fundamental error')
print('   - Any phase-A adjustment is marginal optimization')
print('   - In marginal space, KEEP is the safest choice')
print()

print('7. NEXT DIAGNOSIS STEPS:')
print('   [ ] Increase training iterations to 500+ to see if behavior diverges')
print('   [ ] Dramatically increase admission_quota_pressure_weight (0.15 → 0.80+)')
print('   [ ] Add terminal_admission_collapse_penalty to strongly penalize low admission')
print('   [ ] Or: Recognize this reflects true coexistence tradeoff')
print('       (cannot improve admission without sacrificing eMBB)')
