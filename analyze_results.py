import json
from pathlib import Path

report_file = Path('sr_mappo/results/sr_mappo_report_metrics.json')
with open(report_file) as f:
    data = json.load(f)

print('=== CRITICAL ANALYSIS ===')
print()
print('1. MAPPO Evaluation Status:')
print(f'   sr_mappo dict empty: {len(data.get("sr_mappo", {})) == 0}')
print()

print('2. Greedy Baseline Metrics (25.0 load):')
greedy = data.get('selected_baseline', {})
if greedy and 'urllc_admission' in greedy:
    g_admit = float(greedy['urllc_admission'][0]) if isinstance(greedy['urllc_admission'], list) else float(greedy['urllc_admission'])
    g_embb = float(greedy['embb_rate'][0]) / 1e6 if isinstance(greedy['embb_rate'], list) else float(greedy['embb_rate']) / 1e6
    g_power = float(greedy['total_power'][0]) if isinstance(greedy['total_power'], list) else float(greedy['total_power'])
    g_embb_only = float(greedy['embb_only_fraction'][0]) if isinstance(greedy['embb_only_fraction'], list) else float(greedy['embb_only_fraction'])
    
    print(f'   Admission: {g_admit:.4f}')
    print(f'   eMBB Rate: {g_embb:.2f} Mbps')
    print(f'   Total Power: {g_power:.4f}')
    print(f'   eMBB-only Fraction: {g_embb_only:.4f}')
print()

print('3. Checkpoint Metadata (iter 200):')
ckpt = data.get('checkpoint_meta', {})
print(f'   Phase-A power runtime enabled: {ckpt.get("phase_a_embb_power_runtime_enabled")}')
print(f'   Phase-A power changed ratio: {ckpt.get("phase_a_embb_power_changed_ratio"):.4f}')
print(f'   Phase-A power mean raw delta: {ckpt.get("phase_a_embb_power_mean_raw_delta"):.4f}')
print(f'   Phase-A power mean executed delta: {ckpt.get("phase_a_embb_power_mean_executed_delta"):.4f}')
print()

print('4. ROOT CAUSE HYPOTHESIS:')
print('   Problem 1: MAPPO evaluation data empty')
print('     - evaluate_dual_selection() likely failed silently')
print('     - Policy may not have loaded properly')
print('     - Or environment initialization failed')
print()
print('   Problem 2: Even with checkpoint metadata, policy shows:')
print('     - Only 2.5% phase-A power changes')
print('     - Policy essentially KEEPS actions (does nothing)')
print('     - Result: Identical to greedy (which also has 97% no-op)')
print()
print('   Problem 3: No admission improvement despite:')
print('     - Shield disabled')
print('     - All training enabled')
print('     - admission_quota_pressure_weight = 0.15')
print()
print('   This suggests:')
print('     1. Pure PPO convergence to local optimum (doing nothing)')
print('     2. Admission reward signal insufficient vs throughput signal')
print('     3. Phase-A action space too constrained compared to phase-0')
print('     4. Only 200 iterations - possibly not enough training')
