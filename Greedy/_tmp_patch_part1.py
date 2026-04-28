from pathlib import Path

path = Path(r'd:\URLLC_eMBB_Coexisting\sr_mappo\report.py')
text = path.read_text(encoding='utf-8')

old = "from sr_mappo.networks import SRMAPPOActorCritic\nfrom sr_mappo.types import MODE_KEEP, MODE_OVERLAY, MODE_PUNCTURE, HybridAction\nfrom simulation import create_simulation\n"
new = "from sr_mappo.networks import SRMAPPOActorCritic\nfrom sr_mappo.report_dense import (\n    build_low_damage_diagnostics,\n    build_method_dense_summary,\n    dense_series,\n    extract_method_point_audit,\n    finalize_dense_bundle,\n    nearest_replica_proxy,\n    normalize_dense_record,\n    plot_dense_uncertainty_bands,\n    plot_low_damage_admission_diagnostics,\n    plot_marginal_degradation_slopes,\n    plot_matched_admission_diagnostics,\n    plot_method_decomposition_dense,\n    plot_normalized_gap_diagnostics,\n)\nfrom sr_mappo.types import MODE_KEEP, MODE_OVERLAY, MODE_PUNCTURE, HybridAction\nfrom simulation import create_simulation\n"
if old not in text:
    raise SystemExit("import anchor not found")
text = text.replace(old, new, 1)

old = "        'overlay_selected_pairs': float(metrics['overlay_selected_pairs']),\n        'jain_fairness': float(metrics['jain_fairness']),\n"
new = "        'overlay_selected_pairs': float(metrics['overlay_selected_pairs']),\n        'admission_via_overlay_ratio': float(metrics.get('overlay_count', 0.0) / max(metrics.get('scheduled_urllc_users', 0.0), 1.0)),\n        'admission_via_puncture_ratio': float(metrics.get('puncture_count', 0.0) / max(metrics.get('scheduled_urllc_users', 0.0), 1.0)),\n        'puncture_candidate_pruned_by_loss_ceiling_ratio': 0.0,\n        'jain_fairness': float(metrics['jain_fairness']),\n"
if old not in text:
    raise SystemExit("greedy summary anchor not found")
text = text.replace(old, new, 1)

old = "        'mean_executed_power_delta': float(summary.get('mean_executed_power_delta', 0.0)),\n        'trace': trace,\n"
new = "        'mean_executed_power_delta': float(summary.get('mean_executed_power_delta', 0.0)),\n        'admission_via_overlay_ratio': float(summary.get('admission_via_overlay_ratio', 0.0)),\n        'admission_via_puncture_ratio': float(summary.get('admission_via_puncture_ratio', 0.0)),\n        'puncture_candidate_pruned_by_loss_ceiling_ratio': float(summary.get('puncture_candidate_pruned_by_loss_ceiling_ratio', 0.0)),\n        'trace': trace,\n"
if old not in text:
    raise SystemExit("episode summary anchor not found")
text = text.replace(old, new, 1)

old = """        this_score = float(
            load_aware_selection_score(
                load,
                this_throughput_excess,
                this_admission_gap,
                this_puncture_loss_gap,
                this_overlay_retention_gap,
                this_power_ratio,
            )
        )
        this_contribution = float(load_aware_score_mix(load) * this_score)
"""
new = """        low_damage_objective = bool(getattr(cfg.training, 'low_damage_admission_objective', False))
        this_score = float(
            load_aware_selection_score(
                load,
                this_throughput_excess,
                this_admission_gap,
                this_puncture_loss_gap,
                this_overlay_retention_gap,
                this_power_ratio,
                low_damage=low_damage_objective,
            )
        )
        this_contribution = float(load_aware_score_mix(load, low_damage=low_damage_objective) * this_score)
"""
if old not in text:
    raise SystemExit("tradeoff anchor not found")
text = text.replace(old, new, 1)

path.write_text(text, encoding='utf-8')
print("patched report imports/metrics")
