from pathlib import Path

path = Path(r'd:\URLLC_eMBB_Coexisting\sr_mappo\report.py')
text = path.read_text(encoding='utf-8')

old = "    tradeoff_metrics = build_load_tradeoff_diagnostics(rl_metrics, greedy_metrics, report_cfg)\n"
new = """    tradeoff_metrics = build_load_tradeoff_diagnostics(rl_metrics, greedy_metrics, report_cfg)\n    dense_loads = [float(load) for load in (loads if fast else getattr(report_cfg.training, 'dense_eval_loads', loads))]\n    eval_replicas = max(3, int(getattr(report_cfg.training, 'eval_replicas', 5))) if not fast else 3\n    dense_bundle = run_dense_method_bundle(dense_loads, eval_replicas, checkpoint, checkpoint_reason, report_cfg)\n    method_point_audit = extract_method_point_audit(dense_bundle)\n    matched_proxy = nearest_replica_proxy(dense_bundle)\n    low_damage_metrics = build_low_damage_diagnostics(dense_bundle['rl'], dense_bundle['original_greedy_normal_v2'])\n"""
if old not in text:
    raise SystemExit("tradeoff insertion anchor not found")
text = text.replace(old, new, 1)

old = """        plot_upper_bounds_and_frontier(
            greedy_bundle['original'],
            greedy_bundle.get('original_greedy_normal_v1'),
            greedy_bundle.get('original_greedy_normal_v2'),
            greedy_bundle['matched_fixed_embb'],
            greedy_bundle.get('channel_only_greedy'),
            rl_metrics,
            embb_only_ceiling_metrics,
            throughput_oracle_metrics,
            frontier_bundle,
        ),
        plot_load_tradeoff_diagnostics(tradeoff_metrics),
    ]
"""
new = """        plot_upper_bounds_and_frontier(
            greedy_bundle['original'],
            greedy_bundle.get('original_greedy_normal_v1'),
            greedy_bundle.get('original_greedy_normal_v2'),
            greedy_bundle['matched_fixed_embb'],
            greedy_bundle.get('channel_only_greedy'),
            rl_metrics,
            embb_only_ceiling_metrics,
            throughput_oracle_metrics,
            frontier_bundle,
        ),
        plot_load_tradeoff_diagnostics(tradeoff_metrics),
        plot_low_damage_admission_diagnostics(low_damage_metrics, RESULTS_DIR, _style),
        plot_dense_uncertainty_bands(dense_bundle, RESULTS_DIR, _style, _style_power_axis, _top_axis_lambda),
        plot_normalized_gap_diagnostics(dense_bundle, RESULTS_DIR, _style, _top_axis_lambda),
        plot_matched_admission_diagnostics(matched_proxy, RESULTS_DIR, _style),
        plot_method_decomposition_dense(dense_bundle, RESULTS_DIR, _style, _top_axis_lambda),
        plot_marginal_degradation_slopes(dense_bundle, RESULTS_DIR, _style, _style_power_axis),
    ]
"""
if old not in text:
    raise SystemExit("output path block not found")
text = text.replace(old, new, 1)

old = """        'load_tradeoff_diagnostics': tradeoff_metrics,
        'frozen_greedy_metrics_path': report_cfg.training.frozen_greedy_metrics_path,
        'history_length': len(history),
    })
"""
new = """        'load_tradeoff_diagnostics': tradeoff_metrics,
        'dense_loads': dense_loads,
        'eval_replicas': eval_replicas,
        'dense_method_bundle': dense_bundle,
        'method_point_audit': method_point_audit,
        'matched_admission_proxy': matched_proxy,
        'low_damage_admission_diagnostics': low_damage_metrics,
        'evaluation_protocol': {
            'checkpoint': str(checkpoint),
            'checkpoint_reason': checkpoint_reason,
            'selection_rule': str(getattr(report_cfg.training, 'selection_mode', 'dual_metric')),
            'baseline_mode': greedy_baseline_mode,
            'coarse_sweep_loads': [float(load) for load in loads],
            'dense_sweep_loads': dense_loads,
            'num_eval_seeds': int(eval_replicas),
            'selection_admission_floor': float(getattr(report_cfg.training, 'selection_admission_floor', 0.0) or 0.0),
            'selection_admission_floor_by_load': dict(getattr(report_cfg.training, 'selection_admission_floor_by_load', {})),
            'selection_power_ratio_ceiling_by_load': dict(getattr(report_cfg.training, 'selection_power_ratio_ceiling_by_load', {})),
            'matched_admission_proxy_used': True,
        },
        'frozen_greedy_metrics_path': report_cfg.training.frozen_greedy_metrics_path,
        'history_length': len(history),
    })
"""
if old not in text:
    raise SystemExit("metrics block not found")
text = text.replace(old, new, 1)

path.write_text(text, encoding='utf-8')
print("patched generate_report")
