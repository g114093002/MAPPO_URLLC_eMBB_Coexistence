from pathlib import Path
import textwrap

path = Path(r'd:\URLLC_eMBB_Coexisting\sr_mappo\report.py')
text = path.read_text(encoding='utf-8')

anchor = "\n\ndef run_timeslot_series(\n"
if anchor not in text:
    raise SystemExit("run_timeslot_series anchor not found")

block = textwrap.dedent("""

def run_dense_method_bundle(
    dense_loads: List[float],
    eval_replicas: int,
    checkpoint_path: Path,
    checkpoint_reason: str,
    cfg: SRMAPPOConfig,
) -> Dict[str, Dict]:
    base_sys, base_urllc, base_embb, base_algo, base_sim = _build_main_like_configs()
    checkpoint_cfg = deepcopy(_load_checkpoint_cfg(checkpoint_path))
    checkpoint_cfg.env.include_greedy_reference_in_obs = True
    method_bundle: Dict[str, Dict] = {}
    model = None
    model_cfg = None
    method_keys = [key for key, _label, _color, _marker in dense_series()]
    for method_key in method_keys:
        records_by_load: List[List[Dict[str, float]]] = []
        for load_idx, load in enumerate(dense_loads):
            sys_cfg, urllc_cfg, embb_cfg, algo_cfg, sim_cfg = _configure_density_scenario(
                load, base_sys, base_urllc, base_embb, base_algo, base_sim
            )
            if hasattr(base_sim, 'urllc_user_ratio'):
                sim_cfg.urllc_user_ratio = base_sim.urllc_user_ratio
            env = None
            if method_key in {'rl', 'matched_fixed_embb', 'channel_only_greedy'}:
                env_cfg = deepcopy(checkpoint_cfg if method_key == 'rl' else cfg)
                env_cfg.env.include_greedy_reference_in_obs = True
                env = SRMAPPOPhaseAEnv(sys_cfg, urllc_cfg, embb_cfg, algo_cfg, sim_cfg, env_cfg)
                if method_key == 'rl' and model is None:
                    model_cfg, model = _build_model_for_env(env, checkpoint_path)
            records: List[Dict[str, float]] = []
            seed_base = _report_seed_base(load_idx, model_cfg or checkpoint_cfg) + 6000
            for rep_idx in range(int(eval_replicas)):
                seed = seed_base + rep_idx
                if method_key == 'rl':
                    episode = run_env_episode(env, model, model_cfg, seed=seed, collect_trace=False, use_greedy=False)
                elif method_key == 'matched_fixed_embb':
                    episode = run_env_episode(env, model=None, cfg=cfg, seed=seed, collect_trace=False, use_greedy=True)
                elif method_key == 'channel_only_greedy':
                    episode = run_env_episode(env, model=None, cfg=cfg, seed=seed, collect_trace=False, use_greedy=True, greedy_policy='channel_only')
                elif method_key == 'original':
                    episode = _run_original_greedy_slot(sys_cfg, urllc_cfg, embb_cfg, algo_cfg, sim_cfg, seed=seed, slot_index=rep_idx)
                elif method_key == 'original_greedy_normal_v1':
                    episode = _run_original_greedy_normal_v1_slot(sys_cfg, urllc_cfg, embb_cfg, algo_cfg, sim_cfg, seed=seed, slot_index=rep_idx)
                elif method_key == 'original_greedy_normal_v2':
                    episode = _run_original_greedy_normal_v2_slot(sys_cfg, urllc_cfg, embb_cfg, algo_cfg, sim_cfg, seed=seed, slot_index=rep_idx)
                elif method_key == 'embb_only_ceiling':
                    episode = _run_embb_only_ceiling_slot(sys_cfg, urllc_cfg, embb_cfg, algo_cfg, sim_cfg, seed=seed, slot_index=rep_idx)
                elif method_key == 'throughput_feasible_oracle':
                    episode = _run_throughput_feasible_oracle_slot(sys_cfg, urllc_cfg, embb_cfg, algo_cfg, sim_cfg, seed=seed, slot_index=rep_idx)
                else:
                    raise ValueError(f'Unsupported dense method: {method_key}')
                records.append(normalize_dense_record(episode, sys_cfg.num_uavs))
            records_by_load.append(records)
        method_name = next(label for key, label, _color, _marker in dense_series() if key == method_key)
        method_checkpoint = str(checkpoint_path) if method_key == 'rl' else ''
        method_reason = checkpoint_reason if method_key == 'rl' else 'builtin_baseline'
        method_run_name = cfg.training.run_name if method_key == 'rl' else method_key
        protocol = {
            'selection_rule': str(getattr(cfg.training, 'selection_mode', 'builtin_baseline')) if method_key == 'rl' else 'builtin_baseline',
            'baseline_mode': _greedy_baseline_mode(cfg) if method_key == 'rl' else method_key,
            'dense_sweep': True,
            'num_eval_seeds': int(eval_replicas),
            'admission_floor_applied': bool(getattr(cfg.training, 'selection_admission_floor_by_load', {})) if method_key == 'rl' else False,
            'matched_admission_proxy': True,
        }
        method_bundle[method_key] = build_method_dense_summary(
            method_key,
            method_name,
            dense_loads,
            records_by_load,
            method_checkpoint,
            method_reason,
            method_run_name,
            protocol,
        )
    return finalize_dense_bundle(method_bundle)
""")

text = text.replace(anchor, "\n\n" + block + anchor, 1)
path.write_text(text, encoding='utf-8')
print("inserted run_dense_method_bundle")
