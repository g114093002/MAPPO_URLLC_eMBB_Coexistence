from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import json
import os
import secrets
import sys
from pathlib import Path
from datetime import datetime
from time import perf_counter
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import matplotlib

from sr_mappo.evaluate import _throughput_biased_actions, _myopic_throughput_actions
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import ScalarFormatter
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sr_mappo import _bootstrap  # noqa: F401
from sr_mappo.baseline_catalog import (
    baseline_label as _shared_baseline_label,
    baseline_metadata as _shared_baseline_metadata,
    baseline_narrative as _shared_baseline_narrative,
    normalize_baseline_mode as _shared_normalize_baseline_mode,
)
from sr_mappo.compare import _build_main_like_configs, _configure_density_scenario
from sr_mappo.config import SRMAPPOConfig, cfg_from_dict
from sr_mappo.env import SRMAPPOPhaseAEnv
from sr_mappo.experiments import EXPERIMENT_CHOICES, apply_experiment_preset, experiment_label
from sr_mappo.load_aware import load_aware_score_mix, load_aware_selection_score, selection_floor_for_load
from sr_mappo.networks import SRMAPPOActorCritic
from sr_mappo.report_dense import (
    build_low_damage_diagnostics,
    build_method_dense_summary,
    dense_series,
    extract_method_point_audit,
    finalize_dense_bundle,
    nearest_replica_proxy,
    normalize_dense_record,
    plot_dense_uncertainty_bands,
    plot_low_damage_admission_diagnostics,
    plot_marginal_degradation_slopes,
    plot_matched_admission_diagnostics,
    plot_method_decomposition_dense,
    plot_normalized_gap_diagnostics,
)
from sr_mappo.types import MODE_KEEP, MODE_OVERLAY, MODE_PUNCTURE, HybridAction
from sr_mappo.trainer import _phase_a_embb_power_anchor_targets, phase_a_embb_power_anchor_enabled
from simulation import create_simulation

PACKAGE_DIR = Path(r'd:\URLLC_eMBB_Coexisting\sr_mappo')
PROJECT_ROOT = PACKAGE_DIR.parent

def _resolve_writable_results_dir(preferred: Path) -> Path:
    """Pick a writable results directory for runtime outputs.

    Some environments (Windows ACL / controlled-folder policies) can make the
    package directory non-writable at runtime even though the code itself is
    readable. We prefer `sr_mappo/results` for backward compatibility, but we
    must fall back to a guaranteed-writable location to avoid report crashes.
    """

    candidates = [
        preferred,
        PROJECT_ROOT / 'results',
        PROJECT_ROOT / 'results_sr_mappo',
    ]
    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            probe = candidate / '.write_probe.tmp'
            probe.write_text('ok', encoding='utf-8')
            try:
                probe.unlink()
            except OSError:
                pass
            return candidate
        except OSError:
            continue

    # As a last resort, return the preferred path; callers may still fail, but
    # this keeps the type stable and avoids surprising None.
    return preferred


_RESULTS_DIR_OVERRIDE = os.environ.get("SR_MAPPO_RESULTS_DIR_OVERRIDE", "").strip()
if _RESULTS_DIR_OVERRIDE:
    RESULTS_DIR = _resolve_writable_results_dir(Path(_RESULTS_DIR_OVERRIDE))
else:
    RESULTS_DIR = _resolve_writable_results_dir(PACKAGE_DIR / 'results')
CHECKPOINT_DIR = PROJECT_ROOT / 'checkpoints'
DEFAULT_LOADS = list(SRMAPPOConfig().training.eval_loads)
DEFAULT_EPISODES_PER_LOAD = 20
DIAGNOSTIC_EPISODES_PER_LOAD = 30  # For detailed diagnostic reports with 5x sample size
REPRESENTATIVE_LOAD = 25.0
TIMESLOT_SERIES_LOAD = 25.0
TIMESLOT_SERIES_SLOTS = 51
# Fast mode now sweeps all standard loads (5~25) with minimal episodes per load,
# so plots remain multi-point while runtime stays much lower than full report.
FAST_LOADS = list(DEFAULT_LOADS)
FAST_EPISODES_PER_LOAD = 1
FAST_GREEDY_ONLY_EPISODES_PER_LOAD = 100
FAST_GREEDY_ONLY_VERBOSE_PER_EPISODE = False
REPORT_HEARTBEAT_EVERY_EPISODES = max(1, int(os.environ.get("SR_MAPPO_REPORT_HEARTBEAT_EVERY_EPISODES", "1") or "1"))
REPORT_VERBOSE_VSLOT = str(os.environ.get("SR_MAPPO_REPORT_VERBOSE_VSLOT", "") or "").strip().lower() in {"1", "true", "yes", "on"}
FAST_TIMESLOT_SERIES_LOAD = 20.0
FAST_TIMESLOT_SERIES_SLOTS = 20
MODE_ORDER = [MODE_KEEP, MODE_OVERLAY, MODE_PUNCTURE]
MODE_LABELS = ['KEEP', 'OVERLAY', 'PUNCTURE']
CONDITION_LABELS = ['Overlay feasible exists', 'No feasible overlay']
CURRENT_TOP_LEVEL_REPORT_FILES = {
    '01_core_kpis_vs_load.png',
    '02_mode_diagnostics_vs_load.png',
    '03_fairness_and_uav_vs_load.png',
    '04_training_diagnostics.png',
    '05_slot_timeline_and_activity.png',
    '06_single_slot_mode_maps.png',
    '07_timeslot_kpis_comparison.png',
    '08_timeslot_power_mode_comparison.png',
    '09_timeslot_action_summary.png',
    '10_upper_bounds_and_frontier.png',
    '11_full_mappo_activity_vs_load.png',
    '12_load_tradeoff_diagnostics.png',
    '13_low_damage_admission_diagnostics.png',
    '14_dense_uncertainty_bands.png',
    '15_normalized_gap_diagnostics.png',
    '16_matched_admission_diagnostics.png',
    '17_method_decomposition_dense.png',
    '18_marginal_degradation_slopes.png',
    '19_mode_anchor_debug.png',
    'custom_min_rate_satisfied_count_compare.png',
    'custom_admitted_urllc_packets_compare.png',
    'mode_action_share_compare.png',
    'mode_raw_vs_executed_compare.png',
    'sr_mappo_report_metrics.json',
    'throughput_admission_frontier.json',
    'report_run_manifest.json',
}
_REPORT_TIMING_ENABLED = False
_REPORT_EPISODE_CACHE: Dict[Tuple, Dict] = {}
_REPORT_EPISODE_CACHE_ENABLED = True
_REPORT_PLOT_FALLBACK_WARNINGS: set[str] = set()
_REPORT_RUN_SEED_BASE: Optional[int] = None


def _report_log(message: str) -> None:
    timestamp = datetime.now(ZoneInfo("Asia/Taipei")).isoformat(timespec="seconds")
    print(f"[{timestamp}] [SR-MAPPO][REPORT] {message}", flush=True)


def _report_timing_log(message: str) -> None:
    if not _REPORT_TIMING_ENABLED:
        return
    _report_log(message)


def _report_is_pure_sumrate_policy_name(policy_name: object) -> bool:
    normalized = str(policy_name or "").strip().lower()
    return normalized in {
        "global_sumrate_only",
        "sumrate_only",
        "pure_sumrate",
        "global_tp_only",
    }


def _reset_report_runtime_cache() -> None:
    _REPORT_EPISODE_CACHE.clear()


def _init_report_run_seed_base(cfg: Optional[SRMAPPOConfig] = None) -> int:
    """Pick a fresh seed base per report run.

    Requirements:
    - Different report invocations use different seed streams.
    - Within one report run, all loads share the same episode-indexed seed stream.
    - Different episodes use different seeds (base + ep).
    """
    cfg = cfg or SRMAPPOConfig()
    fixed_seed_env = os.environ.get("SR_MAPPO_REPORT_SEED_BASE", "").strip()
    if fixed_seed_env:
        try:
            return int(fixed_seed_env)
        except ValueError:
            pass
    root = int(getattr(cfg.training, "train_seed", 42))
    # Fresh per-invocation offset from OS entropy.
    offset = int(secrets.randbelow(1_000_000_000))
    return int(root + 1000 + offset)


def _report_episode_cache_key(
    env: SRMAPPOPhaseAEnv,
    cfg: SRMAPPOConfig,
    seed: int,
    collect_trace: bool,
    use_greedy: bool,
    greedy_policy: str,
    cache_tag: str = "",
) -> Tuple:
    return (
        str(getattr(cfg.training, "run_name", "")),
        str(getattr(cfg.training, "experiment_line", "")),
        str(cache_tag or ""),
        str(getattr(cfg.env, "phase", "")),
        str(getattr(cfg.env, "fixed_embb_baseline_policy", "")),
        bool(getattr(cfg.env, "learn_embb_baseline", False)),
        bool(getattr(cfg.shield, "enable_feasibility_shield", False)),
        bool(getattr(cfg.shield, "apply_joint_reliability_rewrite", False)),
        int(seed),
        bool(collect_trace),
        bool(use_greedy),
        str(greedy_policy or "reference").strip().lower(),
        int(env.sys_cfg.num_uavs),
        int(env.sys_cfg.num_subcarriers),
        int(env.sys_cfg.num_minislots),
        int(env.sys_cfg.num_embb_users),
        int(env.sys_cfg.num_urllc_users),
        float(getattr(env.sim_cfg, "urllc_poisson_rate", 0.0)),
    )


def _run_episode_batch_with_representative(episodes_per_load: int, runner) -> Tuple[List[Dict], Dict]:
    episode_count = max(int(episodes_per_load), 1)
    episodes: List[Dict] = []
    representative: Dict = {}
    for episode_idx in range(episode_count):
        collect_trace = episode_idx == 0
        result = runner(episode_idx, collect_trace)
        episodes.append(result)
        if episode_idx == 0:
            representative = deepcopy(result)
    return episodes, representative


def _build_episode_scene_audit(episodes: List[Dict]) -> Dict[str, object]:
    rows: List[Dict[str, object]] = []
    for episode_idx, episode in enumerate(episodes):
        rows.append({
            "episode_index": int(episode_idx),
            "report_episode_seed": int(float(episode.get("report_episode_seed", 0.0) or 0.0)),
            "outer_report_episode_seed": int(float(episode.get("outer_report_episode_seed", episode.get("report_episode_seed", 0.0)) or 0.0)),
            "virtual_slot_reset_seed_first": int(float(episode.get("virtual_slot_reset_seed_first", episode.get("report_episode_seed", 0.0)) or 0.0)),
            "virtual_slots_per_episode": int(float(episode.get("virtual_slots_per_episode", 1.0) or 1.0)),
            "mother_topology_id": str(episode.get("mother_topology_id", "") or ""),
            "mother_topology_seed": float(episode.get("mother_topology_seed", 0.0) or 0.0),
            "same_assoc_hash": str(episode.get("same_assoc_hash", "") or ""),
            "same_channel_hash": str(episode.get("same_channel_hash", "") or ""),
            "same_user_pool_hash": str(episode.get("same_user_pool_hash", "") or ""),
            "mix_user_subset_hash": str(episode.get("mix_user_subset_hash", "") or ""),
            "embb_subset_hash": str(episode.get("embb_subset_hash", "") or ""),
            "same_feasible_graph_hash": str(episode.get("same_feasible_graph_hash", "") or ""),
            "feasible_graph_id": str(episode.get("feasible_graph_id", "") or ""),
            "overlay_graph_hash": str(episode.get("overlay_graph_hash", "") or ""),
            "channel_matrix_hash": str(episode.get("channel_matrix_hash", "") or ""),
            "pathloss_hash": str(episode.get("pathloss_hash", "") or ""),
            "shadowing_hash": str(episode.get("shadowing_hash", "") or ""),
            "sic_order_hash": str(episode.get("sic_order_hash", "") or ""),
            "repair_sequence_hash": str(episode.get("repair_sequence_hash", "") or ""),
            "active_packets": float(episode.get("active_packets", 0.0) or 0.0),
            "scheduled_packets": float(episode.get("scheduled_packets", 0.0) or 0.0),
            "urllc_admission": float(episode.get("urllc_admission", 0.0) or 0.0),
        })

    def _unique_count(key: str) -> int:
        values = [str(row.get(key, "") or "") for row in rows]
        return int(len({value for value in values if value}))

    return {
        "episodes": rows,
        "consistency": {
            "unique_report_episode_seed_count": int(len({int(row["report_episode_seed"]) for row in rows})),
            "unique_outer_report_episode_seed_count": int(len({int(row["outer_report_episode_seed"]) for row in rows})),
            "unique_virtual_slot_reset_seed_first_count": int(len({int(row["virtual_slot_reset_seed_first"]) for row in rows})),
            "unique_mother_topology_id_count": _unique_count("mother_topology_id"),
            "unique_mother_topology_seed_count": int(
                len({float(row["mother_topology_seed"]) for row in rows if float(row["mother_topology_seed"]) != 0.0})
            ),
            "unique_same_assoc_hash_count": _unique_count("same_assoc_hash"),
            "unique_same_channel_hash_count": _unique_count("same_channel_hash"),
            "unique_same_user_pool_hash_count": _unique_count("same_user_pool_hash"),
            "unique_mix_user_subset_hash_count": _unique_count("mix_user_subset_hash"),
            "unique_embb_subset_hash_count": _unique_count("embb_subset_hash"),
            "unique_same_feasible_graph_hash_count": _unique_count("same_feasible_graph_hash"),
        },
    }


def _build_pairing_fairness_audit(
    rl_metrics: Dict[str, object],
    baseline_metrics: Dict[str, object],
) -> Dict[str, object]:
    rl_audit = dict(rl_metrics.get("episode_scene_audit", {}) or {})
    baseline_audit = dict(baseline_metrics.get("episode_scene_audit", {}) or {})
    load_keys = sorted(
        set(rl_audit.keys()) | set(baseline_audit.keys()),
        key=lambda key: float(key),
    )
    compare_keys = [
        "report_episode_seed",
        "outer_report_episode_seed",
        "virtual_slot_reset_seed_first",
        "virtual_slots_per_episode",
        "mother_topology_id",
        "mother_topology_seed",
        "same_assoc_hash",
        "same_channel_hash",
        "same_user_pool_hash",
        "mix_user_subset_hash",
        "embb_subset_hash",
        "same_feasible_graph_hash",
        "feasible_graph_id",
        "overlay_graph_hash",
        "channel_matrix_hash",
        "pathloss_hash",
        "shadowing_hash",
        "sic_order_hash",
        "repair_sequence_hash",
        "active_packets",
    ]
    float_keys = {"mother_topology_seed", "active_packets"}
    per_load: Dict[str, object] = {}
    total_pairs = 0
    total_mismatched_pairs = 0
    total_missing_pairs = 0
    mismatch_key_counts = {key: 0 for key in compare_keys}

    for load_key in load_keys:
        rl_eps = list(((rl_audit.get(load_key) or {}).get("episodes", [])) if isinstance(rl_audit.get(load_key), dict) else [])
        baseline_eps = list(((baseline_audit.get(load_key) or {}).get("episodes", [])) if isinstance(baseline_audit.get(load_key), dict) else [])
        episode_count = max(len(rl_eps), len(baseline_eps))
        load_mismatches: List[Dict[str, object]] = []
        missing_episode_indices: List[int] = []
        matched_pairs = 0

        for episode_idx in range(episode_count):
            rl_ep = rl_eps[episode_idx] if episode_idx < len(rl_eps) else None
            baseline_ep = baseline_eps[episode_idx] if episode_idx < len(baseline_eps) else None
            total_pairs += 1
            if rl_ep is None or baseline_ep is None:
                total_missing_pairs += 1
                missing_episode_indices.append(int(episode_idx))
                continue

            mismatched_keys: List[str] = []
            for key in compare_keys:
                rl_value = rl_ep.get(key)
                baseline_value = baseline_ep.get(key)
                if key in float_keys:
                    rl_float = float(rl_value or 0.0)
                    baseline_float = float(baseline_value or 0.0)
                    if abs(rl_float - baseline_float) > 1.0e-9:
                        mismatched_keys.append(key)
                else:
                    if str(rl_value or "") != str(baseline_value or ""):
                        mismatched_keys.append(key)
            if mismatched_keys:
                total_mismatched_pairs += 1
                for key in mismatched_keys:
                    mismatch_key_counts[key] += 1
                load_mismatches.append({
                    "episode_index": int(episode_idx),
                    "mismatched_keys": mismatched_keys,
                    "rl": {key: rl_ep.get(key) for key in compare_keys},
                    "baseline": {key: baseline_ep.get(key) for key in compare_keys},
                })
            else:
                matched_pairs += 1

        per_load[load_key] = {
            "load": float(load_key),
            "paired_episode_count": int(episode_count),
            "matched_episode_count": int(matched_pairs),
            "mismatched_episode_count": int(len(load_mismatches)),
            "missing_episode_count": int(len(missing_episode_indices)),
            "matched_all": bool((not load_mismatches) and (not missing_episode_indices)),
            "missing_episode_indices": missing_episode_indices,
            "mismatches": load_mismatches,
        }

    nonzero_mismatch_key_counts = {key: int(value) for key, value in mismatch_key_counts.items() if int(value) > 0}
    return {
        "paired_all": bool(total_mismatched_pairs == 0 and total_missing_pairs == 0),
        "loads_compared": int(len(load_keys)),
        "total_episode_pairs": int(total_pairs),
        "mismatched_episode_pairs": int(total_mismatched_pairs),
        "missing_episode_pairs": int(total_missing_pairs),
        "mismatch_key_counts": nonzero_mismatch_key_counts,
        "loads": per_load,
    }


def _aggregate_virtual_slot_results(slot_results: List[Dict], virtual_slots: int) -> Dict:
    """Aggregate multiple single-slot episode summaries into one virtual multi-slot episode.

    We keep action space/sequential decision unchanged, and only extend horizon at report-time.
    """
    if not slot_results:
        return {}
    if virtual_slots <= 1 or len(slot_results) == 1:
        return deepcopy(slot_results[0])

    sum_keys = {
        "active_packets",
        "scheduled_packets",
        "overlay_count",
        "puncture_count",
        "virtual_slot_reset_count_per_episode",
    }
    keep_first_keys = {
        "phase",
        "trace",
        "user_positions",
        "uav_positions",
    }

    agg: Dict[str, object] = {}
    all_keys = set()
    for item in slot_results:
        all_keys.update(item.keys())

    for key in all_keys:
        values = [item.get(key) for item in slot_results if key in item]
        if not values:
            continue
        first = values[0]
        if key in keep_first_keys:
            agg[key] = deepcopy(first)
            continue
        if isinstance(first, np.ndarray):
            try:
                arrs = [np.asarray(v, dtype=float) for v in values]
                if all(arr.shape == arrs[0].shape for arr in arrs):
                    agg[key] = np.mean(np.stack(arrs, axis=0), axis=0)
                    continue
            except Exception:
                agg[key] = deepcopy(first)
                continue
        if isinstance(first, (int, float, np.floating, np.integer, bool)):
            vals = np.asarray(values, dtype=float)
            if key in sum_keys:
                agg[key] = float(np.sum(vals))
            else:
                agg[key] = float(np.mean(vals))
            continue
        agg[key] = deepcopy(first)

    # Recompute key KPIs from accumulated arrivals/admissions across virtual slots.
    active_packets = float(agg.get("active_packets", 0.0) or 0.0)
    scheduled_packets = float(agg.get("scheduled_packets", 0.0) or 0.0)
    if active_packets > 0.0:
        agg["urllc_admission"] = float(scheduled_packets / max(active_packets, 1.0e-12))
    else:
        agg["urllc_admission"] = 1.0

    slot_dur = float(agg.get("urllc_slot_duration_s", 1.0e-3) or 1.0e-3)
    pkt_bits = float(agg.get("urllc_packet_bits_mean", 160.0) or 160.0)
    total_bits = float(scheduled_packets * pkt_bits)
    avg_bps = float(total_bits / max(float(virtual_slots) * slot_dur, 1.0e-12))
    agg["urllc_throughput_bps_slot_est"] = avg_bps
    agg["urllc_throughput_mbps_slot_est"] = float(avg_bps / 1.0e6)
    agg["urllc_throughput_bps_est"] = agg["urllc_throughput_bps_slot_est"]
    agg["virtual_slots_per_episode"] = float(virtual_slots)
    return agg


def _run_env_episode_virtual_slots(
    env: SRMAPPOPhaseAEnv,
    model: Optional[SRMAPPOActorCritic],
    cfg: SRMAPPOConfig,
    seed: int,
    collect_trace: bool,
    use_greedy: bool,
    greedy_policy: str,
    cache_tag: str,
    virtual_slots: int,
) -> Dict:
    _MAX_NUMPY_SEED = (2**32) - 1
    if int(virtual_slots) <= 1:
        return run_env_episode(
            env,
            model=model,
            cfg=cfg,
            seed=seed,
            collect_trace=collect_trace,
            use_greedy=use_greedy,
            greedy_policy=greedy_policy,
            cache_tag=cache_tag,
            reuse_static_context=False,
            reset_count_contribution=1.0,
        )

    slot_results: List[Dict] = []
    for slot_idx in range(int(virtual_slots)):
        # Keep deterministic per-slot perturbation while staying within numpy seed bounds.
        slot_seed = int((int(seed) * 1009 + int(slot_idx)) % _MAX_NUMPY_SEED)
        if slot_seed <= 0:
            slot_seed = int(slot_idx + 1)
        if REPORT_VERBOSE_VSLOT:
            _report_log(
                f"[GREEDY][vs={int(virtual_slots)}] slot {int(slot_idx)+1}/{int(virtual_slots)} "
                f"| seed={int(slot_seed)} | reuse_static_context={int(slot_idx > 0)} | cache_tag={cache_tag}"
            )
        slot_t0 = perf_counter()
        slot_result = run_env_episode(
                env,
                model=model,
                cfg=cfg,
                seed=slot_seed,
                collect_trace=(collect_trace and slot_idx == 0),
                use_greedy=use_greedy,
                greedy_policy=greedy_policy,
                cache_tag=f"{cache_tag}|vs{virtual_slots}|s{slot_idx}",
                reuse_static_context=bool(slot_idx > 0),
                reset_count_contribution=(1.0 if slot_idx == 0 else 0.0),
            )
        slot_results.append(slot_result)
        if REPORT_VERBOSE_VSLOT:
            _report_log(
                f"[GREEDY][vs={int(virtual_slots)}] slot {int(slot_idx)+1}/{int(virtual_slots)} done "
                f"| sec={perf_counter()-slot_t0:.3f} | arrivals={float(slot_result.get('active_packets', 0.0)):.2f} "
                f"| admitted={float(slot_result.get('scheduled_packets', 0.0)):.2f}"
            )
    agg = _aggregate_virtual_slot_results(slot_results, int(virtual_slots))
    agg["outer_report_episode_seed"] = float(seed)
    agg["report_episode_seed"] = float(seed)
    agg["virtual_slot_reset_seed_first"] = float(slot_results[0].get("report_episode_seed", seed)) if slot_results else float(seed)
    return agg


@contextmanager
def _temporary_shared_mother_scene_for_episode(
    env: SRMAPPOPhaseAEnv,
    load: float,
    load_idx: int,
    episode_seed: int,
):
    enabled = bool(_env_bool_override("SR_MAPPO_REPORT_SHARED_MOTHER_RESAMPLE_EACH_EPISODE", False))
    if not enabled:
        yield
        return

    base_mother_id = str(os.environ.get("SR_MAPPO_MOTHER_TOPOLOGY_ID", "") or "").strip()
    base_feasible_graph_id = str(os.environ.get("SR_MAPPO_FEASIBLE_GRAPH_ID", "") or "").strip()
    if not base_mother_id:
        base_mother_id = "shared_mother_scene"
    if not base_feasible_graph_id:
        base_feasible_graph_id = base_mother_id

    scene_suffix = (
        f"__shared_ep_scene"
        f"_load{float(load):.12g}"
        f"_idx{int(load_idx)}"
        f"_seed{int(episode_seed)}"
    )
    previous_values = {
        "SR_MAPPO_MOTHER_TOPOLOGY_FREEZE": os.environ.get("SR_MAPPO_MOTHER_TOPOLOGY_FREEZE"),
        "SR_MAPPO_FEASIBLE_GRAPH_FREEZE": os.environ.get("SR_MAPPO_FEASIBLE_GRAPH_FREEZE"),
        "SR_MAPPO_MOTHER_TOPOLOGY_ID": os.environ.get("SR_MAPPO_MOTHER_TOPOLOGY_ID"),
        "SR_MAPPO_FEASIBLE_GRAPH_ID": os.environ.get("SR_MAPPO_FEASIBLE_GRAPH_ID"),
    }
    os.environ["SR_MAPPO_MOTHER_TOPOLOGY_FREEZE"] = "1"
    os.environ["SR_MAPPO_FEASIBLE_GRAPH_FREEZE"] = "1"
    os.environ["SR_MAPPO_MOTHER_TOPOLOGY_ID"] = f"{base_mother_id}{scene_suffix}"
    os.environ["SR_MAPPO_FEASIBLE_GRAPH_ID"] = f"{base_feasible_graph_id}{scene_suffix}"

    # When we intentionally hop to a new shared mother scene per episode, every
    # env-level cache that depends on the frozen scene must be cleared first.
    # Otherwise a later episode can inherit association/channel/subset/baseline
    # artifacts from the previous seed and produce a hybrid scene that matches
    # neither the old nor the new mother seed.
    for attr_name in (
        "_fixed_association_cache",
        "_fixed_channel_gains_cache",
        "_fixed_last_topology_cache",
        "_fixed_nested_user_indices_cache",
        "_phase0_baseline_snapshot_cache",
        "_nested_canonical_ur_order_cache",
        "_nested_canonical_em_order_cache",
    ):
        if hasattr(env, attr_name):
            setattr(env, attr_name, None)
    try:
        yield
    finally:
        for key, old_value in previous_values.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


def _cleanup_stale_report_artifacts() -> List[str]:
    stale_removed: List[str] = []
    if not RESULTS_DIR.exists():
        return stale_removed
    for path in RESULTS_DIR.iterdir():
        if not path.is_file():
            continue
        if path.suffix.lower() != '.png':
            continue
        prefix = path.stem.split('_', 1)[0]
        if prefix.isdigit() and path.name not in CURRENT_TOP_LEVEL_REPORT_FILES:
            path.unlink(missing_ok=True)
            stale_removed.append(path.name)
    return stale_removed


def _write_report_manifest(payload: Dict) -> Path:
    path = RESULTS_DIR / 'report_run_manifest.json'
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding='utf-8')
    return path


def _safe_mean(values, default=0.0):
    if not values:
        return float(default)
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float(default)
    return float(np.mean(finite))


def _series_with_fallback(
    series: Dict,
    loads: np.ndarray,
    preferred_key: str,
    fallback_keys: List[str] | Tuple[str, ...] = (),
    *,
    context: str = "",
    default_value: float = np.nan,
) -> np.ndarray:
    y = np.asarray(series.get(preferred_key, []), dtype=float)
    if y.size == loads.size:
        return y
    for fb in fallback_keys:
        fb_y = np.asarray(series.get(fb, []), dtype=float)
        if fb_y.size == loads.size:
            warn_key = f"{context}|{preferred_key}|{fb}"
            if warn_key not in _REPORT_PLOT_FALLBACK_WARNINGS:
                _REPORT_PLOT_FALLBACK_WARNINGS.add(warn_key)
                _report_log(
                    f"[plot-fallback] {context}: missing/invalid '{preferred_key}', fallback to '{fb}'."
                )
            return fb_y
    warn_key = f"{context}|{preferred_key}|missing"
    if warn_key not in _REPORT_PLOT_FALLBACK_WARNINGS:
        _REPORT_PLOT_FALLBACK_WARNINGS.add(warn_key)
        _report_log(
            f"[plot-missing] {context}: both '{preferred_key}' and fallback keys missing/invalid; using default."
        )
    return np.full_like(loads, float(default_value), dtype=float)


def _report_plot_key_audit(plot_name: str, pairs: List[Tuple[str, np.ndarray]]) -> None:
    for key_name, values in pairs:
        arr = np.asarray(values, dtype=float)
        _report_log(f"[plot-key-audit] {plot_name} | key={key_name} | values={arr.tolist()}")


def _report_overlap_note(
    ax,
    a: np.ndarray,
    b: np.ndarray,
    *,
    text: str = "positive-rate ratio equals service ratio under current semantics",
) -> None:
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    if aa.shape == bb.shape and aa.size > 0 and np.allclose(aa, bb, atol=1.0e-12, rtol=0.0):
        ax.text(
            0.02,
            0.02,
            text,
            transform=ax.transAxes,
            fontsize=8,
            ha='left',
            va='bottom',
            bbox=dict(boxstyle='round', facecolor='#f7f7f7', edgecolor='#cccccc', alpha=0.9),
        )


def _episode_scalar_mean(episodes: List[Dict], key: str, default: float = 0.0) -> float:
    value = _episode_scalar_aggregate(episodes, key, default=default)
    try:
        return float(value)
    except Exception:
        return float(default)


def _episode_scalar_aggregate(episodes: List[Dict], key: str, default=0.0):
    values = [episode.get(key, default) for episode in episodes]
    if not values:
        return default
    sample = None
    for value in values:
        if value is not None:
            sample = value
            break
    if isinstance(sample, str):
        counts: Dict[str, int] = {}
        for value in values:
            label = str(value)
            counts[label] = counts.get(label, 0) + 1
        return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
    return _safe_mean(values, default=default)


def _metric_scalar_mean(value, default: float = 0.0) -> float:
    if value is None:
        return float(default)
    if isinstance(value, (list, tuple, np.ndarray)):
        if len(value) == 0:
            return float(default)
        return _safe_mean(value, default=default)
    try:
        return float(value)
    except Exception:
        return float(default)


def _metric_scalar_any(value, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, (list, tuple, np.ndarray)):
        if len(value) == 0:
            return bool(default)
        try:
            arr = np.asarray(value, dtype=float)
            return bool(np.any(arr > 0.5))
        except Exception:
            return bool(any(bool(item) for item in value))
    try:
        return bool(float(value) > 0.5)
    except Exception:
        return bool(value)


def _base_profile() -> Tuple[float, float, bool]:
    sys_cfg, _urllc_cfg, _embb_cfg, _algo_cfg, sim_cfg = _build_main_like_configs()
    base_embb_per_uav = max(1, int(np.ceil(sys_cfg.num_embb_users / sys_cfg.num_uavs)))
    base_urllc_per_uav = max(1, int(np.ceil(sys_cfg.num_urllc_users / sys_cfg.num_uavs)))
    return (
        float(base_embb_per_uav + base_urllc_per_uav),
        float(sim_cfg.urllc_poisson_rate),
        bool(getattr(sim_cfg, 'fixed_urllc_poisson_rate', False)),
    )


def _resolve_forced_urllc_ratio(report_cfg: SRMAPPOConfig) -> float:
    """Resolve effective URLLC ratio for greedy report experiments.

    Priority:
    1) explicit `env.urllc_user_ratio_override`
    2) known experiment-name hard mapping
    3) no forcing (-1.0)
    """
    forced_ratio = float(getattr(report_cfg.env, "urllc_user_ratio_override", -1.0) or -1.0)
    if forced_ratio >= 0.0:
        return forced_ratio

    exp_line = str(getattr(report_cfg.training, "experiment_line", "") or "").strip().lower()
    if "v8_greedy_share10_debug" in exp_line or "v8_greedy_mix100_debug" in exp_line:
        return 0.0
    if "v8_greedy_mix73_debug" in exp_line:
        return 0.3
    if "v8_greedy_mix55_debug" in exp_line:
        return 0.5
    if "v8_greedy_mix37_debug" in exp_line:
        return 0.7
    if "v8_greedy_mix010_debug" in exp_line:
        return 1.0
    return -1.0


def _apply_forced_urllc_ratio_to_sim(
    sim_cfg: object,
    report_cfg: SRMAPPOConfig,
    *,
    log_prefix: str,
) -> float:
    """Apply the report-time mix override to a simulation config when present.

    Several report paths rebuild `base_sim` from defaults instead of reusing the
    greedy-sweep path directly. Without reapplying the forced mix ratio here,
    MAPPO-side evaluation can silently fall back to the default 7:3-style user
    split while Greedy is evaluated under the requested mix override.
    """
    forced_ratio = _resolve_forced_urllc_ratio(report_cfg)
    if hasattr(sim_cfg, "urllc_user_ratio") and forced_ratio >= 0.0:
        sim_cfg.urllc_user_ratio = float(np.clip(forced_ratio, 0.0, 1.0))
        _report_log(
            f"[{log_prefix}] forcing urllc_user_ratio={float(sim_cfg.urllc_user_ratio):.3f} "
            f"(override={float(getattr(report_cfg.env, 'urllc_user_ratio_override', -1.0)):.3f})"
        )
    return float(forced_ratio)


def _canonical_greedy_mix_experiment_for_ratio(ratio: float) -> str | None:
    targets = {
        0.0: "phase0_joint_full_power_service_interference_repair_v8_greedy_mix100_debug",
        0.3: "phase0_joint_full_power_service_interference_repair_v8_greedy_mix73_debug",
        0.5: "phase0_joint_full_power_service_interference_repair_v8_greedy_mix55_debug",
        0.7: "phase0_joint_full_power_service_interference_repair_v8_greedy_mix37_debug",
        1.0: "phase0_joint_full_power_service_interference_repair_v8_greedy_mix010_debug",
    }
    for key, exp in targets.items():
        if abs(float(ratio) - float(key)) <= 1.0e-9:
            return exp
    return None


def _maybe_realign_greedy_mix_preset(report_cfg: SRMAPPOConfig) -> SRMAPPOConfig:
    disable_realign = str(
        os.environ.get("SR_MAPPO_REPORT_DISABLE_CANONICAL_GREEDY_MIX_REALIGN", "") or ""
    ).strip().lower()
    if disable_realign in {"1", "true", "yes", "on"}:
        _report_log("[OVERRIDE] disable canonical greedy mix preset realign.")
        return report_cfg
    legacy_mix_override = str(os.environ.get("SR_MAPPO_REPORT_LEGACY_MIX_OVERRIDE", "") or "").strip().lower()
    if legacy_mix_override in {"1", "true", "yes", "on"}:
        _report_log("[OVERRIDE] keep legacy greedy mix preset behavior (skip canonical preset realign).")
        return report_cfg
    forced_ratio = _resolve_forced_urllc_ratio(report_cfg)
    if forced_ratio < 0.0:
        return report_cfg
    current_exp = str(getattr(report_cfg.training, "experiment_line", "") or "").strip().lower()
    if "phase0_joint_full_power_service_interference_repair_v8_greedy_" not in current_exp:
        return report_cfg
    target_exp = _canonical_greedy_mix_experiment_for_ratio(forced_ratio)
    if not target_exp or current_exp == target_exp:
        return report_cfg
    _report_log(
        f"[OVERRIDE] realign greedy mix preset: experiment_line={current_exp} -> {target_exp} "
        f"(forced urllc ratio={float(forced_ratio):.3f})"
    )
    aligned = apply_experiment_preset(SRMAPPOConfig(), target_exp)
    # Preserve explicit runtime overrides already resolved into report_cfg.
    aligned.training.run_name = str(getattr(report_cfg.training, "run_name", aligned.training.run_name))
    aligned.env.urllc_user_ratio_override = float(getattr(report_cfg.env, "urllc_user_ratio_override", -1.0))
    aligned.env.urllc_poisson_rate = float(getattr(report_cfg.env, "urllc_poisson_rate", aligned.env.urllc_poisson_rate))
    aligned.env.fixed_urllc_poisson_rate = bool(
        getattr(report_cfg.env, "fixed_urllc_poisson_rate", aligned.env.fixed_urllc_poisson_rate)
    )
    aligned.env.urllc_poisson_rate_is_per_user = bool(
        getattr(report_cfg.env, "urllc_poisson_rate_is_per_user", aligned.env.urllc_poisson_rate_is_per_user)
    )
    aligned.env.greedy_urllc_share_mode = str(
        getattr(report_cfg.env, "greedy_urllc_share_mode", aligned.env.greedy_urllc_share_mode)
    )
    aligned.env.greedy_urllc_share_ratio = float(
        getattr(report_cfg.env, "greedy_urllc_share_ratio", aligned.env.greedy_urllc_share_ratio)
    )
    aligned.training.experiment_line = target_exp
    return aligned


def _load_to_lambda(load: float) -> float:
    base_total_per_uav, base_poisson, fixed_lambda = _base_profile()
    if fixed_lambda:
        return float(base_poisson)
    return float(base_poisson * load / max(base_total_per_uav, 1.0))


def _normalize_baseline_mode(mode: str | None) -> str:
    return _shared_normalize_baseline_mode(mode, default="original")


def _greedy_baseline_mode(cfg: SRMAPPOConfig) -> str:
    override = str(os.environ.get("SR_MAPPO_REPORT_GREEDY_POLICY_OVERRIDE", "") or "").strip().lower()
    if override in {"global_frontier", "global_greedy"}:
        return "global_frontier_greedy"
    if override in {"throughput_only", "tp_only"}:
        return "throughput_only_greedy"
    if override in {"rate_loss_min", "global_rate_loss", "sumrate_minloss", "global_sumrate_minloss"}:
        return "rate_loss_min_greedy"
    if override in {"force_admit_minloss", "rate_loss_force_admit", "sumrate_force_admit"}:
        return "force_admit_minloss_greedy"
    if override in {"channel_only"}:
        return "channel_only_greedy"
    if override in {"myopic", "myopic_throughput"}:
        return "myopic_throughput_greedy"
    if override in {"hard_feasible", "hard_feasible_throughput"}:
        return "hard_feasible_throughput_greedy"
    return _normalize_baseline_mode(getattr(cfg.training, "greedy_baseline_mode", "original"))


def _baseline_label(mode: str | None) -> str:
    return _shared_baseline_label(mode)


def _baseline_metadata(mode: str | None) -> Dict[str, object]:
    return _shared_baseline_metadata(mode)


def _baseline_narrative(
    mode: str | None,
    *,
    greedy_requires_feasible_admission_only: bool = False,
) -> Dict[str, str]:
    return _shared_baseline_narrative(
        mode,
        greedy_requires_feasible_admission_only=greedy_requires_feasible_admission_only,
    )


def _load_frozen_greedy_payload(cfg: SRMAPPOConfig) -> Dict:
    raw_path = str(getattr(cfg.training, "frozen_greedy_metrics_path", "") or "").strip()
    if not raw_path:
        raise FileNotFoundError("greedy_baseline_mode='frozen_json' requires training.frozen_greedy_metrics_path")
    payload_path = Path(raw_path).expanduser()
    if not payload_path.exists():
        raise FileNotFoundError(f"Frozen greedy metrics not found: {payload_path}")
    return json.loads(payload_path.read_text(encoding="utf-8"))


def _normalize_frozen_representative(payload: Dict) -> Dict[float, Dict]:
    rep = payload.get("greedy_representative", {})
    return {float(load): value for load, value in rep.items()}


def _load_checkpoint_cfg(checkpoint_path: Path) -> SRMAPPOConfig:
    payload = torch.load(checkpoint_path, map_location='cpu')
    return cfg_from_dict(payload.get('cfg'))


def _report_seed_base(load_idx: int, cfg: Optional[SRMAPPOConfig] = None) -> int:
    _ = load_idx
    if _REPORT_RUN_SEED_BASE is not None:
        return int(_REPORT_RUN_SEED_BASE)
    cfg = cfg or SRMAPPOConfig()
    return int(getattr(cfg.training, "train_seed", 42) + 1000)


def _env_bool_override(name: str, default: bool) -> bool:
    raw = str(os.environ.get(name, "") or "").strip().lower()
    if not raw:
        return bool(default)
    return raw in {"1", "true", "yes", "on"}


def _env_int_override(name: str, default: int) -> int:
    raw = str(os.environ.get(name, "") or "").strip()
    if not raw:
        return int(default)
    try:
        return int(raw)
    except Exception:
        return int(default)


def _primary_checkpoint_preference(cfg: Optional[SRMAPPOConfig] = None) -> str:
    cfg = cfg or SRMAPPOConfig()
    preference = str(
        getattr(cfg.training, "primary_checkpoint_preference", "best_throughput") or "best_throughput"
    ).strip().lower()
    allowed = {
        "best_throughput",
        "best_balanced",
        "best_balanced_intercell_aware",
        "best_owner_frozen_action_intercell_balanced",
        "best_v5_balanced_intercell_admission",
        "best_v6_balanced_puncture_accounting",
        "best_admission_service_intercell",
        "best_service_interference_balanced",
        "best_service_power_interference_balanced",
        "best_service_gain_interference_balanced",
        "best_multiload_frontier",
        "best_multiload_tp_power",
        "latest_debug",
    }
    return preference if preference in allowed else "best_throughput"


def _require_primary_checkpoint_match(cfg: Optional[SRMAPPOConfig] = None) -> bool:
    cfg = cfg or SRMAPPOConfig()
    return bool(getattr(cfg.training, "require_primary_checkpoint_match", False))


def _primary_checkpoint_match_warning(cfg: Optional[SRMAPPOConfig], checkpoint_reason: str) -> str:
    cfg = cfg or SRMAPPOConfig()
    if _primary_checkpoint_preference(cfg) not in {
        "best_balanced",
        "best_balanced_intercell_aware",
        "best_owner_frozen_action_intercell_balanced",
        "best_v5_balanced_intercell_admission",
        "best_v6_balanced_puncture_accounting",
        "best_admission_service_intercell",
        "best_service_interference_balanced",
        "best_service_power_interference_balanced",
        "best_service_gain_interference_balanced",
        "best_multiload_frontier",
        "best_multiload_tp_power",
    }:
        return ""
    if not _require_primary_checkpoint_match(cfg):
        return ""
    if any(
        token in str(checkpoint_reason)
        for token in (
            "best_balanced",
            "best_balanced_intercell_aware",
            "best_owner_frozen_action_intercell_balanced",
            "best_v5_balanced_intercell_admission",
            "best_v6_balanced_puncture_accounting",
            "best_admission_service_intercell",
            "best_service_interference_balanced",
            "best_service_power_interference_balanced",
            "best_service_gain_interference_balanced",
            "multiload_frontier",
            "multiload_tp_power",
        )
    ):
        return ""
    return "primary_checkpoint_preference requested but not found; fell back to best_throughput"


def _select_checkpoint(
    cfg: Optional[SRMAPPOConfig] = None,
    checkpoint_path: Optional[str] = None,
    checkpoint_kind: Optional[str] = None,
) -> Tuple[Path, str]:
    cfg = cfg or SRMAPPOConfig()
    run_name = cfg.training.run_name
    if checkpoint_path:
        explicit_path = Path(checkpoint_path).expanduser()
        if not explicit_path.is_absolute():
            explicit_path = (Path.cwd() / explicit_path).resolve()
        if not explicit_path.exists():
            raise FileNotFoundError(f"Checkpoint override path not found: {explicit_path}")
        return explicit_path, "override_checkpoint_path"
    checkpoint_eval_scope = str(
        getattr(cfg.training, "checkpoint_eval_scope", "representative_load") or "representative_load"
    ).strip().lower()
    primary_checkpoint_preference = _primary_checkpoint_preference(cfg)
    normalized_checkpoint_kind = str(checkpoint_kind or "").strip().lower()
    report_best_multiload_frontier = CHECKPOINT_DIR / f'{run_name}_report_best_multiload_frontier.pt'
    report_best_multiload_tp_power = CHECKPOINT_DIR / f'{run_name}_report_best_multiload_tp_power.pt'
    report_best_balanced = CHECKPOINT_DIR / f'{run_name}_report_best_balanced.pt'
    report_best_balanced_intercell_aware = CHECKPOINT_DIR / f'{run_name}_report_best_balanced_intercell_aware.pt'
    report_best_owner_frozen_action_intercell_balanced = CHECKPOINT_DIR / f'{run_name}_report_best_owner_frozen_action_intercell_balanced.pt'
    report_best_v5_balanced_intercell_admission = CHECKPOINT_DIR / f'{run_name}_report_best_v5_balanced_intercell_admission.pt'
    report_best_v6_balanced_puncture_accounting = CHECKPOINT_DIR / f'{run_name}_report_best_v6_balanced_puncture_accounting.pt'
    report_best_admission_service_intercell = CHECKPOINT_DIR / f'{run_name}_report_best_admission_service_intercell.pt'
    report_best_service_interference_balanced = CHECKPOINT_DIR / f'{run_name}_report_best_service_interference_balanced.pt'
    report_best_service_power_interference_balanced = CHECKPOINT_DIR / f'{run_name}_report_best_service_power_interference_balanced.pt'
    report_best_service_gain_interference_balanced = CHECKPOINT_DIR / f'{run_name}_report_best_service_gain_interference_balanced.pt'
    report_best_vs_throughput_feasible = CHECKPOINT_DIR / f'{run_name}_report_best_vs_throughput_feasible_oracle.pt'
    report_best_vs_throughput_only = CHECKPOINT_DIR / f'{run_name}_report_best_vs_throughput_only_greedy.pt'
    report_best_vs_channel_only = CHECKPOINT_DIR / f'{run_name}_report_best_vs_channel_only_greedy.pt'
    report_best_vs_original = CHECKPOINT_DIR / f'{run_name}_report_best_vs_original_greedy.pt'
    report_best_vs_matched = CHECKPOINT_DIR / f'{run_name}_report_best_vs_matched_greedy.pt'
    report_best_floor_throughput = CHECKPOINT_DIR / f'{run_name}_report_best_floor_throughput.pt'
    report_best_throughput = CHECKPOINT_DIR / f'{run_name}_report_best_throughput.pt'
    report_best_reward = CHECKPOINT_DIR / f'{run_name}_report_best_reward.pt'
    report_best_alias = CHECKPOINT_DIR / f'{run_name}_report_best.pt'
    best_multiload_frontier = CHECKPOINT_DIR / f'{run_name}_best_multiload_frontier.pt'
    best_multiload_tp_power = CHECKPOINT_DIR / f'{run_name}_best_multiload_tp_power.pt'
    best_balanced = CHECKPOINT_DIR / f'{run_name}_best_balanced.pt'
    best_balanced_intercell_aware = CHECKPOINT_DIR / f'{run_name}_best_balanced_intercell_aware.pt'
    best_owner_frozen_action_intercell_balanced = CHECKPOINT_DIR / f'{run_name}_best_owner_frozen_action_intercell_balanced.pt'
    best_v5_balanced_intercell_admission = CHECKPOINT_DIR / f'{run_name}_best_v5_balanced_intercell_admission.pt'
    best_v6_balanced_puncture_accounting = CHECKPOINT_DIR / f'{run_name}_best_v6_balanced_puncture_accounting.pt'
    best_admission_service_intercell = CHECKPOINT_DIR / f'{run_name}_best_admission_service_intercell.pt'
    best_service_interference_balanced = CHECKPOINT_DIR / f'{run_name}_best_service_interference_balanced.pt'
    best_service_power_interference_balanced = CHECKPOINT_DIR / f'{run_name}_best_service_power_interference_balanced.pt'
    best_service_gain_interference_balanced = CHECKPOINT_DIR / f'{run_name}_best_service_gain_interference_balanced.pt'
    best_vs_throughput_feasible = CHECKPOINT_DIR / f'{run_name}_best_vs_throughput_feasible_oracle.pt'
    best_vs_throughput_only = CHECKPOINT_DIR / f'{run_name}_best_vs_throughput_only_greedy.pt'
    best_vs_channel_only = CHECKPOINT_DIR / f'{run_name}_best_vs_channel_only_greedy.pt'
    best_vs_original = CHECKPOINT_DIR / f'{run_name}_best_vs_original_greedy.pt'
    best_vs_matched = CHECKPOINT_DIR / f'{run_name}_best_vs_matched_greedy.pt'
    best_floor_throughput = CHECKPOINT_DIR / f'{run_name}_best_floor_throughput.pt'
    best_throughput = CHECKPOINT_DIR / f'{run_name}_best_throughput.pt'
    best_reward = CHECKPOINT_DIR / f'{run_name}_best_reward.pt'
    best_alias = CHECKPOINT_DIR / f'{run_name}_best.pt'
    final_path = CHECKPOINT_DIR / f'{run_name}_final.pt'
    latest_debug = CHECKPOINT_DIR / f'{run_name}_latest_debug.pt'
    if normalized_checkpoint_kind:
        override_candidates = {
            "best_throughput": [(report_best_throughput, "override_best_throughput"), (best_throughput, "override_best_throughput")],
            "best_balanced": [(report_best_balanced, "override_best_balanced"), (best_balanced, "override_best_balanced")],
            "best_balanced_intercell_aware": [(report_best_balanced_intercell_aware, "override_best_balanced_intercell_aware"), (best_balanced_intercell_aware, "override_best_balanced_intercell_aware")],
            "best_owner_frozen_action_intercell_balanced": [(report_best_owner_frozen_action_intercell_balanced, "override_best_owner_frozen_action_intercell_balanced"), (best_owner_frozen_action_intercell_balanced, "override_best_owner_frozen_action_intercell_balanced")],
            "best_v5_balanced_intercell_admission": [(report_best_v5_balanced_intercell_admission, "override_best_v5_balanced_intercell_admission"), (best_v5_balanced_intercell_admission, "override_best_v5_balanced_intercell_admission")],
            "best_v6_balanced_puncture_accounting": [(report_best_v6_balanced_puncture_accounting, "override_best_v6_balanced_puncture_accounting"), (best_v6_balanced_puncture_accounting, "override_best_v6_balanced_puncture_accounting")],
            "best_admission_service_intercell": [(report_best_admission_service_intercell, "override_best_admission_service_intercell"), (best_admission_service_intercell, "override_best_admission_service_intercell")],
            "latest_debug": [(latest_debug, "override_latest_debug")],
            "latest": [],
            "final": [(final_path, "override_final")],
            "best": [(report_best_alias, "override_report_best"), (best_alias, "override_best")],
        }
        if normalized_checkpoint_kind == "latest":
            checkpoints = sorted(CHECKPOINT_DIR.glob(f'{run_name}_iter*.pt'), key=lambda p: p.stat().st_mtime, reverse=True)
            if checkpoints:
                return checkpoints[0], "override_latest_iter"
            if final_path.exists():
                return final_path, "override_final"
            raise FileNotFoundError(f"No latest checkpoint found for run {run_name}")
        if normalized_checkpoint_kind not in override_candidates:
            raise ValueError(f"Unsupported checkpoint kind override: {checkpoint_kind}")
        for path, reason in override_candidates[normalized_checkpoint_kind]:
            if path.exists():
                return path, reason
        raise FileNotFoundError(f"Requested checkpoint kind not found: {checkpoint_kind}")
    selection_mode = str(getattr(cfg.training, "selection_mode", "dual_metric") or "dual_metric").strip().lower()
    selection_admission_floor = float(getattr(cfg.training, "selection_admission_floor", 0.0) or 0.0)
    selection_admission_floor_ratio = float(getattr(cfg.training, "selection_admission_floor_ratio_to_baseline", 0.0) or 0.0)
    has_loadwise_selection_constraints = bool(
        dict(getattr(cfg.training, "selection_admission_floor_by_load", {}) or {})
        or dict(getattr(cfg.training, "selection_power_ratio_ceiling_by_load", {}) or {})
        or dict(getattr(cfg.training, "selection_throughput_ratio_floor_by_load", {}) or {})
        or dict(getattr(cfg.training, "selection_service_ratio_floor_by_load", {}) or {})
        or dict(getattr(cfg.training, "selection_minrate_ratio_floor_by_load", {}) or {})
        or dict(getattr(cfg.training, "selection_puncture_ratio_floor_by_load", {}) or {})
        or dict(getattr(cfg.training, "selection_overlay_ratio_ceiling_by_load", {}) or {})
        or float(getattr(cfg.training, "selection_reliability_floor", 0.0) or 0.0) > 0.0
        or float(getattr(cfg.training, "selection_puncture_ratio_ceiling", 1.0) or 1.0) < 1.0 - 1e-9
        or selection_admission_floor_ratio > 0.0
    )

    baseline_pref = str(getattr(cfg.training, "selection_baseline_mode", "original")).strip().lower()
    comparative_preferred = [
        (report_best_vs_throughput_feasible, 'report_best_vs_throughput_feasible_oracle'),
        (report_best_vs_original, 'report_best_vs_original_greedy'),
        (report_best_vs_matched, 'report_best_vs_matched_greedy'),
        (report_best_vs_throughput_only, 'report_best_vs_throughput_only_greedy'),
        (report_best_vs_channel_only, 'report_best_vs_channel_only_greedy'),
    ]
    comparative_best = [
        (best_vs_throughput_feasible, 'best_vs_throughput_feasible_oracle'),
        (best_vs_original, 'best_vs_original_greedy'),
        (best_vs_matched, 'best_vs_matched_greedy'),
        (best_vs_throughput_only, 'best_vs_throughput_only_greedy'),
        (best_vs_channel_only, 'best_vs_channel_only_greedy'),
    ]
    if baseline_pref in {"throughput_feasible", "throughput_feasible_oracle", "coexistence_oracle"}:
        comparative_preferred = [
            (report_best_vs_throughput_feasible, 'report_best_vs_throughput_feasible_oracle'),
            (report_best_vs_original, 'report_best_vs_original_greedy'),
            (report_best_vs_matched, 'report_best_vs_matched_greedy'),
            (report_best_vs_throughput_only, 'report_best_vs_throughput_only_greedy'),
            (report_best_vs_channel_only, 'report_best_vs_channel_only_greedy'),
        ]
        comparative_best = [
            (best_vs_throughput_feasible, 'best_vs_throughput_feasible_oracle'),
            (best_vs_original, 'best_vs_original_greedy'),
            (best_vs_matched, 'best_vs_matched_greedy'),
            (best_vs_throughput_only, 'best_vs_throughput_only_greedy'),
            (best_vs_channel_only, 'best_vs_channel_only_greedy'),
        ]
    elif baseline_pref in {"matched", "matched_fixed_embb"}:
        comparative_preferred = [
            (report_best_vs_matched, 'report_best_vs_matched_greedy'),
            (report_best_vs_throughput_feasible, 'report_best_vs_throughput_feasible_oracle'),
            (report_best_vs_original, 'report_best_vs_original_greedy'),
            (report_best_vs_throughput_only, 'report_best_vs_throughput_only_greedy'),
            (report_best_vs_channel_only, 'report_best_vs_channel_only_greedy'),
        ]
        comparative_best = [
            (best_vs_matched, 'best_vs_matched_greedy'),
            (best_vs_throughput_feasible, 'best_vs_throughput_feasible_oracle'),
            (best_vs_original, 'best_vs_original_greedy'),
            (best_vs_throughput_only, 'best_vs_throughput_only_greedy'),
            (best_vs_channel_only, 'best_vs_channel_only_greedy'),
        ]
    elif baseline_pref in {"throughput_only", "throughput_only_greedy"}:
        comparative_preferred = [
            (report_best_vs_throughput_only, 'report_best_vs_throughput_only_greedy'),
            (report_best_vs_throughput_feasible, 'report_best_vs_throughput_feasible_oracle'),
            (report_best_vs_original, 'report_best_vs_original_greedy'),
            (report_best_vs_matched, 'report_best_vs_matched_greedy'),
            (report_best_vs_channel_only, 'report_best_vs_channel_only_greedy'),
        ]
        comparative_best = [
            (best_vs_throughput_only, 'best_vs_throughput_only_greedy'),
            (best_vs_throughput_feasible, 'best_vs_throughput_feasible_oracle'),
            (best_vs_original, 'best_vs_original_greedy'),
            (best_vs_matched, 'best_vs_matched_greedy'),
            (best_vs_channel_only, 'best_vs_channel_only_greedy'),
        ]
    elif baseline_pref in {"channel_only", "channel_only_greedy"}:
        comparative_preferred = [
            (report_best_vs_channel_only, 'report_best_vs_channel_only_greedy'),
            (report_best_vs_throughput_feasible, 'report_best_vs_throughput_feasible_oracle'),
            (report_best_vs_original, 'report_best_vs_original_greedy'),
            (report_best_vs_matched, 'report_best_vs_matched_greedy'),
        ]
        comparative_best = [
            (best_vs_channel_only, 'best_vs_channel_only_greedy'),
            (best_vs_throughput_feasible, 'best_vs_throughput_feasible_oracle'),
            (best_vs_original, 'best_vs_original_greedy'),
            (best_vs_matched, 'best_vs_matched_greedy'),
        ]
    multiload_preferred = (
        [
            (report_best_multiload_tp_power, 'report_best_multiload_tp_power'),
            (report_best_multiload_frontier, 'report_best_multiload_frontier'),
            (best_multiload_tp_power, 'best_multiload_tp_power'),
            (best_multiload_frontier, 'best_multiload_frontier'),
        ]
        if checkpoint_eval_scope == "all_loads" and primary_checkpoint_preference in {"best_multiload_frontier", "best_multiload_tp_power"} else []
    )
    balanced_preferred = (
        [
            (report_best_balanced, 'report_best_balanced'),
            (best_balanced, 'best_balanced'),
        ]
        if primary_checkpoint_preference == "best_balanced" else []
    )
    balanced_intercell_preferred = (
        [
            (report_best_balanced_intercell_aware, 'report_best_balanced_intercell_aware'),
            (best_balanced_intercell_aware, 'best_balanced_intercell_aware'),
        ]
        if primary_checkpoint_preference == "best_balanced_intercell_aware" else []
    )
    owner_frozen_action_intercell_preferred = (
        [
            (report_best_owner_frozen_action_intercell_balanced, 'report_best_owner_frozen_action_intercell_balanced'),
            (best_owner_frozen_action_intercell_balanced, 'best_owner_frozen_action_intercell_balanced'),
        ]
        if primary_checkpoint_preference == "best_owner_frozen_action_intercell_balanced" else []
    )
    v5_balanced_intercell_admission_preferred = (
        [
            (report_best_v5_balanced_intercell_admission, 'report_best_v5_balanced_intercell_admission'),
            (best_v5_balanced_intercell_admission, 'best_v5_balanced_intercell_admission'),
        ]
        if primary_checkpoint_preference == "best_v5_balanced_intercell_admission" else []
    )
    v6_balanced_puncture_accounting_preferred = (
        [
            (best_v6_balanced_puncture_accounting, 'best_v6_balanced_puncture_accounting'),
            (report_best_v6_balanced_puncture_accounting, 'report_best_v6_balanced_puncture_accounting'),
        ]
        if primary_checkpoint_preference == "best_v6_balanced_puncture_accounting" else []
    )
    admission_service_intercell_preferred = (
        [
            (report_best_admission_service_intercell, 'report_best_admission_service_intercell'),
            (best_admission_service_intercell, 'best_admission_service_intercell'),
        ]
        if primary_checkpoint_preference == "best_admission_service_intercell" else []
    )
    service_interference_preferred = (
        [
            (report_best_service_interference_balanced, 'report_best_service_interference_balanced'),
            (best_service_interference_balanced, 'best_service_interference_balanced'),
        ]
        if primary_checkpoint_preference == "best_service_interference_balanced" else []
    )
    service_power_interference_preferred = (
        [
            (report_best_service_power_interference_balanced, 'report_best_service_power_interference_balanced'),
            (best_service_power_interference_balanced, 'best_service_power_interference_balanced'),
        ]
        if primary_checkpoint_preference == "best_service_power_interference_balanced" else []
    )
    service_gain_interference_preferred = (
        [
            (report_best_service_gain_interference_balanced, 'report_best_service_gain_interference_balanced'),
            (best_service_gain_interference_balanced, 'best_service_gain_interference_balanced'),
        ]
        if primary_checkpoint_preference == "best_service_gain_interference_balanced" else []
    )
    latest_debug_preferred = (
        [
            (latest_debug, 'latest_debug'),
        ]
        if primary_checkpoint_preference == "latest_debug" else []
    )

    if has_loadwise_selection_constraints:
        if multiload_preferred:
            preferred = [
                *multiload_preferred,
                *latest_debug_preferred,
                *owner_frozen_action_intercell_preferred,
                *v5_balanced_intercell_admission_preferred,
                *v6_balanced_puncture_accounting_preferred,
                *balanced_intercell_preferred,
                *admission_service_intercell_preferred,
                *service_gain_interference_preferred,
                *service_interference_preferred,
                *service_power_interference_preferred,
                *balanced_preferred,
                (latest_debug, 'latest_debug'),
                (report_best_throughput, 'report_best_throughput'),
                (best_throughput, 'best_throughput'),
                (report_best_floor_throughput, 'report_best_floor_throughput'),
                (best_floor_throughput, 'best_floor_throughput'),
                *comparative_preferred,
                (report_best_reward, 'report_best_reward'),
                *comparative_best,
                (best_reward, 'best_reward'),
                (report_best_alias, 'report_best'),
                (best_alias, 'best'),
            ]
        else:
            preferred = [
                *owner_frozen_action_intercell_preferred,
                *v5_balanced_intercell_admission_preferred,
                *v6_balanced_puncture_accounting_preferred,
                *balanced_intercell_preferred,
                *admission_service_intercell_preferred,
                *service_gain_interference_preferred,
                *service_interference_preferred,
                *service_power_interference_preferred,
                *balanced_preferred,
                *latest_debug_preferred,
                (latest_debug, 'latest_debug'),
                (report_best_floor_throughput, 'report_best_floor_throughput'),
                (best_floor_throughput, 'best_floor_throughput'),
                (report_best_throughput, 'report_best_throughput'),
                (best_throughput, 'best_throughput'),
                *comparative_preferred,
                (report_best_reward, 'report_best_reward'),
                *comparative_best,
                (best_reward, 'best_reward'),
                (report_best_alias, 'report_best'),
                (best_alias, 'best'),
            ]
        for path, reason in preferred:
            if path.exists():
                return path, reason

    if selection_admission_floor > 0.0:
        preferred = [
            *multiload_preferred,
            (latest_debug, 'latest_debug'),
            *owner_frozen_action_intercell_preferred,
            *v5_balanced_intercell_admission_preferred,
            *v6_balanced_puncture_accounting_preferred,
            *balanced_intercell_preferred,
            *admission_service_intercell_preferred,
            *service_gain_interference_preferred,
            *service_interference_preferred,
            *service_power_interference_preferred,
            *balanced_preferred,
            (report_best_throughput, 'report_best_throughput'),
            (best_throughput, 'best_throughput'),
            (report_best_floor_throughput, 'report_best_floor_throughput'),
            (best_floor_throughput, 'best_floor_throughput'),
            *comparative_preferred,
            (report_best_reward, 'report_best_reward'),
            *comparative_best,
            (best_reward, 'best_reward'),
        ]
    elif selection_mode == "throughput_only":
        preferred = [
            *multiload_preferred,
            (latest_debug, 'latest_debug'),
            *owner_frozen_action_intercell_preferred,
            *v5_balanced_intercell_admission_preferred,
            *v6_balanced_puncture_accounting_preferred,
            *balanced_intercell_preferred,
            *admission_service_intercell_preferred,
            *service_gain_interference_preferred,
            *service_interference_preferred,
            *service_power_interference_preferred,
            *balanced_preferred,
            (report_best_throughput, 'report_best_throughput'),
            (best_throughput, 'best_throughput'),
            *comparative_preferred,
            *comparative_best,
            (report_best_reward, 'report_best_reward'),
            (best_reward, 'best_reward'),
        ]
    else:
        preferred = [
            *multiload_preferred,
            (latest_debug, 'latest_debug'),
            *owner_frozen_action_intercell_preferred,
            *v5_balanced_intercell_admission_preferred,
            *v6_balanced_puncture_accounting_preferred,
            *balanced_intercell_preferred,
            *admission_service_intercell_preferred,
            *service_gain_interference_preferred,
            *service_interference_preferred,
            *service_power_interference_preferred,
            *balanced_preferred,
            *comparative_preferred,
            (report_best_throughput, 'report_best_throughput'),
            (report_best_reward, 'report_best_reward'),
            *comparative_best,
            (best_throughput, 'best_throughput'),
            (best_reward, 'best_reward'),
        ]
    for path, reason in preferred:
        if path.exists():
            return path, reason

    if report_best_alias.exists():
        return report_best_alias, 'report_best'
    if best_alias.exists():
        return best_alias, 'best'
    if final_path.exists():
        return final_path, 'final'

    checkpoints = sorted(CHECKPOINT_DIR.glob(f'{run_name}_iter*.pt'), key=lambda p: p.stat().st_mtime, reverse=True)
    if checkpoints:
        return checkpoints[0], 'latest_iter'
    checkpoints = sorted(CHECKPOINT_DIR.glob('*.pt'), key=lambda p: p.stat().st_mtime, reverse=True)
    if checkpoints:
        return checkpoints[0], 'latest_any'
    raise FileNotFoundError(f'No SR-MAPPO checkpoint found in {CHECKPOINT_DIR}')


def _checkpoint_metadata(path: Path) -> Dict:
    stat = path.stat()
    meta = {
        'path': str(path.resolve()),
        'mtime': datetime.fromtimestamp(stat.st_mtime).isoformat(timespec='seconds'),
        'size_bytes': int(stat.st_size),
        'iteration': None,
    }
    try:
        payload = torch.load(path, map_location='cpu')
        extra = payload.get('extra', {}) if isinstance(payload, dict) else {}
        meta['iteration'] = extra.get('iteration')
        meta['phase_a_embb_power_runtime_enabled'] = bool(
            extra.get(
                'phase_a_embb_power_runtime_enabled',
                ((extra.get('record') or {}).get('control', {}) or {}).get('phase_a_embb_power_runtime_enabled', False),
            )
        )
        meta['phase_a_embb_power_changed_count'] = float(
            extra.get(
                'phase_a_embb_power_changed_count',
                ((extra.get('evaluation') or {}).get('policy_mean_phase_a_embb_power_changed_count', 0.0)),
            ) or 0.0
        )
        meta['phase_a_embb_power_changed_ratio'] = float(
            extra.get(
                'phase_a_embb_power_changed_ratio',
                ((extra.get('evaluation') or {}).get('policy_mean_phase_a_embb_power_changed_ratio', 0.0)),
            ) or 0.0
        )
        meta['phase_a_embb_power_mean_raw_delta'] = float(
            extra.get(
                'phase_a_embb_power_mean_raw_delta',
                ((extra.get('evaluation') or {}).get('policy_mean_phase_a_embb_power_mean_raw_delta', 0.0)),
            ) or 0.0
        )
        meta['phase_a_embb_power_mean_executed_delta'] = float(
            extra.get(
                'phase_a_embb_power_mean_executed_delta',
                ((extra.get('evaluation') or {}).get('policy_mean_phase_a_embb_power_mean_executed_delta', 0.0)),
            ) or 0.0
        )
        meta['phase_a_embb_power_exercised'] = bool(
            extra.get(
                'phase_a_embb_power_exercised',
                meta['phase_a_embb_power_changed_count'] > 1.0e-9
                or abs(meta['phase_a_embb_power_mean_executed_delta']) > 1.0e-9,
            )
        )
        if isinstance(extra.get('evaluation_config'), dict):
            meta['evaluation_config'] = dict(extra.get('evaluation_config'))
    except Exception:
        pass
    return meta


def _load_history_from_final(cfg: Optional[SRMAPPOConfig] = None, checkpoint_path: Optional[Path] = None) -> List[Dict]:
    cfg = cfg or SRMAPPOConfig()
    candidates: List[Path] = []
    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        candidates.append(checkpoint_path)
    run_name = str(getattr(cfg.training, 'run_name', '') or '').strip()
    if run_name:
        candidates.append(CHECKPOINT_DIR / f'{run_name}_final.pt')
    if checkpoint_path is not None:
        stem = checkpoint_path.stem
        if stem.endswith('_best') or '_best_' in stem or '_iter' in stem:
            prefix = stem.split('_best')[0].split('_iter')[0]
            if prefix:
                candidates.append(CHECKPOINT_DIR / f'{prefix}_final.pt')

    seen = set()
    for path in candidates:
        path = Path(path)
        if path in seen or not path.exists():
            continue
        seen.add(path)
        try:
            payload = torch.load(path, map_location='cpu')
            extra = payload.get('extra', {}) or {}
            history = extra.get('history', []) or []
            if history:
                return history
        except Exception:
            continue
    return []


def _load_history_for_report(cfg: Optional[SRMAPPOConfig] = None, checkpoint_path: Optional[Path] = None) -> List[Dict]:
    """Load training history for report-side diagnostics.

    Prefer history embedded in the selected checkpoint (iter/best/latest). If absent,
    fall back to *_final checkpoint history.
    """
    cfg = cfg or SRMAPPOConfig()
    if checkpoint_path is not None:
        path = Path(checkpoint_path)
        if path.exists():
            try:
                payload = torch.load(path, map_location='cpu')
                extra = payload.get('extra', {}) if isinstance(payload, dict) else {}
                history = extra.get('history', []) or []
                if history:
                    return list(history)
            except Exception:
                pass
    return _load_history_from_final(cfg, checkpoint_path=checkpoint_path)


def plot_training_reward_debug(history: List[Dict]) -> Optional[Path]:
    """Plot training-side reward/loss trends for fast/lite report visibility."""
    if not history:
        return None
    iters: List[int] = []
    rollout_reward: List[float] = []
    eval_policy_score: List[float] = []
    policy_loss: List[float] = []
    value_loss: List[float] = []
    entropy: List[float] = []

    for rec in history:
        if not isinstance(rec, dict):
            continue
        try:
            it = int(rec.get('iteration', 0) or 0)
        except Exception:
            continue
        if it <= 0:
            continue
        iters.append(it)
        roll = rec.get('rollout', {}) if isinstance(rec.get('rollout', {}), dict) else {}
        upd = rec.get('update', {}) if isinstance(rec.get('update', {}), dict) else {}
        ev = rec.get('evaluation', {}) if isinstance(rec.get('evaluation', {}), dict) else {}
        rollout_reward.append(float(roll.get('mean_reward', np.nan)))
        eval_policy_score.append(float(ev.get('policy_score', np.nan)))
        policy_loss.append(float(upd.get('policy_loss', np.nan)))
        value_loss.append(float(upd.get('value_loss', np.nan)))
        entropy.append(float(upd.get('entropy', np.nan)))

    if not iters:
        return None

    x = np.asarray(iters, dtype=float)
    fig, axes = plt.subplots(2, 2, figsize=(14.0, 8.0), constrained_layout=True)
    axes = np.asarray(axes).reshape(2, 2)

    def _plot(ax, y: List[float], title: str, ylabel: str, color: str):
        arr = np.asarray(y, dtype=float)
        mask = np.isfinite(arr) & np.isfinite(x)
        if np.any(mask):
            ax.plot(x[mask], arr[mask], color=color, marker='o', markersize=4, linewidth=1.8)
        else:
            ax.text(0.05, 0.1, "unavailable", transform=ax.transAxes, fontsize=9, color='tab:gray')
        _style(ax, title, 'Iteration', ylabel)

    _plot(axes[0, 0], rollout_reward, 'Training rollout mean reward', 'Reward', 'tab:orange')
    _plot(axes[0, 1], eval_policy_score, 'Eval policy score', 'Score', 'tab:blue')
    _plot(axes[1, 0], policy_loss, 'Policy loss', 'Loss', 'tab:red')
    _plot(axes[1, 1], value_loss, 'Value loss (+ entropy)', 'Loss', 'tab:green')

    # Overlay entropy on value-loss panel (secondary trend, same axis for simplicity).
    ent = np.asarray(entropy, dtype=float)
    mask_e = np.isfinite(ent) & np.isfinite(x)
    if np.any(mask_e):
        axes[1, 1].plot(x[mask_e], ent[mask_e], color='tab:purple', linestyle='--', linewidth=1.4, label='entropy')
        axes[1, 1].legend(loc='best', fontsize=8, frameon=False)

    path = RESULTS_DIR / 'training_reward_debug.png'
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def _compute_cell_edge_served_ratio(sys_cfg, topology, best_uav_per_user, embb_rates):
    embb_rates = np.asarray(embb_rates, dtype=float)
    if topology is None or embb_rates.size == 0:
        return float('nan')
    num_urllc = sys_cfg.num_urllc_users
    num_embb = sys_cfg.num_embb_users
    embb_uavs = np.asarray(best_uav_per_user[num_urllc:num_urllc + num_embb], dtype=int)
    distances = np.asarray(topology['horizontal_distances'][num_urllc:num_urllc + num_embb], dtype=float)
    serving_distances = distances[np.arange(num_embb), embb_uavs]
    threshold = np.percentile(serving_distances, 75)
    edge_mask = serving_distances >= threshold
    if not np.any(edge_mask):
        return float('nan')
    return float(np.mean(embb_rates[edge_mask] > 0))


def _compute_embb_min_rate_satisfaction_ratio(embb_rates, min_rate_bps: float) -> float:
    embb_rates = np.asarray(embb_rates, dtype=float)
    if embb_rates.size == 0:
        return float('nan')
    if float(min_rate_bps) <= 0.0:
        return float(np.mean(embb_rates > 0.0))
    return float(np.mean(embb_rates >= float(min_rate_bps)))


def _compute_embb_min_rate_satisfied_user_count(embb_rates, min_rate_bps: float) -> float:
    embb_rates = np.asarray(embb_rates, dtype=float)
    if embb_rates.size == 0:
        return 0.0
    if float(min_rate_bps) <= 0.0:
        return float(np.count_nonzero(embb_rates > 0.0))
    return float(np.count_nonzero(embb_rates >= float(min_rate_bps)))


def _compute_cell_edge_min_rate_satisfaction_ratio(
    sys_cfg,
    topology,
    best_uav_per_user,
    embb_rates,
    min_rate_bps: float,
):
    embb_rates = np.asarray(embb_rates, dtype=float)
    if topology is None or embb_rates.size == 0:
        return float('nan')
    num_urllc = sys_cfg.num_urllc_users
    num_embb = sys_cfg.num_embb_users
    embb_uavs = np.asarray(best_uav_per_user[num_urllc:num_urllc + num_embb], dtype=int)
    distances = np.asarray(topology['horizontal_distances'][num_urllc:num_urllc + num_embb], dtype=float)
    serving_distances = distances[np.arange(num_embb), embb_uavs]
    threshold = np.percentile(serving_distances, 75)
    edge_mask = serving_distances >= threshold
    if not np.any(edge_mask):
        return float('nan')
    if float(min_rate_bps) <= 0.0:
        return float(np.mean(embb_rates[edge_mask] > 0.0))
    return float(np.mean(embb_rates[edge_mask] >= float(min_rate_bps)))


def _style(ax, title: str, xlabel: str, ylabel: str):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)


def _set_plain_y_ticks(ax):
    formatter = ScalarFormatter(useOffset=False)
    formatter.set_scientific(False)
    ax.yaxis.set_major_formatter(formatter)
    ax.ticklabel_format(axis='y', style='plain', useOffset=False)


def _style_power_axis(ax, title: str, xlabel: str, ylabel: str):
    _style(ax, title, xlabel, ylabel)
    _set_plain_y_ticks(ax)


def _style_timeslot_axis(ax, title: str, ylabel: str, num_slots: int):
    _style(ax, title, 'Time slot index', ylabel)
    ax.set_xlim(0, max(num_slots - 1, 0))
    ax.margins(x=0.0)


def _build_uav_ue_distribution_bundle(rl_rep: Dict[float, Dict], loads: List[float]) -> Dict[str, Dict]:
    bundle: Dict[str, Dict] = {}
    for load in loads:
        rep = rl_rep.get(float(load), {}) if isinstance(rl_rep, dict) else {}
        embb = np.asarray(rep.get('per_uav_associated_embb', []), dtype=float)
        urllc = np.asarray(rep.get('per_uav_associated_urllc', []), dtype=float)
        user_positions = np.asarray(rep.get('user_positions', []), dtype=float)
        uav_positions = np.asarray(rep.get('uav_positions', []), dtype=float)
        embb_user_count = int(float(rep.get('embb_user_count', 0.0) or 0.0))
        urllc_user_count = int(float(rep.get('urllc_user_count', 0.0) or 0.0))
        if embb.size == 0 and urllc.size == 0:
            bundle[str(float(load))] = {
                'load': float(load),
                'num_uavs': 0,
                'per_uav_associated_embb': [],
                'per_uav_associated_urllc': [],
                'per_uav_total_associated_ue': [],
                'embb_total_associated_ue': 0.0,
                'urllc_total_associated_ue': 0.0,
                'total_associated_ue': 0.0,
                'user_positions': [],
                'uav_positions': [],
                'embb_user_count': 0,
                'urllc_user_count': 0,
            }
            continue
        n = int(max(embb.size, urllc.size))
        if embb.size < n:
            embb = np.pad(embb, (0, n - embb.size))
        if urllc.size < n:
            urllc = np.pad(urllc, (0, n - urllc.size))
        total = embb + urllc
        bundle[str(float(load))] = {
            'load': float(load),
            'num_uavs': int(n),
            'per_uav_associated_embb': [float(v) for v in embb.tolist()],
            'per_uav_associated_urllc': [float(v) for v in urllc.tolist()],
            'per_uav_total_associated_ue': [float(v) for v in total.tolist()],
            'embb_total_associated_ue': float(np.sum(embb)),
            'urllc_total_associated_ue': float(np.sum(urllc)),
            'total_associated_ue': float(np.sum(total)),
            'user_positions': user_positions.tolist() if user_positions.size > 0 else [],
            'uav_positions': uav_positions.tolist() if uav_positions.size > 0 else [],
            'embb_user_count': int(max(embb_user_count, 0)),
            'urllc_user_count': int(max(urllc_user_count, 0)),
        }
    return bundle


def plot_uav_ue_distribution(rl_rep: Dict[float, Dict], loads: List[float]) -> List[Path]:
    output_paths: List[Path] = []
    for load in loads:
        rep = rl_rep.get(float(load), {}) if isinstance(rl_rep, dict) else {}
        embb = np.asarray(rep.get('per_uav_associated_embb', []), dtype=float)
        urllc = np.asarray(rep.get('per_uav_associated_urllc', []), dtype=float)
        if embb.size == 0 and urllc.size == 0:
            continue
        n = int(max(embb.size, urllc.size))
        if embb.size < n:
            embb = np.pad(embb, (0, n - embb.size))
        if urllc.size < n:
            urllc = np.pad(urllc, (0, n - urllc.size))
        user_positions = np.asarray(rep.get('user_positions', []), dtype=float)
        uav_positions = np.asarray(rep.get('uav_positions', []), dtype=float)
        embb_user_count = int(float(rep.get('embb_user_count', 0.0) or 0.0))
        urllc_user_count = int(float(rep.get('urllc_user_count', 0.0) or 0.0))
        fig, ax = plt.subplots(figsize=(7.0, 6.0))
        if (
            user_positions.ndim == 2 and user_positions.shape[1] >= 2
            and uav_positions.ndim == 2 and uav_positions.shape[1] >= 2
            and user_positions.shape[0] > 0 and uav_positions.shape[0] > 0
        ):
            embb_n = int(np.clip(embb_user_count, 0, user_positions.shape[0]))
            urllc_n = int(np.clip(urllc_user_count, 0, max(user_positions.shape[0] - embb_n, 0)))
            embb_pts = user_positions[:embb_n, :2]
            urllc_pts = user_positions[embb_n:embb_n + urllc_n, :2]
            if embb_pts.size > 0:
                ax.scatter(embb_pts[:, 0], embb_pts[:, 1], s=12, c='#4c78a8', alpha=0.85, label='eMBB UE', marker='o', edgecolors='none')
            if urllc_pts.size > 0:
                ax.scatter(urllc_pts[:, 0], urllc_pts[:, 1], s=12, c='#e45756', alpha=0.85, label='URLLC UE', marker='o', edgecolors='none')
            ax.scatter(uav_positions[:, 0], uav_positions[:, 1], s=70, c='black', marker='^', label='UAV', zorder=3)
            ax.set_title('UAV and UE Distribution')
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.grid(True, alpha=0.25)
            ax.legend(loc='upper right')
        else:
            x = np.arange(n)
            ax.bar(x, embb, label='eMBB UE', color='#1f77b4')
            ax.bar(x, urllc, bottom=embb, label='URLLC UE', color='#ff7f0e')
            ax.set_title(f'UAV-UE Distribution (load={float(load):.1f})')
            ax.set_xlabel('UAV index')
            ax.set_ylabel('Associated UE count')
            ax.set_xticks(x)
            ax.legend(loc='upper right')
            ax.grid(True, axis='y', alpha=0.25)
        fig.tight_layout()
        out_path = RESULTS_DIR / f"uav_ue_distribution_load_{float(load):.1f}.png"
        fig.savefig(out_path, dpi=180)
        plt.close(fig)
        output_paths.append(out_path)
    return output_paths


def _build_owner_change_detail_bundle(rl_rep: Dict[float, Dict], loads: List[float]) -> Dict[str, Dict]:
    bundle: Dict[str, Dict] = {}
    for load in loads:
        rep = rl_rep.get(float(load), {}) if isinstance(rl_rep, dict) else {}
        details = list(rep.get("phase0_owner_change_detail_top", []) or [])
        harmful_ratio = float(rep.get("phase0_owner_change_harmful_ratio", 0.0) or 0.0)
        bundle[str(float(load))] = {
            "load": float(load),
            "phase0_owner_effective_rate_gain_vs_snapshot_mean": float(
                rep.get("phase0_owner_effective_rate_gain_vs_snapshot_mean", 0.0) or 0.0
            ),
            "phase0_owner_change_harmful_ratio": float(harmful_ratio),
            "phase0_owner_harmful_accepted_ratio": float(rep.get("phase0_owner_harmful_accepted_ratio", 0.0) or 0.0),
            "phase0_owner_objective_gain_accepted_mean": float(rep.get("phase0_owner_objective_gain_accepted_mean", 0.0) or 0.0),
            "owner_limiter_harmful_flag": bool(harmful_ratio > 0.30),
            "details_top": details[:32],
        }
    return bundle


def _top_axis_lambda(ax, loads: List[float]):
    # Debug reports and paper figures previously used a secondary x-axis to show the
    # derived Poisson arrival rate. For fast debugging, this is visual clutter and
    # often confuses interpretation. Keep this as a no-op to avoid any "lambda"
    # annotations on plots.
    return None


def _scenario_info_box_lines(
    sys_cfg,
    sim_cfg,
    load: float,
    checkpoint: Path,
    cfg: SRMAPPOConfig,
    num_slots: int,
    greedy_baseline_mode: str | None = None,
) -> str:
    approx_episodes = int(round(cfg.training.total_iterations * cfg.training.rollout_horizon / max(sys_cfg.num_subcarriers * sys_cfg.num_minislots, 1)))
    lines = [
        f'Load: {load:.0f} UE/UAV',
        f'UAVs: {sys_cfg.num_uavs}',
        f'eMBB UEs: {sys_cfg.num_embb_users}',
        f'URLLC UEs: {sys_cfg.num_urllc_users}',
        f'RB x minislot: {sys_cfg.num_subcarriers} x {sys_cfg.num_minislots}',
        f'Time slots shown: 0..{num_slots - 1}',
        f'Checkpoint: {checkpoint.name}',
        f'Experiment line: {experiment_label(cfg.training.experiment_line)}',
        f'Greedy baseline mode: {greedy_baseline_mode or _greedy_baseline_mode(cfg)}',
        f'BC episodes: {cfg.training.bc_episodes}',
        f'PPO episodes: {cfg.training.total_iterations}',
        f'Horizon: {cfg.training.rollout_horizon}',
        f'Approx slot-episodes: {approx_episodes}',
    ]
    return '\n'.join(lines)


def _add_side_info_box(fig, text: str):
    fig.subplots_adjust(right=0.80)
    fig.text(
        0.82,
        0.98,
        text,
        va='top',
        ha='left',
        fontsize=9,
        family='monospace',
        bbox=dict(boxstyle='round', facecolor='#f6f8fa', edgecolor='#c7cdd4', alpha=0.96),
    )


def _policy_actions(env, model, observations, actor_hidden, critic_hidden):
    device = model.power_log_std.device
    local_obs = torch.from_numpy(np.stack([observations[agent_id].local_obs for agent_id in env.agent_ids]).astype(np.float32)).to(device)
    global_obs = torch.from_numpy(np.stack([observations[agent_id].global_obs for agent_id in env.agent_ids]).astype(np.float32)).to(device)
    mode_mask = torch.from_numpy(np.stack([observations[agent_id].masks.mode_mask for agent_id in env.agent_ids]).astype(np.float32)).to(device)
    packet_mask = torch.from_numpy(np.stack([observations[agent_id].masks.packet_mask for agent_id in env.agent_ids]).astype(np.float32)).to(device)
    embb_owner_mask = torch.from_numpy(np.stack([observations[agent_id].masks.embb_owner_mask for agent_id in env.agent_ids]).astype(np.float32)).to(device)
    output = model.act(
        local_obs=local_obs,
        global_obs=global_obs,
        mode_mask=mode_mask,
        packet_mask=packet_mask,
        embb_owner_mask=embb_owner_mask,
        actor_hidden=actor_hidden,
        critic_hidden=critic_hidden,
        deterministic=True,
    )
    joint_actions = {}
    for idx, agent_id in enumerate(env.agent_ids):
        joint_actions[agent_id] = HybridAction(
            mode=int(output.mode[idx].item()),
            packet_option=int(output.packet_option[idx].item()),
            power_delta=0.0,
            embb_owner_option=int(output.embb_owner_option[idx].item()),
            embb_power_delta=float(output.embb_power_delta[idx].item()),
        )
    planning_phase = all(
        bool(observations[agent_id].metadata.get("planning_phase", 0.0))
        for agent_id in env.agent_ids
    )
    if (not planning_phase) and (not bool(getattr(env.rl_cfg.env, "allow_phase_a_embb_power_adjustment", False))):
        for agent_id in env.agent_ids:
            joint_actions[agent_id].embb_power_delta = 0.0
    return joint_actions, output.actor_hidden.detach(), output.critic_hidden.detach()


def _greedy_actions(env, observations):
    actions = {}
    for agent_id in env.agent_ids:
        ref = observations[agent_id].greedy_reference
        actions[agent_id] = ref if ref is not None else HybridAction()
    return actions


def _channel_only_actions(env, observations):
    """Deliberately weak, conservative, puncture-biased channel heuristic baseline.

    This baseline is intentionally weaker than the greedy reference. It only looks at
    the current observation candidates, narrows the choice to the top-2 by channel
    gain, uses a 70/30 stochastic tie-break, and only allows overlay when retention
    is very strong and clearly better than puncture. The goal is to provide a weaker
    channel-only control line rather than a strong handcrafted baseline.
    """
    actions = {}
    for agent_id, obs in observations.items():
        feasible = [
            (idx, candidate)
            for idx, candidate in enumerate(obs.candidates, start=1)
            if bool(candidate.overlay_feasible) or bool(candidate.puncture_feasible)
        ]
        if not feasible:
            actions[agent_id] = HybridAction(mode=MODE_KEEP, packet_option=0, power_delta=0.0)
            continue
        feasible.sort(key=lambda item: float(item[1].channel_gain), reverse=True)
        shortlist = feasible[:2]
        if len(shortlist) == 1:
            option_idx, best = shortlist[0]
        else:
            base_seed = int(getattr(env, "current_reset_seed", 0))
            cell_index = int(getattr(env, "current_cell_index", 0))
            agent_hash = sum(ord(ch) for ch in str(agent_id))
            rng = np.random.default_rng(
                (base_seed * 73856093 + cell_index * 19349663 + agent_hash * 83492791) & 0xFFFFFFFF
            )
            option_idx, best = shortlist[0] if float(rng.random()) < 0.70 else shortlist[1]
        overlay_margin = float(best.overlay_utility) - float(best.puncture_utility)
        puncture_scale = 0.10 * max(abs(float(best.puncture_utility)), 1.0)
        allow_overlay = bool(
            best.overlay_feasible
            and float(best.overlay_retention) >= 0.93
            and overlay_margin >= puncture_scale
        )
        if allow_overlay:
            mode = MODE_OVERLAY
        elif bool(best.puncture_feasible):
            mode = MODE_PUNCTURE
        elif bool(best.overlay_feasible):
            mode = MODE_OVERLAY
        else:
            mode = MODE_KEEP
        actions[agent_id] = HybridAction(
            mode=int(mode),
            packet_option=int(option_idx),
            power_delta=0.0,
        )
    return actions


def _throughput_only_actions(env, observations):
    actions = {}
    diagnostics = {}
    for agent_id, obs in observations.items():
        action, debug = env.throughput_only_greedy_action(obs)
        actions[agent_id] = action
        diagnostics[agent_id] = debug
    return actions, diagnostics


def _throughput_feasible_actions(env, observations):
    actions = {}
    diagnostics = {}
    for agent_id, obs in observations.items():
        action, debug = env.throughput_feasible_oracle_action(obs)
        actions[agent_id] = action
        diagnostics[agent_id] = debug
    return actions, diagnostics


def _rate_loss_min_actions(env, observations):
    """Pure-sumrate owner baseline + minimum global eMBB-loss admit baseline."""

    actions = {}
    diagnostics = {}
    for agent_id, obs in observations.items():
        action, debug = env.rate_loss_min_greedy_action(obs)
        actions[agent_id] = action
        diagnostics[agent_id] = debug
    return actions, diagnostics


def _force_admit_minloss_actions(env, observations):
    """Pure-sumrate owner baseline + no-KEEP hard-feasible min-loss admit baseline."""

    actions = {}
    diagnostics = {}
    for agent_id, obs in observations.items():
        action, debug = env.force_admit_minloss_greedy_action(obs)
        actions[agent_id] = action
        diagnostics[agent_id] = debug
    return actions, diagnostics


def _hard_feasible_throughput_actions(env, observations):
    actions = {}
    diagnostics = {}
    for agent_id, obs in observations.items():
        action, debug = env.hard_feasible_throughput_greedy_action(obs)
        actions[agent_id] = action
        diagnostics[agent_id] = debug
    return actions, diagnostics


def _global_frontier_actions(env, observations):
    actions = {}
    diagnostics = {}
    for agent_id, obs in observations.items():
        action, debug = env.global_frontier_greedy_action(obs)
        actions[agent_id] = action
        diagnostics[agent_id] = debug
    return actions, diagnostics


def _normalize_urllc_success_metrics(
    active_packets: float,
    scheduled_packets: float,
    admitted_reliability: float,
) -> tuple[float, float, float]:
    if not np.isfinite(admitted_reliability):
        admitted_reliability = float("nan") if active_packets > 0 and scheduled_packets <= 0 else (
            1.0 if active_packets <= 0 else 0.0
        )
    if active_packets <= 0:
        effective_success = 1.0
    elif scheduled_packets <= 0 or not np.isfinite(admitted_reliability):
        effective_success = 0.0
    else:
        effective_success = float(admitted_reliability * scheduled_packets / max(active_packets, 1.0))
    empty_admission_case = float(active_packets > 0 and scheduled_packets <= 0)
    return float(admitted_reliability), float(effective_success), float(empty_admission_case)


def _greedy_summary_from_result(result: Dict, sys_cfg, embb_cfg=None) -> Dict:
    metrics = result['metrics']
    allocation = result['allocation']
    embb_rates = np.asarray(metrics.get('embb_per_user_rate', []), dtype=float)
    min_rate_bps = float(getattr(embb_cfg, 'min_rate_per_user_bps', 0.0) or 0.0) if embb_cfg is not None else 0.0
    # Unify positive-rate ratio with service ratio: same single definition.
    embb_service_ratio = float(metrics['embb_served_users'] / max(sys_cfg.num_embb_users, 1))
    embb_positive_rate_ratio = float(embb_service_ratio)
    embb_min_rate_satisfaction_ratio = _compute_embb_min_rate_satisfaction_ratio(embb_rates, min_rate_bps)
    embb_min_rate_satisfied_user_count = _compute_embb_min_rate_satisfied_user_count(embb_rates, min_rate_bps)
    cell_edge_min_rate_satisfaction_ratio = _compute_cell_edge_min_rate_satisfaction_ratio(
        sys_cfg,
        allocation.get('topology'),
        allocation['best_uav_per_user'],
        embb_rates,
        min_rate_bps,
    )
    active_packets = float(metrics.get('active_urllc_users', 0.0))
    scheduled_packets = float(metrics.get('scheduled_urllc_users', 0.0))
    admitted_reliability, effective_success, empty_admission_case = _normalize_urllc_success_metrics(
        active_packets,
        scheduled_packets,
        float(metrics.get('admitted_urllc_reliability', metrics.get('urllc_success_rate', np.nan))),
    )
    return {
        'embb_rate': float(metrics['embb_total_rate']),
        'embb_rate_after_local_puncture_deduction': float(
            metrics.get(
                'embb_total_rate_after_puncture_deduction',
                metrics.get('embb_rate_after_local_puncture_deduction', metrics['embb_total_rate']),
            )
        ),
        'embb_total_rate_after_puncture_deduction': float(
            metrics.get(
                'embb_total_rate_after_puncture_deduction',
                metrics.get('embb_rate_after_local_puncture_deduction', metrics['embb_total_rate']),
            )
        ),
        'embb_user_rate': float(metrics['embb_user_rate_mean']),
        'embb_user_rate_mean_after_puncture_deduction': float(
            metrics.get('embb_user_rate_mean_after_puncture_deduction', metrics['embb_user_rate_mean'])
        ),
        'embb_service_ratio': embb_service_ratio,
        'embb_service_ratio_after_puncture_deduction': float(
            metrics.get('embb_service_ratio_after_puncture_deduction', embb_service_ratio)
        ),
        'embb_positive_rate_ratio': embb_positive_rate_ratio,
        'embb_min_rate_satisfaction_ratio': float(embb_min_rate_satisfaction_ratio),
        'embb_min_rate_satisfied_user_count': float(embb_min_rate_satisfied_user_count),
        'urllc_admission': float(metrics['urllc_admission_rate']),
        'admitted_urllc_reliability': admitted_reliability,
        'urllc_reliability': admitted_reliability,
        'effective_urllc_success_over_arrivals': effective_success,
        'empty_admission_case': empty_admission_case,
        'effective_lambda_per_user': float(metrics.get('effective_lambda_per_user', 0.0)),
        'effective_lambda_per_user_per_minislot': float(metrics.get('effective_lambda_per_user_per_minislot', 0.0)),
        'expected_total_arrivals_per_minislot': float(metrics.get('expected_total_arrivals_per_minislot', 0.0)),
        'expected_total_arrivals_per_episode': float(metrics.get('expected_total_arrivals_per_episode', 0.0)),
        'active_packets': active_packets,
        'scheduled_packets': scheduled_packets,
        'overlay_count': float(metrics.get('overlay_count', 0.0)),
        'puncture_count': float(metrics.get('puncture_count', 0.0)),
        'total_power': float(metrics['total_power']),
        'embb_power': float(metrics['embb_power']),
        'urllc_power': float(metrics['urllc_power']),
        'overlay_ratio': float(metrics['overlay_ratio']),
        'puncture_ratio': float(metrics['puncture_ratio']),
        'overlay_selection_ratio': float(metrics['overlay_ratio']),
        'puncture_selection_ratio': float(metrics['puncture_ratio']),
        'embb_only_fraction': float(metrics['embb_only_fraction']),
        'avg_puncture_loss': float(metrics['avg_puncture_embb_loss']),
        'avg_overlay_retention': float(metrics['avg_overlay_retention']),
        'overlay_candidate_pairs': float(metrics['overlay_candidate_pairs']),
        'overlay_feasible_pairs': float(metrics['overlay_feasible_pairs']),
        'overlay_selected_pairs': float(metrics['overlay_selected_pairs']),
        'admission_via_overlay_ratio': float(metrics.get('overlay_count', 0.0) / max(metrics.get('scheduled_urllc_users', 0.0), 1.0)),
        'admission_via_puncture_ratio': float(metrics.get('puncture_count', 0.0) / max(metrics.get('scheduled_urllc_users', 0.0), 1.0)),
        'puncture_candidate_pruned_by_loss_ceiling_ratio': 0.0,
        'jain_fairness': float(metrics['jain_fairness']),
        'cell_edge_served_ratio': float(metrics['cell_edge_served_ratio']),
        'cell_edge_min_rate_satisfaction_ratio': float(cell_edge_min_rate_satisfaction_ratio),
        'per_uav_total_load_std': float(metrics['per_uav_total_load_std']),
        'per_uav_urllc_sched_std': float(metrics['per_uav_urllc_sched_std']),
        'per_uav_throughput_std': float(np.std(np.asarray(metrics['per_uav_embb_throughput'], dtype=float))),
        'per_uav_associated_embb': np.asarray(metrics['per_uav_associated_embb'], dtype=float),
        'per_uav_associated_urllc': np.asarray(metrics['per_uav_associated_urllc'], dtype=float),
        'per_uav_scheduled_embb': np.asarray(metrics['per_uav_scheduled_embb'], dtype=float),
        'per_uav_scheduled_urllc': np.asarray(metrics['per_uav_scheduled_urllc'], dtype=float),
        'per_uav_overlay_count': np.asarray(metrics['per_uav_overlay_count'], dtype=float),
        'per_uav_puncture_count': np.asarray(metrics['per_uav_puncture_count'], dtype=float),
        'per_uav_embb_throughput': np.asarray(metrics['per_uav_embb_throughput'], dtype=float),
        'allocation': allocation,
        'best_uav_per_user': np.asarray(allocation['best_uav_per_user'], dtype=int),
        'topology': allocation.get('topology'),
    }


def _run_original_greedy_slot(
    sys_cfg,
    urllc_cfg,
    embb_cfg,
    algo_cfg,
    sim_cfg,
    seed: int,
    slot_index: int = 0,
) -> Dict:
    sys_local = deepcopy(sys_cfg)
    urllc_local = deepcopy(urllc_cfg)
    embb_local = deepcopy(embb_cfg)
    algo_local = deepcopy(algo_cfg)
    sim_local = deepcopy(sim_cfg)
    sim_local.random_seed = int(seed)
    simulation = create_simulation(sys_local, urllc_local, embb_local, algo_local, sim_local)
    result = simulation.run_single_allocation(slot_index=slot_index)
    return _greedy_summary_from_result(result, sys_local, embb_local)


def _run_original_greedy_normal_v1_slot(
    sys_cfg,
    urllc_cfg,
    embb_cfg,
    algo_cfg,
    sim_cfg,
    seed: int,
    slot_index: int = 0,
) -> Dict:
    sys_local = deepcopy(sys_cfg)
    urllc_local = deepcopy(urllc_cfg)
    embb_local = deepcopy(embb_cfg)
    algo_local = deepcopy(algo_cfg)
    sim_local = deepcopy(sim_cfg)
    sim_local.random_seed = int(seed)
    simulation = create_simulation(sys_local, urllc_local, embb_local, algo_local, sim_local)
    result = simulation.run_single_allocation_normal_v1(slot_index=slot_index)
    return _greedy_summary_from_result(result, sys_local, embb_local)


def _run_original_greedy_normal_v2_slot(
    sys_cfg,
    urllc_cfg,
    embb_cfg,
    algo_cfg,
    sim_cfg,
    seed: int,
    slot_index: int = 0,
) -> Dict:
    sys_local = deepcopy(sys_cfg)
    urllc_local = deepcopy(urllc_cfg)
    embb_local = deepcopy(embb_cfg)
    algo_local = deepcopy(algo_cfg)
    sim_local = deepcopy(sim_cfg)
    sim_local.random_seed = int(seed)
    simulation = create_simulation(sys_local, urllc_local, embb_local, algo_local, sim_local)
    result = simulation.run_single_allocation_normal_v2(slot_index=slot_index)
    return _greedy_summary_from_result(result, sys_local, embb_local)


def _run_embb_only_ceiling_slot(
    sys_cfg,
    urllc_cfg,
    embb_cfg,
    algo_cfg,
    sim_cfg,
    seed: int,
    slot_index: int = 0,
) -> Dict:
    sys_local = deepcopy(sys_cfg)
    urllc_local = deepcopy(urllc_cfg)
    embb_local = deepcopy(embb_cfg)
    algo_local = deepcopy(algo_cfg)
    sim_local = deepcopy(sim_cfg)
    sim_local.random_seed = int(seed)
    simulation = create_simulation(sys_local, urllc_local, embb_local, algo_local, sim_local)
    result = simulation.run_embb_only_ceiling(slot_index=slot_index)
    return _greedy_summary_from_result(result, sys_local, embb_local)


def _run_throughput_feasible_oracle_slot(
    sys_cfg,
    urllc_cfg,
    embb_cfg,
    algo_cfg,
    sim_cfg,
    seed: int,
    slot_index: int = 0,
    admission_quota: Optional[int] = None,
) -> Dict:
    sys_local = deepcopy(sys_cfg)
    urllc_local = deepcopy(urllc_cfg)
    embb_local = deepcopy(embb_cfg)
    algo_local = deepcopy(algo_cfg)
    sim_local = deepcopy(sim_cfg)
    sim_local.random_seed = int(seed)
    simulation = create_simulation(sys_local, urllc_local, embb_local, algo_local, sim_local)
    result = simulation.run_throughput_feasible_oracle(slot_index=slot_index, admission_quota=admission_quota)
    return _greedy_summary_from_result(result, sys_local, embb_local)


def run_greedy_sweep(loads: List[float], episodes_per_load: int) -> Tuple[Dict, Dict]:
    sweep_start = perf_counter()
    base_sys, base_urllc, base_embb, base_algo, base_sim = _build_main_like_configs()
    report_cfg = SRMAPPOConfig()
    scalar_keys = [
        'embb_rate', 'embb_rate_after_local_puncture_deduction', 'embb_total_rate_after_puncture_deduction',
        'embb_user_rate', 'embb_user_rate_mean_after_puncture_deduction',
        'embb_service_ratio', 'embb_service_ratio_after_puncture_deduction', 'embb_positive_rate_ratio', 'embb_min_rate_satisfaction_ratio', 'embb_min_rate_satisfied_user_count', 'urllc_admission',
        'admitted_urllc_reliability', 'urllc_reliability', 'effective_urllc_success_over_arrivals', 'empty_admission_case',
        'effective_lambda_per_user', 'effective_lambda_per_user_per_minislot',
        'expected_total_arrivals_per_minislot', 'expected_total_arrivals_per_episode',
        'active_packets', 'scheduled_packets', 'total_power', 'embb_power', 'urllc_power',
        'overlay_ratio', 'puncture_ratio', 'overlay_selection_ratio', 'puncture_selection_ratio',
        'embb_only_fraction', 'avg_puncture_loss',
        'avg_overlay_retention', 'overlay_candidate_pairs', 'overlay_feasible_pairs',
        'overlay_selected_pairs', 'admission_via_overlay_ratio', 'admission_via_puncture_ratio',
        'jain_fairness', 'cell_edge_served_ratio', 'cell_edge_min_rate_satisfaction_ratio',
        'per_uav_total_load_std', 'per_uav_urllc_sched_std', 'per_uav_throughput_std',
    ]
    vector_keys = [
        'per_uav_associated_embb', 'per_uav_associated_urllc', 'per_uav_scheduled_embb',
        'per_uav_scheduled_urllc', 'per_uav_overlay_count', 'per_uav_puncture_count',
        'per_uav_embb_throughput',
    ]
    metrics = {'loads': [], 'lambda': []}
    metrics.update({key: [] for key in scalar_keys + vector_keys})
    representative = {}
    for load_idx, load in enumerate(loads):
        sys_cfg, urllc_cfg, embb_cfg, algo_cfg, sim_cfg = _configure_density_scenario(
            load, base_sys, base_urllc, base_embb, base_algo, base_sim
        )
        if hasattr(base_sim, 'urllc_user_ratio'):
            sim_cfg.urllc_user_ratio = base_sim.urllc_user_ratio
        seed_base = _report_seed_base(load_idx, report_cfg)
        episodes, representative[load] = _run_episode_batch_with_representative(
            episodes_per_load,
            lambda ep, _collect_trace: _run_original_greedy_slot(
                sys_cfg,
                urllc_cfg,
                embb_cfg,
                algo_cfg,
                sim_cfg,
                seed=seed_base + ep,
                slot_index=ep,
            ),
        )
        metrics['loads'].append(float(load))
        metrics['lambda'].append(_load_to_lambda(load))
        for key in scalar_keys:
            metrics[key].append(_episode_scalar_aggregate(episodes, key, default=np.nan))
        for key in vector_keys:
            metrics[key].append(np.mean(np.stack([episode[key] for episode in episodes]), axis=0))
    _report_timing_log(f"run_greedy_sweep loads={len(loads)} episodes_per_load={episodes_per_load} sec={perf_counter() - sweep_start:.3f}")
    return metrics, representative


def _build_model_for_env(env: SRMAPPOPhaseAEnv, checkpoint_path: Path) -> Tuple[SRMAPPOConfig, SRMAPPOActorCritic]:
    from .trainer import set_phase_a_embb_power_runtime

    payload = torch.load(checkpoint_path, map_location='cpu')
    cfg = cfg_from_dict(payload.get('cfg'))
    model = SRMAPPOActorCritic(env.local_obs_dim, env.global_obs_dim, cfg)
    missing, unexpected = model.load_state_dict(payload['model_state_dict'], strict=False)
    if missing:
        _report_log(f"Checkpoint missing keys (initialized to default): {missing}")
    if unexpected:
        _report_log(f"Checkpoint has unexpected keys (ignored): {unexpected}")
    model.to(torch.device(cfg.training.device))
    model.eval()
    extra = payload.get('extra', {}) if isinstance(payload, dict) else {}
    runtime_enabled = bool(
        extra.get(
            'phase_a_embb_power_runtime_enabled',
            getattr(cfg.env, 'allow_phase_a_embb_power_adjustment', False),
        )
    )
    set_phase_a_embb_power_runtime(env, model, runtime_enabled)
    return cfg, model


def _compute_rl_embb_details(env: SRMAPPOPhaseAEnv):
    decisions = np.full((env.sys_cfg.num_uavs, env.sys_cfg.num_subcarriers, env.sys_cfg.num_minislots), 'EMPTY', dtype='<U5')
    decisions[env.mode_grid == MODE_OVERLAY] = 'NOMA'
    decisions[env.mode_grid == MODE_PUNCTURE] = 'PUNCT'
    env.allocator.urllc_power_allocation = env.scheduled_power.copy()
    env.allocator.urllc_selected_uavs = env.scheduled_uavs.copy()
    env.allocator.urllc_packet_sources = env.packet_sources.copy()
    env.allocator.urllc_timefreq_grid = env.packet_grid.copy()
    env.allocator.noma_decisions = decisions
    embb_final = env.allocator.adjust_embb_after_urllc(
        env.embb_result['rb_allocation'],
        env.channel_gains_mag_sq,
        urllc_timefreq_grid=env.packet_grid,
        noma_decisions=decisions,
        associated_uavs=env.best_uav_per_user,
    )
    embb_rates = np.asarray(embb_final['rates'], dtype=float)
    embb_uavs = np.asarray(env.embb_selected_uavs, dtype=int)
    per_uav_embb_throughput = np.bincount(embb_uavs, weights=embb_rates, minlength=env.sys_cfg.num_uavs).astype(float)
    per_uav_scheduled_embb = np.bincount(embb_uavs, weights=(embb_rates > 0).astype(float), minlength=env.sys_cfg.num_uavs).astype(float)
    cell_edge = _compute_cell_edge_served_ratio(env.sys_cfg, env.last_topology, env.best_uav_per_user, embb_rates)
    return embb_final, embb_rates, per_uav_embb_throughput, per_uav_scheduled_embb, cell_edge


def run_env_episode(
    env: SRMAPPOPhaseAEnv,
    model: Optional[SRMAPPOActorCritic],
    cfg: SRMAPPOConfig,
    seed: int,
    collect_trace: bool = False,
    use_greedy: bool = False,
    greedy_policy: str = "reference",
    cache_tag: str = "",
    reuse_static_context: bool = False,
    reset_count_contribution: float = 1.0,
) -> Dict:
    if env is None:
        raise ValueError("run_env_episode received env=None. Check greedy/mappo env initialization in report timeslot/dense paths.")
    episode_start = perf_counter()
    normalized_greedy_policy = str(greedy_policy or "reference").strip().lower()
    cache_key = None
    if _REPORT_EPISODE_CACHE_ENABLED:
        cache_key = _report_episode_cache_key(
            env,
            cfg,
            seed=seed,
            collect_trace=collect_trace,
            use_greedy=use_greedy,
            greedy_policy=normalized_greedy_policy,
            cache_tag=cache_tag,
        )
        cached = _REPORT_EPISODE_CACHE.get(cache_key)
        if cached is not None:
            _report_timing_log(
                f"run_env_episode cache_hit mode={'greedy:' + normalized_greedy_policy if use_greedy else 'policy'} seed={seed}"
            )
            return deepcopy(cached)

    previous_greedy_obs = bool(getattr(env.rl_cfg.env, "include_greedy_reference_in_obs", False))
    previous_fallback = bool(getattr(env.rl_cfg.shield, "enable_greedy_fallback", False))
    previous_mode_correction = bool(getattr(env.rl_cfg.shield, "allow_mode_correction", False))
    previous_training_progress = float(getattr(env, "training_progress_frac", 1.0))
    previous_lightweight_obs_mode = bool(getattr(env, "_lightweight_obs_mode", False))
    if model is not None:
        setattr(
            env,
            "phase_a_embb_power_enabled",
            bool(getattr(model, "phase_a_embb_power_enabled", getattr(env.rl_cfg.env, "allow_phase_a_embb_power_adjustment", False))),
        )
    env.rl_cfg.env.include_greedy_reference_in_obs = bool(use_greedy and normalized_greedy_policy == "reference")
    env._lightweight_obs_mode = bool(use_greedy)
    env.rl_cfg.shield.enable_greedy_fallback = False
    env.rl_cfg.shield.allow_mode_correction = False
    env.training_progress_frac = 1.0
    observations, _info = env.reset(seed=seed, reuse_static_context=bool(reuse_static_context))
    actor_hidden = critic_hidden = None
    if not use_greedy and model is not None:
        actor_hidden, critic_hidden = model.initial_state(batch_size=len(env.agent_ids), device=model.power_log_std.device)

    total_agent_decisions = 0
    shield_corrections = 0
    collision_rewrites = 0
    fallback_count = 0
    mode_correction_count = 0
    packet_invalid_count = 0
    mask_invalid_count = 0
    joint_rewrite_count = 0
    owner_head_active_total = 0
    embb_power_head_active_total = 0
    phase_a_embb_power_head_active_total = 0
    raw_owner_non_null_total = 0
    executed_owner_non_null_total = 0
    raw_embb_power_nonzero_total = 0
    executed_embb_power_nonzero_total = 0
    phase_a_raw_embb_power_nonzero_total = 0
    phase_a_executed_embb_power_nonzero_total = 0
    phase_a_embb_power_eligible_total = 0
    phase_a_raw_embb_power_nonzero_eligible_total = 0
    phase_a_executed_embb_power_nonzero_eligible_total = 0
    raw_executed_any_gap_total = 0
    raw_executed_mode_gap_total = 0
    raw_executed_packet_gap_total = 0
    raw_executed_power_gap_total = 0
    raw_executed_owner_gap_total = 0
    raw_executed_embb_power_gap_total = 0
    urllc_power_abs_change_sum = 0.0
    embb_power_abs_change_sum = 0.0
    overlay_candidate_pairs = 0
    overlay_feasible_pairs = 0
    overlay_selected_pairs = 0
    greedy_phase_a_decisions = 0
    greedy_noop_selected = 0.0
    greedy_admit_selected = 0.0
    greedy_overlay_selected = 0.0
    greedy_puncture_selected = 0.0
    greedy_selected_retention_sum = 0.0
    greedy_selected_loss_sum = 0.0
    greedy_selected_throughput_sum = 0.0
    greedy_selected_reliability_sum = 0.0
    greedy_selected_embb_min_rate_ok_sum = 0.0
    greedy_feasible_admit_count_sum = 0.0
    greedy_no_feasible_admit_sum = 0.0
    greedy_keep_only_when_no_feasible_admit_sum = 0.0
    greedy_rejected_when_noop_better_sum = 0.0
    greedy_noop_available_sum = 0.0
    greedy_noop_better_sum = 0.0
    greedy_requires_feasible_only = 0.0
    greedy_hf_raw_count_sum = 0.0
    greedy_hf_admissible_count_sum = 0.0
    greedy_hf_evaluated_count_sum = 0.0
    greedy_hf_feasible_count_sum = 0.0
    greedy_hf_selected_count_sum = 0.0
    greedy_hf_reject_gate_assoc_sum = 0.0
    greedy_hf_reject_gate_queue_sum = 0.0
    greedy_hf_reject_gate_mode_sum = 0.0
    greedy_hf_reject_gate_owner_sum = 0.0
    greedy_hf_reject_gate_rb_local_sum = 0.0
    greedy_hf_reject_mode_overlay_rel_fail_sum = 0.0
    greedy_hf_reject_mode_overlay_sic_fail_sum = 0.0
    greedy_hf_reject_mode_gain_ratio_fail_sum = 0.0
    greedy_hf_reject_mode_owner_pool_missing_sum = 0.0
    greedy_hf_reject_mode_owner_unresolved_due_to_mode_fail_sum = 0.0
    greedy_hf_reject_mode_puncture_rel_fail_sum = 0.0
    greedy_hf_reject_mode_puncture_sic_fail_sum = 0.0
    greedy_hf_reject_mode_puncture_owner_missing_sum = 0.0
    greedy_hf_reject_owner_missing_sum = 0.0
    greedy_hf_reject_owner_mismatch_sum = 0.0
    greedy_hf_gate_overlay_rel_fail_margin_sum = 0.0
    greedy_hf_gate_overlay_sic_fail_margin_db_sum = 0.0
    greedy_hf_gate_puncture_rel_fail_margin_sum = 0.0
    greedy_hf_gate_target_reliability_sum = 0.0
    greedy_hf_gate_target_sic_db_sum = 0.0
    greedy_hf_gate_overlay_rel_fail_snir_db_sum = 0.0
    greedy_hf_gate_overlay_sic_fail_post_sic_db_sum = 0.0
    greedy_hf_gate_puncture_rel_fail_snir_db_sum = 0.0
    greedy_hf_overlay_sic_trace_pre_sinr_db_sum = 0.0
    greedy_hf_overlay_sic_trace_post_sinr_db_sum = 0.0
    greedy_hf_overlay_sic_trace_noise_power_sum = 0.0
    greedy_hf_overlay_sic_trace_intercell_interference_sum = 0.0
    greedy_hf_overlay_sic_trace_local_interference_sum = 0.0
    greedy_hf_overlay_sic_trace_residual_interference_sum = 0.0
    greedy_hf_overlay_sic_trace_residual_ratio_sum = 0.0
    greedy_hf_overlay_presinr_raw_lt_m10_sum = 0.0
    greedy_hf_overlay_presinr_raw_m10_m6_sum = 0.0
    greedy_hf_overlay_presinr_raw_m6_m2_sum = 0.0
    greedy_hf_overlay_presinr_raw_ge_m2_sum = 0.0
    greedy_hf_overlay_presinr_kept_low_ratio_sum = 0.0
    greedy_hf_overlay_presinr_eval_low_ratio_sum = 0.0
    greedy_hf_quality_priority_enabled_sum = 0.0
    greedy_hf_quality_raw_high_sum = 0.0
    greedy_hf_quality_raw_borderline_sum = 0.0
    greedy_hf_quality_raw_risk_sum = 0.0
    greedy_hf_quality_kept_high_sum = 0.0
    greedy_hf_quality_kept_borderline_sum = 0.0
    greedy_hf_quality_kept_risk_sum = 0.0
    greedy_hf_quality_eval_high_sum = 0.0
    greedy_hf_quality_eval_borderline_sum = 0.0
    greedy_hf_quality_eval_risk_sum = 0.0
    greedy_hf_quality_selected_high_sum = 0.0
    greedy_hf_quality_selected_borderline_sum = 0.0
    greedy_hf_quality_selected_risk_sum = 0.0
    greedy_hf_overlay_blacklist_cell_ratio_sum = 0.0
    greedy_hf_overlay_blacklist_candidate_block_ratio_sum = 0.0
    greedy_hf_overlay_blacklist_saved_mode_fail_ratio_sum = 0.0
    greedy_hf_overlay_only_rb_reservation_enabled_sum = 0.0
    greedy_hf_overlay_only_rb_reservation_ratio_sum = 0.0
    greedy_hf_overlay_only_rb_reserved_count_sum = 0.0
    greedy_hf_overlay_only_rb_reserved_cell_ratio_sum = 0.0
    greedy_hf_overlay_only_rb_mode_block_ratio_sum = 0.0
    phase_a_embb_power_anchor_binding_count = 0
    phase_a_embb_power_anchor_binding_denom = 0
    trace = []
    done = False
    action_select_sec_total = 0.0
    action_resolve_sec_total = 0.0
    env_step_sec_total = 0.0

    while not done:
        current_obs = observations
        planning_phase = all(
            bool(current_obs[agent_id].metadata.get("planning_phase", 0.0))
            for agent_id in env.agent_ids
        )
        cell_index = env.current_cell_index if not planning_phase else -1
        if planning_phase:
            minislot = -1
            rb = env._current_planning_rb()
        else:
            minislot, rb = env._current_cell()

        if (not planning_phase) and phase_a_embb_power_anchor_enabled(cfg, iteration=1):
            _anchor_target, anchor_weight = _phase_a_embb_power_anchor_targets(env, current_obs, cfg, iteration=1)
            for idx, agent_id in enumerate(env.agent_ids):
                head_activity = env.action_head_activity(current_obs[agent_id])
                if not bool(head_activity.get("phase_a_embb_power_active", False)):
                    continue
                phase_a_embb_power_anchor_binding_denom += 1
                if idx < len(anchor_weight) and float(anchor_weight[idx]) > 1e-9:
                    phase_a_embb_power_anchor_binding_count += 1
        greedy_debug = {}
        _action_select_t0 = perf_counter()
        if use_greedy:
            if normalized_greedy_policy == "channel_only":
                joint_actions = _channel_only_actions(env, current_obs)
            elif normalized_greedy_policy in {"hard_feasible", "hard_feasible_throughput"}:
                joint_actions, greedy_debug = _hard_feasible_throughput_actions(env, current_obs)
            elif normalized_greedy_policy in {"global_frontier", "global_greedy"}:
                joint_actions, greedy_debug = _global_frontier_actions(env, current_obs)
            elif normalized_greedy_policy == "throughput_feasible":
                joint_actions, greedy_debug = _throughput_feasible_actions(env, current_obs)
            elif normalized_greedy_policy == "throughput_biased":
                joint_actions, greedy_debug = _throughput_biased_actions(env, current_obs)
            elif normalized_greedy_policy in {"myopic", "myopic_throughput"}:
                joint_actions, greedy_debug = _myopic_throughput_actions(env, current_obs)
            elif normalized_greedy_policy == "throughput_only":
                joint_actions, greedy_debug = _throughput_only_actions(env, current_obs)
            elif normalized_greedy_policy in {"rate_loss_min", "global_rate_loss", "sumrate_minloss", "global_sumrate_minloss"}:
                joint_actions, greedy_debug = _rate_loss_min_actions(env, current_obs)
            elif normalized_greedy_policy in {"force_admit_minloss", "rate_loss_force_admit", "sumrate_force_admit"}:
                joint_actions, greedy_debug = _force_admit_minloss_actions(env, current_obs)
            else:
                joint_actions = _greedy_actions(env, current_obs)
        else:
            joint_actions, actor_hidden, critic_hidden = _policy_actions(env, model, current_obs, actor_hidden, critic_hidden)
        action_select_sec_total += float(perf_counter() - _action_select_t0)
        if greedy_debug and not planning_phase:
            for agent_id in env.agent_ids:
                debug = greedy_debug.get(agent_id)
                if not debug:
                    continue
                greedy_phase_a_decisions += int(debug.get("phase_a_decision", 0.0) > 0.5)
                greedy_noop_selected += float(debug.get("noop_selected", 0.0))
                greedy_admit_selected += float(debug.get("admit_selected", 0.0))
                greedy_overlay_selected += float(debug.get("overlay_selected", 0.0))
                greedy_puncture_selected += float(debug.get("puncture_selected", 0.0))
                greedy_selected_retention_sum += float(debug.get("selected_retention", 0.0))
                greedy_selected_loss_sum += float(debug.get("selected_loss", 0.0))
                greedy_selected_throughput_sum += float(debug.get("selected_throughput", 0.0))
                greedy_selected_reliability_sum += float(
                    debug.get("selected_reliability", debug.get("reliability", 0.0))
                )
                greedy_selected_embb_min_rate_ok_sum += float(debug.get("selected_embb_min_rate_ok", 1.0))
                greedy_feasible_admit_count_sum += float(debug.get("feasible_admit_count", 0.0))
                if float(debug.get("feasible_admit_count", 0.0)) <= 0.5:
                    greedy_no_feasible_admit_sum += 1.0
                greedy_keep_only_when_no_feasible_admit_sum += float(
                    debug.get("keep_selected_due_to_no_feasible_admit", 0.0)
                )
                greedy_rejected_when_noop_better_sum += float(debug.get("rejected_when_noop_better", 0.0))
                greedy_noop_available_sum += float(debug.get("noop_available", 0.0))
                greedy_noop_better_sum += float(debug.get("no_op_better_than_best_admit", 0.0))
                greedy_requires_feasible_only = max(
                    greedy_requires_feasible_only,
                    float(debug.get("current_env_requires_feasible_admission_only", 0.0)),
                )
                greedy_hf_raw_count_sum += float(debug.get("greedy_hf_raw_count", 0.0))
                greedy_hf_admissible_count_sum += float(debug.get("greedy_hf_admissible_count", 0.0))
                greedy_hf_evaluated_count_sum += float(debug.get("greedy_hf_evaluated_count", 0.0))
                greedy_hf_feasible_count_sum += float(debug.get("greedy_hf_feasible_count", 0.0))
                greedy_hf_selected_count_sum += float(debug.get("greedy_hf_selected_count", 0.0))
                greedy_hf_reject_gate_assoc_sum += float(debug.get("greedy_hf_reject_by_gate_association", 0.0))
                greedy_hf_reject_gate_queue_sum += float(debug.get("greedy_hf_reject_by_gate_queue_active", 0.0))
                greedy_hf_reject_gate_mode_sum += float(debug.get("greedy_hf_reject_by_gate_mode_admissible", 0.0))
                greedy_hf_reject_gate_owner_sum += float(debug.get("greedy_hf_reject_by_gate_owner_admissible", 0.0))
                greedy_hf_reject_gate_rb_local_sum += float(debug.get("greedy_hf_reject_by_gate_rb_local", 0.0))
                greedy_hf_reject_mode_overlay_rel_fail_sum += float(debug.get("greedy_hf_reject_by_mode_overlay_rel_fail", 0.0))
                greedy_hf_reject_mode_overlay_sic_fail_sum += float(debug.get("greedy_hf_reject_by_mode_overlay_sic_fail", 0.0))
                greedy_hf_reject_mode_gain_ratio_fail_sum += float(debug.get("greedy_hf_reject_by_mode_gain_ratio_fail", 0.0))
                greedy_hf_reject_mode_owner_pool_missing_sum += float(debug.get("greedy_hf_reject_by_mode_owner_pool_missing", debug.get("greedy_hf_reject_by_mode_overlay_owner_missing", 0.0)))
                greedy_hf_reject_mode_owner_unresolved_due_to_mode_fail_sum += float(debug.get("greedy_hf_reject_by_mode_owner_unresolved_due_to_mode_fail", 0.0))
                greedy_hf_reject_mode_puncture_rel_fail_sum += float(debug.get("greedy_hf_reject_by_mode_puncture_rel_fail", 0.0))
                greedy_hf_reject_mode_puncture_sic_fail_sum += float(debug.get("greedy_hf_reject_by_mode_puncture_sic_fail", 0.0))
                greedy_hf_reject_mode_puncture_owner_missing_sum += float(debug.get("greedy_hf_reject_by_mode_puncture_owner_missing", 0.0))
                greedy_hf_reject_owner_missing_sum += float(debug.get("greedy_hf_reject_by_owner_missing", 0.0))
                greedy_hf_reject_owner_mismatch_sum += float(debug.get("greedy_hf_reject_by_owner_mismatch", 0.0))
                greedy_hf_gate_overlay_rel_fail_margin_sum += float(debug.get("greedy_hf_gate_overlay_rel_fail_margin_mean", 0.0))
                greedy_hf_gate_overlay_sic_fail_margin_db_sum += float(debug.get("greedy_hf_gate_overlay_sic_fail_margin_db_mean", 0.0))
                greedy_hf_gate_puncture_rel_fail_margin_sum += float(debug.get("greedy_hf_gate_puncture_rel_fail_margin_mean", 0.0))
                greedy_hf_gate_target_reliability_sum += float(debug.get("greedy_hf_gate_target_reliability", 0.0))
                greedy_hf_gate_target_sic_db_sum += float(debug.get("greedy_hf_gate_target_sic_snir_db", 0.0))
                greedy_hf_gate_overlay_rel_fail_snir_db_sum += float(debug.get("greedy_hf_gate_overlay_rel_fail_snir_db_mean", 0.0))
                greedy_hf_gate_overlay_sic_fail_post_sic_db_sum += float(debug.get("greedy_hf_gate_overlay_sic_fail_post_sic_db_mean", 0.0))
                greedy_hf_gate_puncture_rel_fail_snir_db_sum += float(debug.get("greedy_hf_gate_puncture_rel_fail_snir_db_mean", 0.0))
                greedy_hf_overlay_sic_trace_pre_sinr_db_sum += float(debug.get("greedy_hf_overlay_sic_trace_pre_sinr_db_mean", 0.0))
                greedy_hf_overlay_sic_trace_post_sinr_db_sum += float(debug.get("greedy_hf_overlay_sic_trace_post_sinr_db_mean", 0.0))
                greedy_hf_overlay_sic_trace_noise_power_sum += float(debug.get("greedy_hf_overlay_sic_trace_noise_power_mean", 0.0))
                greedy_hf_overlay_sic_trace_intercell_interference_sum += float(debug.get("greedy_hf_overlay_sic_trace_intercell_interference_mean", 0.0))
                greedy_hf_overlay_sic_trace_local_interference_sum += float(debug.get("greedy_hf_overlay_sic_trace_local_interference_mean", 0.0))
                greedy_hf_overlay_sic_trace_residual_interference_sum += float(debug.get("greedy_hf_overlay_sic_trace_residual_sic_interference_mean", 0.0))
                greedy_hf_overlay_sic_trace_residual_ratio_sum += float(debug.get("greedy_hf_overlay_sic_trace_sic_residual_ratio", 0.0))
                greedy_hf_overlay_presinr_raw_lt_m10_sum += float(debug.get("greedy_hf_overlay_presinr_raw_lt_m10", 0.0))
                greedy_hf_overlay_presinr_raw_m10_m6_sum += float(debug.get("greedy_hf_overlay_presinr_raw_m10_m6", 0.0))
                greedy_hf_overlay_presinr_raw_m6_m2_sum += float(debug.get("greedy_hf_overlay_presinr_raw_m6_m2", 0.0))
                greedy_hf_overlay_presinr_raw_ge_m2_sum += float(debug.get("greedy_hf_overlay_presinr_raw_ge_m2", 0.0))
                greedy_hf_overlay_presinr_kept_low_ratio_sum += float(debug.get("greedy_hf_overlay_presinr_kept_low_ratio", 0.0))
                greedy_hf_overlay_presinr_eval_low_ratio_sum += float(debug.get("greedy_hf_overlay_presinr_eval_low_ratio", 0.0))
                greedy_hf_quality_priority_enabled_sum += float(debug.get("greedy_hf_quality_priority_enabled", 0.0))
                greedy_hf_quality_raw_high_sum += float(debug.get("greedy_hf_quality_raw_high", 0.0))
                greedy_hf_quality_raw_borderline_sum += float(debug.get("greedy_hf_quality_raw_borderline", 0.0))
                greedy_hf_quality_raw_risk_sum += float(debug.get("greedy_hf_quality_raw_risk", 0.0))
                greedy_hf_quality_kept_high_sum += float(debug.get("greedy_hf_quality_kept_high", 0.0))
                greedy_hf_quality_kept_borderline_sum += float(debug.get("greedy_hf_quality_kept_borderline", 0.0))
                greedy_hf_quality_kept_risk_sum += float(debug.get("greedy_hf_quality_kept_risk", 0.0))
                greedy_hf_quality_eval_high_sum += float(debug.get("greedy_hf_quality_eval_high", 0.0))
                greedy_hf_quality_eval_borderline_sum += float(debug.get("greedy_hf_quality_eval_borderline", 0.0))
                greedy_hf_quality_eval_risk_sum += float(debug.get("greedy_hf_quality_eval_risk", 0.0))
                greedy_hf_quality_selected_high_sum += float(debug.get("greedy_hf_quality_selected_high", 0.0))
                greedy_hf_quality_selected_borderline_sum += float(debug.get("greedy_hf_quality_selected_borderline", 0.0))
                greedy_hf_quality_selected_risk_sum += float(debug.get("greedy_hf_quality_selected_risk", 0.0))
                greedy_hf_overlay_blacklist_cell_ratio_sum += float(debug.get("greedy_hf_overlay_blacklist_cell_ratio", 0.0))
                greedy_hf_overlay_blacklist_candidate_block_ratio_sum += float(debug.get("greedy_hf_overlay_blacklist_candidate_block_ratio", 0.0))
                greedy_hf_overlay_blacklist_saved_mode_fail_ratio_sum += float(debug.get("greedy_hf_overlay_blacklist_saved_mode_fail_ratio", 0.0))
                greedy_hf_overlay_only_rb_reservation_enabled_sum += float(
                    debug.get("greedy_hf_overlay_only_rb_reservation_enabled", 0.0)
                )
                greedy_hf_overlay_only_rb_reservation_ratio_sum += float(
                    debug.get("greedy_hf_overlay_only_rb_reservation_ratio", 0.0)
                )
                greedy_hf_overlay_only_rb_reserved_count_sum += float(
                    debug.get("greedy_hf_overlay_only_rb_reserved_count", 0.0)
                )
                greedy_hf_overlay_only_rb_reserved_cell_ratio_sum += float(
                    debug.get("greedy_hf_overlay_only_rb_reserved_cell_ratio", 0.0)
                )
                greedy_hf_overlay_only_rb_mode_block_ratio_sum += float(
                    debug.get("greedy_hf_overlay_only_rb_mode_block_ratio", 0.0)
                )
        _action_resolve_t0 = perf_counter()
        if planning_phase:
            resolved = {
                agent_id: env._raw_action_to_shielded_action(joint_actions[agent_id], current_obs[agent_id])
                for agent_id in env.agent_ids
            }
        else:
            resolved = env._resolve_executed_actions(
                joint_actions,
                current_obs,
                minislot=minislot,
                rb=rb,
            )
        action_resolve_sec_total += float(perf_counter() - _action_resolve_t0)

        for agent_id in env.agent_ids:
            obs = current_obs[agent_id]
            raw = joint_actions[agent_id]
            final = resolved[agent_id]
            total_agent_decisions += 1
            if not planning_phase:
                overlay_candidate_pairs += len(obs.candidates)
                overlay_feasible_pairs += sum(int(candidate.overlay_feasible) for candidate in obs.candidates)
                if final.candidate is not None and final.action.mode == MODE_OVERLAY:
                    overlay_selected_pairs += 1
            activity = env.action_head_activity(obs)
            if activity["owner_active"]:
                owner_head_active_total += 1
                owner_space = str(getattr(cfg.action, "embb_owner_action_space", "candidate_option_with_null") or "candidate_option_with_null").strip().lower()
                owner_mask = np.asarray(obs.masks.embb_owner_mask, dtype=float)
                if owner_space == "global_owner_id_no_null":
                    raw_idx = int(raw.embb_owner_option)
                    exe_idx = int(final.action.embb_owner_option)
                    raw_owner_non_null_total += int(0 <= raw_idx < owner_mask.size and owner_mask[raw_idx] > 0.5)
                    executed_owner_non_null_total += int(0 <= exe_idx < owner_mask.size and owner_mask[exe_idx] > 0.5)
                else:
                    raw_owner_non_null_total += int(int(raw.embb_owner_option) > 0)
                    executed_owner_non_null_total += int(int(final.action.embb_owner_option) > 0)
            if activity["embb_power_active"]:
                embb_power_head_active_total += 1
                raw_embb_power_nonzero_total += int(abs(float(raw.embb_power_delta)) > 1e-3)
                executed_embb_power_nonzero_total += int(abs(float(final.action.embb_power_delta)) > 1e-3)
            if activity["phase_a_embb_power_active"]:
                phase_a_embb_power_head_active_total += 1
                phase_a_raw_embb_power_nonzero_total += int(abs(float(raw.embb_power_delta)) > 1e-3)
                phase_a_executed_embb_power_nonzero_total += int(abs(float(final.action.embb_power_delta)) > 1e-3)
                # Eligibility: exclude inherently-ineligible cells (inactive/no-owner/no-eMBB) from the denom.
                try:
                    pinfo = dict(getattr(final, "phase_a_embb_power_info", {}) or {})
                except Exception:
                    pinfo = {}
                reason = str(pinfo.get("zeroed_reason", "") or "").strip().lower()
                if reason not in {"inactive_head", "no_embb_active", "no_owner", "invalid_owner"}:
                    phase_a_embb_power_eligible_total += 1
                    phase_a_raw_embb_power_nonzero_eligible_total += int(abs(float(raw.embb_power_delta)) > 1e-3)
                    phase_a_executed_embb_power_nonzero_eligible_total += int(abs(float(final.action.embb_power_delta)) > 1e-3)
            diff_flags = env.action_diff_flags(raw, final.action)
            raw_executed_mode_gap_total += int(diff_flags["mode"])
            raw_executed_packet_gap_total += int(diff_flags["packet"])
            raw_executed_power_gap_total += int(diff_flags["power"])
            raw_executed_owner_gap_total += int(diff_flags["owner"])
            raw_executed_embb_power_gap_total += int(diff_flags["embb_power"])
            urllc_power_abs_change_sum += float(abs(float(raw.power_delta) - float(final.action.power_delta)))
            embb_power_abs_change_sum += float(abs(float(raw.embb_power_delta) - float(final.action.embb_power_delta)))
            changed = any(diff_flags.values())
            raw_executed_any_gap_total += int(changed)
            if changed:
                shield_corrections += 1
            if final.collision_rewritten:
                collision_rewrites += 1
            if final.used_greedy_fallback:
                fallback_count += 1
            if final.mode_corrected:
                mode_correction_count += 1
            if final.packet_invalid_fallback:
                packet_invalid_count += 1
            if final.mask_invalid_fallback:
                mask_invalid_count += 1
            if final.joint_reliability_rewritten:
                joint_rewrite_count += 1

        _env_step_t0 = perf_counter()
        observations, rewards, dones, infos = env.step(
            joint_actions,
            prebuilt_observations=current_obs,
            pre_resolved_actions=resolved,
        )
        env_step_sec_total += float(perf_counter() - _env_step_t0)
        done = all(dones.values())
        if collect_trace and not planning_phase:
            summary = env.summarize_episode()
            utility_gaps = [
                float(resolved[agent_id].utility - current_obs[agent_id].greedy_reference_utility)
                for agent_id in env.agent_ids
            ]
            trace_meta = dict(current_obs[env.agent_ids[0]].metadata or {})
            cell_debug = dict(greedy_debug.get(env.agent_ids[0], {}) or {})
            trace.append({
                'cell_index': cell_index,
                'minislot': minislot,
                'rb': rb,
                'arrivals': float(env.num_packets),
                'scheduled_packets': float(summary['scheduled_packets']),
                'target_quota': float(trace_meta.get('target_quota_packets', 0.0)),
                'target_quota_normalized': float(trace_meta.get('target_quota_normalized', 0.0)),
                'admitted_so_far': float(trace_meta.get('admitted_so_far_packets', 0.0)),
                'admitted_so_far_normalized': float(trace_meta.get('admitted_so_far_normalized', 0.0)),
                'quota_gap': float(trace_meta.get('quota_gap_packets', 0.0)),
                'quota_gap_normalized': float(trace_meta.get('quota_gap_normalized', 0.0)),
                'overlay_count': float(summary['overlay_count']),
                'puncture_count': float(summary['puncture_count']),
                'embb_rate': float(summary['embb_total_rate']),
                'total_power': float(summary['total_power']),
                'utility_gap': float(np.mean(utility_gaps)) if utility_gaps else 0.0,
                'raw_candidate_count': float(cell_debug.get('greedy_hf_raw_count', 0.0)),
                'admissible_candidate_count': float(cell_debug.get('greedy_hf_admissible_count', 0.0)),
                'evaluated_candidate_count': float(cell_debug.get('greedy_hf_evaluated_count', 0.0)),
                'feasible_candidate_count': float(cell_debug.get('greedy_hf_feasible_count', 0.0)),
                'selected_mode': float(cell_debug.get('selected_mode', 0.0)),
                'selected_reliability': float(cell_debug.get('selected_reliability', 0.0)),
                'selected_gain_ratio': float(cell_debug.get('selected_gain_ratio', 0.0)),
                'selected_quality_tier': float(cell_debug.get('selected_quality_tier', 0.0)),
                'top1_mode': float(cell_debug.get('greedy_hf_top1_mode', 0.0)),
                'top2_mode': float(cell_debug.get('greedy_hf_top2_mode', 0.0)),
                'top12_global_loss_gap': float(cell_debug.get('greedy_hf_top12_global_loss_gap', 0.0)),
                'quality_raw_high': float(cell_debug.get('greedy_hf_quality_raw_high', 0.0)),
                'quality_raw_borderline': float(cell_debug.get('greedy_hf_quality_raw_borderline', 0.0)),
                'quality_raw_risk': float(cell_debug.get('greedy_hf_quality_raw_risk', 0.0)),
                'quality_selected_high': float(cell_debug.get('greedy_hf_quality_selected_high', 0.0)),
                'quality_selected_borderline': float(cell_debug.get('greedy_hf_quality_selected_borderline', 0.0)),
                'quality_selected_risk': float(cell_debug.get('greedy_hf_quality_selected_risk', 0.0)),
                'mode_overlay_rel_fail': float(cell_debug.get('greedy_hf_reject_by_mode_overlay_rel_fail', 0.0)),
                'mode_gain_ratio_fail': float(cell_debug.get('greedy_hf_reject_by_mode_gain_ratio_fail', 0.0)),
                'mode_puncture_rel_fail': float(cell_debug.get('greedy_hf_reject_by_mode_puncture_rel_fail', 0.0)),
            })

    summary = env.summarize_episode()
    topo = env.last_topology if isinstance(getattr(env, "last_topology", None), dict) else {}
    user_positions = np.asarray(topo.get("user_positions", []), dtype=float)
    uav_positions = np.asarray(topo.get("uav_positions", []), dtype=float)
    embb_final, embb_rates, per_uav_embb_throughput, per_uav_scheduled_embb, cell_edge = _compute_rl_embb_details(env)
    associated_embb = np.asarray(env.associated_embb_counts, dtype=float)
    associated_urllc = np.asarray(env.associated_urllc_counts, dtype=float)
    scheduled_urllc = np.asarray(env.scheduled_counts, dtype=float)
    overlay_counts = np.asarray(env.overlay_counts, dtype=float)
    puncture_counts = np.asarray(env.puncture_counts, dtype=float)
    embb_min_rate_bps = float(getattr(env.embb_cfg, 'min_rate_per_user_bps', 0.0) or 0.0)
    embb_positive_rate_ratio = float(summary['embb_service_ratio'])
    embb_min_rate_satisfaction_ratio = _compute_embb_min_rate_satisfaction_ratio(embb_rates, embb_min_rate_bps)
    embb_min_rate_satisfied_user_count = _compute_embb_min_rate_satisfied_user_count(embb_rates, embb_min_rate_bps)
    embb_rates_eff = np.asarray(summary.get('embb_user_rates_after_puncture_deduction', []), dtype=float)
    embb_min_rate_satisfied_user_count_after_puncture_deduction = _compute_embb_min_rate_satisfied_user_count(
        embb_rates_eff,
        embb_min_rate_bps,
    )
    cell_edge_min_rate_satisfaction_ratio = _compute_cell_edge_min_rate_satisfaction_ratio(
        env.sys_cfg,
        env.last_topology,
        env.best_uav_per_user,
        embb_rates,
        embb_min_rate_bps,
    )
    env.rl_cfg.env.include_greedy_reference_in_obs = previous_greedy_obs
    env.rl_cfg.shield.enable_greedy_fallback = previous_fallback
    env.rl_cfg.shield.allow_mode_correction = previous_mode_correction
    env.training_progress_frac = previous_training_progress
    env._lightweight_obs_mode = previous_lightweight_obs_mode
    # IMPORTANT: pass through the full `env.summarize_episode()` payload so fast-debug scalar_keys can be
    # plotted reliably (Phase-A suppress reasons, intercell step penalty components, load=10 diagnostics, etc.).
    result = {
        **dict(summary),
        'phase': str(summary.get('phase', env.rl_cfg.env.phase)),
        'learn_embb_baseline': float(summary.get('learn_embb_baseline', float(bool(env.rl_cfg.env.learn_embb_baseline)))),
        'learn_phase0_embb_power': float(summary.get('learn_phase0_embb_power', float(bool(getattr(env.rl_cfg.env, 'learn_phase0_embb_power', True))))),
        'allow_phase_a_embb_power_adjustment': float(summary.get('allow_phase_a_embb_power_adjustment', float(bool(env.rl_cfg.env.allow_phase_a_embb_power_adjustment)))),
        'phase_a_embb_power_runtime_enabled': float(summary.get('phase_a_embb_power_runtime_enabled', 0.0)),
        'phase_a_embb_power_anchor_binding_ratio': float(
            phase_a_embb_power_anchor_binding_count / max(phase_a_embb_power_anchor_binding_denom, 1)
        ),
        'phaseA_pow_anchor_binding_ratio': float(
            phase_a_embb_power_anchor_binding_count / max(phase_a_embb_power_anchor_binding_denom, 1)
        ),
        'enable_action_masking': float(summary.get('enable_action_masking', float(bool(env.rl_cfg.shield.enable_action_masking)))),
        'enable_feasibility_shield': float(summary.get('enable_feasibility_shield', float(bool(env.rl_cfg.shield.enable_feasibility_shield)))),
        'apply_joint_reliability_rewrite': float(summary.get('apply_joint_reliability_rewrite', float(bool(env.rl_cfg.shield.apply_joint_reliability_rewrite)))),
        'enable_greedy_fallback': float(summary.get('enable_greedy_fallback', float(bool(env.rl_cfg.shield.enable_greedy_fallback)))),
        'embb_rate': float(summary['embb_total_rate']),
        'embb_rate_after_local_puncture_deduction': float(
            summary.get(
                'embb_total_rate_after_puncture_deduction',
                summary.get('embb_rate_after_local_puncture_deduction', summary['embb_total_rate']),
            )
        ),
        'embb_total_rate_after_puncture_deduction': float(
            summary.get(
                'embb_total_rate_after_puncture_deduction',
                summary.get('embb_rate_after_local_puncture_deduction', summary['embb_total_rate']),
            )
        ),
        # Same-scenario eMBB baseline before URLLC puncture/admission impact.
        'embb_rate_pre_urllc_admission': float(
            summary.get(
                'embb_rate_raw_before_local_puncture_deduction',
                summary.get('embb_rate_with_intercell', summary.get('embb_total_rate', 0.0)),
            )
        ),
        'embb_user_rate': float(summary['embb_user_rate_mean']),
        'embb_user_rate_mean_after_puncture_deduction': float(
            summary.get('embb_user_rate_mean_after_puncture_deduction', summary['embb_user_rate_mean'])
        ),
        'embb_service_ratio': float(summary['embb_service_ratio']),
        'embb_service_ratio_after_puncture_deduction': float(
            summary.get('embb_service_ratio_after_puncture_deduction', summary['embb_service_ratio'])
        ),
        'embb_positive_rate_ratio': embb_positive_rate_ratio,
        'embb_min_rate_satisfaction_ratio': float(embb_min_rate_satisfaction_ratio),
        'embb_min_rate_satisfied_user_count': float(embb_min_rate_satisfied_user_count),
        'embb_min_rate_satisfied_user_count_after_puncture_deduction': float(
            embb_min_rate_satisfied_user_count_after_puncture_deduction
            if embb_rates_eff.size > 0 else (
                float(summary.get('embb_min_rate_satisfaction_after_puncture_deduction', embb_min_rate_satisfaction_ratio))
                * float(summary.get('embb_user_count', 0.0))
            )
        ),
        'embb_user_count': float(summary.get('embb_user_count', 0.0)),
        'urllc_user_count': float(summary.get('urllc_user_count', 0.0)),
        'embb_urllc_user_ratio': float(summary.get('embb_urllc_user_ratio', 0.0)),
        'effective_lambda_per_user': float(summary.get('effective_lambda_per_user', 0.0)),
        'effective_lambda_per_user_per_minislot': float(summary.get('effective_lambda_per_user_per_minislot', 0.0)),
        'expected_total_arrivals_per_minislot': float(summary.get('expected_total_arrivals_per_minislot', 0.0)),
        'expected_total_arrivals_per_episode': float(summary.get('expected_total_arrivals_per_episode', 0.0)),
        'urllc_admission': float(summary['urllc_admission_rate']),
        'admitted_urllc_reliability': float(
            summary.get('admitted_urllc_reliability', summary.get('urllc_success_rate', np.nan))
        ),
        'urllc_reliability': float(
            summary.get('admitted_urllc_reliability', summary.get('urllc_success_rate', np.nan))
        ),
        'effective_urllc_success_over_arrivals': float(
            summary.get('effective_urllc_success_over_arrivals', summary.get('urllc_success_rate', np.nan))
        ),
        'empty_admission_case': float(summary.get('empty_admission_case', 0.0)),
        # URLLC throughput definition inputs (slot-based; allow robust plot-side recomputation).
        'urllc_slot_duration_s': float(summary.get('urllc_slot_duration_s', 1.0e-3)),
        'urllc_packet_bits_mean': float(summary.get('urllc_packet_bits_mean', 160.0)),
        # Slot-based URLLC throughput estimate: scheduled packets × avg packet bits / 1 ms slot.
        'urllc_throughput_bps_slot_est': float(summary.get('urllc_throughput_bps_slot_est', summary.get('urllc_throughput_bps_est', 0.0))),
        'urllc_throughput_mbps_slot_est': float(summary.get('urllc_throughput_mbps_slot_est', summary.get('urllc_throughput_mbps_est', 0.0))),
        # Backward-compatible aliases (deprecated).
        'urllc_throughput_bps_est': float(summary.get('urllc_throughput_bps_slot_est', summary.get('urllc_throughput_bps_est', 0.0))),
        'active_packets': float(summary['active_packets']),
        'scheduled_packets': float(summary['scheduled_packets']),
        'scheduled_packets_per_uav': float(summary.get('scheduled_packets_per_uav', 0.0)),
        'overlay_count': float(summary['overlay_count']),
        'puncture_count': float(summary['puncture_count']),
        'total_power': float(summary['total_power']),
        'embb_power': float(summary.get('embb_power', np.sum(embb_final['power_allocation']))),
        'urllc_power': float(summary.get('urllc_power', np.sum(env.scheduled_power) / max(env.sys_cfg.num_minislots, 1))),
        'throughput_per_watt': float(summary.get('throughput_per_watt', 0.0)),
        'avg_throughput_per_served_embb_user': float(summary.get('avg_throughput_per_served_embb_user', 0.0)),
        'embb_served_user_count': float(summary.get('embb_served_user_count', 0.0)),
        # Inter-cell interference rate loss (counterfactual: intercell term set to 0; local residual preserved).
        'embb_rate_with_intercell': float(summary.get('embb_rate_with_intercell', summary.get('embb_total_rate', 0.0))),
        'embb_rate_without_intercell_est': float(summary.get('embb_rate_without_intercell_est', summary.get('embb_total_rate', 0.0))),
        'embb_rate_loss_due_to_intercell': float(summary.get('embb_rate_loss_due_to_intercell', 0.0)),
        'embb_rate_loss_due_to_intercell_ratio': float(summary.get('embb_rate_loss_due_to_intercell_ratio', 0.0)),
        'overlay_rate_with_intercell': float(summary.get('overlay_rate_with_intercell', 0.0)),
        'overlay_rate_without_intercell_est': float(summary.get('overlay_rate_without_intercell_est', 0.0)),
        'overlay_rate_loss_due_to_intercell': float(summary.get('overlay_rate_loss_due_to_intercell', 0.0)),
        'puncture_rate_with_intercell': float(summary.get('puncture_rate_with_intercell', 0.0)),
        'puncture_rate_without_intercell_est': float(summary.get('puncture_rate_without_intercell_est', 0.0)),
        'puncture_rate_loss_due_to_intercell': float(summary.get('puncture_rate_loss_due_to_intercell', 0.0)),
        'mean_intercell_interference_power': float(summary.get('mean_intercell_interference_power', 0.0)),
        'mean_intercell_interference_mw': float(summary.get('mean_intercell_interference_mw', 0.0)),
        'mean_intercell_interference_dbm': float(summary.get('mean_intercell_interference_dbm', 0.0)),
        'intercell_interference_nonzero_ratio': float(summary.get('intercell_interference_nonzero_ratio', 0.0)),
        'overlay_intercell_interference_mw': float(summary.get('overlay_intercell_interference_mw', 0.0)),
        'puncture_intercell_interference_mw': float(summary.get('puncture_intercell_interference_mw', 0.0)),
        'overlay_ratio': float(summary['overlay_ratio']),
        'puncture_ratio': float(summary['puncture_ratio']),
        'overlay_selection_ratio': float(summary.get('overlay_selection_ratio', 0.0)),
        'puncture_selection_ratio': float(summary.get('puncture_selection_ratio', 0.0)),
        'embb_only_fraction': float(summary['embb_only_fraction']),
        'avg_puncture_loss': float(summary['avg_puncture_embb_loss']),
        'avg_overlay_retention': float(summary['avg_overlay_retention']),
        'overlay_candidate_pairs': float(overlay_candidate_pairs),
        'overlay_feasible_pairs': float(overlay_feasible_pairs),
        'overlay_selected_pairs': float(overlay_selected_pairs),
        'phase_a_feasible_candidate_pairs': float(summary.get('phase_a_feasible_candidate_pairs', 0.0)),
        'safe_admission_bonus': float(summary.get('safe_admission_bonus', 0.0)),
        'unsafe_admission_penalty': float(summary.get('unsafe_admission_penalty', 0.0)),
        'negative_gap_admission_penalty': float(summary.get('negative_gap_admission_penalty', 0.0)),
        'local_embb_opportunity_cost': float(summary.get('local_embb_opportunity_cost', 0.0)),
        'safe_puncture_preference_penalty': float(summary.get('safe_puncture_preference_penalty', 0.0)),
        'safe_puncture_bonus': float(summary.get('safe_puncture_bonus', 0.0)),
        'overlay_when_safe_puncture_penalty': float(summary.get('overlay_when_safe_puncture_penalty', 0.0)),
        # Service-preserving terminal terms (diagnostics for the current reward shaping).
        'terminal_embb_service_floor_penalty': float(summary.get('terminal_embb_service_floor_penalty', 0.0)),
        'terminal_embb_min_rate_floor_penalty': float(summary.get('terminal_embb_min_rate_floor_penalty', 0.0)),
        'terminal_embb_service_bonus': float(summary.get('terminal_embb_service_bonus', 0.0)),
        'terminal_embb_min_rate_bonus': float(summary.get('terminal_embb_min_rate_bonus', 0.0)),
        'terminal_avg_served_embb_rate_bonus': float(summary.get('terminal_avg_served_embb_rate_bonus', 0.0)),
        'urllc_admission_over_service_tradeoff_penalty': float(summary.get('urllc_admission_over_service_tradeoff_penalty', 0.0)),
        'terminal_embb_service_floor_used': float(summary.get('terminal_embb_service_floor_used', 0.0)),
        'terminal_embb_min_rate_floor_used': float(summary.get('terminal_embb_min_rate_floor_used', 0.0)),
        'terminal_embb_served_user_target': float(summary.get('terminal_embb_served_user_target', 0.0)),
        'admission_band_bonus': float(summary.get('admission_band_bonus', 0.0)),
        'admission_band_penalty': float(summary.get('admission_band_penalty', 0.0)),
        'overlay_gain': float(summary.get('overlay_gain', 0.0)),
        'overlay_margin': float(summary.get('overlay_margin', 0.0)),
        'overlay_retention_gate_bonus': float(summary.get('overlay_retention_gate_bonus', 0.0)),
        'missed_overlay_penalty': float(summary.get('missed_overlay_penalty', 0.0)),
        'terminal_unscheduled_packets': float(summary.get('terminal_unscheduled_packets', 0.0)),
        'terminal_zero_admission_active_penalty': float(summary.get('terminal_zero_admission_active_penalty', 0.0)),
        'terminal_frontier_mode_bonus': float(summary.get('terminal_frontier_mode_bonus', 0.0)),
        'terminal_frontier_mode_penalty': float(summary.get('terminal_frontier_mode_penalty', 0.0)),
        'terminal_admission_collapse_penalty': float(summary.get('terminal_admission_collapse_penalty', 0.0)),
        'both_modes_feasible_ratio': float(summary.get('both_modes_feasible_ratio', 0.0)),
        'safe_puncture_available_ratio': float(summary.get('safe_puncture_available_ratio', 0.0)),
        'overlay_chosen_when_safe_puncture_available_ratio': float(summary.get('overlay_chosen_when_safe_puncture_available_ratio', 0.0)),
        'puncture_chosen_when_safe_puncture_available_ratio': float(summary.get('puncture_chosen_when_safe_puncture_available_ratio', 0.0)),
        'teacher_mode_agreement_ratio': float(summary.get('teacher_mode_agreement_ratio', 0.0)),
        'mode_anchor_active_ratio': float(summary.get('mode_anchor_active_ratio', 0.0)),
        'jain_fairness': float(summary['jain_fairness']),
        'cell_edge_served_ratio': float(cell_edge),
        'cell_edge_min_rate_satisfaction_ratio': float(cell_edge_min_rate_satisfaction_ratio),
        'per_uav_total_load_std': float(np.std(associated_embb + associated_urllc)),
        'per_uav_urllc_sched_std': float(np.std(scheduled_urllc)),
        'per_uav_throughput_std': float(np.std(per_uav_embb_throughput)),
        'per_uav_associated_embb': associated_embb,
        'per_uav_associated_urllc': associated_urllc,
        'user_positions': user_positions[:, :2] if user_positions.ndim == 2 and user_positions.shape[1] >= 2 else np.zeros((0, 2), dtype=float),
        'uav_positions': uav_positions[:, :2] if uav_positions.ndim == 2 and uav_positions.shape[1] >= 2 else np.zeros((0, 2), dtype=float),
        'per_uav_scheduled_embb': per_uav_scheduled_embb,
        'per_uav_scheduled_urllc': scheduled_urllc,
        'per_uav_overlay_count': overlay_counts,
        'per_uav_puncture_count': puncture_counts,
        'per_uav_embb_throughput': per_uav_embb_throughput,
        'shield_correction_ratio': float(shield_corrections / max(total_agent_decisions, 1)),
        'collision_rewrite_ratio': float(collision_rewrites / max(total_agent_decisions, 1)),
        'fallback_ratio': float(fallback_count / max(total_agent_decisions, 1)),
        'mode_correction_ratio': float(mode_correction_count / max(total_agent_decisions, 1)),
        'packet_invalid_ratio': float(packet_invalid_count / max(total_agent_decisions, 1)),
        'mask_invalid_ratio': float(mask_invalid_count / max(total_agent_decisions, 1)),
        'joint_reliability_rewrite_ratio': float(joint_rewrite_count / max(total_agent_decisions, 1)),
        'owner_head_active_ratio': float(owner_head_active_total / max(total_agent_decisions, 1)),
        'embb_power_head_active_ratio': float(embb_power_head_active_total / max(total_agent_decisions, 1)),
        'phase_a_embb_power_head_active_ratio': float(phase_a_embb_power_head_active_total / max(total_agent_decisions, 1)),
        'raw_owner_non_null_ratio': float(raw_owner_non_null_total / max(owner_head_active_total, 1)),
        'executed_owner_non_null_ratio': float(executed_owner_non_null_total / max(owner_head_active_total, 1)),
        'raw_embb_power_nonzero_ratio': float(raw_embb_power_nonzero_total / max(embb_power_head_active_total, 1)),
        'executed_embb_power_nonzero_ratio': float(executed_embb_power_nonzero_total / max(embb_power_head_active_total, 1)),
        # Ratios use the eligible denominator (exclude inactive/no-owner/no-eMBB) to avoid masking suppression.
        'phase_a_raw_embb_power_nonzero_ratio': float(phase_a_raw_embb_power_nonzero_eligible_total / max(phase_a_embb_power_eligible_total, 1)),
        'phase_a_executed_embb_power_nonzero_ratio': float(phase_a_executed_embb_power_nonzero_eligible_total / max(phase_a_embb_power_eligible_total, 1)),
        'phase_a_embb_power_head_active_count': float(phase_a_embb_power_head_active_total),
        'phase_a_embb_power_eligible_count': float(phase_a_embb_power_eligible_total),
        'raw_executed_any_gap_ratio': float(raw_executed_any_gap_total / max(total_agent_decisions, 1)),
        'raw_executed_mode_gap_ratio': float(raw_executed_mode_gap_total / max(total_agent_decisions, 1)),
        'raw_executed_packet_gap_ratio': float(raw_executed_packet_gap_total / max(total_agent_decisions, 1)),
        'raw_executed_power_gap_ratio': float(raw_executed_power_gap_total / max(total_agent_decisions, 1)),
        'raw_executed_owner_gap_ratio': float(raw_executed_owner_gap_total / max(total_agent_decisions, 1)),
        'raw_executed_embb_power_gap_ratio': float(raw_executed_embb_power_gap_total / max(total_agent_decisions, 1)),
        'policy_autonomy_ratio': float(1.0 - shield_corrections / max(total_agent_decisions, 1)),
        'shield_mode_changed_ratio': float(raw_executed_mode_gap_total / max(total_agent_decisions, 1)),
        'shield_packet_changed_ratio': float(raw_executed_packet_gap_total / max(total_agent_decisions, 1)),
        'shield_owner_changed_ratio': float(raw_executed_owner_gap_total / max(total_agent_decisions, 1)),
        'shield_urllc_power_changed_ratio': float(raw_executed_power_gap_total / max(total_agent_decisions, 1)),
        'shield_embb_power_changed_ratio': float(raw_executed_embb_power_gap_total / max(total_agent_decisions, 1)),
        'shield_mean_abs_urllc_power_delta_change': float(urllc_power_abs_change_sum / max(total_agent_decisions, 1)),
        'shield_mean_abs_embb_power_delta_change': float(embb_power_abs_change_sum / max(total_agent_decisions, 1)),
        'intervention_severity': float(
            (raw_executed_mode_gap_total / max(total_agent_decisions, 1))
            + (raw_executed_packet_gap_total / max(total_agent_decisions, 1))
            + 0.5 * (raw_executed_owner_gap_total / max(total_agent_decisions, 1))
            + (urllc_power_abs_change_sum / max(total_agent_decisions, 1))
            + (embb_power_abs_change_sum / max(total_agent_decisions, 1))
        ),
        'planning_owner_non_null_ratio': float(summary.get('planning_owner_non_null_ratio', 0.0)),
        'planning_total_decisions': float(summary.get('planning_total_decisions', 0.0)),
        'planning_owner_non_null_count': float(summary.get('planning_owner_non_null_count', 0.0)),
        'planning_owner_change_count': float(summary.get('planning_owner_change_count', 0.0)),
        'planning_owner_change_ratio': float(summary.get('planning_owner_change_ratio', 0.0)),
        'planning_owner_rewrite_count': float(summary.get('planning_owner_rewrite_count', 0.0)),
        'planning_owner_rewrite_ratio': float(summary.get('planning_owner_rewrite_ratio', 0.0)),
        'planning_embb_power_nonzero_ratio': float(summary.get('planning_embb_power_nonzero_ratio', 0.0)),
        'planning_embb_power_nonzero_count': float(summary.get('planning_embb_power_nonzero_count', 0.0)),
        'planning_embb_power_changed_ratio': float(summary.get('planning_embb_power_changed_ratio', 0.0)),
        'planning_embb_power_changed_count': float(summary.get('planning_embb_power_changed_count', 0.0)),
        'planning_projected_embb_rate_ratio_mean': float(summary.get('planning_projected_embb_rate_ratio_mean', 0.0)),
        'planning_projected_embb_rate_ratio_min': float(summary.get('planning_projected_embb_rate_ratio_min', 0.0)),
        'planning_projected_embb_power_ratio_mean': float(summary.get('planning_projected_embb_power_ratio_mean', 0.0)),
        'planning_projected_embb_power_ratio_max': float(summary.get('planning_projected_embb_power_ratio_max', 0.0)),
        'planning_owner_rate_floor_violation_count': float(summary.get('planning_owner_rate_floor_violation_count', 0.0)),
        'planning_owner_power_ceiling_violation_count': float(summary.get('planning_owner_power_ceiling_violation_count', 0.0)),
        'planning_owner_guard_violation_count': float(summary.get('planning_owner_guard_violation_count', 0.0)),
        'phase0_owner_non_null_ratio_raw': float(summary.get('phase0_owner_non_null_ratio_raw', 0.0)),
        'phase0_owner_non_null_ratio_executed': float(summary.get('phase0_owner_non_null_ratio_executed', 0.0)),
        'phase0_owner_change_ratio_vs_snapshot_raw': float(summary.get('phase0_owner_change_ratio_vs_snapshot_raw', 0.0)),
        'phase0_owner_change_ratio_vs_snapshot_executed': float(summary.get('phase0_owner_change_ratio_vs_snapshot_executed', 0.0)),
        'phase0_owner_fallback_to_candidate0_ratio': float(summary.get('phase0_owner_fallback_to_candidate0_ratio', 0.0)),
        'phase0_owner_invalid_option_ratio': float(summary.get('phase0_owner_invalid_option_ratio', 0.0)),
        'phase0_owner_null_selected_ratio': float(summary.get('phase0_owner_null_selected_ratio', 0.0)),
        'phase0_owner_invalid_to_null_ratio': float(summary.get('phase0_owner_invalid_to_null_ratio', 0.0)),
        'phase0_owner_invalid_to_snapshot_ratio': float(summary.get('phase0_owner_invalid_to_snapshot_ratio', 0.0)),
        'phase0_owner_invalid_to_non_snapshot_ratio': float(summary.get('phase0_owner_invalid_to_non_snapshot_ratio', 0.0)),
        'phase0_owner_restored_to_snapshot_ratio': float(summary.get('phase0_owner_restored_to_snapshot_ratio', 0.0)),
        'phase0_owner_kept_null_ratio': float(summary.get('phase0_owner_kept_null_ratio', 0.0)),
        'phase0_owner_replaced_with_non_snapshot_ratio': float(summary.get('phase0_owner_replaced_with_non_snapshot_ratio', 0.0)),
        'phase0_owner_changed_and_effective_ratio': float(summary.get('phase0_owner_changed_and_effective_ratio', 0.0)),
        'phase0_owner_changed_but_unserved_ratio': float(summary.get('phase0_owner_changed_but_unserved_ratio', 0.0)),
        'phase0_owner_same_as_snapshot_ratio': float(summary.get('phase0_owner_same_as_snapshot_ratio', 0.0)),
        'phase0_owner_effective_service_gain_ratio': float(summary.get('phase0_owner_effective_service_gain_ratio', 0.0)),
        'owner_effective_service_gain_ratio': float(summary.get('owner_effective_service_gain_ratio', summary.get('phase0_owner_effective_service_gain_ratio', 0.0))),
        'phase0_owner_effective_rate_gain_vs_snapshot_mean': float(summary.get('phase0_owner_effective_rate_gain_vs_snapshot_mean', 0.0)),
        'phase0_owner_effective_rate_gain_vs_snapshot_cells_mean_mbps': float(summary.get('phase0_owner_effective_rate_gain_vs_snapshot_cells_mean_mbps', 0.0)),
        'phase0_owner_effective_change_count': float(summary.get('phase0_owner_effective_change_count', 0.0)),
        'phase_a_total_decisions': float(summary.get('phase_a_total_decisions', 0.0)),
        'phase_a_rejected_intercell_per_decision': float(summary.get('phase_a_rejected_intercell_per_decision', 0.0)),
        'phase_a_rejected_min_rate_per_decision': float(summary.get('phase_a_rejected_min_rate_per_decision', 0.0)),
        'phase_a_rejected_power_guard_per_decision': float(summary.get('phase_a_rejected_power_guard_per_decision', 0.0)),
        'phase_a_rejected_collision_per_decision': float(summary.get('phase_a_rejected_collision_per_decision', 0.0)),
        'phase_a_rejected_deadline_per_decision': float(summary.get('phase_a_rejected_deadline_per_decision', 0.0)),
        'phase_a_rejected_other_per_decision': float(summary.get('phase_a_rejected_other_per_decision', 0.0)),
        'phase_a_rejected_other_gain_ratio_per_decision': float(summary.get('phase_a_rejected_other_gain_ratio_per_decision', 0.0)),
        'phase_a_rejected_other_overlay_margin_per_decision': float(summary.get('phase_a_rejected_other_overlay_margin_per_decision', 0.0)),
        'phase_a_rejected_other_overlay_positive_gate_per_decision': float(summary.get('phase_a_rejected_other_overlay_positive_gate_per_decision', 0.0)),
        'phase_a_rejected_other_gain_ratio_given_other_ratio': float(summary.get('phase_a_rejected_other_gain_ratio_given_other_ratio', 0.0)),
        'phase_a_rejected_other_overlay_margin_given_other_ratio': float(summary.get('phase_a_rejected_other_overlay_margin_given_other_ratio', 0.0)),
        'phase_a_rejected_other_overlay_positive_gate_given_other_ratio': float(summary.get('phase_a_rejected_other_overlay_positive_gate_given_other_ratio', 0.0)),
        'phase_a_embb_power_write_count': float(summary.get('phase_a_embb_power_write_count', 0.0)),
        'phase_a_embb_power_changed_count': float(summary.get('phase_a_embb_power_changed_count', 0.0)),
        'phase_a_embb_power_write_ratio': float(summary.get('phase_a_embb_power_write_ratio', 0.0)),
        'phase_a_embb_power_changed_ratio': float(summary.get('phase_a_embb_power_changed_ratio', 0.0)),
        'phase_a_power_zeroed_non_admission_count': float(summary.get('phase_a_power_zeroed_non_admission_count', 0.0)),
        'phase_a_power_zeroed_non_admission_ratio': float(summary.get('phase_a_power_zeroed_non_admission_ratio', 0.0)),
        'phase_a_power_write_on_admission_ratio': float(summary.get('phase_a_power_write_on_admission_ratio', 0.0)),
        'phase_a_power_write_on_keep_ratio': float(summary.get('phase_a_power_write_on_keep_ratio', 0.0)),
        'action_intercell_guard_active_ratio': float(summary.get('action_intercell_guard_active_ratio', 0.0)),
        'action_intercell_guard_masked_option_count': float(summary.get('action_intercell_guard_masked_option_count', 0.0)),
        'phase_a_embb_power_mean_abs_change': float(summary.get('phase_a_embb_power_mean_abs_change', 0.0)),
        'phase_a_embb_power_mean_raw_delta': float(summary.get('phase_a_embb_power_mean_raw_delta', 0.0)),
        'phase_a_embb_power_mean_executed_delta': float(summary.get('phase_a_embb_power_mean_executed_delta', 0.0)),
        'phase_a_embb_power_pre_clip_mean_delta': float(summary.get('phase_a_embb_power_pre_clip_mean_delta', 0.0)),
        'phase_a_embb_power_post_clip_mean_delta': float(summary.get('phase_a_embb_power_post_clip_mean_delta', 0.0)),
        'phase_a_embb_power_post_quant_mean_delta': float(summary.get('phase_a_embb_power_post_quant_mean_delta', 0.0)),
        'phase_a_embb_power_post_projection_mean_delta': float(summary.get('phase_a_embb_power_post_projection_mean_delta', 0.0)),
        'phase_a_embb_power_post_owner_validation_mean_delta': float(summary.get('phase_a_embb_power_post_owner_validation_mean_delta', 0.0)),
        'phase_a_embb_power_final_executed_mean_delta': float(summary.get('phase_a_embb_power_final_executed_mean_delta', 0.0)),
        'phase_a_embb_power_clip_ratio': float(summary.get('phase_a_embb_power_clip_ratio', 0.0)),
        'phase_a_embb_power_quantized_ratio': float(summary.get('phase_a_embb_power_quantized_ratio', 0.0)),
        'phase_a_embb_power_projection_ratio': float(summary.get('phase_a_embb_power_projection_ratio', 0.0)),
        'phase_a_embb_power_owner_invalid_ratio': float(summary.get('phase_a_embb_power_owner_invalid_ratio', 0.0)),
        'phase_a_embb_power_no_candidate_ratio': float(summary.get('phase_a_embb_power_no_candidate_ratio', 0.0)),
        'phase_a_embb_power_keep_mode_zero_ratio': float(summary.get('phase_a_embb_power_keep_mode_zero_ratio', 0.0)),
        'phase_a_embb_power_cap_hit_ratio': float(summary.get('phase_a_embb_power_cap_hit_ratio', 0.0)),
        'phase_a_embb_power_floor_hit_ratio': float(summary.get('phase_a_embb_power_floor_hit_ratio', 0.0)),
        'phase_a_embb_power_sign_flip_ratio': float(summary.get('phase_a_embb_power_sign_flip_ratio', 0.0)),
        'phase_a_embb_power_abs_shrink_ratio': float(summary.get('phase_a_embb_power_abs_shrink_ratio', 0.0)),
        'phase_a_embb_power_projection_l2_mean': float(summary.get('phase_a_embb_power_projection_l2_mean', 0.0)),
        'phase_a_embb_power_pre_vs_final_l1_mean': float(summary.get('phase_a_embb_power_pre_vs_final_l1_mean', 0.0)),
        'phase_a_embb_power_pre_vs_final_sign_consistency': float(summary.get('phase_a_embb_power_pre_vs_final_sign_consistency', 0.0)),
        'phase_a_embb_power_effective_nonzero_ratio': float(summary.get('phase_a_embb_power_effective_nonzero_ratio', 0.0)),
        'phase_a_embb_power_raw_saturation_ratio': float(summary.get('phase_a_embb_power_raw_saturation_ratio', 0.0)),
        'phase_a_embb_power_final_std': float(summary.get('phase_a_embb_power_final_std', 0.0)),
        'phase_a_embb_power_cellwise_diversity': float(summary.get('phase_a_embb_power_cellwise_diversity', 0.0)),
        'phase_a_embb_power_floor_binding_strength': float(summary.get('phase_a_embb_power_floor_binding_strength', 0.0)),
        'phase_a_embb_power_cap_binding_strength': float(summary.get('phase_a_embb_power_cap_binding_strength', 0.0)),
        'phase_a_embb_power_proj_delta_l1': float(summary.get('phase_a_embb_power_proj_delta_l1', 0.0)),
        'phase_a_embb_power_proj_delta_l2': float(summary.get('phase_a_embb_power_proj_delta_l2', 0.0)),
        'phase_a_embb_power_pre_to_floor_delta': float(summary.get('phase_a_embb_power_pre_to_floor_delta', 0.0)),
        'phase_a_embb_power_pre_to_cap_delta': float(summary.get('phase_a_embb_power_pre_to_cap_delta', 0.0)),
        'phase_a_embb_power_final_minus_proj_delta': float(summary.get('phase_a_embb_power_final_minus_proj_delta', 0.0)),
        'phase_a_embb_power_mean_abs_raw_delta': float(summary.get('phase_a_embb_power_mean_abs_raw_delta', 0.0)),
        'phase_a_embb_power_mean_abs_executed_delta': float(summary.get('phase_a_embb_power_mean_abs_executed_delta', 0.0)),
        'phase_a_embb_power_invalid_or_masked_ratio': float(summary.get('phase_a_embb_power_invalid_or_masked_ratio', 0.0)),
        'power_delta_clipped_ratio': float(summary.get('power_delta_clipped_ratio', 0.0)),
        'power_quantized_ratio': float(summary.get('power_quantized_ratio', 0.0)),
        'power_cap_hit_ratio': float(summary.get('power_cap_hit_ratio', 0.0)),
        'power_floor_hit_ratio': float(summary.get('power_floor_hit_ratio', 0.0)),
        'mean_raw_power_delta': float(summary.get('mean_raw_power_delta', 0.0)),
        'mean_executed_power_delta': float(summary.get('mean_executed_power_delta', 0.0)),
        'admission_via_overlay_ratio': float(summary.get('admission_via_overlay_ratio', 0.0)),
        'admission_via_puncture_ratio': float(summary.get('admission_via_puncture_ratio', 0.0)),
        'puncture_candidate_pruned_by_loss_ceiling_ratio': float(summary.get('puncture_candidate_pruned_by_loss_ceiling_ratio', 0.0)),
        'comparison_baseline_key': _normalize_baseline_mode(_greedy_baseline_mode(cfg)),
        'comparison_baseline_label': _baseline_label(_greedy_baseline_mode(cfg)),
        'trace': trace,
        'mode_grid': env.mode_grid.copy(),
        'packet_grid': env.packet_grid.copy(),
        'owner_per_uav_rb': env.owner_per_uav_rb.copy(),
        'snapshot_owner_per_uav_rb': (
            env.phase0_snapshot_owner_per_uav_rb.copy()
            if getattr(env, "phase0_snapshot_owner_per_uav_rb", None) is not None else None
        ),
        'episode_sec': float(perf_counter() - episode_start),
        'report_episode_seed': float(seed),
        'profile_action_select_sec': float(action_select_sec_total),
        'profile_action_resolve_sec': float(action_resolve_sec_total),
        'profile_env_step_sec': float(env_step_sec_total),
        'virtual_slot_reset_count_per_episode': float(reset_count_contribution),
    }
    result.update({
        'phase_a_embb_power_pre_clip_mean_delta': float(summary.get('phase_a_embb_power_pre_clip_mean_delta', 0.0)),
        'phase_a_embb_power_post_clip_mean_delta': float(summary.get('phase_a_embb_power_post_clip_mean_delta', 0.0)),
        'phase_a_embb_power_post_projection_mean_delta': float(summary.get('phase_a_embb_power_post_projection_mean_delta', 0.0)),
        'phase_a_embb_power_final_executed_mean_delta': float(summary.get('phase_a_embb_power_final_executed_mean_delta', 0.0)),
        'phase_a_embb_power_cap_hit_ratio': float(summary.get('phase_a_embb_power_cap_hit_ratio', 0.0)),
        'phase_a_embb_power_floor_hit_ratio': float(summary.get('phase_a_embb_power_floor_hit_ratio', 0.0)),
        'phase_a_embb_power_projection_l2_mean': float(summary.get('phase_a_embb_power_projection_l2_mean', 0.0)),
        'phase_a_embb_power_pre_vs_final_l1_mean': float(summary.get('phase_a_embb_power_pre_vs_final_l1_mean', 0.0)),
        'phase_a_embb_power_pre_vs_final_sign_consistency': float(
            summary.get('phase_a_embb_power_pre_vs_final_sign_consistency', 0.0)
        ),
        'phase_a_embb_power_effective_nonzero_ratio': float(summary.get('phase_a_embb_power_effective_nonzero_ratio', 0.0)),
        'phaseA_pow_ratio_preclip_mean': float(summary.get('phaseA_pow_ratio_preclip_mean', 0.0)),
        'phaseA_pow_ratio_postclip_mean': float(summary.get('phaseA_pow_ratio_postclip_mean', 0.0)),
        'phaseA_pow_ratio_postproj_mean': float(summary.get('phaseA_pow_ratio_postproj_mean', 0.0)),
        'phaseA_pow_ratio_final_mean': float(summary.get('phaseA_pow_ratio_final_mean', 0.0)),
        'phaseA_pow_cap_hit_ratio': float(summary.get('phaseA_pow_cap_hit_ratio', 0.0)),
        'phaseA_pow_floor_hit_ratio': float(summary.get('phaseA_pow_floor_hit_ratio', 0.0)),
        'phaseA_pow_projection_l2_mean': float(summary.get('phaseA_pow_projection_l2_mean', 0.0)),
        'phaseA_pow_pre_vs_final_l1_mean': float(summary.get('phaseA_pow_pre_vs_final_l1_mean', 0.0)),
        'phaseA_pow_pre_vs_final_sign_consistency': float(summary.get('phaseA_pow_pre_vs_final_sign_consistency', 0.0)),
        'phaseA_pow_effective_nonzero_ratio': float(summary.get('phaseA_pow_effective_nonzero_ratio', 0.0)),
    })
    if use_greedy and normalized_greedy_policy in {"hard_feasible", "hard_feasible_throughput", "global_frontier", "global_greedy", "throughput_only", "throughput_biased", "myopic", "myopic_throughput"}:
        decision_denom = max(greedy_phase_a_decisions, 1)
        if normalized_greedy_policy in {"hard_feasible", "hard_feasible_throughput"}:
            baseline_key = "hard_feasible_throughput_greedy"
        elif normalized_greedy_policy in {"global_frontier", "global_greedy"}:
            baseline_key = "global_frontier_greedy"
        elif normalized_greedy_policy == "throughput_biased":
            baseline_key = "throughput_biased_greedy"
        elif normalized_greedy_policy == "throughput_only":
            baseline_key = "throughput_only_greedy"
        else:
            baseline_key = "myopic_throughput_greedy"
        result.update({
            **_baseline_metadata(baseline_key),
            'greedy_noop_selected_ratio': float(greedy_noop_selected / decision_denom),
            'greedy_admit_selected_ratio': float(greedy_admit_selected / decision_denom),
            'greedy_overlay_ratio': float(greedy_overlay_selected / decision_denom),
            'greedy_puncture_ratio': float(greedy_puncture_selected / decision_denom),
            'greedy_avg_embb_retention': float(greedy_selected_retention_sum / decision_denom),
            'greedy_avg_embb_loss': float(greedy_selected_loss_sum / decision_denom),
            'greedy_avg_selected_throughput': float(greedy_selected_throughput_sum / decision_denom),
            'greedy_selected_embb_throughput': float(greedy_selected_throughput_sum / decision_denom),
            'greedy_feasible_admit_count': float(greedy_feasible_admit_count_sum / decision_denom),
            'greedy_no_feasible_admit_ratio': float(greedy_no_feasible_admit_sum / decision_denom),
            'greedy_keep_only_when_no_feasible_admit_ratio': float(
                greedy_keep_only_when_no_feasible_admit_sum / max(greedy_noop_selected, 1.0)
            ),
            'greedy_selected_urllc_reliability': float(greedy_selected_reliability_sum / decision_denom),
            'greedy_selected_embb_min_rate_ok': float(greedy_selected_embb_min_rate_ok_sum / decision_denom),
            'greedy_avg_rejected_urllc_when_noop_better': float(
                greedy_rejected_when_noop_better_sum / decision_denom
            ),
            'greedy_noop_available_ratio': float(greedy_noop_available_sum / decision_denom),
            'greedy_noop_better_ratio': float(greedy_noop_better_sum / decision_denom),
            'greedy_requires_feasible_admission_only': float(greedy_requires_feasible_only),
            'greedy_hf_raw_count': float(greedy_hf_raw_count_sum / decision_denom),
            'greedy_hf_admissible_count': float(greedy_hf_admissible_count_sum / decision_denom),
            'greedy_hf_evaluated_count': float(greedy_hf_evaluated_count_sum / decision_denom),
            'greedy_hf_feasible_count': float(greedy_hf_feasible_count_sum / decision_denom),
            'greedy_hf_selected_count': float(greedy_hf_selected_count_sum / decision_denom),
            'greedy_hf_reject_by_gate_association': float(greedy_hf_reject_gate_assoc_sum / decision_denom),
            'greedy_hf_reject_by_gate_queue_active': float(greedy_hf_reject_gate_queue_sum / decision_denom),
            'greedy_hf_reject_by_gate_mode_admissible': float(greedy_hf_reject_gate_mode_sum / decision_denom),
            'greedy_hf_reject_by_gate_owner_admissible': float(greedy_hf_reject_gate_owner_sum / decision_denom),
            'greedy_hf_reject_by_gate_rb_local': float(greedy_hf_reject_gate_rb_local_sum / decision_denom),
            'greedy_hf_reject_by_mode_overlay_rel_fail': float(greedy_hf_reject_mode_overlay_rel_fail_sum / decision_denom),
            'greedy_hf_reject_by_mode_overlay_sic_fail': float(greedy_hf_reject_mode_overlay_sic_fail_sum / decision_denom),
            'greedy_hf_reject_by_mode_gain_ratio_fail': float(greedy_hf_reject_mode_gain_ratio_fail_sum / decision_denom),
            'greedy_hf_reject_by_mode_owner_pool_missing': float(greedy_hf_reject_mode_owner_pool_missing_sum / decision_denom),
            'greedy_hf_reject_by_mode_owner_unresolved_due_to_mode_fail': float(
                greedy_hf_reject_mode_owner_unresolved_due_to_mode_fail_sum / decision_denom
            ),
            'greedy_hf_reject_by_mode_overlay_owner_missing': float(greedy_hf_reject_mode_owner_pool_missing_sum / decision_denom),
            'greedy_hf_reject_by_mode_puncture_rel_fail': float(greedy_hf_reject_mode_puncture_rel_fail_sum / decision_denom),
            'greedy_hf_reject_by_mode_puncture_sic_fail': float(greedy_hf_reject_mode_puncture_sic_fail_sum / decision_denom),
            'greedy_hf_reject_by_mode_puncture_owner_missing': float(greedy_hf_reject_mode_puncture_owner_missing_sum / decision_denom),
            'greedy_hf_reject_by_owner_missing': float(greedy_hf_reject_owner_missing_sum / decision_denom),
            'greedy_hf_reject_by_owner_mismatch': float(greedy_hf_reject_owner_mismatch_sum / decision_denom),
            'greedy_hf_gate_overlay_rel_fail_margin_mean': float(greedy_hf_gate_overlay_rel_fail_margin_sum / decision_denom),
            'greedy_hf_gate_overlay_sic_fail_margin_db_mean': float(greedy_hf_gate_overlay_sic_fail_margin_db_sum / decision_denom),
            'greedy_hf_gate_puncture_rel_fail_margin_mean': float(greedy_hf_gate_puncture_rel_fail_margin_sum / decision_denom),
            'greedy_hf_gate_target_reliability': float(greedy_hf_gate_target_reliability_sum / decision_denom),
            'greedy_hf_gate_target_sic_snir_db': float(greedy_hf_gate_target_sic_db_sum / decision_denom),
            'greedy_hf_gate_overlay_rel_fail_snir_db_mean': float(greedy_hf_gate_overlay_rel_fail_snir_db_sum / decision_denom),
            'greedy_hf_gate_overlay_sic_fail_post_sic_db_mean': float(greedy_hf_gate_overlay_sic_fail_post_sic_db_sum / decision_denom),
            'greedy_hf_gate_puncture_rel_fail_snir_db_mean': float(greedy_hf_gate_puncture_rel_fail_snir_db_sum / decision_denom),
            'greedy_hf_overlay_sic_trace_pre_sinr_db_mean': float(greedy_hf_overlay_sic_trace_pre_sinr_db_sum / decision_denom),
            'greedy_hf_overlay_sic_trace_post_sinr_db_mean': float(greedy_hf_overlay_sic_trace_post_sinr_db_sum / decision_denom),
            'greedy_hf_overlay_sic_trace_noise_power_mean': float(greedy_hf_overlay_sic_trace_noise_power_sum / decision_denom),
            'greedy_hf_overlay_sic_trace_intercell_interference_mean': float(greedy_hf_overlay_sic_trace_intercell_interference_sum / decision_denom),
            'greedy_hf_overlay_sic_trace_local_interference_mean': float(greedy_hf_overlay_sic_trace_local_interference_sum / decision_denom),
            'greedy_hf_overlay_sic_trace_residual_sic_interference_mean': float(greedy_hf_overlay_sic_trace_residual_interference_sum / decision_denom),
            'greedy_hf_overlay_sic_trace_sic_residual_ratio': float(greedy_hf_overlay_sic_trace_residual_ratio_sum / decision_denom),
            'greedy_hf_overlay_presinr_raw_lt_m10': float(greedy_hf_overlay_presinr_raw_lt_m10_sum / decision_denom),
            'greedy_hf_overlay_presinr_raw_m10_m6': float(greedy_hf_overlay_presinr_raw_m10_m6_sum / decision_denom),
            'greedy_hf_overlay_presinr_raw_m6_m2': float(greedy_hf_overlay_presinr_raw_m6_m2_sum / decision_denom),
            'greedy_hf_overlay_presinr_raw_ge_m2': float(greedy_hf_overlay_presinr_raw_ge_m2_sum / decision_denom),
            'greedy_hf_overlay_presinr_kept_low_ratio': float(greedy_hf_overlay_presinr_kept_low_ratio_sum / decision_denom),
            'greedy_hf_overlay_presinr_eval_low_ratio': float(greedy_hf_overlay_presinr_eval_low_ratio_sum / decision_denom),
            'greedy_hf_quality_priority_enabled': float(greedy_hf_quality_priority_enabled_sum / decision_denom),
            'greedy_hf_quality_raw_high': float(greedy_hf_quality_raw_high_sum / decision_denom),
            'greedy_hf_quality_raw_borderline': float(greedy_hf_quality_raw_borderline_sum / decision_denom),
            'greedy_hf_quality_raw_risk': float(greedy_hf_quality_raw_risk_sum / decision_denom),
            'greedy_hf_quality_kept_high': float(greedy_hf_quality_kept_high_sum / decision_denom),
            'greedy_hf_quality_kept_borderline': float(greedy_hf_quality_kept_borderline_sum / decision_denom),
            'greedy_hf_quality_kept_risk': float(greedy_hf_quality_kept_risk_sum / decision_denom),
            'greedy_hf_quality_eval_high': float(greedy_hf_quality_eval_high_sum / decision_denom),
            'greedy_hf_quality_eval_borderline': float(greedy_hf_quality_eval_borderline_sum / decision_denom),
            'greedy_hf_quality_eval_risk': float(greedy_hf_quality_eval_risk_sum / decision_denom),
            'greedy_hf_quality_selected_high': float(greedy_hf_quality_selected_high_sum / decision_denom),
            'greedy_hf_quality_selected_borderline': float(greedy_hf_quality_selected_borderline_sum / decision_denom),
            'greedy_hf_quality_selected_risk': float(greedy_hf_quality_selected_risk_sum / decision_denom),
            'greedy_hf_overlay_blacklist_cell_ratio': float(greedy_hf_overlay_blacklist_cell_ratio_sum / decision_denom),
            'greedy_hf_overlay_blacklist_candidate_block_ratio': float(greedy_hf_overlay_blacklist_candidate_block_ratio_sum / decision_denom),
            'greedy_hf_overlay_blacklist_saved_mode_fail_ratio': float(greedy_hf_overlay_blacklist_saved_mode_fail_ratio_sum / decision_denom),
            'greedy_hf_overlay_only_rb_reservation_enabled': float(
                greedy_hf_overlay_only_rb_reservation_enabled_sum / decision_denom
            ),
            'greedy_hf_overlay_only_rb_reservation_ratio': float(
                greedy_hf_overlay_only_rb_reservation_ratio_sum / decision_denom
            ),
            'greedy_hf_overlay_only_rb_reserved_count': float(
                greedy_hf_overlay_only_rb_reserved_count_sum / decision_denom
            ),
            'greedy_hf_overlay_only_rb_reserved_cell_ratio': float(
                greedy_hf_overlay_only_rb_reserved_cell_ratio_sum / decision_denom
            ),
            'greedy_hf_overlay_only_rb_mode_block_ratio': float(
                greedy_hf_overlay_only_rb_mode_block_ratio_sum / decision_denom
            ),
        })
        result.update(
            _baseline_narrative(
                baseline_key,
                greedy_requires_feasible_admission_only=bool(greedy_requires_feasible_only),
            )
        )
    if not use_greedy:
        _report_log(
            "[owner-budget] "
            f"seed={seed} "
            f"raw_changed_count={float(result.get('phase0_owner_raw_changed_count_mean', 0.0)):.3f} "
            f"allowed_k={float(result.get('phase0_owner_allowed_k_mean', 0.0)):.3f} "
            f"executed_changed_count={float(result.get('phase0_owner_executed_changed_count_mean', 0.0)):.3f} "
            f"dropped_count={float(result.get('phase0_owner_dropped_count_mean', 0.0)):.3f}"
        )
        _report_log(
            "[owner-ratios] "
            f"raw_change_ratio={float(result.get('phase0_owner_change_ratio_vs_snapshot_raw', 0.0)):.6f} "
            f"executed_change_ratio={float(result.get('phase0_owner_change_ratio_vs_snapshot_executed', 0.0)):.6f} "
            f"changed_effective_ratio={float(result.get('phase0_owner_changed_and_effective_ratio', 0.0)):.6f} "
            f"same_as_snapshot_ratio={float(result.get('phase0_owner_same_as_snapshot_ratio', 0.0)):.6f}"
        )
        _report_log(
            "[owner-gain] "
            f"effective_rate_gain_mean={float(result.get('phase0_owner_effective_rate_gain_vs_snapshot_mean', 0.0)):.6f} "
            f"harmful_change_ratio={float(result.get('phase0_owner_change_harmful_ratio', 0.0)):.6f}"
        )
        _report_log(
            "[phaseA] "
            f"raw_positive_ratio={float(result.get('phase_a_power_raw_positive_ratio', 0.0)):.6f} "
            f"positive_clamped_ratio={float(result.get('phase_a_power_positive_clamped_to_zero_ratio', 0.0)):.6f} "
            f"negative_executed_ratio={float(result.get('phase_a_power_negative_executed_ratio', 0.0)):.6f}"
        )
    if _REPORT_EPISODE_CACHE_ENABLED and cache_key is not None:
        _REPORT_EPISODE_CACHE[cache_key] = deepcopy(result)
    # Per-episode timing line is intentionally suppressed to keep logs concise.
    return result


def run_mappo_sweep(
    loads: List[float],
    episodes_per_load: int,
    checkpoint_path: Path,
    *,
    base_cfg: SRMAPPOConfig | None = None,
) -> Tuple[Dict, Dict]:
    sweep_start = perf_counter()
    base_sys, base_urllc, base_embb, base_algo, base_sim = _build_main_like_configs()
    checkpoint_cfg = deepcopy(_load_checkpoint_cfg(checkpoint_path))
    # Report/eval should use the *requested experiment preset* for reward/env/shield toggles, while
    # keeping action/network compatibility from the checkpoint.
    report_cfg = deepcopy(checkpoint_cfg)
    if base_cfg is not None:
        report_cfg.reward = deepcopy(base_cfg.reward)
        report_cfg.env = deepcopy(base_cfg.env)
        report_cfg.shield = deepcopy(base_cfg.shield)
        report_cfg.training.greedy_baseline_mode = getattr(base_cfg.training, "greedy_baseline_mode", report_cfg.training.greedy_baseline_mode)
        report_cfg.training.report_fast_debug = bool(getattr(base_cfg.training, "report_fast_debug", getattr(report_cfg.training, "report_fast_debug", False)))
        report_cfg.training.report_baseline_mode = getattr(base_cfg.training, "report_baseline_mode", getattr(report_cfg.training, "report_baseline_mode", ""))
        report_cfg.training.primary_checkpoint_preference = getattr(base_cfg.training, "primary_checkpoint_preference", report_cfg.training.primary_checkpoint_preference)
    report_cfg.env.include_greedy_reference_in_obs = False
    _apply_forced_urllc_ratio_to_sim(base_sim, report_cfg, log_prefix="MAPPO")
    checkpoint_cache_tag = str(Path(checkpoint_path).resolve())
    nested_load_enabled = str(os.environ.get("SR_MAPPO_REPORT_NESTED_LOAD_SCENARIO", "1")).strip().lower() not in {"0", "false", "no", "off"}
    max_total_users = 0
    max_embb_users = 0
    max_urllc_users = 0
    if nested_load_enabled and loads:
        _mx_sys, _mx_ur, _mx_em, _mx_algo, _mx_sim = _configure_density_scenario(
            max(float(x) for x in loads), base_sys, base_urllc, base_embb, base_algo, base_sim
        )
        max_embb_users = int(_mx_sys.num_embb_users)
        max_urllc_users = int(_mx_sys.num_urllc_users)
        max_total_users = int(max_embb_users + max_urllc_users)
        fixed_pool_embb = int(_env_int_override("SR_MAPPO_REPORT_NESTED_FIXED_POOL_EMBB_USERS", 0))
        fixed_pool_urllc = int(_env_int_override("SR_MAPPO_REPORT_NESTED_FIXED_POOL_URLLC_USERS", 0))
        fixed_pool_total = int(_env_int_override("SR_MAPPO_REPORT_NESTED_FIXED_POOL_TOTAL_USERS", 0))
        if fixed_pool_embb > 0:
            max_embb_users = int(fixed_pool_embb)
        if fixed_pool_urllc > 0:
            max_urllc_users = int(fixed_pool_urllc)
        if fixed_pool_total > 0:
            max_total_users = int(fixed_pool_total)
        else:
            max_total_users = int(max(max_total_users, max_embb_users + max_urllc_users))
    shared_nested_ur_order_across_loads = None
    shared_nested_em_order_across_loads = None
    prev_nested_embb_served_count: Optional[int] = None
    prev_nested_embb_subset_count: Optional[int] = None
    prev_nested_embb_selected_ids: Optional[list[int]] = None
    virtual_slots_per_episode = max(
        1,
        int(os.environ.get("SR_MAPPO_REPORT_VIRTUAL_SLOTS_PER_EPISODE", "1") or "1"),
    )
    if virtual_slots_per_episode > 1:
        _report_log(
            f"[MAPPO] virtual multi-slot enabled: slots_per_episode={virtual_slots_per_episode}"
        )
    scalar_keys = [
        'learn_embb_baseline', 'learn_phase0_embb_power', 'allow_phase_a_embb_power_adjustment',
        'embb_user_count', 'urllc_user_count', 'embb_urllc_user_ratio',
        'phase_a_embb_power_runtime_enabled', 'phase_a_embb_power_anchor_binding_ratio',
        'phase_a_embb_power_pre_clip_mean_delta', 'phase_a_embb_power_post_clip_mean_delta',
        'phase_a_embb_power_post_projection_mean_delta', 'phase_a_embb_power_final_executed_mean_delta',
        'phase_a_embb_power_cap_hit_ratio', 'phase_a_embb_power_floor_hit_ratio',
        'phase_a_embb_power_projection_l2_mean', 'phase_a_embb_power_pre_vs_final_l1_mean',
        'phase_a_embb_power_pre_vs_final_sign_consistency', 'phase_a_embb_power_effective_nonzero_ratio',
        'phase_a_embb_power_raw_saturation_ratio', 'phase_a_embb_power_final_std', 'phase_a_embb_power_cellwise_diversity',
        'phase_a_power_raw_positive_ratio', 'phase_a_power_positive_clamped_to_zero_ratio',
        'phase_a_power_negative_candidate_ratio', 'phase_a_power_negative_executed_ratio',
        'phase_a_power_mean_negative_delta', 'phase_a_power_service_guard_reject_ratio',
        'phase_a_power_minrate_guard_reject_ratio', 'phase_a_power_reliability_guard_reject_ratio',
        'phase_a_power_total_power_reduction_mean', 'phase_a_power_intercell_reduction_mean',
        'enable_action_masking', 'enable_feasibility_shield', 'apply_joint_reliability_rewrite',
        'enable_greedy_fallback',
        'embb_rate', 'embb_rate_after_local_puncture_deduction', 'embb_total_rate_after_puncture_deduction',
        'embb_user_rate', 'embb_user_rate_mean_after_puncture_deduction',
        'embb_service_ratio', 'embb_positive_rate_ratio', 'embb_min_rate_satisfaction_ratio', 'embb_min_rate_satisfied_user_count', 'urllc_admission',
        'embb_service_ratio_after_puncture_deduction', 'embb_min_rate_satisfaction_after_puncture_deduction',
        'admitted_urllc_reliability', 'urllc_reliability', 'effective_urllc_success_over_arrivals', 'empty_admission_case',
        'effective_lambda_per_user', 'effective_lambda_per_user_per_minislot',
        'expected_total_arrivals_per_minislot', 'expected_total_arrivals_per_episode',
        'active_packets',
        'urllc_slot_duration_s', 'urllc_packet_bits_mean',
        'urllc_throughput_bps_slot_est', 'urllc_throughput_mbps_slot_est',
        'urllc_throughput_bps_est',
        'mean_intercell_interference_power', 'mean_intercell_interference_mw', 'mean_intercell_interference_dbm',
        'intercell_source_mask_excluded_by_puncture_ratio', 'intercell_source_active_ratio',
        'intercell_power_before_puncture_source_mask_mean', 'intercell_power_after_puncture_source_mask_mean', 'intercell_power_reduction_from_source_mask_mean',
        'intercell_interference_nonzero_ratio', 'overlay_intercell_interference_mw', 'puncture_intercell_interference_mw',
        'intercell_penalty_active_ratio', 'intercell_reward_component_mean',
        'selected_action_intercell_cost_mean', 'selected_action_intercell_cost_p95',
        'selected_action_intercell_cost_after_source_mask_mean', 'selected_action_intercell_cost_after_source_mask_p95',
        'selected_action_intercell_cost_before_source_mask_mean', 'selected_action_intercell_cost_before_source_mask_p95',
        'intercell_per_admitted_packet',
        'phase_a_feasible_overlay_candidates_mean', 'phase_a_feasible_puncture_candidates_mean',
        'phase_a_selected_keep_ratio', 'phase_a_selected_overlay_ratio', 'phase_a_selected_puncture_ratio',
        'phase_a_rejected_intercell_per_decision', 'phase_a_rejected_min_rate_per_decision',
        'phase_a_rejected_power_guard_per_decision', 'phase_a_rejected_collision_per_decision',
        'phase_a_rejected_deadline_per_decision', 'phase_a_rejected_other_per_decision',
        'embb_served_user_count', 'embb_min_rate_satisfied_user_count_after_puncture_deduction',
        'embb_rate_with_intercell', 'embb_rate_without_intercell_est', 'embb_rate_loss_due_to_intercell', 'embb_rate_loss_due_to_intercell_ratio',
        'overlay_rate_with_intercell', 'overlay_rate_without_intercell_est', 'overlay_rate_loss_due_to_intercell',
        'puncture_rate_with_intercell', 'puncture_rate_without_intercell_est', 'puncture_rate_loss_due_to_intercell',
        'embb_rate_with_intercell_after_puncture_deduction', 'no_intercell_rate_with_same_puncture_mask', 'intercell_rate_loss_with_same_puncture_mask',
        'local_punctured_embb_airtime_ratio', 'embb_rate_raw_before_local_puncture_deduction', 'embb_rate_after_local_puncture_deduction',
        'embb_rate_loss_due_to_local_puncture', 'embb_rate_loss_due_to_local_puncture_ratio',
        'scheduled_packets', 'scheduled_packets_per_uav',
        'total_power', 'embb_power', 'urllc_power', 'throughput_per_watt', 'avg_throughput_per_served_embb_user',
        'overlay_ratio', 'puncture_ratio', 'overlay_selection_ratio', 'puncture_selection_ratio',
        'embb_only_fraction', 'avg_puncture_loss',
        'avg_overlay_retention', 'overlay_candidate_pairs', 'overlay_feasible_pairs',
        'overlay_selected_pairs', 'admission_via_overlay_ratio', 'admission_via_puncture_ratio', 'phase_a_feasible_candidate_pairs',
        'safe_admission_bonus', 'unsafe_admission_penalty', 'negative_gap_admission_penalty',
        'local_embb_opportunity_cost', 'safe_puncture_preference_penalty',
        'safe_puncture_bonus', 'overlay_when_safe_puncture_penalty',
        'overlay_when_lower_intercell_puncture_available_penalty',
        'missed_feasible_puncture_penalty',
        'admission_band_bonus', 'admission_band_penalty',
        'overlay_gain', 'overlay_margin', 'overlay_retention_gate_bonus', 'missed_overlay_penalty',
        'terminal_embb_service_floor_penalty', 'terminal_embb_min_rate_floor_penalty',
        'terminal_embb_service_bonus', 'terminal_embb_min_rate_bonus', 'terminal_avg_served_embb_rate_bonus',
        'terminal_embb_served_user_deficit_penalty',
        'urllc_admission_over_service_tradeoff_penalty',
        'terminal_embb_service_floor_used', 'terminal_embb_min_rate_floor_used', 'terminal_embb_served_user_target',
        'terminal_intercell_power_penalty', 'terminal_total_power_over_greedy_penalty', 'terminal_embb_power_over_greedy_penalty',
        'terminal_unscheduled_packets', 'terminal_zero_admission_active_penalty',
        'terminal_frontier_mode_bonus', 'terminal_frontier_mode_penalty', 'terminal_admission_collapse_penalty',
        'both_modes_feasible_ratio', 'safe_puncture_available_ratio',
        'feasible_puncture_available_ratio', 'puncture_chosen_when_feasible_ratio',
        'overlay_chosen_when_lower_intercell_puncture_available_ratio',
        'missed_feasible_puncture_ratio',
        'overlay_chosen_when_safe_puncture_available_ratio',
        'puncture_chosen_when_safe_puncture_available_ratio',
        'teacher_mode_agreement_ratio', 'mode_anchor_active_ratio',
        'jain_fairness', 'cell_edge_served_ratio', 'cell_edge_min_rate_satisfaction_ratio',
        'per_uav_total_load_std', 'per_uav_urllc_sched_std', 'per_uav_throughput_std',
        'shield_correction_ratio', 'collision_rewrite_ratio', 'fallback_ratio',
        'mode_correction_ratio', 'packet_invalid_ratio', 'mask_invalid_ratio',
        'joint_reliability_rewrite_ratio', 'owner_head_active_ratio',
        'embb_power_head_active_ratio', 'phase_a_embb_power_head_active_ratio',
        'raw_owner_non_null_ratio', 'executed_owner_non_null_ratio',
        'raw_embb_power_nonzero_ratio', 'executed_embb_power_nonzero_ratio',
        'phase_a_raw_embb_power_nonzero_ratio', 'phase_a_executed_embb_power_nonzero_ratio',
        'raw_executed_any_gap_ratio', 'raw_executed_mode_gap_ratio',
        'raw_executed_packet_gap_ratio', 'raw_executed_power_gap_ratio',
        'raw_executed_owner_gap_ratio', 'raw_executed_embb_power_gap_ratio',
        'policy_autonomy_ratio',
        'shield_mode_changed_ratio', 'shield_packet_changed_ratio', 'shield_owner_changed_ratio',
        'shield_urllc_power_changed_ratio', 'shield_embb_power_changed_ratio',
        'shield_mean_abs_urllc_power_delta_change', 'shield_mean_abs_embb_power_delta_change',
        'intervention_severity',
        'planning_total_decisions', 'planning_owner_non_null_count',
        'planning_owner_change_count', 'planning_owner_rewrite_count',
        'planning_embb_power_nonzero_count', 'planning_embb_power_changed_count',
        'planning_owner_non_null_ratio', 'planning_owner_change_ratio',
        'planning_owner_rewrite_ratio', 'planning_embb_power_nonzero_ratio',
        'planning_embb_power_changed_ratio', 'planning_projected_embb_rate_ratio_mean',
        'planning_projected_embb_rate_ratio_min', 'planning_projected_embb_power_ratio_mean',
        'planning_projected_embb_power_ratio_max', 'planning_owner_rate_floor_violation_count',
        'planning_owner_power_ceiling_violation_count', 'planning_owner_guard_violation_count',
        'phase0_owner_non_null_ratio_raw', 'phase0_owner_non_null_ratio_executed',
        'phase0_owner_change_ratio_vs_snapshot_raw', 'phase0_owner_change_ratio_vs_snapshot_executed',
        'phase0_owner_fallback_to_candidate0_ratio', 'phase0_owner_invalid_option_ratio', 'phase0_owner_null_selected_ratio',
        'phase0_owner_invalid_to_null_ratio', 'phase0_owner_invalid_to_snapshot_ratio', 'phase0_owner_invalid_to_non_snapshot_ratio',
        'phase0_owner_restored_to_snapshot_ratio', 'phase0_owner_kept_null_ratio',
        'phase0_owner_replaced_with_non_snapshot_ratio',
        'phase0_owner_guard_rewrite_ratio', 'phase0_owner_service_violation_ratio', 'phase0_owner_rate_violation_ratio',
        'phase0_owner_change_budget_used', 'phase0_owner_change_budget_allowed',
        'phase0_owner_change_budget_clipped_ratio', 'phase0_owner_change_kept_topk_ratio',
        'phase0_owner_change_dropped_over_budget_ratio',
        'phase0_owner_raw_changed_count_mean', 'phase0_owner_allowed_k_mean',
        'phase0_owner_executed_changed_count_mean', 'phase0_owner_dropped_count_mean',
        'phase0_owner_budget_min_one_rule_eligible_ratio', 'phase0_owner_budget_min_one_rule_applied_ratio',
        'phase0_owner_min_one_blocked_by_no_positive_candidate_ratio',
        'phase0_owner_accepted_positive_service_gain_ratio', 'phase0_owner_accepted_negative_service_gain_ratio',
        'phase0_owner_candidate_positive_objective_ratio', 'phase0_owner_accepted_positive_objective_ratio',
        'phase0_owner_rejected_nonpositive_objective_ratio', 'phase0_owner_objective_gain_mean',
        'phase0_owner_objective_gain_accepted_mean', 'phase0_owner_effective_rate_gain_accepted_mean',
        'phase0_owner_intercell_reduction_accepted_mean', 'phase0_owner_service_gain_accepted_mean',
        'phase0_owner_minrate_gain_accepted_mean', 'phase0_owner_harmful_accepted_ratio',
        'phase0_owner_changed_and_effective_ratio', 'phase0_owner_changed_but_unserved_ratio', 'phase0_owner_same_as_snapshot_ratio',
        'phase0_owner_effective_service_gain_ratio', 'owner_effective_service_gain_ratio', 'phase0_owner_effective_rate_gain_vs_snapshot_mean',
        'phase0_owner_effective_rate_gain_vs_snapshot_cells_mean_mbps',
        'phase0_owner_change_harmful_ratio',
        'phase0_owner_effective_change_count',
        'phase_a_embb_power_write_ratio',
        'phase_a_power_zeroed_non_admission_ratio',
        'phase_a_power_write_on_admission_ratio',
        'phase_a_power_write_on_keep_ratio',
        'action_intercell_guard_active_ratio',
        'action_intercell_guard_candidate_active_ratio',
        'action_intercell_guard_selected_violation_ratio',
        'action_intercell_guard_local_min_cost_mean',
        'action_intercell_guard_selected_excess_mean',
        'phase_a_total_decisions', 'phase_a_embb_power_write_count',
        'phase_a_embb_power_changed_count', 'phase_a_embb_power_changed_ratio', 'phase_a_embb_power_mean_abs_change',
        'phase_a_embb_power_mean_raw_delta', 'phase_a_embb_power_mean_executed_delta',
        'phase_a_embb_power_pre_clip_mean_delta', 'phase_a_embb_power_post_clip_mean_delta',
        'phase_a_embb_power_post_quant_mean_delta', 'phase_a_embb_power_post_projection_mean_delta',
        'phase_a_embb_power_post_owner_validation_mean_delta', 'phase_a_embb_power_final_executed_mean_delta',
        'phase_a_embb_power_clip_ratio', 'phase_a_embb_power_quantized_ratio', 'phase_a_embb_power_invalid_or_masked_ratio',
        'phase_a_embb_power_projection_ratio',
        'phase_a_embb_power_cap_hit_ratio', 'phase_a_embb_power_floor_hit_ratio',
        'phase_a_embb_power_owner_invalid_ratio', 'phase_a_embb_power_no_candidate_ratio', 'phase_a_embb_power_keep_mode_zero_ratio',
        'phase_a_embb_power_sign_flip_ratio', 'phase_a_embb_power_abs_shrink_ratio',
        'phase_a_embb_power_projection_l2_mean', 'phase_a_embb_power_pre_vs_final_l1_mean',
        'phase_a_embb_power_pre_vs_final_sign_consistency', 'phase_a_embb_power_effective_nonzero_ratio',
        'phase_a_embb_power_raw_saturation_ratio', 'phase_a_embb_power_final_std', 'phase_a_embb_power_cellwise_diversity',
        'phase_a_embb_power_floor_binding_strength', 'phase_a_embb_power_cap_binding_strength',
        'phase_a_embb_power_proj_delta_l1', 'phase_a_embb_power_proj_delta_l2',
        'phase_a_embb_power_pre_to_floor_delta', 'phase_a_embb_power_pre_to_cap_delta',
        'phase_a_embb_power_final_minus_proj_delta',
        'phase_a_embb_power_mean_abs_raw_delta', 'phase_a_embb_power_mean_abs_executed_delta',
        'phase_a_embb_power_zeroed_inactive_head_count', 'phase_a_embb_power_zeroed_keep_mode_count',
        'phase_a_embb_power_zeroed_no_candidate_count', 'phase_a_embb_power_zeroed_no_embb_active_count',
        'phase_a_embb_power_zeroed_no_owner_count', 'phase_a_embb_power_zeroed_invalid_owner_count',
        'phase_a_embb_power_zeroed_cap_projection_count', 'phase_a_embb_power_zeroed_floor_projection_count',
        'phase_a_embb_power_zeroed_unknown_count',
        'phase_a_embb_power_zeroed_inactive_head_ratio', 'phase_a_embb_power_zeroed_keep_mode_ratio',
        'phase_a_embb_power_zeroed_no_candidate_ratio', 'phase_a_embb_power_zeroed_no_embb_active_ratio',
        'phase_a_embb_power_zeroed_no_owner_ratio', 'phase_a_embb_power_zeroed_invalid_owner_ratio',
        'phase_a_embb_power_zeroed_cap_projection_ratio', 'phase_a_embb_power_zeroed_floor_projection_ratio',
        'phase_a_embb_power_zeroed_unknown_ratio',
        'power_delta_clipped_ratio', 'power_quantized_ratio',
        'power_cap_hit_ratio', 'power_floor_hit_ratio',
        'mean_raw_power_delta', 'mean_executed_power_delta',
    ]
    # Keep scalar_keys stable and avoid accidental duplicates (duplicates would double-append per load and
    # break fast-debug plots by producing 2x load-length arrays).
    _seen = set()
    scalar_keys = [k for k in scalar_keys if not (k in _seen or _seen.add(k))]
    vector_keys = [
        'per_uav_associated_embb', 'per_uav_associated_urllc', 'per_uav_scheduled_embb',
        'per_uav_scheduled_urllc', 'per_uav_overlay_count', 'per_uav_puncture_count',
        'per_uav_embb_throughput',
    ]
    metrics = {'loads': [], 'lambda': []}
    metrics.update({key: [] for key in scalar_keys + vector_keys})
    metrics['episode_scene_audit'] = {}
    representative = {}
    model = None
    cfg = None
    for load_idx, load in enumerate(loads):
        sys_cfg, urllc_cfg, embb_cfg, algo_cfg, sim_cfg = _configure_density_scenario(
            load, base_sys, base_urllc, base_embb, base_algo, base_sim
        )
        if hasattr(base_sim, 'urllc_user_ratio'):
            sim_cfg.urllc_user_ratio = base_sim.urllc_user_ratio
        if nested_load_enabled and max_total_users > 0:
            setattr(sys_cfg, "nested_load_from_max_users_enabled", True)
            setattr(sys_cfg, "nested_load_max_total_users", int(max_total_users))
            setattr(sys_cfg, "nested_load_max_embb_users", int(max_embb_users))
            setattr(sys_cfg, "nested_load_max_urllc_users", int(max_urllc_users))
            setattr(sys_cfg, "force_serving_hints_association", True)
            freeze_subset_across_episodes = _env_bool_override(
                "SR_MAPPO_REPORT_NESTED_FIXED_SUBSET_ACROSS_EPISODES",
                True,
            )
            setattr(
                report_cfg.env,
                "nested_fixed_user_subset_across_episodes",
                bool(freeze_subset_across_episodes),
            )
            freeze_subset_across_loads = _env_bool_override(
                "SR_MAPPO_REPORT_NESTED_FIXED_SUBSET_ACROSS_LOADS",
                bool(getattr(report_cfg.env, "nested_fixed_user_subset_across_loads", False)),
            )
            setattr(report_cfg.env, "nested_fixed_user_subset_across_loads", bool(freeze_subset_across_loads))
            if bool(freeze_subset_across_loads):
                if shared_nested_ur_order_across_loads is not None and shared_nested_em_order_across_loads is not None:
                    setattr(
                        report_cfg.env,
                        "nested_shared_ur_order_across_loads",
                        np.asarray(shared_nested_ur_order_across_loads, dtype=np.int32).copy(),
                    )
                    setattr(
                        report_cfg.env,
                        "nested_shared_em_order_across_loads",
                        np.asarray(shared_nested_em_order_across_loads, dtype=np.int32).copy(),
                    )
            else:
                if hasattr(report_cfg.env, "nested_shared_ur_order_across_loads"):
                    delattr(report_cfg.env, "nested_shared_ur_order_across_loads")
                if hasattr(report_cfg.env, "nested_shared_em_order_across_loads"):
                    delattr(report_cfg.env, "nested_shared_em_order_across_loads")
            freeze_assoc = _env_bool_override("SR_MAPPO_REPORT_FORCE_FREEZE_ASSOC", True)
            freeze_channel = _env_bool_override("SR_MAPPO_REPORT_FORCE_FREEZE_CHANNEL", True)
            setattr(report_cfg.env, "freeze_association_across_episodes", bool(freeze_assoc))
            setattr(report_cfg.env, "freeze_channel_gains_across_episodes", bool(freeze_channel))
        try:
            nested_embb_max_new_per_load = int(
                str(os.environ.get("SR_MAPPO_NESTED_EMBB_MAX_NEW_PER_LOAD", "0") or "0").strip()
            )
        except Exception:
            nested_embb_max_new_per_load = 0
        if nested_embb_max_new_per_load > 0:
            setattr(report_cfg.env, "nested_embb_max_new_per_load", int(nested_embb_max_new_per_load))
            if prev_nested_embb_served_count is not None:
                setattr(
                    report_cfg.env,
                    "nested_prev_embb_served_count",
                    int(prev_nested_embb_served_count),
                )
            elif hasattr(report_cfg.env, "nested_prev_embb_served_count"):
                delattr(report_cfg.env, "nested_prev_embb_served_count")
            if prev_nested_embb_subset_count is not None:
                setattr(
                    report_cfg.env,
                    "nested_prev_embb_subset_count",
                    int(prev_nested_embb_subset_count),
                )
            elif hasattr(report_cfg.env, "nested_prev_embb_subset_count"):
                delattr(report_cfg.env, "nested_prev_embb_subset_count")
        else:
            if hasattr(report_cfg.env, "nested_embb_max_new_per_load"):
                delattr(report_cfg.env, "nested_embb_max_new_per_load")
            if hasattr(report_cfg.env, "nested_prev_embb_served_count"):
                delattr(report_cfg.env, "nested_prev_embb_served_count")
            if hasattr(report_cfg.env, "nested_prev_embb_subset_count"):
                delattr(report_cfg.env, "nested_prev_embb_subset_count")
        monotone_prerate_guard = bool(
            str(os.environ.get("SR_MAPPO_NESTED_EMBB_PRERATE_GUARD", "0") or "0").strip().lower()
            in {"1", "true", "yes", "on"}
        )
        if monotone_prerate_guard and prev_nested_embb_selected_ids is not None:
            setattr(
                report_cfg.env,
                "nested_prev_embb_selected_ids",
                [int(x) for x in list(prev_nested_embb_selected_ids)],
            )
        elif hasattr(report_cfg.env, "nested_prev_embb_selected_ids"):
            delattr(report_cfg.env, "nested_prev_embb_selected_ids")
        env = SRMAPPOPhaseAEnv(sys_cfg, urllc_cfg, embb_cfg, algo_cfg, sim_cfg, report_cfg)
        if model is None:
            cfg, model = _build_model_for_env(env, checkpoint_path)
        seed_base = _report_seed_base(load_idx, cfg or report_cfg)
        def _episode_runner(ep: int, collect_trace: bool) -> Dict:
            episode_seed = int(seed_base + ep)
            with _temporary_shared_mother_scene_for_episode(
                env=env,
                load=float(load),
                load_idx=int(load_idx),
                episode_seed=episode_seed,
            ):
                return _run_env_episode_virtual_slots(
                    env=env,
                    model=model,
                    cfg=cfg,
                    seed=episode_seed,
                    collect_trace=collect_trace,
                    use_greedy=False,
                    greedy_policy="reference",
                    cache_tag=checkpoint_cache_tag,
                    virtual_slots=virtual_slots_per_episode,
                )
        episodes, representative[load] = _run_episode_batch_with_representative(
            episodes_per_load,
            _episode_runner,
        )
        metrics['loads'].append(float(load))
        metrics['lambda'].append(_load_to_lambda(load))
        metrics['episode_scene_audit'][str(float(load))] = _build_episode_scene_audit(episodes)
        for key in scalar_keys:
            metrics[key].append(_episode_scalar_mean(episodes, key, default=0.0))
        for key in vector_keys:
            metrics[key].append(np.mean(np.stack([episode[key] for episode in episodes]), axis=0))
        if (
            bool(getattr(report_cfg.env, "nested_fixed_user_subset_across_loads", False))
            and (shared_nested_ur_order_across_loads is None or shared_nested_em_order_across_loads is None)
        ):
            try:
                _shared_ur = getattr(env, "_nested_canonical_ur_order_cache", None)
                _shared_em = getattr(env, "_nested_canonical_em_order_cache", None)
                if _shared_ur is not None and _shared_em is not None:
                    shared_nested_ur_order_across_loads = np.asarray(_shared_ur, dtype=np.int32).copy()
                    shared_nested_em_order_across_loads = np.asarray(_shared_em, dtype=np.int32).copy()
            except Exception:
                shared_nested_ur_order_across_loads = None
                shared_nested_em_order_across_loads = None
        rep = representative.get(load, {}) or {}
        try:
            served_counts = [
                float(ep.get("embb_served_user_count", ep.get("embb_served_users", 0.0)) or 0.0)
                for ep in episodes
            ]
            if served_counts:
                prev_nested_embb_served_count = int(max(round(float(np.mean(served_counts))), 0))
            selected_indices_arr = np.asarray(rep.get("nested_selected_user_indices", []), dtype=float)
            if selected_indices_arr.size > 0:
                _nu = int(sys_cfg.num_urllc_users)
                embb_sel = [
                    int(x)
                    for x in selected_indices_arr[_nu:].tolist()
                ]
                prev_nested_embb_subset_count = int(len(embb_sel))
                prev_nested_embb_selected_ids = [int(x) for x in embb_sel]
        except Exception:
            pass
    _report_timing_log(f"run_mappo_sweep loads={len(loads)} episodes_per_load={episodes_per_load} sec={perf_counter() - sweep_start:.3f}")
    return metrics, representative


def run_lambda_sweep_debug(
    load: float,
    lambdas: List[float],
    episodes_per_lambda: int,
    checkpoint_path: Path,
    *,
    baseline_mode: str,
    base_cfg: Optional[SRMAPPOConfig] = None,
) -> Dict[str, object]:
    """Run a small lambda sweep at a fixed load for fast debugging plots."""
    base_sys, base_urllc, base_embb, base_algo, base_sim = _build_main_like_configs()
    sweep_cfg = deepcopy(base_cfg or _load_checkpoint_cfg(checkpoint_path))
    sweep_cfg.env.include_greedy_reference_in_obs = False

    def _baseline_policy_for_mode(mode: str) -> str:
        normalized = str(mode or "").strip().lower()
        if normalized == "myopic_throughput_greedy":
            return "myopic_throughput"
        if normalized == "hard_feasible_throughput_greedy":
            return "hard_feasible_throughput"
        if normalized == "global_frontier_greedy":
            return "global_frontier"
        if normalized == "throughput_only_greedy":
            return "throughput_only"
        if normalized == "channel_only_greedy":
            return "channel_only"
        if normalized == "throughput_feasible_oracle":
            return "throughput_feasible"
        return "reference"

    greedy_policy = _baseline_policy_for_mode(baseline_mode)
    series = {
        "load": float(load),
        "lambda": [],
        "mappo": {
            "urllc_admission": [],
            "embb_rate": [],
            "total_power": [],
            "embb_service_ratio": [],
            "embb_min_rate_satisfaction_ratio": [],
            "overlay_ratio": [],
            "puncture_ratio": [],
            "urllc_slot_duration_s": [],
            "urllc_packet_bits_mean": [],
            "urllc_throughput_bps_slot_est": [],
            "embb_power": [],
            "urllc_power": [],
        },
        "baseline": {
            "urllc_admission": [],
            "embb_rate": [],
            "total_power": [],
            "embb_service_ratio": [],
            "embb_min_rate_satisfaction_ratio": [],
            "overlay_ratio": [],
            "puncture_ratio": [],
            "urllc_slot_duration_s": [],
            "urllc_packet_bits_mean": [],
            "urllc_throughput_bps_slot_est": [],
            "embb_power": [],
            "urllc_power": [],
        },
        "baseline_mode": str(baseline_mode),
        "greedy_policy": str(greedy_policy),
        "embb_user_count": 0.0,
        "urllc_user_count": 0.0,
    }
    model = None
    cfg = None
    for idx, lam in enumerate([float(item) for item in lambdas]):
        sys_cfg, urllc_cfg, embb_cfg, algo_cfg, sim_cfg = _configure_density_scenario(
            float(load), base_sys, base_urllc, base_embb, base_algo, base_sim
        )
        if hasattr(base_sim, "urllc_user_ratio"):
            sim_cfg.urllc_user_ratio = base_sim.urllc_user_ratio
        if hasattr(sim_cfg, "fixed_urllc_poisson_rate"):
            sim_cfg.fixed_urllc_poisson_rate = True
        if hasattr(sim_cfg, "urllc_poisson_rate"):
            sim_cfg.urllc_poisson_rate = float(lam)
        sim_cfg.verbose = False

        env = SRMAPPOPhaseAEnv(sys_cfg, urllc_cfg, embb_cfg, algo_cfg, sim_cfg, sweep_cfg)
        if not series.get("embb_user_count"):
            series["embb_user_count"] = float(int(getattr(sys_cfg, "num_embb_users", 0) or 0))
            series["urllc_user_count"] = float(int(getattr(sys_cfg, "num_urllc_users", 0) or 0))
        if model is None:
            cfg, model = _build_model_for_env(env, checkpoint_path)
        seed_base = 20_000 + 131 * idx
        mappo_runs = []
        baseline_runs = []
        for ep in range(max(int(episodes_per_lambda), 1)):
            mappo_runs.append(
                run_env_episode(
                    env,
                    model=model,
                    cfg=cfg,
                    seed=seed_base + ep,
                    collect_trace=False,
                    use_greedy=False,
                    cache_tag=f"lambda_sweep_{load}_{lam}",
                )
            )
            baseline_runs.append(
                run_env_episode(
                    env,
                    model=None,
                    cfg=cfg,
                    seed=seed_base + ep + 50_000,
                    collect_trace=False,
                    use_greedy=True,
                    greedy_policy=greedy_policy,
                    cache_tag=f"lambda_sweep_{load}_{lam}_baseline",
                )
            )

        def _mean(items: List[Dict], key: str, default: float = 0.0) -> float:
            arr = np.asarray([float(item.get(key, default)) for item in items], dtype=float)
            arr = arr[np.isfinite(arr)]
            return float(np.mean(arr)) if arr.size else float(default)

        series["lambda"].append(float(lam))
        for key in (
            "urllc_admission",
            "embb_rate",
            "total_power",
            "embb_service_ratio",
            "embb_min_rate_satisfaction_ratio",
            "overlay_ratio",
            "puncture_ratio",
            "urllc_slot_duration_s",
            "urllc_packet_bits_mean",
            "urllc_throughput_bps_slot_est",
            "embb_power",
            "urllc_power",
        ):
            series["mappo"][key].append(_mean(mappo_runs, key, default=0.0))
            series["baseline"][key].append(_mean(baseline_runs, key, default=0.0))
    return series


def run_greedy_normal_v1_sweep(loads: List[float], episodes_per_load: int) -> Tuple[Dict, Dict]:
    base_sys, base_urllc, base_embb, base_algo, base_sim = _build_main_like_configs()
    report_cfg = SRMAPPOConfig()
    scalar_keys = [
        'embb_user_count', 'urllc_user_count', 'embb_urllc_user_ratio', 'urllc_throughput_bps_est',
        'embb_rate', 'embb_user_rate', 'embb_service_ratio', 'embb_positive_rate_ratio', 'embb_min_rate_satisfaction_ratio', 'embb_min_rate_satisfied_user_count', 'urllc_admission',
        'admitted_urllc_reliability', 'urllc_reliability', 'effective_urllc_success_over_arrivals', 'empty_admission_case',
        'active_packets', 'scheduled_packets', 'total_power', 'embb_power', 'urllc_power',
        'overlay_ratio', 'puncture_ratio', 'overlay_selection_ratio', 'puncture_selection_ratio', 'embb_only_fraction', 'avg_puncture_loss',
        'avg_overlay_retention', 'overlay_candidate_pairs', 'overlay_feasible_pairs',
        'overlay_selected_pairs', 'admission_via_overlay_ratio', 'admission_via_puncture_ratio', 'jain_fairness', 'cell_edge_served_ratio', 'cell_edge_min_rate_satisfaction_ratio',
        'per_uav_total_load_std', 'per_uav_urllc_sched_std', 'per_uav_throughput_std',
    ]
    vector_keys = [
        'per_uav_associated_embb', 'per_uav_associated_urllc', 'per_uav_scheduled_embb',
        'per_uav_scheduled_urllc', 'per_uav_overlay_count', 'per_uav_puncture_count',
        'per_uav_embb_throughput',
    ]
    metrics = {'loads': [], 'lambda': []}
    metrics.update({key: [] for key in scalar_keys + vector_keys})
    representative = {}
    for load_idx, load in enumerate(loads):
        sys_cfg, urllc_cfg, embb_cfg, algo_cfg, sim_cfg = _configure_density_scenario(
            load, base_sys, base_urllc, base_embb, base_algo, base_sim
        )
        if hasattr(base_sim, 'urllc_user_ratio'):
            sim_cfg.urllc_user_ratio = base_sim.urllc_user_ratio
        seed_base = _report_seed_base(load_idx, report_cfg)
        episodes = [
            _run_original_greedy_normal_v1_slot(
                sys_cfg,
                urllc_cfg,
                embb_cfg,
                algo_cfg,
                sim_cfg,
                seed=seed_base + ep,
                slot_index=ep,
            )
            for ep in range(episodes_per_load)
        ]
        representative[load] = _run_original_greedy_normal_v1_slot(
            sys_cfg,
            urllc_cfg,
            embb_cfg,
            algo_cfg,
            sim_cfg,
            seed=seed_base + 999,
            slot_index=0,
        )
        metrics['loads'].append(float(load))
        metrics['lambda'].append(_load_to_lambda(load))
        for key in scalar_keys:
            metrics[key].append(_episode_scalar_aggregate(episodes, key, default=np.nan))
        for key in vector_keys:
            metrics[key].append(np.mean(np.stack([episode[key] for episode in episodes]), axis=0))
    return metrics, representative


def run_greedy_normal_v2_sweep(loads: List[float], episodes_per_load: int) -> Tuple[Dict, Dict]:
    base_sys, base_urllc, base_embb, base_algo, base_sim = _build_main_like_configs()
    report_cfg = SRMAPPOConfig()
    scalar_keys = [
        'embb_rate', 'embb_user_rate', 'embb_service_ratio', 'embb_positive_rate_ratio', 'embb_min_rate_satisfaction_ratio', 'embb_min_rate_satisfied_user_count', 'urllc_admission',
        'admitted_urllc_reliability', 'urllc_reliability', 'effective_urllc_success_over_arrivals', 'empty_admission_case',
        'active_packets', 'scheduled_packets', 'total_power', 'embb_power', 'urllc_power',
        'overlay_ratio', 'puncture_ratio', 'overlay_selection_ratio', 'puncture_selection_ratio', 'embb_only_fraction', 'avg_puncture_loss',
        'avg_overlay_retention', 'overlay_candidate_pairs', 'overlay_feasible_pairs',
        'overlay_selected_pairs', 'admission_via_overlay_ratio', 'admission_via_puncture_ratio', 'jain_fairness', 'cell_edge_served_ratio', 'cell_edge_min_rate_satisfaction_ratio',
        'per_uav_total_load_std', 'per_uav_urllc_sched_std', 'per_uav_throughput_std',
    ]
    vector_keys = [
        'per_uav_associated_embb', 'per_uav_associated_urllc', 'per_uav_scheduled_embb',
        'per_uav_scheduled_urllc', 'per_uav_overlay_count', 'per_uav_puncture_count',
        'per_uav_embb_throughput',
    ]
    metrics = {'loads': [], 'lambda': []}
    metrics.update({key: [] for key in scalar_keys + vector_keys})
    representative = {}
    for load_idx, load in enumerate(loads):
        sys_cfg, urllc_cfg, embb_cfg, algo_cfg, sim_cfg = _configure_density_scenario(
            load, base_sys, base_urllc, base_embb, base_algo, base_sim
        )
        if hasattr(base_sim, 'urllc_user_ratio'):
            sim_cfg.urllc_user_ratio = base_sim.urllc_user_ratio
        seed_base = _report_seed_base(load_idx, report_cfg)
        episodes = [
            _run_original_greedy_normal_v2_slot(
                sys_cfg,
                urllc_cfg,
                embb_cfg,
                algo_cfg,
                sim_cfg,
                seed=seed_base + ep,
                slot_index=ep,
            )
            for ep in range(episodes_per_load)
        ]
        representative[load] = _run_original_greedy_normal_v2_slot(
            sys_cfg,
            urllc_cfg,
            embb_cfg,
            algo_cfg,
            sim_cfg,
            seed=seed_base + 999,
            slot_index=0,
        )
        metrics['loads'].append(float(load))
        metrics['lambda'].append(_load_to_lambda(load))
        for key in scalar_keys:
            metrics[key].append(_episode_scalar_aggregate(episodes, key, default=np.nan))
        for key in vector_keys:
            metrics[key].append(np.mean(np.stack([episode[key] for episode in episodes]), axis=0))
    return metrics, representative


def run_embb_only_ceiling_sweep(loads: List[float], episodes_per_load: int) -> Tuple[Dict, Dict]:
    sweep_start = perf_counter()
    base_sys, base_urllc, base_embb, base_algo, base_sim = _build_main_like_configs()
    report_cfg = SRMAPPOConfig()
    scalar_keys = [
        'embb_rate', 'embb_user_rate', 'embb_service_ratio', 'embb_positive_rate_ratio', 'embb_min_rate_satisfaction_ratio', 'embb_min_rate_satisfied_user_count', 'urllc_admission',
        'admitted_urllc_reliability', 'urllc_reliability', 'effective_urllc_success_over_arrivals', 'empty_admission_case',
        'active_packets', 'scheduled_packets', 'total_power', 'embb_power', 'urllc_power',
        'overlay_ratio', 'puncture_ratio', 'overlay_selection_ratio', 'puncture_selection_ratio', 'embb_only_fraction', 'avg_puncture_loss',
        'avg_overlay_retention', 'overlay_candidate_pairs', 'overlay_feasible_pairs',
        'overlay_selected_pairs', 'admission_via_overlay_ratio', 'admission_via_puncture_ratio', 'jain_fairness', 'cell_edge_served_ratio', 'cell_edge_min_rate_satisfaction_ratio',
        'per_uav_total_load_std', 'per_uav_urllc_sched_std', 'per_uav_throughput_std',
    ]
    vector_keys = [
        'per_uav_associated_embb', 'per_uav_associated_urllc', 'per_uav_scheduled_embb',
        'per_uav_scheduled_urllc', 'per_uav_overlay_count', 'per_uav_puncture_count',
        'per_uav_embb_throughput',
    ]
    metrics = {'loads': [], 'lambda': []}
    metrics.update({key: [] for key in scalar_keys + vector_keys})
    representative = {}
    for load_idx, load in enumerate(loads):
        sys_cfg, urllc_cfg, embb_cfg, algo_cfg, sim_cfg = _configure_density_scenario(
            load, base_sys, base_urllc, base_embb, base_algo, base_sim
        )
        if hasattr(base_sim, 'urllc_user_ratio'):
            sim_cfg.urllc_user_ratio = base_sim.urllc_user_ratio
        seed_base = _report_seed_base(load_idx, report_cfg)
        episodes, representative[load] = _run_episode_batch_with_representative(
            episodes_per_load,
            lambda ep, _collect_trace: _run_embb_only_ceiling_slot(
                sys_cfg,
                urllc_cfg,
                embb_cfg,
                algo_cfg,
                sim_cfg,
                seed=seed_base + ep,
                slot_index=ep,
            ),
        )
        metrics['loads'].append(float(load))
        metrics['lambda'].append(_load_to_lambda(load))
        for key in scalar_keys:
            metrics[key].append(_episode_scalar_aggregate(episodes, key, default=np.nan))
        for key in vector_keys:
            metrics[key].append(np.mean(np.stack([episode[key] for episode in episodes]), axis=0))
    _report_timing_log(
        f"run_embb_only_ceiling_sweep loads={len(loads)} episodes_per_load={episodes_per_load} sec={perf_counter() - sweep_start:.3f}"
    )
    return metrics, representative


def run_throughput_feasible_oracle_sweep(loads: List[float], episodes_per_load: int) -> Tuple[Dict, Dict]:
    sweep_start = perf_counter()
    base_sys, base_urllc, base_embb, base_algo, base_sim = _build_main_like_configs()
    report_cfg = SRMAPPOConfig()
    scalar_keys = [
        'embb_rate', 'embb_user_rate', 'embb_service_ratio', 'embb_positive_rate_ratio', 'embb_min_rate_satisfaction_ratio', 'embb_min_rate_satisfied_user_count', 'urllc_admission',
        'admitted_urllc_reliability', 'urllc_reliability', 'effective_urllc_success_over_arrivals', 'empty_admission_case',
        'active_packets', 'scheduled_packets', 'total_power', 'embb_power', 'urllc_power',
        'overlay_ratio', 'puncture_ratio', 'overlay_selection_ratio', 'puncture_selection_ratio', 'embb_only_fraction', 'avg_puncture_loss',
        'avg_overlay_retention', 'overlay_candidate_pairs', 'overlay_feasible_pairs',
        'overlay_selected_pairs', 'admission_via_overlay_ratio', 'admission_via_puncture_ratio', 'jain_fairness', 'cell_edge_served_ratio', 'cell_edge_min_rate_satisfaction_ratio',
        'per_uav_total_load_std', 'per_uav_urllc_sched_std', 'per_uav_throughput_std',
    ]
    vector_keys = [
        'per_uav_associated_embb', 'per_uav_associated_urllc', 'per_uav_scheduled_embb',
        'per_uav_scheduled_urllc', 'per_uav_overlay_count', 'per_uav_puncture_count',
        'per_uav_embb_throughput',
    ]
    metrics = {'loads': [], 'lambda': []}
    metrics.update({key: [] for key in scalar_keys + vector_keys})
    representative = {}
    for load_idx, load in enumerate(loads):
        sys_cfg, urllc_cfg, embb_cfg, algo_cfg, sim_cfg = _configure_density_scenario(
            load, base_sys, base_urllc, base_embb, base_algo, base_sim
        )
        if hasattr(base_sim, 'urllc_user_ratio'):
            sim_cfg.urllc_user_ratio = base_sim.urllc_user_ratio
        seed_base = _report_seed_base(load_idx, report_cfg)
        episodes, representative[load] = _run_episode_batch_with_representative(
            episodes_per_load,
            lambda ep, _collect_trace: _run_throughput_feasible_oracle_slot(
                sys_cfg,
                urllc_cfg,
                embb_cfg,
                algo_cfg,
                sim_cfg,
                seed=seed_base + ep,
                slot_index=ep,
            ),
        )
        metrics['loads'].append(float(load))
        metrics['lambda'].append(_load_to_lambda(load))
        for key in scalar_keys:
            metrics[key].append(_episode_scalar_aggregate(episodes, key, default=np.nan))
        for key in vector_keys:
            metrics[key].append(np.mean(np.stack([episode[key] for episode in episodes]), axis=0))
    _report_timing_log(
        f"run_throughput_feasible_oracle_sweep loads={len(loads)} episodes_per_load={episodes_per_load} sec={perf_counter() - sweep_start:.3f}"
    )
    return metrics, representative


def run_throughput_admission_frontier_bundle(loads: List[float]) -> Dict[float, Dict]:
    base_sys, base_urllc, base_embb, base_algo, base_sim = _build_main_like_configs()
    report_cfg = SRMAPPOConfig()
    bundle: Dict[float, Dict] = {}
    for load_idx, load in enumerate(loads):
        sys_cfg, urllc_cfg, embb_cfg, algo_cfg, sim_cfg = _configure_density_scenario(
            load, base_sys, base_urllc, base_embb, base_algo, base_sim
        )
        if hasattr(base_sim, 'urllc_user_ratio'):
            sim_cfg.urllc_user_ratio = base_sim.urllc_user_ratio
        seed = _report_seed_base(load_idx, report_cfg) + 999
        sim_local = deepcopy(sim_cfg)
        sim_local.random_seed = int(seed)
        simulation = create_simulation(sys_cfg, urllc_cfg, embb_cfg, algo_cfg, sim_local)
        slot_context = simulation._generate_slot_context()
        frontier = simulation.run_throughput_admission_frontier(
            slot_index=0,
            slot_context=slot_context,
        )
        active_packets = int(frontier.get('active_packets', 0))
        quota_candidates = sorted(set(
            int(np.clip(round(active_packets * frac), 0, active_packets))
            for frac in (0.0, 0.25, 0.5, 0.75, 1.0)
        ))
        operating_points = []
        for quota in quota_candidates:
            point = next((item for item in frontier['frontier'] if int(item['quota']) == quota), None)
            if point is not None:
                operating_points.append(point)
        frontier['operating_points'] = operating_points
        bundle[float(load)] = frontier
    return bundle


def run_matched_greedy_sweep(
    loads: List[float],
    episodes_per_load: int,
    checkpoint_path: Path,
    base_cfg: Optional[SRMAPPOConfig] = None,
) -> Tuple[Dict, Dict]:
    sweep_start = perf_counter()
    base_sys, base_urllc, base_embb, base_algo, base_sim = _build_main_like_configs()
    report_cfg = deepcopy(base_cfg or _load_checkpoint_cfg(checkpoint_path))
    report_cfg.env.include_greedy_reference_in_obs = False
    report_cfg.training.greedy_baseline_mode = "matched_fixed_embb"
    report_cfg.training.selection_baseline_mode = "matched_fixed_embb"
    checkpoint_cache_tag = str(Path(checkpoint_path).resolve())
    scalar_keys = [
        'embb_rate', 'embb_user_rate', 'embb_service_ratio', 'embb_positive_rate_ratio', 'embb_min_rate_satisfaction_ratio', 'embb_min_rate_satisfied_user_count', 'urllc_admission',
        'admitted_urllc_reliability', 'urllc_reliability', 'effective_urllc_success_over_arrivals', 'empty_admission_case',
        'active_packets', 'scheduled_packets', 'total_power', 'embb_power', 'urllc_power',
        'overlay_ratio', 'puncture_ratio', 'overlay_selection_ratio', 'puncture_selection_ratio', 'embb_only_fraction', 'avg_puncture_loss',
        'avg_overlay_retention', 'overlay_candidate_pairs', 'overlay_feasible_pairs',
        'overlay_selected_pairs', 'admission_via_overlay_ratio', 'admission_via_puncture_ratio', 'jain_fairness', 'cell_edge_served_ratio', 'cell_edge_min_rate_satisfaction_ratio',
        'per_uav_total_load_std', 'per_uav_urllc_sched_std', 'per_uav_throughput_std',
        'shield_correction_ratio', 'collision_rewrite_ratio', 'fallback_ratio',
        'mode_correction_ratio', 'packet_invalid_ratio', 'mask_invalid_ratio',
        'joint_reliability_rewrite_ratio',
    ]
    vector_keys = [
        'per_uav_associated_embb', 'per_uav_associated_urllc', 'per_uav_scheduled_embb',
        'per_uav_scheduled_urllc', 'per_uav_overlay_count', 'per_uav_puncture_count',
        'per_uav_embb_throughput',
    ]
    metrics = {'loads': [], 'lambda': []}
    metrics.update({key: [] for key in scalar_keys + vector_keys})
    representative = {}
    for load_idx, load in enumerate(loads):
        sys_cfg, urllc_cfg, embb_cfg, algo_cfg, sim_cfg = _configure_density_scenario(
            load, base_sys, base_urllc, base_embb, base_algo, base_sim
        )
        if hasattr(base_sim, 'urllc_user_ratio'):
            sim_cfg.urllc_user_ratio = base_sim.urllc_user_ratio
        env = SRMAPPOPhaseAEnv(sys_cfg, urllc_cfg, embb_cfg, algo_cfg, sim_cfg, report_cfg)
        seed_base = _report_seed_base(load_idx, report_cfg)
        episodes, representative[load] = _run_episode_batch_with_representative(
            episodes_per_load,
            lambda ep, collect_trace: run_env_episode(
                env,
                model=None,
                cfg=report_cfg,
                seed=seed_base + ep,
                collect_trace=collect_trace,
                use_greedy=True,
                cache_tag=checkpoint_cache_tag,
            ),
        )
        metrics['loads'].append(float(load))
        metrics['lambda'].append(_load_to_lambda(load))
        for key in scalar_keys:
            metrics[key].append(_episode_scalar_aggregate(episodes, key, default=0.0))
        for key in vector_keys:
            metrics[key].append(np.mean(np.stack([episode[key] for episode in episodes]), axis=0))
    _report_timing_log(
        f"run_matched_greedy_sweep loads={len(loads)} episodes_per_load={episodes_per_load} sec={perf_counter() - sweep_start:.3f}"
    )
    return metrics, representative


def run_channel_only_greedy_sweep(
    loads: List[float],
    episodes_per_load: int,
    checkpoint_path: Path,
    base_cfg: Optional[SRMAPPOConfig] = None,
) -> Tuple[Dict, Dict]:
    sweep_start = perf_counter()
    base_sys, base_urllc, base_embb, base_algo, base_sim = _build_main_like_configs()
    report_cfg = deepcopy(base_cfg or _load_checkpoint_cfg(checkpoint_path))
    report_cfg.env.include_greedy_reference_in_obs = False
    report_cfg.training.greedy_baseline_mode = "channel_only_greedy"
    report_cfg.training.selection_baseline_mode = "channel_only_greedy"
    checkpoint_cache_tag = str(Path(checkpoint_path).resolve())
    scalar_keys = [
        'embb_rate', 'embb_user_rate', 'embb_service_ratio', 'embb_positive_rate_ratio', 'embb_min_rate_satisfaction_ratio', 'embb_min_rate_satisfied_user_count', 'urllc_admission',
        'admitted_urllc_reliability', 'urllc_reliability', 'effective_urllc_success_over_arrivals', 'empty_admission_case',
        'active_packets', 'scheduled_packets', 'total_power', 'embb_power', 'urllc_power',
        'overlay_ratio', 'puncture_ratio', 'overlay_selection_ratio', 'puncture_selection_ratio', 'embb_only_fraction', 'avg_puncture_loss',
        'avg_overlay_retention', 'overlay_candidate_pairs', 'overlay_feasible_pairs',
        'overlay_selected_pairs', 'admission_via_overlay_ratio', 'admission_via_puncture_ratio', 'jain_fairness', 'cell_edge_served_ratio', 'cell_edge_min_rate_satisfaction_ratio',
        'per_uav_total_load_std', 'per_uav_urllc_sched_std', 'per_uav_throughput_std',
        'shield_correction_ratio', 'collision_rewrite_ratio', 'fallback_ratio',
        'mode_correction_ratio', 'packet_invalid_ratio', 'mask_invalid_ratio',
        'joint_reliability_rewrite_ratio',
    ]
    vector_keys = [
        'per_uav_associated_embb', 'per_uav_associated_urllc', 'per_uav_scheduled_embb',
        'per_uav_scheduled_urllc', 'per_uav_overlay_count', 'per_uav_puncture_count',
        'per_uav_embb_throughput',
    ]
    metrics = {'loads': [], 'lambda': []}
    metrics.update({key: [] for key in scalar_keys + vector_keys})
    representative = {}
    for load_idx, load in enumerate(loads):
        sys_cfg, urllc_cfg, embb_cfg, algo_cfg, sim_cfg = _configure_density_scenario(
            load, base_sys, base_urllc, base_embb, base_algo, base_sim
        )
        if hasattr(base_sim, 'urllc_user_ratio'):
            sim_cfg.urllc_user_ratio = base_sim.urllc_user_ratio
        env = SRMAPPOPhaseAEnv(sys_cfg, urllc_cfg, embb_cfg, algo_cfg, sim_cfg, report_cfg)
        seed_base = _report_seed_base(load_idx, report_cfg)
        episodes, representative[load] = _run_episode_batch_with_representative(
            episodes_per_load,
            lambda ep, collect_trace: run_env_episode(
                env,
                model=None,
                cfg=report_cfg,
                seed=seed_base + ep,
                collect_trace=collect_trace,
                use_greedy=True,
                greedy_policy="channel_only",
                cache_tag=checkpoint_cache_tag,
            ),
        )
        metrics['loads'].append(float(load))
        metrics['lambda'].append(_load_to_lambda(load))
        for key in scalar_keys:
            metrics[key].append(_episode_scalar_aggregate(episodes, key, default=0.0))
        for key in vector_keys:
            metrics[key].append(np.mean(np.stack([episode[key] for episode in episodes]), axis=0))
    _report_timing_log(
        f"run_channel_only_greedy_sweep loads={len(loads)} episodes_per_load={episodes_per_load} sec={perf_counter() - sweep_start:.3f}"
    )
    return metrics, representative


def run_throughput_only_greedy_sweep(
    loads: List[float],
    episodes_per_load: int,
    checkpoint_path: Path,
    base_cfg: Optional[SRMAPPOConfig] = None,
) -> Tuple[Dict, Dict]:
    sweep_start = perf_counter()
    base_sys, base_urllc, base_embb, base_algo, base_sim = _build_main_like_configs()
    report_cfg = deepcopy(base_cfg or _load_checkpoint_cfg(checkpoint_path))
    report_cfg.env.include_greedy_reference_in_obs = False
    report_cfg.training.greedy_baseline_mode = "throughput_only_greedy"
    report_cfg.training.selection_baseline_mode = "throughput_only_greedy"
    checkpoint_cache_tag = str(Path(checkpoint_path).resolve())
    scalar_keys = [
        'embb_rate', 'embb_user_rate', 'embb_service_ratio', 'embb_positive_rate_ratio', 'embb_min_rate_satisfaction_ratio', 'embb_min_rate_satisfied_user_count', 'urllc_admission',
        'admitted_urllc_reliability', 'urllc_reliability', 'effective_urllc_success_over_arrivals', 'empty_admission_case',
        'active_packets', 'scheduled_packets', 'total_power', 'embb_power', 'urllc_power',
        'overlay_ratio', 'puncture_ratio', 'overlay_selection_ratio', 'puncture_selection_ratio', 'embb_only_fraction', 'avg_puncture_loss',
        'avg_overlay_retention', 'overlay_candidate_pairs', 'overlay_feasible_pairs',
        'overlay_selected_pairs', 'admission_via_overlay_ratio', 'admission_via_puncture_ratio', 'jain_fairness', 'cell_edge_served_ratio', 'cell_edge_min_rate_satisfaction_ratio',
        'per_uav_total_load_std', 'per_uav_urllc_sched_std', 'per_uav_throughput_std',
        'shield_correction_ratio', 'collision_rewrite_ratio', 'fallback_ratio',
        'mode_correction_ratio', 'packet_invalid_ratio', 'mask_invalid_ratio',
        'joint_reliability_rewrite_ratio', 'greedy_noop_selected_ratio', 'greedy_admit_selected_ratio',
        'greedy_overlay_ratio', 'greedy_puncture_ratio', 'greedy_avg_embb_retention',
        'greedy_avg_embb_loss', 'greedy_avg_selected_throughput',
        'greedy_avg_rejected_urllc_when_noop_better', 'greedy_noop_available_ratio',
        'greedy_noop_better_ratio', 'greedy_requires_feasible_admission_only',
        'urllc_slot_duration_s', 'urllc_packet_bits_mean',
        'urllc_throughput_bps_slot_est', 'urllc_throughput_mbps_slot_est', 'urllc_throughput_bps_est',
        'mean_intercell_interference_power', 'mean_intercell_interference_mw', 'mean_intercell_interference_dbm',
        'intercell_interference_nonzero_ratio', 'overlay_intercell_interference_mw', 'puncture_intercell_interference_mw',
    ]
    vector_keys = [
        'per_uav_associated_embb', 'per_uav_associated_urllc', 'per_uav_scheduled_embb',
        'per_uav_scheduled_urllc', 'per_uav_overlay_count', 'per_uav_puncture_count',
        'per_uav_embb_throughput',
    ]
    metrics = {'loads': [], 'lambda': []}
    metrics.update({key: [] for key in scalar_keys + vector_keys})
    representative = {}
    for load_idx, load in enumerate(loads):
        sys_cfg, urllc_cfg, embb_cfg, algo_cfg, sim_cfg = _configure_density_scenario(
            load, base_sys, base_urllc, base_embb, base_algo, base_sim
        )
        if hasattr(base_sim, 'urllc_user_ratio'):
            sim_cfg.urllc_user_ratio = base_sim.urllc_user_ratio
        env = SRMAPPOPhaseAEnv(sys_cfg, urllc_cfg, embb_cfg, algo_cfg, sim_cfg, report_cfg)
        seed_base = _report_seed_base(load_idx, report_cfg)
        episodes, representative[load] = _run_episode_batch_with_representative(
            episodes_per_load,
            lambda ep, collect_trace: run_env_episode(
                env,
                model=None,
                cfg=report_cfg,
                seed=seed_base + ep,
                collect_trace=collect_trace,
                use_greedy=True,
                greedy_policy="throughput_only",
                cache_tag=checkpoint_cache_tag,
            ),
        )
        metrics['loads'].append(float(load))
        metrics['lambda'].append(_load_to_lambda(load))
        for key in scalar_keys:
            metrics[key].append(_episode_scalar_aggregate(episodes, key, default=0.0))
        for key in vector_keys:
            metrics[key].append(np.mean(np.stack([episode[key] for episode in episodes]), axis=0))
    _report_timing_log(
        f"run_throughput_only_greedy_sweep loads={len(loads)} episodes_per_load={episodes_per_load} sec={perf_counter() - sweep_start:.3f}"
    )
    return metrics, representative


def run_rate_loss_min_greedy_sweep(
    loads: List[float],
    episodes_per_load: int,
    checkpoint_path: Path,
    base_cfg: Optional[SRMAPPOConfig] = None,
) -> Tuple[Dict, Dict]:
    sweep_start = perf_counter()
    base_sys, base_urllc, base_embb, base_algo, base_sim = _build_main_like_configs()
    report_cfg = deepcopy(base_cfg or _load_checkpoint_cfg(checkpoint_path))
    report_cfg.env.include_greedy_reference_in_obs = False
    report_cfg.training.greedy_baseline_mode = "rate_loss_min_greedy"
    report_cfg.training.selection_baseline_mode = "rate_loss_min_greedy"
    checkpoint_cache_tag = str(Path(checkpoint_path).resolve())
    scalar_keys = [
        'embb_rate', 'embb_user_rate', 'embb_service_ratio', 'embb_positive_rate_ratio', 'embb_min_rate_satisfaction_ratio', 'embb_min_rate_satisfied_user_count', 'urllc_admission',
        'admitted_urllc_reliability', 'urllc_reliability', 'effective_urllc_success_over_arrivals', 'empty_admission_case',
        'active_packets', 'scheduled_packets', 'total_power', 'embb_power', 'urllc_power',
        'overlay_ratio', 'puncture_ratio', 'overlay_selection_ratio', 'puncture_selection_ratio', 'embb_only_fraction', 'avg_puncture_loss',
        'avg_overlay_retention', 'overlay_candidate_pairs', 'overlay_feasible_pairs',
        'overlay_selected_pairs', 'admission_via_overlay_ratio', 'admission_via_puncture_ratio', 'jain_fairness', 'cell_edge_served_ratio', 'cell_edge_min_rate_satisfaction_ratio',
        'per_uav_total_load_std', 'per_uav_urllc_sched_std', 'per_uav_throughput_std',
        'shield_correction_ratio', 'collision_rewrite_ratio', 'fallback_ratio',
        'mode_correction_ratio', 'packet_invalid_ratio', 'mask_invalid_ratio',
        'joint_reliability_rewrite_ratio', 'greedy_noop_selected_ratio', 'greedy_admit_selected_ratio',
        'greedy_overlay_ratio', 'greedy_puncture_ratio', 'greedy_avg_embb_retention',
        'greedy_avg_embb_loss', 'greedy_avg_selected_throughput',
        'greedy_avg_rejected_urllc_when_noop_better', 'greedy_noop_available_ratio',
        'greedy_noop_better_ratio', 'greedy_requires_feasible_admission_only',
        'urllc_slot_duration_s', 'urllc_packet_bits_mean',
        'urllc_throughput_bps_slot_est', 'urllc_throughput_mbps_slot_est', 'urllc_throughput_bps_est',
        'mean_intercell_interference_power', 'mean_intercell_interference_mw', 'mean_intercell_interference_dbm',
        'intercell_interference_nonzero_ratio', 'overlay_intercell_interference_mw', 'puncture_intercell_interference_mw',
    ]
    vector_keys = [
        'per_uav_associated_embb', 'per_uav_associated_urllc', 'per_uav_scheduled_embb',
        'per_uav_scheduled_urllc', 'per_uav_overlay_count', 'per_uav_puncture_count',
        'per_uav_embb_throughput',
    ]
    metrics = {'loads': [], 'lambda': []}
    metrics.update({key: [] for key in scalar_keys + vector_keys})
    metrics['episode_scene_audit'] = {}
    representative = {}
    for load_idx, load in enumerate(loads):
        sys_cfg, urllc_cfg, embb_cfg, algo_cfg, sim_cfg = _configure_density_scenario(
            load, base_sys, base_urllc, base_embb, base_algo, base_sim
        )
        if hasattr(base_sim, 'urllc_user_ratio'):
            sim_cfg.urllc_user_ratio = base_sim.urllc_user_ratio
        env = SRMAPPOPhaseAEnv(sys_cfg, urllc_cfg, embb_cfg, algo_cfg, sim_cfg, report_cfg)
        seed_base = _report_seed_base(load_idx, report_cfg)
        episodes, representative[load] = _run_episode_batch_with_representative(
            episodes_per_load,
            lambda ep, collect_trace: run_env_episode(
                env,
                model=None,
                cfg=report_cfg,
                seed=seed_base + ep,
                collect_trace=collect_trace,
                use_greedy=True,
                greedy_policy="rate_loss_min",
                cache_tag=checkpoint_cache_tag,
            ),
        )
        metrics['loads'].append(float(load))
        metrics['lambda'].append(_load_to_lambda(load))
        for key in scalar_keys:
            metrics[key].append(_episode_scalar_aggregate(episodes, key, default=0.0))
        for key in vector_keys:
            metrics[key].append(np.mean(np.stack([episode[key] for episode in episodes]), axis=0))
    _report_timing_log(
        f"run_rate_loss_min_greedy_sweep loads={len(loads)} episodes_per_load={episodes_per_load} sec={perf_counter() - sweep_start:.3f}"
    )
    return metrics, representative


def run_force_admit_minloss_greedy_sweep(
    loads: List[float],
    episodes_per_load: int,
    checkpoint_path: Path,
    base_cfg: Optional[SRMAPPOConfig] = None,
) -> Tuple[Dict, Dict]:
    sweep_start = perf_counter()
    base_sys, base_urllc, base_embb, base_algo, base_sim = _build_main_like_configs()
    report_cfg = deepcopy(base_cfg or _load_checkpoint_cfg(checkpoint_path))
    report_cfg.env.include_greedy_reference_in_obs = False
    report_cfg.training.greedy_baseline_mode = "force_admit_minloss_greedy"
    report_cfg.training.selection_baseline_mode = "force_admit_minloss_greedy"
    checkpoint_cache_tag = str(Path(checkpoint_path).resolve())
    scalar_keys = [
        'embb_rate', 'embb_user_rate', 'embb_service_ratio', 'embb_positive_rate_ratio', 'embb_min_rate_satisfaction_ratio', 'embb_min_rate_satisfied_user_count', 'urllc_admission',
        'admitted_urllc_reliability', 'urllc_reliability', 'effective_urllc_success_over_arrivals', 'empty_admission_case',
        'active_packets', 'scheduled_packets', 'total_power', 'embb_power', 'urllc_power',
        'overlay_ratio', 'puncture_ratio', 'overlay_selection_ratio', 'puncture_selection_ratio', 'embb_only_fraction', 'avg_puncture_loss',
        'avg_overlay_retention', 'overlay_candidate_pairs', 'overlay_feasible_pairs',
        'overlay_selected_pairs', 'admission_via_overlay_ratio', 'admission_via_puncture_ratio', 'jain_fairness', 'cell_edge_served_ratio', 'cell_edge_min_rate_satisfaction_ratio',
        'per_uav_total_load_std', 'per_uav_urllc_sched_std', 'per_uav_throughput_std',
        'shield_correction_ratio', 'collision_rewrite_ratio', 'fallback_ratio',
        'mode_correction_ratio', 'packet_invalid_ratio', 'mask_invalid_ratio',
        'joint_reliability_rewrite_ratio', 'greedy_noop_selected_ratio', 'greedy_admit_selected_ratio',
        'greedy_overlay_ratio', 'greedy_puncture_ratio', 'greedy_avg_embb_retention',
        'greedy_avg_embb_loss', 'greedy_avg_selected_throughput',
        'greedy_avg_rejected_urllc_when_noop_better', 'greedy_noop_available_ratio',
        'greedy_noop_better_ratio', 'greedy_requires_feasible_admission_only',
        'urllc_slot_duration_s', 'urllc_packet_bits_mean',
        'urllc_throughput_bps_slot_est', 'urllc_throughput_mbps_slot_est', 'urllc_throughput_bps_est',
        'mean_intercell_interference_power', 'mean_intercell_interference_mw', 'mean_intercell_interference_dbm',
        'intercell_interference_nonzero_ratio', 'overlay_intercell_interference_mw', 'puncture_intercell_interference_mw',
    ]
    vector_keys = [
        'per_uav_associated_embb', 'per_uav_associated_urllc', 'per_uav_scheduled_embb',
        'per_uav_scheduled_urllc', 'per_uav_overlay_count', 'per_uav_puncture_count',
        'per_uav_embb_throughput',
    ]
    metrics = {'loads': [], 'lambda': []}
    metrics.update({key: [] for key in scalar_keys + vector_keys})
    metrics['episode_scene_audit'] = {}
    representative = {}
    for load_idx, load in enumerate(loads):
        sys_cfg, urllc_cfg, embb_cfg, algo_cfg, sim_cfg = _configure_density_scenario(
            load, base_sys, base_urllc, base_embb, base_algo, base_sim
        )
        if hasattr(base_sim, 'urllc_user_ratio'):
            sim_cfg.urllc_user_ratio = base_sim.urllc_user_ratio
        env = SRMAPPOPhaseAEnv(sys_cfg, urllc_cfg, embb_cfg, algo_cfg, sim_cfg, report_cfg)
        seed_base = _report_seed_base(load_idx, report_cfg)
        episodes, representative[load] = _run_episode_batch_with_representative(
            episodes_per_load,
            lambda ep, collect_trace: run_env_episode(
                env,
                model=None,
                cfg=report_cfg,
                seed=seed_base + ep,
                collect_trace=collect_trace,
                use_greedy=True,
                greedy_policy="force_admit_minloss",
                cache_tag=checkpoint_cache_tag,
            ),
        )
        metrics['loads'].append(float(load))
        metrics['lambda'].append(_load_to_lambda(load))
        for key in scalar_keys:
            metrics[key].append(_episode_scalar_aggregate(episodes, key, default=0.0))
        for key in vector_keys:
            metrics[key].append(np.mean(np.stack([episode[key] for episode in episodes]), axis=0))
    _report_timing_log(
        f"run_force_admit_minloss_greedy_sweep loads={len(loads)} episodes_per_load={episodes_per_load} sec={perf_counter() - sweep_start:.3f}"
    )
    return metrics, representative


def _load_frozen_greedy_for_report(cfg: SRMAPPOConfig, loads: List[float], episodes_per_load: int) -> Tuple[Dict, Dict, Dict]:
    payload = _load_frozen_greedy_payload(cfg)
    payload_loads = [float(load) for load in payload.get('loads', [])]
    if payload_loads != [float(load) for load in loads]:
        raise ValueError(f"Frozen greedy loads mismatch: expected {loads}, got {payload_loads}")
    if int(payload.get('episodes_per_load', episodes_per_load)) != int(episodes_per_load):
        raise ValueError(
            f"Frozen greedy episodes_per_load mismatch: expected {episodes_per_load}, "
            f"got {payload.get('episodes_per_load')}"
        )
    metrics = payload.get('greedy_metrics', {})
    representative = _normalize_frozen_representative(payload)
    return metrics, representative, payload

def run_myopic_throughput_greedy_sweep(
    loads: List[float],
    episodes_per_load: int,
    checkpoint_path: Path,
    base_cfg: Optional[SRMAPPOConfig] = None,
) -> Tuple[Dict, Dict]:
    sweep_start = perf_counter()
    base_sys, base_urllc, base_embb, base_algo, base_sim = _build_main_like_configs()
    report_cfg = deepcopy(base_cfg or _load_checkpoint_cfg(checkpoint_path))
    report_cfg.env.include_greedy_reference_in_obs = False
    report_cfg.training.greedy_baseline_mode = "myopic_throughput_greedy"
    report_cfg.training.selection_baseline_mode = "myopic_throughput_greedy"
    checkpoint_cache_tag = str(Path(checkpoint_path).resolve())
    scalar_keys = [
        'embb_rate', 'embb_user_rate', 'embb_service_ratio', 'embb_positive_rate_ratio', 'embb_min_rate_satisfaction_ratio', 'embb_min_rate_satisfied_user_count', 'urllc_admission',
        'admitted_urllc_reliability', 'urllc_reliability', 'effective_urllc_success_over_arrivals', 'empty_admission_case',
        'active_packets', 'scheduled_packets', 'total_power', 'embb_power', 'urllc_power',
        'overlay_ratio', 'puncture_ratio', 'overlay_selection_ratio', 'puncture_selection_ratio', 'embb_only_fraction', 'avg_puncture_loss',
        'avg_overlay_retention', 'overlay_candidate_pairs', 'overlay_feasible_pairs',
        'overlay_selected_pairs', 'admission_via_overlay_ratio', 'admission_via_puncture_ratio', 'jain_fairness', 'cell_edge_served_ratio', 'cell_edge_min_rate_satisfaction_ratio',
        'per_uav_total_load_std', 'per_uav_urllc_sched_std', 'per_uav_throughput_std',
        'shield_correction_ratio', 'collision_rewrite_ratio', 'fallback_ratio',
        'mode_correction_ratio', 'packet_invalid_ratio', 'mask_invalid_ratio',
        'joint_reliability_rewrite_ratio', 'greedy_noop_selected_ratio', 'greedy_admit_selected_ratio',
        'greedy_overlay_ratio', 'greedy_puncture_ratio', 'greedy_avg_embb_retention',
        'greedy_avg_embb_loss', 'greedy_avg_selected_throughput',
        'greedy_avg_rejected_urllc_when_noop_better', 'greedy_noop_available_ratio',
        'greedy_noop_better_ratio', 'greedy_requires_feasible_admission_only',
    ]
    vector_keys = [
        'per_uav_associated_embb', 'per_uav_associated_urllc', 'per_uav_scheduled_embb',
        'per_uav_scheduled_urllc', 'per_uav_overlay_count', 'per_uav_puncture_count',
        'per_uav_embb_throughput',
    ]
    metrics = {'loads': [], 'lambda': []}
    metrics.update({key: [] for key in scalar_keys + vector_keys})
    representative = {}
    for load_idx, load in enumerate(loads):
        sys_cfg, urllc_cfg, embb_cfg, algo_cfg, sim_cfg = _configure_density_scenario(
            load, base_sys, base_urllc, base_embb, base_algo, base_sim
        )
        if hasattr(base_sim, 'urllc_user_ratio'):
            sim_cfg.urllc_user_ratio = base_sim.urllc_user_ratio
        env = SRMAPPOPhaseAEnv(sys_cfg, urllc_cfg, embb_cfg, algo_cfg, sim_cfg, report_cfg)
        seed_base = _report_seed_base(load_idx, report_cfg)
        episodes, representative[load] = _run_episode_batch_with_representative(
            episodes_per_load,
            lambda ep, collect_trace: run_env_episode(
                env,
                model=None,
                cfg=report_cfg,
                seed=seed_base + ep,
                collect_trace=collect_trace,
                use_greedy=True,
                greedy_policy="myopic_throughput",
                cache_tag=checkpoint_cache_tag,
            ),
        )
        metrics['loads'].append(float(load))
        metrics['lambda'].append(_load_to_lambda(load))
        for key in scalar_keys:
            metrics[key].append(_episode_scalar_aggregate(episodes, key, default=0.0))
        for key in vector_keys:
            metrics[key].append(np.mean(np.stack([episode[key] for episode in episodes]), axis=0))
    _report_timing_log(
        f"run_myopic_throughput_greedy_sweep loads={len(loads)} episodes_per_load={episodes_per_load} sec={perf_counter() - sweep_start:.3f}"
    )
    return metrics, representative


def run_hard_feasible_throughput_greedy_sweep(
    loads: List[float],
    episodes_per_load: int,
    checkpoint_path: Path,
    base_cfg: Optional[SRMAPPOConfig] = None,
    verbose_per_episode: bool = True,
) -> Tuple[Dict, Dict]:
    sweep_start = perf_counter()
    greedy_policy_override_raw = str(
        os.environ.get("SR_MAPPO_REPORT_GREEDY_POLICY_OVERRIDE", "") or ""
    ).strip().lower()
    cfg_mode_normalized = _normalize_baseline_mode(
        getattr((base_cfg or _load_checkpoint_cfg(checkpoint_path)).training, "greedy_baseline_mode", "hard_feasible_throughput_greedy")
    )
    if greedy_policy_override_raw:
        greedy_policy_override = greedy_policy_override_raw
    elif cfg_mode_normalized == "global_frontier_greedy":
        greedy_policy_override = "global_frontier"
    elif cfg_mode_normalized == "throughput_only_greedy":
        greedy_policy_override = "throughput_only"
    elif cfg_mode_normalized == "channel_only_greedy":
        greedy_policy_override = "channel_only"
    elif cfg_mode_normalized == "myopic_throughput_greedy":
        greedy_policy_override = "myopic_throughput"
    else:
        greedy_policy_override = "hard_feasible_throughput"
    virtual_slots_per_episode = max(
        1,
        int(os.environ.get("SR_MAPPO_REPORT_VIRTUAL_SLOTS_PER_EPISODE", "1") or "1"),
    )
    if virtual_slots_per_episode > 1:
        _report_log(
            f"[GREEDY] virtual multi-slot enabled: slots_per_episode={virtual_slots_per_episode}"
        )
    base_sys, base_urllc, base_embb, base_algo, base_sim = _build_main_like_configs()
    report_cfg = deepcopy(base_cfg or _load_checkpoint_cfg(checkpoint_path))
    # Optional report-time hard-feasible greedy gate overrides.
    # This affects only the greedy report sweep path and keeps training/runtime env untouched.
    gain_ratio_override = float(getattr(report_cfg.env, "greedy_hf_min_noma_gain_ratio_override", -1.0) or -1.0)
    gain_ratio_override_env = os.environ.get("SR_MAPPO_REPORT_GREEDY_HF_MIN_NOMA_GAIN_RATIO_OVERRIDE", "").strip()
    if gain_ratio_override_env:
        try:
            gain_ratio_override = float(gain_ratio_override_env)
            _report_log(
                "[OVERRIDE] greedy_hf_min_noma_gain_ratio_override="
                f"{gain_ratio_override:.6f} "
                f"(from SR_MAPPO_REPORT_GREEDY_HF_MIN_NOMA_GAIN_RATIO_OVERRIDE={gain_ratio_override_env})"
            )
        except ValueError:
            _report_log(
                "[OVERRIDE] ignore invalid SR_MAPPO_REPORT_GREEDY_HF_MIN_NOMA_GAIN_RATIO_OVERRIDE="
                f"{gain_ratio_override_env}"
            )
    sic_db_override = float(getattr(report_cfg.env, "greedy_hf_embb_min_sic_snir_db_override", -100.0) or -100.0)
    if gain_ratio_override > 0.0:
        base_algo.min_noma_gain_ratio = float(gain_ratio_override)
        _report_log(f"[GREEDY] override min_noma_gain_ratio={base_algo.min_noma_gain_ratio:.3f}")
    if sic_db_override > -100.0:
        base_algo.embb_min_sic_snir_db = float(sic_db_override)
        _report_log(f"[GREEDY] override embb_min_sic_snir_db={base_algo.embb_min_sic_snir_db:.3f} dB")
    minrate_scale_override = float(getattr(report_cfg.env, "report_embb_min_rate_scale", 1.0) or 1.0)
    if abs(minrate_scale_override - 1.0) > 1.0e-12:
        old_min = float(getattr(base_embb, "min_rate_per_user_bps", getattr(base_embb, "min_rate", 0.0)) or 0.0)
        new_min = float(old_min * minrate_scale_override)
        if hasattr(base_embb, "min_rate_per_user_bps"):
            base_embb.min_rate_per_user_bps = float(new_min)
        if hasattr(base_embb, "min_rate"):
            base_embb.min_rate = float(new_min)
        _report_log(
            f"[GREEDY] override eMBB min-rate scale={minrate_scale_override:.4f} "
            f"({old_min:.3e} -> {new_min:.3e} bps)"
        )
    relax_mode_env = str(os.environ.get("SR_MAPPO_GREEDY_HF_RELAX_MODE_FEASIBLE", "")).strip().lower()
    if relax_mode_env:
        report_cfg.env.greedy_hf_relax_mode_feasible = bool(relax_mode_env in {"1", "true", "yes", "on"})
        _report_log(f"[GREEDY] override greedy_hf_relax_mode_feasible={bool(report_cfg.env.greedy_hf_relax_mode_feasible)}")
    soft_score_env = str(os.environ.get("SR_MAPPO_GREEDY_HF_SOFT_FEASIBLE_SCORING", "")).strip().lower()
    if soft_score_env:
        report_cfg.env.greedy_hf_soft_feasible_scoring = bool(soft_score_env in {"1", "true", "yes", "on"})
        _report_log(f"[GREEDY] override greedy_hf_soft_feasible_scoring={bool(report_cfg.env.greedy_hf_soft_feasible_scoring)}")
    topk_tail_env = str(os.environ.get("SR_MAPPO_GREEDY_HF_TOPK_REPAIR_TAIL", "")).strip()
    if topk_tail_env:
        try:
            report_cfg.env.greedy_hf_topk_repair_tail = max(0, int(topk_tail_env))
            _report_log(f"[GREEDY] override greedy_hf_topk_repair_tail={int(report_cfg.env.greedy_hf_topk_repair_tail)}")
        except Exception:
            _report_log(f"[GREEDY] ignore invalid SR_MAPPO_GREEDY_HF_TOPK_REPAIR_TAIL={topk_tail_env!r}")
    mode_penalty_env = str(os.environ.get("SR_MAPPO_GREEDY_HF_MODE_VIOLATION_PENALTY", "")).strip()
    if mode_penalty_env:
        try:
            report_cfg.env.greedy_hf_soft_penalty_mode_infeasible = float(mode_penalty_env)
            _report_log(
                f"[GREEDY] override greedy_hf_soft_penalty_mode_infeasible="
                f"{float(report_cfg.env.greedy_hf_soft_penalty_mode_infeasible):.3f}"
            )
        except Exception:
            _report_log(f"[GREEDY] ignore invalid SR_MAPPO_GREEDY_HF_MODE_VIOLATION_PENALTY={mode_penalty_env!r}")
    reliability_penalty_env = str(os.environ.get("SR_MAPPO_GREEDY_HF_RELIABILITY_VIOLATION_PENALTY", "")).strip()
    if reliability_penalty_env:
        try:
            report_cfg.env.greedy_hf_soft_penalty_reliability = float(reliability_penalty_env)
            _report_log(
                f"[GREEDY] override greedy_hf_soft_penalty_reliability="
                f"{float(report_cfg.env.greedy_hf_soft_penalty_reliability):.3f}"
            )
        except Exception:
            _report_log(
                f"[GREEDY] ignore invalid SR_MAPPO_GREEDY_HF_RELIABILITY_VIOLATION_PENALTY="
                f"{reliability_penalty_env!r}"
            )
    forced_urllc_ratio = _resolve_forced_urllc_ratio(report_cfg)
    exp_line = str(getattr(report_cfg.training, "experiment_line", "") or "").strip().lower()
    if hasattr(base_sim, "urllc_user_ratio") and forced_urllc_ratio >= 0.0:
        base_sim.urllc_user_ratio = float(np.clip(forced_urllc_ratio, 0.0, 1.0))
        # Strict pure-eMBB guard: disable URLLC arrivals entirely.
        if base_sim.urllc_user_ratio <= 0.0:
            if hasattr(base_sim, "urllc_poisson_rate"):
                base_sim.urllc_poisson_rate = 0.0
            if hasattr(base_sim, "fixed_urllc_poisson_rate"):
                base_sim.fixed_urllc_poisson_rate = True
        _report_log(
            f"[GREEDY] forcing urllc_user_ratio={base_sim.urllc_user_ratio:.3f} "
            f"(override={float(getattr(report_cfg.env, 'urllc_user_ratio_override', -1.0)):.3f}, experiment='{exp_line}')"
        )
    fixed_embb_users = int(_env_int_override("SR_MAPPO_REPORT_FIXED_EMBB_USERS", 0))
    if fixed_embb_users > 0:
        try:
            setattr(base_sim, "fixed_embb_user_count", int(fixed_embb_users))
            _report_log(f"[GREEDY] fixed eMBB user count override enabled: fixed_embb_user_count={int(fixed_embb_users)}")
        except Exception:
            _report_log(f"[GREEDY] failed to apply fixed eMBB user count override: {fixed_embb_users}")
    _owner_policy = str(os.environ.get("SR_MAPPO_OWNER_POLICY", "legacy") or "legacy").strip().lower()
    _owner_pool_cap = str(os.environ.get("SR_MAPPO_OWNER_TOPK_USER_POOL", "0") or "0").strip()
    _owner_lowload_enable = str(os.environ.get("SR_MAPPO_OWNER_TOPK_LOWLOAD_ENABLE", "1") or "1").strip()
    _owner_lowload_max_load = str(os.environ.get("SR_MAPPO_OWNER_TOPK_LOWLOAD_MAX_LOAD", "12") or "12").strip()
    _owner_lowload_max_rb_per_round = str(
        os.environ.get("SR_MAPPO_OWNER_TOPK_LOWLOAD_MAX_RB_PER_USER_PER_ROUND", "0") or "0"
    ).strip()
    _report_log(f"[GREEDY] owner policy: {_owner_policy}")
    _report_log(f"[GREEDY] owner topk user pool cap: {_owner_pool_cap}")
    _report_log(
        "[GREEDY] owner low-load mode: "
        f"enable={_owner_lowload_enable} "
        f"max_load={_owner_lowload_max_load} "
        f"max_rb_per_user_per_round={_owner_lowload_max_rb_per_round}"
    )
    report_cfg.env.include_greedy_reference_in_obs = False
    baseline_mode_name = (
        "global_frontier_greedy"
        if greedy_policy_override in {"global_frontier", "global_greedy"}
        else "hard_feasible_throughput_greedy"
    )
    report_cfg.training.greedy_baseline_mode = baseline_mode_name
    report_cfg.training.selection_baseline_mode = baseline_mode_name
    checkpoint_cache_tag = str(Path(checkpoint_path).resolve())
    nested_load_enabled = str(os.environ.get("SR_MAPPO_REPORT_NESTED_LOAD_SCENARIO", "1")).strip().lower() not in {"0", "false", "no", "off"}
    max_total_users = 0
    max_embb_users = 0
    max_urllc_users = 0
    if nested_load_enabled and loads:
        _mx_sys, _mx_ur, _mx_em, _mx_algo, _mx_sim = _configure_density_scenario(
            max(float(x) for x in loads), base_sys, base_urllc, base_embb, base_algo, base_sim
        )
        max_embb_users = int(_mx_sys.num_embb_users)
        max_urllc_users = int(_mx_sys.num_urllc_users)
        max_total_users = int(max_embb_users + max_urllc_users)
        # Optional canonical nested-pool override for tri-mix comparability:
        # force all mixes to share exactly the same mother-pool boundaries.
        # Useful when we want "only mix ratio/arrival changes" while keeping pool mapping stable.
        fixed_pool_embb = int(_env_int_override("SR_MAPPO_REPORT_NESTED_FIXED_POOL_EMBB_USERS", 0))
        fixed_pool_urllc = int(_env_int_override("SR_MAPPO_REPORT_NESTED_FIXED_POOL_URLLC_USERS", 0))
        fixed_pool_total = int(_env_int_override("SR_MAPPO_REPORT_NESTED_FIXED_POOL_TOTAL_USERS", 0))
        if fixed_pool_embb > 0:
            max_embb_users = int(fixed_pool_embb)
        if fixed_pool_urllc > 0:
            max_urllc_users = int(fixed_pool_urllc)
        if fixed_pool_total > 0:
            max_total_users = int(fixed_pool_total)
        else:
            max_total_users = int(max(max_total_users, max_embb_users + max_urllc_users))
        if (fixed_pool_embb > 0) or (fixed_pool_urllc > 0) or (fixed_pool_total > 0):
            _report_log(
                f"[GREEDY] nested fixed-pool override enabled: "
                f"total/embb/urllc={max_total_users}/{max_embb_users}/{max_urllc_users} "
                f"(env: TOTAL={fixed_pool_total}, EMBB={fixed_pool_embb}, URLLC={fixed_pool_urllc})"
            )
        _report_log(
            f"[GREEDY] nested-load enabled: max_users(total/embb/urllc)="
            f"{max_total_users}/{max_embb_users}/{max_urllc_users}"
        )
    scalar_keys = [
        'embb_rate', 'embb_rate_pre_urllc_admission', 'embb_user_rate', 'embb_service_ratio', 'embb_positive_rate_ratio', 'embb_min_rate_satisfaction_ratio', 'embb_min_rate_satisfied_user_count', 'urllc_admission',
        'admitted_urllc_reliability', 'urllc_reliability', 'effective_urllc_success_over_arrivals', 'empty_admission_case',
        'active_packets', 'scheduled_packets', 'total_power', 'embb_power', 'urllc_power',
        'overlay_ratio', 'puncture_ratio', 'overlay_selection_ratio', 'puncture_selection_ratio', 'embb_only_fraction', 'avg_puncture_loss',
        'avg_overlay_retention', 'overlay_candidate_pairs', 'overlay_feasible_pairs',
        'overlay_selected_pairs', 'admission_via_overlay_ratio', 'admission_via_puncture_ratio', 'jain_fairness', 'cell_edge_served_ratio', 'cell_edge_min_rate_satisfaction_ratio',
        'per_uav_total_load_std', 'per_uav_urllc_sched_std', 'per_uav_throughput_std',
        'shield_correction_ratio', 'collision_rewrite_ratio', 'fallback_ratio',
        'mode_correction_ratio', 'packet_invalid_ratio', 'mask_invalid_ratio',
        'joint_reliability_rewrite_ratio', 'greedy_noop_selected_ratio', 'greedy_admit_selected_ratio',
        'greedy_overlay_ratio', 'greedy_puncture_ratio', 'greedy_avg_embb_retention',
        'greedy_avg_embb_loss', 'greedy_avg_selected_throughput',
        'greedy_selected_embb_throughput', 'greedy_feasible_admit_count', 'greedy_no_feasible_admit_ratio',
        'greedy_keep_only_when_no_feasible_admit_ratio', 'greedy_selected_urllc_reliability', 'greedy_selected_embb_min_rate_ok',
        'greedy_avg_rejected_urllc_when_noop_better', 'greedy_noop_available_ratio',
        'greedy_noop_better_ratio', 'greedy_requires_feasible_admission_only',
        'greedy_urllc_budget_used_ratio', 'greedy_urllc_budget_utilization_ratio',
        'greedy_embb_loss_share_cap_ratio',
        'greedy_hf_reject_reliability_ratio', 'greedy_hf_reject_power_ratio',
        'greedy_hf_reject_min_rate_ratio', 'greedy_hf_reject_share_cap_ratio',
        'greedy_hf_feasible_ratio',
        'greedy_hf_reject_reliability_per_decision', 'greedy_hf_reject_power_per_decision',
        'greedy_hf_reject_min_rate_per_decision', 'greedy_hf_reject_share_cap_per_decision',
        'greedy_hf_candidate_evaluated_per_decision', 'greedy_hf_candidate_feasible_per_decision',
        'greedy_hf_rescan_used', 'greedy_hf_prefilter_truncated',
        'greedy_hf_enforce_min_rate_hard_gate',
        'greedy_hf_pre_admission_all_embb_min_rate_required',
        'greedy_hf_pre_admission_all_embb_min_rate_met',
        'greedy_hf_no_candidate_ratio', 'greedy_hf_all_rejected_ratio', 'greedy_hf_budget_exhausted_keep_ratio',
        'greedy_hf_no_candidate_per_decision', 'greedy_hf_all_rejected_per_decision', 'greedy_hf_budget_exhausted_keep_per_decision',
        'greedy_hf_prefilter_pair_per_decision', 'greedy_hf_prefilter_block_mode_mask_per_decision',
        'greedy_hf_prefilter_block_packet_mask_per_decision', 'greedy_hf_prefilter_block_mode_infeasible_per_decision',
        'greedy_hf_prefilter_block_mode_mask_ratio', 'greedy_hf_prefilter_block_packet_mask_ratio',
        'greedy_hf_prefilter_block_mode_infeasible_ratio',
        'greedy_hf_relaxed_candidate_ratio', 'greedy_hf_selected_relaxed_ratio',
        'greedy_hf_final_gate_reject_ratio', 'greedy_hf_final_gate_keep_ratio',
        'greedy_hf_final_gate_reject_mode_per_decision',
        'greedy_hf_final_gate_reject_reliability_per_decision',
        'greedy_hf_final_gate_reject_power_per_decision',
        'greedy_hf_final_gate_reject_min_rate_per_decision',
        'greedy_hf_final_gate_reject_mode_given_final_gate_ratio',
        'greedy_hf_final_gate_reject_reliability_given_final_gate_ratio',
        'greedy_hf_final_gate_reject_power_given_final_gate_ratio',
        'greedy_hf_final_gate_reject_min_rate_given_final_gate_ratio',
        'greedy_hf_reject_by_mode_overlay_rel_fail',
        'greedy_hf_reject_by_mode_overlay_sic_fail',
        'greedy_hf_reject_by_mode_gain_ratio_fail',
        'greedy_hf_reject_by_mode_owner_pool_missing',
        'greedy_hf_reject_by_mode_owner_unresolved_due_to_mode_fail',
        'greedy_hf_reject_by_mode_overlay_owner_missing',
        'greedy_hf_reject_by_mode_puncture_rel_fail',
        'greedy_hf_reject_by_mode_puncture_sic_fail',
        'greedy_hf_reject_by_mode_puncture_owner_missing',
        'greedy_hf_reject_by_owner_missing',
        'greedy_hf_reject_by_owner_mismatch',
        'greedy_hf_gate_overlay_rel_fail_margin_mean',
        'greedy_hf_gate_overlay_sic_fail_margin_db_mean',
        'greedy_hf_gate_puncture_rel_fail_margin_mean',
        'greedy_hf_gate_target_reliability',
        'greedy_hf_gate_target_sic_snir_db',
        'greedy_hf_gate_overlay_rel_fail_snir_db_mean',
        'greedy_hf_gate_overlay_sic_fail_post_sic_db_mean',
        'greedy_hf_gate_puncture_rel_fail_snir_db_mean',
        'pre_mode_overlay_rel_fail_per_decision',
        'pre_mode_overlay_sic_fail_per_decision',
        'pre_mode_gain_ratio_fail_per_decision',
        'pre_mode_puncture_rel_fail_per_decision',
        'pre_mode_puncture_sic_fail_per_decision',
        'greedy_hf_mode_violation_penalty_avg',
        'greedy_hf_no_candidate_block_mode_mask_per_no_candidate',
        'greedy_hf_no_candidate_block_packet_mask_per_no_candidate',
        'greedy_hf_no_candidate_block_mode_infeasible_per_no_candidate',
        'greedy_hf_no_candidate_empty_observation_per_decision',
        'greedy_hf_no_candidate_mask_block_per_decision',
        'greedy_hf_no_candidate_empty_observation_given_no_candidate_ratio',
        'greedy_hf_no_candidate_mask_block_given_no_candidate_ratio',
        'greedy_hf_no_candidate_given_no_feasible_ratio', 'greedy_hf_all_rejected_given_no_feasible_ratio',
        'greedy_hf_budget_exhausted_given_no_feasible_ratio',
        'phase_a_rejected_intercell_per_decision', 'phase_a_rejected_min_rate_per_decision',
        'phase_a_rejected_power_guard_per_decision', 'phase_a_rejected_collision_per_decision',
        'phase_a_rejected_deadline_per_decision', 'phase_a_rejected_other_per_decision',
        'phase_a_rejected_other_gain_ratio_per_decision', 'phase_a_rejected_other_overlay_margin_per_decision',
        'phase_a_rejected_other_overlay_positive_gate_per_decision',
        'phase_a_rejected_other_no_overlay_owner_per_decision',
        'phase_a_rejected_other_overlay_reliability_per_decision',
        'phase_a_rejected_other_overlay_sic_per_decision',
        'phase_a_rejected_other_gain_ratio_given_other_ratio', 'phase_a_rejected_other_overlay_margin_given_other_ratio',
        'phase_a_rejected_other_overlay_positive_gate_given_other_ratio',
        'phase_a_rejected_other_no_overlay_owner_given_other_ratio',
        'phase_a_rejected_other_overlay_reliability_given_other_ratio',
        'phase_a_rejected_other_overlay_sic_given_other_ratio',
        'urllc_slot_duration_s', 'urllc_packet_bits_mean',
        'urllc_throughput_bps_slot_est', 'urllc_throughput_mbps_slot_est',
        'phase0_baseline_cache_hit_ratio', 'phase0_baseline_cache_hit_count', 'phase0_baseline_cache_total_count',
        'phase0_baseline_minrate_exit_reason',
        'phase0_stage1_handoff_remaining_cells', 'phase0_stage1_handoff_served_users',
        'phase0_stage1_handoff_unmet_users', 'phase0_stage1_handoff_total_rate',
        'phase0_stage1_handoff_service_ratio', 'phase0_stage1_handoff_minrate_ratio',
        'phase0_stage1_handoff_mean_rb_per_served', 'phase0_stage1_handoff_max_rb_single_user',
        'phase0_stage1_handoff_top1_rb_share', 'phase0_stage1_handoff_top2_rb_share',
        'phase0_stage1_served_cap_effective', 'phase0_stage1_served_cap_block_count',
        'phase0_stage1_touch_repeat_block_count', 'phase0_stage1_allow_stage2_after_served_cap',
        'phase0_stage1_transition_progress',
        'phase0_stage2_assigned_cells', 'phase0_stage2_existing_user_assignments',
        'phase0_stage2_new_user_assignments', 'phase0_stage2_existing_user_gain_sum',
        'phase0_stage2_new_user_gain_sum', 'phase0_stage2_existing_user_assignment_ratio',
        'phase0_stage2_new_user_assignment_ratio',
        'virtual_slot_reset_count_per_episode',
        'mean_intercell_interference_power', 'mean_intercell_interference_mw', 'mean_intercell_interference_dbm', 'intercell_interference_nonzero_ratio',
        'overlay_intercell_interference_mw', 'puncture_intercell_interference_mw',
        'embb_served_user_count', 'embb_min_rate_satisfied_user_count', 'embb_min_rate_satisfied_user_count_after_puncture_deduction',
        'embb_rate_with_intercell', 'embb_rate_without_intercell_est', 'embb_rate_loss_due_to_intercell', 'embb_rate_loss_due_to_intercell_ratio',
        'overlay_rate_with_intercell', 'overlay_rate_without_intercell_est', 'overlay_rate_loss_due_to_intercell',
        'puncture_rate_with_intercell', 'puncture_rate_without_intercell_est', 'puncture_rate_loss_due_to_intercell',
        'terminal_embb_service_floor_penalty', 'terminal_embb_min_rate_floor_penalty',
        'terminal_embb_service_bonus', 'terminal_embb_min_rate_bonus', 'terminal_avg_served_embb_rate_bonus',
        'urllc_admission_over_service_tradeoff_penalty',
        'sic_prior_pair_block_ratio', 'sic_prior_saved_mode_fail_ratio', 'sic_prior_owner_total_per_decision',
        'greedy_hf_overlay_blacklist_cell_ratio', 'greedy_hf_overlay_blacklist_candidate_block_ratio', 'greedy_hf_overlay_blacklist_saved_mode_fail_ratio',
        'greedy_hf_overlay_only_rb_reservation_enabled', 'greedy_hf_overlay_only_rb_reservation_ratio',
        'greedy_hf_overlay_only_rb_reserved_count', 'greedy_hf_overlay_only_rb_reserved_cell_ratio',
        'greedy_hf_overlay_only_rb_mode_block_ratio', 'greedy_hf_overlay_only_rb_hard_block_enabled',
        'greedy_hf_overlay_only_rb_soft_preference_enabled', 'greedy_hf_overlay_only_rb_overlay_bonus_bps',
        'greedy_hf_overlay_only_rb_puncture_penalty_bps', 'greedy_hf_overlay_only_rb_soft_preference_applied_ratio',
        'guardrail_enabled', 'guardrail_resample_count', 'guardrail_pass_ratio',
        'guardrail_reject_reason_overlay', 'guardrail_reject_reason_embb_minrate', 'guardrail_reject_reason_uav_imbalance',
        'guardrail_actual_overlay_feasible_ratio', 'guardrail_actual_embb_minrate_ratio', 'guardrail_actual_uav_load_imbalance',
        'guardrail_threshold_overlay', 'guardrail_threshold_embb_minrate', 'guardrail_threshold_uav_imbalance',
        'candidate_pair_count', 'feasible_pair_count', 'feasible_pair_ratio',
        'requested_mix_ratio', 'realized_resource_ratio', 'realized_power_ratio', 'realized_served_users_ratio',
        'feasible_graph_freeze_enabled',
    ]
    vector_keys = [
        'per_uav_associated_embb', 'per_uav_associated_urllc', 'per_uav_scheduled_embb',
        'per_uav_scheduled_urllc', 'per_uav_overlay_count', 'per_uav_puncture_count',
        'per_uav_embb_throughput',
    ]
    if bool(getattr(report_cfg.env, "report_export_embb_user_rates", False)):
        vector_keys.extend([
            'embb_user_rates',
            'embb_user_rates_after_puncture_deduction',
            'embb_user_rb_count',
            'embb_user_rate_per_assigned_rb_est',
            'embb_user_single_rb_true_rate',
            'embb_user_associated_uav',
        ])
    metrics = {'loads': [], 'lambda': []}
    metrics.update({key: [] for key in scalar_keys + vector_keys})
    metrics['mother_topology_id'] = []
    metrics['mother_topology_seed'] = []
    metrics['same_channel_hash'] = []
    metrics['same_assoc_hash'] = []
    metrics['same_user_pool_hash'] = []
    metrics['mix_user_subset_hash'] = []
    metrics['embb_subset_hash'] = []
    metrics['same_feasible_graph_hash'] = []
    metrics['feasible_graph_id'] = []
    metrics['overlay_graph_hash'] = []
    metrics['channel_matrix_hash'] = []
    metrics['pathloss_hash'] = []
    metrics['shadowing_hash'] = []
    metrics['sic_order_hash'] = []
    metrics['repair_sequence_hash'] = []
    metrics['guardrail_pass'] = []
    # Keep per-episode samples for downstream analysis (e.g., share-cap CDF).
    metrics['greedy_episode_arrivals_samples'] = []
    metrics['greedy_episode_admitted_samples'] = []
    metrics['greedy_episode_budget_used_ratio_samples'] = []
    metrics['episode_scene_audit'] = {}
    representative = {}
    shared_nested_ur_order_across_loads = None
    shared_nested_em_order_across_loads = None
    prev_nested_embb_subset_count: Optional[int] = None
    prev_nested_embb_served_count: Optional[int] = None
    prev_nested_embb_selected_ids: Optional[list[int]] = None
    prev_selected_ids: set[int] = set()
    prev_phase0_owner_map: Optional[np.ndarray] = None
    prev_phase0_rate_cap_bps: Optional[float] = None
    cross_mix_rate_cap_map_bps: Dict[float, float] = {}
    try:
        raw_cross_mix_caps = getattr(report_cfg.env, "phase0_cross_mix_rate_cap_map_bps", None)
        if isinstance(raw_cross_mix_caps, dict):
            for k, v in raw_cross_mix_caps.items():
                cross_mix_rate_cap_map_bps[float(k)] = float(v)
    except Exception:
        cross_mix_rate_cap_map_bps = {}
    per_load_poisson_rate_override: Dict[float, float] = {}
    try:
        raw_poisson_rate_map = os.environ.get("SR_MAPPO_REPORT_URLLC_POISSON_RATE_MAP_OVERRIDE", "").strip()
        if raw_poisson_rate_map:
            parsed_poisson_rate_map = json.loads(raw_poisson_rate_map)
            if isinstance(parsed_poisson_rate_map, dict):
                for k, v in parsed_poisson_rate_map.items():
                    per_load_poisson_rate_override[float(k)] = float(v)
    except Exception:
        per_load_poisson_rate_override = {}
    per_load_nested_embb_served_cap_override: Dict[float, int] = {}
    try:
        raw_served_cap_map = os.environ.get("SR_MAPPO_REPORT_NESTED_EMBB_SERVED_CAP_MAP", "").strip()
        if raw_served_cap_map:
            parsed_served_cap_map = json.loads(raw_served_cap_map)
            if isinstance(parsed_served_cap_map, dict):
                for k, v in parsed_served_cap_map.items():
                    per_load_nested_embb_served_cap_override[float(k)] = int(round(float(v)))
    except Exception:
        per_load_nested_embb_served_cap_override = {}
    continuity_prev_scheduled_mean: Optional[float] = None
    continuity_prev_active_mean: Optional[float] = None
    for load_idx, load in enumerate(loads):
        sys_cfg, urllc_cfg, embb_cfg, algo_cfg, sim_cfg = _configure_density_scenario(
            load, base_sys, base_urllc, base_embb, base_algo, base_sim
        )
        if nested_load_enabled and max_total_users > 0:
            setattr(sys_cfg, "nested_load_from_max_users_enabled", True)
            setattr(sys_cfg, "nested_load_max_total_users", int(max_total_users))
            setattr(sys_cfg, "nested_load_max_embb_users", int(max_embb_users))
            setattr(sys_cfg, "nested_load_max_urllc_users", int(max_urllc_users))
            # Keep per-cell composition controlled across load sweep:
            # association follows generated serving hints instead of re-associating
            # by large-scale argmax.
            setattr(sys_cfg, "force_serving_hints_association", True)
            # Hold the selected nested subset fixed across episodes by default so
            # the sweep compares load on the same mother-scene/user ordering.
            # Allow explicit override for experiments that need per-episode subset
            # variation under the same frozen mother scene.
            freeze_subset_across_episodes = _env_bool_override(
                "SR_MAPPO_REPORT_NESTED_FIXED_SUBSET_ACROSS_EPISODES",
                True,
            )
            setattr(
                report_cfg.env,
                "nested_fixed_user_subset_across_episodes",
                bool(freeze_subset_across_episodes),
            )
            # Optional stronger control: reuse the same canonical URLLC/eMBB
            # ordering across load buckets, then let each load take its own
            # class-wise prefix. This reduces cross-load subset jitter without
            # removing the intended load-dependent user-count scaling.
            freeze_subset_across_loads = _env_bool_override(
                "SR_MAPPO_REPORT_NESTED_FIXED_SUBSET_ACROSS_LOADS",
                bool(getattr(report_cfg.env, "nested_fixed_user_subset_across_loads", False)),
            )
            setattr(report_cfg.env, "nested_fixed_user_subset_across_loads", bool(freeze_subset_across_loads))
            if bool(freeze_subset_across_loads):
                if shared_nested_ur_order_across_loads is not None and shared_nested_em_order_across_loads is not None:
                    setattr(
                        report_cfg.env,
                        "nested_shared_ur_order_across_loads",
                        np.asarray(shared_nested_ur_order_across_loads, dtype=np.int32).copy(),
                    )
                    setattr(
                        report_cfg.env,
                        "nested_shared_em_order_across_loads",
                        np.asarray(shared_nested_em_order_across_loads, dtype=np.int32).copy(),
                    )
            else:
                if hasattr(report_cfg.env, "nested_shared_ur_order_across_loads"):
                    delattr(report_cfg.env, "nested_shared_ur_order_across_loads")
                if hasattr(report_cfg.env, "nested_shared_em_order_across_loads"):
                    delattr(report_cfg.env, "nested_shared_em_order_across_loads")
            # Also keep topology/channel fixed per load bucket by default; only arrivals vary.
            # Allow environment overrides for channel-seed sweep experiments.
            freeze_assoc = _env_bool_override("SR_MAPPO_REPORT_FORCE_FREEZE_ASSOC", True)
            freeze_channel = _env_bool_override("SR_MAPPO_REPORT_FORCE_FREEZE_CHANNEL", True)
            setattr(report_cfg.env, "freeze_association_across_episodes", bool(freeze_assoc))
            setattr(report_cfg.env, "freeze_channel_gains_across_episodes", bool(freeze_channel))
        if hasattr(base_sim, 'urllc_user_ratio'):
            sim_cfg.urllc_user_ratio = base_sim.urllc_user_ratio
        if per_load_poisson_rate_override:
            chosen_lambda = None
            for k, v in per_load_poisson_rate_override.items():
                if abs(float(k) - float(load)) <= 1.0e-9:
                    chosen_lambda = float(v)
                    break
            if chosen_lambda is not None and hasattr(sim_cfg, "urllc_poisson_rate"):
                sim_cfg.urllc_poisson_rate = float(chosen_lambda)
                if hasattr(sim_cfg, "fixed_urllc_poisson_rate"):
                    sim_cfg.fixed_urllc_poisson_rate = True
                if hasattr(report_cfg.env, "urllc_poisson_rate"):
                    report_cfg.env.urllc_poisson_rate = float(chosen_lambda)
        chosen_nested_embb_served_cap = None
        if per_load_nested_embb_served_cap_override:
            for k, v in per_load_nested_embb_served_cap_override.items():
                if abs(float(k) - float(load)) <= 1.0e-9:
                    chosen_nested_embb_served_cap = int(v)
                    break
        if chosen_nested_embb_served_cap is not None:
            setattr(
                report_cfg.env,
                "nested_embb_served_cap_for_load",
                int(chosen_nested_embb_served_cap),
            )
        elif hasattr(report_cfg.env, "nested_embb_served_cap_for_load"):
            delattr(report_cfg.env, "nested_embb_served_cap_for_load")
        try:
            nested_embb_max_new_per_load = int(
                str(os.environ.get("SR_MAPPO_NESTED_EMBB_MAX_NEW_PER_LOAD", "0") or "0").strip()
            )
        except Exception:
            nested_embb_max_new_per_load = 0
        if nested_embb_max_new_per_load > 0:
            setattr(report_cfg.env, "nested_embb_max_new_per_load", int(nested_embb_max_new_per_load))
            if prev_nested_embb_served_count is not None:
                setattr(
                    report_cfg.env,
                    "nested_prev_embb_served_count",
                    int(prev_nested_embb_served_count),
                )
            elif hasattr(report_cfg.env, "nested_prev_embb_served_count"):
                delattr(report_cfg.env, "nested_prev_embb_served_count")
            if prev_nested_embb_subset_count is not None:
                setattr(
                    report_cfg.env,
                    "nested_prev_embb_subset_count",
                    int(prev_nested_embb_subset_count),
                )
            elif hasattr(report_cfg.env, "nested_prev_embb_subset_count"):
                delattr(report_cfg.env, "nested_prev_embb_subset_count")
        else:
            if hasattr(report_cfg.env, "nested_embb_max_new_per_load"):
                delattr(report_cfg.env, "nested_embb_max_new_per_load")
            if hasattr(report_cfg.env, "nested_prev_embb_served_count"):
                delattr(report_cfg.env, "nested_prev_embb_served_count")
            if hasattr(report_cfg.env, "nested_prev_embb_subset_count"):
                delattr(report_cfg.env, "nested_prev_embb_subset_count")
        monotone_prerate_guard = bool(
            str(os.environ.get("SR_MAPPO_NESTED_EMBB_PRERATE_GUARD", "0") or "0").strip().lower()
            in {"1", "true", "yes", "on"}
        )
        if monotone_prerate_guard and prev_nested_embb_selected_ids is not None:
            setattr(
                report_cfg.env,
                "nested_prev_embb_selected_ids",
                [int(x) for x in list(prev_nested_embb_selected_ids)],
            )
        elif hasattr(report_cfg.env, "nested_prev_embb_selected_ids"):
            delattr(report_cfg.env, "nested_prev_embb_selected_ids")
        _report_log(
            f"[GREEDY][load={float(load):.1f}] effective_flags "
            f"| nested_load={bool(getattr(sys_cfg, 'nested_load_from_max_users_enabled', False))} "
            f"| nested_fixed_subset={bool(getattr(report_cfg.env, 'nested_fixed_user_subset_across_episodes', False))} "
            f"| force_serving_hints_assoc={bool(getattr(sys_cfg, 'force_serving_hints_association', False))} "
            f"| freeze_assoc={bool(getattr(report_cfg.env, 'freeze_association_across_episodes', False))} "
            f"| freeze_channel={bool(getattr(report_cfg.env, 'freeze_channel_gains_across_episodes', False))}"
        )
        pure_sumrate_phase0 = bool(_env_bool_override("SR_MAPPO_REPORT_PURE_SUMRATE_COLD_START", False))
        if (not pure_sumrate_phase0) and prev_phase0_owner_map is not None and prev_phase0_rate_cap_bps is not None:
            setattr(
                report_cfg.env,
                "phase0_monotone_prev_owner_per_uav_rb",
                np.asarray(prev_phase0_owner_map, dtype=np.int32).copy(),
            )
            setattr(report_cfg.env, "phase0_monotone_prev_rate_cap_bps", float(prev_phase0_rate_cap_bps))
        else:
            if hasattr(report_cfg.env, "phase0_monotone_prev_owner_per_uav_rb"):
                delattr(report_cfg.env, "phase0_monotone_prev_owner_per_uav_rb")
            if hasattr(report_cfg.env, "phase0_monotone_prev_rate_cap_bps"):
                delattr(report_cfg.env, "phase0_monotone_prev_rate_cap_bps")
        cross_mix_cap_bps = cross_mix_rate_cap_map_bps.get(float(load))
        if cross_mix_cap_bps is not None and float(cross_mix_cap_bps) > 0.0:
            setattr(report_cfg.env, "phase0_cross_mix_rate_cap_bps", float(cross_mix_cap_bps))
        else:
            if hasattr(report_cfg.env, "phase0_cross_mix_rate_cap_bps"):
                delattr(report_cfg.env, "phase0_cross_mix_rate_cap_bps")
        setattr(report_cfg.env, "phase0_monotone_rate_cap_tolerance_bps", float(1.0e3))
        env = SRMAPPOPhaseAEnv(sys_cfg, urllc_cfg, embb_cfg, algo_cfg, sim_cfg, report_cfg)
        if continuity_prev_scheduled_mean is not None and continuity_prev_active_mean is not None:
            setattr(env, "greedy_continuity_prev_scheduled_packets", float(continuity_prev_scheduled_mean))
            setattr(env, "greedy_continuity_prev_active_packets", float(continuity_prev_active_mean))
            _report_log(
                f"[GREEDY][load={float(load):.1f}] continuity_seed "
                f"| prev_scheduled_mean={float(continuity_prev_scheduled_mean):.3f} "
                f"| prev_active_mean={float(continuity_prev_active_mean):.3f}"
            )
        seed_base = _report_seed_base(load_idx, report_cfg)
        def _episode_runner(ep: int, collect_trace: bool) -> Dict:
            episode_seed = int(seed_base + ep)
            with _temporary_shared_mother_scene_for_episode(
                env=env,
                load=float(load),
                load_idx=int(load_idx),
                episode_seed=episode_seed,
            ):
                return _run_env_episode_virtual_slots(
                    env=env,
                    model=None,
                    cfg=report_cfg,
                    seed=episode_seed,
                    collect_trace=collect_trace,
                    use_greedy=True,
                    greedy_policy=greedy_policy_override,
                    cache_tag=checkpoint_cache_tag,
                    virtual_slots=virtual_slots_per_episode,
                )
        episodes, representative[load] = _run_episode_batch_with_representative(
            episodes_per_load,
            _episode_runner,
        )
        metrics['episode_scene_audit'][str(float(load))] = _build_episode_scene_audit(episodes)
        rep = representative.get(load, {}) or {}
        try:
            rep_snapshot_owner = rep.get("snapshot_owner_per_uav_rb", None)
            if rep_snapshot_owner is not None:
                prev_phase0_owner_map = np.asarray(rep_snapshot_owner, dtype=np.int32).copy()
            pre_rates = [
                float(
                    ep.get(
                        "embb_rate_pre_urllc_admission",
                        ep.get("embb_rate", 0.0),
                    ) or 0.0
                )
                for ep in episodes
            ]
            if pre_rates:
                prev_phase0_rate_cap_bps = float(max(np.mean(pre_rates), 0.0) * 1.0e6)
        except Exception:
            prev_phase0_owner_map = None
            prev_phase0_rate_cap_bps = None
        try:
            continuity_prev_scheduled_mean = float(
                np.mean([float(ep.get("scheduled_packets", 0.0) or 0.0) for ep in episodes])
            )
            continuity_prev_active_mean = float(
                np.mean([float(ep.get("active_packets", 0.0) or 0.0) for ep in episodes])
            )
        except Exception:
            continuity_prev_scheduled_mean = None
            continuity_prev_active_mean = None
        if (
            bool(getattr(report_cfg.env, "nested_fixed_user_subset_across_loads", False))
            and (shared_nested_ur_order_across_loads is None or shared_nested_em_order_across_loads is None)
        ):
            try:
                _shared_ur = getattr(env, "_nested_canonical_ur_order_cache", None)
                _shared_em = getattr(env, "_nested_canonical_em_order_cache", None)
                if _shared_ur is not None and _shared_em is not None:
                    shared_nested_ur_order_across_loads = np.asarray(_shared_ur, dtype=np.int32).copy()
                    shared_nested_em_order_across_loads = np.asarray(_shared_em, dtype=np.int32).copy()
            except Exception:
                shared_nested_ur_order_across_loads = None
                shared_nested_em_order_across_loads = None
        try:
            served_counts = [
                float(ep.get("embb_served_user_count", ep.get("embb_served_users", 0.0)) or 0.0)
                for ep in episodes
            ]
            if served_counts:
                prev_nested_embb_served_count = int(max(round(float(np.mean(served_counts))), 0))
            selected_ids = {
                int(x) for x in np.asarray(rep.get("nested_selected_user_indices", []), dtype=float).tolist()
            }
            selected_indices_arr = np.asarray(rep.get("nested_selected_user_indices", []), dtype=float)
            if selected_indices_arr.size > 0:
                _nu = int(sys_cfg.num_urllc_users)
                embb_sel = [
                    int(x)
                    for x in selected_indices_arr[_nu:].tolist()
                ]
                prev_nested_embb_subset_count = int(len(embb_sel))
                prev_nested_embb_selected_ids = [int(x) for x in embb_sel]
        except Exception:
            selected_ids = set()
        new_ids = sorted(int(x) for x in (selected_ids - prev_selected_ids))
        removed_ids = sorted(int(x) for x in (prev_selected_ids - selected_ids))
        prev_selected_ids = set(selected_ids)
        _report_log(
            f"[GREEDY][load={float(load):.1f}] nested_audit "
            f"| selected_users={len(selected_ids)} | newly_added={len(new_ids)} | removed_vs_prev={len(removed_ids)}"
            f" | embb_served_ref={int(prev_nested_embb_served_count or 0)}"
        )
        if new_ids:
            _report_log(f"[GREEDY][load={float(load):.1f}] nested_audit_new_ids={new_ids}")
        if removed_ids:
            _report_log(f"[GREEDY][load={float(load):.1f}] nested_audit_removed_ids={removed_ids}")
        try:
            embb_ids_per_uav = rep.get("audit_embb_global_ids_per_uav", [])
            urllc_ids_per_uav = rep.get("audit_urllc_global_ids_per_uav", [])
            embb_show = [
                [int(x) for x in np.asarray(v, dtype=float).tolist()] for v in embb_ids_per_uav
            ]
            urllc_show = [
                [int(x) for x in np.asarray(v, dtype=float).tolist()] for v in urllc_ids_per_uav
            ]
            _report_log(
                f"[GREEDY][load={float(load):.1f}] per_uav_global_ids "
                f"| embb={embb_show} | urllc={urllc_show}"
            )
        except Exception:
            pass
        try:
            _ov = np.asarray(rep.get("per_uav_overlay_count", []), dtype=float).tolist()
            _pu = np.asarray(rep.get("per_uav_puncture_count", []), dtype=float).tolist()
            _ad = np.asarray(rep.get("per_uav_scheduled_urllc", []), dtype=float).tolist()
            _report_log(
                f"[GREEDY][load={float(load):.1f}] per_uav_mode_admit "
                f"| overlay={_ov} | puncture={_pu} | admitted_urllc={_ad} "
                f"| overlay_feasible_pairs={float(rep.get('overlay_feasible_pairs', 0.0)):.3f} "
                f"| overlay_candidate_pairs={float(rep.get('overlay_candidate_pairs', 0.0)):.3f} "
                f"| scheduled_packets={float(rep.get('scheduled_packets', 0.0)):.3f}"
            )
        except Exception:
            pass
        # Per-episode greedy debug log (requested): arrivals/admission/throughput visibility.
        running_arrivals: List[float] = []
        running_admitted: List[float] = []
        running_admission_ratio: List[float] = []
        running_embb_rate_mbps: List[float] = []
        running_urllc_tp_mbps: List[float] = []
        running_urllc_users: List[float] = []
        running_embb_users: List[float] = []
        running_budget_used_ratio: List[float] = []
        running_episode_sec: List[float] = []
        running_reset_total_sec: List[float] = []
        running_prepare_slot_ctx_sec: List[float] = []
        running_arrival_gen_sec: List[float] = []
        running_reset_state_sec: List[float] = []
        running_reset_greedy_ref_sec: List[float] = []
        running_reset_build_obs_sec: List[float] = []
        running_reset_misc_sec: List[float] = []
        running_action_select_sec: List[float] = []
        running_action_resolve_sec: List[float] = []
        running_env_step_sec: List[float] = []
        running_hf_eval_sec: List[float] = []
        running_hf_prefilter_sec: List[float] = []
        running_hf_fastpath_sec: List[float] = []
        running_hf_rescan_used: List[float] = []
        running_step_profile_total_sec: List[float] = []
        running_step_profile_obs_build_sec: List[float] = []
        running_step_profile_obs_enum_sec: List[float] = []
        running_step_profile_obs_candidate_enum_sec: List[float] = []
        running_step_profile_obs_agent_loop_sec: List[float] = []
        running_step_profile_obs_select_sec: List[float] = []
        running_step_profile_obs_greedy_ref_sec: List[float] = []
        running_step_profile_obs_local_sec: List[float] = []
        running_step_profile_obs_global_sec: List[float] = []
        running_step_profile_obs_mask_sec: List[float] = []
        running_step_profile_obs_meta_sec: List[float] = []
        running_step_profile_obs_pack_sec: List[float] = []
        running_step_profile_obs_flatten_concat_sec: List[float] = []
        running_step_profile_apply_action_sec: List[float] = []
        running_step_profile_state_update_sec: List[float] = []
        running_step_profile_reward_compute_sec: List[float] = []
        running_step_profile_reward_delta_sec: List[float] = []
        running_step_profile_reward_fullscan_sec: List[float] = []
        running_step_profile_interference_sec: List[float] = []
        running_step_profile_rate_sec: List[float] = []
        running_step_profile_interference_calls: List[float] = []
        running_step_profile_interference_cache_hit_calls: List[float] = []
        running_step_profile_rate_calls: List[float] = []
        running_other_sec: List[float] = []
        running_baseline_cache_hit_ratio: List[float] = []
        running_virtual_slot_reset_count: List[float] = []
        running_hf_raw_count: List[float] = []
        running_hf_admissible_count: List[float] = []
        running_hf_evaluated_count: List[float] = []
        running_hf_feasible_count: List[float] = []
        running_hf_selected_count: List[float] = []
        running_hf_reject_gate_assoc: List[float] = []
        running_hf_reject_gate_queue: List[float] = []
        running_hf_reject_gate_mode: List[float] = []
        running_hf_reject_gate_owner: List[float] = []
        running_hf_reject_gate_rb_local: List[float] = []
        running_hf_mode_overlay_rel_fail: List[float] = []
        running_hf_mode_overlay_sic_fail: List[float] = []
        running_hf_mode_gain_ratio_fail: List[float] = []
        running_hf_mode_owner_pool_missing: List[float] = []
        running_hf_mode_owner_unresolved_due_to_mode_fail: List[float] = []
        running_hf_mode_puncture_rel_fail: List[float] = []
        running_hf_mode_puncture_sic_fail: List[float] = []
        running_hf_mode_puncture_owner_missing: List[float] = []
        running_hf_owner_missing: List[float] = []
        running_hf_owner_mismatch: List[float] = []
        running_pre_mode_raw_pair: List[float] = []
        running_pre_mode_overlay_rel: List[float] = []
        running_pre_mode_overlay_sic: List[float] = []
        running_pre_mode_gain_ratio: List[float] = []
        running_pre_mode_owner_pool_missing: List[float] = []
        running_pre_mode_owner_unresolved_due_to_mode_fail: List[float] = []
        running_pre_mode_puncture_rel: List[float] = []
        running_pre_mode_puncture_sic: List[float] = []
        running_pre_mode_puncture_owner_missing: List[float] = []
        running_pre_mode_owner_missing: List[float] = []
        running_pre_mode_owner_mismatch: List[float] = []
        running_sic_prior_pair_block_ratio: List[float] = []
        running_sic_prior_saved_mode_fail_ratio: List[float] = []
        running_sic_prior_owner_total_per_decision: List[float] = []
        running_hf_gate_overlay_rel_fail_margin: List[float] = []
        running_hf_gate_overlay_sic_fail_margin_db: List[float] = []
        running_hf_gate_puncture_rel_fail_margin: List[float] = []
        running_hf_gate_target_reliability: List[float] = []
        running_hf_gate_target_sic_db: List[float] = []
        running_hf_gate_overlay_rel_fail_snir_db: List[float] = []
        running_hf_gate_overlay_sic_fail_post_sic_db: List[float] = []
        running_hf_gate_puncture_rel_fail_snir_db: List[float] = []
        running_hf_selected_minus_admitted_reliability: List[float] = []
        running_hf_overlay_sic_trace_pre_sinr_db: List[float] = []
        running_hf_overlay_sic_trace_post_sinr_db: List[float] = []
        running_hf_overlay_sic_trace_noise_power: List[float] = []
        running_hf_overlay_sic_trace_intercell_interference: List[float] = []
        running_hf_overlay_sic_trace_local_interference: List[float] = []
        running_hf_overlay_sic_trace_residual_interference: List[float] = []
        running_hf_overlay_sic_trace_residual_ratio: List[float] = []
        running_hf_overlay_presinr_raw_lt_m10: List[float] = []
        running_hf_overlay_presinr_raw_m10_m6: List[float] = []
        running_hf_overlay_presinr_raw_m6_m2: List[float] = []
        running_hf_overlay_presinr_raw_ge_m2: List[float] = []
        running_hf_overlay_presinr_kept_low_ratio: List[float] = []
        running_hf_overlay_presinr_eval_low_ratio: List[float] = []
        running_hf_quality_priority_enabled: List[float] = []
        running_hf_quality_raw_high: List[float] = []
        running_hf_quality_raw_borderline: List[float] = []
        running_hf_quality_raw_risk: List[float] = []
        running_hf_quality_kept_high: List[float] = []
        running_hf_quality_kept_borderline: List[float] = []
        running_hf_quality_kept_risk: List[float] = []
        running_hf_quality_eval_high: List[float] = []
        running_hf_quality_eval_borderline: List[float] = []
        running_hf_quality_eval_risk: List[float] = []
        running_hf_quality_selected_high: List[float] = []
        running_hf_quality_selected_borderline: List[float] = []
        running_hf_quality_selected_risk: List[float] = []
        running_hf_overlay_blacklist_cell_ratio: List[float] = []
        running_hf_overlay_blacklist_candidate_block_ratio: List[float] = []
        running_hf_overlay_blacklist_saved_mode_fail_ratio: List[float] = []
        running_hf_overlay_rb_reservation_enabled: List[float] = []
        running_hf_overlay_rb_reservation_ratio: List[float] = []
        running_hf_overlay_rb_reserved_count: List[float] = []
        running_hf_overlay_rb_reserved_cell_ratio: List[float] = []
        running_hf_overlay_rb_mode_block_ratio: List[float] = []
        running_hf_overlay_rb_hard_block_enabled: List[float] = []
        running_hf_overlay_rb_soft_preference_enabled: List[float] = []
        running_hf_overlay_rb_soft_preference_applied_ratio: List[float] = []
        running_guardrail_enabled: List[float] = []
        running_guardrail_resample_count: List[float] = []
        running_guardrail_pass_ratio: List[float] = []
        running_guardrail_reject_overlay: List[float] = []
        running_guardrail_reject_embb_minrate: List[float] = []
        running_guardrail_reject_uav_imbalance: List[float] = []
        running_guardrail_actual_overlay: List[float] = []
        running_guardrail_actual_minrate: List[float] = []
        running_guardrail_actual_imbalance: List[float] = []
        running_guardrail_threshold_overlay: List[float] = []
        running_guardrail_threshold_minrate: List[float] = []
        running_guardrail_threshold_imbalance: List[float] = []
        running_mother_topology_id: List[str] = []
        running_mother_topology_seed: List[float] = []
        running_same_channel_hash: List[str] = []
        running_same_assoc_hash: List[str] = []
        running_same_user_pool_hash: List[str] = []
        running_mix_user_subset_hash: List[str] = []
        running_embb_subset_hash: List[str] = []
        running_same_feasible_graph_hash: List[str] = []
        running_feasible_graph_id: List[str] = []
        running_overlay_graph_hash: List[str] = []
        running_channel_matrix_hash: List[str] = []
        running_pathloss_hash: List[str] = []
        running_shadowing_hash: List[str] = []
        running_sic_order_hash: List[str] = []
        running_repair_sequence_hash: List[str] = []
        running_guardrail_pass_flag: List[float] = []
        heartbeat_every = max(1, _env_int_override("SR_MAPPO_REPORT_HEARTBEAT_EVERY_EPISODES", REPORT_HEARTBEAT_EVERY_EPISODES))
        load_loop_t0 = perf_counter()
        for ep_idx, episode in enumerate(episodes, start=1):
            arrivals = float(episode.get('active_packets', 0.0) or 0.0)
            admitted = float(episode.get('scheduled_packets', 0.0) or 0.0)
            admission_ratio = float(episode.get('urllc_admission', 0.0) or 0.0)
            embb_rate_mbps = float((episode.get('embb_rate', 0.0) or 0.0) / 1.0e6)
            urllc_tp_mbps = float(episode.get('urllc_throughput_mbps_slot_est', 0.0) or 0.0)
            urllc_users = int(episode.get('urllc_user_count', 0) or 0)
            embb_users = int(episode.get('embb_user_count', 0) or 0)
            budget_used_ratio = float(
                episode.get(
                    "greedy_urllc_budget_utilization_ratio",
                    episode.get("greedy_urllc_budget_used_ratio", 0.0),
                )
                or 0.0
            )
            episode_sec = float(episode.get("episode_sec", 0.0) or 0.0)
            reset_total_sec = float(episode.get("profile_reset_total_sec", 0.0) or 0.0)
            prepare_slot_ctx_sec = float(episode.get("profile_prepare_slot_context_sec", 0.0) or 0.0)
            arrival_gen_sec = float(episode.get("profile_arrival_generation_sec", 0.0) or 0.0)
            reset_state_sec = float(episode.get("profile_reset_episode_state_sec", 0.0) or 0.0)
            reset_greedy_ref_sec = float(episode.get("profile_reset_greedy_reference_sec", 0.0) or 0.0)
            reset_build_obs_sec = float(episode.get("profile_reset_build_observations_sec", 0.0) or 0.0)
            reset_misc_sec = float(episode.get("profile_reset_misc_sec", 0.0) or 0.0)
            action_select_sec = float(episode.get("profile_action_select_sec", 0.0) or 0.0)
            action_resolve_sec = float(episode.get("profile_action_resolve_sec", 0.0) or 0.0)
            env_step_sec = float(episode.get("profile_env_step_sec", 0.0) or 0.0)
            hf_eval_sec = float(episode.get("profile_hf_eval_sec", 0.0) or 0.0)
            hf_prefilter_sec = float(episode.get("profile_hf_prefilter_sec", 0.0) or 0.0)
            hf_fastpath_sec = float(episode.get("profile_hf_fastpath_sec", 0.0) or 0.0)
            hf_rescan_used = float(episode.get("greedy_hf_rescan_used", 0.0) or 0.0)
            step_profile_total_sec = float(episode.get("profile_step_total_sec", 0.0) or 0.0)
            step_profile_obs_build_sec = float(episode.get("profile_step_obs_build_sec", 0.0) or 0.0)
            step_profile_obs_enum_sec = float(episode.get("profile_obs_enum_sec", 0.0) or 0.0)
            step_profile_obs_candidate_enum_sec = float(episode.get("profile_obs_candidate_enum_sec", step_profile_obs_enum_sec) or 0.0)
            step_profile_obs_agent_loop_sec = float(episode.get("profile_obs_agent_loop_sec", 0.0) or 0.0)
            step_profile_obs_select_sec = float(episode.get("profile_obs_select_sec", 0.0) or 0.0)
            step_profile_obs_greedy_ref_sec = float(episode.get("profile_obs_greedy_ref_sec", 0.0) or 0.0)
            step_profile_obs_local_sec = float(episode.get("profile_obs_local_sec", 0.0) or 0.0)
            step_profile_obs_global_sec = float(episode.get("profile_obs_global_sec", 0.0) or 0.0)
            step_profile_obs_mask_sec = float(episode.get("profile_obs_mask_sec", 0.0) or 0.0)
            step_profile_obs_meta_sec = float(episode.get("profile_obs_meta_sec", 0.0) or 0.0)
            step_profile_obs_pack_sec = float(episode.get("profile_obs_pack_sec", 0.0) or 0.0)
            step_profile_obs_flatten_concat_sec = float(episode.get("profile_obs_flatten_concat_sec", 0.0) or 0.0)
            step_profile_apply_action_sec = float(episode.get("profile_step_apply_action_sec", 0.0) or 0.0)
            step_profile_state_update_sec = float(episode.get("profile_step_state_update_sec", 0.0) or 0.0)
            step_profile_reward_compute_sec = float(episode.get("profile_step_reward_compute_sec", 0.0) or 0.0)
            step_profile_reward_delta_sec = float(episode.get("profile_step_reward_delta_sec", 0.0) or 0.0)
            step_profile_reward_fullscan_sec = float(episode.get("profile_step_reward_fullscan_sec", 0.0) or 0.0)
            step_profile_interference_sec = float(episode.get("profile_interference_update_sec", 0.0) or 0.0)
            step_profile_rate_sec = float(episode.get("profile_rate_update_sec", 0.0) or 0.0)
            step_profile_interference_calls = float(episode.get("profile_interference_update_calls", 0.0) or 0.0)
            step_profile_interference_cache_hit_calls = float(episode.get("profile_interference_cache_hit_calls", 0.0) or 0.0)
            step_profile_rate_calls = float(episode.get("profile_rate_update_calls", 0.0) or 0.0)
            baseline_cache_hit_ratio = float(episode.get("phase0_baseline_cache_hit_ratio", 0.0) or 0.0)
            virtual_slot_reset_count = float(episode.get("virtual_slot_reset_count_per_episode", 1.0) or 1.0)
            hf_raw_count = float(episode.get("greedy_hf_raw_count", 0.0) or 0.0)
            hf_admissible_count = float(episode.get("greedy_hf_admissible_count", 0.0) or 0.0)
            hf_evaluated_count = float(episode.get("greedy_hf_evaluated_count", 0.0) or 0.0)
            hf_feasible_count = float(episode.get("greedy_hf_feasible_count", 0.0) or 0.0)
            hf_selected_count = float(episode.get("greedy_hf_selected_count", 0.0) or 0.0)
            hf_reject_gate_assoc = float(episode.get("greedy_hf_reject_by_gate_association", 0.0) or 0.0)
            hf_reject_gate_queue = float(episode.get("greedy_hf_reject_by_gate_queue_active", 0.0) or 0.0)
            hf_reject_gate_mode = float(episode.get("greedy_hf_reject_by_gate_mode_admissible", 0.0) or 0.0)
            hf_reject_gate_owner = float(episode.get("greedy_hf_reject_by_gate_owner_admissible", 0.0) or 0.0)
            hf_reject_gate_rb_local = float(episode.get("greedy_hf_reject_by_gate_rb_local", 0.0) or 0.0)
            hf_mode_overlay_rel_fail = float(episode.get("greedy_hf_reject_by_mode_overlay_rel_fail", 0.0) or 0.0)
            hf_mode_overlay_sic_fail = float(episode.get("greedy_hf_reject_by_mode_overlay_sic_fail", 0.0) or 0.0)
            hf_mode_gain_ratio_fail = float(episode.get("greedy_hf_reject_by_mode_gain_ratio_fail", 0.0) or 0.0)
            hf_mode_owner_pool_missing = float(episode.get("greedy_hf_reject_by_mode_owner_pool_missing", episode.get("greedy_hf_reject_by_mode_overlay_owner_missing", 0.0)) or 0.0)
            hf_mode_owner_unresolved_due_to_mode_fail = float(episode.get("greedy_hf_reject_by_mode_owner_unresolved_due_to_mode_fail", 0.0) or 0.0)
            hf_mode_puncture_rel_fail = float(episode.get("greedy_hf_reject_by_mode_puncture_rel_fail", 0.0) or 0.0)
            hf_mode_puncture_sic_fail = float(episode.get("greedy_hf_reject_by_mode_puncture_sic_fail", 0.0) or 0.0)
            hf_mode_puncture_owner_missing = float(episode.get("greedy_hf_reject_by_mode_puncture_owner_missing", 0.0) or 0.0)
            hf_owner_missing = float(episode.get("greedy_hf_reject_by_owner_missing", 0.0) or 0.0)
            hf_owner_mismatch = float(episode.get("greedy_hf_reject_by_owner_mismatch", 0.0) or 0.0)
            pre_mode_raw_pair = float(episode.get("pre_mode_raw_pair_per_decision", 0.0) or 0.0)
            pre_mode_overlay_rel = float(episode.get("pre_mode_overlay_rel_fail_per_decision", 0.0) or 0.0)
            pre_mode_overlay_sic = float(episode.get("pre_mode_overlay_sic_fail_per_decision", 0.0) or 0.0)
            pre_mode_gain_ratio = float(episode.get("pre_mode_gain_ratio_fail_per_decision", 0.0) or 0.0)
            pre_mode_owner_pool_missing = float(episode.get("pre_mode_owner_pool_missing_per_decision", episode.get("pre_mode_overlay_owner_missing_per_decision", 0.0)) or 0.0)
            pre_mode_owner_unresolved_due_to_mode_fail = float(episode.get("pre_mode_owner_unresolved_due_to_mode_fail_per_decision", 0.0) or 0.0)
            pre_mode_puncture_rel = float(episode.get("pre_mode_puncture_rel_fail_per_decision", 0.0) or 0.0)
            pre_mode_puncture_sic = float(episode.get("pre_mode_puncture_sic_fail_per_decision", 0.0) or 0.0)
            pre_mode_puncture_owner_missing = float(episode.get("pre_mode_puncture_owner_missing_per_decision", 0.0) or 0.0)
            pre_mode_owner_missing = float(episode.get("pre_mode_owner_missing_per_decision", 0.0) or 0.0)
            pre_mode_owner_mismatch = float(episode.get("pre_mode_owner_mismatch_per_decision", 0.0) or 0.0)
            sic_prior_pair_block_ratio = float(episode.get("sic_prior_pair_block_ratio", 0.0) or 0.0)
            sic_prior_saved_mode_fail_ratio = float(episode.get("sic_prior_saved_mode_fail_ratio", 0.0) or 0.0)
            sic_prior_owner_total_per_decision = float(episode.get("sic_prior_owner_total_per_decision", 0.0) or 0.0)
            hf_gate_overlay_rel_fail_margin = float(episode.get("greedy_hf_gate_overlay_rel_fail_margin_mean", 0.0) or 0.0)
            hf_gate_overlay_sic_fail_margin_db = float(episode.get("greedy_hf_gate_overlay_sic_fail_margin_db_mean", 0.0) or 0.0)
            hf_gate_puncture_rel_fail_margin = float(episode.get("greedy_hf_gate_puncture_rel_fail_margin_mean", 0.0) or 0.0)
            hf_gate_target_reliability = float(episode.get("greedy_hf_gate_target_reliability", 0.0) or 0.0)
            hf_gate_target_sic_db = float(episode.get("greedy_hf_gate_target_sic_snir_db", 0.0) or 0.0)
            hf_gate_overlay_rel_fail_snir_db = float(episode.get("greedy_hf_gate_overlay_rel_fail_snir_db_mean", 0.0) or 0.0)
            hf_gate_overlay_sic_fail_post_sic_db = float(episode.get("greedy_hf_gate_overlay_sic_fail_post_sic_db_mean", 0.0) or 0.0)
            hf_gate_puncture_rel_fail_snir_db = float(episode.get("greedy_hf_gate_puncture_rel_fail_snir_db_mean", 0.0) or 0.0)
            selected_rel = float(episode.get("greedy_selected_urllc_reliability", 0.0) or 0.0)
            admitted_rel = float(episode.get("admitted_urllc_reliability", episode.get("urllc_reliability", 0.0)) or 0.0)
            selected_minus_admitted_rel = float(selected_rel - admitted_rel)
            overlay_sic_trace_pre = float(episode.get("greedy_hf_overlay_sic_trace_pre_sinr_db_mean", 0.0) or 0.0)
            overlay_sic_trace_post = float(episode.get("greedy_hf_overlay_sic_trace_post_sinr_db_mean", 0.0) or 0.0)
            overlay_sic_trace_noise = float(episode.get("greedy_hf_overlay_sic_trace_noise_power_mean", 0.0) or 0.0)
            overlay_sic_trace_intercell = float(episode.get("greedy_hf_overlay_sic_trace_intercell_interference_mean", 0.0) or 0.0)
            overlay_sic_trace_local = float(episode.get("greedy_hf_overlay_sic_trace_local_interference_mean", 0.0) or 0.0)
            overlay_sic_trace_residual = float(episode.get("greedy_hf_overlay_sic_trace_residual_sic_interference_mean", 0.0) or 0.0)
            overlay_sic_trace_ratio = float(episode.get("greedy_hf_overlay_sic_trace_sic_residual_ratio", 0.0) or 0.0)
            overlay_presinr_raw_lt_m10 = float(episode.get("greedy_hf_overlay_presinr_raw_lt_m10", 0.0) or 0.0)
            overlay_presinr_raw_m10_m6 = float(episode.get("greedy_hf_overlay_presinr_raw_m10_m6", 0.0) or 0.0)
            overlay_presinr_raw_m6_m2 = float(episode.get("greedy_hf_overlay_presinr_raw_m6_m2", 0.0) or 0.0)
            overlay_presinr_raw_ge_m2 = float(episode.get("greedy_hf_overlay_presinr_raw_ge_m2", 0.0) or 0.0)
            overlay_presinr_kept_low_ratio = float(episode.get("greedy_hf_overlay_presinr_kept_low_ratio", 0.0) or 0.0)
            overlay_presinr_eval_low_ratio = float(episode.get("greedy_hf_overlay_presinr_eval_low_ratio", 0.0) or 0.0)
            quality_priority_enabled = float(episode.get("greedy_hf_quality_priority_enabled", 0.0) or 0.0)
            quality_raw_high = float(episode.get("greedy_hf_quality_raw_high", 0.0) or 0.0)
            quality_raw_borderline = float(episode.get("greedy_hf_quality_raw_borderline", 0.0) or 0.0)
            quality_raw_risk = float(episode.get("greedy_hf_quality_raw_risk", 0.0) or 0.0)
            quality_kept_high = float(episode.get("greedy_hf_quality_kept_high", 0.0) or 0.0)
            quality_kept_borderline = float(episode.get("greedy_hf_quality_kept_borderline", 0.0) or 0.0)
            quality_kept_risk = float(episode.get("greedy_hf_quality_kept_risk", 0.0) or 0.0)
            quality_eval_high = float(episode.get("greedy_hf_quality_eval_high", 0.0) or 0.0)
            quality_eval_borderline = float(episode.get("greedy_hf_quality_eval_borderline", 0.0) or 0.0)
            quality_eval_risk = float(episode.get("greedy_hf_quality_eval_risk", 0.0) or 0.0)
            quality_selected_high = float(episode.get("greedy_hf_quality_selected_high", 0.0) or 0.0)
            quality_selected_borderline = float(episode.get("greedy_hf_quality_selected_borderline", 0.0) or 0.0)
            quality_selected_risk = float(episode.get("greedy_hf_quality_selected_risk", 0.0) or 0.0)
            overlay_blacklist_cell_ratio = float(episode.get("greedy_hf_overlay_blacklist_cell_ratio", 0.0) or 0.0)
            overlay_blacklist_candidate_block_ratio = float(
                episode.get("greedy_hf_overlay_blacklist_candidate_block_ratio", 0.0) or 0.0
            )
            overlay_blacklist_saved_mode_fail_ratio = float(
                episode.get("greedy_hf_overlay_blacklist_saved_mode_fail_ratio", 0.0) or 0.0
            )
            overlay_rb_reservation_enabled = float(
                episode.get("greedy_hf_overlay_only_rb_reservation_enabled", 0.0) or 0.0
            )
            overlay_rb_reservation_ratio = float(
                episode.get("greedy_hf_overlay_only_rb_reservation_ratio", 0.0) or 0.0
            )
            overlay_rb_reserved_count = float(
                episode.get("greedy_hf_overlay_only_rb_reserved_count", 0.0) or 0.0
            )
            overlay_rb_reserved_cell_ratio = float(
                episode.get("greedy_hf_overlay_only_rb_reserved_cell_ratio", 0.0) or 0.0
            )
            overlay_rb_mode_block_ratio = float(
                episode.get("greedy_hf_overlay_only_rb_mode_block_ratio", 0.0) or 0.0
            )
            overlay_rb_hard_block_enabled = float(
                episode.get("greedy_hf_overlay_only_rb_hard_block_enabled", 0.0) or 0.0
            )
            overlay_rb_soft_preference_enabled = float(
                episode.get("greedy_hf_overlay_only_rb_soft_preference_enabled", 0.0) or 0.0
            )
            overlay_rb_soft_preference_applied_ratio = float(
                episode.get("greedy_hf_overlay_only_rb_soft_preference_applied_ratio", 0.0) or 0.0
            )
            guardrail_enabled = float(episode.get("guardrail_enabled", 0.0) or 0.0)
            guardrail_resample_count = float(episode.get("guardrail_resample_count", 0.0) or 0.0)
            guardrail_pass_ratio = float(episode.get("guardrail_pass_ratio", 1.0) or 0.0)
            guardrail_reject_overlay = float(episode.get("guardrail_reject_reason_overlay", 0.0) or 0.0)
            guardrail_reject_embb_minrate = float(episode.get("guardrail_reject_reason_embb_minrate", 0.0) or 0.0)
            guardrail_reject_uav_imbalance = float(episode.get("guardrail_reject_reason_uav_imbalance", 0.0) or 0.0)
            guardrail_actual_overlay = float(episode.get("guardrail_actual_overlay_feasible_ratio", 0.0) or 0.0)
            guardrail_actual_minrate = float(episode.get("guardrail_actual_embb_minrate_ratio", 0.0) or 0.0)
            guardrail_actual_imbalance = float(episode.get("guardrail_actual_uav_load_imbalance", 0.0) or 0.0)
            guardrail_threshold_overlay = float(episode.get("guardrail_threshold_overlay", 0.0) or 0.0)
            guardrail_threshold_minrate = float(episode.get("guardrail_threshold_embb_minrate", 0.0) or 0.0)
            guardrail_threshold_imbalance = float(episode.get("guardrail_threshold_uav_imbalance", 0.0) or 0.0)
            mother_topology_id = str(episode.get("mother_topology_id", "") or "")
            mother_topology_seed = float(episode.get("mother_topology_seed", 0.0) or 0.0)
            same_channel_hash = str(episode.get("same_channel_hash", "") or "")
            same_assoc_hash = str(episode.get("same_assoc_hash", "") or "")
            same_user_pool_hash = str(episode.get("same_user_pool_hash", "") or "")
            mix_user_subset_hash = str(episode.get("mix_user_subset_hash", "") or "")
            embb_subset_hash = str(episode.get("embb_subset_hash", "") or "")
            same_feasible_graph_hash = str(episode.get("same_feasible_graph_hash", "") or "")
            feasible_graph_id = str(episode.get("feasible_graph_id", "") or "")
            overlay_graph_hash = str(episode.get("overlay_graph_hash", "") or "")
            channel_matrix_hash = str(episode.get("channel_matrix_hash", "") or "")
            pathloss_hash = str(episode.get("pathloss_hash", "") or "")
            shadowing_hash = str(episode.get("shadowing_hash", "") or "")
            sic_order_hash = str(episode.get("sic_order_hash", "") or "")
            repair_sequence_hash = str(episode.get("repair_sequence_hash", "") or "")
            guardrail_pass_flag = float(episode.get("guardrail_pass", episode.get("guardrail_pass_ratio", 0.0)) or 0.0)
            accounted_major_sec = reset_total_sec + action_select_sec + action_resolve_sec + env_step_sec
            other_sec = max(episode_sec - accounted_major_sec, 0.0)
            running_arrivals.append(arrivals)
            running_admitted.append(admitted)
            running_admission_ratio.append(admission_ratio)
            running_embb_rate_mbps.append(embb_rate_mbps)
            running_urllc_tp_mbps.append(urllc_tp_mbps)
            running_urllc_users.append(float(urllc_users))
            running_embb_users.append(float(embb_users))
            running_budget_used_ratio.append(budget_used_ratio)
            running_episode_sec.append(episode_sec)
            running_reset_total_sec.append(reset_total_sec)
            running_prepare_slot_ctx_sec.append(prepare_slot_ctx_sec)
            running_arrival_gen_sec.append(arrival_gen_sec)
            running_reset_state_sec.append(reset_state_sec)
            running_reset_greedy_ref_sec.append(reset_greedy_ref_sec)
            running_reset_build_obs_sec.append(reset_build_obs_sec)
            running_reset_misc_sec.append(reset_misc_sec)
            running_action_select_sec.append(action_select_sec)
            running_action_resolve_sec.append(action_resolve_sec)
            running_env_step_sec.append(env_step_sec)
            running_hf_eval_sec.append(hf_eval_sec)
            running_hf_prefilter_sec.append(hf_prefilter_sec)
            running_hf_fastpath_sec.append(hf_fastpath_sec)
            running_hf_rescan_used.append(hf_rescan_used)
            running_step_profile_total_sec.append(step_profile_total_sec)
            running_step_profile_obs_build_sec.append(step_profile_obs_build_sec)
            running_step_profile_obs_enum_sec.append(step_profile_obs_enum_sec)
            running_step_profile_obs_candidate_enum_sec.append(step_profile_obs_candidate_enum_sec)
            running_step_profile_obs_agent_loop_sec.append(step_profile_obs_agent_loop_sec)
            running_step_profile_obs_select_sec.append(step_profile_obs_select_sec)
            running_step_profile_obs_greedy_ref_sec.append(step_profile_obs_greedy_ref_sec)
            running_step_profile_obs_local_sec.append(step_profile_obs_local_sec)
            running_step_profile_obs_global_sec.append(step_profile_obs_global_sec)
            running_step_profile_obs_mask_sec.append(step_profile_obs_mask_sec)
            running_step_profile_obs_meta_sec.append(step_profile_obs_meta_sec)
            running_step_profile_obs_pack_sec.append(step_profile_obs_pack_sec)
            running_step_profile_obs_flatten_concat_sec.append(step_profile_obs_flatten_concat_sec)
            running_step_profile_apply_action_sec.append(step_profile_apply_action_sec)
            running_step_profile_state_update_sec.append(step_profile_state_update_sec)
            running_step_profile_reward_compute_sec.append(step_profile_reward_compute_sec)
            running_step_profile_reward_delta_sec.append(step_profile_reward_delta_sec)
            running_step_profile_reward_fullscan_sec.append(step_profile_reward_fullscan_sec)
            running_step_profile_interference_sec.append(step_profile_interference_sec)
            running_step_profile_rate_sec.append(step_profile_rate_sec)
            running_step_profile_interference_calls.append(step_profile_interference_calls)
            running_step_profile_interference_cache_hit_calls.append(step_profile_interference_cache_hit_calls)
            running_step_profile_rate_calls.append(step_profile_rate_calls)
            running_other_sec.append(other_sec)
            running_baseline_cache_hit_ratio.append(baseline_cache_hit_ratio)
            running_virtual_slot_reset_count.append(virtual_slot_reset_count)
            running_hf_raw_count.append(hf_raw_count)
            running_hf_admissible_count.append(hf_admissible_count)
            running_hf_evaluated_count.append(hf_evaluated_count)
            running_hf_feasible_count.append(hf_feasible_count)
            running_hf_selected_count.append(hf_selected_count)
            running_hf_reject_gate_assoc.append(hf_reject_gate_assoc)
            running_hf_reject_gate_queue.append(hf_reject_gate_queue)
            running_hf_reject_gate_mode.append(hf_reject_gate_mode)
            running_hf_reject_gate_owner.append(hf_reject_gate_owner)
            running_hf_reject_gate_rb_local.append(hf_reject_gate_rb_local)
            running_hf_mode_overlay_rel_fail.append(hf_mode_overlay_rel_fail)
            running_hf_mode_overlay_sic_fail.append(hf_mode_overlay_sic_fail)
            running_hf_mode_gain_ratio_fail.append(hf_mode_gain_ratio_fail)
            running_hf_mode_owner_pool_missing.append(hf_mode_owner_pool_missing)
            running_hf_mode_owner_unresolved_due_to_mode_fail.append(hf_mode_owner_unresolved_due_to_mode_fail)
            running_hf_mode_puncture_rel_fail.append(hf_mode_puncture_rel_fail)
            running_hf_mode_puncture_sic_fail.append(hf_mode_puncture_sic_fail)
            running_hf_mode_puncture_owner_missing.append(hf_mode_puncture_owner_missing)
            running_hf_owner_missing.append(hf_owner_missing)
            running_hf_owner_mismatch.append(hf_owner_mismatch)
            running_pre_mode_raw_pair.append(pre_mode_raw_pair)
            running_pre_mode_overlay_rel.append(pre_mode_overlay_rel)
            running_pre_mode_overlay_sic.append(pre_mode_overlay_sic)
            running_pre_mode_gain_ratio.append(pre_mode_gain_ratio)
            running_pre_mode_owner_pool_missing.append(pre_mode_owner_pool_missing)
            running_pre_mode_owner_unresolved_due_to_mode_fail.append(pre_mode_owner_unresolved_due_to_mode_fail)
            running_pre_mode_puncture_rel.append(pre_mode_puncture_rel)
            running_pre_mode_puncture_sic.append(pre_mode_puncture_sic)
            running_pre_mode_puncture_owner_missing.append(pre_mode_puncture_owner_missing)
            running_pre_mode_owner_missing.append(pre_mode_owner_missing)
            running_pre_mode_owner_mismatch.append(pre_mode_owner_mismatch)
            running_sic_prior_pair_block_ratio.append(sic_prior_pair_block_ratio)
            running_sic_prior_saved_mode_fail_ratio.append(sic_prior_saved_mode_fail_ratio)
            running_sic_prior_owner_total_per_decision.append(sic_prior_owner_total_per_decision)
            running_hf_gate_overlay_rel_fail_margin.append(hf_gate_overlay_rel_fail_margin)
            running_hf_gate_overlay_sic_fail_margin_db.append(hf_gate_overlay_sic_fail_margin_db)
            running_hf_gate_puncture_rel_fail_margin.append(hf_gate_puncture_rel_fail_margin)
            running_hf_gate_target_reliability.append(hf_gate_target_reliability)
            running_hf_gate_target_sic_db.append(hf_gate_target_sic_db)
            running_hf_gate_overlay_rel_fail_snir_db.append(hf_gate_overlay_rel_fail_snir_db)
            running_hf_gate_overlay_sic_fail_post_sic_db.append(hf_gate_overlay_sic_fail_post_sic_db)
            running_hf_gate_puncture_rel_fail_snir_db.append(hf_gate_puncture_rel_fail_snir_db)
            running_hf_selected_minus_admitted_reliability.append(selected_minus_admitted_rel)
            running_hf_overlay_sic_trace_pre_sinr_db.append(overlay_sic_trace_pre)
            running_hf_overlay_sic_trace_post_sinr_db.append(overlay_sic_trace_post)
            running_hf_overlay_sic_trace_noise_power.append(overlay_sic_trace_noise)
            running_hf_overlay_sic_trace_intercell_interference.append(overlay_sic_trace_intercell)
            running_hf_overlay_sic_trace_local_interference.append(overlay_sic_trace_local)
            running_hf_overlay_sic_trace_residual_interference.append(overlay_sic_trace_residual)
            running_hf_overlay_sic_trace_residual_ratio.append(overlay_sic_trace_ratio)
            running_hf_overlay_presinr_raw_lt_m10.append(overlay_presinr_raw_lt_m10)
            running_hf_overlay_presinr_raw_m10_m6.append(overlay_presinr_raw_m10_m6)
            running_hf_overlay_presinr_raw_m6_m2.append(overlay_presinr_raw_m6_m2)
            running_hf_overlay_presinr_raw_ge_m2.append(overlay_presinr_raw_ge_m2)
            running_hf_overlay_presinr_kept_low_ratio.append(overlay_presinr_kept_low_ratio)
            running_hf_overlay_presinr_eval_low_ratio.append(overlay_presinr_eval_low_ratio)
            running_hf_quality_priority_enabled.append(quality_priority_enabled)
            running_hf_quality_raw_high.append(quality_raw_high)
            running_hf_quality_raw_borderline.append(quality_raw_borderline)
            running_hf_quality_raw_risk.append(quality_raw_risk)
            running_hf_quality_kept_high.append(quality_kept_high)
            running_hf_quality_kept_borderline.append(quality_kept_borderline)
            running_hf_quality_kept_risk.append(quality_kept_risk)
            running_hf_quality_eval_high.append(quality_eval_high)
            running_hf_quality_eval_borderline.append(quality_eval_borderline)
            running_hf_quality_eval_risk.append(quality_eval_risk)
            running_hf_quality_selected_high.append(quality_selected_high)
            running_hf_quality_selected_borderline.append(quality_selected_borderline)
            running_hf_quality_selected_risk.append(quality_selected_risk)
            running_hf_overlay_blacklist_cell_ratio.append(overlay_blacklist_cell_ratio)
            running_hf_overlay_blacklist_candidate_block_ratio.append(overlay_blacklist_candidate_block_ratio)
            running_hf_overlay_blacklist_saved_mode_fail_ratio.append(overlay_blacklist_saved_mode_fail_ratio)
            running_hf_overlay_rb_reservation_enabled.append(overlay_rb_reservation_enabled)
            running_hf_overlay_rb_reservation_ratio.append(overlay_rb_reservation_ratio)
            running_hf_overlay_rb_reserved_count.append(overlay_rb_reserved_count)
            running_hf_overlay_rb_reserved_cell_ratio.append(overlay_rb_reserved_cell_ratio)
            running_hf_overlay_rb_mode_block_ratio.append(overlay_rb_mode_block_ratio)
            running_hf_overlay_rb_hard_block_enabled.append(overlay_rb_hard_block_enabled)
            running_hf_overlay_rb_soft_preference_enabled.append(overlay_rb_soft_preference_enabled)
            running_hf_overlay_rb_soft_preference_applied_ratio.append(overlay_rb_soft_preference_applied_ratio)
            running_guardrail_enabled.append(guardrail_enabled)
            running_guardrail_resample_count.append(guardrail_resample_count)
            running_guardrail_pass_ratio.append(guardrail_pass_ratio)
            running_guardrail_reject_overlay.append(guardrail_reject_overlay)
            running_guardrail_reject_embb_minrate.append(guardrail_reject_embb_minrate)
            running_guardrail_reject_uav_imbalance.append(guardrail_reject_uav_imbalance)
            running_guardrail_actual_overlay.append(guardrail_actual_overlay)
            running_guardrail_actual_minrate.append(guardrail_actual_minrate)
            running_guardrail_actual_imbalance.append(guardrail_actual_imbalance)
            running_guardrail_threshold_overlay.append(guardrail_threshold_overlay)
            running_guardrail_threshold_minrate.append(guardrail_threshold_minrate)
            running_guardrail_threshold_imbalance.append(guardrail_threshold_imbalance)
            running_mother_topology_id.append(mother_topology_id)
            running_mother_topology_seed.append(mother_topology_seed)
            running_same_channel_hash.append(same_channel_hash)
            running_same_assoc_hash.append(same_assoc_hash)
            running_same_user_pool_hash.append(same_user_pool_hash)
            running_mix_user_subset_hash.append(mix_user_subset_hash)
            running_embb_subset_hash.append(embb_subset_hash)
            running_same_feasible_graph_hash.append(same_feasible_graph_hash)
            running_feasible_graph_id.append(feasible_graph_id)
            running_overlay_graph_hash.append(overlay_graph_hash)
            running_channel_matrix_hash.append(channel_matrix_hash)
            running_pathloss_hash.append(pathloss_hash)
            running_shadowing_hash.append(shadowing_hash)
            running_sic_order_hash.append(sic_order_hash)
            running_repair_sequence_hash.append(repair_sequence_hash)
            running_guardrail_pass_flag.append(guardrail_pass_flag)
            if verbose_per_episode:
                _report_log(
                    f"[GREEDY][load={float(load):.1f}] episode {ep_idx}/{int(episodes_per_load)} "
                    f"| users(urllc/embb)={urllc_users}/{embb_users} "
                    f"| arrivals={arrivals:.2f} | admitted={admitted:.2f} | admission={admission_ratio:.4f} "
                    f"| embb_rate={embb_rate_mbps:.3f} Mbps | urllc_tp={urllc_tp_mbps:.3f} Mbps"
                )
            if (ep_idx % heartbeat_every == 0) or (ep_idx == int(episodes_per_load)):
                elapsed_sec = max(perf_counter() - load_loop_t0, 1.0e-12)
                avg_ep_sec = float(elapsed_sec / max(ep_idx, 1))
                remain_ep = max(int(episodes_per_load) - int(ep_idx), 0)
                eta_sec = float(avg_ep_sec * remain_ep)
                _report_log(
                    f"[GREEDY][load={float(load):.1f}] progress {ep_idx}/{int(episodes_per_load)} "
                    f"| elapsed={elapsed_sec:.1f}s | avg_ep={avg_ep_sec:.3f}s | eta={eta_sec:.1f}s "
                    f"| arrivals_mean={float(np.mean(running_arrivals)):.2f} | admitted_mean={float(np.mean(running_admitted)):.2f}"
                )
            if ep_idx % 10 == 0:
                mean_urllc_users = float(np.mean(running_urllc_users)) if running_urllc_users else 0.0
                mean_embb_users = float(np.mean(running_embb_users)) if running_embb_users else 0.0
                embb_to_urllc_ratio = float(mean_embb_users / max(mean_urllc_users, 1.0e-9))
                total_episode_sec = max(float(np.sum(running_episode_sec)), 1.0e-12)
                total_action_select_sec = float(np.sum(running_action_select_sec))
                total_hf_prefilter_sec = float(np.sum(running_hf_prefilter_sec))
                total_hf_eval_sec = float(np.sum(running_hf_eval_sec))
                total_hf_fastpath_sec = float(np.sum(running_hf_fastpath_sec))
                total_action_other_sec = max(
                    total_action_select_sec - total_hf_prefilter_sec - total_hf_eval_sec - total_hf_fastpath_sec,
                    0.0,
                )
                _report_log(
                    f"[GREEDY][load={float(load):.1f}] time_breakdown(1-{ep_idx}) "
                    f"| reset_total={float(np.sum(running_reset_total_sec)) / total_episode_sec:.3%} "
                    f"| action_select={total_action_select_sec / total_episode_sec:.3%} "
                    f"| action_resolve={float(np.sum(running_action_resolve_sec)) / total_episode_sec:.3%} "
                    f"| env_step={float(np.sum(running_env_step_sec)) / total_episode_sec:.3%} "
                    f"| residual_other={float(np.sum(running_other_sec)) / total_episode_sec:.3%}"
                )
                _report_log(
                    f"[GREEDY][load={float(load):.1f}] time_breakdown_detail(1-{ep_idx}) "
                    f"| reset(state/slot_ctx/arrival/greedy_ref/build_obs/misc)="
                    f"{float(np.sum(running_reset_state_sec)) / total_episode_sec:.3%}/"
                    f"{float(np.sum(running_prepare_slot_ctx_sec)) / total_episode_sec:.3%}/"
                    f"{float(np.sum(running_arrival_gen_sec)) / total_episode_sec:.3%}/"
                    f"{float(np.sum(running_reset_greedy_ref_sec)) / total_episode_sec:.3%}/"
                    f"{float(np.sum(running_reset_build_obs_sec)) / total_episode_sec:.3%}/"
                    f"{float(np.sum(running_reset_misc_sec)) / total_episode_sec:.3%} "
                    f"| action(hf_prefilter/hf_eval/hf_fastpath/other)="
                    f"{total_hf_prefilter_sec / total_episode_sec:.3%}/"
                    f"{total_hf_eval_sec / total_episode_sec:.3%}/"
                    f"{total_hf_fastpath_sec / total_episode_sec:.3%}/"
                    f"{total_action_other_sec / total_episode_sec:.3%}"
                )
                _report_log(
                    f"[GREEDY][load={float(load):.1f}] env_step_breakdown(1-{ep_idx}) "
                    f"| step(obs_build/apply/state/reward/interference/rate)="
                    f"{float(np.sum(running_step_profile_obs_build_sec)) / total_episode_sec:.3%}/"
                    f"{float(np.sum(running_step_profile_apply_action_sec)) / total_episode_sec:.3%}/"
                    f"{float(np.sum(running_step_profile_state_update_sec)) / total_episode_sec:.3%}/"
                    f"{float(np.sum(running_step_profile_reward_compute_sec)) / total_episode_sec:.3%}/"
                    f"{float(np.sum(running_step_profile_interference_sec)) / total_episode_sec:.3%}/"
                    f"{float(np.sum(running_step_profile_rate_sec)) / total_episode_sec:.3%} "
                    f"| calls(interference/rate)="
                    f"{float(np.sum(running_step_profile_interference_calls)):.0f}/"
                    f"{float(np.sum(running_step_profile_rate_calls)):.0f}"
                    f" | cache_hits(interference)="
                    f"{float(np.sum(running_step_profile_interference_cache_hit_calls)):.0f}"
                )
                reward_total_sec = float(np.sum(running_step_profile_reward_compute_sec))
                reward_delta_sec = float(np.sum(running_step_profile_reward_delta_sec))
                reward_fullscan_sec = float(np.sum(running_step_profile_reward_fullscan_sec))
                reward_other_sec = max(reward_total_sec - reward_delta_sec - reward_fullscan_sec, 0.0)
                reward_denom = max(reward_total_sec, 1.0e-12)
                _report_log(
                    f"[GREEDY][load={float(load):.1f}] reward_sub_breakdown(1-{ep_idx}) "
                    f"| reward(delta/fullscan/other)="
                    f"{reward_delta_sec / reward_denom:.3%}/"
                    f"{reward_fullscan_sec / reward_denom:.3%}/"
                    f"{reward_other_sec / reward_denom:.3%}"
                )
                obs_total_sec = float(np.sum(running_step_profile_obs_build_sec))
                obs_select_sec = float(np.sum(running_step_profile_obs_select_sec))
                obs_greedy_ref_sec = float(np.sum(running_step_profile_obs_greedy_ref_sec))
                obs_local_sec = float(np.sum(running_step_profile_obs_local_sec))
                obs_global_sec = float(np.sum(running_step_profile_obs_global_sec))
                obs_mask_sec = float(np.sum(running_step_profile_obs_mask_sec))
                # enum_other = obs_build minus already-explicit major observation builders.
                obs_enum_other_sec = max(
                    obs_total_sec - (
                        obs_select_sec
                        + obs_greedy_ref_sec
                        + obs_local_sec
                        + obs_global_sec
                        + obs_mask_sec
                    ),
                    0.0,
                )
                obs_candidate_enum_sec = float(np.sum(running_step_profile_obs_candidate_enum_sec))
                obs_agent_loop_sec = float(np.sum(running_step_profile_obs_agent_loop_sec))
                obs_pack_sec = float(np.sum(running_step_profile_obs_pack_sec))
                obs_flatten_concat_sec = float(np.sum(running_step_profile_obs_flatten_concat_sec))
                obs_metadata_sec = float(np.sum(running_step_profile_obs_meta_sec))
                obs_other_sec = max(
                    obs_enum_other_sec - (
                        obs_agent_loop_sec
                        + obs_candidate_enum_sec
                        + obs_pack_sec
                        + obs_flatten_concat_sec
                        + obs_metadata_sec
                    ),
                    0.0,
                )
                obs_denom = max(obs_enum_other_sec, 1.0e-12)
                _report_log(
                    f"[GREEDY][load={float(load):.1f}] obs_enum_other_sub_breakdown(1-{ep_idx}) "
                    f"| obs_enum_other(obs_build-local-global-mask-select-greedy_ref)="
                    f"{obs_enum_other_sec / max(obs_total_sec, 1.0e-12):.3%} of obs_build "
                    f"| obs(agent_loop/candidate_enum/obs_pack/flatten_concat/metadata/other)="
                    f"{obs_agent_loop_sec / obs_denom:.3%}/"
                    f"{obs_candidate_enum_sec / obs_denom:.3%}/"
                    f"{obs_pack_sec / obs_denom:.3%}/"
                    f"{obs_flatten_concat_sec / obs_denom:.3%}/"
                    f"{obs_metadata_sec / obs_denom:.3%}/"
                    f"{obs_other_sec / obs_denom:.3%}"
                )
                _report_log(
                    f"[GREEDY][load={float(load):.1f}] episodes 1-{ep_idx} summary "
                    f"| users_mean(urllc/embb)={mean_urllc_users:.2f}/{mean_embb_users:.2f} "
                    f"| embb:urllc={embb_to_urllc_ratio:.3f} "
                    f"| arrivals_mean={float(np.mean(running_arrivals)):.2f} "
                    f"| admitted_mean={float(np.mean(running_admitted)):.2f} "
                    f"| admission_mean={float(np.mean(running_admission_ratio)):.4f} "
                    f"| embb_rate_mean={float(np.mean(running_embb_rate_mbps)):.3f} Mbps "
                    f"| urllc_tp_mean={float(np.mean(running_urllc_tp_mbps)):.3f} Mbps "
                    f"| budget_used_ratio_mean={float(np.mean(running_budget_used_ratio)):.4f} "
                    f"| budget_used_ratio_max={float(np.max(running_budget_used_ratio)):.4f} "
                    f"| hf_rescan_ratio={float(np.mean(running_hf_rescan_used)):.4f} "
                    f"| baseline_cache_hit_ratio={float(np.mean(running_baseline_cache_hit_ratio)):.4f} "
                    f"| virtual_slot_reset_count_per_episode={float(np.mean(running_virtual_slot_reset_count)):.2f}"
                )
                _report_log(
                    f"[GREEDY][load={float(load):.1f}] ssot_invariant(1-{ep_idx}) "
                    f"| raw/admissible/evaluated/feasible/selected="
                    f"{float(np.mean(running_hf_raw_count)):.2f}/"
                    f"{float(np.mean(running_hf_admissible_count)):.2f}/"
                    f"{float(np.mean(running_hf_evaluated_count)):.2f}/"
                    f"{float(np.mean(running_hf_feasible_count)):.2f}/"
                    f"{float(np.mean(running_hf_selected_count)):.2f} "
                    f"| reject_by_gate(association/queue_active/mode/owner/rb_local)="
                    f"{float(np.mean(running_hf_reject_gate_assoc)):.2f}/"
                    f"{float(np.mean(running_hf_reject_gate_queue)):.2f}/"
                    f"{float(np.mean(running_hf_reject_gate_mode)):.2f}/"
                    f"{float(np.mean(running_hf_reject_gate_owner)):.2f}/"
                    f"{float(np.mean(running_hf_reject_gate_rb_local)):.2f}"
                )
                _report_log(
                    f"[GREEDY][load={float(load):.1f}] ssot_reason_breakdown(1-{ep_idx}) "
                    f"| mode(overlay_rel/overlay_sic/gain_ratio/owner_pool_missing/owner_unresolved_due_to_mode_fail/puncture_rel/puncture_sic/puncture_owner_missing)="
                    f"{float(np.mean(running_hf_mode_overlay_rel_fail)):.2f}/"
                    f"{float(np.mean(running_hf_mode_overlay_sic_fail)):.2f}/"
                    f"{float(np.mean(running_hf_mode_gain_ratio_fail)):.2f}/"
                    f"{float(np.mean(running_hf_mode_owner_pool_missing)):.2f}/"
                    f"{float(np.mean(running_hf_mode_owner_unresolved_due_to_mode_fail)):.2f}/"
                    f"{float(np.mean(running_hf_mode_puncture_rel_fail)):.2f}/"
                    f"{float(np.mean(running_hf_mode_puncture_sic_fail)):.2f}/"
                    f"{float(np.mean(running_hf_mode_puncture_owner_missing)):.2f} "
                    f"| owner(missing/mismatch)="
                    f"{float(np.mean(running_hf_owner_missing)):.2f}/"
                    f"{float(np.mean(running_hf_owner_mismatch)):.2f}"
                )
                _report_log(
                    f"[GREEDY][load={float(load):.1f}] pre_mode_reason_breakdown(1-{ep_idx}) "
                    f"| raw_pair={float(np.mean(running_pre_mode_raw_pair)):.2f} "
                    f"| mode(overlay_rel/overlay_sic/gain_ratio/owner_pool_missing/owner_unresolved_due_to_mode_fail/puncture_rel/puncture_sic/puncture_owner_missing)="
                    f"{float(np.mean(running_pre_mode_overlay_rel)):.2f}/"
                    f"{float(np.mean(running_pre_mode_overlay_sic)):.2f}/"
                    f"{float(np.mean(running_pre_mode_gain_ratio)):.2f}/"
                    f"{float(np.mean(running_pre_mode_owner_pool_missing)):.2f}/"
                    f"{float(np.mean(running_pre_mode_owner_unresolved_due_to_mode_fail)):.2f}/"
                    f"{float(np.mean(running_pre_mode_puncture_rel)):.2f}/"
                    f"{float(np.mean(running_pre_mode_puncture_sic)):.2f}/"
                    f"{float(np.mean(running_pre_mode_puncture_owner_missing)):.2f} "
                    f"| owner(missing/mismatch)="
                    f"{float(np.mean(running_pre_mode_owner_missing)):.2f}/"
                    f"{float(np.mean(running_pre_mode_owner_mismatch)):.2f}"
                )
                _report_log(
                    f"[GREEDY][load={float(load):.1f}] sic_pairing_prior_trace(1-{ep_idx}) "
                    f"| sic_prior_owner_total_per_decision={float(np.mean(running_sic_prior_owner_total_per_decision)):.2f} "
                    f"| sic_prior_pair_block_ratio={float(np.mean(running_sic_prior_pair_block_ratio)):.3f} "
                    f"| sic_prior_saved_mode_fail_ratio={float(np.mean(running_sic_prior_saved_mode_fail_ratio)):.3f}"
                )
                _report_log(
                    f"[GREEDY][load={float(load):.1f}] gate_margin_breakdown(1-{ep_idx}) "
                    f"| target(rel/sic_db)="
                    f"{float(np.mean(running_hf_gate_target_reliability)):.6f}/"
                    f"{float(np.mean(running_hf_gate_target_sic_db)):.3f} "
                    f"| fail_margin_mean(overlay_rel/overlay_sic_db/puncture_rel)="
                    f"{float(np.mean(running_hf_gate_overlay_rel_fail_margin)):.6f}/"
                    f"{float(np.mean(running_hf_gate_overlay_sic_fail_margin_db)):.3f}/"
                    f"{float(np.mean(running_hf_gate_puncture_rel_fail_margin)):.6f} "
                    f"| fail_snir_db_mean(overlay_rel/overlay_sic_post/puncture_rel)="
                    f"{float(np.mean(running_hf_gate_overlay_rel_fail_snir_db)):.3f}/"
                    f"{float(np.mean(running_hf_gate_overlay_sic_fail_post_sic_db)):.3f}/"
                    f"{float(np.mean(running_hf_gate_puncture_rel_fail_snir_db)):.3f} "
                    f"| selected_minus_admitted_rel_mean={float(np.mean(running_hf_selected_minus_admitted_reliability)):.6f}"
                )
                _report_log(
                    f"[GREEDY][load={float(load):.1f}] overlay_sic_trace(1-{ep_idx}) "
                    f"| pre_sinr_db_mean={float(np.mean(running_hf_overlay_sic_trace_pre_sinr_db)):.3f} "
                    f"| post_sinr_db_mean={float(np.mean(running_hf_overlay_sic_trace_post_sinr_db)):.3f} "
                    f"| noise_power_mean={float(np.mean(running_hf_overlay_sic_trace_noise_power)):.6e} "
                    f"| intercell_interference_mean={float(np.mean(running_hf_overlay_sic_trace_intercell_interference)):.6e} "
                    f"| local_interference_mean={float(np.mean(running_hf_overlay_sic_trace_local_interference)):.6e} "
                    f"| residual_sic_interference_mean={float(np.mean(running_hf_overlay_sic_trace_residual_interference)):.6e} "
                    f"| sic_residual_ratio={float(np.mean(running_hf_overlay_sic_trace_residual_ratio)):.6f} "
                    f"| threshold_db={float(np.mean(running_hf_gate_target_sic_db)):.3f} "
                    f"| margin_db_mean={float(np.mean(running_hf_gate_overlay_sic_fail_margin_db)):.3f} "
                    f"| pre_sinr_bucket(raw <-10/-10~-6/-6~-2/>=-2)="
                    f"{float(np.mean(running_hf_overlay_presinr_raw_lt_m10)):.2f}/"
                    f"{float(np.mean(running_hf_overlay_presinr_raw_m10_m6)):.2f}/"
                    f"{float(np.mean(running_hf_overlay_presinr_raw_m6_m2)):.2f}/"
                    f"{float(np.mean(running_hf_overlay_presinr_raw_ge_m2)):.2f} "
                    f"| low_pre_sinr_ratio(kept/eval)="
                    f"{float(np.mean(running_hf_overlay_presinr_kept_low_ratio)):.3f}/"
                    f"{float(np.mean(running_hf_overlay_presinr_eval_low_ratio)):.3f}"
                )
                _report_log(
                    f"[GREEDY][load={float(load):.1f}] quality_tier_trace(1-{ep_idx}) "
                    f"| priority_enabled={float(np.mean(running_hf_quality_priority_enabled)):.3f} "
                    f"| raw(high/borderline/risk)="
                    f"{float(np.mean(running_hf_quality_raw_high)):.2f}/"
                    f"{float(np.mean(running_hf_quality_raw_borderline)):.2f}/"
                    f"{float(np.mean(running_hf_quality_raw_risk)):.2f} "
                    f"| kept(high/borderline/risk)="
                    f"{float(np.mean(running_hf_quality_kept_high)):.2f}/"
                    f"{float(np.mean(running_hf_quality_kept_borderline)):.2f}/"
                    f"{float(np.mean(running_hf_quality_kept_risk)):.2f} "
                    f"| eval(high/borderline/risk)="
                    f"{float(np.mean(running_hf_quality_eval_high)):.2f}/"
                    f"{float(np.mean(running_hf_quality_eval_borderline)):.2f}/"
                    f"{float(np.mean(running_hf_quality_eval_risk)):.2f} "
                    f"| selected(high/borderline/risk)="
                    f"{float(np.mean(running_hf_quality_selected_high)):.2f}/"
                    f"{float(np.mean(running_hf_quality_selected_borderline)):.2f}/"
                    f"{float(np.mean(running_hf_quality_selected_risk)):.2f}"
                )
                _report_log(
                    f"[GREEDY][load={float(load):.1f}] overlay_blacklist_trace(1-{ep_idx}) "
                    f"| overlay_blacklist_cell_ratio={float(np.mean(running_hf_overlay_blacklist_cell_ratio)):.3f} "
                    f"| overlay_blacklist_candidate_block_ratio={float(np.mean(running_hf_overlay_blacklist_candidate_block_ratio)):.3f} "
                    f"| overlay_blacklist_saved_mode_fail_ratio={float(np.mean(running_hf_overlay_blacklist_saved_mode_fail_ratio)):.3f}"
                )
                _report_log(
                    f"[GREEDY][load={float(load):.1f}] overlay_rb_reservation_trace(1-{ep_idx}) "
                    f"| enabled={float(np.mean(running_hf_overlay_rb_reservation_enabled)):.3f} "
                    f"| hard_block={float(np.mean(running_hf_overlay_rb_hard_block_enabled)):.3f} "
                    f"| soft_pref={float(np.mean(running_hf_overlay_rb_soft_preference_enabled)):.3f} "
                    f"| ratio={float(np.mean(running_hf_overlay_rb_reservation_ratio)):.3f} "
                    f"| reserved_count={float(np.mean(running_hf_overlay_rb_reserved_count)):.2f} "
                    f"| reserved_cell_ratio={float(np.mean(running_hf_overlay_rb_reserved_cell_ratio)):.3f} "
                    f"| mode_block_ratio={float(np.mean(running_hf_overlay_rb_mode_block_ratio)):.3f} "
                    f"| soft_pref_applied_ratio={float(np.mean(running_hf_overlay_rb_soft_preference_applied_ratio)):.3f}"
                )
                _report_log(
                    f"[GREEDY][load={float(load):.1f}] guardrail_trace(1-{ep_idx}) "
                    f"| enabled={float(np.mean(running_guardrail_enabled)):.3f} "
                    f"| resample_count={float(np.mean(running_guardrail_resample_count)):.2f} "
                    f"| pass_ratio={float(np.mean(running_guardrail_pass_ratio)):.3f} "
                    f"| reject_reason(overlay/minrate/imbalance)="
                    f"{float(np.mean(running_guardrail_reject_overlay)):.2f}/"
                    f"{float(np.mean(running_guardrail_reject_embb_minrate)):.2f}/"
                    f"{float(np.mean(running_guardrail_reject_uav_imbalance)):.2f} "
                    f"| actual(overlay/minrate/imbalance)="
                    f"{float(np.mean(running_guardrail_actual_overlay)):.3f}/"
                    f"{float(np.mean(running_guardrail_actual_minrate)):.3f}/"
                    f"{float(np.mean(running_guardrail_actual_imbalance)):.3f} "
                    f"| threshold(overlay/minrate/imbalance)="
                    f"{float(np.mean(running_guardrail_threshold_overlay)):.3f}/"
                    f"{float(np.mean(running_guardrail_threshold_minrate)):.3f}/"
                    f"{float(np.mean(running_guardrail_threshold_imbalance)):.3f}"
                )
        # Always emit one load-end breakdown, even when episodes_per_load < 10.
        total_episode_sec = max(float(np.sum(running_episode_sec)), 1.0e-12)
        total_action_select_sec = float(np.sum(running_action_select_sec))
        total_hf_prefilter_sec = float(np.sum(running_hf_prefilter_sec))
        total_hf_eval_sec = float(np.sum(running_hf_eval_sec))
        total_hf_fastpath_sec = float(np.sum(running_hf_fastpath_sec))
        total_action_other_sec = max(
            total_action_select_sec - total_hf_prefilter_sec - total_hf_eval_sec - total_hf_fastpath_sec,
            0.0,
        )
        ep_range_label = f"1-{len(episodes)}"
        _report_log(
            f"[GREEDY][load={float(load):.1f}] time_breakdown({ep_range_label}) "
            f"| reset_total={float(np.sum(running_reset_total_sec)) / total_episode_sec:.3%} "
            f"| action_select={total_action_select_sec / total_episode_sec:.3%} "
            f"| action_resolve={float(np.sum(running_action_resolve_sec)) / total_episode_sec:.3%} "
            f"| env_step={float(np.sum(running_env_step_sec)) / total_episode_sec:.3%} "
            f"| residual_other={float(np.sum(running_other_sec)) / total_episode_sec:.3%} "
            f"| baseline_cache_hit_ratio={float(np.mean(running_baseline_cache_hit_ratio)):.4f} "
            f"| virtual_slot_reset_count_per_episode={float(np.mean(running_virtual_slot_reset_count)):.2f}"
        )
        _report_log(
            f"[GREEDY][load={float(load):.1f}] time_breakdown_detail({ep_range_label}) "
            f"| reset(state/slot_ctx/arrival/greedy_ref/build_obs/misc)="
            f"{float(np.sum(running_reset_state_sec)) / total_episode_sec:.3%}/"
            f"{float(np.sum(running_prepare_slot_ctx_sec)) / total_episode_sec:.3%}/"
            f"{float(np.sum(running_arrival_gen_sec)) / total_episode_sec:.3%}/"
            f"{float(np.sum(running_reset_greedy_ref_sec)) / total_episode_sec:.3%}/"
            f"{float(np.sum(running_reset_build_obs_sec)) / total_episode_sec:.3%}/"
            f"{float(np.sum(running_reset_misc_sec)) / total_episode_sec:.3%} "
            f"| action(hf_prefilter/hf_eval/hf_fastpath/other)="
            f"{total_hf_prefilter_sec / total_episode_sec:.3%}/"
            f"{total_hf_eval_sec / total_episode_sec:.3%}/"
            f"{total_hf_fastpath_sec / total_episode_sec:.3%}/"
            f"{total_action_other_sec / total_episode_sec:.3%}"
        )
        _report_log(
            f"[GREEDY][load={float(load):.1f}] env_step_breakdown({ep_range_label}) "
            f"| step(obs_build/apply/state/reward/interference/rate)="
            f"{float(np.sum(running_step_profile_obs_build_sec)) / total_episode_sec:.3%}/"
            f"{float(np.sum(running_step_profile_apply_action_sec)) / total_episode_sec:.3%}/"
            f"{float(np.sum(running_step_profile_state_update_sec)) / total_episode_sec:.3%}/"
            f"{float(np.sum(running_step_profile_reward_compute_sec)) / total_episode_sec:.3%}/"
            f"{float(np.sum(running_step_profile_interference_sec)) / total_episode_sec:.3%}/"
            f"{float(np.sum(running_step_profile_rate_sec)) / total_episode_sec:.3%} "
            f"| calls(interference/rate)="
            f"{float(np.sum(running_step_profile_interference_calls)):.0f}/"
            f"{float(np.sum(running_step_profile_rate_calls)):.0f}"
            f" | cache_hits(interference)="
            f"{float(np.sum(running_step_profile_interference_cache_hit_calls)):.0f}"
        )
        reward_total_sec = float(np.sum(running_step_profile_reward_compute_sec))
        reward_delta_sec = float(np.sum(running_step_profile_reward_delta_sec))
        reward_fullscan_sec = float(np.sum(running_step_profile_reward_fullscan_sec))
        reward_other_sec = max(reward_total_sec - reward_delta_sec - reward_fullscan_sec, 0.0)
        reward_denom = max(reward_total_sec, 1.0e-12)
        _report_log(
            f"[GREEDY][load={float(load):.1f}] reward_sub_breakdown({ep_range_label}) "
            f"| reward(delta/fullscan/other)="
            f"{reward_delta_sec / reward_denom:.3%}/"
            f"{reward_fullscan_sec / reward_denom:.3%}/"
            f"{reward_other_sec / reward_denom:.3%}"
        )
        obs_total_sec = float(np.sum(running_step_profile_obs_build_sec))
        obs_select_sec = float(np.sum(running_step_profile_obs_select_sec))
        obs_greedy_ref_sec = float(np.sum(running_step_profile_obs_greedy_ref_sec))
        obs_local_sec = float(np.sum(running_step_profile_obs_local_sec))
        obs_global_sec = float(np.sum(running_step_profile_obs_global_sec))
        obs_mask_sec = float(np.sum(running_step_profile_obs_mask_sec))
        obs_enum_other_sec = max(
            obs_total_sec - (
                obs_select_sec
                + obs_greedy_ref_sec
                + obs_local_sec
                + obs_global_sec
                + obs_mask_sec
            ),
            0.0,
        )
        obs_candidate_enum_sec = float(np.sum(running_step_profile_obs_candidate_enum_sec))
        obs_agent_loop_sec = float(np.sum(running_step_profile_obs_agent_loop_sec))
        obs_pack_sec = float(np.sum(running_step_profile_obs_pack_sec))
        obs_flatten_concat_sec = float(np.sum(running_step_profile_obs_flatten_concat_sec))
        obs_metadata_sec = float(np.sum(running_step_profile_obs_meta_sec))
        obs_other_sec = max(
            obs_enum_other_sec - (
                obs_agent_loop_sec
                + obs_candidate_enum_sec
                + obs_pack_sec
                + obs_flatten_concat_sec
                + obs_metadata_sec
            ),
            0.0,
        )
        obs_denom = max(obs_enum_other_sec, 1.0e-12)
        _report_log(
            f"[GREEDY][load={float(load):.1f}] obs_enum_other_sub_breakdown({ep_range_label}) "
            f"| obs_enum_other(obs_build-local-global-mask-select-greedy_ref)="
            f"{obs_enum_other_sec / max(obs_total_sec, 1.0e-12):.3%} of obs_build "
            f"| obs(agent_loop/candidate_enum/obs_pack/flatten_concat/metadata/other)="
            f"{obs_agent_loop_sec / obs_denom:.3%}/"
            f"{obs_candidate_enum_sec / obs_denom:.3%}/"
            f"{obs_pack_sec / obs_denom:.3%}/"
            f"{obs_flatten_concat_sec / obs_denom:.3%}/"
            f"{obs_metadata_sec / obs_denom:.3%}/"
            f"{obs_other_sec / obs_denom:.3%}"
        )
        _report_log(
            f"[GREEDY][load={float(load):.1f}] ssot_invariant({ep_range_label}) "
            f"| raw/admissible/evaluated/feasible/selected="
            f"{float(np.mean(running_hf_raw_count)):.2f}/"
            f"{float(np.mean(running_hf_admissible_count)):.2f}/"
            f"{float(np.mean(running_hf_evaluated_count)):.2f}/"
            f"{float(np.mean(running_hf_feasible_count)):.2f}/"
            f"{float(np.mean(running_hf_selected_count)):.2f} "
            f"| reject_by_gate(association/queue_active/mode/owner/rb_local)="
            f"{float(np.mean(running_hf_reject_gate_assoc)):.2f}/"
            f"{float(np.mean(running_hf_reject_gate_queue)):.2f}/"
            f"{float(np.mean(running_hf_reject_gate_mode)):.2f}/"
            f"{float(np.mean(running_hf_reject_gate_owner)):.2f}/"
            f"{float(np.mean(running_hf_reject_gate_rb_local)):.2f}"
        )
        _report_log(
            f"[GREEDY][load={float(load):.1f}] ssot_reason_breakdown({ep_range_label}) "
            f"| mode(overlay_rel/overlay_sic/gain_ratio/owner_pool_missing/owner_unresolved_due_to_mode_fail/puncture_rel/puncture_sic/puncture_owner_missing)="
            f"{float(np.mean(running_hf_mode_overlay_rel_fail)):.2f}/"
            f"{float(np.mean(running_hf_mode_overlay_sic_fail)):.2f}/"
            f"{float(np.mean(running_hf_mode_gain_ratio_fail)):.2f}/"
            f"{float(np.mean(running_hf_mode_owner_pool_missing)):.2f}/"
            f"{float(np.mean(running_hf_mode_owner_unresolved_due_to_mode_fail)):.2f}/"
            f"{float(np.mean(running_hf_mode_puncture_rel_fail)):.2f}/"
            f"{float(np.mean(running_hf_mode_puncture_sic_fail)):.2f}/"
            f"{float(np.mean(running_hf_mode_puncture_owner_missing)):.2f} "
            f"| owner(missing/mismatch)="
            f"{float(np.mean(running_hf_owner_missing)):.2f}/"
            f"{float(np.mean(running_hf_owner_mismatch)):.2f}"
        )
        _report_log(
            f"[GREEDY][load={float(load):.1f}] pre_mode_reason_breakdown({ep_range_label}) "
            f"| raw_pair={float(np.mean(running_pre_mode_raw_pair)):.2f} "
            f"| mode(overlay_rel/overlay_sic/gain_ratio/owner_pool_missing/owner_unresolved_due_to_mode_fail/puncture_rel/puncture_sic/puncture_owner_missing)="
            f"{float(np.mean(running_pre_mode_overlay_rel)):.2f}/"
            f"{float(np.mean(running_pre_mode_overlay_sic)):.2f}/"
            f"{float(np.mean(running_pre_mode_gain_ratio)):.2f}/"
            f"{float(np.mean(running_pre_mode_owner_pool_missing)):.2f}/"
            f"{float(np.mean(running_pre_mode_owner_unresolved_due_to_mode_fail)):.2f}/"
            f"{float(np.mean(running_pre_mode_puncture_rel)):.2f}/"
            f"{float(np.mean(running_pre_mode_puncture_sic)):.2f}/"
            f"{float(np.mean(running_pre_mode_puncture_owner_missing)):.2f} "
            f"| owner(missing/mismatch)="
            f"{float(np.mean(running_pre_mode_owner_missing)):.2f}/"
            f"{float(np.mean(running_pre_mode_owner_mismatch)):.2f}"
        )
        _report_log(
            f"[GREEDY][load={float(load):.1f}] sic_pairing_prior_trace({ep_range_label}) "
            f"| sic_prior_owner_total_per_decision={float(np.mean(running_sic_prior_owner_total_per_decision)):.2f} "
            f"| sic_prior_pair_block_ratio={float(np.mean(running_sic_prior_pair_block_ratio)):.3f} "
            f"| sic_prior_saved_mode_fail_ratio={float(np.mean(running_sic_prior_saved_mode_fail_ratio)):.3f}"
        )
        _report_log(
            f"[GREEDY][load={float(load):.1f}] gate_margin_breakdown({ep_range_label}) "
            f"| target(rel/sic_db)="
            f"{float(np.mean(running_hf_gate_target_reliability)):.6f}/"
            f"{float(np.mean(running_hf_gate_target_sic_db)):.3f} "
            f"| fail_margin_mean(overlay_rel/overlay_sic_db/puncture_rel)="
            f"{float(np.mean(running_hf_gate_overlay_rel_fail_margin)):.6f}/"
            f"{float(np.mean(running_hf_gate_overlay_sic_fail_margin_db)):.3f}/"
            f"{float(np.mean(running_hf_gate_puncture_rel_fail_margin)):.6f} "
            f"| fail_snir_db_mean(overlay_rel/overlay_sic_post/puncture_rel)="
            f"{float(np.mean(running_hf_gate_overlay_rel_fail_snir_db)):.3f}/"
            f"{float(np.mean(running_hf_gate_overlay_sic_fail_post_sic_db)):.3f}/"
            f"{float(np.mean(running_hf_gate_puncture_rel_fail_snir_db)):.3f} "
            f"| selected_minus_admitted_rel_mean={float(np.mean(running_hf_selected_minus_admitted_reliability)):.6f}"
        )
        _report_log(
            f"[GREEDY][load={float(load):.1f}] overlay_sic_trace({ep_range_label}) "
            f"| pre_sinr_db_mean={float(np.mean(running_hf_overlay_sic_trace_pre_sinr_db)):.3f} "
            f"| post_sinr_db_mean={float(np.mean(running_hf_overlay_sic_trace_post_sinr_db)):.3f} "
            f"| noise_power_mean={float(np.mean(running_hf_overlay_sic_trace_noise_power)):.6e} "
            f"| intercell_interference_mean={float(np.mean(running_hf_overlay_sic_trace_intercell_interference)):.6e} "
            f"| local_interference_mean={float(np.mean(running_hf_overlay_sic_trace_local_interference)):.6e} "
            f"| residual_sic_interference_mean={float(np.mean(running_hf_overlay_sic_trace_residual_interference)):.6e} "
            f"| sic_residual_ratio={float(np.mean(running_hf_overlay_sic_trace_residual_ratio)):.6f} "
            f"| threshold_db={float(np.mean(running_hf_gate_target_sic_db)):.3f} "
            f"| margin_db_mean={float(np.mean(running_hf_gate_overlay_sic_fail_margin_db)):.3f} "
            f"| pre_sinr_bucket(raw <-10/-10~-6/-6~-2/>=-2)="
            f"{float(np.mean(running_hf_overlay_presinr_raw_lt_m10)):.2f}/"
            f"{float(np.mean(running_hf_overlay_presinr_raw_m10_m6)):.2f}/"
            f"{float(np.mean(running_hf_overlay_presinr_raw_m6_m2)):.2f}/"
            f"{float(np.mean(running_hf_overlay_presinr_raw_ge_m2)):.2f} "
            f"| low_pre_sinr_ratio(kept/eval)="
            f"{float(np.mean(running_hf_overlay_presinr_kept_low_ratio)):.3f}/"
            f"{float(np.mean(running_hf_overlay_presinr_eval_low_ratio)):.3f}"
        )
        _report_log(
            f"[GREEDY][load={float(load):.1f}] quality_tier_trace({ep_range_label}) "
            f"| priority_enabled={float(np.mean(running_hf_quality_priority_enabled)):.3f} "
            f"| raw(high/borderline/risk)="
            f"{float(np.mean(running_hf_quality_raw_high)):.2f}/"
            f"{float(np.mean(running_hf_quality_raw_borderline)):.2f}/"
            f"{float(np.mean(running_hf_quality_raw_risk)):.2f} "
            f"| kept(high/borderline/risk)="
            f"{float(np.mean(running_hf_quality_kept_high)):.2f}/"
            f"{float(np.mean(running_hf_quality_kept_borderline)):.2f}/"
            f"{float(np.mean(running_hf_quality_kept_risk)):.2f} "
            f"| eval(high/borderline/risk)="
            f"{float(np.mean(running_hf_quality_eval_high)):.2f}/"
            f"{float(np.mean(running_hf_quality_eval_borderline)):.2f}/"
            f"{float(np.mean(running_hf_quality_eval_risk)):.2f} "
            f"| selected(high/borderline/risk)="
            f"{float(np.mean(running_hf_quality_selected_high)):.2f}/"
            f"{float(np.mean(running_hf_quality_selected_borderline)):.2f}/"
            f"{float(np.mean(running_hf_quality_selected_risk)):.2f}"
        )
        _report_log(
            f"[GREEDY][load={float(load):.1f}] overlay_blacklist_trace({ep_range_label}) "
            f"| overlay_blacklist_cell_ratio={float(np.mean(running_hf_overlay_blacklist_cell_ratio)):.3f} "
            f"| overlay_blacklist_candidate_block_ratio={float(np.mean(running_hf_overlay_blacklist_candidate_block_ratio)):.3f} "
            f"| overlay_blacklist_saved_mode_fail_ratio={float(np.mean(running_hf_overlay_blacklist_saved_mode_fail_ratio)):.3f}"
        )
        _report_log(
            f"[GREEDY][load={float(load):.1f}] overlay_rb_reservation_trace({ep_range_label}) "
            f"| enabled={float(np.mean(running_hf_overlay_rb_reservation_enabled)):.3f} "
            f"| hard_block={float(np.mean(running_hf_overlay_rb_hard_block_enabled)):.3f} "
            f"| soft_pref={float(np.mean(running_hf_overlay_rb_soft_preference_enabled)):.3f} "
            f"| ratio={float(np.mean(running_hf_overlay_rb_reservation_ratio)):.3f} "
            f"| reserved_count={float(np.mean(running_hf_overlay_rb_reserved_count)):.2f} "
            f"| reserved_cell_ratio={float(np.mean(running_hf_overlay_rb_reserved_cell_ratio)):.3f} "
            f"| mode_block_ratio={float(np.mean(running_hf_overlay_rb_mode_block_ratio)):.3f} "
            f"| soft_pref_applied_ratio={float(np.mean(running_hf_overlay_rb_soft_preference_applied_ratio)):.3f}"
        )
        _report_log(
            f"[GREEDY][load={float(load):.1f}] guardrail_trace({ep_range_label}) "
            f"| enabled={float(np.mean(running_guardrail_enabled)):.3f} "
            f"| resample_count={float(np.mean(running_guardrail_resample_count)):.2f} "
            f"| pass_ratio={float(np.mean(running_guardrail_pass_ratio)):.3f} "
            f"| reject_reason(overlay/minrate/imbalance)="
            f"{float(np.mean(running_guardrail_reject_overlay)):.2f}/"
            f"{float(np.mean(running_guardrail_reject_embb_minrate)):.2f}/"
            f"{float(np.mean(running_guardrail_reject_uav_imbalance)):.2f} "
            f"| actual(overlay/minrate/imbalance)="
            f"{float(np.mean(running_guardrail_actual_overlay)):.3f}/"
            f"{float(np.mean(running_guardrail_actual_minrate)):.3f}/"
            f"{float(np.mean(running_guardrail_actual_imbalance)):.3f} "
            f"| threshold(overlay/minrate/imbalance)="
            f"{float(np.mean(running_guardrail_threshold_overlay)):.3f}/"
            f"{float(np.mean(running_guardrail_threshold_minrate)):.3f}/"
            f"{float(np.mean(running_guardrail_threshold_imbalance)):.3f}"
        )
        _guardrail_total = int(len(running_guardrail_pass_ratio))
        _guardrail_pass = int(np.count_nonzero(np.asarray(running_guardrail_pass_ratio, dtype=float) > 0.5))
        _guardrail_reject = int(max(_guardrail_total - _guardrail_pass, 0))
        _guardrail_reject_overlay = int(np.count_nonzero(np.asarray(running_guardrail_reject_overlay, dtype=float) > 0.5))
        _guardrail_reject_minrate = int(np.count_nonzero(np.asarray(running_guardrail_reject_embb_minrate, dtype=float) > 0.5))
        _guardrail_reject_imbalance = int(np.count_nonzero(np.asarray(running_guardrail_reject_uav_imbalance, dtype=float) > 0.5))
        _report_log(
            f"[GREEDY][load={float(load):.1f}] feasible_regime_audit({ep_range_label}) "
            f"| episodes={_guardrail_total} "
            f"| pass/reject={_guardrail_pass}/{_guardrail_reject} "
            f"| reject_reason_count(overlay/minrate/imbalance)="
            f"{_guardrail_reject_overlay}/{_guardrail_reject_minrate}/{_guardrail_reject_imbalance}"
        )
        def _same_ratio_str(vals: List[str]) -> float:
            if not vals:
                return 1.0
            ref = vals[0]
            return float(np.mean([1.0 if str(v) == str(ref) else 0.0 for v in vals]))
        _same_channel_ratio = _same_ratio_str(running_same_channel_hash)
        _same_assoc_ratio = _same_ratio_str(running_same_assoc_hash)
        _same_user_pool_ratio = _same_ratio_str(running_same_user_pool_hash)
        _mother_id_ref = running_mother_topology_id[0] if running_mother_topology_id else ""
        _mother_seed_ref = float(running_mother_topology_seed[0]) if running_mother_topology_seed else 0.0
        _channel_hash_ref = running_same_channel_hash[0] if running_same_channel_hash else ""
        _assoc_hash_ref = running_same_assoc_hash[0] if running_same_assoc_hash else ""
        _user_pool_hash_ref = running_same_user_pool_hash[0] if running_same_user_pool_hash else ""
        _subset_hash_ref = running_mix_user_subset_hash[0] if running_mix_user_subset_hash else ""
        _embb_subset_hash_ref = running_embb_subset_hash[0] if running_embb_subset_hash else ""
        _same_feasible_graph_hash_ref = running_same_feasible_graph_hash[0] if running_same_feasible_graph_hash else ""
        _feasible_graph_id_ref = running_feasible_graph_id[0] if running_feasible_graph_id else ""
        _overlay_graph_hash_ref = running_overlay_graph_hash[0] if running_overlay_graph_hash else ""
        _channel_matrix_hash_ref = running_channel_matrix_hash[0] if running_channel_matrix_hash else ""
        _pathloss_hash_ref = running_pathloss_hash[0] if running_pathloss_hash else ""
        _shadowing_hash_ref = running_shadowing_hash[0] if running_shadowing_hash else ""
        _sic_order_hash_ref = running_sic_order_hash[0] if running_sic_order_hash else ""
        _repair_sequence_hash_ref = running_repair_sequence_hash[0] if running_repair_sequence_hash else ""
        _guardrail_pass_mean = float(np.mean(running_guardrail_pass_flag)) if running_guardrail_pass_flag else 0.0
        _report_log(
            f"[GREEDY][load={float(load):.1f}] mix_comparability_audit({ep_range_label}) "
            f"| mother_topology_id={_mother_id_ref} "
            f"| mother_topology_seed={_mother_seed_ref:.0f} "
            f"| same_channel_hash={_channel_hash_ref} "
            f"| same_assoc_hash={_assoc_hash_ref} "
            f"| same_user_pool_hash={_user_pool_hash_ref} "
            f"| mix_user_subset_hash={_subset_hash_ref} "
            f"| embb_subset_hash={_embb_subset_hash_ref} "
            f"| feasible_graph_id={_feasible_graph_id_ref} "
            f"| same_feasible_graph_hash={_same_feasible_graph_hash_ref} "
            f"| overlay_graph_hash={_overlay_graph_hash_ref} "
            f"| channel_matrix_hash={_channel_matrix_hash_ref} "
            f"| pathloss_hash={_pathloss_hash_ref} "
            f"| shadowing_hash={_shadowing_hash_ref} "
            f"| sic_order_hash={_sic_order_hash_ref} "
            f"| repair_sequence_hash={_repair_sequence_hash_ref} "
            f"| consistency_ratio(channel/assoc/user_pool)="
            f"{_same_channel_ratio:.3f}/{_same_assoc_ratio:.3f}/{_same_user_pool_ratio:.3f} "
            f"| guardrail_pass={_guardrail_pass_mean:.3f}"
        )
        if len(episodes) < 10:
            mean_urllc_users = float(np.mean(running_urllc_users)) if running_urllc_users else 0.0
            mean_embb_users = float(np.mean(running_embb_users)) if running_embb_users else 0.0
            embb_to_urllc_ratio = float(mean_embb_users / max(mean_urllc_users, 1.0e-9))
            _report_log(
                f"[GREEDY][load={float(load):.1f}] episodes {ep_range_label} summary "
                f"| users_mean(urllc/embb)={mean_urllc_users:.2f}/{mean_embb_users:.2f} "
                f"| embb:urllc={embb_to_urllc_ratio:.3f} "
                f"| arrivals_mean={float(np.mean(running_arrivals)):.2f} "
                f"| admitted_mean={float(np.mean(running_admitted)):.2f} "
                f"| admission_mean={float(np.mean(running_admission_ratio)):.4f} "
                f"| embb_rate_mean={float(np.mean(running_embb_rate_mbps)):.3f} Mbps "
                f"| urllc_tp_mean={float(np.mean(running_urllc_tp_mbps)):.3f} Mbps "
                f"| budget_used_ratio_mean={float(np.mean(running_budget_used_ratio)):.4f} "
                f"| budget_used_ratio_max={float(np.max(running_budget_used_ratio)):.4f} "
                f"| hf_rescan_ratio={float(np.mean(running_hf_rescan_used)):.4f} "
                f"| baseline_cache_hit_ratio={float(np.mean(running_baseline_cache_hit_ratio)):.4f} "
                f"| virtual_slot_reset_count_per_episode={float(np.mean(running_virtual_slot_reset_count)):.2f}"
            )
        # Persist per-episode samples for post-hoc plotting/comparison.
        metrics['greedy_episode_arrivals_samples'].append([float(x) for x in running_arrivals])
        metrics['greedy_episode_admitted_samples'].append([float(x) for x in running_admitted])
        metrics['greedy_episode_budget_used_ratio_samples'].append([float(x) for x in running_budget_used_ratio])
        metrics['loads'].append(float(load))
        metrics['lambda'].append(_load_to_lambda(load))
        for key in scalar_keys:
            if key == 'greedy_urllc_budget_used_ratio':
                metrics[key].append(
                    _safe_mean([
                        episode.get(
                            "greedy_urllc_budget_utilization_ratio",
                            episode.get("greedy_urllc_budget_used_ratio", 0.0),
                        )
                        for episode in episodes
                    ])
                )
            else:
                metrics[key].append(_episode_scalar_aggregate(episodes, key, default=0.0))
        for key in vector_keys:
            metrics[key].append(np.mean(np.stack([episode[key] for episode in episodes]), axis=0))
        # Mix comparability fields (string/numeric metadata).
        metrics['mother_topology_id'].append(_mother_id_ref)
        metrics['mother_topology_seed'].append(_mother_seed_ref)
        metrics['same_channel_hash'].append(_channel_hash_ref)
        metrics['same_assoc_hash'].append(_assoc_hash_ref)
        metrics['same_user_pool_hash'].append(_user_pool_hash_ref)
        metrics['mix_user_subset_hash'].append(_subset_hash_ref)
        metrics['embb_subset_hash'].append(_embb_subset_hash_ref)
        metrics['same_feasible_graph_hash'].append(_same_feasible_graph_hash_ref)
        metrics['feasible_graph_id'].append(_feasible_graph_id_ref)
        metrics['overlay_graph_hash'].append(_overlay_graph_hash_ref)
        metrics['channel_matrix_hash'].append(_channel_matrix_hash_ref)
        metrics['pathloss_hash'].append(_pathloss_hash_ref)
        metrics['shadowing_hash'].append(_shadowing_hash_ref)
        metrics['sic_order_hash'].append(_sic_order_hash_ref)
        metrics['repair_sequence_hash'].append(_repair_sequence_hash_ref)
        metrics['guardrail_pass'].append(_guardrail_pass_mean)
    _report_timing_log(
        f"run_hard_feasible_throughput_greedy_sweep loads={len(loads)} episodes_per_load={episodes_per_load} sec={perf_counter() - sweep_start:.3f}"
    )
    return metrics, representative

def run_selected_greedy_sweep(
    loads: List[float],
    episodes_per_load: int,
    cfg: SRMAPPOConfig,
    checkpoint_path: Path,
) -> Tuple[Dict, Dict, Optional[Dict]]:
    mode = _greedy_baseline_mode(cfg)
    if mode == "original_greedy_normal_v1":
        metrics, representative = run_greedy_normal_v1_sweep(loads, episodes_per_load)
        return metrics, representative, None
    if mode == "original_greedy_normal_v2":
        metrics, representative = run_greedy_normal_v2_sweep(loads, episodes_per_load)
        return metrics, representative, None
    if mode == "matched_fixed_embb":
        metrics, representative = run_matched_greedy_sweep(loads, episodes_per_load, checkpoint_path, cfg)
        return metrics, representative, None
    if mode == "throughput_feasible_oracle":
        metrics, representative = run_throughput_feasible_oracle_sweep(loads, episodes_per_load)
        return metrics, representative, None
    if mode == "hard_feasible_throughput_greedy":
        metrics, representative = run_hard_feasible_throughput_greedy_sweep(
            loads, episodes_per_load, checkpoint_path, cfg
        )
        return metrics, representative, None
    if mode == "global_frontier_greedy":
        metrics, representative = run_hard_feasible_throughput_greedy_sweep(
            loads, episodes_per_load, checkpoint_path, cfg
        )
        return metrics, representative, None
    if mode == "throughput_only_greedy":
        metrics, representative = run_throughput_only_greedy_sweep(loads, episodes_per_load, checkpoint_path, cfg)
        return metrics, representative, None
    if mode == "channel_only_greedy":
        metrics, representative = run_channel_only_greedy_sweep(loads, episodes_per_load, checkpoint_path, cfg)
        return metrics, representative, None
    if mode == "frozen_json":
        metrics, representative, payload = _load_frozen_greedy_for_report(cfg, loads, episodes_per_load)
        return metrics, representative, payload
    if mode == "myopic_throughput_greedy":
        metrics, representative = run_myopic_throughput_greedy_sweep(
            loads, episodes_per_load, checkpoint_path, cfg
        )
        return metrics, representative, None
    metrics, representative = run_greedy_sweep(loads, episodes_per_load)
    return metrics, representative, None


def run_report_greedy_bundle(
    loads: List[float],
    episodes_per_load: int,
    cfg: SRMAPPOConfig,
    checkpoint_path: Path,
) -> Tuple[Dict[str, Dict], Dict[str, Dict], Optional[Dict]]:
    bundle_start = perf_counter()
    selected_metrics, selected_rep, frozen_payload = run_selected_greedy_sweep(
        loads,
        episodes_per_load,
        cfg,
        checkpoint_path,
    )
    selected_mode = _greedy_baseline_mode(cfg)
    frozen_mode = str((frozen_payload or {}).get('greedy_baseline_mode', '') or '').strip().lower()

    metrics_bundle: Dict[str, Dict] = {}
    representative_bundle: Dict[str, Dict] = {}

    if selected_mode == "original" or frozen_mode == "original":
        metrics_bundle['original'] = selected_metrics
        representative_bundle['original'] = selected_rep
    else:
        original_metrics, original_rep = run_greedy_sweep(loads, episodes_per_load)
        metrics_bundle['original'] = original_metrics
        representative_bundle['original'] = original_rep

    if selected_mode == "original_greedy_normal_v1" or frozen_mode == "original_greedy_normal_v1":
        metrics_bundle['original_greedy_normal_v1'] = selected_metrics
        representative_bundle['original_greedy_normal_v1'] = selected_rep
    else:
        original_lite_metrics, original_lite_rep = run_greedy_normal_v1_sweep(loads, episodes_per_load)
        metrics_bundle['original_greedy_normal_v1'] = original_lite_metrics
        representative_bundle['original_greedy_normal_v1'] = original_lite_rep

    if selected_mode == "original_greedy_normal_v2" or frozen_mode == "original_greedy_normal_v2":
        metrics_bundle['original_greedy_normal_v2'] = selected_metrics
        representative_bundle['original_greedy_normal_v2'] = selected_rep
    else:
        normal_v2_metrics, normal_v2_rep = run_greedy_normal_v2_sweep(loads, episodes_per_load)
        metrics_bundle['original_greedy_normal_v2'] = normal_v2_metrics
        representative_bundle['original_greedy_normal_v2'] = normal_v2_rep

    if selected_mode == "matched_fixed_embb" or frozen_mode == "matched_fixed_embb":
        metrics_bundle['matched_fixed_embb'] = selected_metrics
        representative_bundle['matched_fixed_embb'] = selected_rep
    else:
        matched_metrics, matched_rep = run_matched_greedy_sweep(loads, episodes_per_load, checkpoint_path, cfg)
        metrics_bundle['matched_fixed_embb'] = matched_metrics
        representative_bundle['matched_fixed_embb'] = matched_rep

    if selected_mode == "throughput_feasible_oracle" or frozen_mode == "throughput_feasible_oracle":
        metrics_bundle['throughput_feasible_oracle'] = selected_metrics
        representative_bundle['throughput_feasible_oracle'] = selected_rep
    else:
        throughput_feasible_metrics, throughput_feasible_rep = run_throughput_feasible_oracle_sweep(loads, episodes_per_load)
        metrics_bundle['throughput_feasible_oracle'] = throughput_feasible_metrics
        representative_bundle['throughput_feasible_oracle'] = throughput_feasible_rep

    if selected_mode == "hard_feasible_throughput_greedy" or frozen_mode == "hard_feasible_throughput_greedy":
        metrics_bundle['hard_feasible_throughput_greedy'] = selected_metrics
        representative_bundle['hard_feasible_throughput_greedy'] = selected_rep
    else:
        hard_metrics, hard_rep = run_hard_feasible_throughput_greedy_sweep(
            loads, episodes_per_load, checkpoint_path, cfg
        )
        metrics_bundle['hard_feasible_throughput_greedy'] = hard_metrics
        representative_bundle['hard_feasible_throughput_greedy'] = hard_rep

    if selected_mode == "global_frontier_greedy" or frozen_mode == "global_frontier_greedy":
        metrics_bundle['global_frontier_greedy'] = selected_metrics
        representative_bundle['global_frontier_greedy'] = selected_rep
    else:
        global_cfg = deepcopy(cfg)
        global_cfg.training.greedy_baseline_mode = "global_frontier_greedy"
        global_cfg.training.selection_baseline_mode = "global_frontier_greedy"
        global_metrics, global_rep = run_hard_feasible_throughput_greedy_sweep(
            loads, episodes_per_load, checkpoint_path, global_cfg
        )
        metrics_bundle['global_frontier_greedy'] = global_metrics
        representative_bundle['global_frontier_greedy'] = global_rep

    if selected_mode == "myopic_throughput_greedy" or frozen_mode == "myopic_throughput_greedy":
        metrics_bundle['myopic_throughput_greedy'] = selected_metrics
        representative_bundle['myopic_throughput_greedy'] = selected_rep
    else:
        myopic_metrics, myopic_rep = run_myopic_throughput_greedy_sweep(
            loads, episodes_per_load, checkpoint_path, cfg
        )
        metrics_bundle['myopic_throughput_greedy'] = myopic_metrics
        representative_bundle['myopic_throughput_greedy'] = myopic_rep

    if selected_mode == "throughput_only_greedy" or frozen_mode == "throughput_only_greedy":
        metrics_bundle['throughput_only_greedy'] = selected_metrics
        representative_bundle['throughput_only_greedy'] = selected_rep
    else:
        throughput_only_metrics, throughput_only_rep = run_throughput_only_greedy_sweep(loads, episodes_per_load, checkpoint_path, cfg)
        metrics_bundle['throughput_only_greedy'] = throughput_only_metrics
        representative_bundle['throughput_only_greedy'] = throughput_only_rep

    if selected_mode == "rate_loss_min_greedy" or frozen_mode == "rate_loss_min_greedy":
        metrics_bundle['rate_loss_min_greedy'] = selected_metrics
        representative_bundle['rate_loss_min_greedy'] = selected_rep
    else:
        rate_loss_metrics, rate_loss_rep = run_rate_loss_min_greedy_sweep(loads, episodes_per_load, checkpoint_path, cfg)
        metrics_bundle['rate_loss_min_greedy'] = rate_loss_metrics
        representative_bundle['rate_loss_min_greedy'] = rate_loss_rep

    if selected_mode == "force_admit_minloss_greedy" or frozen_mode == "force_admit_minloss_greedy":
        metrics_bundle['force_admit_minloss_greedy'] = selected_metrics
        representative_bundle['force_admit_minloss_greedy'] = selected_rep
    else:
        force_admit_metrics, force_admit_rep = run_force_admit_minloss_greedy_sweep(loads, episodes_per_load, checkpoint_path, cfg)
        metrics_bundle['force_admit_minloss_greedy'] = force_admit_metrics
        representative_bundle['force_admit_minloss_greedy'] = force_admit_rep

    if selected_mode == "channel_only_greedy" or frozen_mode == "channel_only_greedy":
        metrics_bundle['channel_only_greedy'] = selected_metrics
        representative_bundle['channel_only_greedy'] = selected_rep
    else:
        channel_only_metrics, channel_only_rep = run_channel_only_greedy_sweep(loads, episodes_per_load, checkpoint_path, cfg)
        metrics_bundle['channel_only_greedy'] = channel_only_metrics
        representative_bundle['channel_only_greedy'] = channel_only_rep

    for baseline_key, payload in metrics_bundle.items():
        payload.update(_baseline_metadata(baseline_key))
        requires_feasible_only = False
        requires_value = payload.get('greedy_requires_feasible_admission_only', 0.0)
        if isinstance(requires_value, (list, tuple, np.ndarray)):
            arr = np.asarray(requires_value, dtype=float)
            finite = arr[np.isfinite(arr)]
            requires_feasible_only = bool(finite.size and np.max(finite) > 0.5)
        else:
            try:
                requires_feasible_only = bool(float(requires_value) > 0.5)
            except (TypeError, ValueError):
                requires_feasible_only = False
        payload.update(_baseline_narrative(
            baseline_key,
            greedy_requires_feasible_admission_only=requires_feasible_only,
        ))
        payload['method_name'] = _baseline_label(baseline_key)
    for baseline_key, rep_payload in representative_bundle.items():
        for episode_payload in rep_payload.values():
            if isinstance(episode_payload, dict):
                episode_payload.update(_baseline_metadata(baseline_key))
                episode_payload['comparison_baseline_label'] = _baseline_label(baseline_key)

    _report_timing_log(
        f"run_report_greedy_bundle loads={len(loads)} episodes_per_load={episodes_per_load} sec={perf_counter() - bundle_start:.3f}"
    )
    return metrics_bundle, representative_bundle, frozen_payload


def _baseline_style(mode: str | None) -> tuple[str, str]:
    style_map = {
        'original': ('tab:blue', 'o'),
        'original_greedy_normal_v1': ('tab:cyan', 'P'),
        'original_greedy_normal_v2': ('tab:pink', 'X'),
        'myopic_throughput_greedy': ('tab:brown', 'h'),
        'hard_feasible_throughput_greedy': ('saddlebrown', 'H'),
        'global_frontier_greedy': ('darkorange', 's'),
        'matched_fixed_embb': ('tab:green', '^'),
        'throughput_feasible_oracle': ('tab:red', '*'),
        'throughput_only_greedy': ('tab:purple', 'D'),
        'rate_loss_min_greedy': ('tab:olive', '>'),
        'force_admit_minloss_greedy': ('tab:red', 'P'),
        'channel_only_greedy': ('tab:gray', 'v'),
    }
    return style_map.get(_normalize_baseline_mode(mode), ('tab:purple', 'D'))


def _baseline_requires_feasible_only(payload: Optional[Dict]) -> bool:
    if not payload:
        return False
    requires_value = payload.get('greedy_requires_feasible_admission_only', 0.0)
    if isinstance(requires_value, (list, tuple, np.ndarray)):
        arr = np.asarray(requires_value, dtype=float)
        finite = arr[np.isfinite(arr)]
        return bool(finite.size and np.max(finite) > 0.5)
    try:
        return bool(float(requires_value) > 0.5)
    except (TypeError, ValueError):
        return False


def _comparison_series(greedy_baseline_mode: str):
    baseline_key = _normalize_baseline_mode(greedy_baseline_mode)
    baseline_color, baseline_marker = _baseline_style(baseline_key)
    return [
        (baseline_key, _baseline_label(baseline_key), baseline_color, baseline_marker),
        ('rl', 'MAPPO', 'tab:orange', 's'),
    ]


def _comparison_source(
    series_key: str,
    greedy_original: Dict,
    greedy_original_lite: Optional[Dict],
    greedy_original_normal_v2: Optional[Dict],
    greedy_myopic: Optional[Dict],
    greedy_matched: Dict,
    greedy_throughput_feasible: Optional[Dict],
    greedy_throughput_only: Optional[Dict],
    greedy_channel_only: Optional[Dict],
    rl: Dict,
):
    if series_key == 'rl':
        return rl
    if series_key == 'original':
        return greedy_original
    if series_key == 'original_greedy_normal_v1':
        return greedy_original_lite
    if series_key == 'original_greedy_normal_v2':
        return greedy_original_normal_v2
    if series_key == 'myopic_throughput_greedy':
        return greedy_myopic
    if series_key == 'matched_fixed_embb':
        return greedy_matched
    if series_key == 'throughput_feasible_oracle':
        return greedy_throughput_feasible
    if series_key == 'throughput_only_greedy':
        return greedy_throughput_only
    if series_key == 'rate_loss_min_greedy':
        return greedy_throughput_only
    return greedy_channel_only


def _selected_baseline_source(
    greedy_original: Dict,
    greedy_original_lite: Optional[Dict],
    greedy_original_normal_v2: Optional[Dict],
    greedy_myopic: Optional[Dict],
    greedy_matched: Dict,
    greedy_throughput_feasible: Optional[Dict],
    greedy_throughput_only: Optional[Dict],
    greedy_channel_only: Optional[Dict],
    greedy_baseline_mode: str,
):
    baseline_key = _normalize_baseline_mode(greedy_baseline_mode)
    baseline_color, baseline_marker = _baseline_style(baseline_key)
    return (
        _comparison_source(
            baseline_key,
            greedy_original,
            greedy_original_lite,
            greedy_original_normal_v2,
            greedy_myopic,
            greedy_matched,
            greedy_throughput_feasible,
            greedy_throughput_only,
            greedy_channel_only,
            {},
        ),
        _baseline_label(baseline_key),
        baseline_color,
        baseline_marker,
    )




def run_dense_method_bundle(
    dense_loads: List[float],
    eval_replicas: int,
    checkpoint_path: Path,
    checkpoint_reason: str,
    cfg: SRMAPPOConfig,
) -> Dict[str, Dict]:
    bundle_start = perf_counter()
    base_sys, base_urllc, base_embb, base_algo, base_sim = _build_main_like_configs()
    checkpoint_cfg = deepcopy(_load_checkpoint_cfg(checkpoint_path))
    checkpoint_cfg.env.include_greedy_reference_in_obs = False
    _apply_forced_urllc_ratio_to_sim(base_sim, cfg, log_prefix="DENSE")
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
            if method_key in {'rl', 'matched_fixed_embb', 'myopic_throughput_greedy', 'throughput_only_greedy', 'channel_only_greedy', 'throughput_biased_greedy'}:
                env_cfg = deepcopy(checkpoint_cfg if method_key == 'rl' else cfg)
                env_cfg.env.include_greedy_reference_in_obs = False
                env = SRMAPPOPhaseAEnv(sys_cfg, urllc_cfg, embb_cfg, algo_cfg, sim_cfg, env_cfg)
                if method_key == 'rl' and model is None:
                    model_cfg, model = _build_model_for_env(env, checkpoint_path)
            records: List[Dict[str, float]] = []
            seed_base = _report_seed_base(load_idx, model_cfg or checkpoint_cfg) + 6000
            for rep_idx in range(int(eval_replicas)):
                seed = seed_base + rep_idx
                if method_key == 'rl':
                    episode = run_env_episode(
                        env,
                        model,
                        model_cfg,
                        seed=seed,
                        collect_trace=False,
                        use_greedy=False,
                        cache_tag=str(Path(checkpoint_path).resolve()),
                    )
                elif method_key == 'matched_fixed_embb':
                    episode = run_env_episode(
                        env,
                        model=None,
                        cfg=cfg,
                        seed=seed,
                        collect_trace=False,
                        use_greedy=True,
                        cache_tag=str(Path(checkpoint_path).resolve()),
                    )
                elif method_key == 'throughput_only_greedy':
                    episode = run_env_episode(
                        env,
                        model=None,
                        cfg=cfg,
                        seed=seed,
                        collect_trace=False,
                        use_greedy=True,
                        greedy_policy='throughput_only',
                        cache_tag=str(Path(checkpoint_path).resolve()),
                    )
                elif method_key == 'myopic_throughput_greedy':
                    episode = run_env_episode(
                        env,
                        model=None,
                        cfg=cfg,
                        seed=seed,
                        collect_trace=False,
                        use_greedy=True,
                        greedy_policy='myopic_throughput',
                        cache_tag=str(Path(checkpoint_path).resolve()),
                    )
                elif method_key == 'channel_only_greedy':
                    episode = run_env_episode(
                        env,
                        model=None,
                        cfg=cfg,
                        seed=seed,
                        collect_trace=False,
                        use_greedy=True,
                        greedy_policy='channel_only',
                        cache_tag=str(Path(checkpoint_path).resolve()),
                    )
                elif method_key == 'original':
                    episode = _run_original_greedy_slot(sys_cfg, urllc_cfg, embb_cfg, algo_cfg, sim_cfg, seed=seed, slot_index=rep_idx)
                elif method_key == 'original_greedy_normal_v1':
                    episode = _run_original_greedy_normal_v1_slot(sys_cfg, urllc_cfg, embb_cfg, algo_cfg, sim_cfg, seed=seed, slot_index=rep_idx)
                elif method_key == 'original_greedy_normal_v2':
                    episode = _run_original_greedy_normal_v2_slot(sys_cfg, urllc_cfg, embb_cfg, algo_cfg, sim_cfg, seed=seed, slot_index=rep_idx)
                elif method_key == 'embb_only_ceiling':
                    episode = _run_embb_only_ceiling_slot(sys_cfg, urllc_cfg, embb_cfg, algo_cfg, sim_cfg, seed=seed, slot_index=rep_idx)
                elif method_key == 'throughput_biased_greedy':
                    episode = run_env_episode(
                        env,
                        model=None,
                        cfg=cfg,
                        seed=seed,
                        collect_trace=False,
                        use_greedy=True,
                        greedy_policy='throughput_biased',
                        cache_tag=str(Path(checkpoint_path).resolve()),
                    )
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
    result = finalize_dense_bundle(method_bundle)
    _report_timing_log(
        f"run_dense_method_bundle loads={len(dense_loads)} eval_replicas={eval_replicas} sec={perf_counter() - bundle_start:.3f}"
    )
    return result


def run_timeslot_series(
    load: float,
    num_slots: int,
    checkpoint_path: Path,
    cfg: Optional[SRMAPPOConfig] = None,
    frozen_greedy_payload: Optional[Dict] = None,
):
    series_start = perf_counter()
    greedy_series, meta = run_greedy_timeslot_series(
        load,
        num_slots,
        checkpoint_path,
        cfg=cfg,
        frozen_greedy_payload=frozen_greedy_payload,
    )
    base_sys, base_urllc, base_embb, base_algo, base_sim = _build_main_like_configs()
    _apply_forced_urllc_ratio_to_sim(base_sim, deepcopy(cfg or _load_checkpoint_cfg(checkpoint_path)), log_prefix="TIMESLOT")
    sys_cfg, urllc_cfg, embb_cfg, algo_cfg, sim_cfg = _configure_density_scenario(
        load, base_sys, base_urllc, base_embb, base_algo, base_sim
    )

    report_cfg = deepcopy(cfg or _load_checkpoint_cfg(checkpoint_path))
    report_cfg.env.include_greedy_reference_in_obs = False
    env = SRMAPPOPhaseAEnv(sys_cfg, urllc_cfg, embb_cfg, algo_cfg, sim_cfg, report_cfg)
    cfg, model = _build_model_for_env(env, checkpoint_path)

    rl_series = []
    seed_base = 8000
    checkpoint_cache_tag = str(Path(checkpoint_path).resolve())
    for slot_idx in range(num_slots):
        seed = seed_base + slot_idx
        rl_summary = run_env_episode(
            env,
            model,
            cfg,
            seed=seed,
            collect_trace=False,
            use_greedy=False,
            cache_tag=checkpoint_cache_tag,
        )
        rl_series.append(rl_summary)

    meta['cfg'] = cfg
    _report_timing_log(
        f"run_timeslot_series load={float(load):.1f} num_slots={num_slots} sec={perf_counter() - series_start:.3f}"
    )
    return greedy_series, rl_series, meta


def run_greedy_timeslot_series(
    load: float,
    num_slots: int,
    checkpoint_path: Path,
    cfg: Optional[SRMAPPOConfig] = None,
    frozen_greedy_payload: Optional[Dict] = None,
):
    base_sys, base_urllc, base_embb, base_algo, base_sim = _build_main_like_configs()
    sys_cfg, urllc_cfg, embb_cfg, algo_cfg, sim_cfg = _configure_density_scenario(
        load, base_sys, base_urllc, base_embb, base_algo, base_sim
    )

    report_cfg = deepcopy(cfg or _load_checkpoint_cfg(checkpoint_path))
    greedy_series = []
    seed_base = 8000
    greedy_mode = _greedy_baseline_mode(report_cfg)
    greedy_env = None
    if greedy_mode == "matched_fixed_embb":
        greedy_cfg = deepcopy(report_cfg)
        greedy_cfg.env.include_greedy_reference_in_obs = False
        greedy_env = SRMAPPOPhaseAEnv(sys_cfg, urllc_cfg, embb_cfg, algo_cfg, sim_cfg, greedy_cfg)
    elif greedy_mode == "throughput_feasible_oracle":
        greedy_cfg = deepcopy(report_cfg)
        greedy_cfg.env.include_greedy_reference_in_obs = False
        greedy_env = SRMAPPOPhaseAEnv(sys_cfg, urllc_cfg, embb_cfg, algo_cfg, sim_cfg, greedy_cfg)
    elif greedy_mode == "hard_feasible_throughput_greedy":
        greedy_cfg = deepcopy(report_cfg)
        greedy_cfg.env.include_greedy_reference_in_obs = False
        greedy_env = SRMAPPOPhaseAEnv(sys_cfg, urllc_cfg, embb_cfg, algo_cfg, sim_cfg, greedy_cfg)
    elif greedy_mode == "throughput_only_greedy":
        greedy_cfg = deepcopy(report_cfg)
        greedy_cfg.env.include_greedy_reference_in_obs = False
        greedy_env = SRMAPPOPhaseAEnv(sys_cfg, urllc_cfg, embb_cfg, algo_cfg, sim_cfg, greedy_cfg)
    elif greedy_mode == "rate_loss_min_greedy":
        greedy_cfg = deepcopy(report_cfg)
        greedy_cfg.env.include_greedy_reference_in_obs = False
        greedy_env = SRMAPPOPhaseAEnv(sys_cfg, urllc_cfg, embb_cfg, algo_cfg, sim_cfg, greedy_cfg)
    elif greedy_mode == "force_admit_minloss_greedy":
        greedy_cfg = deepcopy(report_cfg)
        greedy_cfg.env.include_greedy_reference_in_obs = False
        greedy_env = SRMAPPOPhaseAEnv(sys_cfg, urllc_cfg, embb_cfg, algo_cfg, sim_cfg, greedy_cfg)
    elif greedy_mode == "myopic_throughput_greedy":
        greedy_cfg = deepcopy(report_cfg)
        greedy_cfg.env.include_greedy_reference_in_obs = False
        greedy_env = SRMAPPOPhaseAEnv(sys_cfg, urllc_cfg, embb_cfg, algo_cfg, sim_cfg, greedy_cfg)
    elif greedy_mode == "channel_only_greedy":
        greedy_cfg = deepcopy(report_cfg)
        greedy_cfg.env.include_greedy_reference_in_obs = False
        greedy_env = SRMAPPOPhaseAEnv(sys_cfg, urllc_cfg, embb_cfg, algo_cfg, sim_cfg, greedy_cfg)
    elif greedy_mode == "frozen_json":
        if frozen_greedy_payload is None:
            frozen_greedy_payload = _load_frozen_greedy_payload(report_cfg)
        frozen_load = float(frozen_greedy_payload.get('timeslot_series_load', -1.0))
        frozen_slots = int(frozen_greedy_payload.get('timeslot_series_slots', -1))
        if frozen_load != float(load) or frozen_slots != int(num_slots):
            raise ValueError(
                f"Frozen greedy payload timeslot series mismatch: expected load={load}, slots={num_slots}, "
                f"got load={frozen_load}, slots={frozen_slots}"
            )
        greedy_series = list(frozen_greedy_payload.get('slot_greedy', []))
    for slot_idx in range(num_slots):
        seed = seed_base + slot_idx
        if greedy_mode == "original":
            greedy_summary = _run_original_greedy_slot(
                sys_cfg,
                urllc_cfg,
                embb_cfg,
                algo_cfg,
                sim_cfg,
                seed=seed,
                slot_index=slot_idx,
            )
        elif greedy_mode == "original_greedy_normal_v1":
            greedy_summary = _run_original_greedy_normal_v1_slot(
                sys_cfg,
                urllc_cfg,
                embb_cfg,
                algo_cfg,
                sim_cfg,
                seed=seed,
                slot_index=slot_idx,
            )
        elif greedy_mode == "original_greedy_normal_v2":
            greedy_summary = _run_original_greedy_normal_v2_slot(
                sys_cfg,
                urllc_cfg,
                embb_cfg,
                algo_cfg,
                sim_cfg,
                seed=seed,
                slot_index=slot_idx,
            )
        elif greedy_mode == "matched_fixed_embb":
            greedy_summary = run_env_episode(
                greedy_env,
                model=None,
                cfg=report_cfg,
                seed=seed,
                collect_trace=False,
                use_greedy=True,
                greedy_policy="reference",
                cache_tag=str(Path(checkpoint_path).resolve()),
            )
        elif greedy_mode == "throughput_feasible_oracle":
            greedy_summary = run_env_episode(
                greedy_env,
                model=None,
                cfg=report_cfg,
                seed=seed,
                collect_trace=False,
                use_greedy=True,
                greedy_policy="throughput_feasible",
                cache_tag=str(Path(checkpoint_path).resolve()),
            )
        elif greedy_mode == "hard_feasible_throughput_greedy":
            greedy_summary = run_env_episode(
                greedy_env,
                model=None,
                cfg=report_cfg,
                seed=seed,
                collect_trace=False,
                use_greedy=True,
                greedy_policy="hard_feasible_throughput",
                cache_tag=str(Path(checkpoint_path).resolve()),
            )
        elif greedy_mode == "myopic_throughput_greedy":
            greedy_summary = run_env_episode(
                greedy_env,
                model=None,
                cfg=report_cfg,
                seed=seed,
                collect_trace=False,
                use_greedy=True,
                greedy_policy="myopic_throughput",
                cache_tag=str(Path(checkpoint_path).resolve()),
            )
        elif greedy_mode == "throughput_only_greedy":
            greedy_summary = run_env_episode(
                greedy_env,
                model=None,
                cfg=report_cfg,
                seed=seed,
                collect_trace=False,
                use_greedy=True,
                greedy_policy="throughput_only",
                cache_tag=str(Path(checkpoint_path).resolve()),
            )
        elif greedy_mode == "rate_loss_min_greedy":
            greedy_summary = run_env_episode(
                greedy_env,
                model=None,
                cfg=report_cfg,
                seed=seed,
                collect_trace=False,
                use_greedy=True,
                greedy_policy="rate_loss_min",
                cache_tag=str(Path(checkpoint_path).resolve()),
            )
        elif greedy_mode == "force_admit_minloss_greedy":
            greedy_summary = run_env_episode(
                greedy_env,
                model=None,
                cfg=report_cfg,
                seed=seed,
                collect_trace=False,
                use_greedy=True,
                greedy_policy="force_admit_minloss",
                cache_tag=str(Path(checkpoint_path).resolve()),
            )
        elif greedy_mode == "channel_only_greedy":
            greedy_summary = run_env_episode(
                greedy_env,
                model=None,
                cfg=report_cfg,
                seed=seed,
                collect_trace=False,
                use_greedy=True,
                greedy_policy="channel_only",
                cache_tag=str(Path(checkpoint_path).resolve()),
            )
        else:
            greedy_summary = greedy_series[slot_idx]
        if greedy_mode != "frozen_json":
            greedy_series.append(greedy_summary)

    meta = {
        'load': float(load),
        'num_slots': int(num_slots),
        'sys_cfg': sys_cfg,
        'sim_cfg': sim_cfg,
        'checkpoint': checkpoint_path,
        'cfg': report_cfg,
        'greedy_baseline_mode': greedy_mode,
        'greedy_baseline_label': _baseline_label(greedy_mode),
    }
    return greedy_series, meta


def plot_core_kpis(
    greedy_original: Dict,
    greedy_original_lite: Optional[Dict],
    greedy_original_normal_v2: Optional[Dict],
    greedy_myopic: Optional[Dict],
    greedy_matched: Dict,
    greedy_throughput_feasible: Optional[Dict],
    greedy_throughput_only: Optional[Dict],
    greedy_channel_only: Optional[Dict],
    rl: Dict,
    embb_only_ceiling: Optional[Dict] = None,
    throughput_oracle: Optional[Dict] = None,
    greedy_baseline_mode: str = "original",
):
    fig, axes = plt.subplots(3, 2, figsize=(14, 14), constrained_layout=True)
    loads = rl['loads']
    baseline_label = _baseline_label(greedy_baseline_mode)
    panels = [
        ('Aggregate eMBB throughput', 'embb_rate', 1e6, 'Mbps'),
        ('URLLC admission ratio', 'urllc_admission', 1.0, 'Ratio'),
        ('Admitted URLLC reliability', 'admitted_urllc_reliability', 1.0, 'Reliability'),
        ('eMBB positive-rate ratio', 'embb_positive_rate_ratio', 1.0, 'Ratio'),
        ('Per-user eMBB rate', 'embb_user_rate', 1e6, 'Mbps'),
        ('Total transmit power', 'total_power', 1e3, 'mW'),
    ]
    for ax, (title, key, scale, ylabel) in zip(axes.flat, panels):
        for series_key, label, color, marker in _comparison_series(greedy_baseline_mode):
            data = _comparison_source(
                series_key,
                greedy_original,
                greedy_original_lite,
                greedy_original_normal_v2,
                greedy_myopic,
                greedy_matched,
                greedy_throughput_feasible,
                greedy_throughput_only,
                greedy_channel_only,
                rl,
            )
            if data is None:
                continue
            values = np.asarray(data[key]) * scale if key == 'total_power' else np.asarray(data[key]) / scale
            ax.plot(loads, values, marker=marker, color=color, label=label)
        if key == 'total_power':
            _style_power_axis(ax, title, 'Average UE load per UAV', ylabel)
        else:
            _style(ax, title, 'Average UE load per UAV', ylabel)
        _top_axis_lambda(ax, loads)
        ax.legend()
    fig.suptitle(f'Core KPIs vs Load | {baseline_label} vs MAPPO | Selected baseline: {baseline_label}', fontsize=14)
    path = RESULTS_DIR / '01_core_kpis_vs_load.png'
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_core_kpi_debug_fast(
    rl: Dict,
    baseline: Dict,
    *,
    baseline_label: str = "Greedy",
    legend_title: str = "",
    greedy_only: bool = False,
) -> Path:
    """Fast core KPI plot (MAPPO vs selected baseline)."""
    loads = np.asarray(rl.get('loads', []), dtype=float)
    fig, axes = plt.subplots(2, 4, figsize=(20, 9.0), constrained_layout=True)
    axes = np.asarray(axes).reshape(2, 4)

    def _series(data: Dict, key: str) -> np.ndarray:
        arr = np.asarray(data.get(key, []), dtype=float)
        return arr if arr.size == loads.size else np.zeros_like(loads, dtype=float)

    def _urllc_tp_bps_slot_est(data: Dict) -> np.ndarray:
        """Robust slot-based URLLC throughput estimate for plotting."""
        tp = _series(data, 'urllc_throughput_bps_slot_est')
        if tp.size != loads.size:
            tp = np.zeros_like(loads, dtype=float)
        if np.any(tp > 0.0):
            return tp
        scheduled = _series(data, 'scheduled_packets')
        if not np.any(scheduled > 0.0):
            return tp
        packet_bits = _series(data, 'urllc_packet_bits_mean')
        slot_dur = _series(data, 'urllc_slot_duration_s')
        if packet_bits.size != loads.size:
            packet_bits = np.full_like(loads, 160.0, dtype=float)
        if slot_dur.size != loads.size:
            slot_dur = np.full_like(loads, 1.0e-3, dtype=float)
        return scheduled * packet_bits / np.maximum(slot_dur, 1.0e-12)

    def _actual_arrivals(data: Dict) -> np.ndarray:
        arrivals = _series(data, 'active_packets')
        if np.any(arrivals > 0.0):
            return arrivals
        scheduled = _series(data, 'scheduled_packets')
        admission = _series(data, 'urllc_admission')
        if not np.any(scheduled > 0.0) or not np.any(admission > 0.0):
            return arrivals
        return np.divide(
            scheduled,
            np.maximum(admission, 1.0e-12),
            out=np.zeros_like(loads, dtype=float),
            where=np.maximum(admission, 1.0e-12) > 0.0,
        )

    def _plot2(ax, key: str, title: str, ylabel: str, *, scale_div: float = 1.0, scale_mul: float = 1.0):
        y_rl = _series(rl, key) / scale_div * scale_mul
        y_base = _series(baseline, key) / scale_div * scale_mul
        if not greedy_only:
            ax.plot(loads, y_rl, color='tab:orange', marker='s', markersize=6, linewidth=2.0, label='MAPPO (solid)')
        ax.plot(loads, y_base, color='tab:brown', marker='o', markersize=6, linewidth=2.0, linestyle='--', label=f'{baseline_label} (dashed)')
        _style(ax, title, 'Average UE load per UAV', ylabel)
        ax.legend(loc="best", fontsize=8, frameon=False)

    _plot2(axes[0, 0], 'embb_rate', 'Aggregate eMBB throughput', 'Mbps', scale_div=1e6)
    # URLLC admission (keep reliability in a separate fast debug figure to avoid confusion).
    ax = axes[0, 1]
    m_adm = _series(rl, 'urllc_admission')
    g_adm = _series(baseline, 'urllc_admission')
    if not greedy_only:
        ax.plot(loads, m_adm, color='tab:orange', marker='s', markersize=6, linewidth=2.0, label='MAPPO admission (solid)')
    ax.plot(loads, g_adm, color='tab:brown', marker='o', markersize=6, linewidth=2.0, linestyle='--', label=f'{baseline_label} admission (dashed)')
    # Ratio companion totals (arrivals / admitted).
    ax2 = ax.twinx()
    m_arrivals = _actual_arrivals(rl)
    m_admitted = _series(rl, 'scheduled_packets')
    g_arrivals = _actual_arrivals(baseline)
    g_admitted = _series(baseline, 'scheduled_packets')
    if not greedy_only:
        ax2.plot(loads, m_arrivals, color='tab:blue', marker='^', markersize=4, linewidth=1.6, alpha=0.55, label='MAPPO arrivals (count)')
        ax2.plot(loads, m_admitted, color='tab:green', marker='v', markersize=4, linewidth=1.6, alpha=0.55, label='MAPPO admitted (count)')
    ax2.plot(loads, g_arrivals, color='tab:purple', marker='^', markersize=4, linewidth=1.3, alpha=0.45, linestyle='--', label=f'{baseline_label} arrivals (count)')
    ax2.plot(loads, g_admitted, color='tab:olive', marker='v', markersize=4, linewidth=1.3, alpha=0.45, linestyle='--', label=f'{baseline_label} admitted (count)')
    _style(ax, 'URLLC admission ratio', 'Average UE load per UAV', 'Ratio')
    ax2.set_ylabel('Packets')
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="best", fontsize=8, frameon=False)

    # Service ratio panel: service ratio + min-rate satisfaction ratio (both methods).
    ax = axes[0, 2]
    m_srv = _series_with_fallback(
        rl,
        loads,
        'embb_service_ratio_after_puncture_deduction',
        ('embb_service_ratio',),
        context='core_kpi_debug.mappo.service_corrected',
    )
    g_srv = _series_with_fallback(
        baseline,
        loads,
        'embb_service_ratio_after_puncture_deduction',
        ('embb_service_ratio',),
        context='core_kpi_debug.greedy.service_corrected',
    )
    m_min = _series_with_fallback(
        rl,
        loads,
        'embb_min_rate_satisfaction_after_puncture_deduction',
        ('embb_min_rate_satisfaction_ratio',),
        context='core_kpi_debug.mappo.minrate_corrected',
    )
    g_min = _series_with_fallback(
        baseline,
        loads,
        'embb_min_rate_satisfaction_after_puncture_deduction',
        ('embb_min_rate_satisfaction_ratio',),
        context='core_kpi_debug.greedy.minrate_corrected',
    )
    if not greedy_only:
        ax.plot(loads, m_srv, color='tab:orange', marker='s', markersize=6, linewidth=2.0, label='MAPPO service ratio (solid)')
    ax.plot(loads, g_srv, color='tab:brown', marker='o', markersize=6, linewidth=2.0, linestyle='--', label=f'{baseline_label} service ratio (dashed)')
    if not greedy_only:
        ax.plot(loads, m_min, color='tab:green', marker='^', markersize=6, linewidth=2.0, alpha=0.90, zorder=3, label='MAPPO min-rate satisfaction (solid)')
    ax.plot(loads, g_min, color='tab:gray', marker='d', markersize=6, linewidth=2.0, linestyle='--', alpha=0.85, zorder=2, label=f'{baseline_label} min-rate satisfaction (dashed)')
    # Ratio companion totals (served users / min-rate-satisfied users).
    ax2 = ax.twinx()
    m_embb_n = _series(rl, 'embb_user_count')
    g_embb_n = _series(baseline, 'embb_user_count')
    m_served_n = m_srv * np.maximum(m_embb_n, 0.0)
    g_served_n = g_srv * np.maximum(g_embb_n, 0.0)
    m_min_ok_n = _series_with_fallback(
        rl,
        loads,
        'embb_min_rate_satisfied_user_count_after_puncture_deduction',
        ('embb_min_rate_satisfied_user_count',),
        context='core_kpi_debug.mappo.minrate_count',
    )
    g_min_ok_n = _series_with_fallback(
        baseline,
        loads,
        'embb_min_rate_satisfied_user_count_after_puncture_deduction',
        ('embb_min_rate_satisfied_user_count',),
        context='core_kpi_debug.greedy.minrate_count',
    )
    if m_min_ok_n.size != loads.size or not np.any(np.isfinite(m_min_ok_n)):
        m_min_ok_n = m_min * np.maximum(m_embb_n, 0.0)
    if g_min_ok_n.size != loads.size or not np.any(np.isfinite(g_min_ok_n)):
        g_min_ok_n = g_min * np.maximum(g_embb_n, 0.0)
    if not greedy_only:
        ax2.plot(loads, m_served_n, color='tab:cyan', marker='P', markersize=4, linewidth=1.5, alpha=0.55, label='MAPPO served eMBB (count)')
        ax2.plot(loads, m_min_ok_n, color='tab:blue', marker='X', markersize=4, linewidth=1.5, alpha=0.55, label='MAPPO min-rate OK (count)')
    ax2.plot(loads, g_served_n, color='tab:pink', marker='P', markersize=4, linewidth=1.3, alpha=0.45, linestyle='--', label=f'{baseline_label} served eMBB (count)')
    ax2.plot(loads, g_min_ok_n, color='tab:purple', marker='X', markersize=4, linewidth=1.3, alpha=0.45, linestyle='--', label=f'{baseline_label} min-rate OK (count)')
    _style(ax, 'eMBB service & min-rate satisfaction (corrected)', 'Average UE load per UAV', 'Ratio')
    ax2.set_ylabel('Users')
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="best", fontsize=8, frameon=False)
    _report_plot_key_audit(
        "core_kpi_debug.service_minrate_corrected",
        [
            ("sr_mappo.embb_service_ratio_after_puncture_deduction_or_fallback", m_srv),
            ("greedy.embb_service_ratio_after_puncture_deduction_or_fallback", g_srv),
            ("sr_mappo.embb_min_rate_satisfaction_after_puncture_deduction_or_fallback", m_min),
            ("greedy.embb_min_rate_satisfaction_after_puncture_deduction_or_fallback", g_min),
        ],
    )

    _plot2(axes[0, 3], 'total_power', 'Total transmit power', 'mW', scale_mul=1e3)

    # eMBB:URLLC user ratio (counts + ratio).
    ax = axes[1, 0]
    embb_count = _series(rl, 'embb_user_count')
    urllc_count = _series(rl, 'urllc_user_count')
    ratio = _series(rl, 'embb_urllc_user_ratio')
    ax.plot(loads, embb_count, color='tab:orange', marker='s', markersize=6, linewidth=2.0, label='eMBB users (solid)')
    ax.plot(loads, urllc_count, color='tab:brown', marker='o', markersize=6, linewidth=2.0, label='URLLC users (solid)')
    ax2 = ax.twinx()
    ax2.plot(loads, ratio, color='tab:blue', linestyle='--', marker='^', markersize=5, linewidth=1.8, label='eMBB/URLLC ratio (dashed)')
    _style(ax, 'User mix', 'Average UE load per UAV', 'Users')
    ax2.set_ylabel('Ratio')
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="best", fontsize=8, frameon=False)

    # Coexistence-only mode mix: KEEP is excluded from the denominator.
    ax = axes[1, 1]
    if not greedy_only:
        ax.plot(loads, _series(rl, 'overlay_ratio'), color='tab:orange', marker='s', markersize=6, linewidth=2.0, label='MAPPO superpose/overlay (solid)')
        ax.plot(loads, _series(rl, 'puncture_ratio'), color='tab:red', marker='x', markersize=6, linewidth=2.0, label='MAPPO puncture (solid)')
    ax.plot(loads, _series(baseline, 'overlay_ratio'), color='tab:brown', marker='o', markersize=6, linewidth=2.0, linestyle='--', label=f'{baseline_label} superpose/overlay (dashed)')
    ax.plot(loads, _series(baseline, 'puncture_ratio'), color='tab:gray', marker='d', markersize=6, linewidth=2.0, linestyle='--', label=f'{baseline_label} puncture (dashed)')
    _style(ax, 'Coexistence mode mix (excluding KEEP)', 'Average UE load per UAV', 'Ratio')
    ax.legend(loc="best", fontsize=8, frameon=False)

    # URLLC throughput (Mbps estimate).
    ax = axes[1, 2]
    m_tp = _urllc_tp_bps_slot_est(rl)
    g_tp = _urllc_tp_bps_slot_est(baseline)
    if not greedy_only:
        ax.plot(loads, m_tp / 1.0e6, color='tab:orange', marker='s', markersize=6, linewidth=2.0, label='MAPPO URLLC throughput (solid)')
    ax.plot(loads, g_tp / 1.0e6, color='tab:brown', marker='o', markersize=6, linewidth=2.0, linestyle='--', label=f'{baseline_label} URLLC throughput (dashed)')
    _style(ax, 'URLLC throughput (slot est.)', 'Average UE load per UAV', 'Mbps')
    ax.legend(loc="best", fontsize=8, frameon=False)

    # Power split (eMBB / URLLC / total).
    ax = axes[1, 3]
    if not greedy_only:
        ax.plot(loads, _series(rl, 'embb_power') * 1e3, color='tab:orange', marker='s', markersize=6, linewidth=2.0, label='MAPPO eMBB power (solid)')
        ax.plot(loads, _series(rl, 'urllc_power') * 1e3, color='tab:green', marker='^', markersize=6, linewidth=2.0, label='MAPPO URLLC power (solid)')
        ax.plot(loads, _series(rl, 'total_power') * 1e3, color='tab:blue', marker='P', markersize=6, linewidth=2.0, label='MAPPO total power (solid)')
    ax.plot(loads, _series(baseline, 'embb_power') * 1e3, color='tab:brown', marker='o', markersize=6, linewidth=2.0, linestyle='--', label=f'{baseline_label} eMBB power (dashed)')
    ax.plot(loads, _series(baseline, 'urllc_power') * 1e3, color='tab:olive', marker='d', markersize=6, linewidth=2.0, linestyle='--', label=f'{baseline_label} URLLC power (dashed)')
    ax.plot(loads, _series(baseline, 'total_power') * 1e3, color='tab:gray', marker='x', markersize=6, linewidth=2.0, linestyle='--', label=f'{baseline_label} total power (dashed)')
    _style(ax, 'Power split', 'Average UE load per UAV', 'mW')
    ax.legend(loc="best", fontsize=8, frameon=False)

    if legend_title:
        fig.suptitle(str(legend_title), fontsize=10)
    for ax in axes.ravel():
        ax.set_xticks(loads)
    path = RESULTS_DIR / 'core_kpi_debug.png'
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_urllc_arrival_admit_debug(
    metrics: Dict,
    *,
    episodes_per_load: int,
    title_suffix: str = "",
) -> Path:
    """Plot URLLC arrivals/admissions and their ratio per load."""
    loads = np.asarray(metrics.get("loads", []), dtype=float)

    def _series(key: str) -> np.ndarray:
        arr = np.asarray(metrics.get(key, []), dtype=float)
        if arr.size == loads.size:
            return arr
        return np.zeros_like(loads, dtype=float)

    mean_arrivals = _series("active_packets")
    mean_admits = _series("scheduled_packets")
    total_arrivals = mean_arrivals * max(int(episodes_per_load), 1)
    total_admits = mean_admits * max(int(episodes_per_load), 1)
    ratio = np.divide(total_admits, np.maximum(total_arrivals, 1.0), out=np.zeros_like(total_admits), where=total_arrivals > 0.0)

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6), constrained_layout=True)
    ax0, ax1, ax2 = axes

    ax0.plot(loads, total_arrivals, color="tab:blue", marker="o", linewidth=2.0, label="Total arrivals")
    _style(ax0, "URLLC arrivals (total)", "Average UE load per UAV", "Packets")
    ax0.legend(loc="best", fontsize=8, frameon=False)

    ax1.plot(loads, total_admits, color="tab:green", marker="s", linewidth=2.0, label="Total admitted")
    _style(ax1, "URLLC admitted (total)", "Average UE load per UAV", "Packets")
    ax1.legend(loc="best", fontsize=8, frameon=False)

    ax2.plot(loads, ratio, color="tab:brown", marker="d", linewidth=2.0, linestyle="--", label="Admit/Arrival ratio")
    _style(ax2, "URLLC admit ratio", "Average UE load per UAV", "Ratio")
    ax2.legend(loc="best", fontsize=8, frameon=False)

    if title_suffix:
        fig.suptitle(str(title_suffix), fontsize=10)
    path = RESULTS_DIR / "urllc_arrival_admit_debug.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def _report_compare_total_user_axis(rl: Dict, baseline: Dict) -> tuple[np.ndarray, str]:
    loads = np.asarray(rl.get('loads', baseline.get('loads', [])), dtype=float)
    rl_total = np.asarray(rl.get('embb_user_count', []), dtype=float) + np.asarray(rl.get('urllc_user_count', []), dtype=float)
    baseline_total = np.asarray(baseline.get('embb_user_count', []), dtype=float) + np.asarray(baseline.get('urllc_user_count', []), dtype=float)
    if rl_total.size == loads.size and np.all(np.isfinite(rl_total)) and np.all(rl_total > 0.0):
        return rl_total, 'Total users in system'
    if baseline_total.size == loads.size and np.all(np.isfinite(baseline_total)) and np.all(baseline_total > 0.0):
        return baseline_total, 'Total users in system'
    return loads, 'Average UE load per UAV'


def plot_min_rate_satisfied_count_compare(
    rl: Dict,
    baseline: Dict,
    *,
    baseline_label: str = "Greedy",
) -> Path:
    x, xlabel = _report_compare_total_user_axis(rl, baseline)
    rl_count = np.asarray(rl.get('embb_min_rate_satisfied_user_count', []), dtype=float)
    baseline_count = np.asarray(baseline.get('embb_min_rate_satisfied_user_count', []), dtype=float)
    if rl_count.size != x.size:
        rl_ratio = np.asarray(rl.get('embb_min_rate_satisfaction_ratio', []), dtype=float)
        rl_users = np.asarray(rl.get('embb_user_count', []), dtype=float)
        rl_count = rl_ratio * rl_users if rl_ratio.size == x.size and rl_users.size == x.size else np.zeros_like(x)
    if baseline_count.size != x.size:
        baseline_ratio = np.asarray(baseline.get('embb_min_rate_satisfaction_ratio', []), dtype=float)
        baseline_users = np.asarray(baseline.get('embb_user_count', []), dtype=float)
        baseline_count = baseline_ratio * baseline_users if baseline_ratio.size == x.size and baseline_users.size == x.size else np.zeros_like(x)

    fig, ax = plt.subplots(figsize=(9.2, 5.4), constrained_layout=True)
    ax.plot(x, rl_count, color='tab:orange', marker='s', markersize=6, linewidth=2.1, label='MAPPO')
    ax.plot(x, baseline_count, color='tab:brown', marker='o', markersize=6, linewidth=2.1, linestyle='--', label=baseline_label)
    ax.set_title('Min-rate satisfied eMBB user count')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Users')
    ax.grid(True, alpha=0.25)
    ax.legend(loc='best', fontsize=9, frameon=False)
    path = RESULTS_DIR / 'custom_min_rate_satisfied_count_compare.png'
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_admitted_urllc_packets_compare(
    rl: Dict,
    baseline: Dict,
    *,
    baseline_label: str = "Greedy",
) -> Path:
    x, xlabel = _report_compare_total_user_axis(rl, baseline)
    rl_packets = np.asarray(rl.get('scheduled_packets', []), dtype=float)
    baseline_packets = np.asarray(baseline.get('scheduled_packets', []), dtype=float)

    fig, ax = plt.subplots(figsize=(9.2, 5.4), constrained_layout=True)
    ax.plot(x, rl_packets, color='tab:orange', marker='s', markersize=6, linewidth=2.1, label='MAPPO')
    ax.plot(x, baseline_packets, color='tab:brown', marker='o', markersize=6, linewidth=2.1, linestyle='--', label=baseline_label)
    ax.set_title('Admitted URLLC packet count')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Packets')
    ax.grid(True, alpha=0.25)
    ax.legend(loc='best', fontsize=9, frameon=False)
    path = RESULTS_DIR / 'custom_admitted_urllc_packets_compare.png'
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_mode_action_share_compare(
    rl: Dict,
    baseline: Dict,
    *,
    baseline_label: str = "Greedy",
) -> Path:
    x, xlabel = _report_compare_total_user_axis(rl, baseline)

    def _series(data: Dict, key: str) -> np.ndarray:
        arr = np.asarray(data.get(key, []), dtype=float)
        return arr if arr.size == x.size else np.zeros_like(x, dtype=float)

    m_overlay = _series(rl, 'overlay_selection_ratio')
    m_puncture = _series(rl, 'puncture_selection_ratio')
    g_overlay = _series(baseline, 'overlay_selection_ratio')
    g_puncture = _series(baseline, 'puncture_selection_ratio')
    m_keep = np.clip(1.0 - m_overlay - m_puncture, 0.0, 1.0)
    g_keep = np.clip(1.0 - g_overlay - g_puncture, 0.0, 1.0)

    fig, ax = plt.subplots(figsize=(10.2, 5.8), constrained_layout=True)
    ax.plot(x, m_keep, color='tab:blue', marker='o', linewidth=2.0, label='MAPPO keep')
    ax.plot(x, m_overlay, color='tab:orange', marker='s', linewidth=2.0, label='MAPPO overlay')
    ax.plot(x, m_puncture, color='tab:red', marker='x', linewidth=2.0, label='MAPPO puncture')
    ax.plot(x, g_keep, color='tab:cyan', marker='o', linewidth=1.8, linestyle='--', label=f'{baseline_label} keep')
    ax.plot(x, g_overlay, color='tab:brown', marker='s', linewidth=1.8, linestyle='--', label=f'{baseline_label} overlay')
    ax.plot(x, g_puncture, color='tab:gray', marker='d', linewidth=1.8, linestyle='--', label=f'{baseline_label} puncture')
    ax.set_title('Mode action share (including KEEP)')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Share of all Phase-A decisions')
    ax.grid(True, alpha=0.25)
    ax.legend(loc='best', fontsize=8, frameon=False, ncol=2)
    path = RESULTS_DIR / 'mode_action_share_compare.png'
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_mode_raw_vs_executed_compare(rl: Dict) -> Path:
    loads = np.asarray(rl.get('loads', []), dtype=float)
    total_users = (
        np.asarray(rl.get('embb_user_count', []), dtype=float)
        + np.asarray(rl.get('urllc_user_count', []), dtype=float)
    )
    use_total_users = bool(total_users.size == loads.size and np.all(total_users > 0.0))
    x = total_users if use_total_users else loads
    xlabel = 'Total users in system' if use_total_users else 'Average UE load per UAV'

    def _series(key: str) -> np.ndarray:
        arr = np.asarray(rl.get(key, []), dtype=float)
        return arr if arr.size == loads.size else np.zeros_like(loads, dtype=float)

    raw_overlay = _series('raw_overlay_ratio')
    raw_puncture = _series('raw_puncture_ratio')
    exe_overlay = _series('executed_overlay_ratio')
    exe_puncture = _series('executed_puncture_ratio')
    raw_keep = np.clip(1.0 - raw_overlay - raw_puncture, 0.0, 1.0)
    exe_keep = np.clip(1.0 - exe_overlay - exe_puncture, 0.0, 1.0)

    fig, ax = plt.subplots(figsize=(10.2, 5.8), constrained_layout=True)
    ax.plot(x, raw_keep, color='tab:blue', marker='o', linewidth=2.0, label='Raw keep')
    ax.plot(x, raw_overlay, color='tab:orange', marker='s', linewidth=2.0, label='Raw overlay')
    ax.plot(x, raw_puncture, color='tab:red', marker='x', linewidth=2.0, label='Raw puncture')
    ax.plot(x, exe_keep, color='tab:cyan', marker='o', linewidth=1.8, linestyle='--', label='Executed keep')
    ax.plot(x, exe_overlay, color='tab:brown', marker='s', linewidth=1.8, linestyle='--', label='Executed overlay')
    ax.plot(x, exe_puncture, color='tab:gray', marker='d', linewidth=1.8, linestyle='--', label='Executed puncture')
    ax.set_title('MAPPO raw vs executed mode share')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Share of all Phase-A decisions')
    ax.grid(True, alpha=0.25)
    ax.legend(loc='best', fontsize=8, frameon=False, ncol=2)
    path = RESULTS_DIR / 'mode_raw_vs_executed_compare.png'
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_greedy_candidate_rejection_debug(metrics: Dict, *, title_suffix: str = "") -> Path:
    """Plot greedy hard-feasible candidate rejection decomposition vs load."""
    loads = np.asarray(metrics.get("loads", []), dtype=float)
    if loads.size == 0:
        loads = np.asarray(DEFAULT_LOADS, dtype=float)
    rej_rel = np.asarray(metrics.get("greedy_hf_reject_reliability_ratio", np.zeros_like(loads)), dtype=float)
    rej_pwr = np.asarray(metrics.get("greedy_hf_reject_power_ratio", np.zeros_like(loads)), dtype=float)
    rej_min = np.asarray(metrics.get("greedy_hf_reject_min_rate_ratio", np.zeros_like(loads)), dtype=float)
    rej_share = np.asarray(metrics.get("greedy_hf_reject_share_cap_ratio", np.zeros_like(loads)), dtype=float)
    feasible = np.asarray(metrics.get("greedy_hf_feasible_ratio", np.zeros_like(loads)), dtype=float)
    eval_per_dec = np.asarray(metrics.get("greedy_hf_candidate_evaluated_per_decision", np.zeros_like(loads)), dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax = axes[0]
    ax.stackplot(
        loads,
        rej_rel,
        rej_pwr,
        rej_min,
        rej_share,
        labels=["reject: reliability", "reject: power", "reject: min-rate", "reject: share-cap"],
        alpha=0.85,
    )
    ax.plot(loads, feasible, color="black", linewidth=2.0, marker="o", label="feasible ratio")
    ax.set_title("Greedy candidate rejection decomposition")
    ax.set_xlabel("Average UE load per UAV")
    ax.set_ylabel("Ratio")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8, frameon=False)
    _report_plot_key_audit(
        "greedy_candidate_rejection_debug.ratios",
        [
            ("greedy_hf_reject_reliability_ratio", rej_rel),
            ("greedy_hf_reject_power_ratio", rej_pwr),
            ("greedy_hf_reject_min_rate_ratio", rej_min),
            ("greedy_hf_reject_share_cap_ratio", rej_share),
            ("greedy_hf_feasible_ratio", feasible),
        ],
    )

    ax = axes[1]
    ax.plot(loads, eval_per_dec, color="tab:blue", marker="s", linewidth=2.0, label="evaluated candidates / decision")
    ax.plot(
        loads,
        np.asarray(metrics.get("greedy_hf_candidate_feasible_per_decision", np.zeros_like(loads)), dtype=float),
        color="tab:green",
        marker="^",
        linewidth=2.0,
        label="feasible candidates / decision",
    )
    ax.set_title("Greedy candidate volume")
    ax.set_xlabel("Average UE load per UAV")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8, frameon=False)

    if title_suffix:
        fig.suptitle(str(title_suffix), fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = RESULTS_DIR / "greedy_candidate_rejection_debug.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_urllc_reliability_debug_fast(
    rl: Dict,
    baseline: Dict,
    *,
    baseline_label: str = "Greedy",
    reliability_target: float = 0.0,
) -> Path:
    """Fast debug plot: URLLC reliability (admitted + overall) vs load."""
    loads = np.asarray(rl.get('loads', []), dtype=float)
    fig, ax = plt.subplots(1, 1, figsize=(10.2, 4.6), constrained_layout=True)

    def _series(data: Dict, key: str) -> np.ndarray:
        arr = np.asarray(data.get(key, []), dtype=float)
        return arr if arr.size == loads.size else np.full_like(loads, np.nan, dtype=float)

    m_adm = _series(rl, 'admitted_urllc_reliability')
    g_adm = _series(baseline, 'admitted_urllc_reliability')
    m_all = _series(rl, 'urllc_reliability')
    g_all = _series(baseline, 'urllc_reliability')
    ax.plot(loads, m_adm, color='tab:orange', marker='s', markersize=6, linewidth=2.0, label='MAPPO admitted reliability (solid)')
    ax.plot(loads, g_adm, color='tab:brown', marker='o', markersize=6, linewidth=2.0, linestyle='--', label=f'{baseline_label} admitted reliability (dashed)')
    ax.plot(loads, m_all, color='tab:blue', marker='^', markersize=6, linewidth=2.0, label='MAPPO overall reliability (solid)')
    ax.plot(loads, g_all, color='tab:gray', marker='d', markersize=6, linewidth=2.0, linestyle='--', label=f'{baseline_label} overall reliability (dashed)')
    if reliability_target > 0.0:
        ax.axhline(float(reliability_target), color='tab:red', linestyle=':', linewidth=1.8, label='reliability target')
    _style(ax, 'URLLC reliability', 'Average UE load per UAV', 'Reliability')
    ax.legend(loc="best", fontsize=8, frameon=False, ncol=2)
    path = RESULTS_DIR / 'urllc_reliability_debug.png'
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_intercell_vs_load_debug_fast(
    rl: Dict,
    baseline: Dict,
    *,
    baseline_label: str = "Greedy",
) -> Path:
    """Fast debug plot: action-level intercell damage vs load (MAPPO vs selected baseline)."""
    loads = np.asarray(rl.get('loads', []), dtype=float)
    fig, axes = plt.subplots(1, 3, figsize=(18.0, 4.8), constrained_layout=True)
    axes = np.asarray(axes).reshape(1, 3)

    def _get(series: Dict, key: str) -> np.ndarray:
        y = np.asarray(series.get(key, []), dtype=float)
        return y if y.size == loads.size else np.zeros_like(loads, dtype=float)

    ax = axes[0, 0]
    m_mean = _get(rl, 'selected_action_intercell_cost_after_source_mask_mean')
    g_mean = _get(baseline, 'selected_action_intercell_cost_after_source_mask_mean')
    m_p95 = _get(rl, 'selected_action_intercell_cost_after_source_mask_p95')
    g_p95 = _get(baseline, 'selected_action_intercell_cost_after_source_mask_p95')
    ax.plot(loads, m_mean, color='tab:orange', marker='s', markersize=6, linewidth=2.0, label='MAPPO mean (solid)')
    ax.plot(loads, g_mean, color='tab:brown', marker='o', markersize=6, linewidth=2.0, linestyle='--', label=f'{baseline_label} mean (dashed)')
    ax.plot(loads, m_p95, color='tab:orange', marker='s', markersize=5, linewidth=1.4, linestyle=':', label='MAPPO p95 (dotted)')
    ax.plot(loads, g_p95, color='tab:brown', marker='o', markersize=5, linewidth=1.4, linestyle=':', label=f'{baseline_label} p95 (dotted)')
    _style(ax, 'Selected-action intercell cost (after source mask)', 'Average UE load per UAV', 'W')
    ax.legend(loc="best", fontsize=8, frameon=False)

    ax = axes[0, 1]
    ax.plot(loads, _get(rl, 'intercell_per_admitted_packet'), color='tab:orange', marker='s', markersize=6, linewidth=2.0, label='MAPPO (solid)')
    ax.plot(loads, _get(baseline, 'intercell_per_admitted_packet'), color='tab:brown', marker='o', markersize=6, linewidth=2.0, linestyle='--', label=f'{baseline_label} (dashed)')
    _style(ax, 'Intercell per admitted packet', 'Average UE load per UAV', 'W/admit')
    ax.legend(loc="best", fontsize=8, frameon=False)

    ax = axes[0, 2]
    m = _get(rl, 'intercell_per_admitted_packet')
    g = _get(baseline, 'intercell_per_admitted_packet')
    ratio = np.zeros_like(loads, dtype=float)
    mask = np.isfinite(m) & np.isfinite(g)
    ratio[mask] = (m[mask] - g[mask]) / np.maximum(g[mask], 1.0e-12)
    ax.plot(loads, ratio, color='tab:purple', marker='D', markersize=6, linewidth=2.0, label='(MAPPO-Greedy)/Greedy')
    ax.axhline(0.0, color='tab:gray', linestyle='--', linewidth=1.2, label='0')
    _style(ax, 'Intercell excess vs greedy (ratio)', 'Average UE load per UAV', 'Ratio')
    ax.legend(loc="best", fontsize=8, frameon=False)

    path = RESULTS_DIR / 'intercell_vs_load_debug.png'
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_intercell_rate_loss_debug_fast(
    rl: Dict,
    baseline: Dict,
    *,
    baseline_label: str = "Greedy",
) -> Path:
    """Fast debug plot: counterfactual eMBB rate loss due to inter-cell interference."""
    loads = np.asarray(rl.get('loads', []), dtype=float)
    fig, axes = plt.subplots(2, 3, figsize=(18.0, 8.4), constrained_layout=True)
    axes = np.asarray(axes).reshape(2, 3)

    def _get(series: Dict, key: str) -> np.ndarray:
        y = np.asarray(series.get(key, []), dtype=float)
        return y if y.size == loads.size else np.zeros_like(loads, dtype=float)

    def _plot(ax, key: str, title: str, ylabel: str, *, div: float = 1.0, mul: float = 1.0):
        m = _get(rl, key) / div * mul
        g = _get(baseline, key) / div * mul
        ax.plot(loads, m, color='tab:orange', marker='s', markersize=6, linewidth=2.0, label='MAPPO (solid)')
        ax.plot(loads, g, color='tab:brown', marker='o', markersize=6, linewidth=2.0, linestyle='--', label=f'{baseline_label} (dashed)')
        _style(ax, title, 'Average UE load per UAV', ylabel)
        ax.legend(loc="best", fontsize=8, frameon=False)

    _plot(axes[0, 0], 'embb_rate_with_intercell', 'eMBB rate (with intercell; legacy semantics)', 'Mbps', div=1.0e6)
    _plot(axes[0, 1], 'embb_rate_without_intercell_est', 'eMBB rate (no-intercell upper bound)', 'Mbps', div=1.0e6)
    _plot(axes[0, 2], 'embb_rate_loss_due_to_intercell', 'System rate gap vs no-intercell upper bound', 'Mbps', div=1.0e6)
    ax = axes[1, 0]
    m = _get(rl, 'embb_rate_loss_due_to_intercell_ratio')
    g = _get(baseline, 'embb_rate_loss_due_to_intercell_ratio')
    ax.plot(loads, m, color='tab:orange', marker='s', markersize=6, linewidth=2.0, label='MAPPO (solid)')
    ax.plot(loads, g, color='tab:brown', marker='o', markersize=6, linewidth=2.0, linestyle='--', label=f'{baseline_label} (dashed)')
    ax.axhline(0.7, color='tab:red', linestyle=':', linewidth=1.6, label='target 0.7')
    _style(ax, 'Rate loss ratio (loss/ideal)', 'Average UE load per UAV', 'Ratio')
    ax.legend(loc="best", fontsize=8, frameon=False)
    _plot(axes[1, 1], 'overlay_rate_loss_due_to_intercell', 'Overlay rate loss vs no-intercell upper bound', 'Mbps', div=1.0e6)
    _plot(axes[1, 2], 'intercell_rate_loss_with_same_puncture_mask', 'Intercell rate loss (same puncture mask)', 'Mbps', div=1.0e6)

    path = RESULTS_DIR / 'intercell_rate_loss_debug.png'
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_local_puncture_deduction_debug_fast(
    rl: Dict,
    baseline: Dict,
    *,
    baseline_label: str = "Greedy",
) -> Path:
    """Fast debug plot: local puncture airtime deduction diagnostics (raw vs corrected)."""
    loads = np.asarray(rl.get('loads', []), dtype=float)
    fig, axes = plt.subplots(1, 3, figsize=(18.0, 4.6), constrained_layout=True)
    axes = np.asarray(axes).reshape(1, 3)

    def _series_or_nan(series: Dict, key: str) -> np.ndarray:
        y = np.asarray(series.get(key, []), dtype=float)
        if y.size == loads.size:
            return y
        return np.full(loads.shape, np.nan, dtype=float)

    def _resolve_local_puncture_series(series: Dict, key: str) -> np.ndarray:
        # Prefer explicit diagnostics when available.
        direct = _series_or_nan(series, key)
        if np.isfinite(direct).any():
            return direct

        # Fallbacks for older/partial baseline payloads:
        # - Use eMBB rate-with-intercell as effective rate proxy.
        # - If loss is requested and we have raw/effective, derive as difference.
        if key == 'embb_rate_after_local_puncture_deduction':
            proxy = _series_or_nan(series, 'embb_rate_with_intercell')
            if not np.isfinite(proxy).any():
                proxy = _series_or_nan(series, 'embb_rate')
            return proxy

        if key == 'embb_rate_raw_before_local_puncture_deduction':
            proxy = _series_or_nan(series, 'embb_rate_raw_before_local_puncture_deduction')
            if np.isfinite(proxy).any():
                return proxy
            # If raw is missing, fall back to effective so we don't draw fake zeros.
            eff = _resolve_local_puncture_series(series, 'embb_rate_after_local_puncture_deduction')
            return eff

        if key == 'embb_rate_loss_due_to_local_puncture':
            raw = _resolve_local_puncture_series(series, 'embb_rate_raw_before_local_puncture_deduction')
            eff = _resolve_local_puncture_series(series, 'embb_rate_after_local_puncture_deduction')
            if np.isfinite(raw).any() and np.isfinite(eff).any():
                return raw - eff
            return np.full(loads.shape, np.nan, dtype=float)

        return np.full(loads.shape, np.nan, dtype=float)

    def _plot(ax, key: str, title: str, ylabel: str, *, div: float = 1.0):
        m = _resolve_local_puncture_series(rl, key) / div
        g = _resolve_local_puncture_series(baseline, key) / div
        ax.plot(loads, m, color='tab:orange', marker='s', markersize=6, linewidth=2.0, label='MAPPO (solid)')
        if np.isfinite(g).any():
            ax.plot(loads, g, color='tab:brown', marker='o', markersize=6, linewidth=2.0, linestyle='--', label=f'{baseline_label} (dashed)')
        else:
            ax.text(
                0.03,
                0.08,
                f'{baseline_label} metric unavailable',
                transform=ax.transAxes,
                fontsize=8,
                color='tab:brown',
            )
        _style(ax, title, 'Average UE load per UAV', ylabel)
        ax.legend(loc="best", fontsize=8, frameon=False)

    _plot(axes[0, 0], 'embb_rate_raw_before_local_puncture_deduction', 'Raw eMBB rate (before puncture deduction)', 'Mbps', div=1.0e6)
    _plot(axes[0, 1], 'embb_rate_after_local_puncture_deduction', 'Effective eMBB rate (after puncture deduction)', 'Mbps', div=1.0e6)
    _plot(axes[0, 2], 'embb_rate_loss_due_to_local_puncture', 'Rate loss due to local puncture', 'Mbps', div=1.0e6)

    path = RESULTS_DIR / 'local_puncture_deduction_debug.png'
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_phaseA_negative_only_debug_fast(
    rl: Dict,
    baseline: Dict,
    *,
    baseline_label: str = "Greedy",
) -> Path:
    """Fast debug plot: Phase-A negative-only power repair diagnostics."""
    loads = np.asarray(rl.get('loads', []), dtype=float)
    fig, axes = plt.subplots(1, 4, figsize=(20.0, 4.6), constrained_layout=True)
    axes = np.asarray(axes).reshape(1, 4)

    def _get(series: Dict, key: str) -> np.ndarray:
        y = np.asarray(series.get(key, []), dtype=float)
        return y if y.size == loads.size else np.zeros_like(loads, dtype=float)

    def _plot(ax, key: str, title: str, ylabel: str, *, mul: float = 1.0):
        m = _get(rl, key) * mul
        g = _get(baseline, key) * mul
        ax.plot(loads, m, color='tab:orange', marker='s', markersize=6, linewidth=2.0, label='MAPPO (solid)')
        ax.plot(loads, g, color='tab:brown', marker='o', markersize=6, linewidth=2.0, linestyle='--', label=f'{baseline_label} (dashed)')
        _style(ax, title, 'Average UE load per UAV', ylabel)
        ax.legend(loc="best", fontsize=8, frameon=False)

    _plot(axes[0, 0], 'phase_a_power_positive_clamped_to_zero_ratio', 'Phase-A positive clamp ratio', 'Ratio')
    _plot(axes[0, 1], 'phase_a_power_negative_executed_ratio', 'Phase-A negative executed ratio', 'Ratio')
    _plot(axes[0, 2], 'phase_a_power_total_power_reduction_mean', 'Phase-A total power reduction', 'W/decision')
    _plot(axes[0, 3], 'phase_a_power_intercell_reduction_mean', 'Phase-A intercell reduction', 'W/decision')

    path = RESULTS_DIR / 'phaseA_negative_only_debug.png'
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_service_recovery_debug_fast(
    rl: Dict,
    baseline: Dict,
    *,
    baseline_label: str = "Greedy",
) -> Path:
    """Fast debug plot: service recovery + owner effectiveness + admission/service tradeoff diagnostics."""
    loads = np.asarray(rl.get('loads', []), dtype=float)
    fig, axes = plt.subplots(2, 4, figsize=(20.0, 9.0), constrained_layout=True)
    axes = np.asarray(axes).reshape(2, 4)

    def _get(series: Dict, key: str) -> np.ndarray:
        y = np.asarray(series.get(key, []), dtype=float)
        return y if y.size == loads.size else np.zeros_like(loads, dtype=float)

    def _plot(ax, key: str, title: str, ylabel: str, *, div: float = 1.0, mul: float = 1.0):
        m = _get(rl, key) / div * mul
        g = _get(baseline, key) / div * mul
        ax.plot(loads, m, color='tab:orange', marker='s', markersize=6, linewidth=2.0, label='MAPPO (solid)')
        ax.plot(loads, g, color='tab:brown', marker='o', markersize=6, linewidth=2.0, linestyle='--', label=f'{baseline_label} (dashed)')
        _style(ax, title, 'Average UE load per UAV', ylabel)
        ax.legend(loc="best", fontsize=8, frameon=False)

    ax = axes[0, 0]
    m = _series_with_fallback(
        rl, loads, 'embb_service_ratio_after_puncture_deduction', ('embb_service_ratio',),
        context='service_recovery_debug.mappo.service_corrected'
    )
    g = _series_with_fallback(
        baseline, loads, 'embb_service_ratio_after_puncture_deduction', ('embb_service_ratio',),
        context='service_recovery_debug.greedy.service_corrected'
    )
    ax.plot(loads, m, color='tab:orange', marker='s', markersize=6, linewidth=2.0, label='MAPPO (solid)')
    ax.plot(loads, g, color='tab:brown', marker='o', markersize=6, linewidth=2.0, linestyle='--', label=f'{baseline_label} (dashed)')
    _style(ax, 'eMBB service ratio (corrected)', 'Average UE load per UAV', 'Ratio')
    ax.legend(loc="best", fontsize=8, frameon=False)

    ax = axes[0, 1]
    m = _series_with_fallback(
        rl, loads, 'embb_min_rate_satisfaction_after_puncture_deduction', ('embb_min_rate_satisfaction_ratio',),
        context='service_recovery_debug.mappo.minrate_corrected'
    )
    g = _series_with_fallback(
        baseline, loads, 'embb_min_rate_satisfaction_after_puncture_deduction', ('embb_min_rate_satisfaction_ratio',),
        context='service_recovery_debug.greedy.minrate_corrected'
    )
    ax.plot(loads, m, color='tab:orange', marker='s', markersize=6, linewidth=2.0, label='MAPPO (solid)')
    ax.plot(loads, g, color='tab:brown', marker='o', markersize=6, linewidth=2.0, linestyle='--', label=f'{baseline_label} (dashed)')
    _style(ax, 'eMBB min-rate satisfaction (corrected)', 'Average UE load per UAV', 'Ratio')
    ax.legend(loc="best", fontsize=8, frameon=False)
    _plot(axes[0, 2], 'avg_throughput_per_served_embb_user', 'Avg rate per served eMBB', 'Mbps', div=1.0e6)
    _plot(axes[0, 3], 'embb_served_user_count', 'Served eMBB user count', 'Users')

    _plot(axes[1, 0], 'phase0_owner_effective_service_gain_ratio', 'Owner effective service gain (vs snapshot)', 'Ratio')
    _plot(axes[1, 1], 'phase0_owner_effective_rate_gain_vs_snapshot_mean', 'Owner effective rate gain (vs snapshot)', 'Ratio')
    _plot(axes[1, 2], 'urllc_admission_over_service_tradeoff_penalty', 'Admission-over-service tradeoff penalty', 'Reward')

    ax = axes[1, 3]
    embb_n_m = _get(rl, 'embb_user_count')
    embb_n_g = _get(baseline, 'embb_user_count')
    avg_m = _get(rl, 'avg_throughput_per_served_embb_user')
    avg_g = _get(baseline, 'avg_throughput_per_served_embb_user')
    m_025 = embb_n_m * 0.25 * avg_m / 1.0e6
    m_030 = embb_n_m * 0.30 * avg_m / 1.0e6
    g_025 = embb_n_g * 0.25 * avg_g / 1.0e6
    g_030 = embb_n_g * 0.30 * avg_g / 1.0e6
    ax.plot(loads, m_025, color='tab:orange', linestyle='-', marker='s', markersize=6, linewidth=2.0, label='MAPPO est@service=0.25 (solid)')
    ax.plot(loads, m_030, color='tab:red', linestyle='-', marker='^', markersize=6, linewidth=2.0, label='MAPPO est@service=0.30 (solid)')
    ax.plot(loads, g_025, color='tab:brown', linestyle='--', marker='o', markersize=6, linewidth=2.0, label=f'{baseline_label} est@service=0.25 (dashed)')
    ax.plot(loads, g_030, color='tab:gray', linestyle='--', marker='d', markersize=6, linewidth=2.0, label=f'{baseline_label} est@service=0.30 (dashed)')
    _style(ax, 'Estimated eMBB throughput at service floors', 'Average UE load per UAV', 'Mbps')
    ax.legend(loc="best", fontsize=8, frameon=False)

    path = RESULTS_DIR / 'service_recovery_debug.png'
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_service_separation_debug_fast(
    rl: Dict,
    baseline: Dict,
    *,
    baseline_label: str = "Greedy",
) -> Path:
    """Fast debug plot: service/min-rate separation vs greedy + owner->service conversion."""
    loads = np.asarray(rl.get('loads', []), dtype=float)
    fig, axes = plt.subplots(2, 4, figsize=(20.0, 9.0), constrained_layout=True)
    axes = np.asarray(axes).reshape(2, 4)

    def _get(series: Dict, key: str) -> np.ndarray:
        y = np.asarray(series.get(key, []), dtype=float)
        if y.size == loads.size:
            return y
        if y.size == 2 * loads.size and loads.size > 0:
            try:
                return y.reshape(int(loads.size), 2).mean(axis=1)
            except Exception:
                pass
        return np.zeros_like(loads, dtype=float)

    m_srv = _get(rl, 'embb_service_ratio')
    g_srv = _get(baseline, 'embb_service_ratio')
    m_min = _get(rl, 'embb_min_rate_satisfaction_ratio')
    g_min = _get(baseline, 'embb_min_rate_satisfaction_ratio')
    m_served = _get(rl, 'embb_served_user_count')
    g_served = _get(baseline, 'embb_served_user_count')

    srv_gain = m_srv - g_srv
    min_gain = m_min - g_min
    served_gain = m_served - g_served

    # Panel 1: service
    ax = axes[0, 0]
    ax.plot(loads, m_srv, color='tab:orange', marker='s', markersize=6, linewidth=2.0, label='MAPPO service (solid)')
    ax.plot(loads, g_srv, color='tab:brown', marker='o', markersize=6, linewidth=2.0, linestyle='--', label=f'{baseline_label} service (dashed)')
    _style(ax, 'eMBB service ratio', 'Average UE load per UAV', 'Ratio')
    ax.legend(loc="best", fontsize=8, frameon=False)

    # Panel 2: service gain vs greedy
    ax = axes[0, 1]
    ax.plot(loads, srv_gain, color='tab:blue', marker='^', markersize=6, linewidth=2.0, label='service gain vs greedy')
    ax.axhline(0.0, color='tab:gray', linestyle=':', linewidth=1.2)
    _style(ax, 'Service gain vs greedy', 'Average UE load per UAV', 'Δ ratio')
    ax.legend(loc="best", fontsize=8, frameon=False)

    # Panel 3: min-rate satisfaction
    ax = axes[0, 2]
    ax.plot(loads, m_min, color='tab:green', marker='s', markersize=6, linewidth=2.0, label='MAPPO min-rate (solid)')
    ax.plot(loads, g_min, color='tab:gray', marker='o', markersize=6, linewidth=2.0, linestyle='--', label=f'{baseline_label} min-rate (dashed)')
    _style(ax, 'eMBB min-rate satisfaction', 'Average UE load per UAV', 'Ratio')
    ax.legend(loc="best", fontsize=8, frameon=False)

    # Panel 4: min-rate gain vs greedy
    ax = axes[0, 3]
    ax.plot(loads, min_gain, color='tab:purple', marker='d', markersize=6, linewidth=2.0, label='min-rate gain vs greedy')
    ax.axhline(0.0, color='tab:gray', linestyle=':', linewidth=1.2)
    _style(ax, 'Min-rate gain vs greedy', 'Average UE load per UAV', 'Δ ratio')
    ax.legend(loc="best", fontsize=8, frameon=False)

    # Panel 5: served user gain vs greedy
    ax = axes[1, 0]
    ax.plot(loads, served_gain, color='tab:orange', marker='s', markersize=6, linewidth=2.0, label='served user gain vs greedy')
    ax.axhline(0.0, color='tab:gray', linestyle=':', linewidth=1.2)
    _style(ax, 'Served-user gain vs greedy', 'Average UE load per UAV', 'Users')
    ax.legend(loc="best", fontsize=8, frameon=False)

    # Panel 6: owner executed change ratio
    ax = axes[1, 1]
    ax.plot(loads, _get(rl, 'phase0_owner_change_ratio_vs_snapshot_executed'), color='tab:blue', marker='^', markersize=6, linewidth=2.0, label='owner executed change ratio')
    _style(ax, 'Owner executed change', 'Average UE load per UAV', 'Ratio')
    ax.legend(loc="best", fontsize=8, frameon=False)

    # Panel 7: conversion ratio
    ax = axes[1, 2]
    ax.plot(loads, _get(rl, 'owner_change_service_conversion_ratio'), color='tab:red', marker='x', markersize=6, linewidth=2.0, label='owner->service conversion')
    _style(ax, 'Owner->service conversion', 'Average UE load per UAV', 'Ratio')
    ax.legend(loc="best", fontsize=8, frameon=False)

    axes[1, 3].axis('off')

    path = RESULTS_DIR / 'service_separation_debug.png'
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_service_target_debug_fast(
    rl: Dict,
    baseline: Dict,
    *,
    baseline_label: str = "Greedy",
    urllc_admission_floor: float = 0.0,
) -> Path:
    """Fast debug plot: service curriculum targets + admission floor (MAPPO vs selected baseline)."""
    loads = np.asarray(rl.get('loads', []), dtype=float)
    fig, axes = plt.subplots(2, 3, figsize=(18.0, 8.0), constrained_layout=True)
    axes = np.asarray(axes).reshape(2, 3)

    def _get(series: Dict, key: str) -> np.ndarray:
        y = np.asarray(series.get(key, []), dtype=float)
        return y if y.size == loads.size else np.zeros_like(loads, dtype=float)

    # Service ratio vs floor.
    ax = axes[0, 0]
    ax.plot(
        loads,
        _series_with_fallback(
            rl,
            loads,
            'embb_service_ratio_after_puncture_deduction',
            ('embb_service_ratio',),
            context='service_target_debug.mappo.service_corrected',
        ),
        color='tab:orange', marker='s', markersize=6, linewidth=2.0, label='MAPPO service (solid)'
    )
    ax.plot(
        loads,
        _series_with_fallback(
            baseline,
            loads,
            'embb_service_ratio_after_puncture_deduction',
            ('embb_service_ratio',),
            context='service_target_debug.greedy.service_corrected',
        ),
        color='tab:brown', marker='o', markersize=6, linewidth=2.0, linestyle='--', label=f'{baseline_label} service (dashed)'
    )
    srv_floor = _get(rl, 'terminal_embb_service_floor_used')
    if np.any(srv_floor > 0.0):
        ax.plot(loads, srv_floor, color='tab:red', linestyle=':', linewidth=2.0, label='service floor (curriculum)')
    _style(ax, 'eMBB service vs floor', 'Average UE load per UAV', 'Ratio')
    ax.legend(loc="best", fontsize=8, frameon=False)

    # Min-rate satisfaction vs floor.
    ax = axes[0, 1]
    ax.plot(
        loads,
        _series_with_fallback(
            rl,
            loads,
            'embb_min_rate_satisfaction_after_puncture_deduction',
            ('embb_min_rate_satisfaction_ratio',),
            context='service_target_debug.mappo.minrate_corrected',
        ),
        color='tab:orange', marker='s', markersize=6, linewidth=2.0, label='MAPPO min-rate (solid)'
    )
    ax.plot(
        loads,
        _series_with_fallback(
            baseline,
            loads,
            'embb_min_rate_satisfaction_after_puncture_deduction',
            ('embb_min_rate_satisfaction_ratio',),
            context='service_target_debug.greedy.minrate_corrected',
        ),
        color='tab:brown', marker='o', markersize=6, linewidth=2.0, linestyle='--', label=f'{baseline_label} min-rate (dashed)'
    )
    min_floor = _get(rl, 'terminal_embb_min_rate_floor_used')
    if np.any(min_floor > 0.0):
        ax.plot(loads, min_floor, color='tab:red', linestyle=':', linewidth=2.0, label='min-rate floor (curriculum)')
    _style(ax, 'Min-rate satisfaction vs floor', 'Average UE load per UAV', 'Ratio')
    ax.legend(loc="best", fontsize=8, frameon=False)

    # Served user count vs target.
    ax = axes[0, 2]
    ax.plot(loads, _get(rl, 'embb_served_user_count'), color='tab:orange', marker='s', markersize=6, linewidth=2.0, label='MAPPO served users (solid)')
    ax.plot(loads, _get(baseline, 'embb_served_user_count'), color='tab:brown', marker='o', markersize=6, linewidth=2.0, linestyle='--', label=f'{baseline_label} served users (dashed)')
    target = _get(rl, 'terminal_embb_served_user_target')
    if np.any(target > 0.0):
        ax.plot(loads, target, color='tab:blue', linestyle=':', linewidth=2.0, label='served-user target')
    _style(ax, 'Served eMBB users vs target', 'Average UE load per UAV', 'Users')
    ax.legend(loc="best", fontsize=8, frameon=False)

    # URLLC admission vs floor.
    ax = axes[1, 0]
    ax.plot(loads, _get(rl, 'urllc_admission'), color='tab:orange', marker='s', markersize=6, linewidth=2.0, label='MAPPO admission (solid)')
    ax.plot(loads, _get(baseline, 'urllc_admission'), color='tab:brown', marker='o', markersize=6, linewidth=2.0, linestyle='--', label=f'{baseline_label} admission (dashed)')
    _style(ax, 'URLLC admission', 'Average UE load per UAV', 'Ratio')
    ax.legend(loc="best", fontsize=8, frameon=False)

    # Aggregate throughput.
    ax = axes[1, 1]
    ax.plot(loads, _get(rl, 'embb_rate') / 1.0e6, color='tab:orange', marker='s', markersize=6, linewidth=2.0, label='MAPPO throughput (solid)')
    ax.plot(loads, _get(baseline, 'embb_rate') / 1.0e6, color='tab:brown', marker='o', markersize=6, linewidth=2.0, linestyle='--', label=f'{baseline_label} throughput (dashed)')
    _style(ax, 'Aggregate eMBB throughput', 'Average UE load per UAV', 'Mbps')
    ax.legend(loc="best", fontsize=8, frameon=False)

    # Leave one panel unused (reserved for future curriculum plots).
    axes[1, 2].axis('off')

    path = RESULTS_DIR / 'service_target_debug.png'
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_service_oracle_debug_fast(
    rl: Dict,
    baseline: Dict,
    *,
    baseline_label: str = "Greedy",
    num_uavs: int = 0,
    num_rbs: int = 0,
) -> Path:
    """Fast debug plot: feasibility ceilings (resource upper bounds) vs service floors.

    Note: this is a lightweight upper-bound estimate (not a full optimization oracle). It helps detect
    whether a service-floor target is likely infeasible given (num_uavs, num_rbs).
    """
    loads = np.asarray(rl.get('loads', []), dtype=float)
    fig, axes = plt.subplots(2, 2, figsize=(16.8, 8.0), constrained_layout=True)
    axes = np.asarray(axes).reshape(2, 2)

    def _get(series: Dict, key: str) -> np.ndarray:
        y = np.asarray(series.get(key, []), dtype=float)
        return y if y.size == loads.size else np.full_like(loads, np.nan, dtype=float)

    embb_n = _get(rl, 'embb_user_count')
    ceiling_count = float(max(int(num_uavs) * int(num_rbs), 0))
    ceiling_ratio = np.where(embb_n > 0.0, np.minimum(embb_n, ceiling_count) / np.maximum(embb_n, 1.0e-9), np.nan)

    # Service ratio vs floor + ceiling.
    ax = axes[0, 0]
    ax.plot(loads, _get(rl, 'embb_service_ratio'), color='tab:orange', marker='s', markersize=6, linewidth=2.0, label='MAPPO service (solid)')
    ax.plot(loads, _get(baseline, 'embb_service_ratio'), color='tab:brown', marker='o', markersize=6, linewidth=2.0, linestyle='--', label=f'{baseline_label} service (dashed)')
    floor = _get(rl, 'terminal_embb_service_floor_used')
    if np.any(np.isfinite(floor)) and np.nanmax(floor) > 0.0:
        ax.plot(loads, floor, color='tab:red', linestyle=':', linewidth=2.0, label='service floor')
    ax.plot(loads, ceiling_ratio, color='tab:blue', linestyle=':', linewidth=2.0, label='resource upper bound')
    _style(ax, 'Service ratio vs targets', 'Average UE load per UAV', 'Ratio')
    ax.legend(loc="best", fontsize=8, frameon=False)

    # Min-rate satisfaction vs floor + ceiling.
    ax = axes[0, 1]
    ax.plot(loads, _get(rl, 'embb_min_rate_satisfaction_ratio'), color='tab:orange', marker='s', markersize=6, linewidth=2.0, label='MAPPO min-rate (solid)')
    ax.plot(loads, _get(baseline, 'embb_min_rate_satisfaction_ratio'), color='tab:brown', marker='o', markersize=6, linewidth=2.0, linestyle='--', label=f'{baseline_label} min-rate (dashed)')
    floor = _get(rl, 'terminal_embb_min_rate_floor_used')
    if np.any(np.isfinite(floor)) and np.nanmax(floor) > 0.0:
        ax.plot(loads, floor, color='tab:red', linestyle=':', linewidth=2.0, label='min-rate floor')
    ax.plot(loads, ceiling_ratio, color='tab:blue', linestyle=':', linewidth=2.0, label='resource upper bound')
    _style(ax, 'Min-rate satisfaction vs targets', 'Average UE load per UAV', 'Ratio')
    ax.legend(loc="best", fontsize=8, frameon=False)

    # Served user count vs target + ceiling count.
    ax = axes[1, 0]
    ax.plot(loads, _get(rl, 'embb_served_user_count'), color='tab:orange', marker='s', markersize=6, linewidth=2.0, label='MAPPO served users (solid)')
    ax.plot(loads, _get(baseline, 'embb_served_user_count'), color='tab:brown', marker='o', markersize=6, linewidth=2.0, linestyle='--', label=f'{baseline_label} served users (dashed)')
    target = _get(rl, 'terminal_embb_served_user_target')
    if np.any(np.isfinite(target)) and np.nanmax(target) > 0.0:
        ax.plot(loads, target, color='tab:red', linestyle=':', linewidth=2.0, label='served-user target')
    ax.axhline(ceiling_count, color='tab:blue', linestyle=':', linewidth=2.0, label='resource upper bound')
    _style(ax, 'Served eMBB user count', 'Average UE load per UAV', 'Users')
    ax.legend(loc="best", fontsize=8, frameon=False)

    # Aggregate throughput (sanity).
    ax = axes[1, 1]
    ax.plot(loads, _get(rl, 'embb_rate') / 1.0e6, color='tab:orange', marker='s', markersize=6, linewidth=2.0, label='MAPPO throughput (solid)')
    ax.plot(loads, _get(baseline, 'embb_rate') / 1.0e6, color='tab:brown', marker='o', markersize=6, linewidth=2.0, linestyle='--', label=f'{baseline_label} throughput (dashed)')
    _style(ax, 'Aggregate eMBB throughput', 'Average UE load per UAV', 'Mbps')
    ax.legend(loc="best", fontsize=8, frameon=False)

    path = RESULTS_DIR / 'service_oracle_debug.png'
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def _plot_lambda_sweep_debug_fast_v2(series: Dict[str, object], baseline_label: str = "Greedy") -> Path:
    """Fast lambda sweep plot (v2): per-subplot legends + slot-based URLLC throughput keys."""
    x = np.asarray(series.get('lambda', []), dtype=float)
    rl = series.get('mappo', {}) if isinstance(series.get('mappo', {}), dict) else {}
    base = series.get('baseline', {}) if isinstance(series.get('baseline', {}), dict) else {}
    fig, axes = plt.subplots(3, 3, figsize=(18, 12.2), constrained_layout=True)
    axes = np.asarray(axes).reshape(3, 3)

    def _series(d: Dict, key: str) -> np.ndarray:
        arr = np.asarray(d.get(key, []), dtype=float)
        return arr if arr.size == x.size else np.zeros_like(x, dtype=float)

    def _plot2(ax, key: str, title: str, ylabel: str, *, scale_div: float = 1.0, scale_mul: float = 1.0):
        y_rl = _series(rl, key) / scale_div * scale_mul
        y_base = _series(base, key) / scale_div * scale_mul
        ax.plot(x, y_rl, color='tab:orange', marker='s', markersize=6, linewidth=2.0, label='MAPPO (solid)')
        ax.plot(x, y_base, color='tab:brown', marker='o', markersize=6, linewidth=2.0, linestyle='--', label=f'{baseline_label} (dashed)')
        _style(ax, title, 'Poisson λ (pkts/slot)', ylabel)
        ax.legend(loc="best", fontsize=8, frameon=False)

    _plot2(axes[0, 0], 'urllc_admission', 'URLLC admission ratio', 'Ratio')
    _plot2(axes[0, 1], 'embb_rate', 'Aggregate eMBB throughput', 'Mbps', scale_div=1e6)
    _plot2(axes[0, 2], 'total_power', 'Total transmit power', 'mW', scale_mul=1e3)

    ax = axes[1, 0]
    ax.plot(x, _series(rl, 'embb_service_ratio'), color='tab:orange', marker='s', markersize=6, linewidth=2.0, label='MAPPO service ratio (solid)')
    ax.plot(x, _series(base, 'embb_service_ratio'), color='tab:brown', marker='o', markersize=6, linewidth=2.0, linestyle='--', label=f'{baseline_label} service ratio (dashed)')
    ax.plot(x, _series(rl, 'embb_min_rate_satisfaction_ratio'), color='tab:green', marker='^', markersize=6, linewidth=2.0, label='MAPPO min-rate satisfaction (solid)')
    ax.plot(x, _series(base, 'embb_min_rate_satisfaction_ratio'), color='tab:gray', marker='d', markersize=6, linewidth=2.0, linestyle='--', label=f'{baseline_label} min-rate satisfaction (dashed)')
    _style(ax, 'eMBB service & min-rate satisfaction', 'Poisson λ (pkts/slot)', 'Ratio')
    ax.legend(loc="best", fontsize=8, frameon=False)

    ax = axes[1, 1]
    ax.plot(x, _series(rl, 'overlay_ratio'), color='tab:orange', marker='s', markersize=6, linewidth=2.0, label='MAPPO superpose/overlay (solid)')
    ax.plot(x, _series(rl, 'puncture_ratio'), color='tab:red', marker='x', markersize=6, linewidth=2.0, label='MAPPO puncture (solid)')
    ax.plot(x, _series(base, 'overlay_ratio'), color='tab:brown', marker='o', markersize=6, linewidth=2.0, linestyle='--', label=f'{baseline_label} superpose/overlay (dashed)')
    ax.plot(x, _series(base, 'puncture_ratio'), color='tab:gray', marker='d', markersize=6, linewidth=2.0, linestyle='--', label=f'{baseline_label} puncture (dashed)')
    _style(ax, 'Coexistence mode mix (excluding KEEP)', 'Poisson λ (pkts/slot)', 'Ratio')
    ax.legend(loc="best", fontsize=8, frameon=False)

    ax = axes[1, 2]
    m_tp = _series(rl, 'urllc_throughput_bps_slot_est')
    g_tp = _series(base, 'urllc_throughput_bps_slot_est')
    if (m_tp.size == x.size and np.all(m_tp == 0.0)) and np.any(_series(rl, 'scheduled_packets') > 0.0):
        bits = _series(rl, 'urllc_packet_bits_mean')
        dur = _series(rl, 'urllc_slot_duration_s')
        bits = bits if bits.size == x.size else np.full_like(x, 160.0, dtype=float)
        dur = dur if dur.size == x.size else np.full_like(x, 1.0e-3, dtype=float)
        m_tp = _series(rl, 'scheduled_packets') * bits / np.maximum(dur, 1.0e-12)
    if (g_tp.size == x.size and np.all(g_tp == 0.0)) and np.any(_series(base, 'scheduled_packets') > 0.0):
        bits = _series(base, 'urllc_packet_bits_mean')
        dur = _series(base, 'urllc_slot_duration_s')
        bits = bits if bits.size == x.size else np.full_like(x, 160.0, dtype=float)
        dur = dur if dur.size == x.size else np.full_like(x, 1.0e-3, dtype=float)
        g_tp = _series(base, 'scheduled_packets') * bits / np.maximum(dur, 1.0e-12)
    ax.plot(x, m_tp / 1.0e6, color='tab:orange', marker='s', markersize=6, linewidth=2.0, label='MAPPO URLLC throughput (solid)')
    ax.plot(x, g_tp / 1.0e6, color='tab:brown', marker='o', markersize=6, linewidth=2.0, linestyle='--', label=f'{baseline_label} URLLC throughput (dashed)')
    _style(ax, 'URLLC throughput (slot est.)', 'Poisson λ (pkts/slot)', 'Mbps')
    ax.legend(loc="best", fontsize=8, frameon=False)

    ax = axes[2, 0]
    ax.plot(x, _series(rl, 'embb_power') * 1e3, color='tab:orange', marker='s', markersize=6, linewidth=2.0, label='MAPPO eMBB power (solid)')
    ax.plot(x, _series(rl, 'urllc_power') * 1e3, color='tab:green', marker='^', markersize=6, linewidth=2.0, label='MAPPO URLLC power (solid)')
    ax.plot(x, _series(rl, 'total_power') * 1e3, color='tab:blue', marker='P', markersize=6, linewidth=2.0, label='MAPPO total power (solid)')
    ax.plot(x, _series(base, 'embb_power') * 1e3, color='tab:brown', marker='o', markersize=6, linewidth=2.0, linestyle='--', label=f'{baseline_label} eMBB power (dashed)')
    ax.plot(x, _series(base, 'urllc_power') * 1e3, color='tab:olive', marker='d', markersize=6, linewidth=2.0, linestyle='--', label=f'{baseline_label} URLLC power (dashed)')
    ax.plot(x, _series(base, 'total_power') * 1e3, color='tab:gray', marker='x', markersize=6, linewidth=2.0, linestyle='--', label=f'{baseline_label} total power (dashed)')
    _style(ax, 'Power split', 'Poisson λ (pkts/slot)', 'mW')
    ax.legend(loc="best", fontsize=8, frameon=False)

    axes[2, 1].axis('off')
    axes[2, 2].axis('off')

    load = float(series.get("load", float("nan")))
    embb_n = int(float(series.get("embb_user_count", 0.0) or 0.0))
    urllc_n = int(float(series.get("urllc_user_count", 0.0) or 0.0))
    baseline_mode = str(series.get("baseline_mode", "") or "")
    fig.suptitle(
        f"load={load:.1f}, eMBB:URLLC={embb_n}:{urllc_n}, baseline={baseline_mode}",
        fontsize=10,
    )

    path = RESULTS_DIR / 'lambda_sweep_debug.png'
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_lambda_sweep_debug_fast(series: Dict[str, object], baseline_label: str = "Greedy") -> Path:
    return _plot_lambda_sweep_debug_fast_v2(series, baseline_label=baseline_label)
    """Fast lambda sweep plot at a fixed load (MAPPO vs selected baseline)."""
    x = np.asarray(series.get('lambda', []), dtype=float)
    rl = series.get('mappo', {}) if isinstance(series.get('mappo', {}), dict) else {}
    base = series.get('baseline', {}) if isinstance(series.get('baseline', {}), dict) else {}
    fig, axes = plt.subplots(3, 3, figsize=(18, 12.2), constrained_layout=True)
    axes = np.asarray(axes).reshape(3, 3)

    def _series(d: Dict, key: str) -> np.ndarray:
        arr = np.asarray(d.get(key, []), dtype=float)
        return arr if arr.size == x.size else np.zeros_like(x, dtype=float)

    def _plot(ax, key: str, title: str, ylabel: str, *, scale_div: float = 1.0, scale_mul: float = 1.0):
        y_rl = np.asarray(rl.get(key, []), dtype=float)
        y_base = np.asarray(base.get(key, []), dtype=float)
        if y_rl.size != x.size:
            y_rl = np.zeros_like(x, dtype=float)
        if y_base.size != x.size:
            y_base = np.zeros_like(x, dtype=float)
        y_rl = y_rl / scale_div * scale_mul
        y_base = y_base / scale_div * scale_mul
        ax.plot(x, y_rl, color='tab:orange', marker='s', markersize=6, linewidth=2.0, label='MAPPO')
        ax.plot(x, y_base, color='tab:brown', marker='o', markersize=6, linewidth=2.0, label=baseline_label)
        _style(ax, title, 'Poisson λ (pkts/slot)', ylabel)

    _plot(axes[0, 0], 'urllc_admission', 'URLLC admission ratio', 'Ratio')
    _plot(axes[0, 1], 'embb_rate', 'Aggregate eMBB throughput', 'Mbps', scale_div=1e6)
    _plot(axes[0, 2], 'total_power', 'Total transmit power', 'mW', scale_mul=1e3)
    _plot(axes[1, 0], 'embb_service_ratio', 'eMBB service ratio', 'Ratio')

    # Mode ratio panel.
    ax = axes[1, 1]
    ax.plot(x, _series(rl, 'overlay_ratio'), color='tab:orange', marker='s', markersize=6, linewidth=2.0, label='MAPPO superpose/overlay')
    ax.plot(x, _series(rl, 'puncture_ratio'), color='tab:red', marker='x', markersize=6, linewidth=2.0, label='MAPPO puncture')
    ax.plot(x, _series(base, 'overlay_ratio'), color='tab:brown', marker='o', markersize=6, linewidth=2.0, linestyle='--', label=f'{baseline_label} superpose/overlay')
    ax.plot(x, _series(base, 'puncture_ratio'), color='tab:gray', marker='d', markersize=6, linewidth=2.0, linestyle='--', label=f'{baseline_label} puncture')
    _style(ax, 'Coexistence mode mix (excluding KEEP)', 'Poisson λ (pkts/slot)', 'Ratio')

    # URLLC throughput.
    ax = axes[1, 2]
    ax.plot(x, _series(rl, 'urllc_throughput_bps_est') / 1.0e6, color='tab:orange', marker='s', markersize=6, linewidth=2.0, label='MAPPO URLLC throughput')
    ax.plot(x, _series(base, 'urllc_throughput_bps_est') / 1.0e6, color='tab:brown', marker='o', markersize=6, linewidth=2.0, label=f'{baseline_label} URLLC throughput')
    _style(ax, 'URLLC throughput (est.)', 'Poisson λ (pkts/slot)', 'Mbps')

    # Power split.
    ax = axes[2, 0]
    ax.plot(x, _series(rl, 'embb_power') * 1e3, color='tab:orange', marker='s', markersize=6, linewidth=2.0, label='MAPPO eMBB power')
    ax.plot(x, _series(rl, 'urllc_power') * 1e3, color='tab:green', marker='^', markersize=6, linewidth=2.0, label='MAPPO URLLC power')
    ax.plot(x, _series(rl, 'total_power') * 1e3, color='tab:blue', marker='P', markersize=6, linewidth=2.0, label='MAPPO total power')
    ax.plot(x, _series(base, 'embb_power') * 1e3, color='tab:brown', marker='o', markersize=6, linewidth=2.0, linestyle='--', label=f'{baseline_label} eMBB power')
    ax.plot(x, _series(base, 'urllc_power') * 1e3, color='tab:olive', marker='d', markersize=6, linewidth=2.0, linestyle='--', label=f'{baseline_label} URLLC power')
    ax.plot(x, _series(base, 'total_power') * 1e3, color='tab:gray', marker='x', markersize=6, linewidth=2.0, linestyle='--', label=f'{baseline_label} total power')
    _style(ax, 'Power split', 'Poisson λ (pkts/slot)', 'mW')

    # Hide unused axes.
    axes[2, 1].axis('off')
    axes[2, 2].axis('off')

    load = float(series.get("load", float("nan")))
    embb_n = int(float(series.get("embb_user_count", 0.0) or 0.0))
    urllc_n = int(float(series.get("urllc_user_count", 0.0) or 0.0))
    baseline_mode = str(series.get("baseline_mode", "") or "")
    fig.suptitle(
        f"load={load:.1f}, eMBB:URLLC={embb_n}:{urllc_n}, baseline={baseline_mode}",
        fontsize=10,
    )

    handle_label = {}
    for row in axes:
        for ax in row:
            h, l = ax.get_legend_handles_labels()
            for hh, ll in zip(h, l):
                handle_label.setdefault(ll, hh)
            if ax.get_legend():
                ax.legend_.remove()
    handles = list(handle_label.values())
    labels = list(handle_label.keys())
    fig.legend(handles, labels, loc='upper center', ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.01))
    path = RESULTS_DIR / 'lambda_sweep_debug.png'
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_owner_effective_debug_fast(rl: Dict) -> Path:
    """Fast debug plot: owner-change execution attribution (MAPPO only)."""
    loads = np.asarray(rl.get('loads', []), dtype=float)
    fig, ax = plt.subplots(1, 1, figsize=(9.6, 4.4), constrained_layout=True)

    def _get(key: str) -> np.ndarray:
        y = np.asarray(rl.get(key, []), dtype=float)
        return y if y.size == loads.size else np.zeros_like(loads, dtype=float)

    ax.plot(loads, _get('phase0_owner_change_ratio_vs_snapshot_raw'), color='tab:blue', marker='^', markersize=6, linewidth=2.0, label='raw owner change ratio')
    ax.plot(loads, _get('phase0_owner_change_ratio_vs_snapshot_executed'), color='tab:orange', marker='s', markersize=6, linewidth=2.0, label='executed owner change ratio')
    ax.plot(loads, _get('phase0_owner_changed_and_effective_ratio'), color='tab:purple', marker='d', markersize=6, linewidth=2.0, label='changed+effective ratio')
    ax.plot(loads, _get('phase0_owner_changed_but_unserved_ratio'), color='tab:red', marker='x', markersize=6, linewidth=2.0, label='changed but unserved ratio')
    same_exec = _get('phase0_owner_same_as_snapshot_ratio')
    ax.plot(loads, same_exec, color='tab:green', marker='o', markersize=6, linewidth=2.0, label='same-as-snapshot ratio (executed)')
    ax.plot(loads, _get('phase0_owner_restored_to_snapshot_ratio'), color='tab:cyan', marker='P', markersize=6, linewidth=1.8, label='restored-to-snapshot ratio')
    _report_plot_key_audit(
        "owner_effective_debug",
        [
            ("phase0_owner_change_ratio_vs_snapshot_raw", _get('phase0_owner_change_ratio_vs_snapshot_raw')),
            ("phase0_owner_change_ratio_vs_snapshot_executed", _get('phase0_owner_change_ratio_vs_snapshot_executed')),
            ("phase0_owner_changed_and_effective_ratio", _get('phase0_owner_changed_and_effective_ratio')),
            ("phase0_owner_changed_but_unserved_ratio", _get('phase0_owner_changed_but_unserved_ratio')),
            ("phase0_owner_same_as_snapshot_ratio", same_exec),
            ("phase0_owner_restored_to_snapshot_ratio", _get('phase0_owner_restored_to_snapshot_ratio')),
        ],
    )
    _style(ax, 'Owner execution attribution', 'Average UE load per UAV', 'Ratio')
    ax.legend(frameon=False, ncol=2)
    path = RESULTS_DIR / 'owner_effective_debug.png'
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_phaseA_power_debug_fast(rl: Dict) -> Path:
    """Fast debug plot: Phase-A power activation/shrink summary (MAPPO only)."""
    loads = np.asarray(rl.get('loads', []), dtype=float)
    fig, axes = plt.subplots(2, 5, figsize=(26.0, 7.8), constrained_layout=True)
    axes = np.asarray(axes).reshape(2, 5)

    def _get(key: str) -> np.ndarray:
        y = np.asarray(rl.get(key, []), dtype=float)
        if y.size == loads.size:
            return y
        if y.size == 2 * loads.size and loads.size > 0:
            try:
                return y.reshape(int(loads.size), 2).mean(axis=1)
            except Exception:
                pass
        if y.size <= 0 or loads.size <= 0:
            return np.zeros_like(loads, dtype=float)
        # Robust fallback: resample by index so fast-debug plots never render empty.
        x_old = np.linspace(0.0, 1.0, num=int(y.size), dtype=float)
        x_new = np.linspace(0.0, 1.0, num=int(loads.size), dtype=float)
        return np.interp(x_new, x_old, y).astype(float)

    axes[0, 0].plot(loads, _get('phase_a_raw_embb_power_nonzero_ratio'), color='tab:blue', marker='o', markersize=6, linewidth=2.0)
    _style(axes[0, 0], 'Phase-A raw nonzero ratio', 'Average UE load per UAV', 'Ratio')

    axes[0, 1].plot(loads, _get('phase_a_executed_embb_power_nonzero_ratio'), color='tab:orange', marker='s', markersize=6, linewidth=2.0)
    _style(axes[0, 1], 'Phase-A executed nonzero ratio', 'Average UE load per UAV', 'Ratio')

    axes[0, 2].plot(loads, _get('phase_a_embb_power_effective_nonzero_ratio'), color='tab:green', marker='^', markersize=6, linewidth=2.0)
    _style(axes[0, 2], 'Phase-A effective nonzero ratio', 'Average UE load per UAV', 'Ratio')

    axes[0, 3].plot(loads, _get('phase_a_embb_power_raw_saturation_ratio'), color='tab:red', marker='x', markersize=6, linewidth=2.0)
    _style(axes[0, 3], 'Phase-A raw saturation ratio', 'Average UE load per UAV', 'Ratio')

    axes[0, 4].plot(loads, _get('phase_a_embb_power_cap_hit_ratio'), color='tab:orange', marker='s', markersize=6, linewidth=2.0, label='cap hit')
    axes[0, 4].plot(loads, _get('phase_a_embb_power_floor_hit_ratio'), color='tab:blue', marker='o', markersize=6, linewidth=2.0, label='floor hit')
    _style(axes[0, 4], 'Phase-A cap/floor hit ratio', 'Average UE load per UAV', 'Ratio')
    axes[0, 4].legend(frameon=False, fontsize=8, loc='best')

    axes[1, 0].plot(loads, _get('phase_a_embb_power_write_ratio'), color='tab:purple', marker='d', markersize=6, linewidth=2.0)
    _style(axes[1, 0], 'Phase-A write ratio', 'Average UE load per UAV', 'Ratio')

    axes[1, 1].plot(loads, _get('phase_a_embb_power_final_std'), color='tab:green', marker='^', markersize=6, linewidth=2.0)
    _style(axes[1, 1], 'Phase-A final delta std', 'Average UE load per UAV', 'Std')

    axes[1, 2].plot(loads, _get('phase_a_embb_power_cellwise_diversity'), color='tab:cyan', marker='v', markersize=6, linewidth=2.0)
    _style(axes[1, 2], 'Phase-A cellwise diversity', 'Average UE load per UAV', 'Std')

    axes[1, 3].plot(loads, _get('phase_a_embb_power_mean_abs_change'), color='tab:red', marker='x', markersize=6, linewidth=2.0)
    _style(axes[1, 3], 'Phase-A mean abs executed change', 'Average UE load per UAV', 'Abs')

    # Suppression breakdown (stacked; ratios are per decision cell, so sum<1 means "not suppressed / nonzero").
    ax = axes[1, 4]
    reasons = [
        ('inactive', 'phase_a_embb_power_zeroed_inactive_head_ratio', 'tab:gray'),
        ('no_eMBB', 'phase_a_embb_power_zeroed_no_embb_active_ratio', 'tab:blue'),
        ('no_owner', 'phase_a_embb_power_zeroed_no_owner_ratio', 'tab:purple'),
        ('inv_owner', 'phase_a_embb_power_zeroed_invalid_owner_ratio', 'tab:pink'),
        ('cap', 'phase_a_embb_power_zeroed_cap_projection_ratio', 'tab:orange'),
        ('floor', 'phase_a_embb_power_zeroed_floor_projection_ratio', 'tab:cyan'),
        ('no_cand', 'phase_a_embb_power_zeroed_no_candidate_ratio', 'tab:green'),
        ('keep_blk', 'phase_a_embb_power_zeroed_keep_mode_ratio', 'tab:brown'),
        ('unknown', 'phase_a_embb_power_zeroed_unknown_ratio', 'tab:red'),
    ]
    bottom = np.zeros_like(loads, dtype=float)
    for label, key, color in reasons:
        y = np.clip(_get(key), 0.0, 1.0)
        ax.bar(loads, y, bottom=bottom, width=0.9, color=color, alpha=0.75, label=label)
        bottom = bottom + y
    _style(ax, 'Phase-A suppress breakdown', 'Average UE load per UAV', 'Ratio')
    ax.set_ylim(0.0, 1.05)
    ax.legend(frameon=False, fontsize=7, loc='upper right')

    path = RESULTS_DIR / 'phaseA_power_debug.png'
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_owner_map_slot_debug(
    *,
    greedy_owner_map: np.ndarray,
    policy_owner_map: np.ndarray,
    experiment_line: str,
    load: float,
    poisson_rate: float,
    baseline_label: str,
) -> Path:
    """(UAV,RB) owner maps for a single episode/slot: greedy snapshot vs MAPPO, plus a diff map."""
    greedy = np.asarray(greedy_owner_map, dtype=int)
    policy = np.asarray(policy_owner_map, dtype=int)
    if greedy.ndim != 2 or policy.ndim != 2:
        raise ValueError(f"Expected 2D owner maps, got greedy={greedy.shape}, policy={policy.shape}")
    if greedy.shape != policy.shape:
        raise ValueError(f"Owner map shape mismatch: greedy={greedy.shape}, policy={policy.shape}")

    diff = (greedy != policy).astype(int)
    num_uavs, num_rbs = int(greedy.shape[0]), int(greedy.shape[1])
    max_owner = int(max(int(np.max(greedy, initial=0)), int(np.max(policy, initial=0)), 0))
    cmap = plt.get_cmap('tab20', max(max_owner + 1, 2))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), constrained_layout=True)
    im0 = axes[0].imshow(greedy, aspect='auto', interpolation='nearest', cmap=cmap, vmin=0, vmax=max(max_owner, 1))
    axes[0].set_title('Greedy snapshot owner map')
    axes[0].set_xlabel('RB index')
    axes[0].set_ylabel('UAV index')

    im1 = axes[1].imshow(policy, aspect='auto', interpolation='nearest', cmap=cmap, vmin=0, vmax=max(max_owner, 1))
    axes[1].set_title('MAPPO executed owner map')
    axes[1].set_xlabel('RB index')
    axes[1].set_ylabel('UAV index')

    im2 = axes[2].imshow(diff, aspect='auto', interpolation='nearest', cmap='Greys', vmin=0, vmax=1)
    axes[2].set_title('Difference (1=different)')
    axes[2].set_xlabel('RB index')
    axes[2].set_ylabel('UAV index')

    for ax in axes:
        ax.set_xticks(np.linspace(0, max(num_rbs - 1, 0), num=min(num_rbs, 9), dtype=int))
        ax.set_yticks(np.linspace(0, max(num_uavs - 1, 0), num=min(num_uavs, 9), dtype=int))

    fig.suptitle(
        f"{experiment_label(experiment_line)} | load={float(load):.1f} | poisson_rate={float(poisson_rate):.0f} pkts/slot | baseline={baseline_label}",
        fontsize=10,
    )
    fig.colorbar(im1, ax=axes[:2], shrink=0.85, pad=0.02, label='eMBB owner id')
    fig.colorbar(im2, ax=axes[2], shrink=0.85, pad=0.02, ticks=[0, 1])

    path = RESULTS_DIR / 'owner_map_slot_debug.png'
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_owner_vs_service_debug_fast(rl: Dict, baseline: Dict, baseline_label: str = "Greedy") -> Path:
    """Fast debug plot: tie owner executed change to service/admission separation."""
    loads = np.asarray(rl.get('loads', []), dtype=float)
    fig, ax = plt.subplots(1, 1, figsize=(10.6, 4.4), constrained_layout=True)

    def _get(series: Dict, key: str) -> np.ndarray:
        y = np.asarray(series.get(key, []), dtype=float)
        return y if y.size == loads.size else np.zeros_like(loads, dtype=float)

    # Owner executed change (MAPPO)
    ax.plot(loads, _get(rl, 'phase0_owner_change_ratio_vs_snapshot_executed'), color='tab:purple', marker='d', markersize=6, linewidth=2.0, label='MAPPO owner change (exe)')

    # Service ratio (MAPPO vs baseline)
    ax.plot(loads, _get(rl, 'embb_service_ratio'), color='tab:orange', marker='s', markersize=6, linewidth=2.0, label='MAPPO service ratio')
    ax.plot(loads, _get(baseline, 'embb_service_ratio'), color='tab:brown', marker='o', markersize=6, linewidth=2.0, label=f'{baseline_label} service ratio')

    # URLLC admission (MAPPO vs baseline)
    ax.plot(loads, _get(rl, 'urllc_admission'), color='tab:orange', linestyle='--', marker='s', markersize=6, linewidth=2.0, label='MAPPO URLLC admission')
    ax.plot(loads, _get(baseline, 'urllc_admission'), color='tab:brown', linestyle='--', marker='o', markersize=6, linewidth=2.0, label=f'{baseline_label} URLLC admission')
    _report_plot_key_audit(
        "owner_vs_service_debug",
        [
            ("phase0_owner_change_ratio_vs_snapshot_executed", _get(rl, 'phase0_owner_change_ratio_vs_snapshot_executed')),
            ("sr_mappo.embb_service_ratio", _get(rl, 'embb_service_ratio')),
            ("greedy.embb_service_ratio", _get(baseline, 'embb_service_ratio')),
            ("sr_mappo.urllc_admission", _get(rl, 'urllc_admission')),
            ("greedy.urllc_admission", _get(baseline, 'urllc_admission')),
        ],
    )

    _style(ax, 'Owner vs Service separation', 'Average UE load per UAV', 'Ratio')
    ax.legend(frameon=False, ncol=2)
    path = RESULTS_DIR / 'owner_vs_service_debug.png'
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_positive_rate_decomposition_debug(rl: Dict, baseline: Dict, baseline_label: str = "Greedy") -> Path:
    """Fast debug plot to explain why positive-rate ratio overlaps (MAPPO vs baseline)."""
    loads = np.asarray(rl.get('loads', []), dtype=float)
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.2), constrained_layout=True)

    def _series(data: Dict, key: str, *, default: float = float("nan")) -> np.ndarray:
        raw = data.get(key, [])
        if isinstance(raw, (int, float, np.floating, np.integer)):
            return np.full(loads.shape, float(raw), dtype=float)
        arr = np.asarray(raw, dtype=float) if raw is not None else np.asarray([], dtype=float)
        if arr.size == loads.size:
            return arr
        if arr.size == 0:
            return np.full(loads.shape, float(default), dtype=float)
        # Best-effort: truncate or pad to match load grid.
        if arr.size > loads.size:
            return arr[: loads.size]
        padded = np.full(loads.shape, float(default), dtype=float)
        padded[: arr.size] = arr
        return padded

    panels = [
        ('eMBB positive-rate ratio', 'embb_positive_rate_ratio', 1.0, 'Ratio'),
        ('eMBB service ratio', 'embb_service_ratio', 1.0, 'Ratio'),
        ('Avg throughput per served eMBB user', 'avg_throughput_per_served_embb_user', 1e6, 'Mbps'),
        ('Phase-0 owner change vs snapshot (exe)', 'phase0_owner_change_ratio_vs_snapshot_executed', 1.0, 'Ratio'),
        ('Phase-A power eff nonzero', 'phase_a_embb_power_effective_nonzero_ratio', 1.0, 'Ratio'),
    ]

    for ax, (title, key, scale, ylabel) in zip(axes, panels):
        y_rl = _series(rl, key, default=float("nan"))
        # Greedy baselines do not define owner-change / Phase-A power diagnostics; treat them as 0.
        base_default = 0.0 if key in {"phase0_owner_change_ratio_vs_snapshot_executed", "phase_a_embb_power_effective_nonzero_ratio"} else float("nan")
        y_base = _series(baseline, key, default=base_default)
        if scale != 1.0:
            y_rl = y_rl / scale
            y_base = y_base / scale
        ax.plot(loads, y_rl, color='tab:orange', marker='s', markersize=6, linewidth=2.2, alpha=0.95, zorder=4, label='MAPPO')
        ax.plot(loads, y_base, color='tab:brown', marker='o', markersize=5.5, linewidth=1.8, alpha=0.75, zorder=3, label=baseline_label)
        _style(ax, title, 'Average UE load per UAV', ylabel)
        _report_plot_key_audit(
            f"positive_rate_decomposition.{key}",
            [(f"sr_mappo.{key}", y_rl), (f"greedy.{key}", y_base)],
        )
        if key == "embb_positive_rate_ratio":
            srv_rl = _series(rl, "embb_service_ratio", default=float("nan"))
            srv_base = _series(baseline, "embb_service_ratio", default=float("nan"))
            _report_overlap_note(ax, y_rl, srv_rl)
            _report_overlap_note(ax, y_base, srv_base)

    handles, labels = axes[0].get_legend_handles_labels()
    for ax in axes:
        if ax.get_legend():
            ax.legend_.remove()
    fig.legend(handles, labels, loc='upper center', ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.10))
    path = RESULTS_DIR / 'positive_rate_decomposition_debug.png'
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_mode_diagnostics(
    greedy_original: Dict,
    greedy_original_lite: Optional[Dict],
    greedy_original_normal_v2: Optional[Dict],
    greedy_myopic: Optional[Dict],
    greedy_matched: Dict,
    greedy_throughput_feasible: Optional[Dict],
    greedy_throughput_only: Optional[Dict],
    greedy_channel_only: Optional[Dict],
    rl: Dict,
    greedy_baseline_mode: str = "original",
):
    fig, axes = plt.subplots(3, 2, figsize=(14, 14), constrained_layout=True)
    loads = rl['loads']
    baseline_data, baseline_label, baseline_color, baseline_marker = _selected_baseline_source(
        greedy_original,
        greedy_original_lite,
        greedy_original_normal_v2,
        greedy_myopic,
        greedy_matched,
        greedy_throughput_feasible,
        greedy_throughput_only,
        greedy_channel_only,
        greedy_baseline_mode,
    )
    for data, prefix, color, marker in (
        (baseline_data, baseline_label, baseline_color, baseline_marker),
        (rl, 'MAPPO', 'tab:orange', 's'),
    ):
        if data is None:
            continue
        axes[0, 0].plot(loads, data['overlay_ratio'], marker=marker, color=color, linestyle='-', label=f'{prefix} overlay ratio')
        axes[0, 0].plot(loads, data['puncture_ratio'], marker=marker, color=color, linestyle='--', label=f'{prefix} puncture ratio')
    _style(axes[0, 0], 'Coexistence mode mix (excluding KEEP)', 'Average UE load per UAV', 'Ratio')
    axes[0, 0].legend(fontsize=10, ncol=2)

    for data, label, color, marker in _comparison_series(greedy_baseline_mode):
        source = _comparison_source(
            data,
            greedy_original,
            greedy_original_lite,
            greedy_original_normal_v2,
            greedy_myopic,
            greedy_matched,
            greedy_throughput_feasible,
            greedy_throughput_only,
            greedy_channel_only,
            rl,
        )
        if source is None:
            continue
        axes[0, 1].plot(loads, np.asarray(source['avg_puncture_loss']) / 1e6, marker=marker, color=color, label=label)
    _style(axes[0, 1], 'Average eMBB loss per puncture', 'Average UE load per UAV', 'Mbps loss/action')
    axes[0, 1].legend()

    for data, label, color, marker in _comparison_series(greedy_baseline_mode):
        source = _comparison_source(
            data,
            greedy_original,
            greedy_original_lite,
            greedy_original_normal_v2,
            greedy_myopic,
            greedy_matched,
            greedy_throughput_feasible,
            greedy_throughput_only,
            greedy_channel_only,
            rl,
        )
        if source is None:
            continue
        axes[1, 0].plot(loads, source['avg_overlay_retention'], marker=marker, color=color, label=label)
    _style(axes[1, 0], 'Average eMBB retention under overlay', 'Average UE load per UAV', 'Retention ratio')
    axes[1, 0].legend()

    for source, prefix, color, marker in (
        (baseline_data, baseline_label, baseline_color, baseline_marker),
        (rl, 'MAPPO', 'tab:orange', 's'),
    ):
        if source is None:
            continue
        axes[1, 1].plot(loads, source['overlay_candidate_pairs'], marker=marker, color=color, label=f'{prefix} candidate')
        axes[1, 1].plot(loads, source['overlay_feasible_pairs'], marker=marker, color=color, linestyle='--', label=f'{prefix} feasible')
        axes[1, 1].plot(loads, source['overlay_selected_pairs'], marker=marker, color=color, linestyle=':', label=f'{prefix} selected')
    _style(axes[1, 1], 'Overlay feasibility / selection', 'Average UE load per UAV', 'Pairs per slot')
    axes[1, 1].legend(fontsize=8)

    for source, prefix, color, marker in (
        (baseline_data, baseline_label, baseline_color, baseline_marker),
        (rl, 'MAPPO', 'tab:orange', 's'),
    ):
        if source is None:
            continue
        axes[2, 0].plot(loads, np.asarray(source['embb_power']) * 1e3, marker=marker, color=color, label=f'{prefix} eMBB')
        axes[2, 0].plot(loads, np.asarray(source['urllc_power']) * 1e3, marker=marker, color=color, linestyle='--', label=f'{prefix} URLLC')
    _style_power_axis(axes[2, 0], 'Power split by traffic type', 'Average UE load per UAV', 'mW')
    axes[2, 0].legend(fontsize=8)

    axes[2, 1].plot(loads, rl['shield_correction_ratio'], marker='s', label='Shield correction')
    axes[2, 1].plot(loads, rl['joint_reliability_rewrite_ratio'], marker='D', linestyle='--', label='Joint rewrite')
    axes[2, 1].plot(loads, rl['mode_correction_ratio'], marker='v', linestyle='--', label='Mode corrected')
    axes[2, 1].plot(loads, rl['packet_invalid_ratio'], marker='P', linestyle=':', label='Packet invalid')
    axes[2, 1].plot(loads, rl['mask_invalid_ratio'], marker='X', linestyle=':', label='Mask invalid')
    axes[2, 1].plot(loads, rl['collision_rewrite_ratio'], marker='^', label='Collision rewrite')
    axes[2, 1].plot(loads, rl['fallback_ratio'], marker='o', label='Fallback ratio')
    _style(axes[2, 1], 'Shield / fallback activation', 'Average UE load per UAV', 'Ratio')
    axes[2, 1].legend(fontsize=7, ncol=2)
    fig.suptitle(f'Mode Diagnostics | {baseline_label} vs MAPPO | Selected baseline: {baseline_label}', fontsize=14)

    path = RESULTS_DIR / '02_mode_diagnostics_vs_load.png'
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_upper_bounds_and_frontier(
    greedy_original: Dict,
    greedy_original_lite: Optional[Dict],
    greedy_original_normal_v2: Optional[Dict],
    greedy_myopic: Optional[Dict],
    greedy_matched: Dict,
    greedy_throughput_feasible: Optional[Dict],
    greedy_throughput_only: Optional[Dict],
    greedy_channel_only: Optional[Dict],
    rl: Dict,
    embb_only_ceiling: Dict,
    throughput_oracle: Dict,
    frontier_bundle: Dict[float, Dict],
    greedy_baseline_mode: str = "original",
):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    loads = rl['loads']
    baseline_data, baseline_label, baseline_color, baseline_marker = _selected_baseline_source(
        greedy_original,
        greedy_original_lite,
        greedy_original_normal_v2,
        greedy_myopic,
        greedy_matched,
        greedy_throughput_feasible,
        greedy_throughput_only,
        greedy_channel_only,
        greedy_baseline_mode,
    )
    reference_series = [
        (baseline_data, baseline_label, baseline_color, baseline_marker),
        (rl, 'MAPPO', 'tab:orange', 's'),
    ]
    if greedy_throughput_feasible is not None and _normalize_baseline_mode(greedy_baseline_mode) != "throughput_feasible_oracle":
        reference_series.append((greedy_throughput_feasible, 'Throughput-feasible Oracle', 'tab:red', '*'))
    if greedy_throughput_only is not None and _normalize_baseline_mode(greedy_baseline_mode) != "throughput_only_greedy":
        reference_series.append((greedy_throughput_only, 'Throughput-only Greedy (eMBB-only ceiling)', 'tab:purple', 'D'))
    if embb_only_ceiling is not None:
        reference_series.append((embb_only_ceiling, 'eMBB-only ceiling', 'black', 'x'))

    for data, label, color, marker in reference_series:
        if data is None:
            continue
        axes[0].plot(loads, np.asarray(data['embb_rate']) / 1e6, marker=marker, linewidth=1.8, label=label, color=color)
        axes[1].plot(loads, np.asarray(data['urllc_admission']), marker=marker, linewidth=1.8, label=label, color=color)
    _style(axes[0], 'Upper-bound throughput references', 'Average UE load per UAV', 'Mbps')
    _style(axes[1], 'Coexistence vs ceiling admission', 'Average UE load per UAV', 'Ratio')
    _top_axis_lambda(axes[0], loads)
    _top_axis_lambda(axes[1], loads)
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    fig.suptitle(f'Upper Bounds and Coexistence References | Selected baseline: {baseline_label}', fontsize=14)
    path = RESULTS_DIR / '10_upper_bounds_and_frontier.png'
    fig.savefig(path, dpi=210, bbox_inches='tight')
    plt.close(fig)
    return path


def plot_full_mappo_activity(rl_metrics: Dict):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    loads = rl_metrics['loads']

    axes[0, 0].plot(loads, rl_metrics['owner_head_active_ratio'], marker='o', label='Owner head active')
    axes[0, 0].plot(loads, rl_metrics['raw_owner_non_null_ratio'], marker='s', label='Raw owner non-null')
    axes[0, 0].plot(loads, rl_metrics['executed_owner_non_null_ratio'], marker='^', label='Executed owner non-null')
    axes[0, 0].plot(loads, rl_metrics['planning_owner_non_null_ratio'], marker='D', linestyle='--', label='Planning owner committed')
    _style(axes[0, 0], 'Planning owner-head usage', 'Average UE load per UAV', 'Ratio')
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].plot(loads, rl_metrics['embb_power_head_active_ratio'], marker='o', label='eMBB power head active')
    axes[0, 1].plot(loads, rl_metrics['raw_embb_power_nonzero_ratio'], marker='s', label='Raw eMBB power non-zero')
    axes[0, 1].plot(loads, rl_metrics['executed_embb_power_nonzero_ratio'], marker='^', label='Executed eMBB power non-zero')
    axes[0, 1].plot(loads, rl_metrics['phase_a_raw_embb_power_nonzero_ratio'], marker='D', linestyle='--', label='Phase-A raw power non-zero')
    axes[0, 1].plot(loads, rl_metrics['phase_a_executed_embb_power_nonzero_ratio'], marker='X', linestyle='--', label='Phase-A executed power non-zero')
    axes[0, 1].plot(loads, rl_metrics['phase_a_embb_power_write_ratio'], marker='P', linestyle=':', label='Phase-A grid write')
    _style(axes[0, 1], 'eMBB power-head usage', 'Average UE load per UAV', 'Ratio')
    axes[0, 1].legend(fontsize=8)

    axes[1, 0].plot(loads, rl_metrics['raw_executed_any_gap_ratio'], marker='o', label='Any raw/executed gap')
    axes[1, 0].plot(loads, rl_metrics['raw_executed_mode_gap_ratio'], marker='s', linestyle='--', label='Mode gap')
    axes[1, 0].plot(loads, rl_metrics['raw_executed_packet_gap_ratio'], marker='^', linestyle='--', label='Packet gap')
    axes[1, 0].plot(loads, rl_metrics['raw_executed_power_gap_ratio'], marker='D', linestyle='--', label='URLLC power gap')
    _style(axes[1, 0], 'Raw vs executed gap (coexistence)', 'Average UE load per UAV', 'Ratio')
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].plot(loads, rl_metrics['power_floor_hit_ratio'], marker='o', label='Feasible floor hit')
    axes[1, 1].plot(loads, rl_metrics['power_cap_hit_ratio'], marker='s', label='Upper-bound hit')
    axes[1, 1].plot(loads, rl_metrics['power_quantized_ratio'], marker='^', linestyle='--', label='Discrete bin used')
    axes[1, 1].plot(loads, rl_metrics['power_delta_clipped_ratio'], marker='D', linestyle='--', label='Delta clipped')
    axes[1, 1].plot(loads, rl_metrics['mean_raw_power_delta'], marker='x', linestyle=':', label='Mean raw delta')
    axes[1, 1].plot(loads, rl_metrics['mean_executed_power_delta'], marker='P', linestyle=':', label='Mean executed delta')
    _style(axes[1, 1], 'URLLC power projection diagnostics', 'Average UE load per UAV', 'Ratio or delta')
    axes[1, 1].legend(fontsize=8)

    fig.suptitle('Full MAPPO Action-Head Activity and Execution Gap', fontsize=14)
    path = RESULTS_DIR / '11_full_mappo_activity_vs_load.png'
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def build_load_tradeoff_diagnostics(rl_metrics: Dict, baseline_metrics: Dict, cfg: SRMAPPOConfig) -> Dict[str, List[float] | float]:
    loads = [float(load) for load in rl_metrics.get('loads', [])]
    throughput_excess = []
    admission_gap = []
    puncture_loss_gap = []
    overlay_retention_gap = []
    floor_values = []
    floor_pass = []
    floor_violation = []
    loadwise_score = []
    score_contribution = []

    for idx, load in enumerate(loads):
        rl_embb = float(rl_metrics['embb_rate'][idx])
        base_embb = float(baseline_metrics['embb_rate'][idx])
        rl_adm = float(rl_metrics['urllc_admission'][idx])
        base_adm = float(baseline_metrics['urllc_admission'][idx])
        rl_loss = float(rl_metrics['avg_puncture_loss'][idx])
        base_loss = float(baseline_metrics['avg_puncture_loss'][idx])
        rl_retention = float(rl_metrics['avg_overlay_retention'][idx])
        base_retention = float(baseline_metrics['avg_overlay_retention'][idx])
        rl_power = float(rl_metrics['total_power'][idx])
        base_power = float(baseline_metrics['total_power'][idx])

        this_throughput_excess = rl_embb / max(base_embb, 1e-9) - 1.0
        this_admission_gap = rl_adm - base_adm
        this_puncture_loss_gap = rl_loss - base_loss
        this_overlay_retention_gap = rl_retention - base_retention
        this_power_ratio = rl_power / max(base_power, 1e-9)
        this_floor = float(
            selection_floor_for_load(
                load,
                getattr(cfg.training, 'selection_admission_floor_by_load', {}),
                fallback_floor=float(getattr(cfg.training, 'selection_admission_floor', 0.0) or 0.0),
            )
        )
        this_floor_pass = float(rl_adm >= this_floor - 1e-9)
        this_floor_violation = float(max(this_floor - rl_adm, 0.0))
        low_damage_objective = bool(getattr(cfg.training, 'low_damage_admission_objective', False))
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

        throughput_excess.append(this_throughput_excess)
        admission_gap.append(this_admission_gap)
        puncture_loss_gap.append(this_puncture_loss_gap)
        overlay_retention_gap.append(this_overlay_retention_gap)
        floor_values.append(this_floor)
        floor_pass.append(this_floor_pass)
        floor_violation.append(this_floor_violation)
        loadwise_score.append(this_score)
        score_contribution.append(this_contribution)

    return {
        'loads': loads,
        'throughput_excess': throughput_excess,
        'admission_gap': admission_gap,
        'avg_puncture_loss_gap': puncture_loss_gap,
        'overlay_retention_gap': overlay_retention_gap,
        'selection_floor': floor_values,
        'selection_floor_pass': floor_pass,
        'selection_floor_violation': floor_violation,
        'loadwise_selection_score': loadwise_score,
        'score_contribution': score_contribution,
        'weighted_selection_score': float(np.sum(score_contribution)),
    }


def plot_load_tradeoff_diagnostics(tradeoff: Dict):
    fig, axes = plt.subplots(3, 2, figsize=(14, 12), constrained_layout=True)
    loads = tradeoff['loads']

    axes[0, 0].plot(loads, tradeoff['throughput_excess'], marker='o')
    _style(axes[0, 0], 'Per-load throughput excess', 'Average UE load per UAV', 'Rate ratio - 1')

    axes[0, 1].plot(loads, tradeoff['admission_gap'], marker='s', color='tab:green')
    _style(axes[0, 1], 'Per-load admission gap', 'Average UE load per UAV', 'Admission gap')

    axes[1, 0].plot(loads, np.asarray(tradeoff['avg_puncture_loss_gap']) / 1.0e6, marker='^', color='tab:red')
    _style(axes[1, 0], 'Per-load puncture-loss gap', 'Average UE load per UAV', 'Loss gap (Mbps)')

    axes[1, 1].plot(loads, tradeoff['overlay_retention_gap'], marker='D', color='tab:purple')
    _style(axes[1, 1], 'Per-load overlay-retention gap', 'Average UE load per UAV', 'Retention gap')

    axes[2, 0].plot(loads, tradeoff['selection_floor_violation'], marker='o', label='Floor violation')
    _style(axes[2, 0], 'Per-load floor violation', 'Average UE load per UAV', 'Ratio')
    axes[2, 0].legend(fontsize=8)

    axes[2, 1].bar([str(int(load)) for load in loads], tradeoff['score_contribution'], color='tab:orange')
    _style(axes[2, 1], 'Contribution to weighted selection score', 'Average UE load per UAV', 'Weighted contribution')

    fig.suptitle('Load-aware Tradeoff Diagnostics', fontsize=14)
    path = RESULTS_DIR / '12_load_tradeoff_diagnostics.png'
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def build_baseline_reference_diagnostics(
    rl_metrics: Dict,
    matched_metrics: Dict,
    throughput_feasible_metrics: Dict,
    throughput_only_metrics: Dict,
) -> Dict[str, List[float]]:
    loads = [float(load) for load in rl_metrics.get('loads', [])]
    return {
        'loads': loads,
        'throughput_only_admission': [float(value) for value in throughput_only_metrics.get('urllc_admission', [])],
        'throughput_feasible_admission': [float(value) for value in throughput_feasible_metrics.get('urllc_admission', [])],
        'matched_admission': [float(value) for value in matched_metrics.get('urllc_admission', [])],
        'mappo_admission': [float(value) for value in rl_metrics.get('urllc_admission', [])],
        'throughput_feasible_power_mw': [float(value) * 1.0e3 for value in throughput_feasible_metrics.get('total_power', [])],
        'matched_power_mw': [float(value) * 1.0e3 for value in matched_metrics.get('total_power', [])],
        'mappo_power_mw': [float(value) * 1.0e3 for value in rl_metrics.get('total_power', [])],
        'throughput_only_vs_throughput_feasible_admission_gap': [
            float(throughput_only_metrics['urllc_admission'][idx] - throughput_feasible_metrics['urllc_admission'][idx])
            for idx in range(len(loads))
        ],
        'throughput_feasible_vs_matched_power_gap_mw': [
            float((throughput_feasible_metrics['total_power'][idx] - matched_metrics['total_power'][idx]) * 1.0e3)
            for idx in range(len(loads))
        ],
        'mappo_vs_matched_throughput_gap_mbps': [
            float((rl_metrics['embb_rate'][idx] - matched_metrics['embb_rate'][idx]) / 1.0e6)
            for idx in range(len(loads))
        ],
        'mappo_vs_matched_admission_gap': [
            float(rl_metrics['urllc_admission'][idx] - matched_metrics['urllc_admission'][idx])
            for idx in range(len(loads))
        ],
        'mappo_vs_matched_power_gap_mw': [
            float((rl_metrics['total_power'][idx] - matched_metrics['total_power'][idx]) * 1.0e3)
            for idx in range(len(loads))
        ],
    }


def plot_baseline_reference_story(diagnostics: Dict):
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), constrained_layout=True)
    loads = diagnostics['loads']

    axes[0, 0].plot(loads, diagnostics['throughput_only_admission'], marker='D', color='tab:purple', label='Throughput-only Greedy (eMBB-only ceiling)')
    axes[0, 0].plot(loads, diagnostics['throughput_feasible_admission'], marker='*', color='tab:red', label='Throughput-feasible Oracle')
    axes[0, 0].plot(loads, diagnostics['matched_admission'], marker='^', color='tab:green', label='Matched Fixed-Power Throughput Oracle')
    axes[0, 0].plot(loads, diagnostics['mappo_admission'], marker='s', color='tab:orange', label='MAPPO')
    _style(axes[0, 0], 'Admission ratio across baseline roles', 'Average UE load per UAV', 'Admission ratio')
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].plot(loads, diagnostics['throughput_feasible_power_mw'], marker='*', color='tab:red', label='Throughput-feasible Oracle')
    axes[0, 1].plot(loads, diagnostics['matched_power_mw'], marker='^', color='tab:green', label='Matched Fixed-Power Throughput Oracle')
    axes[0, 1].plot(loads, diagnostics['mappo_power_mw'], marker='s', color='tab:orange', label='MAPPO')
    _style_power_axis(axes[0, 1], 'Power: optimistic oracle vs matched coexistence reference', 'Average UE load per UAV', 'mW')
    axes[0, 1].legend(fontsize=8)

    axes[1, 0].plot(loads, diagnostics['mappo_vs_matched_throughput_gap_mbps'], marker='s', color='tab:orange')
    _style(axes[1, 0], 'MAPPO throughput gap vs matched oracle', 'Average UE load per UAV', 'Gap (Mbps)')

    axes[1, 1].plot(loads, diagnostics['mappo_vs_matched_admission_gap'], marker='o', color='tab:blue', label='Admission gap')
    axes[1, 1].plot(loads, diagnostics['mappo_vs_matched_power_gap_mw'], marker='x', color='tab:red', linestyle='--', label='Power gap (mW)')
    _style(axes[1, 1], 'MAPPO coexistence gaps vs matched oracle', 'Average UE load per UAV', 'Gap')
    axes[1, 1].legend(fontsize=8)

    fig.suptitle('Baseline Role Cleanup: Ceiling vs Constrained vs Matched Coexistence Reference', fontsize=14)
    path = RESULTS_DIR / '19_baseline_reference_story.png'
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_fairness_uav(
    greedy_original: Dict,
    greedy_original_lite: Optional[Dict],
    greedy_original_normal_v2: Optional[Dict],
    greedy_myopic: Optional[Dict],
    greedy_matched: Dict,
    greedy_throughput_feasible: Optional[Dict],
    greedy_throughput_only: Optional[Dict],
    greedy_channel_only: Optional[Dict],
    rl: Dict,
    greedy_baseline_mode: str = "original",
):
    fig, axes = plt.subplots(3, 2, figsize=(14, 14), constrained_layout=True)
    loads = rl['loads']
    baseline_data, baseline_label, baseline_color, baseline_marker = _selected_baseline_source(
        greedy_original,
        greedy_original_lite,
        greedy_original_normal_v2,
        greedy_myopic,
        greedy_matched,
        greedy_throughput_feasible,
        greedy_throughput_only,
        greedy_channel_only,
        greedy_baseline_mode,
    )
    for data, label, color, marker in _comparison_series(greedy_baseline_mode):
        source = _comparison_source(
            data,
            greedy_original,
            greedy_original_lite,
            greedy_original_normal_v2,
            greedy_myopic,
            greedy_matched,
            greedy_throughput_feasible,
            greedy_throughput_only,
            greedy_channel_only,
            rl,
        )
        if source is None:
            continue
        axes[0, 0].plot(loads, source['jain_fairness'], marker=marker, color=color, label=label)
    _style(axes[0, 0], "Jain's fairness index", 'Average UE load per UAV', 'Fairness')
    axes[0, 0].legend()

    for data, label, color, marker in _comparison_series(greedy_baseline_mode):
        source = _comparison_source(
            data,
            greedy_original,
            greedy_original_lite,
            greedy_original_normal_v2,
            greedy_myopic,
            greedy_matched,
            greedy_throughput_feasible,
            greedy_throughput_only,
            greedy_channel_only,
            rl,
        )
        if source is None:
            continue
        axes[0, 1].plot(loads, source['cell_edge_min_rate_satisfaction_ratio'], marker=marker, color=color, label=label)
    _style(axes[0, 1], 'Cell-edge eMBB min-rate satisfaction', 'Average UE load per UAV', 'Ratio')
    axes[0, 1].legend()

    for source, prefix, color, marker in (
        (baseline_data, baseline_label, baseline_color, baseline_marker),
        (rl, 'MAPPO', 'tab:orange', 's'),
    ):
        if source is None:
            continue
        axes[1, 0].plot(loads, source['per_uav_total_load_std'], marker=marker, color=color, label=f'{prefix} assoc. std')
        axes[1, 0].plot(loads, source['per_uav_urllc_sched_std'], marker=marker, color=color, linestyle='--', label=f'{prefix} URLLC sched std')
    _style(axes[1, 0], 'Per-UAV load imbalance', 'Average UE load per UAV', 'Std. dev.')
    axes[1, 0].legend(fontsize=8)

    for data, label, color, marker in _comparison_series(greedy_baseline_mode):
        source = _comparison_source(
            data,
            greedy_original,
            greedy_original_lite,
            greedy_original_normal_v2,
            greedy_myopic,
            greedy_matched,
            greedy_throughput_feasible,
            greedy_throughput_only,
            greedy_channel_only,
            rl,
        )
        if source is None:
            continue
        axes[1, 1].plot(loads, source['per_uav_throughput_std'], marker=marker, color=color, label=label)
    _style(axes[1, 1], 'Per-UAV throughput imbalance', 'Average UE load per UAV', 'Std. dev. (bps)')
    axes[1, 1].legend()

    greedy_channel_sched = np.stack(baseline_data['per_uav_scheduled_urllc'], axis=0) if baseline_data is not None else None
    rl_sched = np.stack(rl['per_uav_scheduled_urllc'], axis=0)
    for uav_idx in range(rl_sched.shape[1]):
        if greedy_channel_sched is not None:
            axes[2, 0].plot(loads, greedy_channel_sched[:, uav_idx], marker=baseline_marker, color=baseline_color, alpha=0.35)
        axes[2, 0].plot(loads, rl_sched[:, uav_idx], marker='s', color='tab:orange', alpha=0.35)
    _style(axes[2, 0], 'Per-UAV scheduled URLLC packets', 'Average UE load per UAV', 'Packets/slot')
    axes[2, 0].legend(
        handles=[
            Line2D([0], [0], color=baseline_color, marker=baseline_marker, label=baseline_label),
            Line2D([0], [0], color='tab:orange', marker='s', label='MAPPO'),
        ],
        fontsize=8,
    )

    for source, prefix, color, marker in (
        (baseline_data, baseline_label, baseline_color, baseline_marker),
        (rl, 'MAPPO', 'tab:orange', 's'),
    ):
        if source is None:
            continue
        axes[2, 1].plot(loads, np.mean(np.stack(source['per_uav_associated_embb'], axis=0), axis=1), marker=marker, color=color, label=f'{prefix} assoc. eMBB')
        axes[2, 1].plot(loads, np.mean(np.stack(source['per_uav_associated_urllc'], axis=0), axis=1), marker=marker, color=color, linestyle='--', label=f'{prefix} assoc. URLLC')
    _style(axes[2, 1], 'Average per-UAV association load', 'Average UE load per UAV', 'Users/UAV')
    axes[2, 1].legend(fontsize=8)
    fig.suptitle(f'Fairness and Per-UAV Diagnostics | {baseline_label} vs MAPPO | Selected baseline: {baseline_label}', fontsize=14)

    path = RESULTS_DIR / '03_fairness_and_uav_vs_load.png'
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_training_diagnostics(history: List[Dict], rl_metrics: Dict):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    episodes = [record['iteration'] for record in history]
    rollout_rewards = [record.get('update', {}).get('mean_reward', np.nan) for record in history]
    axes[0, 0].plot(episodes, rollout_rewards, marker='o')
    _style(axes[0, 0], 'Learning curve: rollout reward', 'Training episode', 'Mean team reward')

    eval_iters, eval_embb, eval_adm = [], [], []
    for record in history:
        evaluation = record.get('evaluation')
        if evaluation:
            eval_iters.append(record['iteration'])
            eval_embb.append(float(evaluation.get('policy_mean_embb_rate', 0.0)) / 1e6)
            eval_adm.append(float(evaluation.get('policy_mean_scheduled_ratio', 0.0)))
    if eval_iters:
        axes[0, 1].plot(eval_iters, eval_embb, marker='s', label='eMBB throughput')
        ax2 = axes[0, 1].twinx()
        ax2.plot(eval_iters, eval_adm, marker='^', color='tab:orange', label='URLLC admission')
        axes[0, 1].set_title('Performance vs training steps')
        axes[0, 1].set_xlabel('Training episode')
        axes[0, 1].set_ylabel('eMBB throughput (Mbps)')
        ax2.set_ylabel('Admission ratio')
        axes[0, 1].grid(True, alpha=0.25)
    else:
        axes[0, 1].text(0.5, 0.5, 'No evaluation history available', ha='center', va='center')

    if eval_iters:
        overlay_counts = [float(record.get('evaluation', {}).get('policy_mean_overlay', 0.0)) for record in history if record.get('evaluation')]
        punct_counts = [float(record.get('evaluation', {}).get('policy_mean_puncture', 0.0)) for record in history if record.get('evaluation')]
        total = np.maximum(np.asarray(overlay_counts) + np.asarray(punct_counts), 1e-9)
        axes[1, 0].plot(eval_iters, np.asarray(overlay_counts) / total, marker='s', label='Overlay ratio')
        axes[1, 0].plot(eval_iters, np.asarray(punct_counts) / total, marker='^', label='Puncture ratio')
        _style(axes[1, 0], 'Action distribution evolution', 'Training episode', 'Ratio')
        axes[1, 0].legend()
    else:
        axes[1, 0].text(0.5, 0.5, 'No action history available', ha='center', va='center')

    loads = rl_metrics['loads']
    axes[1, 1].plot(loads, rl_metrics['shield_correction_ratio'], marker='s', label='Shield correction')
    axes[1, 1].plot(loads, rl_metrics['joint_reliability_rewrite_ratio'], marker='D', linestyle='--', label='Joint rewrite')
    axes[1, 1].plot(loads, rl_metrics['mode_correction_ratio'], marker='v', linestyle='--', label='Mode corrected')
    axes[1, 1].plot(loads, rl_metrics['collision_rewrite_ratio'], marker='^', label='Collision rewrite')
    axes[1, 1].plot(loads, rl_metrics['fallback_ratio'], marker='o', label='Fallback ratio')
    _style(axes[1, 1], 'Shield / fallback activation', 'Average UE load per UAV', 'Ratio')
    axes[1, 1].legend(fontsize=8, ncol=2)

    path = RESULTS_DIR / '04_training_diagnostics.png'
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_training_reward_curve(history: List[Dict]):
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    episodes = [record.get('iteration', idx + 1) for idx, record in enumerate(history)]
    rewards = [record.get('update', {}).get('mean_reward', np.nan) for record in history]
    if episodes:
        y = np.asarray(rewards, dtype=float)
        ax.plot(episodes, y, marker='o', linewidth=1.0, alpha=0.65, label='rollout reward')
        finite = y[np.isfinite(y)]
        if finite.size >= 10:
            win = min(51, max(11, (len(y) // 20) * 2 + 1))
            fill_value = float(np.mean(finite))
            y_clean = np.where(np.isfinite(y), y, fill_value)
            kernel = np.ones(win, dtype=float) / float(win)
            y_pad = np.pad(y_clean, (win // 2, win // 2), mode='edge')
            smooth = np.convolve(y_pad, kernel, mode='valid')[:len(y)]
            ax.plot(episodes, smooth, linewidth=2.0, label=f'moving average ({win})')
        ax.legend(fontsize=9)
    else:
        ax.text(0.5, 0.5, 'No training history available', ha='center', va='center', transform=ax.transAxes)
    _style(ax, 'Training Reward Curve', 'Training episode', 'Mean team reward')
    path = RESULTS_DIR / 'training_reward_curve.png'
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_mode_anchor_debug(history: List[Dict]):
    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    eval_iters: List[int] = []
    overlay_ratio: List[float] = []
    puncture_ratio: List[float] = []
    overlay_when_safe: List[float] = []
    teacher_agreement: List[float] = []
    for record in history:
        evaluation = record.get('evaluation') or {}
        if not evaluation:
            continue
        eval_iters.append(int(record.get('iteration', len(eval_iters) + 1)))
        overlay_ratio.append(float(evaluation.get('policy_mean_overlay_ratio', evaluation.get('policy_mean_overlay', 0.0))))
        puncture_ratio.append(float(evaluation.get('policy_mean_puncture_ratio', evaluation.get('policy_mean_puncture', 0.0))))
        overlay_when_safe.append(
            float(evaluation.get('policy_mean_overlay_chosen_when_safe_puncture_available_ratio', np.nan))
        )
        teacher_agreement.append(
            float(evaluation.get('policy_mean_teacher_mode_agreement_ratio', np.nan))
        )
    if eval_iters:
        ax.plot(eval_iters, overlay_ratio, marker='o', linestyle='-', label='Overlay ratio')
        ax.plot(eval_iters, puncture_ratio, marker='s', linestyle='--', label='Puncture ratio')
        ax.plot(
            eval_iters,
            overlay_when_safe,
            marker='D',
            linestyle='-',
            label='Overlay chosen when safe puncture available',
        )
        ax.plot(
            eval_iters,
            teacher_agreement,
            marker='^',
            linestyle='--',
            label='Teacher mode agreement ratio',
        )
    else:
        ax.text(0.5, 0.5, 'No evaluation history available', ha='center', va='center', transform=ax.transAxes)
    _style(ax, 'Hard mode-anchor debug', 'Training iteration', 'Ratio / count proxy')
    ax.legend(fontsize=9)
    path = RESULTS_DIR / '19_mode_anchor_debug.png'
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_slot_dynamics(rep_greedy: Dict, rep_rl: Dict):
    baseline_label = rep_rl.get('comparison_baseline_label', 'Baseline')
    fig, axes = plt.subplots(3, 2, figsize=(14, 14), constrained_layout=True)
    trace = rep_rl['trace']
    x = [item['cell_index'] for item in trace]
    axes[0, 0].plot(x, np.asarray([item['embb_rate'] for item in trace]) / 1e6, marker='o')
    _style(axes[0, 0], 'MAPPO slot timeline: eMBB throughput', 'Cell index (RB×minislot)', 'Mbps')

    axes[0, 1].plot(x, [item['arrivals'] for item in trace], marker='o', label='Arrivals')
    axes[0, 1].plot(x, [item['scheduled_packets'] for item in trace], marker='s', label='Admitted')
    _style(axes[0, 1], 'MAPPO slot timeline: URLLC arrivals / admitted', 'Cell index (RB×minislot)', 'Packets')
    axes[0, 1].legend()

    axes[1, 0].plot(x, [item['overlay_count'] for item in trace], marker='o', label='Overlay count')
    axes[1, 0].plot(x, [item['puncture_count'] for item in trace], marker='s', label='Puncture count')
    _style(axes[1, 0], 'MAPPO slot timeline: mode counts', 'Cell index (RB×minislot)', 'Count')
    axes[1, 0].legend()

    axes[1, 1].plot(x, np.asarray([item['total_power'] for item in trace]) * 1e3, marker='o', label='Total power')
    ax2 = axes[1, 1].twinx()
    ax2.plot(x, [item['utility_gap'] for item in trace], color='tab:red', marker='s', label='Utility gap')
    axes[1, 1].set_title('MAPPO slot timeline: power and one-step utility gap')
    axes[1, 1].set_xlabel('Cell index (RB×minislot)')
    axes[1, 1].set_ylabel('Power (mW)')
    ax2.set_ylabel('Utility gap vs greedy')
    axes[1, 1].grid(True, alpha=0.25)
    _set_plain_y_ticks(axes[1, 1])

    minislot_activity = np.sum(rep_rl['packet_grid'] >= 0, axis=1)
    im = axes[2, 0].imshow(minislot_activity, aspect='auto', cmap='YlOrBr')
    axes[2, 0].set_title('URLLC scheduled packet activity by minislot')
    axes[2, 0].set_xlabel('Minislot')
    axes[2, 0].set_ylabel('UAV')
    fig.colorbar(im, ax=axes[2, 0], fraction=0.046)

    throughput_ratio = rep_rl['embb_rate'] / max(rep_greedy['embb_rate'], 1e-9)
    axes[2, 1].bar([baseline_label, 'MAPPO'], [rep_greedy['embb_rate'] / 1e6, rep_rl['embb_rate'] / 1e6], color=['tab:blue', 'tab:orange'])
    axes[2, 1].set_title(f'Representative slot throughput comparison (ratio={throughput_ratio:.2f})')
    axes[2, 1].set_ylabel('eMBB throughput (Mbps)')
    axes[2, 1].grid(True, axis='y', alpha=0.25)

    path = RESULTS_DIR / '05_slot_timeline_and_activity.png'
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_single_slot_mode_maps(rep_rl: Dict):
    owner = np.asarray(rep_rl['owner_per_uav_rb'], dtype=int)
    mode_grid = np.asarray(rep_rl['mode_grid'], dtype=int)
    mode_map = np.zeros_like(mode_grid, dtype=int)
    for uav_idx in range(owner.shape[0]):
        for rb_idx in range(owner.shape[1]):
            for minislot in range(mode_grid.shape[2]):
                if mode_grid[uav_idx, rb_idx, minislot] == MODE_OVERLAY:
                    mode_map[uav_idx, rb_idx, minislot] = 2
                elif mode_grid[uav_idx, rb_idx, minislot] == MODE_PUNCTURE:
                    mode_map[uav_idx, rb_idx, minislot] = 3
                elif owner[uav_idx, rb_idx] >= 0:
                    mode_map[uav_idx, rb_idx, minislot] = 1
                else:
                    mode_map[uav_idx, rb_idx, minislot] = 0
    fig, axes = plt.subplots(1, mode_map.shape[0], figsize=(16, 4), constrained_layout=True)
    if mode_map.shape[0] == 1:
        axes = [axes]
    cmap = matplotlib.colors.ListedColormap(['#f5f5f5', '#9ec5fe', '#ffb86b', '#ef6f6c'])
    labels = ['Idle', 'eMBB only', 'Overlay', 'Puncture']
    for uav_idx, ax in enumerate(axes):
        im = ax.imshow(mode_map[uav_idx], aspect='auto', cmap=cmap, vmin=0, vmax=3)
        ax.set_title(f'UAV {uav_idx + 1}')
        ax.set_xlabel('Minislot')
        ax.set_ylabel('RB')
    cbar = fig.colorbar(im, ax=axes, fraction=0.03, ticks=[0, 1, 2, 3])
    cbar.ax.set_yticklabels(labels)
    path = RESULTS_DIR / '06_single_slot_mode_maps.png'
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_timeslot_kpi_comparison(greedy_series: List[Dict], rl_series: List[Dict], meta: Dict):
    num_slots = int(meta['num_slots'])
    x = np.arange(num_slots)
    baseline_label = str(meta.get('greedy_baseline_label', 'Greedy'))
    fig, axes = plt.subplots(2, 2, figsize=(16, 9), constrained_layout=False)
    panels = [
        ('Aggregate eMBB Throughput', 'embb_rate', 1e6, 'Mbps'),
        ('Per-User eMBB Rate', 'embb_user_rate', 1e6, 'Mbps'),
        ('eMBB Positive-Rate Ratio', 'embb_positive_rate_ratio', 1.0, 'Ratio'),
        ('URLLC Admission Ratio', 'urllc_admission', 1.0, 'Ratio'),
    ]
    for ax, (title, key, scale, ylabel) in zip(axes.flat, panels):
        ax.plot(x, np.asarray([item[key] for item in greedy_series]) / scale, marker='o', markersize=3, linewidth=1.5, label=baseline_label)
        ax.plot(x, np.asarray([item[key] for item in rl_series]) / scale, marker='s', markersize=3, linewidth=1.5, label='MAPPO')
        _style_timeslot_axis(ax, title, ylabel, num_slots)
        ax.legend(fontsize=8)

    fig.suptitle(f'MAPPO and {baseline_label} Dynamics | load={meta["load"]:.0f} UE/UAV', fontsize=14)
    side_text = _scenario_info_box_lines(
        meta['sys_cfg'],
        meta['sim_cfg'],
        float(meta['load']),
        meta['checkpoint'],
        meta['cfg'],
        num_slots,
        meta.get('greedy_baseline_mode'),
    )
    _add_side_info_box(fig, side_text)
    path = RESULTS_DIR / '07_timeslot_kpis_comparison.png'
    fig.savefig(path, dpi=210, bbox_inches='tight')
    plt.close(fig)
    return path


def plot_timeslot_power_comparison(greedy_series: List[Dict], rl_series: List[Dict], meta: Dict):
    num_slots = int(meta['num_slots'])
    x = np.arange(num_slots)
    baseline_label = str(meta.get('greedy_baseline_label', 'Greedy'))
    fig, axes = plt.subplots(2, 2, figsize=(16, 9), constrained_layout=False)

    axes[0, 0].plot(x, np.asarray([item['total_power'] for item in greedy_series]) * 1e3, marker='o', markersize=3, linewidth=1.5, label=baseline_label)
    axes[0, 0].plot(x, np.asarray([item['total_power'] for item in rl_series]) * 1e3, marker='s', markersize=3, linewidth=1.5, label='MAPPO')
    _style_timeslot_axis(axes[0, 0], 'Total Transmit Power', 'mW', num_slots)
    _set_plain_y_ticks(axes[0, 0])
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].plot(x, np.asarray([item['embb_power'] for item in greedy_series]) * 1e3, marker='o', markersize=3, linewidth=1.5, label=f'{baseline_label} eMBB')
    axes[0, 1].plot(x, np.asarray([item['embb_power'] for item in rl_series]) * 1e3, marker='s', markersize=3, linewidth=1.5, label='MAPPO eMBB')
    axes[0, 1].plot(x, np.asarray([item['urllc_power'] for item in greedy_series]) * 1e3, marker='o', markersize=3, linewidth=1.0, linestyle='--', label=f'{baseline_label} URLLC')
    axes[0, 1].plot(x, np.asarray([item['urllc_power'] for item in rl_series]) * 1e3, marker='s', markersize=3, linewidth=1.0, linestyle='--', label='MAPPO URLLC')
    _style_timeslot_axis(axes[0, 1], 'Traffic-Type Power Split', 'mW', num_slots)
    _set_plain_y_ticks(axes[0, 1])
    axes[0, 1].legend(fontsize=8, ncol=2)

    axes[1, 0].plot(x, [item['overlay_ratio'] for item in greedy_series], marker='o', markersize=3, linewidth=1.5, label=f'{baseline_label} overlay ratio')
    axes[1, 0].plot(x, [item['overlay_ratio'] for item in rl_series], marker='s', markersize=3, linewidth=1.5, label='MAPPO overlay ratio')
    axes[1, 0].plot(x, [item['puncture_ratio'] for item in greedy_series], marker='o', markersize=3, linewidth=1.0, linestyle='--', label=f'{baseline_label} puncture ratio')
    axes[1, 0].plot(x, [item['puncture_ratio'] for item in rl_series], marker='s', markersize=3, linewidth=1.0, linestyle='--', label='MAPPO puncture ratio')
    _style_timeslot_axis(axes[1, 0], 'Mode Ratio', 'Ratio', num_slots)
    axes[1, 0].legend(fontsize=8, ncol=2)

    axes[1, 1].plot(x, np.asarray([item['avg_puncture_loss'] for item in greedy_series]) / 1e6, marker='o', markersize=3, linewidth=1.5, label=f'{baseline_label} puncture loss')
    axes[1, 1].plot(x, np.asarray([item['avg_puncture_loss'] for item in rl_series]) / 1e6, marker='s', markersize=3, linewidth=1.5, label='MAPPO puncture loss')
    axes[1, 1].plot(x, [item['avg_overlay_retention'] for item in greedy_series], marker='o', markersize=3, linewidth=1.0, linestyle='--', label=f'{baseline_label} overlay retention')
    axes[1, 1].plot(x, [item['avg_overlay_retention'] for item in rl_series], marker='s', markersize=3, linewidth=1.0, linestyle='--', label='MAPPO overlay retention')
    _style_timeslot_axis(axes[1, 1], 'Puncture Loss / Overlay Retention', 'Loss (Mbps) or retention', num_slots)
    axes[1, 1].legend(fontsize=8, ncol=2)

    fig.suptitle(f'Power and Mode Timeline | load={meta["load"]:.0f} UE/UAV', fontsize=14)
    side_text = _scenario_info_box_lines(
        meta['sys_cfg'],
        meta['sim_cfg'],
        float(meta['load']),
        meta['checkpoint'],
        meta['cfg'],
        num_slots,
        meta.get('greedy_baseline_mode'),
    )
    _add_side_info_box(fig, side_text)
    path = RESULTS_DIR / '08_timeslot_power_mode_comparison.png'
    fig.savefig(path, dpi=210, bbox_inches='tight')
    plt.close(fig)
    return path


def plot_timeslot_action_summary(greedy_series: List[Dict], rl_series: List[Dict], meta: Dict):
    num_slots = int(meta['num_slots'])
    x = np.arange(num_slots)
    baseline_label = str(meta.get('greedy_baseline_label', 'Greedy'))
    fig, axes = plt.subplots(3, 1, figsize=(16, 10), constrained_layout=False, sharex=True)

    width = 0.38
    axes[0].bar(x - width / 2, [item['active_packets'] for item in greedy_series], width=width, alpha=0.60, label=f'{baseline_label} arrivals', color='tab:blue')
    axes[0].bar(x + width / 2, [item['active_packets'] for item in rl_series], width=width, alpha=0.60, label='MAPPO arrivals', color='tab:orange')
    axes[0].plot(x, [item['scheduled_packets'] for item in greedy_series], color='tab:blue', marker='o', linewidth=1.5, markersize=3, label=f'{baseline_label} admitted')
    axes[0].plot(x, [item['scheduled_packets'] for item in rl_series], color='tab:orange', marker='s', linewidth=1.5, markersize=3, label='MAPPO admitted')
    _style_timeslot_axis(axes[0], 'URLLC Arrivals and Admitted Packets', 'Packets', num_slots)
    axes[0].legend(fontsize=8, ncol=2)

    axes[1].plot(x, [item['overlay_count'] for item in greedy_series], marker='o', linewidth=1.5, markersize=3, label=f'{baseline_label} overlay count')
    axes[1].plot(x, [item['overlay_count'] for item in rl_series], marker='s', linewidth=1.5, markersize=3, label='MAPPO overlay count')
    axes[1].plot(x, [item['puncture_count'] for item in greedy_series], marker='o', linewidth=1.0, markersize=3, linestyle='--', label=f'{baseline_label} puncture count')
    axes[1].plot(x, [item['puncture_count'] for item in rl_series], marker='s', linewidth=1.0, markersize=3, linestyle='--', label='MAPPO puncture count')
    _style_timeslot_axis(axes[1], 'Mode Action Counts', 'Count', num_slots)
    axes[1].legend(fontsize=8, ncol=2)

    axes[2].plot(x, np.asarray([item['embb_rate'] for item in greedy_series]) / 1e6, marker='o', linewidth=1.5, markersize=3, label=f'{baseline_label} eMBB throughput')
    axes[2].plot(x, np.asarray([item['embb_rate'] for item in rl_series]) / 1e6, marker='s', linewidth=1.5, markersize=3, label='MAPPO eMBB throughput')
    _style_timeslot_axis(axes[2], 'Aggregate eMBB Throughput', 'Mbps', num_slots)
    axes[2].legend(fontsize=8)

    fig.suptitle(f'URLLC, Mode, and Throughput Timeline | load={meta["load"]:.0f} UE/UAV', fontsize=14)
    side_text = _scenario_info_box_lines(
        meta['sys_cfg'],
        meta['sim_cfg'],
        float(meta['load']),
        meta['checkpoint'],
        meta['cfg'],
        num_slots,
        meta.get('greedy_baseline_mode'),
    )
    _add_side_info_box(fig, side_text)
    path = RESULTS_DIR / '09_timeslot_action_summary.png'
    fig.savefig(path, dpi=210, bbox_inches='tight')
    plt.close(fig)
    return path


def save_metrics_json(payload: Dict):
    path = RESULTS_DIR / 'sr_mappo_report_metrics.json'
    def _convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_convert(v) for v in obj]
        return obj
    path.write_text(json.dumps(_convert(payload), indent=2), encoding='utf-8')
    return path


def append_experiment_history_entry(entry: Dict[str, object]) -> Path:
    path = RESULTS_DIR / "sr_mappo_experiment_history.json"
    history: List[Dict[str, object]] = []
    if path.exists():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, list):
                history = loaded
        except Exception:
            history = []
    history.append(entry)
    path.write_text(json.dumps(history, indent=2, default=_json_default), encoding="utf-8")
    return path


def _json_default(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def generate_report(
    loads: Optional[List[float]] = None,
    episodes_per_load: int = DEFAULT_EPISODES_PER_LOAD,
    fast: bool = False,
    experiment_line: str | None = None,
    checkpoint_path: str | None = None,
    checkpoint_kind: str | None = None,
    greedy_only: bool = False,
    output_dir: str | None = None,
):
    global _REPORT_TIMING_ENABLED, _REPORT_RUN_SEED_BASE, _REPORT_EPISODE_CACHE_ENABLED
    report_start = perf_counter()
    _report_log("Starting report generation.")
    if output_dir:
        globals()['RESULTS_DIR'] = _resolve_writable_results_dir(Path(output_dir))
    _report_log(f"Results output dir: {RESULTS_DIR}")
    try:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        # One more chance to fall back at runtime (e.g., if ACL changes after import).
        fallback_dir = _resolve_writable_results_dir(PROJECT_ROOT / 'results')
        _report_log(f"WARNING: results dir not writable ({exc}). Falling back to: {fallback_dir}")
        globals()['RESULTS_DIR'] = fallback_dir
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stale_removed = _cleanup_stale_report_artifacts()
    if stale_removed:
        _report_log(f"Removed stale top-level report artifacts: {', '.join(sorted(stale_removed))}")
    report_cfg = apply_experiment_preset(SRMAPPOConfig(), experiment_line)
    # Optional runtime overrides for batch experiment sweeps.
    forced_ratio_env = os.environ.get("SR_MAPPO_REPORT_URLLC_RATIO_OVERRIDE", "").strip()
    if forced_ratio_env:
        try:
            forced_ratio_val = float(forced_ratio_env)
            report_cfg.env.urllc_user_ratio_override = float(np.clip(forced_ratio_val, 0.0, 1.0))
            _report_log(
                f"[OVERRIDE] urllc_user_ratio_override={report_cfg.env.urllc_user_ratio_override:.3f} "
                f"(from SR_MAPPO_REPORT_URLLC_RATIO_OVERRIDE={forced_ratio_env})"
            )
        except ValueError:
            _report_log(f"[OVERRIDE] ignore invalid SR_MAPPO_REPORT_URLLC_RATIO_OVERRIDE={forced_ratio_env!r}")
    forced_poisson_env = os.environ.get("SR_MAPPO_REPORT_URLLC_POISSON_RATE_OVERRIDE", "").strip()
    if forced_poisson_env:
        try:
            report_cfg.env.urllc_poisson_rate = float(forced_poisson_env)
            _report_log(
                f"[OVERRIDE] urllc_poisson_rate={float(report_cfg.env.urllc_poisson_rate):.6f} "
                f"(from SR_MAPPO_REPORT_URLLC_POISSON_RATE_OVERRIDE={forced_poisson_env})"
            )
        except ValueError:
            _report_log(f"[OVERRIDE] ignore invalid SR_MAPPO_REPORT_URLLC_POISSON_RATE_OVERRIDE={forced_poisson_env!r}")
    forced_poisson_fixed_env = os.environ.get("SR_MAPPO_REPORT_FIXED_URLLC_POISSON_RATE", "").strip()
    if forced_poisson_fixed_env:
        v = forced_poisson_fixed_env.lower()
        report_cfg.env.fixed_urllc_poisson_rate = bool(v in {"1", "true", "yes", "on"})
        _report_log(
            f"[OVERRIDE] fixed_urllc_poisson_rate={int(bool(report_cfg.env.fixed_urllc_poisson_rate))} "
            f"(from SR_MAPPO_REPORT_FIXED_URLLC_POISSON_RATE={forced_poisson_fixed_env})"
        )
    forced_per_user_env = os.environ.get("SR_MAPPO_REPORT_URLLC_POISSON_PER_USER", "").strip()
    if forced_per_user_env:
        v = forced_per_user_env.lower()
        report_cfg.env.urllc_poisson_rate_is_per_user = bool(v in {"1", "true", "yes", "on"})
        _report_log(
            f"[OVERRIDE] urllc_poisson_rate_is_per_user={int(bool(report_cfg.env.urllc_poisson_rate_is_per_user))} "
            f"(from SR_MAPPO_REPORT_URLLC_POISSON_PER_USER={forced_per_user_env})"
        )
    forced_cross_mix_cap_map_env = os.environ.get("SR_MAPPO_REPORT_PHASE0_CROSS_MIX_RATE_CAP_MAP_BPS", "").strip()
    if forced_cross_mix_cap_map_env:
        try:
            raw_map = json.loads(forced_cross_mix_cap_map_env)
            cap_map_bps: Dict[float, float] = {}
            if isinstance(raw_map, dict):
                for k, v in raw_map.items():
                    cap_map_bps[float(k)] = float(v)
            if cap_map_bps:
                report_cfg.env.phase0_cross_mix_rate_cap_map_bps = dict(cap_map_bps)
                _report_log(
                    "[OVERRIDE] phase0_cross_mix_rate_cap_map_bps="
                    + json.dumps({f"{float(k):.12g}": float(v) for k, v in cap_map_bps.items()})
                    + " (from SR_MAPPO_REPORT_PHASE0_CROSS_MIX_RATE_CAP_MAP_BPS)"
                )
        except Exception:
            _report_log(
                "[OVERRIDE] ignore invalid SR_MAPPO_REPORT_PHASE0_CROSS_MIX_RATE_CAP_MAP_BPS="
                + repr(forced_cross_mix_cap_map_env)
            )
    forced_slot_level_env = os.environ.get("SR_MAPPO_REPORT_URLLC_POISSON_SLOT_LEVEL", "").strip()
    if forced_slot_level_env:
        v = forced_slot_level_env.lower()
        report_cfg.env.urllc_poisson_rate_is_slot_level = bool(v in {"1", "true", "yes", "on"})
        _report_log(
            f"[OVERRIDE] urllc_poisson_rate_is_slot_level={int(bool(report_cfg.env.urllc_poisson_rate_is_slot_level))} "
            f"(from SR_MAPPO_REPORT_URLLC_POISSON_SLOT_LEVEL={forced_slot_level_env})"
        )
    forced_fixed_embb_baseline_policy_env = os.environ.get("SR_MAPPO_REPORT_FIXED_EMBB_BASELINE_POLICY", "").strip()
    if forced_fixed_embb_baseline_policy_env:
        report_cfg.env.fixed_embb_baseline_policy = str(forced_fixed_embb_baseline_policy_env).strip()
        _report_log(
            f"[OVERRIDE] fixed_embb_baseline_policy={report_cfg.env.fixed_embb_baseline_policy} "
            f"(from SR_MAPPO_REPORT_FIXED_EMBB_BASELINE_POLICY={forced_fixed_embb_baseline_policy_env})"
        )
    forced_disallow_keep_pending_env = os.environ.get("SR_MAPPO_REPORT_DISALLOW_KEEP_WHEN_URLLC_PENDING", "").strip()
    if forced_disallow_keep_pending_env:
        report_cfg.env.disallow_keep_when_urllc_pending = bool(
            forced_disallow_keep_pending_env.lower() in {"1", "true", "yes", "on"}
        )
        _report_log(
            f"[OVERRIDE] disallow_keep_when_urllc_pending={int(bool(report_cfg.env.disallow_keep_when_urllc_pending))} "
            f"(from SR_MAPPO_REPORT_DISALLOW_KEEP_WHEN_URLLC_PENDING={forced_disallow_keep_pending_env})"
        )
    frozen_greedy_json_env = os.environ.get("SR_MAPPO_REPORT_FROZEN_GREEDY_JSON", "").strip()
    if frozen_greedy_json_env:
        report_cfg.training.frozen_greedy_metrics_path = str(frozen_greedy_json_env)
        _report_log(
            "[OVERRIDE] frozen_greedy_metrics_path="
            f"{report_cfg.training.frozen_greedy_metrics_path} "
            f"(from SR_MAPPO_REPORT_FROZEN_GREEDY_JSON={frozen_greedy_json_env})"
        )
    report_cfg = _maybe_realign_greedy_mix_preset(report_cfg)

    # Share semantics default to disabled for backward compatibility.
    # Explicit env override can enable share-mode for controlled A/B comparisons.
    share_enable_env = os.environ.get("SR_MAPPO_REPORT_ENABLE_GREEDY_SHARE", "").strip().lower()
    share_enabled = share_enable_env in {"1", "true", "yes", "on"}
    report_cfg.env.greedy_urllc_share_mode = "none"
    report_cfg.env.greedy_urllc_share_ratio = 0.0
    report_cfg.env.greedy_share_reference_pre_mbps_by_load = {}
    if share_enabled:
        forced_share_mode = os.environ.get("SR_MAPPO_REPORT_GREEDY_SHARE_MODE_OVERRIDE", "").strip().lower()
        forced_share_ratio = os.environ.get("SR_MAPPO_REPORT_GREEDY_SHARE_RATIO_OVERRIDE", "").strip()
        forced_share_ref_env = os.environ.get("SR_MAPPO_REPORT_GREEDY_SHARE_REFERENCE_PRE_MBPS_BY_LOAD", "").strip()
        share_mode = forced_share_mode if forced_share_mode else "fixed_share"
        if share_mode not in {"none", "fixed_share"}:
            _report_log(f"[OVERRIDE] invalid share mode {share_mode!r}; fallback to 'fixed_share'")
            share_mode = "fixed_share"
        report_cfg.env.greedy_urllc_share_mode = share_mode
        if forced_share_ratio:
            try:
                report_cfg.env.greedy_urllc_share_ratio = float(max(0.0, float(forced_share_ratio)))
            except ValueError:
                _report_log(f"[OVERRIDE] ignore invalid SR_MAPPO_REPORT_GREEDY_SHARE_RATIO_OVERRIDE={forced_share_ratio!r}")
        if forced_share_ref_env:
            parsed_ref: Dict[float, float] = {}
            try:
                for token in forced_share_ref_env.split(","):
                    token = token.strip()
                    if not token:
                        continue
                    load_s, mbps_s = token.split(":")
                    parsed_ref[float(load_s.strip())] = float(mbps_s.strip())
            except Exception:
                parsed_ref = {}
                _report_log(
                    f"[OVERRIDE] ignore invalid SR_MAPPO_REPORT_GREEDY_SHARE_REFERENCE_PRE_MBPS_BY_LOAD={forced_share_ref_env!r}"
                )
            report_cfg.env.greedy_share_reference_pre_mbps_by_load = parsed_ref
        _report_log(
            f"[OVERRIDE] greedy share enabled: mode={report_cfg.env.greedy_urllc_share_mode} "
            f"ratio={float(report_cfg.env.greedy_urllc_share_ratio):.4f} "
            f"ref_points={len(getattr(report_cfg.env, 'greedy_share_reference_pre_mbps_by_load', {}) or {})}"
        )
    # Optional report-time override for hard-feasible greedy SIC gate.
    sic_override_env = os.environ.get("SR_MAPPO_REPORT_GREEDY_HF_EMBB_MIN_SIC_SNIR_DB_OVERRIDE", "").strip()
    if sic_override_env:
        try:
            report_cfg.env.greedy_hf_embb_min_sic_snir_db_override = float(sic_override_env)
            _report_log(
                "[OVERRIDE] greedy_hf_embb_min_sic_snir_db_override="
                f"{float(report_cfg.env.greedy_hf_embb_min_sic_snir_db_override):.3f} dB "
                f"(from SR_MAPPO_REPORT_GREEDY_HF_EMBB_MIN_SIC_SNIR_DB_OVERRIDE={sic_override_env})"
            )
        except ValueError:
            _report_log(
                "[OVERRIDE] ignore invalid SR_MAPPO_REPORT_GREEDY_HF_EMBB_MIN_SIC_SNIR_DB_OVERRIDE="
                f"{sic_override_env!r}"
            )
    minrate_scale_env = os.environ.get("SR_MAPPO_REPORT_EMBB_MIN_RATE_SCALE", "").strip()
    if minrate_scale_env:
        try:
            report_cfg.env.report_embb_min_rate_scale = float(minrate_scale_env)
            _report_log(
                "[OVERRIDE] report_embb_min_rate_scale="
                f"{float(report_cfg.env.report_embb_min_rate_scale):.4f} "
                f"(from SR_MAPPO_REPORT_EMBB_MIN_RATE_SCALE={minrate_scale_env})"
            )
        except ValueError:
            _report_log(
                "[OVERRIDE] ignore invalid SR_MAPPO_REPORT_EMBB_MIN_RATE_SCALE="
                f"{minrate_scale_env!r}"
            )
    # Report-only smoothing guard:
    # In greedy-only + pure eMBB (URLLC ratio forced to 0), if topology/channel are frozen,
    # episodes become near-identical and mean curves look stair-like/noisy across loads.
    # Disable freeze here so episodes_per_load performs true Monte Carlo averaging.
    if bool(greedy_only):
        forced_ratio = _resolve_forced_urllc_ratio(report_cfg)
        exp_line = str(getattr(report_cfg.training, "experiment_line", "") or "").strip().lower()
        pure_embb_mode = (forced_ratio == 0.0) or ("v8_greedy_share10_debug" in exp_line)
        if pure_embb_mode and (
            bool(getattr(report_cfg.env, "freeze_association_across_episodes", False))
            or bool(getattr(report_cfg.env, "freeze_channel_gains_across_episodes", False))
        ):
            report_cfg.env.freeze_association_across_episodes = False
            report_cfg.env.freeze_channel_gains_across_episodes = False
            _report_log(
                "[GREEDY] pure-eMBB report mode: disable freeze_assoc/freeze_channel for episode averaging."
            )
    _REPORT_TIMING_ENABLED = bool(getattr(report_cfg.training, "enable_timing_logs", False))
    _reset_report_runtime_cache()
    _REPORT_RUN_SEED_BASE = _init_report_run_seed_base(report_cfg)
    _report_log(f"Report run seed base: {_REPORT_RUN_SEED_BASE} (per-load paired; per-episode increments)")
    # Default to lite/fast-debug style report unless explicitly disabled by preset.
    # This keeps report generation concise without requiring `--fast` every run.
    report_lite_default = bool(getattr(report_cfg.training, "report_lite_default", True))
    fast_debug = bool(fast) or bool(getattr(report_cfg.training, "report_fast_debug", False)) or report_lite_default
    _REPORT_EPISODE_CACHE_ENABLED = not (bool(fast_debug) and bool(greedy_only))
    loads = loads or (FAST_LOADS if fast else DEFAULT_LOADS)
    loads_override_env = os.environ.get("SR_MAPPO_REPORT_LOADS_OVERRIDE", "").strip()
    if loads_override_env:
        try:
            parsed_loads = [float(x.strip()) for x in loads_override_env.split(",") if x.strip()]
            if parsed_loads:
                loads = parsed_loads
                _report_log(
                    f"[OVERRIDE] loads={loads} "
                    f"(from SR_MAPPO_REPORT_LOADS_OVERRIDE={loads_override_env})"
                )
        except ValueError:
            _report_log(
                f"[OVERRIDE] ignore invalid SR_MAPPO_REPORT_LOADS_OVERRIDE={loads_override_env!r}"
            )
    exp_line_norm = str(getattr(report_cfg.training, "experiment_line", "") or "").strip().lower()
    if bool(greedy_only) and bool(fast_debug) and (
        ("v8_greedy_mix37_debug" in exp_line_norm)
        or ("v8_greedy_mix55_debug" in exp_line_norm)
        or ("v8_greedy_mix73_debug" in exp_line_norm)
        or ("v8_greedy_mix010_debug" in exp_line_norm)
        or ("v8_greedy_mix100_debug" in exp_line_norm)
    ):
        episodes_per_load = 100
    # Runtime override for batch scripts.
    episodes_override_env = os.environ.get("SR_MAPPO_REPORT_EPISODES_PER_LOAD_OVERRIDE", "").strip()
    if episodes_override_env:
        try:
            episodes_per_load = max(1, int(episodes_override_env))
            _report_log(
                f"[OVERRIDE] episodes_per_load={episodes_per_load} "
                f"(from SR_MAPPO_REPORT_EPISODES_PER_LOAD_OVERRIDE={episodes_override_env})"
            )
        except ValueError:
            _report_log(
                f"[OVERRIDE] ignore invalid SR_MAPPO_REPORT_EPISODES_PER_LOAD_OVERRIDE={episodes_override_env!r}"
            )
    if fast_debug:
        if greedy_only:
            # Greedy-only fast debug should respect provided episodes_per_load for
            # quick iterative debugging.
            episodes_per_load = max(1, int(episodes_per_load))
        else:
            # Fast mode should still honor user-specified episode overrides so
            # comparisons can be made with stable statistics (e.g., ep10/ep20).
            episodes_per_load = max(1, int(episodes_per_load))
    representative_load = float(loads[-1]) if loads else REPRESENTATIVE_LOAD
    checkpoint, checkpoint_reason = _select_checkpoint(
        report_cfg,
        checkpoint_path=checkpoint_path,
        checkpoint_kind=checkpoint_kind,
    )
    primary_checkpoint_match_warning = _primary_checkpoint_match_warning(report_cfg, checkpoint_reason)
    if primary_checkpoint_match_warning and _require_primary_checkpoint_match(report_cfg):
        raise RuntimeError(
            f"{primary_checkpoint_match_warning}. "
            f"Requested primary_checkpoint_preference={getattr(report_cfg.training, 'primary_checkpoint_preference', None)!r}. "
            f"No matching primary checkpoint file was found in {str(CHECKPOINT_DIR.resolve())}."
        )
    checkpoint_cfg = _load_checkpoint_cfg(checkpoint)
    if _require_primary_checkpoint_match(report_cfg):
        ckpt_line = str(getattr(checkpoint_cfg.training, "experiment_line", "") or "")
        req_line = str(getattr(report_cfg.training, "experiment_line", "") or "")
        if ckpt_line and req_line and ckpt_line.strip() != req_line.strip():
            raise RuntimeError(
                "Checkpoint experiment_line mismatch: "
                f"checkpoint has '{ckpt_line}', but report requested '{req_line}'."
            )
    # Freeze a baseline-only config *before* checkpoint-driven env overrides.
    # This keeps greedy baseline reproducible across checkpoint changes.
    baseline_report_cfg = deepcopy(report_cfg)

    report_cfg.env.fixed_embb_baseline_policy = checkpoint_cfg.env.fixed_embb_baseline_policy
    report_cfg.env.include_frontier_progress_obs = bool(getattr(checkpoint_cfg.env, "include_frontier_progress_obs", False))
    report_cfg.env.include_quota_progress_obs = bool(getattr(checkpoint_cfg.env, "include_quota_progress_obs", False))
    checkpoint_meta = _checkpoint_metadata(checkpoint)
    best_throughput_meta = _checkpoint_metadata(CHECKPOINT_DIR / f"{report_cfg.training.run_name}_best_throughput.pt") if (CHECKPOINT_DIR / f"{report_cfg.training.run_name}_best_throughput.pt").exists() else {}
    best_balanced_meta = _checkpoint_metadata(CHECKPOINT_DIR / f"{report_cfg.training.run_name}_best_balanced.pt") if (CHECKPOINT_DIR / f"{report_cfg.training.run_name}_best_balanced.pt").exists() else {}
    greedy_baseline_mode = _greedy_baseline_mode(report_cfg)
    _report_log(f"Checkpoint selected: {checkpoint.name}")
    _report_log(f"Checkpoint selection reason: {checkpoint_reason}")
    if primary_checkpoint_match_warning:
        _report_log(f"Checkpoint selection warning: {primary_checkpoint_match_warning}")
    _report_log(f"Experiment line: {experiment_label(report_cfg.training.experiment_line)}")
    _report_log(f"Greedy baseline mode: {greedy_baseline_mode}")
    _report_log(f"Comparison baseline label: {_baseline_label(greedy_baseline_mode)}")
    _report_log(
        "Phase control: "
        f"phase={report_cfg.env.phase} | "
        f"learn_embb_baseline={bool(report_cfg.env.learn_embb_baseline)} | "
        f"learn_phase0_embb_power={bool(getattr(report_cfg.env, 'learn_phase0_embb_power', True))} | "
        f"allow_phase_a_embb_power_adjustment={bool(report_cfg.env.allow_phase_a_embb_power_adjustment)} | "
        f"freeze_assoc={bool(getattr(report_cfg.env, 'freeze_association_across_episodes', False))} | "
        f"freeze_channel={bool(getattr(report_cfg.env, 'freeze_channel_gains_across_episodes', False))}"
    )
    _report_log(
        "Shield control: "
        f"action_masking={bool(report_cfg.shield.enable_action_masking)} | "
        f"feasibility_shield={bool(report_cfg.shield.enable_feasibility_shield)} | "
        f"joint_reliability_rewrite={bool(report_cfg.shield.apply_joint_reliability_rewrite)} | "
        f"greedy_fallback={bool(report_cfg.shield.enable_greedy_fallback)}"
    )
    _report_log(
        "Checkpoint meta: "
        f"path={checkpoint_meta['path']} | "
        f"mtime={checkpoint_meta['mtime']} | "
        f"size={checkpoint_meta['size_bytes']} bytes | "
        f"iteration={checkpoint_meta['iteration']} | "
        f"phaseA_embb_power_active={checkpoint_meta.get('phase_a_embb_power_runtime_enabled', False)} | "
        f"phaseA_embb_power_exercised={checkpoint_meta.get('phase_a_embb_power_exercised', False)} | "
        f"phaseA_changed_ratio={float(checkpoint_meta.get('phase_a_embb_power_changed_ratio', 0.0)):.3f}"
    )

    if fast_debug:
        # Fast debug report: only compute the selected baseline vs MAPPO core KPIs
        # and a small lambda sweep at a representative load.
        _report_log("Fast debug report enabled: generating only core KPIs + owner debug + Phase-A power + lambda sweep + owner map.")
        _base_total_per_uav, base_poisson_rate, base_poisson_fixed = _base_profile()
        debug_num_uavs = 0
        debug_num_rbs = 0
        debug_reliability_target = 0.0
        try:
            sys0, _urllc0, _embb0, _algo0, sim0 = _build_main_like_configs()
            debug_num_uavs = int(getattr(sys0, "num_uavs", 0) or 0)
            debug_num_rbs = int(getattr(sys0, "num_subcarriers", 0) or 0)
            debug_reliability_target = float(
                getattr(_urllc0, "target_reliability", getattr(_urllc0, "reliability_target", 0.0)) or 0.0
            )
            _report_log(
                f"Scenario dims: num_uavs={debug_num_uavs} | "
                f"num_subcarriers={debug_num_rbs} | "
                f"num_minislots={int(getattr(sys0, 'num_minislots', 0))} | "
                f"default_urllc_poisson_rate={float(getattr(sim0, 'urllc_poisson_rate', float('nan'))):.0f}"
            )
        except Exception:
            _report_log("Scenario dims: (unavailable)")
        _report_log("URLLC throughput is slot-based estimate: scheduled packets x avg packet bits / 1 ms slot.")
        _report_log(f"Core KPI loads: {[float(x) for x in list(loads)]} | episodes_per_load={int(episodes_per_load)}")
        if greedy_only:
            _report_log("Fast greedy-only mode: skipping SR-MAPPO sweep.")
            rl_metrics = {}
            _rl_rep = {}
        else:
            rl_metrics, _rl_rep = run_mappo_sweep(loads, episodes_per_load, checkpoint, base_cfg=report_cfg)
            rl_metrics["terminal_admission_floor_soft_penalty_floor"] = float(
                getattr(report_cfg.reward, "terminal_admission_floor_soft_penalty_floor", 0.0) or 0.0
            )
        if greedy_baseline_mode == "hard_feasible_throughput_greedy":
            baseline_metrics, _baseline_rep = run_hard_feasible_throughput_greedy_sweep(
                loads,
                episodes_per_load,
                checkpoint,
                base_cfg=baseline_report_cfg,
                verbose_per_episode=(not (fast_debug and greedy_only and (not FAST_GREEDY_ONLY_VERBOSE_PER_EPISODE))),
            )
        elif greedy_baseline_mode == "myopic_throughput_greedy":
            baseline_metrics, _baseline_rep = run_myopic_throughput_greedy_sweep(
                loads,
                episodes_per_load,
                checkpoint,
                base_cfg=baseline_report_cfg,
            )
        elif greedy_baseline_mode == "original":
            baseline_metrics, _baseline_rep = run_greedy_sweep(loads, episodes_per_load)
        else:
            baseline_metrics, _baseline_rep, _frozen = run_selected_greedy_sweep(
                loads,
                episodes_per_load,
                baseline_report_cfg,
                checkpoint,
            )
        # Keep UAV-UE distribution available in fast-debug mode.
        if greedy_only:
            uav_ue_distribution_bundle = {}
            uav_ue_distribution_paths = []
        else:
            uav_ue_distribution_bundle = _build_uav_ue_distribution_bundle(_rl_rep, loads)
            uav_ue_distribution_paths = plot_uav_ue_distribution(_rl_rep, loads)

        if greedy_only:
            # In greedy-only mode, use baseline as both inputs so no MAPPO series appears.
            rl_metrics = dict(baseline_metrics)

        # Derived separation diagnostics (arrays aligned to `loads`).
        try:
            l = np.asarray(rl_metrics.get("loads", []), dtype=float)
            def _align(series: Dict, key: str) -> np.ndarray:
                y = np.asarray(series.get(key, []), dtype=float)
                if y.size == l.size:
                    return y
                if y.size == 2 * l.size and l.size > 0:
                    return y.reshape(int(l.size), 2).mean(axis=1)
                return np.zeros_like(l, dtype=float)
            srv_gain = _align(rl_metrics, "embb_service_ratio") - _align(baseline_metrics, "embb_service_ratio")
            min_gain = _align(rl_metrics, "embb_min_rate_satisfaction_ratio") - _align(baseline_metrics, "embb_min_rate_satisfaction_ratio")
            served_gain = _align(rl_metrics, "embb_served_user_count") - _align(baseline_metrics, "embb_served_user_count")
            rl_metrics["service_gain_vs_greedy"] = [float(x) for x in srv_gain.tolist()]
            rl_metrics["minrate_gain_vs_greedy"] = [float(x) for x in min_gain.tolist()]
            rl_metrics["served_user_gain_vs_greedy"] = [float(x) for x in served_gain.tolist()]
        except Exception:
            pass

        legend_title = (
            f"{report_cfg.training.experiment_line} | "
            f"baseline={greedy_baseline_mode} | "
            f"lambda={float(base_poisson_rate):.0f} (fixed={bool(base_poisson_fixed)})"
        )
        if greedy_only:
            output_paths = [
                str(plot_core_kpi_debug_fast(
                    rl_metrics,
                    baseline_metrics,
                    baseline_label="Greedy",
                    legend_title=legend_title,
                    greedy_only=True,
                )),
                str(plot_urllc_arrival_admit_debug(
                    baseline_metrics,
                    episodes_per_load=int(episodes_per_load),
                    title_suffix=f"{report_cfg.training.experiment_line} | Greedy only",
                )),
                str(plot_greedy_candidate_rejection_debug(
                    baseline_metrics,
                    title_suffix=f"{report_cfg.training.experiment_line} | Greedy candidate diagnostics",
                )),
                str(plot_mode_action_share_compare(
                    baseline_metrics,
                    baseline_metrics,
                    baseline_label="Greedy",
                )),
                str(plot_mode_raw_vs_executed_compare(baseline_metrics)),
                str(plot_min_rate_satisfied_count_compare(
                    baseline_metrics,
                    baseline_metrics,
                    baseline_label="Greedy",
                )),
                str(plot_admitted_urllc_packets_compare(
                    baseline_metrics,
                    baseline_metrics,
                    baseline_label="Greedy",
                )),
            ]
        else:
            output_paths = [
                str(plot_core_kpi_debug_fast(
                    rl_metrics,
                    baseline_metrics,
                    baseline_label="Greedy",
                    legend_title=legend_title,
                )),
                str(plot_urllc_reliability_debug_fast(
                    rl_metrics,
                    baseline_metrics,
                    baseline_label="Greedy",
                    reliability_target=float(debug_reliability_target),
                )),
                str(plot_intercell_vs_load_debug_fast(
                    rl_metrics,
                    baseline_metrics,
                    baseline_label="Greedy",
                )),
                str(plot_intercell_rate_loss_debug_fast(
                    rl_metrics,
                    baseline_metrics,
                    baseline_label="Greedy",
                )),
                str(plot_local_puncture_deduction_debug_fast(
                    rl_metrics,
                    baseline_metrics,
                    baseline_label="Greedy",
                )),
                str(plot_owner_effective_debug_fast(rl_metrics)),
                str(plot_phaseA_power_debug_fast(rl_metrics)),
                str(plot_phaseA_negative_only_debug_fast(
                    rl_metrics,
                    baseline_metrics,
                    baseline_label="Greedy",
                )),
                str(plot_service_recovery_debug_fast(
                    rl_metrics,
                    baseline_metrics,
                    baseline_label="Greedy",
                )),
                str(plot_service_separation_debug_fast(
                    rl_metrics,
                    baseline_metrics,
                    baseline_label="Greedy",
                )),
                str(plot_service_target_debug_fast(
                    rl_metrics,
                    baseline_metrics,
                    baseline_label="Greedy",
                    urllc_admission_floor=float(getattr(report_cfg.reward, "terminal_admission_floor_soft_penalty_floor", 0.0) or 0.0),
                )),
                str(plot_service_oracle_debug_fast(
                    rl_metrics,
                    baseline_metrics,
                    baseline_label="Greedy",
                    num_uavs=int(debug_num_uavs),
                    num_rbs=int(debug_num_rbs),
                )),
                str(plot_mode_action_share_compare(
                    rl_metrics,
                    baseline_metrics,
                    baseline_label="Greedy",
                )),
                str(plot_mode_raw_vs_executed_compare(rl_metrics)),
                str(plot_min_rate_satisfied_count_compare(
                    rl_metrics,
                    baseline_metrics,
                    baseline_label="Greedy",
                )),
                str(plot_admitted_urllc_packets_compare(
                    rl_metrics,
                    baseline_metrics,
                    baseline_label="Greedy",
                )),
            ]
        sweep_enabled_cfg = bool(getattr(report_cfg.training, "report_enable_lambda_sweep", False))
        sweep_load = float(getattr(report_cfg.training, "report_lambda_sweep_load", 15.0) or 15.0)
        sweep_values = list(getattr(report_cfg.training, "report_lambda_sweep_values", [4.0, 8.0, 12.0, 16.0]) or [4.0, 8.0, 12.0, 16.0])
        sweep_episodes = int(getattr(report_cfg.training, "report_lambda_sweep_episodes_per_lambda", 50) or 50)
        sweep_enable_env = os.environ.get("SR_MAPPO_REPORT_ENABLE_LAMBDA_SWEEP", "").strip().lower()
        sweep_enabled_env = sweep_enable_env in {"1", "true", "yes", "on"}
        sweep_enabled = sweep_enabled_cfg or sweep_enabled_env
        sweep_disable_env = os.environ.get("SR_MAPPO_REPORT_DISABLE_LAMBDA_SWEEP", "").strip().lower()
        sweep_disabled = sweep_disable_env in {"1", "true", "yes", "on"}
        sweep_episodes_override_env = os.environ.get("SR_MAPPO_REPORT_LAMBDA_SWEEP_EPISODES_OVERRIDE", "").strip()
        if sweep_episodes_override_env:
            try:
                sweep_episodes = max(1, int(sweep_episodes_override_env))
                _report_log(
                    f"[OVERRIDE] lambda_sweep_episodes_per_lambda={sweep_episodes} "
                    f"(from SR_MAPPO_REPORT_LAMBDA_SWEEP_EPISODES_OVERRIDE={sweep_episodes_override_env})"
                )
            except ValueError:
                _report_log(
                    "[OVERRIDE] ignore invalid SR_MAPPO_REPORT_LAMBDA_SWEEP_EPISODES_OVERRIDE="
                    f"{sweep_episodes_override_env!r}"
                )
        lambda_series = {}
        if not greedy_only and sweep_enabled and not sweep_disabled:
            _report_log(
                f"Lambda sweep: load={float(sweep_load):.1f} | values={[float(v) for v in list(sweep_values)]} | episodes_per_lambda={int(sweep_episodes)}"
            )
            lambda_series = run_lambda_sweep_debug(
                sweep_load,
                [float(v) for v in sweep_values],
                episodes_per_lambda=max(1, sweep_episodes),
                checkpoint_path=checkpoint,
                baseline_mode=greedy_baseline_mode,
                base_cfg=baseline_report_cfg,
            )
            output_paths.append(str(plot_lambda_sweep_debug_fast(lambda_series, baseline_label="Greedy")))
            output_paths.extend([str(path) for path in uav_ue_distribution_paths])
        elif not greedy_only:
            if sweep_disabled:
                _report_log(
                    "[OVERRIDE] lambda sweep disabled "
                    f"(from SR_MAPPO_REPORT_DISABLE_LAMBDA_SWEEP={sweep_disable_env!r})"
                )
            else:
                _report_log(
                    "Lambda sweep skipped by default "
                    "(set report_enable_lambda_sweep=true or SR_MAPPO_REPORT_ENABLE_LAMBDA_SWEEP=1 to enable)."
                )
            output_paths.extend([str(path) for path in uav_ue_distribution_paths])

        # Owner map slot visualization (episode=0, slot=0): greedy snapshot vs MAPPO.
        if not greedy_only:
            try:
                base_sys, base_urllc, base_embb, base_algo, base_sim = _build_main_like_configs()
                sys_cfg, urllc_cfg, embb_cfg, algo_cfg, sim_cfg = _configure_density_scenario(
                    float(sweep_load), base_sys, base_urllc, base_embb, base_algo, base_sim
                )
                if hasattr(base_sim, "urllc_user_ratio"):
                    sim_cfg.urllc_user_ratio = base_sim.urllc_user_ratio
                if hasattr(sim_cfg, "fixed_urllc_poisson_rate"):
                    sim_cfg.fixed_urllc_poisson_rate = bool(base_poisson_fixed)
                if hasattr(sim_cfg, "urllc_poisson_rate"):
                    sim_cfg.urllc_poisson_rate = float(base_poisson_rate)
                sim_cfg.verbose = False
                owner_env = SRMAPPOPhaseAEnv(sys_cfg, urllc_cfg, embb_cfg, algo_cfg, sim_cfg, report_cfg)
                _owner_cfg, owner_model = _build_model_for_env(owner_env, checkpoint)
                episode = run_env_episode(
                    owner_env,
                    model=owner_model,
                    cfg=_owner_cfg,
                    seed=0,
                    collect_trace=False,
                    use_greedy=False,
                    cache_tag=f"owner_map_slot_{sweep_load}_{base_poisson_rate}",
                )
                greedy_owner_map = episode.get("snapshot_owner_per_uav_rb", None)
                policy_owner_map = episode.get("owner_per_uav_rb", None)
                if greedy_owner_map is not None and policy_owner_map is not None:
                    output_paths.append(str(plot_owner_map_slot_debug(
                        greedy_owner_map=np.asarray(greedy_owner_map, dtype=int),
                        policy_owner_map=np.asarray(policy_owner_map, dtype=int),
                        experiment_line=str(report_cfg.training.experiment_line),
                        load=float(sweep_load),
                        poisson_rate=float(base_poisson_rate),
                        baseline_label=str(_baseline_label(greedy_baseline_mode)),
                    )))
            except Exception as exc:
                _report_log(f"[WARN] owner_map_slot_debug generation failed: {exc}")

        try:
            sys0, _urllc0, _embb0, _algo0, sim0 = _build_main_like_configs()
            default_poisson_rate = float(getattr(sim0, "urllc_poisson_rate", float("nan")))
            num_uavs = int(getattr(sys0, "num_uavs", 0))
            num_subcarriers = int(getattr(sys0, "num_subcarriers", 0))
            num_minislots = int(getattr(sys0, "num_minislots", 0))
        except Exception:
            default_poisson_rate = float("nan")
            num_uavs = 0
            num_subcarriers = 0
            num_minislots = 0

        pairing_fairness_audit = _build_pairing_fairness_audit(rl_metrics, baseline_metrics)
        if bool(pairing_fairness_audit.get("paired_all", False)):
            _report_log(
                "[PAIRING] fairness audit passed "
                f"| loads={int(pairing_fairness_audit.get('loads_compared', 0))} "
                f"| episode_pairs={int(pairing_fairness_audit.get('total_episode_pairs', 0))}"
            )
        else:
            _report_log(
                "[PAIRING][WARN] fairness audit failed "
                f"| mismatched_episode_pairs={int(pairing_fairness_audit.get('mismatched_episode_pairs', 0))} "
                f"| missing_episode_pairs={int(pairing_fairness_audit.get('missing_episode_pairs', 0))} "
                f"| mismatch_key_counts={json.dumps(pairing_fairness_audit.get('mismatch_key_counts', {}), sort_keys=True)}"
            )

        metrics_payload = {
            "checkpoint": str(checkpoint),
            "checkpoint_selection_reason": str(checkpoint_reason),
            "experiment_line": str(report_cfg.training.experiment_line),
            "selected_baseline_key": str(greedy_baseline_mode),
            "selected_baseline_label": str(_baseline_label(greedy_baseline_mode)),
            "comparison_baseline_key": str(greedy_baseline_mode),
            "comparison_baseline_label": str(_baseline_label(greedy_baseline_mode)),
            "urllc_poisson_rate": float(base_poisson_rate),
            "fixed_urllc_poisson_rate": bool(base_poisson_fixed),
            "default_urllc_poisson_rate": float(default_poisson_rate) if np.isfinite(default_poisson_rate) else None,
            "num_uavs": int(num_uavs),
            "num_subcarriers": int(num_subcarriers),
            "num_minislots": int(num_minislots),
            "urllc_throughput_definition": "slot_based_estimate: scheduled_packets * avg_packet_bits / 1ms_slot",
            "loads": [float(x) for x in list(loads)],
            "episodes_per_load": int(episodes_per_load),
            "same_scenario_pairing_enabled": True,
            "pairing_fairness_audit": pairing_fairness_audit,
            "report_run_seed_base": int(_REPORT_RUN_SEED_BASE if _REPORT_RUN_SEED_BASE is not None else 0),
            "seeds_used_by_load": {
                str(float(load)): [int(_report_seed_base(i, report_cfg) + ep) for ep in range(int(episodes_per_load))]
                for i, load in enumerate(loads)
            },
            "lambda_sweep_load": float(sweep_load),
            "lambda_sweep_values": [float(v) for v in list(sweep_values)],
            "lambda_sweep_episodes_per_lambda": int(sweep_episodes),
            "sr_mappo": rl_metrics,
            "greedy": baseline_metrics,
            "greedy_representative": {str(float(k)): v for k, v in _baseline_rep.items()},
            "lambda_sweep_debug": lambda_series,
            "uav_ue_distribution": uav_ue_distribution_bundle,
            "uav_ue_distribution_plot_paths": [str(path) for path in uav_ue_distribution_paths],
        }
        metrics_path = RESULTS_DIR / "sr_mappo_report_metrics.json"
        metrics_path.write_text(json.dumps(metrics_payload, indent=2, default=_json_default), encoding="utf-8")
        summary_lines = []
        summary_lines.append("SR-MAPPO fast debug summary")
        summary_lines.append(f"checkpoint: {checkpoint}")
        summary_lines.append(f"checkpoint_selection_reason: {checkpoint_reason}")
        summary_lines.append(f"experiment_line: {report_cfg.training.experiment_line}")
        summary_lines.append(f"selected_baseline_key: {greedy_baseline_mode}")
        summary_lines.append(f"selected_baseline_label: {_baseline_label(greedy_baseline_mode)}")
        summary_lines.append("URLLC throughput is slot-based estimate: scheduled packets x avg packet bits / 1 ms slot.")
        summary_lines.append(f"urllc_poisson_rate: {float(base_poisson_rate):.0f} (fixed={bool(base_poisson_fixed)})")
        summary_lines.append(f"report_run_seed_base: {int(_REPORT_RUN_SEED_BASE if _REPORT_RUN_SEED_BASE is not None else 0)}")
        try:
            _sys0, _urllc0, _embb0, _algo0, _sim0 = _build_main_like_configs()
            default_poisson = float(getattr(_sim0, "urllc_poisson_rate", float("nan")))
            summary_lines.append(f"default_urllc_poisson_rate: {default_poisson:.0f}")
            summary_lines.append(f"num_uavs: {int(getattr(_sys0, 'num_uavs', 0))}")
            summary_lines.append(f"num_subcarriers: {int(getattr(_sys0, 'num_subcarriers', 0))}")
            summary_lines.append(f"num_minislots: {int(getattr(_sys0, 'num_minislots', 0))}")
        except Exception:
            summary_lines.append("default_urllc_poisson_rate: n/a")
            summary_lines.append("num_uavs: n/a")
            summary_lines.append("num_subcarriers: n/a")
            summary_lines.append("num_minislots: n/a")
        summary_lines.append(f"loads: {[float(x) for x in list(loads)]}")
        summary_lines.append(f"episodes_per_load: {int(episodes_per_load)}")
        summary_lines.append(
            "pairing_fairness: "
            f"paired_all={int(bool(pairing_fairness_audit.get('paired_all', False)))} "
            f"mismatched_episode_pairs={int(pairing_fairness_audit.get('mismatched_episode_pairs', 0))} "
            f"missing_episode_pairs={int(pairing_fairness_audit.get('missing_episode_pairs', 0))}"
        )
        if pairing_fairness_audit.get("mismatch_key_counts"):
            summary_lines.append(
                "pairing_fairness_mismatch_keys: "
                + json.dumps(pairing_fairness_audit.get("mismatch_key_counts", {}), sort_keys=True)
            )
        summary_lines.append(f"lambda_sweep_load: {float(sweep_load)}")
        summary_lines.append(f"lambda_sweep_values: {[float(v) for v in list(sweep_values)]}")
        summary_lines.append(f"lambda_sweep_episodes_per_lambda: {int(sweep_episodes)}")
        # URLLC throughput definition inputs (slot-based estimate).
        try:
            slot_dur = float(np.asarray(rl_metrics.get("urllc_slot_duration_s", [1.0e-3]), dtype=float)[0])
        except Exception:
            slot_dur = 1.0e-3
        try:
            pkt_bits = float(np.asarray(rl_metrics.get("urllc_packet_bits_mean", [160.0]), dtype=float)[0])
        except Exception:
            pkt_bits = 160.0
        summary_lines.append(f"urllc_slot_duration_s: {slot_dur:.6f}")
        summary_lines.append(f"urllc_packet_bits_mean: {pkt_bits:.1f}")
        summary_lines.append("")
        summary_lines.append("Per-load summary:")
        summary_lines.append(
            "load | eMBB:URLLC | total_load/UAV | MAPPO embb(Mbps) | Greedy embb(Mbps) | "
            "MAPPO adm | Greedy adm | MAPPO srv | Greedy srv | MAPPO min | Greedy min | "
            "MAPPO served | Greedy served | MAPPO avg_srv(Mbps) | Greedy avg_srv(Mbps) | "
            "MAPPO ceil(Mbps) | Greedy ceil(Mbps) | "
            "MAPPO urllc(Mbps) | Greedy urllc(Mbps) | MAPPO pwr(mW) | Greedy pwr(mW) | ov/pu (M) | ov/pu (G)"
        )

        rl_loads = np.asarray(rl_metrics.get("loads", []), dtype=float)
        for idx, load_val in enumerate(rl_loads.tolist()):
            def _at(data: Dict, key: str, default: float = 0.0) -> float:
                arr = data.get(key, [])
                if isinstance(arr, list) and idx < len(arr):
                    return float(arr[idx])
                return float(default)
            embb_n = int(_at(rl_metrics, "embb_user_count", 0.0))
            urllc_n = int(_at(rl_metrics, "urllc_user_count", 0.0))
            m_embb = _at(rl_metrics, "embb_rate") / 1.0e6
            g_embb = _at(baseline_metrics, "embb_rate") / 1.0e6
            m_adm = _at(rl_metrics, "urllc_admission")
            g_adm = _at(baseline_metrics, "urllc_admission")
            m_srv = _at(rl_metrics, "embb_service_ratio")
            g_srv = _at(baseline_metrics, "embb_service_ratio")
            m_min = _at(rl_metrics, "embb_min_rate_satisfaction_ratio")
            g_min = _at(baseline_metrics, "embb_min_rate_satisfaction_ratio")
            m_served = _at(rl_metrics, "embb_served_user_count", m_srv * embb_n)
            g_served = _at(baseline_metrics, "embb_served_user_count", g_srv * embb_n)
            m_avg_srv = _at(rl_metrics, "avg_throughput_per_served_embb_user") / 1.0e6
            g_avg_srv = _at(baseline_metrics, "avg_throughput_per_served_embb_user") / 1.0e6
            m_ceil = (m_served * (m_avg_srv * 1.0e6)) / 1.0e6
            g_ceil = (g_served * (g_avg_srv * 1.0e6)) / 1.0e6
            m_urllc = _at(rl_metrics, "urllc_throughput_bps_slot_est", _at(rl_metrics, "urllc_throughput_bps_est")) / 1.0e6
            g_urllc = _at(baseline_metrics, "urllc_throughput_bps_slot_est", _at(baseline_metrics, "urllc_throughput_bps_est")) / 1.0e6
            m_pwr = _at(rl_metrics, "total_power") * 1e3
            g_pwr = _at(baseline_metrics, "total_power") * 1e3
            m_ov = _at(rl_metrics, "overlay_ratio")
            m_pu = _at(rl_metrics, "puncture_ratio")
            g_ov = _at(baseline_metrics, "overlay_ratio")
            g_pu = _at(baseline_metrics, "puncture_ratio")
            try:
                _sys0, _urllc0, _embb0, _algo0, _sim0 = _build_main_like_configs()
                uavs = max(int(getattr(_sys0, "num_uavs", 1) or 1), 1)
            except Exception:
                uavs = 1
            total_load_per_uav = float((embb_n + urllc_n) / max(uavs, 1))
            summary_lines.append(
                f"{load_val:>4.1f} | {embb_n}:{urllc_n} | {total_load_per_uav:>6.1f} | {m_embb:>7.2f} | {g_embb:>7.2f} | "
                f"{m_adm:>5.3f} | {g_adm:>5.3f} | {m_srv:>5.3f} | {g_srv:>5.3f} | {m_min:>5.3f} | {g_min:>5.3f} | "
                f"{m_served:>6.1f} | {g_served:>6.1f} | {m_avg_srv:>7.2f} | {g_avg_srv:>7.2f} | "
                f"{m_ceil:>7.2f} | {g_ceil:>7.2f} | "
                f"{m_urllc:>7.3f} | {g_urllc:>7.3f} | {m_pwr:>7.1f} | {g_pwr:>7.1f} | {m_ov:>4.2f}/{m_pu:>4.2f} | {g_ov:>4.2f}/{g_pu:>4.2f}"
            )

        summary_lines.append("")
        summary_lines.append("Throughput ceiling what-ifs (Mbps): using avg_throughput_per_served_embb_user × (embb_user_count × service_ratio).")
        summary_lines.append("load | MAPPO est@0.25 | MAPPO est@0.30 | Greedy est@0.25 | Greedy est@0.30")
        rl_loads = np.asarray(rl_metrics.get("loads", []), dtype=float)
        for idx, load_val in enumerate(rl_loads.tolist()):
            def _at(data: Dict, key: str, default: float = 0.0) -> float:
                arr = data.get(key, [])
                if isinstance(arr, list) and idx < len(arr):
                    return float(arr[idx])
                return float(default)
            embb_n = float(_at(rl_metrics, "embb_user_count", 0.0))
            m_avg = float(_at(rl_metrics, "avg_throughput_per_served_embb_user", 0.0))
            g_avg = float(_at(baseline_metrics, "avg_throughput_per_served_embb_user", 0.0))
            m_025 = (embb_n * 0.25 * m_avg) / 1.0e6
            m_030 = (embb_n * 0.30 * m_avg) / 1.0e6
            g_025 = (embb_n * 0.25 * g_avg) / 1.0e6
            g_030 = (embb_n * 0.30 * g_avg) / 1.0e6
            summary_lines.append(f"{load_val:>4.1f} | {m_025:>10.2f} | {m_030:>10.2f} | {g_025:>11.2f} | {g_030:>11.2f}")

        summary_lines.append("")
        # Automatic diagnostics text.
        owner_chg = float(np.nanmean(np.asarray(rl_metrics.get("phase0_owner_change_ratio_vs_snapshot_executed", []), dtype=float))) if rl_metrics.get("phase0_owner_change_ratio_vs_snapshot_executed") else 0.0
        owner_eff = float(np.nanmean(np.asarray(rl_metrics.get("phase0_owner_changed_and_effective_ratio", []), dtype=float))) if rl_metrics.get("phase0_owner_changed_and_effective_ratio") else 0.0
        sat = float(np.nanmean(np.asarray(rl_metrics.get("phase_a_embb_power_raw_saturation_ratio", []), dtype=float))) if rl_metrics.get("phase_a_embb_power_raw_saturation_ratio") else 0.0
        cap = float(np.nanmean(np.asarray(rl_metrics.get("phase_a_embb_power_cap_hit_ratio", []), dtype=float))) if rl_metrics.get("phase_a_embb_power_cap_hit_ratio") else 0.0
        final_std = float(np.nanmean(np.asarray(rl_metrics.get("phase_a_embb_power_final_std", []), dtype=float))) if rl_metrics.get("phase_a_embb_power_final_std") else 0.0
        summary_lines.append("Auto-diagnostics:")
        summary_lines.append(f"- owner_executed_change_ratio_vs_snapshot (mean): {owner_chg:.3f}")
        summary_lines.append(f"- owner_changed_and_effective_ratio (mean): {owner_eff:.3f}")
        summary_lines.append(f"- phaseA_raw_saturation_ratio (mean): {sat:.3f}")
        summary_lines.append(f"- phaseA_cap_hit_ratio (mean): {cap:.3f}")
        summary_lines.append(f"- phaseA_final_delta_std (mean): {final_std:.4f}")
        summary_lines.append(f"- admission_gain_mean (MAPPO - Greedy): {float(np.nanmean(np.asarray(rl_metrics.get('urllc_admission', []), dtype=float) - np.asarray(baseline_metrics.get('urllc_admission', []), dtype=float))):.4f}" if rl_metrics.get('urllc_admission') and baseline_metrics.get('urllc_admission') else "- admission_gain_mean: n/a")
        summary_lines.append(f"- embb_throughput_diff_mean_Mbps (MAPPO - Greedy): {float(np.nanmean((np.asarray(rl_metrics.get('embb_rate', []), dtype=float) - np.asarray(baseline_metrics.get('embb_rate', []), dtype=float)) / 1.0e6)):.3f}" if rl_metrics.get('embb_rate') and baseline_metrics.get('embb_rate') else "- embb_throughput_diff_mean_Mbps: n/a")
        summary_lines.append(f"- total_power_diff_mean_mW (MAPPO - Greedy): {float(np.nanmean((np.asarray(rl_metrics.get('total_power', []), dtype=float) - np.asarray(baseline_metrics.get('total_power', []), dtype=float)) * 1e3)):.2f}" if rl_metrics.get('total_power') and baseline_metrics.get('total_power') else "- total_power_diff_mean_mW: n/a")

        summary_path = RESULTS_DIR / "fast_debug_summary.txt"
        summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
        output_paths.append(str(summary_path))
        reward_plot = plot_training_reward_debug(
            _load_history_for_report(report_cfg, checkpoint_path=checkpoint)
        )
        if reward_plot is not None:
            output_paths.append(str(reward_plot))
        frontier_json_path = RESULTS_DIR / "throughput_admission_frontier.json"
        frontier_json_path.write_text(json.dumps({}, indent=2), encoding="utf-8")
        _write_report_manifest({
            "fast_debug": True,
            "checkpoint": str(checkpoint),
            "checkpoint_reason": str(checkpoint_reason),
            "experiment_line": str(report_cfg.training.experiment_line),
            "selected_baseline_key": str(greedy_baseline_mode),
            "loads": [float(x) for x in list(loads)],
            "episodes_per_load": int(episodes_per_load),
            "outputs": output_paths,
        })
        _report_timing_log(f"generate_report fast_debug sec={perf_counter() - report_start:.3f}")
        return {
            "checkpoint": str(checkpoint),
            "output_paths": output_paths,
            "metrics_path": str(metrics_path),
            "frontier_json_path": str(frontier_json_path),
        }
    _report_log(
        "Checkpoint overview: "
        f"best_throughput_iter={best_throughput_meta.get('iteration')} | "
        f"best_balanced_iter={best_balanced_meta.get('iteration')}"
    )
    _report_log(f"Loads: {loads}")
    _report_log(f"Episodes per load: {episodes_per_load}")
    if fast:
        _report_log("FAST mode enabled: reduced loads/episodes/slots.")
    frozen_greedy_payload = None
    greedy_bundle, greedy_reps, frozen_greedy_payload = run_report_greedy_bundle(loads, episodes_per_load, report_cfg, checkpoint)
    selected_mode = greedy_baseline_mode
    if selected_mode == 'frozen_json':
        selected_mode = str((frozen_greedy_payload or {}).get('greedy_baseline_mode', 'original') or 'original').strip().lower()
    if selected_mode not in {
        'original',
        'original_greedy_normal_v1',
        'original_greedy_normal_v2',
        'matched_fixed_embb',
        'throughput_feasible_oracle',
        'throughput_biased_greedy',
        'hard_feasible_throughput_greedy',
        'global_frontier_greedy',
        'myopic_throughput_greedy',
        'throughput_only_greedy',
        'rate_loss_min_greedy',
        'force_admit_minloss_greedy',
        'channel_only_greedy',
    }:
        selected_mode = 'original'
    comparison_baseline_mode = selected_mode
    _report_log(f"Selected baseline label: {_baseline_label(comparison_baseline_mode)}")
    if selected_mode == 'original_greedy_normal_v1':
        greedy_metrics = greedy_bundle['original_greedy_normal_v1']
        greedy_rep = greedy_reps['original_greedy_normal_v1']
    elif selected_mode == 'throughput_feasible_oracle':
        greedy_metrics = greedy_bundle['throughput_feasible_oracle']
        greedy_rep = greedy_reps['throughput_feasible_oracle']
    elif selected_mode == 'throughput_biased_greedy':
        greedy_metrics = greedy_bundle['throughput_biased_greedy']
        greedy_rep = greedy_reps['throughput_biased_greedy']
    elif selected_mode == 'original_greedy_normal_v2':
        greedy_metrics = greedy_bundle['original_greedy_normal_v2']
        greedy_rep = greedy_reps['original_greedy_normal_v2']
    elif selected_mode == 'matched_fixed_embb':
        greedy_metrics = greedy_bundle['matched_fixed_embb']
        greedy_rep = greedy_reps['matched_fixed_embb']
    elif selected_mode == 'throughput_feasible_oracle':
        greedy_metrics = greedy_bundle['throughput_feasible_oracle']
        greedy_rep = greedy_reps['throughput_feasible_oracle']
    elif selected_mode == 'throughput_only_greedy':
        greedy_metrics = greedy_bundle['throughput_only_greedy']
        greedy_rep = greedy_reps['throughput_only_greedy']
    elif selected_mode == 'rate_loss_min_greedy':
        greedy_metrics = greedy_bundle['rate_loss_min_greedy']
        greedy_rep = greedy_reps['rate_loss_min_greedy']
    elif selected_mode == 'force_admit_minloss_greedy':
        greedy_metrics = greedy_bundle['force_admit_minloss_greedy']
        greedy_rep = greedy_reps['force_admit_minloss_greedy']
    elif selected_mode == 'hard_feasible_throughput_greedy':
        greedy_metrics = greedy_bundle['hard_feasible_throughput_greedy']
        greedy_rep = greedy_reps['hard_feasible_throughput_greedy']
    elif selected_mode == 'myopic_throughput_greedy':
        greedy_metrics = greedy_bundle['myopic_throughput_greedy']
        greedy_rep = greedy_reps['myopic_throughput_greedy']
    elif selected_mode == 'channel_only_greedy':
        greedy_metrics = greedy_bundle['channel_only_greedy']
        greedy_rep = greedy_reps['channel_only_greedy']
    else:
        greedy_metrics = greedy_bundle['original']
        greedy_rep = greedy_reps['original']
    _report_log("Greedy variant sweeps completed.")
    embb_only_ceiling_metrics, embb_only_ceiling_rep = run_embb_only_ceiling_sweep(loads, episodes_per_load)
    _report_log("eMBB-only ceiling sweep completed.")
    throughput_oracle_metrics, throughput_oracle_rep = run_throughput_feasible_oracle_sweep(loads, episodes_per_load)
    throughput_oracle_metrics.update(_baseline_metadata('throughput_feasible_oracle'))
    throughput_oracle_metrics.update(_baseline_narrative('throughput_feasible_oracle'))
    throughput_oracle_metrics['method_name'] = _baseline_label('throughput_feasible_oracle')
    _report_log("Throughput-feasible Oracle sweep completed.")
    frontier_bundle = run_throughput_admission_frontier_bundle(loads)
    frontier_json_path = RESULTS_DIR / 'throughput_admission_frontier.json'
    frontier_payload = json.dumps(
        {str(k): v for k, v in frontier_bundle.items()},
        indent=2,
        default=_json_default,
    )
    try:
        frontier_json_path.write_text(frontier_payload, encoding='utf-8')
        _report_log(f"Frontier saved: {frontier_json_path}")
    except PermissionError:
        # Windows often denies overwriting if the json is open in another program.
        fallback = RESULTS_DIR / f"throughput_admission_frontier_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        fallback.write_text(frontier_payload, encoding='utf-8')
        _report_log(f"Frontier saved (fallback): {fallback}")
    if greedy_only:
        _report_log("Greedy-only mode enabled: skipping SR-MAPPO sweep.")
        rl_metrics = dict(greedy_metrics)
        rl_rep = dict(greedy_rep)
    else:
        rl_metrics, rl_rep = run_mappo_sweep(loads, episodes_per_load, checkpoint, base_cfg=report_cfg)
        _report_log("SR-MAPPO sweep completed.")
    rl_phase_a_runtime_enabled = _metric_scalar_any(
        rl_metrics.get('phase_a_embb_power_runtime_enabled', checkpoint_meta.get('phase_a_embb_power_runtime_enabled', False)),
        default=bool(checkpoint_meta.get('phase_a_embb_power_runtime_enabled', False)),
    )
    rl_phase_a_changed_count = _metric_scalar_mean(
        rl_metrics.get('phase_a_embb_power_changed_count', checkpoint_meta.get('phase_a_embb_power_changed_count', 0.0)),
        default=float(checkpoint_meta.get('phase_a_embb_power_changed_count', 0.0)),
    )
    rl_phase_a_changed_ratio = _metric_scalar_mean(
        rl_metrics.get('phase_a_embb_power_changed_ratio', checkpoint_meta.get('phase_a_embb_power_changed_ratio', 0.0)),
        default=float(checkpoint_meta.get('phase_a_embb_power_changed_ratio', 0.0)),
    )
    rl_phase_a_mean_raw_delta = _metric_scalar_mean(
        rl_metrics.get('phase_a_embb_power_mean_raw_delta', checkpoint_meta.get('phase_a_embb_power_mean_raw_delta', 0.0)),
        default=float(checkpoint_meta.get('phase_a_embb_power_mean_raw_delta', 0.0)),
    )
    rl_phase_a_mean_executed_delta = _metric_scalar_mean(
        rl_metrics.get('phase_a_embb_power_mean_executed_delta', checkpoint_meta.get('phase_a_embb_power_mean_executed_delta', 0.0)),
        default=float(checkpoint_meta.get('phase_a_embb_power_mean_executed_delta', 0.0)),
    )
    rl_phase_a_zero_keep = _metric_scalar_mean(rl_metrics.get('phase_a_embb_power_zeroed_keep_mode_ratio', 0.0))
    rl_phase_a_zero_no_candidate = _metric_scalar_mean(rl_metrics.get('phase_a_embb_power_zeroed_no_candidate_ratio', 0.0))
    rl_phase_a_zero_invalid_owner = _metric_scalar_mean(rl_metrics.get('phase_a_embb_power_zeroed_invalid_owner_ratio', 0.0))
    rl_phase_a_zero_inactive = _metric_scalar_mean(rl_metrics.get('phase_a_embb_power_zeroed_inactive_head_ratio', 0.0))
    _report_log(
        "Selected checkpoint Phase-A eMBB power: "
        f"runtime_enabled={rl_phase_a_runtime_enabled} | "
        f"changed_count={rl_phase_a_changed_count:.3f} | "
        f"changed_ratio={rl_phase_a_changed_ratio:.3f} | "
        f"mean_raw_delta={rl_phase_a_mean_raw_delta:.3f} | "
        f"mean_executed_delta={rl_phase_a_mean_executed_delta:.3f} | "
        f"zeroed(inactive/keep/no_candidate/invalid_owner)="
        f"{rl_phase_a_zero_inactive:.3f}/{rl_phase_a_zero_keep:.3f}/{rl_phase_a_zero_no_candidate:.3f}/{rl_phase_a_zero_invalid_owner:.3f}"
    )
    if rl_phase_a_runtime_enabled and rl_phase_a_changed_ratio <= 1.0e-9:
        _report_log("WARNING: selected checkpoint had Phase-A eMBB power runtime enabled, but no nonzero Phase-A eMBB power changes were executed.")
    if not rl_phase_a_runtime_enabled:
        _report_log("WARNING: selected checkpoint did not have Phase-A eMBB power runtime enabled.")
    phase_a_pos_clamp_ratio = _metric_scalar_mean(rl_metrics.get('phase_a_power_positive_clamped_to_zero_ratio', 0.0))
    if phase_a_pos_clamp_ratio >= 0.9:
        _report_log("Phase-A currently learns power reduction only; positive boost is blocked by clamp.")
    tradeoff_metrics = build_load_tradeoff_diagnostics(rl_metrics, greedy_metrics, report_cfg)

    dense_loads = [float(load) for load in (loads if fast else getattr(report_cfg.training, 'dense_eval_loads', loads))]
    eval_replicas = max(3, int(getattr(report_cfg.training, 'eval_replicas', 5))) if not fast else 3
    dense_bundle = run_dense_method_bundle(dense_loads, eval_replicas, checkpoint, checkpoint_reason, report_cfg)
    method_point_audit = extract_method_point_audit(dense_bundle)
    matched_proxy = nearest_replica_proxy(dense_bundle)
    low_damage_metrics = build_low_damage_diagnostics(dense_bundle['rl'], dense_bundle['matched_fixed_embb'])
    baseline_reference_metrics = build_baseline_reference_diagnostics(
        dense_bundle['rl'],
        dense_bundle['matched_fixed_embb'],
        dense_bundle['throughput_feasible_oracle'],
        dense_bundle['throughput_only_greedy'],
    )
    rl_metrics.update({
        'loadwise_selection_score': list(tradeoff_metrics['loadwise_selection_score']),
        'loadwise_admission_floor': list(tradeoff_metrics['selection_floor']),
        'loadwise_floor_pass': list(tradeoff_metrics['selection_floor_pass']),
        'loadwise_puncture_loss_gap': list(tradeoff_metrics['avg_puncture_loss_gap']),
        'loadwise_overlay_retention_gap': list(tradeoff_metrics['overlay_retention_gap']),
        'weighted_selection_score': float(tradeoff_metrics['weighted_selection_score']),
        'selection_floor_violation': list(tradeoff_metrics['selection_floor_violation']),
        'weighted_selection_contribution': list(tradeoff_metrics['score_contribution']),
    })
    try:
        load_arr = np.asarray(rl_metrics.get('loads', []), dtype=float)
        _report_plot_key_audit(
            "owner_main_series",
            [
                ("phase0_owner_change_ratio_vs_snapshot_raw", np.asarray(rl_metrics.get('phase0_owner_change_ratio_vs_snapshot_raw', []), dtype=float)),
                ("phase0_owner_change_ratio_vs_snapshot_executed", np.asarray(rl_metrics.get('phase0_owner_change_ratio_vs_snapshot_executed', []), dtype=float)),
                ("phase0_owner_changed_and_effective_ratio", np.asarray(rl_metrics.get('phase0_owner_changed_and_effective_ratio', []), dtype=float)),
                ("phase0_owner_changed_but_unserved_ratio", np.asarray(rl_metrics.get('phase0_owner_changed_but_unserved_ratio', []), dtype=float)),
                ("phase0_owner_same_as_snapshot_ratio", np.asarray(rl_metrics.get('phase0_owner_same_as_snapshot_ratio', []), dtype=float)),
                ("phase0_owner_restored_to_snapshot_ratio", np.asarray(rl_metrics.get('phase0_owner_restored_to_snapshot_ratio', []), dtype=float)),
            ],
        )
        m_srv = np.asarray(rl_metrics.get('embb_service_ratio', []), dtype=float)
        g_srv = np.asarray(greedy_metrics.get('embb_service_ratio', []), dtype=float)
        m_min = np.asarray(rl_metrics.get('embb_min_rate_satisfaction_ratio', []), dtype=float)
        g_min = np.asarray(greedy_metrics.get('embb_min_rate_satisfaction_ratio', []), dtype=float)
        m_pos = np.asarray(rl_metrics.get('embb_positive_rate_ratio', []), dtype=float)
        g_pos = np.asarray(greedy_metrics.get('embb_positive_rate_ratio', []), dtype=float)
        _report_plot_key_audit(
            "service_minrate_series",
            [
                ("sr_mappo.embb_service_ratio", m_srv),
                ("greedy.embb_service_ratio", g_srv),
                ("sr_mappo.embb_min_rate_satisfaction_ratio", m_min),
                ("greedy.embb_min_rate_satisfaction_ratio", g_min),
                ("sr_mappo.embb_positive_rate_ratio", m_pos),
                ("greedy.embb_positive_rate_ratio", g_pos),
            ],
        )
        if m_pos.size == m_srv.size and m_pos.size > 0 and np.allclose(m_pos, m_srv, atol=1.0e-12, rtol=0.0):
            _report_log("positive-rate ratio equals service ratio under current semantics (MAPPO).")
        if g_pos.size == g_srv.size and g_pos.size > 0 and np.allclose(g_pos, g_srv, atol=1.0e-12, rtol=0.0):
            _report_log("positive-rate ratio equals service ratio under current semantics (Greedy).")
    except Exception:
        pass
    timeslot_load = FAST_TIMESLOT_SERIES_LOAD if fast else TIMESLOT_SERIES_LOAD
    timeslot_slots = FAST_TIMESLOT_SERIES_SLOTS if fast else TIMESLOT_SERIES_SLOTS
    slot_greedy, slot_rl, slot_meta = run_timeslot_series(
        timeslot_load,
        timeslot_slots,
        checkpoint,
        cfg=report_cfg,
        frozen_greedy_payload=frozen_greedy_payload,
    )
    _report_log("Timeslot series completed.")
    uav_ue_distribution_bundle = _build_uav_ue_distribution_bundle(rl_rep, loads)
    uav_ue_distribution_paths = plot_uav_ue_distribution(rl_rep, loads)
    owner_change_detail_bundle = _build_owner_change_detail_bundle(rl_rep, loads)
    owner_change_detail_path = RESULTS_DIR / "owner_change_detail_debug.json"
    owner_change_detail_path.write_text(
        json.dumps(owner_change_detail_bundle, indent=2, default=_json_default),
        encoding="utf-8",
    )
    _report_log(f"Owner change detail saved: {owner_change_detail_path}")
    for load_key, payload in owner_change_detail_bundle.items():
        eff_gain = float(payload.get("phase0_owner_effective_rate_gain_vs_snapshot_mean", 0.0) or 0.0)
        harmful_ratio = float(payload.get("phase0_owner_change_harmful_ratio", 0.0) or 0.0)
        if eff_gain < 0.0:
            _report_log(f"[owner-warning] load={load_key}: owner limiter accepted harmful changes (effective_rate_gain_mean<0).")
        if harmful_ratio > 0.30:
            _report_log(f"[owner-warning] load={load_key}: harmful_change_ratio={harmful_ratio:.3f} > 0.30; tune top-K scoring before widening budget.")
    history = _load_history_from_final(report_cfg, checkpoint)
    plot_start = perf_counter()
    output_paths = [
        plot_full_mappo_activity(rl_metrics),
        plot_core_kpis(
            greedy_bundle['original'],
            greedy_bundle.get('original_greedy_normal_v1'),
            greedy_bundle.get('original_greedy_normal_v2'),
            greedy_bundle.get('myopic_throughput_greedy'),
            greedy_bundle['matched_fixed_embb'],
            greedy_bundle.get('throughput_feasible_oracle'),
            greedy_bundle.get('throughput_only_greedy'),
            greedy_bundle.get('channel_only_greedy'),
            rl_metrics,
            embb_only_ceiling_metrics,
            throughput_oracle_metrics,
            comparison_baseline_mode,
        ),
        plot_mode_diagnostics(
            greedy_bundle['original'],
            greedy_bundle.get('original_greedy_normal_v1'),
            greedy_bundle.get('original_greedy_normal_v2'),
            greedy_bundle.get('myopic_throughput_greedy'),
            greedy_bundle['matched_fixed_embb'],
            greedy_bundle.get('throughput_feasible_oracle'),
            greedy_bundle.get('throughput_only_greedy'),
            greedy_bundle.get('channel_only_greedy'),
            rl_metrics,
            comparison_baseline_mode,
        ),
        plot_fairness_uav(
            greedy_bundle['original'],
            greedy_bundle.get('original_greedy_normal_v1'),
            greedy_bundle.get('original_greedy_normal_v2'),
            greedy_bundle.get('myopic_throughput_greedy'),
            greedy_bundle['matched_fixed_embb'],
            greedy_bundle.get('throughput_feasible_oracle'),
            greedy_bundle.get('throughput_only_greedy'),
            greedy_bundle.get('channel_only_greedy'),
            rl_metrics,
            comparison_baseline_mode,
        ),
        plot_training_diagnostics(history, rl_metrics),
        plot_training_reward_curve(history),
        plot_mode_anchor_debug(history),
        plot_slot_dynamics(greedy_rep[representative_load], rl_rep[representative_load]),
        plot_single_slot_mode_maps(rl_rep[representative_load]),
        plot_timeslot_kpi_comparison(slot_greedy, slot_rl, slot_meta),
        plot_timeslot_power_comparison(slot_greedy, slot_rl, slot_meta),
        plot_timeslot_action_summary(slot_greedy, slot_rl, slot_meta),
        plot_upper_bounds_and_frontier(
            greedy_bundle['original'],
            greedy_bundle.get('original_greedy_normal_v1'),
            greedy_bundle.get('original_greedy_normal_v2'),
            greedy_bundle.get('myopic_throughput_greedy'),
            greedy_bundle['matched_fixed_embb'],
            greedy_bundle.get('throughput_feasible_oracle'),
            greedy_bundle.get('throughput_only_greedy'),
            greedy_bundle.get('channel_only_greedy'),
            rl_metrics,
            embb_only_ceiling_metrics,
            throughput_oracle_metrics,
            frontier_bundle,
            comparison_baseline_mode,
        ),
        plot_load_tradeoff_diagnostics(tradeoff_metrics),
        plot_low_damage_admission_diagnostics(low_damage_metrics, RESULTS_DIR, _style),
        plot_dense_uncertainty_bands(dense_bundle, RESULTS_DIR, _style, _style_power_axis, _top_axis_lambda),
        plot_normalized_gap_diagnostics(dense_bundle, RESULTS_DIR, _style, _top_axis_lambda),
        plot_matched_admission_diagnostics(matched_proxy, RESULTS_DIR, _style),
        plot_method_decomposition_dense(dense_bundle, RESULTS_DIR, _style, _top_axis_lambda),
        plot_marginal_degradation_slopes(dense_bundle, RESULTS_DIR, _style, _style_power_axis),
        plot_baseline_reference_story(baseline_reference_metrics),
        plot_mode_action_share_compare(
            rl_metrics,
            greedy_metrics,
            baseline_label=_baseline_label(comparison_baseline_mode),
        ),
        plot_mode_raw_vs_executed_compare(rl_metrics),
        plot_min_rate_satisfied_count_compare(
            rl_metrics,
            greedy_metrics,
            baseline_label=_baseline_label(comparison_baseline_mode),
        ),
        plot_admitted_urllc_packets_compare(
            rl_metrics,
            greedy_metrics,
            baseline_label=_baseline_label(comparison_baseline_mode),
        ),
    ]
    output_paths.extend(uav_ue_distribution_paths)
    _report_log("Figures saved.")
    _report_timing_log(f"plotting_and_save_figures sec={perf_counter() - plot_start:.3f}")
    metrics_start = perf_counter()
    selected_baseline_metadata = _baseline_metadata(comparison_baseline_mode)
    selected_baseline_narrative = _baseline_narrative(
        comparison_baseline_mode,
        greedy_requires_feasible_admission_only=_baseline_requires_feasible_only(greedy_metrics),
    )
    baseline_catalog_payload = {
        mode: {
            **_baseline_metadata(mode),
            **_baseline_narrative(mode),
        }
        for mode in (
            'matched_fixed_embb',
            'throughput_feasible_oracle',
            'throughput_only_greedy',
        )
    }
    metrics_path = save_metrics_json({
        'checkpoint': str(checkpoint),
        'checkpoint_selection_reason': checkpoint_reason,
        'checkpoint_meta': checkpoint_meta,
        'best_throughput_checkpoint_meta': best_throughput_meta,
        'best_balanced_checkpoint_meta': best_balanced_meta,
        'experiment_line': report_cfg.training.experiment_line,
        'urllc_poisson_rate': float(_base_profile()[1]),
        'fixed_urllc_poisson_rate': bool(_base_profile()[2]),
        'greedy_baseline_mode': greedy_baseline_mode,
        'greedy_baseline': greedy_baseline_mode,
        'comparison_baseline_key': comparison_baseline_mode,
        'comparison_baseline_label': _baseline_label(comparison_baseline_mode),
        'selected_baseline_key': comparison_baseline_mode,
        'selected_baseline_label': _baseline_label(comparison_baseline_mode),
        **selected_baseline_metadata,
        **selected_baseline_narrative,
        'phase': report_cfg.env.phase,
        'learn_embb_baseline': bool(report_cfg.env.learn_embb_baseline),
        'learn_phase0_embb_power': bool(getattr(report_cfg.env, 'learn_phase0_embb_power', True)),
        'allow_phase_a_embb_power_adjustment': bool(report_cfg.env.allow_phase_a_embb_power_adjustment),
        'enable_action_masking': bool(report_cfg.shield.enable_action_masking),
        'enable_feasibility_shield': bool(report_cfg.shield.enable_feasibility_shield),
        'apply_joint_reliability_rewrite': bool(report_cfg.shield.apply_joint_reliability_rewrite),
        'enable_greedy_fallback': bool(report_cfg.shield.enable_greedy_fallback),
        'loads': loads,
        'episodes_per_load': episodes_per_load,
        'same_scenario_pairing_enabled': True,
        'report_run_seed_base': int(_REPORT_RUN_SEED_BASE if _REPORT_RUN_SEED_BASE is not None else 0),
        'seeds_used_by_load': {
            str(float(load)): [int(_report_seed_base(i, report_cfg) + ep) for ep in range(int(episodes_per_load))]
            for i, load in enumerate(loads)
        },
        'single_episode_warning': (
            "single episode per load; curves are noisy and should not be used for final conclusion."
            if int(episodes_per_load) == 1 else ""
        ),
        'representative_load': representative_load,
        'timeslot_series_load': timeslot_load,
        'timeslot_series_slots': timeslot_slots,
        'baseline_catalog': baseline_catalog_payload,
        'selected_baseline': greedy_metrics,
        'greedy': greedy_metrics,
        'greedy_original': greedy_bundle['original'],
        'greedy_original_normal_v1': greedy_bundle.get('original_greedy_normal_v1'),
        'greedy_original_normal_v2': greedy_bundle.get('original_greedy_normal_v2'),
        'greedy_matched_fixed_embb': greedy_bundle['matched_fixed_embb'],
        'greedy_throughput_feasible': greedy_bundle.get('throughput_feasible_oracle'),
        'greedy_throughput_only': greedy_bundle.get('throughput_only_greedy'),
        'greedy_channel_only': greedy_bundle.get('channel_only_greedy'),
        'embb_only_ceiling': embb_only_ceiling_metrics,
        'throughput_feasible_oracle': throughput_oracle_metrics,
        'throughput_admission_frontier': {str(k): v for k, v in frontier_bundle.items()},
        'throughput_admission_frontier_json': str(frontier_json_path),
        'sr_mappo': rl_metrics,
        'loadwise_selection_score': tradeoff_metrics['loadwise_selection_score'],
        'loadwise_admission_floor': tradeoff_metrics['selection_floor'],
        'loadwise_floor_pass': tradeoff_metrics['selection_floor_pass'],
        'loadwise_puncture_loss_gap': tradeoff_metrics['avg_puncture_loss_gap'],
        'loadwise_overlay_retention_gap': tradeoff_metrics['overlay_retention_gap'],
        'weighted_selection_score': tradeoff_metrics['weighted_selection_score'],
        'load_tradeoff_diagnostics': tradeoff_metrics,
        'dense_loads': dense_loads,
        'eval_replicas': eval_replicas,
        'dense_method_bundle': dense_bundle,
        'method_point_audit': method_point_audit,
        'uav_ue_distribution': uav_ue_distribution_bundle,
        'uav_ue_distribution_plot_paths': [str(path) for path in uav_ue_distribution_paths],
        'owner_change_detail_debug': owner_change_detail_bundle,
        'owner_change_detail_debug_json': str(owner_change_detail_path),
        'matched_admission_proxy': matched_proxy,
        'low_damage_admission_diagnostics': low_damage_metrics,
        'baseline_reference_story': baseline_reference_metrics,
        'evaluation_protocol': {
            'checkpoint': str(checkpoint),
            'checkpoint_reason': checkpoint_reason,
            'actually_loaded_checkpoint_reason': checkpoint_reason,
            'best_throughput_iteration': best_throughput_meta.get('iteration'),
            'best_balanced_iteration': best_balanced_meta.get('iteration'),
            'best_throughput_eval_config': dict(best_throughput_meta.get('evaluation_config', {}) or {}),
            'best_balanced_eval_config': dict(best_balanced_meta.get('evaluation_config', {}) or {}),
            'primary_checkpoint_preference': str(getattr(report_cfg.training, 'primary_checkpoint_preference', 'best_throughput') or 'best_throughput'),
            'require_primary_checkpoint_match': bool(getattr(report_cfg.training, 'require_primary_checkpoint_match', False)),
            'primary_checkpoint_match_warning': primary_checkpoint_match_warning,
            'selection_rule': str(getattr(report_cfg.training, 'selection_mode', 'dual_metric')),
            'baseline_mode': comparison_baseline_mode,
            'comparison_baseline_key': comparison_baseline_mode,
            'comparison_baseline_label': _baseline_label(comparison_baseline_mode),
            'selected_baseline_key': comparison_baseline_mode,
            'selected_baseline_label': _baseline_label(comparison_baseline_mode),
            **selected_baseline_metadata,
            **selected_baseline_narrative,
            'coarse_sweep_loads': [float(load) for load in loads],
            'same_scenario_pairing_enabled': True,
            'report_run_seed_base': int(_REPORT_RUN_SEED_BASE if _REPORT_RUN_SEED_BASE is not None else 0),
            'seeds_used_by_load': {
                str(float(load)): [int(_report_seed_base(i, report_cfg) + ep) for ep in range(int(episodes_per_load))]
                for i, load in enumerate(loads)
            },
            'dense_sweep_loads': dense_loads,
            'num_eval_seeds': int(eval_replicas),
            'selection_admission_floor': float(getattr(report_cfg.training, 'selection_admission_floor', 0.0) or 0.0),
            'selection_admission_floor_ratio_to_baseline': float(getattr(report_cfg.training, 'selection_admission_floor_ratio_to_baseline', 0.0) or 0.0),
            'selection_admission_floor_by_load': dict(getattr(report_cfg.training, 'selection_admission_floor_by_load', {})),
            'checkpoint_eval_scope': str(getattr(report_cfg.training, 'checkpoint_eval_scope', 'representative_load') or 'representative_load'),
            'checkpoint_eval_loads': [float(load) for load in list(getattr(report_cfg.training, 'checkpoint_eval_loads', []) or [])],
            'checkpoint_eval_episodes_per_load': int(getattr(report_cfg.training, 'checkpoint_eval_episodes_per_load', 1) or 1),
            'selection_score_weights_by_load': dict(getattr(report_cfg.training, 'selection_score_weights_by_load', {})),
            'selection_throughput_ratio_floor_by_load': dict(getattr(report_cfg.training, 'selection_throughput_ratio_floor_by_load', {})),
            'selection_service_ratio_floor_by_load': dict(getattr(report_cfg.training, 'selection_service_ratio_floor_by_load', {})),
            'selection_minrate_ratio_floor_by_load': dict(getattr(report_cfg.training, 'selection_minrate_ratio_floor_by_load', {})),
            'selection_reliability_floor': float(getattr(report_cfg.training, 'selection_reliability_floor', 0.0) or 0.0),
            'selection_power_ratio_ceiling_by_load': dict(getattr(report_cfg.training, 'selection_power_ratio_ceiling_by_load', {})),
            'selection_puncture_ratio_floor_by_load': dict(getattr(report_cfg.training, 'selection_puncture_ratio_floor_by_load', {})),
            'selection_overlay_ratio_ceiling_by_load': dict(getattr(report_cfg.training, 'selection_overlay_ratio_ceiling_by_load', {})),
            'use_teacher_distillation': bool(getattr(report_cfg.training, 'use_teacher_distillation', False)),
            'teacher_policy': str(getattr(report_cfg.training, 'teacher_policy', 'channel_only_greedy') or 'channel_only_greedy'),
            'teacher_distill_coef_start': float(getattr(report_cfg.training, 'teacher_distill_coef_start', 0.0) or 0.0),
            'teacher_distill_coef_end': float(getattr(report_cfg.training, 'teacher_distill_coef_end', 0.0) or 0.0),
            'teacher_distill_end_frac': float(getattr(report_cfg.training, 'teacher_distill_end_frac', 0.0) or 0.0),
            'greedy_reference_bc_enabled': bool(getattr(report_cfg.training, 'use_greedy_reference_bc', False)),
            'greedy_bc_coef_start': float(getattr(report_cfg.training, 'greedy_bc_coef_start', 0.0) or 0.0),
            'greedy_bc_coef_end': float(getattr(report_cfg.training, 'greedy_bc_coef_end', 0.0) or 0.0),
            'greedy_bc_end_frac': float(getattr(report_cfg.training, 'greedy_bc_end_frac', 0.0) or 0.0),
            'greedy_bc_warmup_iters': int(getattr(report_cfg.training, 'greedy_bc_warmup_iters', 0) or 0),
            'matched_admission_proxy_used': True,
            'phase': report_cfg.env.phase,
            'learn_embb_baseline': bool(report_cfg.env.learn_embb_baseline),
            'learn_phase0_embb_power': bool(getattr(report_cfg.env, 'learn_phase0_embb_power', True)),
            'allow_phase_a_embb_power_adjustment': bool(report_cfg.env.allow_phase_a_embb_power_adjustment),
            'enable_action_masking': bool(report_cfg.shield.enable_action_masking),
            'enable_feasibility_shield': bool(report_cfg.shield.enable_feasibility_shield),
            'apply_joint_reliability_rewrite': bool(report_cfg.shield.apply_joint_reliability_rewrite),
            'enable_greedy_fallback': bool(report_cfg.shield.enable_greedy_fallback),
        },
        'frozen_greedy_metrics_path': str(getattr(report_cfg.training, 'frozen_greedy_metrics_path', "") or ""),
        'history_length': len(history),
        'actually_loaded_checkpoint_reason': checkpoint_reason,
        'primary_checkpoint_preference': str(getattr(report_cfg.training, 'primary_checkpoint_preference', 'best_throughput') or 'best_throughput'),
        'checkpoint_override_path': str(checkpoint_path or ""),
        'checkpoint_override_kind': str(checkpoint_kind or ""),
        'require_primary_checkpoint_match': bool(getattr(report_cfg.training, 'require_primary_checkpoint_match', False)),
        'primary_checkpoint_match_warning': primary_checkpoint_match_warning,
    })
    _report_log(f"Metrics saved: {metrics_path}")
    _report_timing_log(f"save_metrics_json sec={perf_counter() - metrics_start:.3f}")
    history_entry = {
        "timestamp": datetime.now().isoformat(),
        "experiment_line": str(report_cfg.training.experiment_line),
        "checkpoint": str(checkpoint),
        "checkpoint_selection_reason": str(checkpoint_reason),
        "greedy_only": bool(greedy_only),
        "loads": [float(load) for load in loads],
        "episodes_per_load": int(episodes_per_load),
        "greedy_baseline_mode": str(greedy_baseline_mode),
        "comparison_baseline_key": str(comparison_baseline_mode),
        "metrics_path": str(metrics_path),
        "core_series": {
            "greedy_embb_rate_mbps": [float(x) / 1.0e6 for x in list(greedy_metrics.get("embb_rate", []))],
            "greedy_urllc_admission": [float(x) for x in list(greedy_metrics.get("urllc_admission", []))],
            "greedy_embb_service_ratio": [float(x) for x in list(greedy_metrics.get("embb_service_ratio", []))],
            "greedy_urllc_tp_mbps": [float(x) for x in list(greedy_metrics.get("urllc_throughput_mbps_slot_est", []))],
            "mappo_embb_rate_mbps": [float(x) / 1.0e6 for x in list(rl_metrics.get("embb_rate", []))] if isinstance(rl_metrics, dict) else [],
            "mappo_urllc_admission": [float(x) for x in list(rl_metrics.get("urllc_admission", []))] if isinstance(rl_metrics, dict) else [],
            "mappo_embb_service_ratio": [float(x) for x in list(rl_metrics.get("embb_service_ratio", []))] if isinstance(rl_metrics, dict) else [],
            "mappo_urllc_tp_mbps": [float(x) for x in list(rl_metrics.get("urllc_throughput_mbps_slot_est", []))] if isinstance(rl_metrics, dict) else [],
        },
    }
    history_path = append_experiment_history_entry(history_entry)
    _report_log(f"Experiment history appended: {history_path}")
    manifest_path = _write_report_manifest({
        'generated_at': datetime.now().isoformat(),
        'experiment_line': report_cfg.training.experiment_line,
        'checkpoint': str(checkpoint),
        'checkpoint_selection_reason': checkpoint_reason,
        'actually_loaded_checkpoint_reason': checkpoint_reason,
        'primary_checkpoint_preference': str(getattr(report_cfg.training, 'primary_checkpoint_preference', 'best_throughput') or 'best_throughput'),
        'checkpoint_override_path': str(checkpoint_path or ""),
        'checkpoint_override_kind': str(checkpoint_kind or ""),
        'require_primary_checkpoint_match': bool(getattr(report_cfg.training, 'require_primary_checkpoint_match', False)),
        'primary_checkpoint_match_warning': primary_checkpoint_match_warning,
        'greedy_baseline_mode': greedy_baseline_mode,
        'comparison_baseline_key': comparison_baseline_mode,
        'comparison_baseline_label': _baseline_label(comparison_baseline_mode),
        'selected_baseline_key': comparison_baseline_mode,
        'selected_baseline_label': _baseline_label(comparison_baseline_mode),
        **selected_baseline_metadata,
        'loads': [float(load) for load in loads],
        'episodes_per_load': int(episodes_per_load),
        'output_paths': [str(path) for path in output_paths],
        'metrics_path': str(metrics_path),
        'frontier_json_path': str(frontier_json_path),
        'owner_change_detail_debug_json': str(owner_change_detail_path),
        'experiment_history_json': str(history_path),
        'stale_top_level_artifacts_removed': sorted(stale_removed),
    })
    _report_log(f"Manifest saved: {manifest_path}")
    _report_timing_log(f"generate_report total_sec={perf_counter() - report_start:.3f}")
    return {
        'checkpoint': str(checkpoint),
        'checkpoint_selection_reason': checkpoint_reason,
        'actually_loaded_checkpoint_reason': checkpoint_reason,
        'primary_checkpoint_match_warning': primary_checkpoint_match_warning,
        'checkpoint_override_path': str(checkpoint_path or ""),
        'checkpoint_override_kind': str(checkpoint_kind or ""),
        'checkpoint_meta': checkpoint_meta,
        'experiment_line': report_cfg.training.experiment_line,
        'greedy_baseline_mode': greedy_baseline_mode,
        'comparison_baseline_key': comparison_baseline_mode,
        'comparison_baseline_label': _baseline_label(comparison_baseline_mode),
        'output_paths': [str(path) for path in output_paths],
        'metrics_path': str(metrics_path),
        'frontier_json_path': str(frontier_json_path),
        'manifest_path': str(manifest_path),
        'experiment_history_path': str(history_path),
    }


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Generate SR-MAPPO report figures.")
    parser.add_argument("--fast", action="store_true", help="Run a fast, low-sample report for quick checks.")
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        choices=EXPERIMENT_CHOICES,
        help="Experiment preset.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="Override report checkpoint with an explicit path.",
    )
    parser.add_argument(
        "--checkpoint-kind",
        type=str,
        default=None,
        choices=["best_throughput", "best_balanced", "best_v5_balanced_intercell_admission", "best_v6_balanced_puncture_accounting", "latest", "final", "best"],
        help="Override report checkpoint selection with a named checkpoint kind.",
    )
    parser.add_argument(
        "--greedy-only",
        action="store_true",
        help="Skip SR-MAPPO sweep and generate report using greedy baseline only.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Optional output directory for this report run.",
    )
    args = parser.parse_args()

    output_dir = args.out_dir
    if not output_dir:
        # Keep compatibility with orchestrators that set SR_MAPPO_RESULTS_DIR_OVERRIDE
        # (e.g., greedy share grid). Only auto-create a run folder for direct CLI usage.
        if not os.environ.get("SR_MAPPO_RESULTS_DIR_OVERRIDE", "").strip():
            exp_tag = str(args.experiment or "default").replace(":", "_").replace("/", "_")
            ts = datetime.now(ZoneInfo("Asia/Taipei")).strftime("%Y%m%d_%H%M%S")
            output_dir = str((RESULTS_DIR / f"report_{exp_tag}_{ts}").resolve())

    result = generate_report(
        fast=bool(args.fast),
        experiment_line=args.experiment,
        checkpoint_path=args.checkpoint_path,
        checkpoint_kind=args.checkpoint_kind,
        greedy_only=bool(args.greedy_only),
        output_dir=output_dir,
    )
    _report_log("Report generated with checkpoint:")
    _report_log(result['checkpoint'])
    _report_log("Artifacts:")
    for path in result['output_paths']:
        _report_log(path)
    _report_log(result['metrics_path'])
    _report_log(result['frontier_json_path'])
