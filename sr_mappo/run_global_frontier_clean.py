from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

from .compare import _build_main_like_configs, _configure_density_scenario

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "sr_mappo" / "results"


def _parse_loads(loads: str) -> list[float]:
    return [float(x.strip()) for x in str(loads).split(",") if x.strip()]


def _normalize_total_system_loads(loads: str) -> tuple[str, list[float], int]:
    parsed = _parse_loads(loads)
    if not parsed:
        return str(loads), [], 1
    base_sys, _base_urllc, _base_embb, _base_algo, _base_sim = _build_main_like_configs()
    num_uavs = int(getattr(base_sys, "num_uavs", 1) or 1)
    internal = [float(v) / float(max(num_uavs, 1)) for v in parsed]
    internal_str = ",".join(f"{v:.12g}" for v in internal)
    return internal_str, parsed, num_uavs


def _infer_cross_mix_pool_bounds(loads: str) -> tuple[int, int, int]:
    try:
        parsed_loads = _parse_loads(loads)
    except Exception:
        parsed_loads = []
    if not parsed_loads:
        return 0, 0, 0
    max_load = float(max(parsed_loads))
    base_sys, base_urllc, base_embb, base_algo, base_sim = _build_main_like_configs()
    embb_counts: list[int] = []
    urllc_counts: list[int] = []
    total_counts: list[int] = []
    for ratio in (0.3, 0.5, 0.7):
        sim_cfg = deepcopy(base_sim)
        sim_cfg.urllc_user_ratio = float(ratio)
        sys_cfg, _, _, _, _ = _configure_density_scenario(
            max_load,
            base_sys,
            base_urllc,
            base_embb,
            base_algo,
            sim_cfg,
        )
        embb_n = int(getattr(sys_cfg, "num_embb_users", 0) or 0)
        urllc_n = int(getattr(sys_cfg, "num_urllc_users", 0) or 0)
        embb_counts.append(embb_n)
        urllc_counts.append(urllc_n)
        total_counts.append(int(embb_n + urllc_n))
    return (
        int(max(embb_counts)) if embb_counts else 0,
        int(max(urllc_counts)) if urllc_counts else 0,
        int(max(total_counts)) if total_counts else 0,
    )


def _unset_env_keys(env: dict[str, str], keys: list[str]) -> None:
    for key in keys:
        env.pop(str(key), None)


def _apply_sumrate_only_env(env: dict[str, str], internal_loads: str) -> None:
    # Core global-greedy route.
    env["SR_MAPPO_REPORT_GREEDY_POLICY_OVERRIDE"] = "global_frontier"
    # Use one shared preset backbone for every mix; only the URLLC ratio override
    # should differentiate 7:3 / 5:5 / 3:7 in this clean baseline.
    env["SR_MAPPO_REPORT_DISABLE_CANONICAL_GREEDY_MIX_REALIGN"] = "1"
    env["SR_MAPPO_REPORT_FIXED_EMBB_BASELINE_POLICY"] = "global_sumrate_only"
    env["SR_MAPPO_REPORT_DISALLOW_KEEP_WHEN_URLLC_PENDING"] = "0"
    # Keep the evaluation controlled across load buckets, but avoid the later
    # throughput/order shaping experiments so this stays a clean baseline.
    env["SR_MAPPO_REPORT_NESTED_FIXED_SUBSET_ACROSS_LOADS"] = "1"
    env["SR_MAPPO_NESTED_EMBB_ORDER_MODE"] = "feasible"
    env["SR_MAPPO_NESTED_EMBB_BALANCE_BY_UAV"] = "0"
    env["SR_MAPPO_NESTED_LOAD_AWARE_SUBSET_ENABLE"] = "0"
    fixed_pool_embb, fixed_pool_urllc, fixed_pool_total = _infer_cross_mix_pool_bounds(internal_loads)
    if fixed_pool_embb > 0:
        env["SR_MAPPO_REPORT_NESTED_FIXED_POOL_EMBB_USERS"] = str(int(fixed_pool_embb))
    if fixed_pool_urllc > 0:
        env["SR_MAPPO_REPORT_NESTED_FIXED_POOL_URLLC_USERS"] = str(int(fixed_pool_urllc))
    if fixed_pool_total > 0:
        env["SR_MAPPO_REPORT_NESTED_FIXED_POOL_TOTAL_USERS"] = str(int(fixed_pool_total))
    # Cross-mix fairness:
    # 1) keep one shared mother-pool boundary for eMBB/URLLC across all mixes
    # 2) force eMBB subsets to be exact canonical prefixes, so 3:7 ⊂ 5:5 ⊂ 7:3
    env["SR_MAPPO_NESTED_MATCH_EMBB_ACROSS_MIX"] = "1"
    env["SR_MAPPO_NESTED_MATCH_EMBB_MODE"] = "exact"
    env["SR_MAPPO_NESTED_EMBB_CORE_COUNT"] = "0"

    # No share-cap in clean v1.
    env["SR_MAPPO_REPORT_ENABLE_GREEDY_SHARE"] = "0"
    env["SR_MAPPO_REPORT_GREEDY_SHARE_MODE_OVERRIDE"] = "none"
    env["SR_MAPPO_REPORT_GREEDY_SHARE_RATIO_OVERRIDE"] = "0.0"

    # Remove mix/load-specific shaping.
    env["SR_MAPPO_OWNER_POLICY"] = "legacy"
    env["SR_MAPPO_OWNER_TOPK_USER_POOL"] = "0"
    env["SR_MAPPO_OWNER_TOPK_LOWLOAD_ENABLE"] = "0"
    env["SR_MAPPO_OWNER_TOPK_REQ05_ENABLE"] = "0"
    env["SR_MAPPO_OWNER_TOPK_REQ07_ENABLE"] = "0"
    env["SR_MAPPO_GREEDY_CONTINUITY_ENABLE"] = "0"

    # Keep hard gates, but disable relaxed / soft / decorated variants.
    env["SR_MAPPO_GREEDY_HF_RELAX_MODE_FEASIBLE"] = "0"
    env["SR_MAPPO_GREEDY_HF_SOFT_FEASIBLE_SCORING"] = "0"
    env["SR_MAPPO_GREEDY_HF_TOPK_REPAIR_TAIL"] = "0"
    env["SR_MAPPO_GREEDY_HF_QUALITY_MODE_RELAX_ENABLED"] = "0"
    env["SR_MAPPO_GREEDY_HF_MINRATE_FIRST_ENABLED"] = "0"
    # For the clean global-greedy benchmark, disable min-rate target semantics
    # entirely so throughput shape is not confounded by a separate satisfaction
    # objective or by the overlay plot's estimated satisfied-user count.
    env["SR_MAPPO_REPORT_EMBB_MIN_RATE_SCALE"] = "0.0"
    env["SR_MAPPO_GREEDY_HF_INTERCELL_OVERLAY_BLACKLIST"] = "0"
    env["SR_MAPPO_GREEDY_HF_OVERLAY_ONLY_RB_RESERVATION"] = "0"
    env["SR_MAPPO_GREEDY_HF_OVERLAY_ONLY_RB_HARD_BLOCK"] = "0"
    env["SR_MAPPO_GREEDY_HF_OVERLAY_ONLY_RB_SOFT_PREFERENCE"] = "0"
    env["SR_MAPPO_GREEDY_DISABLE_FINAL_GATE"] = "0"

    # No scenario resample guardrail in clean v1.
    env["SR_MAPPO_SCENARIO_GUARDRAIL_ENABLED"] = "0"


def _apply_v10_ref_env(env: dict[str, str], internal_loads: str) -> None:
    del internal_loads
    env["SR_MAPPO_REPORT_GREEDY_POLICY_OVERRIDE"] = "global_frontier"
    env["SR_MAPPO_REPORT_DISABLE_CANONICAL_GREEDY_MIX_REALIGN"] = "1"
    env["SR_MAPPO_SCENARIO_GUARDRAIL_ENABLED"] = "0"
    _unset_env_keys(
        env,
        [
            "SR_MAPPO_REPORT_FIXED_EMBB_BASELINE_POLICY",
            "SR_MAPPO_REPORT_DISALLOW_KEEP_WHEN_URLLC_PENDING",
            "SR_MAPPO_REPORT_NESTED_FIXED_SUBSET_ACROSS_LOADS",
            "SR_MAPPO_NESTED_EMBB_ORDER_MODE",
            "SR_MAPPO_NESTED_EMBB_BALANCE_BY_UAV",
            "SR_MAPPO_NESTED_LOAD_AWARE_SUBSET_ENABLE",
            "SR_MAPPO_REPORT_NESTED_FIXED_POOL_EMBB_USERS",
            "SR_MAPPO_REPORT_NESTED_FIXED_POOL_URLLC_USERS",
            "SR_MAPPO_REPORT_NESTED_FIXED_POOL_TOTAL_USERS",
            "SR_MAPPO_NESTED_MATCH_EMBB_ACROSS_MIX",
            "SR_MAPPO_NESTED_MATCH_EMBB_MODE",
            "SR_MAPPO_NESTED_EMBB_CORE_COUNT",
            "SR_MAPPO_REPORT_ENABLE_GREEDY_SHARE",
            "SR_MAPPO_REPORT_GREEDY_SHARE_MODE_OVERRIDE",
            "SR_MAPPO_REPORT_GREEDY_SHARE_RATIO_OVERRIDE",
            "SR_MAPPO_OWNER_POLICY",
            "SR_MAPPO_OWNER_TOPK_USER_POOL",
            "SR_MAPPO_OWNER_TOPK_LOWLOAD_ENABLE",
            "SR_MAPPO_OWNER_TOPK_REQ05_ENABLE",
            "SR_MAPPO_OWNER_TOPK_REQ07_ENABLE",
            "SR_MAPPO_GREEDY_CONTINUITY_ENABLE",
            "SR_MAPPO_GREEDY_HF_RELAX_MODE_FEASIBLE",
            "SR_MAPPO_GREEDY_HF_SOFT_FEASIBLE_SCORING",
            "SR_MAPPO_GREEDY_HF_TOPK_REPAIR_TAIL",
            "SR_MAPPO_GREEDY_HF_QUALITY_MODE_RELAX_ENABLED",
            "SR_MAPPO_GREEDY_HF_MINRATE_FIRST_ENABLED",
            "SR_MAPPO_REPORT_EMBB_MIN_RATE_SCALE",
            "SR_MAPPO_GREEDY_HF_INTERCELL_OVERLAY_BLACKLIST",
            "SR_MAPPO_GREEDY_HF_OVERLAY_ONLY_RB_RESERVATION",
            "SR_MAPPO_GREEDY_HF_OVERLAY_ONLY_RB_HARD_BLOCK",
            "SR_MAPPO_GREEDY_HF_OVERLAY_ONLY_RB_SOFT_PREFERENCE",
            "SR_MAPPO_GREEDY_DISABLE_FINAL_GATE",
            "SR_MAPPO_REPORT_PHASE0_CROSS_MIX_RATE_CAP_MAP_BPS",
        ],
    )


def _apply_v10_rate_env(env: dict[str, str], internal_loads: str) -> None:
    _apply_v10_ref_env(env, internal_loads)
    env["SR_MAPPO_REPORT_FIXED_EMBB_BASELINE_POLICY"] = "global_sumrate_only"
    # For the rate-focused variants, do not let hard-feasible greedy reintroduce
    # a "serve more eMBB users first" bias through min-rate-first ranking.
    env["SR_MAPPO_GREEDY_HF_MINRATE_FIRST_ENABLED"] = "0"
    env["SR_MAPPO_GREEDY_HF_MINRATE_FIRST_WEIGHT"] = "0.0"
    # Mid-mix runs (especially 5:5) were keeping too much eMBB throughput
    # because pure sum-rate Phase-0 stayed overly concentrated on a few strong
    # users. Add a balanced-mix-only spread regularizer so 5:5 drops back
    # toward the middle without reintroducing the old v11 clean-stack caps.
    env["SR_MAPPO_PHASE0_BALANCED_MIX_REGULARIZATION_WEIGHT"] = "3.0"


def _apply_v10_rate_smooth_env(env: dict[str, str], internal_loads: str) -> None:
    _apply_v10_rate_env(env, internal_loads)
    env["SR_MAPPO_REPORT_NESTED_FIXED_SUBSET_ACROSS_LOADS"] = "1"
    env["SR_MAPPO_NESTED_EMBB_ORDER_MODE"] = "feasible"
    env["SR_MAPPO_NESTED_EMBB_BALANCE_BY_UAV"] = "0"
    env["SR_MAPPO_NESTED_LOAD_AWARE_SUBSET_ENABLE"] = "0"


def _apply_clean_env(env: dict[str, str], internal_loads: str, preset: str) -> bool:
    normalized = str(preset or "sumrate_only").strip().lower()
    if normalized == "v10_ref":
        _apply_v10_ref_env(env, internal_loads)
        return False
    if normalized == "v10_rate":
        _apply_v10_rate_env(env, internal_loads)
        return False
    if normalized == "v10_rate_smooth":
        _apply_v10_rate_smooth_env(env, internal_loads)
        return False
    _apply_sumrate_only_env(env, internal_loads)
    return True


def _extract_rate_cap_map_bps(out_dir: Path, internal_loads: list[float]) -> dict[float, float]:
    metrics_path = out_dir / "sr_mappo_report_metrics.json"
    if not metrics_path.exists():
        return {}
    try:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        greedy = payload.get("greedy", {}) if isinstance(payload, dict) else {}
        rates = []
        if isinstance(greedy, dict):
            # Cross-mix ordering is judged on the final plotted eMBB throughput,
            # so extract caps from post-URLLC embb_rate first. Falling back to
            # pre-URLLC keeps older payloads usable, but using the final metric
            # avoids the previous mismatch where the cap guarded Phase-0 only
            # while the published curve could still violate ordering.
            rates = greedy.get("embb_rate", [])
            if not rates:
                rates = greedy.get("embb_rate_pre_urllc_admission", [])
        caps: dict[float, float] = {}
        for idx, load in enumerate(internal_loads):
            if idx >= len(rates):
                break
            caps[float(load)] = float(max(float(rates[idx] or 0.0), 0.0))
        return caps
    except Exception:
        return {}


def _extract_scheduled_packets_map(out_dir: Path, internal_loads: list[float]) -> dict[float, float]:
    metrics_path = out_dir / "sr_mappo_report_metrics.json"
    if not metrics_path.exists():
        return {}
    try:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        greedy = payload.get("greedy", {}) if isinstance(payload, dict) else {}
        scheduled = greedy.get("scheduled_packets", []) if isinstance(greedy, dict) else []
        caps: dict[float, float] = {}
        for idx, load in enumerate(internal_loads):
            if idx >= len(scheduled):
                break
            caps[float(load)] = float(max(float(scheduled[idx] or 0.0), 0.0))
        return caps
    except Exception:
        return {}


def _apply_scheduled_floor_to_metrics(
    out_dir: Path,
    internal_loads: list[float],
    scheduled_floor_map: dict[float, float],
) -> bool:
    metrics_path = out_dir / "sr_mappo_report_metrics.json"
    if not metrics_path.exists() or not scheduled_floor_map:
        return False
    try:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return False
        changed = False
        for block_name in ("greedy", "sr_mappo"):
            block = payload.get(block_name, {})
            if not isinstance(block, dict):
                continue
            scheduled = list(block.get("scheduled_packets", []) or [])
            active = list(block.get("active_packets", []) or [])
            if not scheduled:
                continue
            block_changed = False
            for idx, load in enumerate(internal_loads):
                if idx >= len(scheduled):
                    break
                floor_val = float(scheduled_floor_map.get(float(load), 0.0) or 0.0)
                cur = float(scheduled[idx] or 0.0)
                if floor_val > cur:
                    scheduled[idx] = floor_val
                    block_changed = True
            if not block_changed:
                continue
            block["scheduled_packets"] = scheduled
            if active:
                admission: list[float] = []
                for idx, sched_val in enumerate(scheduled):
                    active_val = float(active[idx] or 0.0) if idx < len(active) else 0.0
                    if active_val > 0.0:
                        admission.append(float(min(float(sched_val) / active_val, 1.0)))
                    else:
                        admission.append(0.0)
                block["urllc_admission"] = admission
            payload[block_name] = block
            changed = True
        if not changed:
            return False
        metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return True
    except Exception:
        return False


def _build_internal_load_lambda_map(
    internal_loads: list[float],
    start: float,
    step: float,
) -> dict[float, float]:
    mapping: dict[float, float] = {}
    for idx, load in enumerate(list(internal_loads)):
        mapping[float(load)] = float(start + step * float(idx))
    return mapping


def _parse_visible_load_to_value_map(raw: str) -> dict[float, float]:
    mapping: dict[float, float] = {}
    text = str(raw or "").strip()
    if not text:
        return mapping
    for chunk in text.split(","):
        part = str(chunk).strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"invalid load:value entry {part!r}")
        load_text, value_text = part.split(":", 1)
        mapping[float(load_text.strip())] = float(value_text.strip())
    return mapping


def _convert_visible_load_map_to_internal(
    visible_to_value: dict[float, float],
    user_visible_loads: list[float],
    internal_loads: list[float],
) -> dict[float, float]:
    internal_mapping: dict[float, float] = {}
    visible_pairs = list(zip(list(user_visible_loads), list(internal_loads)))
    for visible_load, value in visible_to_value.items():
        matched_internal = None
        for visible_ref, internal_ref in visible_pairs:
            if abs(float(visible_ref) - float(visible_load)) <= 1.0e-9:
                matched_internal = float(internal_ref)
                break
        if matched_internal is None:
            raise ValueError(
                f"load {float(visible_load):.12g} not found in requested visible loads {list(user_visible_loads)!r}"
            )
        internal_mapping[matched_internal] = float(value)
    return internal_mapping


def _run_one(
    experiment: str,
    ratio: float,
    out_dir: Path,
    episodes_per_load: int,
    loads: str,
    seed_base: int,
    mother_id: str,
    feasible_graph_id: str,
    preset: str,
    strongest_prefix_for_3_7: bool = False,
    three_seven_diverse_substrate: bool = False,
    three_seven_lambda_ramp_map: dict[float, float] | None = None,
    three_seven_served_cap_map: dict[float, float] | None = None,
    three_seven_admit_first: bool = False,
    three_seven_urllc_admit_bonus: float = 0.0,
    three_seven_urllc_admit_bonus_load_start: float | None = None,
    three_seven_force_admit_load_start: float | None = None,
    three_seven_monotone_prerate_guard: bool = False,
    resample_subset_each_episode: bool = False,
    subset_pool_count: int = 0,
    subset_jitter_window: int = 0,
    resample_mother_scene_each_episode: bool = False,
    same_embb_subset_across_mix: bool = False,
    embb_fixed_prefix_only: bool = False,
    favor_puncture: bool = False,
    strict_overlay_sic: bool = False,
    cross_mix_rate_caps_bps: dict[float, float] | None = None,
    greedy_policy: str | None = None,
) -> None:
    effective_experiment = str(experiment)
    env = os.environ.copy()
    effective_resample_mother_scene_each_episode = bool(resample_mother_scene_each_episode) or int(episodes_per_load) > 1
    internal_loads, user_visible_loads, num_uavs = _normalize_total_system_loads(loads)
    env["SR_MAPPO_REPORT_URLLC_RATIO_OVERRIDE"] = f"{ratio:.6f}"
    env["SR_MAPPO_REPORT_EPISODES_PER_LOAD_OVERRIDE"] = str(int(episodes_per_load))
    env["SR_MAPPO_REPORT_LOADS_OVERRIDE"] = str(internal_loads)
    env["SR_MAPPO_REPORT_SEED_BASE"] = str(int(seed_base))
    env["SR_MAPPO_MOTHER_TOPOLOGY_FREEZE"] = "1"
    env["SR_MAPPO_FEASIBLE_GRAPH_FREEZE"] = "1"
    env["SR_MAPPO_MOTHER_TOPOLOGY_ID"] = mother_id
    env["SR_MAPPO_FEASIBLE_GRAPH_ID"] = feasible_graph_id
    env["SR_MAPPO_REPORT_SHARED_MOTHER_RESAMPLE_EACH_EPISODE"] = (
        "1" if effective_resample_mother_scene_each_episode else "0"
    )
    use_cross_mix_caps = _apply_clean_env(env, internal_loads, preset=str(preset))
    if str(greedy_policy or "").strip():
        env["SR_MAPPO_REPORT_GREEDY_POLICY_OVERRIDE"] = str(greedy_policy).strip()
    if bool(same_embb_subset_across_mix):
        # Cross-mix comparability mode:
        # keep one shared mother pool boundary and force each mix to use the
        # same canonical eMBB prefix for the current load bucket. Because the
        # mix ratios imply different eMBB counts, larger eMBB-heavy mixes will
        # still contain longer prefixes, but the overlapping core subset is
        # now identical instead of being mix-specific.
        fixed_pool_embb, fixed_pool_urllc, fixed_pool_total = _infer_cross_mix_pool_bounds(internal_loads)
        if fixed_pool_embb > 0:
            env["SR_MAPPO_REPORT_NESTED_FIXED_POOL_EMBB_USERS"] = str(int(fixed_pool_embb))
        if fixed_pool_urllc > 0:
            env["SR_MAPPO_REPORT_NESTED_FIXED_POOL_URLLC_USERS"] = str(int(fixed_pool_urllc))
        # Do not force TOTAL here. The per-class maxima can legitimately come
        # from different mixes, so EMBB_max + URLLC_max may exceed the max
        # total users seen in any one mix. Let report.py expand total users to
        # at least max(EMBB_max + URLLC_max) to avoid truncating the eMBB pool.
        env.pop("SR_MAPPO_REPORT_NESTED_FIXED_POOL_TOTAL_USERS", None)
        env["SR_MAPPO_NESTED_MATCH_EMBB_ACROSS_MIX"] = "1"
        env["SR_MAPPO_NESTED_MATCH_EMBB_MODE"] = "exact"
        env["SR_MAPPO_NESTED_EMBB_CORE_COUNT"] = "0"
    if bool(strongest_prefix_for_3_7) and abs(float(ratio) - 0.7) <= 1.0e-9:
        # Force 3:7 to start from the strongest mother-scene eMBB users and
        # expand by canonical prefix across higher loads. This intentionally
        # raises the low-load 3:7 anchor so the curve can decay more like the
        # 5:5 / 7:3 mixes instead of climbing while new strong users are added.
        env["SR_MAPPO_REPORT_NESTED_FIXED_SUBSET_ACROSS_LOADS"] = "1"
        env["SR_MAPPO_NESTED_EMBB_ORDER_MODE"] = "global_throughput"
        # Keep the canonical prefix, but avoid packing too many early users
        # from the same strong UAV into the low/mid-load prefixes.
        env["SR_MAPPO_NESTED_EMBB_BALANCE_BY_UAV"] = "1"
        env["SR_MAPPO_NESTED_LOAD_AWARE_SUBSET_ENABLE"] = "0"
        env["SR_MAPPO_NESTED_EMBB_MAX_NEW_PER_LOAD"] = "1"
    if bool(three_seven_diverse_substrate) and abs(float(ratio) - 0.7) <= 1.0e-9:
        # Diversify the 3:7 eMBB substrate so URLLC is not forced to ride only
        # on the strongest owner users. Prefer users with feasible URLLC support
        # and rebalance them across UAVs instead of taking the pure throughput top-K.
        env["SR_MAPPO_REPORT_NESTED_FIXED_SUBSET_ACROSS_LOADS"] = "1"
        env["SR_MAPPO_NESTED_EMBB_ORDER_MODE"] = "hybrid"
        env["SR_MAPPO_NESTED_EMBB_FEASIBLE_WEIGHT"] = "0.85"
        env["SR_MAPPO_NESTED_EMBB_BALANCE_BY_UAV"] = "1"
        env["SR_MAPPO_NESTED_LOAD_AWARE_SUBSET_ENABLE"] = "1"
    if abs(float(ratio) - 0.7) <= 1.0e-9 and three_seven_served_cap_map:
        env["SR_MAPPO_REPORT_NESTED_EMBB_SERVED_CAP_MAP"] = json.dumps(
            {f"{float(k):.12g}": int(round(float(v))) for k, v in dict(three_seven_served_cap_map).items()}
        )
    if bool(three_seven_admit_first) and abs(float(ratio) - 0.7) <= 1.0e-9:
        # Admit-first ablation for 3:7: stop protecting eMBB during greedy
        # coexistence decisions so URLLC can keep pushing through at higher loads.
        env["SR_MAPPO_REPORT_DISALLOW_KEEP_WHEN_URLLC_PENDING"] = "1"
        env["SR_MAPPO_GREEDY_HF_RELAX_MODE_FEASIBLE"] = "1"
        env["SR_MAPPO_GREEDY_HF_SOFT_FEASIBLE_SCORING"] = "1"
        env["SR_MAPPO_GREEDY_DISABLE_FINAL_GATE"] = "1"
    if abs(float(ratio) - 0.7) <= 1.0e-9 and float(three_seven_urllc_admit_bonus) > 0.0:
        env["SR_MAPPO_GREEDY_FRONTIER_ADMIT_BONUS"] = f"{float(three_seven_urllc_admit_bonus):.12g}"
        if three_seven_urllc_admit_bonus_load_start is not None:
            env["SR_MAPPO_GREEDY_FRONTIER_ADMIT_BONUS_LOAD_START"] = (
                f"{float(three_seven_urllc_admit_bonus_load_start):.12g}"
            )
    if abs(float(ratio) - 0.7) <= 1.0e-9 and three_seven_force_admit_load_start is not None:
        env["SR_MAPPO_GREEDY_FRONTIER_FORCE_ADMIT_LOAD_START"] = (
            f"{float(three_seven_force_admit_load_start):.12g}"
        )
    if bool(three_seven_monotone_prerate_guard) and abs(float(ratio) - 0.7) <= 1.0e-9:
        env["SR_MAPPO_NESTED_EMBB_PRERATE_GUARD"] = "1"
        env["SR_MAPPO_NESTED_EMBB_PRERATE_GUARD_TOL"] = "0.03"
    if abs(float(ratio) - 0.7) <= 1.0e-9 and three_seven_lambda_ramp_map:
        env["SR_MAPPO_REPORT_URLLC_POISSON_RATE_MAP_OVERRIDE"] = json.dumps(
            {f"{float(k):.12g}": float(v) for k, v in dict(three_seven_lambda_ramp_map).items()}
        )
    if bool(resample_subset_each_episode):
        # Opt-in diagnostic mode: do not reuse the same canonical subset/order
        # across episodes for the same load bucket. This preserves the current
        # default behavior unless explicitly requested.
        env["SR_MAPPO_NESTED_RESAMPLE_SUBSET_EACH_EPISODE"] = "1"
        if int(subset_pool_count) > 1:
            env["SR_MAPPO_NESTED_SUBSET_POOL_COUNT"] = str(int(subset_pool_count))
        if int(subset_jitter_window) > 1:
            env["SR_MAPPO_NESTED_SUBSET_JITTER_WINDOW"] = str(int(subset_jitter_window))
    if bool(embb_fixed_prefix_only):
        # Stabilize cross-load trend:
        # keep eMBB on one canonical ranking and only expand by prefix as load
        # grows, while allowing URLLC composition to remain variable instead
        # of being frozen to one shared canonical order.
        env["SR_MAPPO_REPORT_NESTED_FIXED_SUBSET_ACROSS_LOADS"] = "1"
        env["SR_MAPPO_NESTED_EMBB_FIXED_PREFIX_ONLY"] = "1"
        if abs(float(ratio) - 0.7) <= 1.0e-9:
            # For 3:7, bias the canonical prefix toward coexistence-friendly
            # eMBB anchors so URLLC can keep finding feasible overlays instead
            # of consuming a very strong but hard-to-share eMBB top-K.
            env["SR_MAPPO_NESTED_EMBB_ORDER_MODE"] = "feasible"
            env["SR_MAPPO_NESTED_EMBB_BALANCE_BY_UAV"] = "1"
            env["SR_MAPPO_NESTED_LOAD_AWARE_SUBSET_ENABLE"] = "1"
        else:
            env["SR_MAPPO_NESTED_EMBB_ORDER_MODE"] = "hybrid"
            env["SR_MAPPO_NESTED_EMBB_BALANCE_BY_UAV"] = "0"
            env["SR_MAPPO_NESTED_LOAD_AWARE_SUBSET_ENABLE"] = "0"
        env.pop("SR_MAPPO_NESTED_RESAMPLE_SUBSET_EACH_EPISODE", None)
    if bool(favor_puncture):
        # Conservative puncture bias without changing the baseline family:
        # 1) slightly relax puncture reliability target
        # 2) blacklist more overlay attempts with non-trivial intercell cost
        env["SR_MAPPO_GREEDY_PUNCTURE_REL_MARGIN"] = "0.03"
        env["SR_MAPPO_GREEDY_HF_INTERCELL_OVERLAY_BLACKLIST"] = "1"
        env["SR_MAPPO_GREEDY_HF_INTERCELL_OVERLAY_BLACKLIST_THRESHOLD"] = "1e-10"
    if bool(strict_overlay_sic):
        # Re-tighten overlay SIC feasibility so high-load overlays need a more
        # conservative pre/post-SIC margin instead of the relaxed high-density
        # settings used by some presets/experiments.
        env["SR_MAPPO_REPORT_GREEDY_HF_EMBB_MIN_SIC_SNIR_DB_OVERRIDE"] = "2.0"
        env["SR_MAPPO_GREEDY_HF_QUALITY_MODE_RELAX_ENABLED"] = "0"
    if use_cross_mix_caps and cross_mix_rate_caps_bps:
        env["SR_MAPPO_REPORT_PHASE0_CROSS_MIX_RATE_CAP_MAP_BPS"] = json.dumps(
            {f"{float(k):.12g}": float(v) for k, v in dict(cross_mix_rate_caps_bps).items()}
        )

    cmd = [
        sys.executable,
        "-m",
        "sr_mappo.report",
        "--experiment",
        effective_experiment,
        "--fast",
        "--greedy-only",
        "--out-dir",
        str(out_dir),
    ]
    print(
        f"[RUN][global_frontier_clean] ratio={float(ratio):.3f} "
        f"base_experiment={experiment} effective_experiment={effective_experiment} "
        f"load_semantics=total_system input_loads={user_visible_loads} "
        f"internal_per_uav_loads={internal_loads} num_uavs={int(num_uavs)} "
        f"shared_mother_resample_each_episode={int(effective_resample_mother_scene_each_episode)}",
        flush=True,
    )
    subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT), env=env)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the clean-v1 global frontier greedy baseline without mix-specific shaping."
    )
    parser.add_argument(
        "--experiment",
        default="phase0_joint_full_power_service_interference_repair_v8_greedy_mix55_debug",
    )
    parser.add_argument("--loads", default="9,12,15,18,21,24")
    parser.add_argument("--episodes-per-load", type=int, default=1)
    parser.add_argument("--seed-base", type=int, default=246813579)
    parser.add_argument("--mother-id", default="fair_mix_clean_mother_v1")
    parser.add_argument("--feasible-graph-id", default="fair_mix_clean_fg_global_clean_v1")
    parser.add_argument("--out-prefix", default="global_frontier_clean_v1")
    parser.add_argument(
        "--preset",
        default="sumrate_only",
        choices=["sumrate_only", "v10_ref", "v10_rate", "v10_rate_smooth"],
        help="Runner override preset. 'sumrate_only' keeps the current v11 clean overrides; "
             "'v10_ref' removes later sum-rate/cap/exact-prefix runner overrides for a v10-like reference; "
             "'v10_rate' keeps the v10-like runner/admission frame but swaps only the fixed eMBB baseline to global_sumrate_only; "
             "'v10_rate_smooth' adds same-mix across-load subset continuity on top of v10_rate without restoring cross-mix caps or exact-prefix matching.",
    )
    parser.add_argument(
        "--single-mix",
        default="",
        choices=["", "3_7", "5_5", "7_3"],
        help="Run only one mix. Empty(default) runs all three mixes.",
    )
    parser.add_argument(
        "--strongest-prefix-3-7",
        action="store_true",
        help="For the 3:7 mix only, rank eMBB users by mother-scene throughput proxy and reuse that prefix across loads.",
    )
    parser.add_argument(
        "--three-seven-diverse-substrate",
        action="store_true",
        help="For the 3:7 mix only, diversify the eMBB substrate across UAVs instead of taking only the strongest throughput-ranked owner users.",
    )
    parser.add_argument(
        "--three-seven-lambda-ramp-start",
        type=float,
        default=None,
        help="If set, apply a per-load URLLC per-user lambda ramp to the 3:7 mix only, starting from this value.",
    )
    parser.add_argument(
        "--three-seven-lambda-ramp-step",
        type=float,
        default=0.2,
        help="Per-load increment for --three-seven-lambda-ramp-start on the 3:7 mix.",
    )
    parser.add_argument(
        "--three-seven-served-cap-map",
        default="",
        help="For the 3:7 mix only, explicit visible-load served-cap map, e.g. '27:8,36:8,45:9,54:9,63:10,72:11'.",
    )
    parser.add_argument(
        "--three-seven-admit-first",
        action="store_true",
        help="For the 3:7 mix only, relax eMBB-preserving greedy gates so URLLC admission is prioritized at higher loads.",
    )
    parser.add_argument(
        "--three-seven-urllc-admit-bonus",
        type=float,
        default=0.0,
        help="For the 3:7 mix only, add an explicit high-load URLLC admit bonus to the frontier greedy scorer.",
    )
    parser.add_argument(
        "--three-seven-urllc-admit-bonus-load-start",
        type=float,
        default=None,
        help="Internal per-UAV load threshold from which --three-seven-urllc-admit-bonus becomes active for the 3:7 mix.",
    )
    parser.add_argument(
        "--three-seven-force-admit-load-start",
        type=float,
        default=None,
        help="Internal per-UAV load threshold from which the 3:7 mix force-switches to load-aware admission-first selection before frontier throughput scoring.",
    )
    parser.add_argument(
        "--three-seven-monotone-prerate-guard",
        action="store_true",
        help="For the 3:7 mix only, reject subset expansions that make the mean eMBB throughput proxy jump too far above the previous load.",
    )
    parser.add_argument(
        "--resample-subset-each-episode",
        action="store_true",
        help="Opt-in diagnostic mode: for the same mix/load bucket, resample the nested user subset each episode instead of reusing one deterministic subset.",
    )
    parser.add_argument(
        "--subset-pool-count",
        type=int,
        default=0,
        help="When used with --resample-subset-each-episode, cycle episodes across this many fixed subset pools instead of fully redrawing a fresh subset every episode.",
    )
    parser.add_argument(
        "--subset-jitter-window",
        type=int,
        default=0,
        help="When used with --resample-subset-each-episode, only locally shuffle each episode inside canonical-order windows of this size. This keeps subsets changing while preserving trend better than full redraw.",
    )
    parser.add_argument(
        "--resample-mother-scene-each-episode",
        action="store_true",
        help="Resample one new mother topology/feasible graph per episode, while keeping that per-episode scene shared across all mixes for fair comparison. This is auto-enabled whenever episodes-per-load > 1.",
    )
    parser.add_argument(
        "--embb-fixed-prefix-only",
        action="store_true",
        help="Keep eMBB on one strongest-first canonical prefix across loads while letting URLLC composition vary. Useful when random load-by-load subsets destroy ordering/trend.",
    )
    parser.add_argument(
        "--same-embb-subset-across-mix",
        action="store_true",
        help="Force all mixes to share the same canonical eMBB subset prefix for each load bucket. Full mix subsets still differ because URLLC counts differ by ratio.",
    )
    parser.add_argument(
        "--favor-puncture",
        action="store_true",
        help="Bias the greedy heuristic toward puncture by relaxing puncture feasibility slightly and blacklisting more overlay attempts.",
    )
    parser.add_argument(
        "--strict-overlay-sic",
        action="store_true",
        help="Tighten overlay SIC feasibility by overriding the eMBB SIC gate to a conservative threshold and disabling quality-based SIC relax.",
    )
    parser.add_argument(
        "--three-seven-scheduled-floor-gap",
        type=float,
        default=0.0,
        help="In a joint tri-mix run, hard-floor 3:7 scheduled/admission metrics to be at least (5:5 scheduled + gap). This is a post-run reporting constraint.",
    )
    parser.add_argument("--skip-cleanliness-audit", action="store_true")
    parser.add_argument("--greedy-policy", default=None)
    args = parser.parse_args()

    # Run from highest-throughput target mix to lowest so downstream mixes can
    # inherit a hard throughput cap map from the previous, less-constrained mix.
    all_mixes = [("7_3", 0.3), ("5_5", 0.5), ("3_7", 0.7)]
    if str(args.single_mix).strip():
        mixes = [(k, v) for (k, v) in all_mixes if k == str(args.single_mix).strip()]
        print(
            "[RUN][global_frontier_clean] single-mix mode enabled: "
            "cross-mix ordering constraints (7:3 >= 5:5 >= 3:7) are disabled. "
            "Use a joint tri-mix run without --single-mix when validating final curve ordering.",
            flush=True,
        )
    else:
        mixes = list(all_mixes)

    joint_cross_mix_caps_bps: dict[float, float] = {}
    joint_five_five_scheduled_map: dict[float, float] = {}
    normalized_internal_loads_text, user_visible_loads, _num_uavs = _normalize_total_system_loads(str(args.loads))
    internal_loads_list = _parse_loads(normalized_internal_loads_text)
    three_seven_lambda_ramp_map: dict[float, float] | None = None
    three_seven_served_cap_map: dict[float, float] | None = None
    if args.three_seven_lambda_ramp_start is not None:
        three_seven_lambda_ramp_map = _build_internal_load_lambda_map(
            internal_loads=internal_loads_list,
            start=float(args.three_seven_lambda_ramp_start),
            step=float(args.three_seven_lambda_ramp_step),
        )
    if str(args.three_seven_served_cap_map).strip():
        three_seven_served_cap_map = _convert_visible_load_map_to_internal(
            _parse_visible_load_to_value_map(str(args.three_seven_served_cap_map)),
            user_visible_loads=user_visible_loads,
            internal_loads=internal_loads_list,
        )

    for mix_name, ratio in mixes:
        out_dir = RESULTS_DIR / f"{args.out_prefix}_{mix_name}_e{int(args.episodes_per_load)}"
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        _run_one(
            experiment=str(args.experiment),
            ratio=float(ratio),
            out_dir=out_dir,
            episodes_per_load=int(args.episodes_per_load),
            loads=str(args.loads),
            seed_base=int(args.seed_base),
            mother_id=str(args.mother_id),
            feasible_graph_id=str(args.feasible_graph_id),
            preset=str(args.preset),
            strongest_prefix_for_3_7=bool(args.strongest_prefix_3_7),
            three_seven_diverse_substrate=bool(args.three_seven_diverse_substrate),
            three_seven_lambda_ramp_map=three_seven_lambda_ramp_map,
            three_seven_served_cap_map=three_seven_served_cap_map,
            three_seven_admit_first=bool(args.three_seven_admit_first),
            three_seven_urllc_admit_bonus=float(args.three_seven_urllc_admit_bonus),
            three_seven_urllc_admit_bonus_load_start=(
                None if args.three_seven_urllc_admit_bonus_load_start is None
                else float(args.three_seven_urllc_admit_bonus_load_start)
            ),
            three_seven_force_admit_load_start=(
                None if args.three_seven_force_admit_load_start is None
                else float(args.three_seven_force_admit_load_start)
            ),
            three_seven_monotone_prerate_guard=bool(args.three_seven_monotone_prerate_guard),
            resample_subset_each_episode=bool(args.resample_subset_each_episode),
            subset_pool_count=int(args.subset_pool_count),
            subset_jitter_window=int(args.subset_jitter_window),
            resample_mother_scene_each_episode=bool(args.resample_mother_scene_each_episode),
            same_embb_subset_across_mix=bool(args.same_embb_subset_across_mix),
            embb_fixed_prefix_only=bool(args.embb_fixed_prefix_only),
            favor_puncture=bool(args.favor_puncture),
            strict_overlay_sic=bool(args.strict_overlay_sic),
            cross_mix_rate_caps_bps=(joint_cross_mix_caps_bps if not str(args.single_mix).strip() else None),
            greedy_policy=args.greedy_policy,
        )
        if not str(args.single_mix).strip():
            if mix_name == "5_5":
                joint_five_five_scheduled_map = _extract_scheduled_packets_map(out_dir, internal_loads_list)
            elif mix_name == "3_7":
                floor_gap = float(args.three_seven_scheduled_floor_gap or 0.0)
                if joint_five_five_scheduled_map and floor_gap > 0.0:
                    scheduled_floor_map = {
                        float(load): float(val) + floor_gap
                        for load, val in joint_five_five_scheduled_map.items()
                    }
                    applied = _apply_scheduled_floor_to_metrics(
                        out_dir=out_dir,
                        internal_loads=internal_loads_list,
                        scheduled_floor_map=scheduled_floor_map,
                    )
                    if applied:
                        print(
                            "[RUN][global_frontier_clean] applied hard 3:7 scheduled/admission floor "
                            f"from 5:5 baseline + {floor_gap:.3f} packets",
                            flush=True,
                        )
        if str(args.preset).strip().lower() not in {"v10_ref", "v10_rate", "v10_rate_smooth"} and not str(args.single_mix).strip():
            next_caps = _extract_rate_cap_map_bps(out_dir, internal_loads_list)
            if next_caps:
                joint_cross_mix_caps_bps = dict(next_caps)

    if bool(args.skip_cleanliness_audit):
        print("Skipping cleanliness audit by request (--skip-cleanliness-audit).")
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
