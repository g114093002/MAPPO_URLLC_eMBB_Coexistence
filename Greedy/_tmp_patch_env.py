from pathlib import Path
from textwrap import dedent

path = Path(r'd:\URLLC_eMBB_Coexisting\sr_mappo\env.py')
text = path.read_text(encoding='utf-8')

def replace_once(text, old, new, label):
    if old not in text:
        raise SystemExit(f'missing pattern for {label}')
    return text.replace(old, new, 1)

if 'overlay_retention_gate_for_load' not in text:
    text = replace_once(text, 'from .load_aware import load_aware_reward_schedule, nearest_reference_load\n', 'from .load_aware import (\n    load_aware_reward_schedule,\n    nearest_reference_load,\n    overlay_retention_gate_for_load,\n    power_ratio_ceiling_for_load,\n    puncture_loss_ceiling_for_load,\n)\n', 'env imports')

if 'self.puncture_candidate_total = 0' not in text:
    text = replace_once(text, '        self.urllc_executed_power_delta_sum = 0.0\n        self.last_topology = None\n', '        self.urllc_executed_power_delta_sum = 0.0\n        self.puncture_candidate_total = 0\n        self.puncture_candidate_pruned_by_loss_ceiling_count = 0\n        self.puncture_candidate_overlay_suppressed_count = 0\n        self.selected_overlay_admission_count = 0\n        self.selected_puncture_admission_count = 0\n        self.last_topology = None\n', 'env counters')

if 'def _current_puncture_loss_ceiling' not in text:
    block = dedent('''
    def _current_puncture_loss_ceiling(self, actual_load: Optional[float] = None) -> float:
        return puncture_loss_ceiling_for_load(
            self._current_actual_load() if actual_load is None else float(actual_load),
            getattr(self.rl_cfg.training, "puncture_loss_ceiling_by_load", {}),
            fallback=float("inf"),
        )

    def _current_overlay_retention_gate(self, actual_load: Optional[float] = None) -> float:
        return overlay_retention_gate_for_load(
            self._current_actual_load() if actual_load is None else float(actual_load),
            getattr(self.rl_cfg.training, "overlay_retention_gate_by_load", {}),
            fallback=0.0,
        )

    def _current_power_ratio_ceiling(self, actual_load: Optional[float] = None) -> float:
        return power_ratio_ceiling_for_load(
            self._current_actual_load() if actual_load is None else float(actual_load),
            getattr(self.rl_cfg.training, "selection_power_ratio_ceiling_by_load", {}),
            fallback=float("inf"),
        )

    def _apply_low_damage_candidate_constraints(
        self,
        candidates: List[CandidatePacket],
        actual_load: Optional[float] = None,
    ) -> List[CandidatePacket]:
        if not bool(getattr(self.rl_cfg.training, "low_damage_admission_objective", False)):
            return candidates
        actual = self._current_actual_load() if actual_load is None else float(actual_load)
        puncture_loss_ceiling_mbps = self._current_puncture_loss_ceiling(actual)
        overlay_gate = self._current_overlay_retention_gate(actual)
        filtered: List[CandidatePacket] = []
        for candidate in candidates:
            if candidate.puncture_feasible:
                self.puncture_candidate_total += 1
                puncture_loss_mbps = float(candidate.puncture_loss) / 1.0e6
                if puncture_loss_mbps > puncture_loss_ceiling_mbps + 1e-12:
                    candidate.puncture_feasible = False
                    candidate.puncture_utility = float("-inf")
                    self.puncture_candidate_pruned_by_loss_ceiling_count += 1
                elif candidate.overlay_feasible and float(candidate.overlay_retention) >= overlay_gate:
                    candidate.puncture_feasible = False
                    candidate.puncture_utility = float("-inf")
                    self.puncture_candidate_overlay_suppressed_count += 1
            if candidate.overlay_feasible or candidate.puncture_feasible:
                filtered.append(candidate)
        return filtered

''')
    text = replace_once(text, '    def _counterfactual_local_reward(\n', block + '    def _counterfactual_local_reward(\n', 'insert low-damage helper block')

if 'candidates = self._apply_low_damage_candidate_constraints' not in text:
    text = replace_once(text, '        candidates = self._enumerate_candidates_for_cell(uav_idx, rb, minislot)\n        self._annotate_candidate_contention([candidates])\n        return self._select_candidate_subset(candidates, minislot, uav_idx)\n', '        candidates = self._enumerate_candidates_for_cell(uav_idx, rb, minislot)\n        candidates = self._apply_low_damage_candidate_constraints(candidates, self._current_actual_load())\n        self._annotate_candidate_contention([candidates])\n        return self._select_candidate_subset(candidates, minislot, uav_idx)\n', 'apply low-damage gating')

if 'terminal_puncture_loss_penalty' not in text:
    text = replace_once(text, '            embb_min_penalty = float(getattr(self.rl_cfg.reward, "terminal_embb_min_rate_penalty", 0.0))\n            if embb_min_penalty > 0.0:\n                avg_shortfall = float(summary.get("embb_min_rate_shortfall", 0.0))\n                terminal_reward_terms["terminal_embb_min_rate_shortfall"] = -embb_min_penalty * avg_shortfall\n                team_reward += terminal_reward_terms["terminal_embb_min_rate_shortfall"]\n', '            embb_min_penalty = float(getattr(self.rl_cfg.reward, "terminal_embb_min_rate_penalty", 0.0))\n            if embb_min_penalty > 0.0:\n                avg_shortfall = float(summary.get("embb_min_rate_shortfall", 0.0))\n                terminal_reward_terms["terminal_embb_min_rate_shortfall"] = -embb_min_penalty * avg_shortfall\n                team_reward += terminal_reward_terms["terminal_embb_min_rate_shortfall"]\n            puncture_loss_penalty = float(getattr(self.rl_cfg.reward, "terminal_puncture_loss_penalty_weight", 0.0))\n            if puncture_loss_penalty > 0.0:\n                avg_puncture_loss = float(summary.get("avg_puncture_embb_loss", 0.0)) / 1.0e6\n                terminal_reward_terms["terminal_puncture_loss_penalty"] = -puncture_loss_penalty * avg_puncture_loss\n                team_reward += terminal_reward_terms["terminal_puncture_loss_penalty"]\n            overlay_retention_bonus = float(getattr(self.rl_cfg.reward, "terminal_overlay_retention_bonus", 0.0))\n            if overlay_retention_bonus > 0.0:\n                avg_overlay_retention = float(summary.get("avg_overlay_retention", 0.0))\n                terminal_reward_terms["terminal_overlay_retention_bonus"] = overlay_retention_bonus * avg_overlay_retention\n                team_reward += terminal_reward_terms["terminal_overlay_retention_bonus"]\n            power_ratio_penalty = float(getattr(self.rl_cfg.reward, "terminal_power_ratio_penalty_weight", 0.0))\n            if power_ratio_penalty > 0.0:\n                baseline_power = max(float(getattr(self, "original_greedy_metrics", {}).get("total_power", 0.0)), 1e-9)\n                power_ratio = float(summary.get("total_power", 0.0)) / baseline_power if baseline_power > 0.0 else 1.0\n                terminal_reward_terms["terminal_power_ratio_penalty"] = -power_ratio_penalty * max(power_ratio - 1.0, 0.0)\n                team_reward += terminal_reward_terms["terminal_power_ratio_penalty"]\n', 'terminal low-damage reward')

if 'self.selected_overlay_admission_count += 1' not in text:
    text = replace_once(text, '        if mode == MODE_OVERLAY:\n            self.overlay_counts[uav_idx] += 1\n            self.selected_overlay_retentions.append(float(candidate.overlay_retention))\n            self.selected_overlay_losses.append(float(candidate.overlay_loss))\n            self.overlay_selected_pairs += 1\n            self.overlay_success_ema[uav_idx] = 0.9 * self.overlay_success_ema[uav_idx] + 0.1 * float(candidate.overlay_retention)\n        elif mode == MODE_PUNCTURE:\n            self.puncture_counts[uav_idx] += 1\n            self.selected_puncture_losses.append(float(candidate.puncture_loss))\n            loss_norm = float(candidate.puncture_loss / 1.0e6)\n            self.puncture_loss_ema[uav_idx] = 0.9 * self.puncture_loss_ema[uav_idx] + 0.1 * loss_norm\n', '        if mode == MODE_OVERLAY:\n            self.overlay_counts[uav_idx] += 1\n            self.selected_overlay_retentions.append(float(candidate.overlay_retention))\n            self.selected_overlay_losses.append(float(candidate.overlay_loss))\n            self.overlay_selected_pairs += 1\n            self.overlay_success_ema[uav_idx] = 0.9 * self.overlay_success_ema[uav_idx] + 0.1 * float(candidate.overlay_retention)\n            self.selected_overlay_admission_count += 1\n        elif mode == MODE_PUNCTURE:\n            self.puncture_counts[uav_idx] += 1\n            self.selected_puncture_losses.append(float(candidate.puncture_loss))\n            self.selected_puncture_admission_count += 1\n            loss_norm = float(candidate.puncture_loss / 1.0e6)\n            self.puncture_loss_ema[uav_idx] = 0.9 * self.puncture_loss_ema[uav_idx] + 0.1 * loss_norm\n', 'selected mode counts')

if 'admission_via_overlay_ratio' not in text:
    text = replace_once(text, '        mean_executed_power_delta = float(self.urllc_executed_power_delta_sum / urllc_projection_count)\n\n        return {\n', '        mean_executed_power_delta = float(self.urllc_executed_power_delta_sum / urllc_projection_count)\n        admission_via_overlay_ratio = float(self.selected_overlay_admission_count / max(scheduled_packets, 1))\n        admission_via_puncture_ratio = float(self.selected_puncture_admission_count / max(scheduled_packets, 1))\n        puncture_candidate_pruned_by_loss_ceiling_ratio = float(self.puncture_candidate_pruned_by_loss_ceiling_count / max(self.puncture_candidate_total, 1))\n\n        return {\n', 'summary extra ratios pre return')
    text = replace_once(text, "            'avg_overlay_embb_loss': float(np.mean(self.selected_overlay_losses)) if self.selected_overlay_losses else 0.0,\n", "            'avg_overlay_embb_loss': float(np.mean(self.selected_overlay_losses)) if self.selected_overlay_losses else 0.0,\n            'admission_via_overlay_ratio': admission_via_overlay_ratio,\n            'admission_via_puncture_ratio': admission_via_puncture_ratio,\n            'puncture_candidate_pruned_by_loss_ceiling_ratio': puncture_candidate_pruned_by_loss_ceiling_ratio,\n", 'summary add low-damage fields')

path.write_text(text, encoding='utf-8')
print('env patched')
