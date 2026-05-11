from pathlib import Path
import re
from sr_mappo.run_greedy_mix_share_grid import (
    _load_greedy_metrics,
    _plot_grid,
    _plot_per_mix_share_comparison,
    _plot_per_share_mix_comparison,
)

out_dir = Path(r"D:\URLLC_eMBB_Coexisting\sr_mappo\results\m37_l15_35_merged")
pat = re.compile(r"share(\d+)_mix_([0-9_]+)_sr_mappo_report_metrics\.json$")

all_data = {}
files = sorted(out_dir.glob("share*_mix_*_sr_mappo_report_metrics.json"))
for f in files:
    m = pat.match(f.name)
    if not m:
        continue
    share_key = f"share{int(m.group(1))}"
    mix_key = m.group(2).replace("_", ":")
    all_data.setdefault(share_key, {})[mix_key] = _load_greedy_metrics(f)

share_keys = sorted(all_data.keys(), key=lambda k: int(k.replace("share", "")))
mixes = sorted({mk for sv in all_data.values() for mk in sv.keys()})

_plot_grid(all_data, out_dir / "mix_share_grid_comparison.png")
for mix in mixes:
    mix_safe = mix.replace(":", "_")
    _plot_per_mix_share_comparison(
        all_data,
        mix,
        out_dir / f"mix_{mix_safe}_share_comparison.png",
        share_keys,
    )
for share in share_keys:
    _plot_per_share_mix_comparison(all_data, share, out_dir / f"{share}_mix_comparison.png")

print("done")
