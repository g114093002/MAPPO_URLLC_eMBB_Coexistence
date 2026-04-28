from pathlib import Path

path = Path(r'd:\URLLC_eMBB_Coexisting\sr_mappo\env.py')
text = path.read_text(encoding='utf-8')
marker = "\ndef _current_puncture_loss_ceiling(self, actual_load: Optional[float] = None) -> float:\n"
idx = text.find(marker)
if idx < 0:
    raise SystemExit("indent marker not found")
head = text[: idx + 1]
tail = text[idx + 1 :]
tail = "".join(("    " + line) if line.strip() else line for line in tail.splitlines(True))
path.write_text(head + tail, encoding='utf-8')
print("indented env tail into class scope")
