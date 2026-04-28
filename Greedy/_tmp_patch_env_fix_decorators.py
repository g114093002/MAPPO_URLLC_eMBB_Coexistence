from pathlib import Path

path = Path(r'd:\URLLC_eMBB_Coexisting\sr_mappo\env.py')
lines = path.read_text(encoding='utf-8').splitlines(True)
fixed = []
for line in lines:
    if line.startswith("        @"):
        fixed.append("    " + line[8:])
    else:
        fixed.append(line)
path.write_text("".join(fixed), encoding='utf-8')
print("deindented env decorators to class scope")
