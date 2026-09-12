"""Fuehrt alle Notebook-Code-Zellen sequenziell aus (Smoke-Test fuer die Deliverables)."""
import json
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
nb = json.load(open(os.path.join(HERE, "Dynamic_NN_Pipeline.ipynb"), encoding="utf-8"))

ns = {"__name__": "__main__", "__file__": "Dynamic_NN_Pipeline.ipynb"}
sys.path.insert(0, HERE)
os.chdir(HERE)
import matplotlib
matplotlib.use("Agg")

failures = 0
for idx, cell in enumerate(nb["cells"]):
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell["source"])
    print(f"[cell {idx}] executing ({len(src)} chars)...")
    try:
        exec(compile(src, f"nb-cell-{idx}", "exec"), ns)
    except Exception as e:
        failures += 1
        print(f"!!! CELL {idx} FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        break
print("FAILURES:", failures)
sys.exit(1 if failures else 0)