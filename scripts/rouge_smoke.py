"""Smoke test pyrouge via the Windows/Cygwin patch."""
import sys, os, tempfile
sys.path.insert(0, os.path.dirname(__file__))
import rouge_patch

ROUGE_HOME = "/d/KHDL/LuanAn/tesing/downloads/rouge_setup/ROUGE-1.5.5"
rouge_patch.apply(rouge_home=ROUGE_HOME)
from pyrouge import Rouge155

tmp = tempfile.mkdtemp()
sys_dir = os.path.join(tmp, "system"); os.makedirs(sys_dir)
mod_dir = os.path.join(tmp, "model"); os.makedirs(mod_dir)
with open(os.path.join(sys_dir, "text.001.txt"), "w", encoding="utf-8") as f:
    f.write("the hotel room was clean and quiet with a comfortable bed")
with open(os.path.join(mod_dir, "text.A.001.txt"), "w", encoding="utf-8") as f:
    f.write("the room was clean quiet and the bed was very comfortable")

r = Rouge155(rouge_dir=ROUGE_HOME.replace("/d/", "D:/"))
r.system_dir = sys_dir
r.model_dir = mod_dir
r.system_filename_pattern = r"text.(\d+).txt"
r.model_filename_pattern = "text.[A-Z].#ID#.txt"
out = r.convert_and_evaluate()
print(out)
print("SMOKE OK")
