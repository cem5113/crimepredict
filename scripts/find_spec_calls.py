# scripts/find_spec_calls.py
import os, re
root = os.path.dirname(os.path.abspath(__file__)) or "."
pat = re.compile(r"spec\.loader\.exec_module|spec_from_file_location")
hits = []
for base, _, files in os.walk(root):
    for f in files:
        if f.endswith(".py"):
            p = os.path.join(base, f)
            try:
                with open(p, "r", encoding="utf-8") as fh:
                    txt = fh.read()
                if pat.search(txt):
                    hits.append(p)
            except Exception:
                pass
print("\n".join(hits) or "No matches")
