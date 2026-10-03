"""
Scale the demand of a SUMO route file without touching the original.
  python scripts/make_scaled_routes.py --in sumo_4x4_network/routes.rou.xml --factor 1.5 \
         --out sumo_4x4_network/routes_x1.5.rou.xml
factor >= 1: every vehicle is kept, floor(f)-1 extra copies are added, plus one more with probability f-floor(f).
factor <  1: each vehicle is kept with probability f (thinning).
Copies keep the original's route/vType and get depart = original + U(0, 3) s (capped at 3599) so they do not
all appear on the same tick; ids are "<orig>_c<k>". Output is sorted by depart (SUMO requires that). Seeded.
"""
import argparse
import copy
import math
import random
import xml.etree.ElementTree as ET

p = argparse.ArgumentParser()
p.add_argument("--in", dest="inp", required=True); p.add_argument("--out", required=True)
p.add_argument("--factor", type=float, required=True); p.add_argument("--seed", type=int, default=12345)
a = p.parse_args()
assert a.inp != a.out, "refusing to overwrite the input file"
rng = random.Random(a.seed)
tree = ET.parse(a.inp); root = tree.getroot()
vehicles = root.findall("vehicle")
non_veh = [c for c in list(root) if c.tag != "vehicle"]
out = []
for v in vehicles:
    f = a.factor
    if f < 1.0:
        if rng.random() < f: out.append(v)
        continue
    out.append(v)
    n_extra = int(math.floor(f)) - 1 + (1 if rng.random() < f - math.floor(f) else 0)
    for k in range(1, n_extra + 1):
        c = copy.deepcopy(v); c.set("id", f"{v.get('id')}_c{k}")
        c.set("depart", f"{min(float(v.get('depart')) + rng.uniform(0, 3), 3599.0):.2f}")
        out.append(c)
out.sort(key=lambda e: float(e.get("depart")))
for e in list(root): root.remove(e)
for e in non_veh + out: root.append(e)
ET.indent(tree, space="    ")
tree.write(a.out, encoding="UTF-8", xml_declaration=True)
print(f"{a.inp}: {len(vehicles)} vehicles -> {a.out}: {len(out)} vehicles (x{len(out) / len(vehicles):.3f}, target x{a.factor})")
