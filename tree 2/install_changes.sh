#!/bin/bash
# Usage (from your PROJECT ROOT):  bash /path/to/tree/install_changes.sh /path/to/tree
# Copies every changed/new file into place. Does NOT overwrite existing __init__.py files.
set -e
T="${1:-$(dirname "$0")}"
[ -d environment ] && [ -d dqn ] && [ -d ppo ] || { echo "run this from the project root (needs environment/ dqn/ ppo/)"; exit 1; }
mkdir -p diagnostics tests scripts docs
while IFS= read -r f; do
  mkdir -p "$(dirname "$f")"; cp "$T/$f" "$f"; echo "copied $f"
done < <(cd "$T" && find environment dqn ppo diagnostics tests scripts docs -type f ! -name '__init__.py' | sort)
for d in environment dqn ppo diagnostics; do [ -f "$d/__init__.py" ] || { touch "$d/__init__.py"; echo "created $d/__init__.py"; }; done
chmod +x scripts/smoke_test.sh
[ -f dqn/analyze_switching_behavior.py ] && git rm -q dqn/analyze_switching_behavior.py && echo "removed dqn/analyze_switching_behavior.py (replaced by diagnostics/)"
python3 scripts/make_scaled_routes.py --in sumo_4x4_network/routes.rou.xml --factor 1.5 --out sumo_4x4_network/routes_x1.5.rou.xml
python3 scripts/make_scaled_routes.py --in sumo_4x4_network/routes.rou.xml --factor 2.0 --out sumo_4x4_network/routes_x2.0.rou.xml
echo; echo "Next:  python tests/test_masking.py     (expect 9 PASS)"
echo "       git diff -- dqn/train.py ppo/train.py environment/eval_common.py   (review, then paste to me)"
