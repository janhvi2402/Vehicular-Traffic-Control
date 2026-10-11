"""
scripts/write_missing_config.py -- Play button, no arguments. ONLY for runs trained with an OLDER train.py that did not
write runs/<algo>_seed<N>/config.json (scripts/check_matched_budget.py reports those as "no config.json").
It writes the config from DQNConfig() / PPOConfig() with that run's seed. Correct ONLY if that run used the default
config files (no edits to dqn/config.py or ppo/config.py since it started). Runs that already have config.json are skipped.
"""
import dataclasses, glob, json, os, re, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from dqn.config import DQNConfig
from ppo.config import PPOConfig

wrote = 0
for algo, cls in (("dqn", DQNConfig), ("ppo", PPOConfig)):
    for d in sorted(glob.glob(os.path.join(ROOT, "runs", f"{algo}_seed*"))):
        m = re.search(r"_seed(\d+)$", d); path = os.path.join(d, "config.json")
        if not m or not os.path.isdir(d) or os.path.exists(path): continue
        cfg = cls(); cfg.seed = int(m.group(1))
        json.dump(dataclasses.asdict(cfg), open(path, "w"), indent=2)
        print("wrote", path); wrote += 1
print(f"done: {wrote} config.json file(s) written")
