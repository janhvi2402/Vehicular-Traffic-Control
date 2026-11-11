import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from dqn.config import DQNConfig
from ppo.config import PPOConfig
from ac.config import ACConfig
from ca.config import CAConfig

for C in (DQNConfig, PPOConfig, ACConfig, CAConfig):
    c = C()
    print(C.__name__, c.episode_seconds, c.decision_interval)