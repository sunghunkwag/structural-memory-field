"""Comparison baseline: SMF vs trivial action selectors.

Task: train on pattern A for 40 steps, silence for 40 steps, then
test if method correctly selects action 0 (the trained pattern's channel).

Methods compared:
1. Random: picks actions uniformly at random
2. Most-frequent: always picks the most common historical action
3. SMF: uses structural memory field physics
"""
import sys
import numpy as np
from collections import Counter
from smf.engine.v2 import EngineV2
from smf.config.params import EngineConfig

TRAIN_STEPS = 40
SILENCE_STEPS = 40
TEST_STEPS = 20
NUM_TRIALS = 5

s_A = np.array([3, 0, 0, 0, 0, 0, 0, 0])  # target: action 0
empty = np.zeros(8)


def run_smf_trial():
    """Run SMF engine: train, silence, test recall."""
    cfg = EngineConfig()
    cfg.stress.knot_threshold = 0.3
    e = EngineV2(cfg=cfg)
    for _ in range(TRAIN_STEPS):
        e.step(s_A)
    for _ in range(SILENCE_STEPS):
        e.step(empty)
    actions = [e.step(s_A) for _ in range(TEST_STEPS)]
    return Counter(actions).most_common(1)[0][0]


def run_random_trial():
    """Random baseline: pick actions uniformly."""
    return np.random.randint(0, 4)


def run_most_frequent_trial():
    """Most-frequent baseline: train phase always gives 0 (it's the only input),
    but after silence there's no memory mechanism — just return last seen."""
    # This baseline has no memory — it just returns action 0 because
    # during training that was the only input. But it can't adapt to
    # new patterns or distinguish them.
    return 0


print("=== Comparison: SMF vs Baselines ===")
print(f"Task: train A={list(s_A[:4])}..., silence {SILENCE_STEPS} steps, recall A")
print(f"Target action: 0 (pattern A's channel)")
print()

# Run trials
np.random.seed(42)
smf_correct = sum(run_smf_trial() == 0 for _ in range(NUM_TRIALS))
random_correct = sum(run_random_trial() == 0 for _ in range(NUM_TRIALS))
freq_correct = sum(run_most_frequent_trial() == 0 for _ in range(NUM_TRIALS))

print(f"  Random baseline:         {random_correct}/{NUM_TRIALS} correct")
print(f"  Most-frequent baseline:  {freq_correct}/{NUM_TRIALS} correct")
print(f"  SMF engine:              {smf_correct}/{NUM_TRIALS} correct")
print()

# SMF advantage: test with MULTIPLE patterns where most-frequent fails
print("=== Multi-pattern test (where most-frequent fails) ===")
s_B = np.array([0, 0, 3, 0, 0, 0, 0, 0])  # target: action 2

cfg = EngineConfig()
cfg.stress.knot_threshold = 0.3
e = EngineV2(cfg=cfg)

# Train A then B
for _ in range(TRAIN_STEPS):
    e.step(s_A)
for _ in range(TRAIN_STEPS):
    e.step(s_B)
for _ in range(SILENCE_STEPS):
    e.step(empty)

# Test: does SMF distinguish A from B?
smf_action_A = Counter([e.step(s_A) for _ in range(TEST_STEPS)]).most_common(1)[0][0]
smf_action_B = Counter([e.step(s_B) for _ in range(TEST_STEPS)]).most_common(1)[0][0]
smf_discriminates = smf_action_A != smf_action_B

print(f"  SMF action for A: {smf_action_A}")
print(f"  SMF action for B: {smf_action_B}")
print(f"  SMF discriminates A vs B: {smf_discriminates}")
print(f"  Random would discriminate: ~6% chance (1/16)")
print(f"  Most-frequent cannot discriminate: always same action")

print()
if smf_correct >= random_correct:
    print("SMF beats or matches random baseline.")
else:
    print("WARNING: SMF underperformed random.")
    sys.exit(1)
