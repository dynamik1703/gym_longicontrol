import copy

import gymnasium as gym
import numpy as np
import pytest
import torch

from benchmarks.bankers_viability_shield_v1.backup_policy import CausalMemory
from benchmarks.bankers_viability_shield_v1.fixtures import make_track
from benchmarks.bankers_viability_shield_v4.serialization import canonical_bytes
from gym_longicontrol.envs.longicontrol import LongiControlEnv

from benchmarks.bankers_masked_rl_pilot_v2.checkpointing import load, save
from benchmarks.bankers_masked_rl_pilot_v2.integrity import guard_panel, guard_seed, owner
from benchmarks.bankers_masked_rl_pilot_v2.masked_distribution import distribution, sample
from benchmarks.bankers_masked_rl_pilot_v2.sac_agent import GAMMA_NATIVE, DiscreteSAC, targets
from benchmarks.bankers_masked_rl_pilot_v2.shield_treatments import METHODS, Treatment
from benchmarks.bankers_masked_rl_pilot_v2.training import Runner, stream_rows, update_target
from benchmarks.bankers_masked_rl_pilot_v2.verification import Auditor


def synthetic_track(_):
    return make_track([0], [20])


def fixture_environment():
    return gym.wrappers.TimeLimit(LongiControlEnv(track_generator=synthetic_track), 1800)


@pytest.mark.parametrize('count', [1, 3, 21])
def test_masked_distribution(count):
    logits = torch.arange(21, dtype=torch.float64, requires_grad=True)
    mask = torch.arange(21) < count
    p, lp, h, target = distribution(logits, mask)
    assert torch.equal(p[~mask], torch.zeros_like(p[~mask]))
    assert torch.isclose(p.sum(), torch.tensor(1.0, dtype=p.dtype))
    assert torch.isfinite(lp).all() and torch.isfinite(h).all()
    assert target.item() == pytest.approx(0.98 * np.log(count))
    rng = torch.Generator().manual_seed(11)
    for _ in range(100):
        assert mask[sample(logits, mask, rng)]
    (-h).backward()
    assert torch.equal(logits.grad[~mask], torch.zeros_like(logits.grad[~mask]))
    if count == 1:
        assert p[0] == 1 and h == 0 and target == 0


def test_synthetic_bellman():
    p = torch.tensor([[0.25, 0.75]])
    lp = p.log()
    qr = torch.tensor([[2.0, 4.0]])
    qc = torch.tensor([[[2.0, 6.0], [3.0, 7.0], [0.0, 0.0]]])
    reward = torch.tensor([[1.0]])
    cost = torch.tensor([[0.0, 1.0, 0.0]])
    yr, yc = targets(reward, cost, torch.tensor([[GAMMA_NATIVE]]), p, lp, qr, qc, 0.2)
    entropy = -0.25 * np.log(0.25) - 0.75 * np.log(0.75)
    assert yr.item() == pytest.approx(1 + GAMMA_NATIVE * (3.5 + 0.2 * entropy))
    np.testing.assert_allclose(yc.numpy(), [[GAMMA_NATIVE * 5, 1 + GAMMA_NATIVE * 6, 0]])
    yr, yc = targets(reward, cost, torch.zeros(1, 1), p, lp, qr, qc, 0.2)
    assert torch.equal(yr, reward) and torch.equal(yc, cost)
    assert GAMMA_NATIVE ** 10 == pytest.approx(0.99)


def test_schedule_and_stream():
    targets_ = [update_target(i) for i in range(50001)]
    assert all(x == 0 for x in targets_[:10001])
    assert targets_[10008] == 1 and targets_[-1] == 5000
    assert max(np.diff(targets_)) == 1
    assert len(stream_rows()) == 865


@pytest.mark.parametrize('method', METHODS)
def test_native_evidence_and_recovery(method, tmp_path):
    runner = Runner(method, 'V2_SYNTHETIC_RECOVERY_' + method, fixture_environment(), True, 512)
    for _ in range(260):
        row = runner.advance(full_evidence=True, learn=False)
    # Include real optimizer states and replay RNG in a mid-episode checkpoint.
    runner.agent.update(runner.replay.sample(runner.replay_rng))
    path = tmp_path / 'technical.pkl'
    meta = save(path, runner)
    expected = []
    for _ in range(3):
        expected.append(runner.advance(full_evidence=True, learn=False))
        runner.agent.update(runner.replay.sample(runner.replay_rng))
    recovered = load(path, meta['sha256'], runner.attempt)
    for row in expected:
        actual = recovered.advance(full_evidence=True, learn=False)
        assert canonical_bytes(actual) == canonical_bytes(row)
        recovered.agent.update(recovered.replay.sample(recovered.replay_rng))
    for k, module in runner.agent.modules().items():
        for key, value in module.state_dict().items():
            assert torch.equal(value, recovered.agent.modules()[k].state_dict()[key])
    assert torch.equal(runner.agent.log_alpha, recovered.agent.log_alpha)
    assert torch.equal(runner.agent.multipliers, recovered.agent.multipliers)
    for k, array in runner.replay.arrays.items():
        assert np.array_equal(array, recovered.replay.arrays[k])
    auditor = Auditor(synthetic=True)
    fresh = Runner(method, 'V2_SYNTHETIC_AUDIT_' + method, fixture_environment(), True, 512)
    for _ in range(5):
        row = fresh.advance(full_evidence=True, learn=False)
        auditor.verify(row, fresh.environment.unwrapped.track, method, full=True)
    assert auditor.rows == 5


def test_post_tie_and_stale_binding():
    env = fixture_environment()
    o, _ = env.reset(seed=11)
    m = CausalMemory()
    m.ingest(o)
    t = Treatment(METHODS[1], {})
    mask, certificates, binding = t.mask(o, m)
    with pytest.raises(RuntimeError, match='STALE'):
        t.execute(10, o, m, mask, certificates, '0' * 64)
    from benchmarks.bankers_viability_shield_v3.safe_action_mask import closest_certified
    bits = [False] * 21
    bits[0] = bits[20] = True
    assert closest_certified(bits, 0) == 0


def test_held_out_and_lock(tmp_path):
    for panel in ('VALIDATION', 'PAPER', 'HYBRID_RL_VALIDATION_V1'):
        with pytest.raises(Exception):
            guard_panel(panel)
    for seed in range(4000, 4018):
        with pytest.raises(Exception):
            guard_seed(seed)
    with owner(tmp_path):
        with pytest.raises(FileExistsError):
            with owner(tmp_path):
                pass


def test_successor_corruption():
    r = Runner(METHODS[2], 'V2_SYNTHETIC_CORRUPTION', fixture_environment(), True, 512)
    row = r.advance(full_evidence=True, learn=False)
    bad = copy.deepcopy(row)
    bad['next_binding'] = 'f' * 64
    with pytest.raises(Exception):
        Auditor(synthetic=True).verify(bad, r.environment.unwrapped.track, METHODS[2], True)


def test_finite_update():
    r = Runner(METHODS[2], 'V2_SYNTHETIC_OPTIMIZER', fixture_environment(), True, 512)
    for _ in range(256):
        r.advance(learn=False)
    result = r.agent.update(r.replay.sample(r.replay_rng))
    assert all(np.isfinite(x) for x in result.values())
