"""Independent native physics and selected-action audit; declared grid sampling."""

from dataclasses import asdict

import numpy as np

from benchmarks.bankers_viability_shield_v1.backup_policy import CONFIG, CausalMemory
from benchmarks.bankers_viability_shield_v1.certificate import identities
from benchmarks.bankers_viability_shield_v1.safe_action_set import ACTION_GRID
from benchmarks.bankers_viability_shield_v2.independent_verification import verify_envelope
from benchmarks.bankers_viability_shield_v4.independent_verification import (
    physical_certificate, verify_compact,
)
from benchmarks.bankers_viability_shield_v4.serialization import canonical_bytes, digest, require
from gym_longicontrol.domain.dynamics import advance
from gym_longicontrol.domain.observation import observation
from gym_longicontrol.domain.vehicle import VehicleModel

from .shield_treatments import METHODS, causal_binding
from .training import signals


class Auditor:
    def __init__(self):
        self.vehicle = VehicleModel()
        self.original_identity = identities()
        self.rows = self.grid_states = self.candidates = self.selected = 0
        self.previous = None

    def verify(self, row, track, method, full=False):
        require(row['contract'] == 'bankers-training-evidence-v1', 'evidence contract')
        require(row['native'] == row['replay_insertion'], 'replay insertion identity')
        require(row['native'] == self.rows, 'missing/duplicate native evidence')
        memory = CausalMemory.restore(row['memory'])
        ordinary = row['observation']
        expected_o = observation(memory.state, track.sense(memory.state.position_m, 150),
                                 self.vehicle.specs, CONFIG, 1.0)
        require(np.array_equal(ordinary, expected_o), 'actual current observation')
        require(causal_binding(ordinary, memory) == row['binding'], 'current state-mask binding')
        verify_envelope(memory, track)
        if self.previous is not None and row['episode_step'] > 0:
            require(row['memory'] == self.previous['successor_memory'], 'physical episode continuity')
            require(row['binding'] == self.previous['next_binding'], 'successor mask alignment')
            require(row['physical_mask'] == self.previous['next_physical_mask'], 'cached successor mask')
        proposal, selected = row['proposal'], row['executed']
        require(type(proposal) is int and type(selected) is int and 0 <= proposal < 21 and 0 <= selected < 21,
                'native grid indices')
        if method == METHODS[0]:
            require(proposal == selected and row['selected_certificate'] is None, 'RAW altered action')
        else:
            rebuilt, _ = physical_certificate(ordinary, memory, ACTION_GRID[selected], self.vehicle, self.original_identity)
            require(canonical_bytes(rebuilt) == canonical_bytes(row['selected_certificate']), 'selected independent certificate')
            require(rebuilt['accepted'], 'unsafe selected action')
            require(digest(rebuilt) == row['selected_certificate_sha256'], 'selected certificate binding')
            require(row['physical_mask'][selected], 'selected mask bit')
            self.selected += 1
            if method == METHODS[2]:
                require(proposal == selected and row['physical_mask'][proposal], 'PRE replacement')
            else:
                expected = min((i for i, b in enumerate(row['physical_mask']) if b),
                    key=lambda i: (abs(ACTION_GRID[i] - ACTION_GRID[proposal]), ACTION_GRID[i]))
                require(selected == expected, 'POST nearest/lower tie rule')
        actual, power, energy = advance(memory.state, ACTION_GRID[selected], self.vehicle, CONFIG)
        require(asdict(actual) == row['actual_successor'], 'actual native physics')
        require(energy == row['signed_energy'], 'signed native energy')
        limit = track.sense(actual.position_m, 150).current_limit_m_s
        excess = max(0.0, actual.velocity_m_s - limit)
        require(excess == row['speed_excess'] and row['costs'][0] == excess, 'exact positive speed excess')
        require(limit == row['actual_limit'], 'physical native limit')
        if method != METHODS[0]:
            require(excess == 0.0, 'BANKERS_MASKED_RL_PILOT_SAFETY_COUNTEREXAMPLE')
        successor = CausalMemory.restore(row['successor_memory'])
        require(asdict(successor.state) == asdict(actual), 'causal successor state')
        next_o = observation(actual, track.sense(actual.position_m, 150), self.vehicle.specs, CONFIG, 1.0)
        require(np.array_equal(next_o, row['successor_observation']), 'successor observation')
        memory.execute(ACTION_GRID[selected], self.vehicle)
        memory.ingest(next_o)
        require(memory.snapshot() == successor.snapshot(), 'successor causal memory propagation')
        require(causal_binding(next_o, successor) == row['next_binding'], 'successor binding')
        require(row['completed'] == (actual.position_m >= 1000), 'completion')
        reward, costs = signals(dict(step_energy_kwh=energy, speed_excess_m_s=excess,
            elapsed_time_s=actual.elapsed_time_s), row['completed'], row['terminated'],
            row['truncated'], proposal, selected)
        require(row['reward'] == reward and row['costs'] == costs, 'reward/cost accounting')
        sampled = full or row['native'] % 100 == 0
        if method != METHODS[0] and sampled:
            for m, o, key, cert_key in (
                (CausalMemory.restore(row['memory']), ordinary, 'physical_mask', 'certificates'),
                (successor, next_o, 'next_physical_mask', 'next_certificates'),
            ):
                bits = []
                for i, action in enumerate(ACTION_GRID):
                    rebuilt, _ = physical_certificate(o, m, action, self.vehicle, self.original_identity)
                    bits.append(rebuilt['accepted'])
                    if full:
                        verify_compact(row[cert_key][i], o, m, self.vehicle, self.original_identity,
                                       row['identity'], action_index=i)
                    self.candidates += 1
                require(bits == row[key], 'independently rebuilt full mask')
                self.grid_states += 1
        expected_support = row['physical_mask'] if method == METHODS[2] else [True] * 21
        expected_next = row['next_physical_mask'] if method == METHODS[2] else [True] * 21
        require(row['policy_mask'] == expected_support and row['next_policy_mask'] == expected_next,
                'Bellman support contract')
        self.previous = row
        self.rows += 1
        return power

    def report(self):
        return dict(native_steps=self.rows, selected_exhaustive=self.selected,
                    full_grid_states=self.grid_states, reconstructed_candidates=self.candidates,
                    training_grid_scope='native insertion % 100 == 0; both current and successor',
                    training_grid_exhaustive=False)
