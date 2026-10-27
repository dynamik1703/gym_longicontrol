# Training evidence

Independent native selected-action verification, predecessor/envelope reconstruction and sampled full-grid audits are required online and from stored evidence; every replay slot is verified before final sealing.

{
  "RAW_DISCRETE_SAC": {
    "episodes": 27,
    "full_grid_states": 0,
    "journal_head": "798519c87d908e9b81661a761182a9e37e6cc77ffca8d50a06df6611b1d2803c",
    "native_steps": 50000,
    "reconstructed_candidates": 0,
    "replay_slots_verified": 50000,
    "selected_exhaustive": 0,
    "training_grid_exhaustive": false,
    "training_grid_scope": "native insertion % 100 == 0, current and successor; Diagnostic all states"
  },
  "BANKER_POST_SAC": {
    "episodes": 27,
    "full_grid_states": 1000,
    "journal_head": "ea94aafe1d5624e002c1c251aa9dec14b034e6bee69944e49115e976429088db",
    "native_steps": 50000,
    "reconstructed_candidates": 21000,
    "replay_slots_verified": 50000,
    "selected_exhaustive": 50000,
    "training_grid_exhaustive": false,
    "training_grid_scope": "native insertion % 100 == 0, current and successor; Diagnostic all states"
  },
  "BANKER_PRE_SAC": {
    "episodes": 27,
    "full_grid_states": 1000,
    "journal_head": "88c16c44884d5f1001d511215784fc61f6c66d54619399fd65b3e5ce2040f21f",
    "native_steps": 50000,
    "reconstructed_candidates": 21000,
    "replay_slots_verified": 50000,
    "selected_exhaustive": 50000,
    "training_grid_exhaustive": false,
    "training_grid_scope": "native insertion % 100 == 0, current and successor; Diagnostic all states"
  }
}
