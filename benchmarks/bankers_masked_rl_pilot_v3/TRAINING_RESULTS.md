# Training results

{
  "RAW_DISCRETE_SAC": {
    "native": 50000,
    "updates": 5000,
    "actor_sha256": "7832c01746ff1a7f8607ef99b4a7178ff3e6332c1d3c68974633b85de1ee7c8d",
    "checkpoint": {
      "attempt": "BANKERS_MASKED_RL_PILOT_V3__RAW_DISCRETE_SAC__SEED_11__ATTEMPT_01",
      "bytes": 11829187,
      "native": 50000,
      "sha256": "850c23734f870d77bbe8d99aa95e3b3f8764d807a0ec631c3681aed565848913",
      "uncompressed_bytes": 96020430,
      "updates": 5000
    },
    "completed": true,
    "verification": {
      "episodes": 27,
      "full_grid_states": 0,
      "journal_head": "798519c87d908e9b81661a761182a9e37e6cc77ffca8d50a06df6611b1d2803c",
      "native_steps": 50000,
      "reconstructed_candidates": 0,
      "replay_slots_verified": 50000,
      "selected_exhaustive": 0,
      "training_grid_exhaustive": false,
      "training_grid_scope": "native insertion % 100 == 0, current and successor; Diagnostic all states"
    }
  },
  "BANKER_POST_SAC": {
    "native": 50000,
    "updates": 5000,
    "actor_sha256": "c40bd4f65abb1f47e86e8538aeda2fb951f5427ce77e5f4918e8d0ff748be1cb",
    "checkpoint": {
      "attempt": "BANKERS_MASKED_RL_PILOT_V3__BANKER_POST_SAC__SEED_11__ATTEMPT_01",
      "bytes": 11802926,
      "native": 50000,
      "sha256": "7dd2a62e6768b7f209d2581be91fbb0ff95c423fa0c328346d82db7dcf609a29",
      "uncompressed_bytes": 96029273,
      "updates": 5000
    },
    "completed": true,
    "verification": {
      "episodes": 27,
      "full_grid_states": 1000,
      "journal_head": "ea94aafe1d5624e002c1c251aa9dec14b034e6bee69944e49115e976429088db",
      "native_steps": 50000,
      "reconstructed_candidates": 21000,
      "replay_slots_verified": 50000,
      "selected_exhaustive": 50000,
      "training_grid_exhaustive": false,
      "training_grid_scope": "native insertion % 100 == 0, current and successor; Diagnostic all states"
    }
  },
  "BANKER_PRE_SAC": {
    "native": 50000,
    "updates": 5000,
    "actor_sha256": "0d6a9683b6e21fbada4f079ef2804a92ae2f806d5b7ec9da06a02b3d53a26464",
    "checkpoint": {
      "attempt": "BANKERS_MASKED_RL_PILOT_V3__BANKER_PRE_SAC__SEED_11__ATTEMPT_01",
      "bytes": 11811290,
      "native": 50000,
      "sha256": "93cd8120fff5b6d11537b5310469abffb5f5dbede4b7c8f96b30c09e6a1901e4",
      "uncompressed_bytes": 96029271,
      "updates": 5000
    },
    "completed": true,
    "verification": {
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
}

Counts are actual committed native journal events. No final actor/checkpoint exists for an uncompleted arm.
