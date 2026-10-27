# Frozen optimization diagnostics

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
    "task_prefix_length": 28,
    "completed_training_episodes": 27,
    "observed_execution": {
      "cpu_s": 348.29355899999996,
      "from_native": 0,
      "includes_final_independent_audit": true,
      "wall_s": 917.3477914580144
    },
    "independent_verification": {
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
    "optimizer_statistics": {
      "actor_loss": {
        "first": -2.998419761657715,
        "last": -44.37825012207031,
        "minimum": -44.546600341796875,
        "maximum": -2.998419761657715,
        "last100_mean": -43.65571949005127
      },
      "alpha": {
        "first": 0.9997000694274902,
        "last": 0.5678713321685791,
        "minimum": 0.5212000012397766,
        "maximum": 0.9997000694274902,
        "last100_mean": 0.5711133319139481
      },
      "cost_loss": {
        "first": 0.16558146476745605,
        "last": 0.00674948375672102,
        "minimum": 0.00048435249482281506,
        "maximum": 0.49500030279159546,
        "last100_mean": 0.005354387588449754
      },
      "cost_mean": {
        "first": 0.005017641931772232,
        "last": 0.02833893708884716,
        "minimum": -0.016060521826148033,
        "maximum": 0.18310311436653137,
        "last100_mean": 0.042531434330157936
      },
      "entropy": {
        "first": 3.0413968563079834,
        "last": 3.0054502487182617,
        "minimum": 2.9059815406799316,
        "maximum": 3.0430212020874023,
        "last100_mean": 2.989423701763153
      },
      "q_max": {
        "first": 0.10711139440536499,
        "last": 44.67768096923828,
        "minimum": 0.10711139440536499,
        "maximum": 45.12281036376953,
        "last100_mean": 44.12889919281006
      },
      "q_mean": {
        "first": -0.043724555522203445,
        "last": 43.02134323120117,
        "minimum": -0.043724555522203445,
        "maximum": 43.02134323120117,
        "last100_mean": 42.47228088378906
      },
      "q_min": {
        "first": -0.19734762609004974,
        "last": 38.467559814453125,
        "minimum": -0.19734762609004974,
        "maximum": 38.849544525146484,
        "last100_mean": 37.34450714111328
      },
      "reward_loss": {
        "first": 16.354610443115234,
        "last": 0.06499990820884705,
        "minimum": 0.007065360434353352,
        "maximum": 33.89828872680664,
        "last100_mean": 1.516555517576635
      },
      "speed_multiplier": {
        "first": 0.0,
        "last": 4.69221830368042,
        "minimum": 0.0,
        "maximum": 4.69221830368042,
        "last100_mean": 4.658053898811341
      },
      "target_entropy": {
        "first": 2.9836316108703613,
        "last": 2.9836316108703613,
        "minimum": 2.9836316108703613,
        "maximum": 2.9836316108703613,
        "last100_mean": 2.9836316108703613
      },
      "terminal_multiplier": {
        "first": 0.000999999581836164,
        "last": 5.1190643310546875,
        "minimum": 0.000999999581836164,
        "maximum": 5.1190643310546875,
        "last100_mean": 5.069062452316285
      }
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
    "task_prefix_length": 28,
    "completed_training_episodes": 27,
    "observed_execution": {
      "cpu_s": 401.87148699999995,
      "from_native": 0,
      "includes_final_independent_audit": true,
      "wall_s": 924.1629678339814
    },
    "independent_verification": {
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
    "optimizer_statistics": {
      "actor_loss": {
        "first": -2.9978280067443848,
        "last": -36.861724853515625,
        "minimum": -36.861724853515625,
        "maximum": -2.9978280067443848,
        "last100_mean": -36.458800315856934
      },
      "alpha": {
        "first": 0.9997000694274902,
        "last": 0.3134725093841553,
        "minimum": 0.3134725093841553,
        "maximum": 0.9997000694274902,
        "last100_mean": 0.31612626403570177
      },
      "cost_loss": {
        "first": 0.027325132861733437,
        "last": 0.0021796729415655136,
        "minimum": 9.702329407446086e-06,
        "maximum": 0.027325132861733437,
        "last100_mean": 0.0006576456317998236
      },
      "cost_mean": {
        "first": 0.0033022386487573385,
        "last": 0.006464097648859024,
        "minimum": 0.00020052575564477593,
        "maximum": 0.008650297299027443,
        "last100_mean": 0.0063603238202631475
      },
      "entropy": {
        "first": 3.0413331985473633,
        "last": 3.0098063945770264,
        "minimum": 2.9542322158813477,
        "maximum": 3.0429599285125732,
        "last100_mean": 2.9966516661643983
      },
      "q_max": {
        "first": 0.10029344260692596,
        "last": 37.409915924072266,
        "minimum": 0.10029344260692596,
        "maximum": 38.226707458496094,
        "last100_mean": 37.38818244934082
      },
      "q_mean": {
        "first": -0.0442606545984745,
        "last": 36.04353713989258,
        "minimum": -0.0442606545984745,
        "maximum": 36.04353713989258,
        "last100_mean": 35.63322292327881
      },
      "q_min": {
        "first": -0.19749967753887177,
        "last": 32.49449920654297,
        "minimum": -0.19749967753887177,
        "maximum": 32.59284210205078,
        "last100_mean": 31.7558939743042
      },
      "reward_loss": {
        "first": 16.451908111572266,
        "last": 0.10501886904239655,
        "minimum": 0.007718396373093128,
        "maximum": 21.301963806152344,
        "last100_mean": 1.5618340701423585
      },
      "speed_multiplier": {
        "first": 0.0,
        "last": 0.0,
        "minimum": 0.0,
        "maximum": 0.0,
        "last100_mean": 0.0
      },
      "target_entropy": {
        "first": 2.9836316108703613,
        "last": 2.9836316108703613,
        "minimum": 2.9836316108703613,
        "maximum": 2.9836316108703613,
        "last100_mean": 2.9836316108703613
      },
      "terminal_multiplier": {
        "first": 0.000999999581836164,
        "last": 5.127732753753662,
        "minimum": 0.000999999581836164,
        "maximum": 5.127732753753662,
        "last100_mean": 5.077384705543518
      }
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
    "task_prefix_length": 28,
    "completed_training_episodes": 27,
    "observed_execution": {
      "cpu_s": 418.448128,
      "from_native": 0,
      "includes_final_independent_audit": true,
      "wall_s": 1006.7595957079902
    },
    "independent_verification": {
      "episodes": 27,
      "full_grid_states": 1000,
      "journal_head": "88c16c44884d5f1001d511215784fc61f6c66d54619399fd65b3e5ce2040f21f",
      "native_steps": 50000,
      "reconstructed_candidates": 21000,
      "replay_slots_verified": 50000,
      "selected_exhaustive": 50000,
      "training_grid_exhaustive": false,
      "training_grid_scope": "native insertion % 100 == 0, current and successor; Diagnostic all states"
    },
    "optimizer_statistics": {
      "actor_loss": {
        "first": -2.9952306747436523,
        "last": -36.88286209106445,
        "minimum": -36.88286209106445,
        "maximum": -2.9952306747436523,
        "last100_mean": -36.492716751098634
      },
      "alpha": {
        "first": 0.9997000694274902,
        "last": 0.31462186574935913,
        "minimum": 0.31462186574935913,
        "maximum": 0.9997000694274902,
        "last100_mean": 0.3172431030869484
      },
      "cost_loss": {
        "first": 0.023412559181451797,
        "last": 5.666379365720786e-05,
        "minimum": 5.524719654204091e-06,
        "maximum": 0.023412559181451797,
        "last100_mean": 0.0004935958728765399
      },
      "cost_mean": {
        "first": 0.003206526627764106,
        "last": 0.0052176290191709995,
        "minimum": 6.47204797132872e-05,
        "maximum": 0.007836966775357723,
        "last100_mean": 0.005484923971816897
      },
      "entropy": {
        "first": 3.0385990142822266,
        "last": 3.0087156295776367,
        "minimum": 2.950655937194824,
        "maximum": 3.042722463607788,
        "last100_mean": 2.996279513835907
      },
      "q_max": {
        "first": 0.10032819211483002,
        "last": 37.44756317138672,
        "minimum": 0.10032819211483002,
        "maximum": 38.296512603759766,
        "last100_mean": 37.44600269317627
      },
      "q_mean": {
        "first": -0.04428694769740105,
        "last": 36.060028076171875,
        "minimum": -0.04428694769740105,
        "maximum": 36.060028076171875,
        "last100_mean": 35.66215412139893
      },
      "q_min": {
        "first": -0.1970365196466446,
        "last": 31.63486671447754,
        "minimum": -0.1970365196466446,
        "maximum": 32.47299575805664,
        "last100_mean": 31.731390686035155
      },
      "reward_loss": {
        "first": 16.395923614501953,
        "last": 0.10120996832847595,
        "minimum": 0.007963964715600014,
        "maximum": 21.337371826171875,
        "last100_mean": 1.557793896868825
      },
      "speed_multiplier": {
        "first": 0.0,
        "last": 0.0,
        "minimum": 0.0,
        "maximum": 0.0,
        "last100_mean": 0.0
      },
      "target_entropy": {
        "first": 2.9809446334838867,
        "last": 2.983248233795166,
        "minimum": 2.9743645191192627,
        "maximum": 2.9836316108703613,
        "last100_mean": 2.9833472990989685
      },
      "terminal_multiplier": {
        "first": 0.0009999994654208422,
        "last": 5.115715026855469,
        "minimum": 0.0009999994654208422,
        "maximum": 5.115715026855469,
        "last100_mean": 5.065010952949524
      }
    }
  }
}

Every scientific update checks all reported diagnostics and all learned network parameters for finite values. Min/max/final/last-100 summaries are descriptive; no checkpoint selection or tuning follows.
