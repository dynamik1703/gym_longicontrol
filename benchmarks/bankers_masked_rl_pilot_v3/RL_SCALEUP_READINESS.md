# Prospective scale-up

{
  "synthetic_admission_forecast": {
    "scaleup_one_method_three_seeds_300k_s": 181125.99635683422,
    "scaleup_three_methods_three_seeds_300k_s": 376613.1108106298
  },
  "observed_pilot_forecast": {
    "cpu_s": {
      "pre_three_seeds_300k": 20096.404834,
      "all_three_seeds_300k": 51712.22955999999,
      "measured_training": {
        "RAW_DISCRETE_SAC": 348.29355899999996,
        "BANKER_POST_SAC": 401.87148699999995,
        "BANKER_PRE_SAC": 418.448128
      },
      "measured_diagnostic": {
        "RAW_DISCRETE_SAC": 199.41388400000005,
        "BANKER_POST_SAC": 568.8999609999996,
        "BANKER_PRE_SAC": 428.795335,
        "CAUSAL60": 34.460227000000486,
        "BANKERS_MASKED_CAUSAL60": 301.1967289999998,
        "ADAPTIVE_PRICE_MPC_V2": 294.09315199999946
      }
    },
    "wall_s": {
      "pre_three_seeds_300k": 43355.485080911545,
      "all_three_seeds_300k": 125269.216609445,
      "measured_training": {
        "RAW_DISCRETE_SAC": 917.3477914580144,
        "BANKER_POST_SAC": 924.1629678339814,
        "BANKER_PRE_SAC": 1006.7595957079902
      },
      "measured_diagnostic": {
        "RAW_DISCRETE_SAC": 1116.3852742109739,
        "BANKER_POST_SAC": 1486.8387581259594,
        "BANKER_PRE_SAC": 692.1258224169724,
        "CAUSAL60": 85.4164350440551,
        "BANKERS_MASKED_CAUSAL60": 354.33766250198823,
        "ADAPTIVE_PRICE_MPC_V2": 439.93825291498797
      }
    }
  }
}

Forecasts include online masks, optimization, evidence, independent audits, checkpoint/recovery and physical baselines. If pilot did not execute these are synthetic forecasts, not actual pilot extrapolations. V2 110/230-hour projection remains historical. No 300k performance inference or automatic scale-up authorization.
