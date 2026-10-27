# Explorative Auswertung des kontrollierten Piloten

BANKERS_MASKED_RL_PILOT_V3_NO_LEARNING_SIGNAL

Ein Trainingsseed, zwölf Aufgaben; Split e5fb92ae36eb9b57eb3be2ffe5e801b32ae9de425118fc32b7538d7f5a1ad5a4. Alle Rohnenner bleiben 12. Keine Überlegenheits- oder Äquivalenzbehauptung.

| Controller | RSR /12 | Completion /12 | Deadline /12 | Speed /12 | Median Fortschritt m | Feasible Energie kWh (n) |
|---|---:|---:|---:|---:|---:|---:|
| RAW_DISCRETE_SAC | 0 | 0 | 0 | 12 | 0.000 | nicht verfügbar |
| BANKER_POST_SAC | 0 | 0 | 0 | 12 | 24.176 | nicht verfügbar |
| BANKER_PRE_SAC | 0 | 0 | 0 | 12 | 29.042 | nicht verfügbar |
| CAUSAL60 | 12 | 12 | 12 | 12 | 1000.980 | 0.196299 (12) |
| BANKERS_MASKED_CAUSAL60 | 12 | 12 | 12 | 12 | 1000.827 | 0.198182 (12) |
| ADAPTIVE_PRICE_MPC_V2 | 12 | 12 | 12 | 12 | 1000.165 | 0.167539 (12) |

Aufgabenpaarige Primärvergleiche:

BANKER_PRE_SAC__VS__RAW_DISCRETE_SAC: A allein erfolgreich 0, B allein 0, beide 0, beide fehlgeschlagen 12; mittlere gepaarte Fortschrittsdifferenz 16.452 m. Energie wird ausschließlich auf gemeinsam feasible Aufgaben verglichen (n=0).

BANKER_PRE_SAC__VS__BANKER_POST_SAC: A allein erfolgreich 0, B allein 0, beide 0, beide fehlgeschlagen 12; mittlere gepaarte Fortschrittsdifferenz 9.864 m. Energie wird ausschließlich auf gemeinsam feasible Aufgaben verglichen (n=0).

BANKER_POST_SAC__VS__RAW_DISCRETE_SAC: A allein erfolgreich 0, B allein 0, beide 0, beide fehlgeschlagen 12; mittlere gepaarte Fortschrittsdifferenz 6.588 m. Energie wird ausschließlich auf gemeinsam feasible Aufgaben verglichen (n=0).

Fehleridentitäten:

RAW_DISCRETE_SAC: [{"numeric_seed": 12100000, "reasons": ["INCOMPLETE", "DEADLINE"]}, {"numeric_seed": 12100001, "reasons": ["INCOMPLETE", "DEADLINE"]}, {"numeric_seed": 12100003, "reasons": ["INCOMPLETE", "DEADLINE"]}, {"numeric_seed": 12100004, "reasons": ["INCOMPLETE", "DEADLINE"]}, {"numeric_seed": 12100005, "reasons": ["INCOMPLETE", "DEADLINE"]}, {"numeric_seed": 12100006, "reasons": ["INCOMPLETE", "DEADLINE"]}, {"numeric_seed": 12100007, "reasons": ["INCOMPLETE", "DEADLINE"]}, {"numeric_seed": 12100009, "reasons": ["INCOMPLETE", "DEADLINE"]}, {"numeric_seed": 12100010, "reasons": ["INCOMPLETE", "DEADLINE"]}, {"numeric_seed": 12100011, "reasons": ["INCOMPLETE", "DEADLINE"]}, {"numeric_seed": 12100014, "reasons": ["INCOMPLETE", "DEADLINE"]}, {"numeric_seed": 12100015, "reasons": ["INCOMPLETE", "DEADLINE"]}]

BANKER_POST_SAC: [{"numeric_seed": 12100000, "reasons": ["INCOMPLETE", "DEADLINE"]}, {"numeric_seed": 12100001, "reasons": ["INCOMPLETE", "DEADLINE"]}, {"numeric_seed": 12100003, "reasons": ["INCOMPLETE", "DEADLINE"]}, {"numeric_seed": 12100004, "reasons": ["INCOMPLETE", "DEADLINE"]}, {"numeric_seed": 12100005, "reasons": ["INCOMPLETE", "DEADLINE"]}, {"numeric_seed": 12100006, "reasons": ["INCOMPLETE", "DEADLINE"]}, {"numeric_seed": 12100007, "reasons": ["INCOMPLETE", "DEADLINE"]}, {"numeric_seed": 12100009, "reasons": ["INCOMPLETE", "DEADLINE"]}, {"numeric_seed": 12100010, "reasons": ["INCOMPLETE", "DEADLINE"]}, {"numeric_seed": 12100011, "reasons": ["INCOMPLETE", "DEADLINE"]}, {"numeric_seed": 12100014, "reasons": ["INCOMPLETE", "DEADLINE"]}, {"numeric_seed": 12100015, "reasons": ["INCOMPLETE", "DEADLINE"]}]

BANKER_PRE_SAC: [{"numeric_seed": 12100000, "reasons": ["INCOMPLETE", "DEADLINE"]}, {"numeric_seed": 12100001, "reasons": ["INCOMPLETE", "DEADLINE"]}, {"numeric_seed": 12100003, "reasons": ["INCOMPLETE", "DEADLINE"]}, {"numeric_seed": 12100004, "reasons": ["INCOMPLETE", "DEADLINE"]}, {"numeric_seed": 12100005, "reasons": ["INCOMPLETE", "DEADLINE"]}, {"numeric_seed": 12100006, "reasons": ["INCOMPLETE", "DEADLINE"]}, {"numeric_seed": 12100007, "reasons": ["INCOMPLETE", "DEADLINE"]}, {"numeric_seed": 12100009, "reasons": ["INCOMPLETE", "DEADLINE"]}, {"numeric_seed": 12100010, "reasons": ["INCOMPLETE", "DEADLINE"]}, {"numeric_seed": 12100011, "reasons": ["INCOMPLETE", "DEADLINE"]}, {"numeric_seed": 12100014, "reasons": ["INCOMPLETE", "DEADLINE"]}, {"numeric_seed": 12100015, "reasons": ["INCOMPLETE", "DEADLINE"]}]

CAUSAL60: []

BANKERS_MASKED_CAUSAL60: []

ADAPTIVE_PRICE_MPC_V2: []
