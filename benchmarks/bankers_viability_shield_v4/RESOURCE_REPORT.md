# Frozen V4 resource gate

Source/protocol freeze commit 6c266f6a33aecaddfe387505f93033da3b2076f6. Projected CPU 11379.276836s=3.160910h. Projected peak retained 728018493bytes=0.678020GiB. Gate PASS. V3 historical stop remains unchanged.

|Controller|V3 gzip B/step|V4 gzip B/step|Reduction|V4 raw journal B/step|
|---|---:|---:|---:|---:|
|CAUSAL60|352.8|397.4|-12.64%|3116.6|
|LEGACY_NATIVE_POST_GREEDY|457.4|498.6|-9.01%|3615.4|
|BANKERS_GREEDY|34074.6|4607.0|86.48%|41422.8|
|BANKERS_MASKED_CAUSAL60|33065.6|4752.0|85.63%|43010.6|

The added provenance fields can enlarge baseline evidence; storage savings are assessed separately per controller. All28 speed-band records, actual speed distributions and measured mask/selected/replay/write costs are retained in resource_measurements.json. Formula/accounting differences were committed before measurement. Peak preflight RSS 450592768bytes. Final measured CPU, retained package, live-journal/publication peak and source/sealing overhead appear in resource_accounting.json and budget_monitor.json where Development opened. No hard real-time guarantee is inferred.
