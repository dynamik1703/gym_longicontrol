# Numerical contract
Production acceleration_from_action, np.interp, min/max clipping and the original
Python x/v arithmetic order are retained. dt remains 0.1; speed tolerance remains
exactly zero; certificate velocity margin 1e-6 and sign enclosure 1e-7 are inherited.
The old absolute frontier is checked without receiving a new observation. Memory
ingest, pointwise-maximum lower envelope and remembered signs are unchanged.

Test runtime: Python 3.11.13, NumPy 2.4.6, Gymnasium 1.4.0. The binary64 argument
assumes the same deterministic runtime for prediction and execution, no model
mutation and the inherited sensing model. Power-model layer absolute bounds are
computed with exact rational sums and upward conversion. A new platform needs
new verification; finite test agreement is not a machine-checked hardware proof.
