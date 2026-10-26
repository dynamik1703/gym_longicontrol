# Controller comparison remains untested on Development
The four prepared control laws are frozen in controller_adapters.py and evaluation.py.
CAUSAL60 remains unchanged and has no Banker shadow mask. The legacy arm is the
native+1 experimental wrapper with inherited predicates and CAUSAL60 replacement;
it is not the original capped target-speed controller. Banker greedy and masked
CAUSAL60 use the identical 21-action safety predicate and full final certification.

No matched18-task comparison, joint-feasible energy comparison, added masked
failure identity, deadline-loss count or finite-grid progress conclusion exists.
These quantities are null in results.json, not zero. A speed certificate cannot
substitute for the missing progress experiment.

Inherited safety-predicate diagnostics were preregistered for Banker native steps
0,100,200,... . No Development states were sampled (0/0; fraction undefined).
Synthetic probe/fixture records mark each actually sampled row explicitly; those
records are not a census. Endpoint, backup, preview-assumption, deadline-only,
domain/proposal and numerical categories remain declared for future analysis;
no full-state predicate agreement is inferred from partial synthetic diagnostics.
