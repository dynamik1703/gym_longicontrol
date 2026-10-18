# Future split policy and repair recommendation

## Policy comparison

| Policy | Rejection rule | Scientific meaning | Distribution effect |
|---|---|---|---|
| S — Seed only | reject reserved numeric seeds | Tests new RNG labels, not necessarily new task content. | Negligible, but permits route and exact-task leakage. |
| F — Full-task clean | S plus reject reserved `full_task_sha256` | Excludes MDP-equivalent tasks while allowing the same infrastructure under different requirements. | Tiny for fixed V2; broader task families retain route reuse. |
| R — Route clean | S plus reject reserved `route_sha256` (and full-task hash defensively) | Tests on unseen operational infrastructure as well as unseen exact tasks. | Removes all requirements on a matching route; slight conditional distortion, concentrated in common low-complexity routes. |

For a paper claiming generalization to **unseen constrained-control tasks**, use **Policy
R (Route clean)**. Here the route supplies nearly all stochastic task variation and can
materially simplify control even when a deadline changes. Full-task-clean is necessary
but not sufficient for an unseen-content claim. Route exclusions should be checked
before replay insertion, with numeric exclusion retained as defense in depth.

This choice is based on generator/task semantics, not observed policy performance. The
rejected set is only 18 fixed Development+Validation routes relative to a 265,949,468
route support, although the generator's non-uniformity means rejection rates must be
reported rather than assumed negligible.

## Reusable hard gate

`fingerprints.assert_task_not_reserved(...)` accepts a `TaskFingerprint` and policy
`S`, `F`, or `R`. Every policy always checks numeric seeds; F additionally checks the
full-task hash; R checks both full-task and route hashes. It raises
`ReservedTaskError` before a task can be inserted into training data. Unit tests cover
all rejection modes and allowed route reuse under F.

Completed historical runners were deliberately not modified.

## Stopped 2×2 repair

Recommend **Option 1: preserve T_HIST and T_CURRENT episode-for-episode and replace the
primary diagnostic evaluation set with a pre-frozen route-clean set**. That preserves
the factorial's causal treatment. Original Development can remain a disclosed secondary
calibration result, but cannot be called content-clean.

- Option 2, filtering/resampling colliding training tasks, keeps Development clean but
  changes the very historical/current streams the factorial is intended to isolate.
- Option 3, seed-only interpretation, is defensible only for expected performance under
  a repetitive generator distribution. It does not match an unseen-task generalization
  claim.

## Proposed diagnostic set (designed, not used)

Candidates come from the separate sequential namespace beginning at 9,000,000. All 18
numeric seeds are absent from the two 500-task critical streams and all reserved seed
ranges. Their route and full-task hashes are absent from both critical streams,
Development, and Validation. Candidate routes are mutually unique. No policy was
evaluated on them.

| Seed | `route_sha256` | `full_task_sha256` |
|---:|---|---|
| 9000000 | `206e570674a9fb95cd872543962e8a6ba65194eebc2506c3941379e64a68d1c5` | `54a8373206b1dad57f2f70b8ec0509cd8f012675c1c462275e61773db7c55b78` |
| 9000001 | `4e85508ada30c32d13018c190e80dbf3e514214ea02f160093bcf66d0a832dd2` | `5511964dbf4c1ecf02529c6883ca902561a68fc00e2925130515767779b92aeb` |
| 9000002 | `b8b3d1cab9efa229ea797fa4b96805cb5d311831587c3d9714bd27f0f4344b6e` | `210fc7ca5e9dc721b3cb6cba2c0bc4a4e058e89ead866449e8776f8d34c072db` |
| 9000003 | `a8e5d3ed8bfc27e281252d870637ef9969b921e72b7fbc41025d4ec7fda6f1be` | `702fceb9afb0ed24680bef78e0006e9002a75ecf02b379fdef9336f24987f002` |
| 9000004 | `65e6bd02d21db427a994f98096ce1417f4fe7d0f40b45a27e61bf6b4b09cb90d` | `0f0e914b6d99b67898c5dc3a7e0dd88a170529ff908b1cfab47a0d2ae782d6f3` |
| 9000005 | `37421e43f27b4e72f5516f962be36e23c86f15be602f66ecd23151c561b6369c` | `3a04942965fea88d7969485a38756d84bb473f0ee60d1757572557d31f094c6b` |
| 9000006 | `5a68958089d8406f6cde6296be061367f68fc5af42ea461c82d329eaa4a70e56` | `d3a35dedb4a5989bfe52cc0d654c6a46cd6d540a4bf0336c7a72147f8d732639` |
| 9000007 | `69d80b033cb20b510feb8440e526025de8829604d4052f3d67263ec7f4dff530` | `b03ef4b72880efead3216710c66f6d4b9f84bcbd6e507ce33aa0d9a8bf525468` |
| 9000008 | `d09b99f6cba37402bab8cfafd803599d609b45f650cd71aead79a048f6fde925` | `1334d3dab2e5d15e875e8614ab101d87166458f5a8978af8483bb6f32ef1b58d` |
| 9000009 | `9f49dd287866ba92d9c9f50739752de2e5943fde001c9d7401a7e9dcd5261f7e` | `1d3d93f5d27c25d798fe4a3f47fc08e6830520713ff39b7d16c141e0e48a4637` |
| 9000010 | `f547a45b815831601b56ecc479897d9318dc0ae6e0e5cecbf8c3cd930e126143` | `dee0fd567eceb6e618cd814dc60463167de6357e760e4883ea0f39498ae05e2d` |
| 9000011 | `d3d27268460c013162fe528781896e8df9bae97e9c17a3e8f4b31dee88eb38ba` | `dfca82e1194bddd28e1f6f73e0eae4a02440b94a7cb42f2d8d9ddca376df9b4e` |
| 9000012 | `51048a51ca2ad6e9da47aff7d71b9a2ec9fb28cef58522313b0ceaa5d3a527dc` | `c7aed07cd386288b86a516d6b6013adc05c8ec59c839c78f7a61669f29e23cfd` |
| 9000013 | `306506971a513d840414dadac626546d6c746aa33747d2f0fd89a4c81d60060e` | `844cae4a8acd0b2855d7707401045b950108c2200a309c556a6f1ef8bc15c064` |
| 9000014 | `f065b20b28901cb2a0cc04efd873c982f93a3148a6ce2ee8235f6e87784bbc00` | `76b22f6142d45e8582f79f57ade34ed927e8c75a541d9f434004e656c7e80400` |
| 9000015 | `282457e690fd82d7f4dcf76db02bb451ffc792d043de75a8a5c90ceb55b15b5c` | `1f15facc6c0281a00dd0cde6a737bd55ae179d2e097074a72703e220e6a35f86` |
| 9000016 | `ec4b2be34f23dbc0986dc08e940aa4cc81a3e731b78f84581838e72ac866dc2b` | `39b96deaf7a7ac391733471a71aff7c1e3d4657a998d7acb49d59da0a6db7150` |
| 9000017 | `e895e24f2a2c640a51bcd88acc7d65d31671e4adbde62d92346644e4a6ba2281` | `764e5549b4e511b6fee153e12c75c2a135710e8c9d87116cd4c9acaacd7843eb` |

This is a design artifact, not authorization to evaluate. Before any future training,
freeze the set in the new study protocol and enforce Policy R bidirectionally: reject
training tasks matching any diagnostic route, and reject any proposed diagnostic route
matching the fully frozen training streams.
