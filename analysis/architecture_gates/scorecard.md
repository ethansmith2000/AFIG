# AFIG architecture gate scorecard

Optimizer steps are the comparison axis. Results below 30k steps are exploratory and must not determine a final promotion alone.

| Arm | Step | Tier | Seeds | Physical NRMSE | Log-amp MAE | Phase error | Radial error |
|---|---:|---|---:|---:|---:|---:|---:|
| b-beta | 5000 | exploratory | 1 | 0.284179 | 0.509020 | 0.290349 | 0.313727 |
| b-default | 5000 | exploratory | 1 | 0.282499 | 0.497263 | 0.288383 | 0.303262 |
| b-polar-off | 5000 | exploratory | 1 | 0.284005 | 0.505231 | 0.288981 | 0.307713 |
| b-target-off | 5000 | exploratory | 1 | 0.283025 | 0.482199 | 0.273572 | 0.301230 |
| f-alpha0 | 5000 | exploratory | 1 | 0.284817 | 0.496709 | 0.287776 | 0.307605 |
| f-alpha02 | 5000 | exploratory | 1 | 0.282671 | 0.495498 | 0.284259 | 0.299479 |
| f-alpha1 | 5000 | exploratory | 1 | 0.289232 | 0.628555 | 0.361849 | 0.367487 |
| f-gain | 5000 | exploratory | 1 | 0.282277 | 0.498112 | 0.284570 | 0.298647 |
| g-clean | 5000 | exploratory | 1 | 0.279664 | 0.495967 | 0.283771 | 0.300956 |
| g-noise | 5000 | exploratory | 1 | 0.281163 | 0.495092 | 0.281754 | 0.298911 |
| n0 | 5000 | exploratory | 1 | 0.284004 | 0.497710 | 0.289195 | 0.299787 |
| n1 | 5000 | exploratory | 1 | 0.283367 | 0.502160 | 0.289111 | 0.306109 |
| n2 | 5000 | exploratory | 1 | 0.284401 | 0.501242 | 0.289468 | 0.301124 |
| p0 | 5000 | exploratory | 1 | 0.294123 | 0.616459 | 0.357978 | 0.375636 |
| p0 | 30000 | medium | 2 | 0.297826 | 0.567996 | 0.362032 | 0.340905 |
| p1 | 5000 | exploratory | 1 | 0.285630 | 0.502603 | 0.290624 | 0.305671 |
| p1 | 30000 | medium | 2 | 0.289162 | 0.404883 | 0.233344 | 0.210599 |
| p2 | 5000 | exploratory | 1 | 0.285550 | 0.515627 | 0.296956 | 0.312686 |
| r0 | 5000 | exploratory | 1 | 0.284641 | 0.502742 | 0.288695 | 0.306050 |
| r1 | 5000 | exploratory | 1 | 0.284517 | 0.502012 | 0.289124 | 0.306912 |
| r2 | 5000 | exploratory | 1 | 0.284067 | 0.497582 | 0.289303 | 0.306673 |
| s0 | 5000 | exploratory | 1 | 0.285178 | 0.499652 | 0.287893 | 0.305559 |
| s1 | 5000 | exploratory | 1 | 0.283815 | 0.494172 | 0.284647 | 0.302475 |
