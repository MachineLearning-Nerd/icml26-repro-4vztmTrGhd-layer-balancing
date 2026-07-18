# Conclusion


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_1a7ff476a3e5", "created_at": "2026-07-17T03:31:44+00:00", "title": "Executive summary"}
-->
**Both official claims are verified.** Across 12 valid settings, equal layer rates change from a one-step local maximum to a two-step local minimum. Gradients match autograd to 1.74e-17, the polynomial identity is exact, and six direct Haar-matrix checks agree within 0.77 standard errors. Seventeen tests and two fail-closed controls pass.

## Scope & cost

| | This reproduction | Full replication |
|---|---|---|
| Scope | Exact two-layer theorem instance; 12 landscapes; gradients; formula triangulation; controls | All two-, three-, four-, and eight-layer figures plus Gaussian initialization |
| Hardware | 4-core CPU | CPU or GPU for large-h figures |
| Time | About 13 seconds | Several hours |
| Cost | $0 | Hardware-dependent |
| Outcome | Both challenge claims verified at theorem scale | Extends empirical figures |


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_060bf3b24e6b", "created_at": "2026-07-18T08:56:50+00:00", "title": "Re-judge executive summary", "pinned": true, "pinned_at": "2026-07-18T08:56:51+00:00"}
-->
**Both challenge claims are now verified across their full stated two- and three-layer scope.** The prior 2/4 verdict accepted the two-layer result but found no three-layer experiment. The revision adds ten independent three-layer checks: 30/30 dense Figure-2 curves, 24/24 Corollary curvature checks, symbolic differentiation, pinned-source drift audit, an independent p/q identity, 80-digit arithmetic, 24/24 autograd Hessians, literal-X autograd parity (maximum update error 6.11e-16), and direct clean/noisy h=d=n=1000 matrix GD over five seeds. Both full-scale direct conditions show equal rates as a one-step local maximum and a two-step global minimum. All 37 tests pass.

## Scope & cost

| | This revision | Full paper replication |
|---|---|---|
| Scope | Both two- and three-layer scored claims; exact formulas; direct clean/noisy full-width GD; 10 three-layer proof routes | Also reproduce every 4/8-layer, Gaussian, nonlinear, and ResNet figure |
| Hardware | 4-core CPU, 15 GB RAM | GPU useful for all ancillary figures |
| Time | Three-layer evidence 10 s; complete test suite 2 s | Several hours |
| Cost | $0 | Hardware-dependent |
| Outcome | Closes the judge-identified three-layer gap | Extends beyond scored claims |
