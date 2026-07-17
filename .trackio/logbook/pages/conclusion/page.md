# Conclusion


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_1a7ff476a3e5", "created_at": "2026-07-17T03:31:44+00:00", "title": "Executive summary", "pinned": true, "pinned_at": "2026-07-17T03:31:45+00:00"}
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
