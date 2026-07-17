# Claim 2 — exact gradients and losses


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_155351fe8ddc", "created_at": "2026-07-17T03:31:39+00:00", "title": "Independent triangulation"}
-->
Three independent checks are used: exact whitened-data gradients versus PyTorch autograd in float64; Equation 82 reduced to a compact polynomial in p = eta1 + eta2 and q = eta1 eta2; and direct Haar-matrix signal-only updates from Equations 12-14 compared statistically with Theorem 5.3.


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_f2d74fcf24ba", "created_at": "2026-07-17T03:31:40+00:00", "title": "Result"}
-->
**VERIFIED:** 10 gradient trials have maximum absolute error 1.735e-17; the independent one-step identity has zero error; all six direct-matrix comparisons are within 0.77 standard errors over 1000 Haar seeds each.


---
<!-- trackio-cell
{"type": "code", "id": "cell_04991e08918d", "created_at": "2026-07-17T03:31:42+00:00", "title": "Run 17 independent checks", "command": ["python", "-m", "pytest", "repro/tests", "-q"], "exit_code": 0, "duration_s": 2.102}
-->
````bash
$ python -m pytest repro/tests -q
````

exit 0 · 2.1s


````output
.................                                                        [100%]
17 passed in 1.50s

````
