# STATUS — Layer-Balancing (`4vztmTrGhd`)

**Session:** autoloop. **Last updated:** 2026-07-18. **State:** judge-gap revision synced; awaiting re-judge.

## Paper and source

- arXiv 2606.00340; OpenReview `4vztmTrGhd`.
- Official repository `TDCSZ327/Layer-Balancing`, pinned commit
  `97997b4307187836b18ce0269bb7b4275f16db9e`.

## Current evidence

- Claim 1: exact constrained landscapes across 12 valid `(h, alpha)` settings.
- Claim 1 three-layer repair: ten independent approaches all pass. The strongest
  are direct clean and label-noise (`rho=0.01`) matrix-GD runs at `h=d=n=1000`
  over seeds 2020–2024; equal rates are a one-step local maximum and a two-step
  global minimum in both conditions. The exact reduced-gradient implementation
  matches literal full-`X` autograd updates within `6.11e-16`.
- Three-layer source audit: the pinned plotting source omits the paper's
  `2 eta1 eta2 / h^3` one-step term. The discrepancy is exactly isolated and
  does not alter any of the 30 tested Figure-2 landscape classifications.
- Claim 2: independent autograd gradient checks, compact polynomial identity,
  and direct Haar-matrix Monte Carlo evaluation of signal-only dynamics.
- Fail-closed controls: theorem boundary and corrupted formula coefficient.
- Full run: 12/12 one-step local maxima and 12/12 two-step local minima;
  max gradient error `1.735e-17`; maximum matrix/formula discrepancy `0.77`
  standard errors; 17/17 tests and 2/2 controls pass.
- The official full-scale empirical path (`h=n=d=1000`, test size 4000, five
  seeds) was also captured: the theoretical loss differs from exact GD by 0.145%
  after one step and 1.633% after two. This is consistent with the paper's bounded
  surrogate error (`O(h^-1)` / `O(h^-1/2)`) and is disclosed rather than called
  bit-exact. The helper script's `not-verified` label comes from its own stricter
  0.1% cutoff, which is not a condition in the paper.
- The first official verdict at Space SHA `9fca7ac` was medium quality, 2/4:
  Claim 2 verified and Claim 1 inconclusive solely because the logbook tested
  two-layer networks but not the explicitly claimed three-layer networks.
- The revision's captured ten-approach run, raw CSV/JSON artifacts, source-drift
  disclosure, and updated pinned executive summary are synced to
  `DineshAI/4vztmTrGhd` at Space SHA `89a07a0`. The public bucket contains all
  four new raw artifacts. The evidence commit is public on GitHub at `5a054b1`.
- Publish gate passed: both end-to-end reproducers reran, 37/37 tests passed,
  Python compilation passed, the artifact row-count/readback audit passed, the
  secret/absolute-path scan was clean, and the public Space/tag/bucket readback
  matched the local revision.

## Next

- Poll the official verdict for Space SHA `89a07a0`; if Claim 1 is not upgraded
  to verified, use the new rationale to choose the next materially different
  attempt. Expected score: 4/4.
