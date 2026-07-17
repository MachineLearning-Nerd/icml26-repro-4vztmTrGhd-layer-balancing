# STATUS — Layer-Balancing (`4vztmTrGhd`)

**Session:** autoloop. **Last updated:** 2026-07-17. **State:** locally complete; publication queued.

## Paper and source

- arXiv 2606.00340; OpenReview `4vztmTrGhd`.
- Official repository `TDCSZ327/Layer-Balancing`, pinned commit
  `97997b4307187836b18ce0269bb7b4275f16db9e`.

## Current evidence

- Claim 1: exact constrained landscapes across 12 valid `(h, alpha)` settings.
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
- Trackio logbook is complete, tagged, pinned, and secret-scanned. Publishing to
  `DineshAI/4vztmTrGhd` was attempted on 2026-07-17 but Hugging Face rejected
  Space creation because the account reached its 20-Spaces-per-day limit. The
  server requested retry in about 23 hours; no Space currently exists.

## Next

- Retry `trackio logbook publish DineshAI/4vztmTrGhd` after the daily Space-creation
  quota resets, verify the public tags and bucket, then move the registry row to
  `under_verdict`.
