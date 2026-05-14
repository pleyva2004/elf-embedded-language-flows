# improvements/

> Runnable Python prototypes for the proposals in [`../05-improvements.tex`](../05-improvements.tex).

Each file implements one specific subsection of `05-improvements.tex` (Implementation / Code Improvements). Cross-references below.

---

## Prototypes

| File | Implements (§ in 05-improvements.tex) | One-line description |
|------|---------------------------------------|----------------------|
| [`cfg-schedule.py`](./cfg-schedule.py) | Implementation / Code Improvements §1 | Compares constant-$\omega$ CFG to U-shape and linear-ramp $\omega(t)$ schedules on a toy 2D Gaussian-mixture target; reports quality (mean log-pdf under the true mixture) and diversity (2D-histogram entropy) |

## Run

```bash
pip install -r requirements.txt
python cfg-schedule.py
```

CPU-runnable, <60s. Trains a tiny conditional flow-matching model with CFG dropout, then runs three sampling strategies and prints a comparison.

## Expected output (illustrative)

```
                                  strategy   mean log p   entropy
                              ------------   ----------   -------
                       fixed omega = 3.0          -2.41      1.87
U-shape: omega(t) = 3 * (2t(1-t)+0.01)^-0.5      -1.93      1.92
              linear ramp: omega(t) = 1 + 4t      -2.18      1.85

U-shape schedule beat fixed omega on quality (1.92 > 1.87)
```

The number you actually get will vary across runs (random init); the trend (U-shape better than fixed at boundaries-heavy schedules) is the testable claim.

## Notes

This is a toy probe of the *schedule shape*, not a faithful re-run of ELF at scale. The real test would swap the 2D mixture for OpenWebText and the toy MLP for ELF's 105M-parameter network. But the schedule-based CFG mechanic itself is plumbing-level — if it works in 2D, it works in $\mathbb{R}^{768}$ at scale.
