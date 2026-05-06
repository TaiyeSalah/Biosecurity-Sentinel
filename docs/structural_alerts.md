# Structural Alert Library

The Biosecurity Sentinel screens every molecule against a library of
SMARTS patterns derived from Chemical Weapons Convention (CWC) Schedule 1
analog structures and published biosecurity literature.

This screening runs **before the ML model** — it is fast, deterministic,
and does not depend on model training status. A CRITICAL hit triggers an
immediate HALT regardless of what the ML model would predict.

## Why deterministic pre-screening matters

ML models can be wrong. A SchNet model trained on Tox21 has never seen
most novel organophosphate scaffolds and will have high uncertainty on them.
The structural alert layer provides a hard rule-based backstop that catches
known chemical weapons precursor scaffolds with 100% recall on any compound
matching the pattern — no probabilistic threshold, no false negatives from
distribution shift.

## Alert Library

| SMARTS | Name | Severity | Source |
|--------|------|----------|--------|
| `P(=O)(F)` | Organophosphate fluoride | CRITICAL | CWC Schedule 1A — nerve agent scaffold (sarin, soman, cyclosarin all contain this) |
| `P(=O)(CN(C)C)` | Novichok precursor analog | CRITICAL | CWC Schedule 1A analog — A-series nerve agents |
| `[As](=O)(O)O` | Arsenical warfare agent | CRITICAL | CWC Schedule 1B — lewisite class |
| `ClC(Cl)(Cl)Cl` | Carbon tetrachloride scaffold | HIGH | CWC Schedule 3 — phosgene precursor pathway |
| `S(=O)(=O)F` | Sulfonyl fluoride | HIGH | Cholinesterase inhibitor scaffold — published SAR |
| `C(=O)Cl` | Acyl chloride | MODERATE | OPCW precursor screening — reactive pathway |
| `N(=O)[O-]` | Nitro energetic scaffold | MODERATE | Dual-use energetics monitoring |
| `c1cn[nH]c1` | Pyrazole scaffold | LOW | Monitoring flag — low specificity, contextual |

## Severity definitions

- **CRITICAL**: Immediate HALT. Structural match to known chemical weapons precursor or
  warfare agent class. Do not synthesise without direct biosafety officer involvement.
- **HIGH**: High-priority review required. Structural match to cholinesterase inhibitor or
  vesicant precursor. Full safety review before any laboratory work.
- **MODERATE**: Elevated caution. Reactive or potentially dual-use scaffold. Standard
  review recommended.
- **LOW**: Monitoring flag only. Low specificity — common in benign compounds but
  worth noting for context.

## Extending the library

To add a new alert, add an entry to `STRUCTURAL_ALERTS` in `config/settings.py`:

```python
{
    "smarts": "YOUR_SMARTS_PATTERN",
    "name": "Human-readable name",
    "category": "Category string",
    "severity": "CRITICAL",  # or HIGH / MODERATE / LOW
    "reference": "Source citation",
}
```

Then run `pytest tests/test_sentinel.py::TestStructuralAlerts::test_all_smarts_patterns_compile`
to verify the new pattern compiles correctly before committing.

## Limitations

- The library screens for **structural similarity** to known threat scaffolds.
  It does not catch entirely novel scaffolds with no structural precedent.
- The ML model (SchNet + MC Dropout) is designed to handle novel scaffolds
  by returning UNKNOWN rather than a confident prediction.
- The combination of deterministic pre-screening + probabilistic ML + explicit
  uncertainty quantification is the design response to both failure modes.
