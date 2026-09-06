# Flow 0 baseline report

- git: `paper/receiver-oriented-adaptive-debye-mvp` `72c065bff7aaefd852182df153f9462f918fa960`
- python: `3.12.3 (main, Mar 23 2026, 19:04:32) [GCC 13.3.0]`
- empymod: `2.6.0`
- numpy: `2.4.4`
- scipy: `1.18.1`
- pytest: `tests/adaptive_debye_mvp` including official PR 9 `test_layered_forward.py` (15 passed, ~7 min)
- official smoke shared_survey_hash: `1704f69ccc648cbb...` IP increment `2.546e-04`
- layered smoke case: `PG01` IP increment peak `3.459e-06` shared_survey_hash match `7aee8d77a6d71e2f...`
- API: `atem3d.adaptive_debye_mvp.layered_forward.compute_layered_response`
- 3-D sweep not run.
- Solver code was not modified in Flow 0.
