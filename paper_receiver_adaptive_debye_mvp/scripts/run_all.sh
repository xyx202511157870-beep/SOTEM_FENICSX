#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

GEN="$ROOT/generated/receiver_adaptive_debye_mvp"
mkdir -p "$GEN"

python "$ROOT/paper_receiver_adaptive_debye_mvp/scripts/run_flow0.py"
python "$ROOT/paper_receiver_adaptive_debye_mvp/scripts/freeze_protocol.py"
python "$ROOT/paper_receiver_adaptive_debye_mvp/scripts/run_oracle_gap.py"

if [[ ! -f "$GEN/flow2_oracle_gap/L0_summary.json" ]]; then
  echo "L0 summary missing" >&2
  exit 1
fi

L0_PASSED="$(python -c "import json; print(json.load(open('$GEN/flow2_oracle_gap/L0_summary.json'))['passed'])")"
if [[ "$L0_PASSED" != "True" ]]; then
  python - <<'PY'
from pathlib import Path
import json
root = Path("generated/receiver_adaptive_debye_mvp")
payload = {
    "status": "STOP_LAYERED_NO_ACTIONABLE_GAP",
    "gate": "L0",
    "three_d_run": False,
    "later_stages": "refused",
}
(root / "STOP_REASON.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print("STOP_LAYERED_NO_ACTIONABLE_GAP: refusing Flow 3/4/5")
PY
  exit 2
fi

if [[ ! -f "$GEN/flow3_selector/incoming_train.csv" ]]; then
  echo "Flow 3 inputs are not present; refusing unauthorized later stages" >&2
  exit 3
fi

python "$ROOT/paper_receiver_adaptive_debye_mvp/scripts/train_selector.py"
python "$ROOT/paper_receiver_adaptive_debye_mvp/scripts/run_layered_test.py"
echo "run_all finished without starting 3-D"
