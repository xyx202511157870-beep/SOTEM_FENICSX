"""检查公开核心算法文件是否包含面向中文读者的物理说明。"""

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]

CORE_FILES = (
    "dolfinx/sotem_pipeline.py",
    "src/atem3d/simulation.py",
    "src/atem3d/hj.py",
    "src/atem3d/materials/cole_cole.py",
    "src/atem3d/materials/prony.py",
    "src/atem3d/primary/dc.py",
    "src/atem3d/sources.py",
    "src/atem3d/receivers.py",
    "src/atem3d/magnetic_recovery.py",
    "sotem_ip/cole_cole.py",
    "sotem_ip/debye.py",
    "sotem_ip/forward.py",
)


def test_core_algorithm_files_contain_chinese_physical_explanations():
    missing = []
    for relative_path in CORE_FILES:
        text = (ROOT / relative_path).read_text(encoding="utf-8")
        if re.search(r"[\u4e00-\u9fff]", text) is None:
            missing.append(relative_path)

    assert missing == []
