from tools.run_atem3d_with_runtime import _pardiso_ooc_settings


def test_pardiso_ooc_settings_records_intel_environment(monkeypatch):
    monkeypatch.setenv("MKL_PARDISO_OOC_PATH", r"D:\pardiso-ooc")
    monkeypatch.setenv("MKL_PARDISO_OOC_MAX_CORE_SIZE", "16384")
    monkeypatch.setenv("MKL_PARDISO_OOC_MAX_SWAP_SIZE", "0")
    monkeypatch.setenv("MKL_PARDISO_OOC_KEEP_FILE", "0")

    assert _pardiso_ooc_settings() == {
        "path": r"D:\pardiso-ooc",
        "max_core_size_mb": 16384,
        "max_swap_size_mb": 0,
        "keep_file": False,
    }
