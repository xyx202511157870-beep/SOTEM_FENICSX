# Lei/Song first independent SOTEM benchmark failure

Status: **DONE_WITH_CONCERNS -- failed_with_reproducible_evidence**

The independent-reference gate failed for Lei noIP. The fixed Task 10 comparison at
`srcpts=9 -> 17` remains failed; no SimPEG, FEniCSx source-only, or five-channel
FEniCSx run was started. Higher-order reference runs were retained as diagnostics
and do not replace or waive the original gate.

## Gate result

The comparison used every observation time without scaling, sign fitting, time
translation, point removal, smoothing, or threshold changes. For each component,
the reported value is
`max(abs(srcpts_low - srcpts_high) / max(abs(srcpts_high), floor))`, using the
repository's `src/atem3d/sotem_metrics.py::compare_signed_response` floor of
`0.01 * peak(abs(srcpts_high))` for each component. The fixed threshold was
`0.005`. Ey is excluded from this gate because it is a near-zero symmetry diagnostic,
not one of Task 10's required Ex/Hz/dBzdt convergence components.

| Case | Pair | Ex | Hz | dBzdt | Required Ex/Hz/dBzdt |
|---|---:|---:|---:|---:|---|
| Lei noIP | 9 -> 17 | 0.15806579470979232 | 5.05111505285459e-6 | 0.02358102281597996 | **FAIL** |
| Song noIP | 9 -> 17 | 0.0033395295228093678 | 2.877462353005613e-7 | 0.00046426123098783616 | PASS |
| Song exact Cole-Cole IP | 9 -> 17 | 0.00029705135651102773 | 3.015718759764101e-7 | 3.638568056519078e-6 | PASS |
| Lei noIP higher-order diagnostic | 33 -> 65 | 0.025758874457864886 | 2.867551660265967e-7 | 0.0031690488297546043 | **FAIL** |
| Lei noIP higher-order diagnostic | 65 -> 129 | 0.010871727484161172 | 9.94654270321229e-8 | 0.0015432620703936433 | **FAIL** |

The first fixed-gate Lei failures are both at `time_obs_s=1.0000000000000001e-05`:
Ex is `2.6684091025941119e-05` at 9 points versus
`2.3041947312352895e-05` at 17 points; dBzdt is
`-8.1609403435994625e-08` versus `-7.9729304878551284e-08`. At the diagnostic
cap, the 65 -> 129 Ex maximum is still `0.010871727484161172` (at
`1.5848931924611141e-05`). Thus `srcpts=129` is not proposed as a converged
production reference.

Reason codes:

- `EMPYMOD_REFERENCE_CONVERGENCE_GATE_FAILED_LEI_NOIP`
- `EMPYMOD_REFERENCE_HIGH_ORDER_CAP_EXCEEDED_LEI_NOIP`
- `DOWNSTREAM_SKIPPED_REFERENCE_GATE`
- `PRELIMINARY_REFERENCE_PUBLISH_FAILURE_UNCAPTURED`
- `WINDOWS_ATOMIC_STAGE_PATH_TOO_LONG` (diagnostic inference, not independently preserved traceback evidence)

## Commands and run IDs

All successful reference runs used Windows CPython with `PYTHONPATH=src` and the
following exact command shape, with the values in the table substituted literally:

```powershell
python -m atem3d.sotem_validation_cli prepare --case <case> --solver empymod --level S0T0B0 --run-dir generated/t10/r/<run-id>
python -m atem3d.sotem_validation_cli reference --run-dir generated/t10/r/<run-id> --case <case> --resume --variant <variant> --srcpts <srcpts>
```

| Run ID | Case argument | Variant | srcpts | Manifest status |
|---|---|---|---:|---|
| `l5-691bfe7c` | `benchmarks/sotem/lei2023_noip.yaml` | `noip` | 5 | `reference_complete` |
| `l9-00f2310b` | `benchmarks/sotem/lei2023_noip.yaml` | `noip` | 9 | `reference_complete` |
| `l17-7d716f20` | `benchmarks/sotem/lei2023_noip.yaml` | `noip` | 17 | `reference_complete` |
| `sn5-229f43b0` | `benchmarks/sotem/song2025_layered_pair.yaml` | `noip` | 5 | `reference_complete` |
| `sn9-ab9ed8eb` | `benchmarks/sotem/song2025_layered_pair.yaml` | `noip` | 9 | `reference_complete` |
| `sn17-f6ee60e4` | `benchmarks/sotem/song2025_layered_pair.yaml` | `noip` | 17 | `reference_complete` |
| `si5-1ca3b9f3` | `benchmarks/sotem/song2025_layered_pair.yaml` | `cole-cole-exact` | 5 | `reference_complete` |
| `si9-02870841` | `benchmarks/sotem/song2025_layered_pair.yaml` | `cole-cole-exact` | 9 | `reference_complete` |
| `si17-041fb75f` | `benchmarks/sotem/song2025_layered_pair.yaml` | `cole-cole-exact` | 17 | `reference_complete` |
| `l33-623eb4f6` | `benchmarks/sotem/lei2023_noip.yaml` | `noip` | 33 | `reference_complete` |
| `l65-6152078c` | `benchmarks/sotem/lei2023_noip.yaml` | `noip` | 65 | `reference_complete` |
| `l129-f9bdd9ec` | `benchmarks/sotem/lei2023_noip.yaml` | `noip` | 129 | `reference_complete` |

For the preliminary long-path run
`generated/task10/references/20260720T130856.738892Z-lei-noip-srcpts5-8493a5c8`
the interactive terminal reported empymod completion followed by `FileNotFoundError`
while creating the atomic staging CSV. That terminal output was not persisted, so
neither the exception nor completion of the in-memory calculation is independently
auditable from the run directory. The observed target path was 265 characters, which
supports but does not prove the Windows path-length diagnosis. The preserved manifest
independently establishes only `status=prepared` with no reference stage; its SHA-256 is
`9e9169e040a885d5519114d65154fba8e858755a69dc08325a7b0a193972f039`.
It was not reused or deleted.

## Provenance hashes

Hashes below are SHA-256 in the order `manifest / empymod.csv /
empymod_metadata.json / reference_empymod_or_1d.csv`.

| Run ID | SHA-256 hashes |
|---|---|
| `l5-691bfe7c` | `8d5745af172982fbfd18b12f5ebe0d8387ac8f2e0445ca22444892b89c02c3af` / `ccbd6a9996999df0d50ad1a28e4428b56220bb93c9a660883895977f42ec7a28` / `7f34238d1b45f838e4a5c318913f8064ab32f90a93b51930ce176eabf169aae9` / `98d0d865064603910c31b79d094ea6ad4176e48f9167d0e0e262407b744fc2ec` |
| `l9-00f2310b` | `573c8d6fc7f4b0a28c1e665229a4797833101c6bd1ad023f251207e346a306ea` / `aee05c3e095a9f1318293a65bb2ee6fdc41c492415f0dd6ecd39b0790f84a48e` / `a56b869270f7172bc648220fd2765f3561fa79bec1455cb42a5e5487071be7dd` / `350dcfe94e9e59efc3306888f7d1cd9eeb19648fb51d741c50653ab39c0f964f` |
| `l17-7d716f20` | `b6f11ad87496a6caef7d51baa04f38135efc7fef874aa30bd83a982ff7fd5984` / `807e9e31d6493bef99739285a1b510626fb8d9c300f3f02c14ce170d75c94f2e` / `5d72cce9dafdb8003611a2ed67b6262c506f36a22e744d9591f2ab7b003edead` / `09141ee64b3705490e5c21dc92bf6740260b68a44f78b7f447e9926bd92e3704` |
| `sn5-229f43b0` | `902455a475c3a261475ffba2039ef6b0c92aac0c529dc06e5fea21e11d1a7753` / `9088932abd6f264759b047521c26c87731cbc9de0b58446163e1366fb3521eb1` / `5d8b72c3cc10a38ba75fbd4b6a495c1ee86df8d7375e74cdb8bd8fc74cba48bd` / `6e8ca17fb5a5ff9dd694bde2f4ff8be647b8fddda405db503fb2b9cee39da405` |
| `sn9-ab9ed8eb` | `348eb3acfa7a33353e9207c0819fa0f9119f5ce4f66d7af2471fb5c74d9f2597` / `8a22342971895f4827d0fc744066758aeffff19593599899fbf375c971e0e4b9` / `3f6f08e8c0f3e9aa096e4e2de6a05737031c64566f537e7aebb0fb0db8262fd9` / `bab7a88073c0462cb94dd1c2a8808fbce4acf156d1f4227d2694ec1c0004f145` |
| `sn17-f6ee60e4` | `ff7d733faba9a9143cdd1288cdafef5da336f29d695c04c6793873f854403d14` / `1bc1263d88f4aaaa4a0791abf75cd505a26bd16a49e36838e20a3344357b3a89` / `b4945e7d4894a0b59101779272af692b7833447110b1c9bec7c8dfc8c3a90b5d` / `2ac5a5778cc634336d52b04c187bf2e54debf53210e9efdf2c8eb6a8f5cb5f92` |
| `si5-1ca3b9f3` | `acd848bbf933596cc271ed3751bb74c989a532dd1d70ccc37e1c3b87c882bc92` / `71acef057f3686fd5ab28406da81ca7eb54f8cfb69025c51e0c465628fe9ae7e` / `218b8765504fc8e712857de9b658be75fa4ff15a07dd35948213cd2a2629a391` / `949ed61f5b261e081ee9439e9fda9b16f9b55fd92ed53d54f6aa429cb704e539` |
| `si9-02870841` | `9a878b74ed0849a711aaa2a7f6acd8ff3f81f27ca65a95f92656cf0982eaa64e` / `2e63cfab4fef3933a4b9692f04053784ca69fbbc111aeceefd774197c43614b7` / `f53a0857251242b454b7a16ac4fd0cc155ea45aaa5e6314983a5d616945d9ab7` / `e3807e9599bf86bc8c7a81e021c5c65ab018305fd9a9db060ff20455827d677a` |
| `si17-041fb75f` | `ef5aecfbd745e9de3dc8b213475fce74c6575c92a4e96cd0f9bde1f6259affd4` / `fde9d075ff2c7215e649a0b8ddd8f1fd1919757a08c72ca738f6d83198f34312` / `df7739b2203816f8248cd865467d973b24ab0fe7b04639fdb889994a9b4b7849` / `23355000536561e583e50675c1d1686702cbe03f4fc8fb4bd77d397d5b1de233` |
| `l33-623eb4f6` | `09bf71de0492a465d21629866c95405716ee4ab12fbd612de8fca8524b5a93e3` / `0f6dd77a535892afe9b1d1835565efbca8440c218766bc68f6fba52092a5407f` / `4233d46f83bea5ea547d6ba0de436f2ff9472ff16b84b683c356d57a26b0c730` / `21cb862804420bcc227213ab5d33bbd788498ae89fe7212d1d005debee6a5dc4` |
| `l65-6152078c` | `e500f558178dcb9f7a773c5dca036dcc518e3f28c694a5f7e4d257348bf281eb` / `aba8583755a1149be562f14f83f7bd2db47b2b6c43e54862c26ebffa7054cfad` / `d0dc56b0520c0de3066b95090202aef772151d24ed1f8f6a3d3d53fff676c133` / `83b477605346bb7cd94dfaec72299e6b68fb045ed28cdad5431083e1ab00c285` |
| `l129-f9bdd9ec` | `6e97047f82f32706a765d63fe356c30810b0448025d071f4c06b2a4f0a4a5b30` / `0b3984f213b94a456b501efbb4db3c130d3760d2e913f5063572b1feeff50637` / `ca9243cca8b4faffc51f3c2a93f287b24a3608c8e9b880595be92f916d90be31` / `3e3b766ba60f5e09cdca24e08b673d93b661dda57f30b6fbbc47bbc333a8d50f` |

The normalized Lei model SHA-256 is
`b3dc883b9f58497e7ffa538db3b1151c6e4a2af5095d0f5a86cb06b120eab721`;
its case snapshot SHA-256 is
`5808806f3895af5e3be397c60cfee347d2438b88b54550dd6656e15bd37e8ef7`.
The normalized Song model SHA-256 is
`abaf77507e6f49d576699d42844170256058775e118867cb04e8df5e32d12de8`;
its case snapshot SHA-256 is
`7995a168d4b525e71810afbb1b699233a805dc1ad39e5192584a9a6218813af0`.

No `error_summary.json` exists because the independent-reference gate precedes
SimPEG/FEniCSx comparison. No zero-error shortcut or primary-as-FEniCSx artifact was
created.

## Versions and verification state

- Reference commit: `6c7c836a0774538adafaa40c48dac73a42969c88`, clean in every successful manifest.
- Windows reference environment: CPython `3.12.10`, empymod `2.6.0`, NumPy `2.4.3`, SciPy `1.17.1`, discretize `0.12.0`, SimPEG `0.25.2`, pymatsolver `0.4.0`, PyYAML `6.0.3`.
- WSL executable preflight only: `/home/paidaxin/miniconda3/envs/fenicsx/bin/python --version` returned Python `3.10.20`; it was not used after the reference gate failed.
- Supplied full Windows baseline evidence: `1206 passed, 10 skipped, 164 warnings in 63.76s`.
- Fresh task-related verification: `python -m pytest tests/test_sotem_validation_cli.py tests/test_run_sotem_benchmark.py tests/test_error_metric_floor.py tests/test_sotem_metrics.py` returned `166 passed, 2 skipped in 31.29s`.
- A fresh hash audit re-read all 12 successful manifests and verified every recorded stage-output SHA-256 against the files on disk.

Generated CSV/JSON/mesh/NPZ artifacts remain ignored. Only this summary and the
small CLI-control implementation are committed.

## Follow-up root cause and quasistatic analytical gate

The failed runs above are preserved unchanged, but they are **not eligible as final
references**. Their empymod calls omitted explicit constitutive and Fourier-transform
arguments. Consequently empymod used relative electric permittivity 1 instead of the
quasistatic value 0 and its lagged Fourier DLF default (`pts_per_dec=-1`). FEniCSx and
SimPEG solve the quasistatic problem, so the old runs did not share their governing
equation. Their metadata also lacked a transform/constitutive identity capable of
detecting that mismatch.

The approved replacement identity is fixed independently of source quadrature:

- equation: quasistatic;
- every layer: `epermH=epermV=0` and `mpermH=mpermV=1`;
- Hankel transform: DLF, `key_201_2009`, standard mode (`pts_per_dec=0`);
- Fourier transform: DLF, `key_201_2012`, standard mode (`pts_per_dec=0`).

The ignored diagnostic script
`generated/lei_quasistatic_transform_diag.py` has SHA-256
`a4485cc6fa06b8a0745aad6facacb5356a606599762d61d73b567e2246c8eced`.
For Ex at all 41 observation times, its independent oracle integrates the analytical
diffusive-half-space solution along the finite wire with GL2049. The metric headed
"max" below is `max(abs(candidate-oracle)/abs(oracle))`; L2 is
`norm(candidate-oracle)/norm(oracle)`.

| Standard-DLF quasistatic Ex candidate | max | relative L2 |
|---|---:|---:|
| srcpts 257 | 3.04434e-5 | 7.14062e-6 |
| srcpts 513 | 3.04437e-5 | 7.15652e-6 |
| srcpts 1025 | 3.04437e-5 | 7.15576e-6 |

The direct source-quadrature comparisons use the same pointwise relative metric:
Ex 513 -> 1025 has max `1.14276e-9`, dBzdt 513 -> 1025 has max
`1.71337e-9`, and evaluating Ex at `1e-5 s` alone versus inside the complete time
array differs by `1.28643e-13`. Thus standard Fourier DLF removes the material
time-array dependence seen in the lagged transform. These analytical checks approve
the physical/transform identity; they do not change the separately controlled CLI
default `srcpts=5` or waive the required 5/9/17 source-convergence rerun.

The final independent analytical-curl plateau check covers all 41 times and selects
GL257 with 2 m curl spacing as its oracle. For spatial differencing, 4 m -> 2 m has
peak-normalized maximum `1.4043e-5` and relative L2 `9.67e-6`; 2 m -> 1 m has
peak-normalized maximum `1.9741e-5` and relative L2 `1.354e-5`. For line quadrature,
GL65 -> GL129 has peak-normalized maximum `2.3576e-5` and relative L2 `2.034e-5`;
GL129 -> GL257 has peak-normalized maximum `1.6267e-5` and relative L2 `8.628e-6`.
Against that oracle, layered standard-DLF srcpts 129 with quasistatic permittivity
has peak-normalized maximum `8.1834e-6`, relative L2 `5.8476e-6`, and maximum under
the 1%-of-peak floor metric `7.2277e-4` (`0.0723%`). Keeping empymod's default unit
permittivity instead gives peak-normalized maximum `18.94094%` and relative L2
`4.84656%`. Raw curl cancellation in the weak `0.1 s` tail is not spatially stable;
that sample remains in the comparison but uses the stated 1%-of-peak floor in the
denominator, so unstable raw weak-tail relative error is not substituted for the
strong-signal gate. These differently defined metrics are reported separately and
are not substituted into the original Task 10 gate table.

The production reference path now records this identity in stage inputs, the durable
transaction journal, and `empymod_metadata.json`; resume, recovery, and polarization-
effect source selection require exact agreement and reject legacy evidence with a
missing identity. This follow-up establishes an independent-reference prerequisite
only. It does **not** claim that a FEniCSx or SimPEG forward response has been run or
validated.

## Clean-provenance source-quadrature rerun

The original failure evidence above is retained unchanged. A later rerun from the
clean detached WSL clone `/home/paidaxin/codex-sotem-clean-20260720-i30Hx9` at
commit `55b6389c2234cf86f86cabbd57c8fae97bdcb980` verified the approved quasistatic
identity and the fixed `srcpts=9 -> 17` source-quadrature gate. The audit found an
empty Git porcelain status and independently checked every run's commit, working
directory, Python executable, case/model hashes, reference identity, metadata,
manifest state, transaction inputs, and recorded stage-output hashes.

The local audit evidence is
`generated/t10q_clean/reference_audit.json`, SHA-256
`62566930591df1b2a26eacb4d565ac13fb3a49cb2f60dd7129a717b088932e1f`.
It was produced with WSL Python `3.10.20`, empymod `2.5.4`, NumPy `2.2.6`, and
SciPy `1.15.2`. The fixed threshold remained `0.005`, with the denominator floor
`0.01 * peak(abs(srcpts-17 reference))` per component.

| Clean run pair | Ex max | Hz max | dBzdt max | Result |
|---|---:|---:|---:|---|
| Lei noIP, 9 -> 17 | 1.4850153399660493e-7 | 1.47790257094914e-7 | 2.531256148552455e-7 | PASS |
| Song noIP, 9 -> 17 | 3.2668459368876505e-7 | 2.858960581161179e-7 | 3.638300895863172e-6 | PASS |
| Song exact Cole-Cole IP, 9 -> 17 | 2.9704396822899094e-4 | 3.015777705142679e-7 | 3.6385732025732505e-6 | PASS |

All three fixed 9 -> 17 comparisons pass. The old low-order default is not safe:
the earlier quasistatic audit's Song exact Cole-Cole IP 5 -> 9 comparison gave
`Ex=0.15991651860664247`, far above the same `0.005` threshold. Therefore the
pipeline now uses primary `srcpts=9` with a mandatory higher-order `srcpts=17`
acceptance audit, while the standalone validation CLI publishes the high-order
`srcpts=17` reference directly.

For a formal pipeline forward, both references and the fixed acceptance gate are
evaluated after the source-only exit and before either E- or H-form time stepping.
The gate requires all of `Ex`, `Hz`, and `dBzdt`; a missing component or a component
above `0.005` fails closed before any forward partial, checkpoint, response NPZ,
figure, or acceptance artifact can be written. A passing run reuses the references
already computed for the full requested schedule, selecting completed rows for a
segmented run instead of recomputing them after the solver.

This clean rerun computed empymod references only. It did **not** run a FEniCSx or
SimPEG forward solve, and it does not claim either solver has passed response
validation.
