# LAYERED_INDEPENDENT_TEST_REPORT

Official 10-case L2 with W0-W3, point, disk_1.0, disk_4.0, and tilted-coil projection.
B0 is not defined in protocol.md and was not invented. B1 is the frozen log-uniform time-window template `K{K}_tw_span8.0_shift+0.0_dens1.00` when present in the shortlist.

- L2 status: `L2_PASS`
- passed A: `True`
- passed B: `False`
- n cases: `10`
- best same-K P-R/B2 median ratio: `0.382566169465141` at K=`10`
- bootstrap 95% CI: `[0.163401575828889, 0.7770330921368851]`
- win rate: `0.9`
- median K_qual_B2 - K_qual_PR: `0.0`
- nonnegative K_qual rate: `1.0`
- group_ok: `{'waveforms': True, 'receivers': True, 'components': True, 'ip': True}`
- pr_by_k: `{'10': 'K10_cc_span4.0_shift-0.5_dens1.00', '12': 'K12_cc_span6.0_shift+0.0_dens1.25', '4': 'K04_cc_span4.0_shift-0.5_dens1.25', '6': 'K06_cc_span4.0_shift+0.0_dens1.25', '8': 'K08_cc_span4.0_shift-0.5_dens1.00'}`
- b2_by_k: `{'10': 'K10_tw_span6.0_shift+0.5_dens1.25', '12': 'K12_tw_span6.0_shift+0.5_dens1.25', '4': 'K04_cc_span4.0_shift-0.5_dens1.25', '6': 'K06_tw_span4.0_shift+0.5_dens1.25', '8': 'K08_cc_span4.0_shift-0.5_dens1.00'}`
- reselection: `False`
- bootstrap: 2000 case-level paired resamples, seed 202609116
- OR: upper bound only (not a gate input; see or_upper_bound_pr13.json)

## L2 A/B checklist (P-R vs frozen B2; OR is upper bound only)

- A1 median P-R/B2 ratio <= 0.80: `0.382566169465141` -> `True`
- A2 bootstrap 95% CI upper < 1.00: `0.7770330921368851` -> `True`
- A3 P-R better than B2 in >= 70% of cases: `0.9` -> `True`
- A4 improvement on at least two waveforms: `True`
- A5 improvement on point AND at least one disk: `True`
- A6 H or dB/dt clearly improved, other not worse >10%: `True`
- A7 IP-increment p95 not worse by >5%: `True`
- A overall: `True`
- B1 median(K_qual_B2 - K_qual_P-R) >= 2: `0.0` -> `False`
- B2 nonnegative qualifying-K difference in >= 70% of cases: `1.0` -> `True`
- B overall: `False`

## Same-K A rows

- K=6: median_ratio=0.4868492097268449, CI=[0.22354483741659018, 1.3283459552678822], win_rate=0.7, n=10
- K=8: median_ratio=1.0, CI=[1.0, 1.0], win_rate=0.0, n=10
- K=10: median_ratio=0.382566169465141, CI=[0.163401575828889, 0.7770330921368851], win_rate=0.9, n=10
- K=12: median_ratio=0.7866275346261127, CI=[0.2855577228750221, 1.058884411555189], win_rate=0.7, n=10

## OR upper bound (not a gate input)

Receiver-oracle upper bound from PR13 independent_test JSON (point tasks, official reduce_case_error). Not used for the L2 gate. Frozen P-R/B2 from L1.

- TE01: K=4 OR/B2=1.0 P-R/B2=1.0; K=6 OR/B2=0.4895325600196173 P-R/B2=0.4895325600196173; K=8 OR/B2=0.8872349477981988 P-R/B2=1.0; K=10 OR/B2=0.10619296938391602 P-R/B2=0.29293657429582737; K=12 OR/B2=0.2819787110416383 P-R/B2=0.9090713964405378
- TE02: K=4 OR/B2=1.0 P-R/B2=1.0; K=6 OR/B2=0.062457962871123214 P-R/B2=0.3690310195211588; K=8 OR/B2=1.0 P-R/B2=1.0; K=10 OR/B2=0.09865847244617802 P-R/B2=0.10936905049202868; K=12 OR/B2=0.07508068432699094 P-R/B2=0.17686130008076473
- TE03: K=4 OR/B2=0.8510587638851077 P-R/B2=1.0; K=6 OR/B2=0.6612193735515658 P-R/B2=1.6148081334972286; K=8 OR/B2=0.6259471078106144 P-R/B2=1.0; K=10 OR/B2=0.6503723624684751 P-R/B2=0.8406538133620948; K=12 OR/B2=1.0 P-R/B2=1.7209448149839337
- TE04: K=4 OR/B2=0.7755948384665562 P-R/B2=1.0; K=6 OR/B2=0.49157833376691706 P-R/B2=0.49157833376691706; K=8 OR/B2=0.4815912964351779 P-R/B2=1.0; K=10 OR/B2=0.12485153873677933 P-R/B2=0.7364745406918063; K=12 OR/B2=0.24196225325526144 P-R/B2=0.6313395949777991
- TE05: K=4 OR/B2=0.5251486329270605 P-R/B2=1.0; K=6 OR/B2=0.08115883678460833 P-R/B2=0.08115883678460833; K=8 OR/B2=0.3317128086391213 P-R/B2=1.0; K=10 OR/B2=0.05162722994334352 P-R/B2=0.1659506391348824; K=12 OR/B2=0.02608169087920709 P-R/B2=0.12555554922048037
- TE06: K=4 OR/B2=1.0 P-R/B2=1.0; K=6 OR/B2=0.9190794975382859 P-R/B2=1.2399619382663012; K=8 OR/B2=0.9253906504601275 P-R/B2=1.0; K=10 OR/B2=1.0 P-R/B2=1.5714062326285203; K=12 OR/B2=0.8486738957658171 P-R/B2=0.8486738957658171
- TE07: K=4 OR/B2=0.3941922544632639 P-R/B2=1.0; K=6 OR/B2=0.2099472545564428 P-R/B2=0.22015025464659593; K=8 OR/B2=0.8784520710214567 P-R/B2=1.0; K=10 OR/B2=0.03852026384498526 P-R/B2=0.07297106884782217; K=12 OR/B2=0.040056196188034826 P-R/B2=1.2862305526483855
- TE08: K=4 OR/B2=0.7685839655981472 P-R/B2=1.0; K=6 OR/B2=0.9498333920785772 P-R/B2=1.3133536535185721; K=8 OR/B2=0.6423878682247008 P-R/B2=1.0; K=10 OR/B2=0.6185605814317483 P-R/B2=1.3781087213175593; K=12 OR/B2=0.8826356771231136 P-R/B2=0.8826356771231136

- median OR/B2 across 8 cases × K: `0.6222538446211814` (n=40; upper bound only)

3-D was not run.
