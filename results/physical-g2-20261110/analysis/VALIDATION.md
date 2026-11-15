# Analysis validation

The original evidence archive and its inventory are unchanged. The reproduction script verified SHA-256 and byte counts for all 330 archived files, every block's bound raw/profile/network/GPU artifacts, and all nine controlled profile gates. It analyzed exactly 8,100 successful measured rows and excluded 27 warmups. Every additive per-request budget closes to the original observed latency; maximum absolute closure error was 0 ms at stored precision.

The independently saved run summaries match all 27 newly computed mean latency budgets. Five analysis checks passed, including synthetic exclusive-stage accounting, rejection of a negative transfer residual, preservation of the cloud combined stage, a known constant paired bootstrap, and agreement with the archived summaries. The checks deliberately prevent counting server processing twice through the client receive wait.

Commands used from the repository root:

```bash
MPLCONFIGDIR=/tmp/g2-matplotlib /usr/bin/python3 \
  results/physical-g2-20261110/analysis/analyze_stages.py
MPLCONFIGDIR=/tmp/g2-matplotlib /usr/bin/python3 -m unittest discover \
  -s results/physical-g2-20261110/analysis -v
.venv/bin/python -m ruff check results/physical-g2-20261110/analysis/*.py --no-cache
```

All 21 generated tables, JSON outputs, report and figure files reproduced byte-for-byte on a second invocation under the recorded software versions. Local report links were checked and the stage/placement figures were visually inspected. No GPU/cloud access or inference rerun was involved.

This validates the analysis against the preserved evidence, not the unmeasured scientific claims: cloud-only visual sub-stages, isolated one-way network delay, queue time, shared-memory page-fault causality, official accuracy, live policy switching cost and held-out adaptive gains remain outside what G2 identifies. Confidence intervals resample request IDs conditional on this campaign; they do not substitute for independent host/network-day replications.
