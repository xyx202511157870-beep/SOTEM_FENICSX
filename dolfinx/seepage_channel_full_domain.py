#!/usr/bin/env python3
"""Full-domain FEniCSx adapter for the seepage-channel benchmark.

The adapter intentionally contains no half-domain or receiver-mirroring path.
Pure geometry/material and receiver-set helpers are added here as their
test-driven implementation tasks are completed.
"""

from __future__ import annotations
