# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Compact, bounded measurement buffer; disk writes occur after hardware shutdown."""

import math

import numpy as np


class BufferedRecording:
    """Preallocated float64 samples at 100 Hz, with no per-pass serialization."""

    def __init__(self, fields, seconds):
        if not math.isfinite(seconds) or not 0 < seconds <= 600:
            raise ValueError("Buffered recording duration must be in (0, 600] seconds")
        self.fields = tuple(fields)
        self.samples = np.empty((math.ceil(seconds * 100) + 200, len(fields)), dtype=np.float64)
        self.count = 0

    def append(self, values):
        if len(values) != len(self.fields):
            raise ValueError("Recording fields do not match sample values")
        if self.count == len(self.samples):
            raise RuntimeError("Recording buffer is full")
        self.samples[self.count] = values
        self.count += 1

    def write_to(self, writer):
        """Write all recorded passes after pump/output shutdown."""
        for sample in self.samples[: self.count]:
            row = dict(zip(self.fields, sample.tolist(), strict=True))
            row["armed"] = bool(row["armed"])
            for key in ("pass_index", "device_ts_us"):
                if key in row:
                    row[key] = int(row[key])
            writer.writerow(row)
