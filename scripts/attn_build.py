#!/usr/bin/env python3
"""M3 Triton DSL build provider for scripts/build_dsl.py (--milestone M3)."""

from __future__ import annotations


def specifications(load_source):
    rel = load_source("rel_attention")
    return [
        {
            "name": name,
            "milestone": "M3",
            "kernel": kernel,
            "signature": sig,
            "num_warps": warps,
            "constants": constants,
            "options": {"num_warps": warps, "num_stages": stages, "enable_fp_fusion": fp_fusion},
        }
        for name, kernel, sig, constants, warps, stages, fp_fusion in rel.get_dsl_specs()
    ]
