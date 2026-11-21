#!/usr/bin/env python3
"""M3 Triton DSL build provider for scripts/build_dsl.py (--milestone M3)."""

from __future__ import annotations


def specifications(load_source):
    rel = load_source("rel_attention")
    out = []
    for spec in rel.get_dsl_specs():
        name, kernel, sig, constants, warps, stages, fp_fusion = spec[:7]
        entry = {
            "name": name,
            "milestone": "M3",
            "kernel": kernel,
            "signature": sig,
            "num_warps": warps,
            "constants": constants,
            "options": {"num_warps": warps, "num_stages": stages, "enable_fp_fusion": fp_fusion},
        }
        if len(spec) > 7 and spec[7]:
            entry["attrs"] = spec[7]
        out.append(entry)
    return out
