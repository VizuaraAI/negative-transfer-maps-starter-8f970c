"""Negative Transfer Maps Under Capacity Pressure.

Measuring which training domains damage each other when model capacity is
scarce, and whether that damage disappears with scale.

Module map:
    config.py       capacity presets, mixture specs, token budgets
    data.py         the eight-domain registry and streaming token pipeline
    model.py        a plain decoder-only transformer with a width/depth knob
    train.py        train one cell of the capacity x mixture grid
    evaluate.py     score a checkpoint on all eight domains
    transfer.py     assemble runs into the K x K interference map, fit decay
    diagnostics.py  gradient conflict, CKA, polysemanticity on existing weights
    plotting.py     the paper's two figures, straight from the report JSON
"""

__version__ = "0.1.0"

DOMAIN_COUNT = 8
