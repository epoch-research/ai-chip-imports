"""Estimate where AI accelerators arrive, from national customs records.

The pipeline isolates the AI-attributable slice of customs trade in servers and accelerator
boards, converts it into accelerator dollars per destination, compares the total with Epoch
AI's Chip Sales data, and distributes Epoch's per-model sales across destinations in proportion
to those dollars. The method is documented at
https://epoch.ai/data/ai-chip-imports-documentation/methodology;
docs/data.md lists the inputs.

Run ``python -m ai_chip_imports`` (estimator and allocator) or ``python -m ai_chip_imports.sources``
(raw customs files to clean CSVs).
"""

__version__ = "1.0.0"
