"""Uncertainty of the country estimates: parameter and sales draws within structural cases.

The estimate can be wrong in two ways: its inputs can be off (sales volumes, prices, reference
levels, the imported content of exports), and the trade can be misread (expensive memory read
as accelerator boards, an average server price hiding very different products). The run
addresses the first with random draws and the second with structural cases, and reports them
separately:

* **Draws** (``sampling``): each varies the baseline reference, k, the classifier's thresholds,
  selected content readings and Epoch's sales and prices, and reruns the whole pipeline --
  netting, composite reference, ledger, content, estimator and allocation (``draw``).
* **Cases** (``cases``): 144 alternative readings of the trade -- references, server mixtures,
  selection rules, composition stresses, the Malaysia deduction -- each held fixed over the whole
  history; every case runs the same draws.

The conditional 90% interval is the central case's 5th-95th percentile; the structural envelope
spans the lowest case 5th to the highest case 95th percentile, with no probability attached to
the cases (``report``). Every draw allocates exactly its own sampled supply, so world totals
share one interval across cases.

Usage (see ``__main__``)::

    python -m ai_chip_imports.uncertainty prepare
    python -m ai_chip_imports.uncertainty probe
    python -m ai_chip_imports.uncertainty run --samples 64 --workers 6
    python -m ai_chip_imports.uncertainty report

With 64 draws per case each 5% tail rests on about three draws: endpoints are approximate, and
the report states their sensitivity to the sample. A full run of 9,216 draws takes about a day
on six cores.
"""
