"""Preparation of the 2021 reference inputs in ``data/external/baseline_2021/``.

Run in this order (each step reads the previous one's output):

1. ``books_2021`` -- the three prepared 2021 books (UN Comtrade, Eurostat, HMRC) from raw files.
2. ``uk_index`` -- the UK's 2021 baseline-index levels, ``uk_index_2021.csv``.
3. ``history_2021`` -- ``measured_history.csv.gz``, the measured books' 2021 months, with the
   UK's imports from the EU reconstructed (``uk_reconstruction``).
"""
