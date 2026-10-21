"""The HS code basket, per-code constants and period helpers.

AI accelerators cross borders in one of a few forms, each under its own HS-6 code:

    847149  data-processing systems entered as such
    847150  processing units (servers, GPU baseboards / HGX trays)
    847180  other data-processing units
    847330  parts and accessories of data-processing machines (accelerator cards, boards)
    854231  processors and controllers (bare chips)

The first four are "systems" and are counted; bare chips are carried through the loaders but
count toward no destination: accelerator silicon makes its last border crossing as a board or
a system, so a bare-chip crossing is either mid-supply-chain (and the same silicon is counted
again as a board) or non-AI trade.
"""

SYSTEMS = ["847149", "847150", "847180", "847330"]
CHIPS = ["854231"]
BASKET = SYSTEMS + CHIPS
# Machine codes: whole systems, whose content share follows the accelerator product mix.
SERVER_CODES = ["847149", "847150", "847180"]

# First quarter of the AI window used for summaries and pooled priors.
AI_LO = "2023Q3"

# Parametric non-AI growth per year, used where no measured index exists: systems 5% (ordinary
# servers and PCs), bare chips 3%.
NONAI_GROWTH = {"systems": 0.05, "chips": 0.03}

# Flat accelerator-content share of border value per code: an 8-GPU server's GPUs are ~65% of
# its price, a GPU board is nearly all GPU, a bare chip is all chip. Used as a fallback and for
# the low-content ("mfg") band member; measured content replaces it wherever it exists.
CHIP_CONTENT = {"847149": 0.65, "847150": 0.65, "847180": 0.65, "847330": 0.90, "854231": 1.00}


def quarter(period):
    """'YYYY-MM' -> 'YYYYQn'."""
    y, m = period[:4], int(period[5:7])
    return f"{y}Q{(m - 1) // 3 + 1}"


def channel(code):
    """'chips' for bare chips, 'systems' for every other code (selects the growth rate)."""
    return "chips" if code in CHIPS else "systems"


def years_since_mid2022(period):
    """Years from mid-2022 to the middle of month ``period`` (for parametric growth)."""
    y, m = int(period[:4]), int(period[5:7])
    return ((y - 2022) * 12 + m - 6.5) / 12.0
