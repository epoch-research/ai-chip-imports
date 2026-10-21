"""The HS code basket, per-code constants and period helpers."""
import pandas as pd
import pytest

from ai_chip_imports import codes


@pytest.mark.parametrize("period, expected", [("2024-01", "2024Q1"), ("2024-03", "2024Q1"),
                                              ("2024-04", "2024Q2"), ("2024-09", "2024Q3"),
                                              ("2024-10", "2024Q4"), ("2024-12", "2024Q4")])
def test_quarter_maps_months_to_calendar_quarters(period, expected):
    """A YYYY-MM month maps to its calendar quarter, matching pandas' quarterly periods."""
    assert codes.quarter(period) == expected
    assert codes.quarter(period) == str(pd.Period(period, freq="M").asfreq("Q"))


def test_channel_separates_bare_chips_from_systems():
    """Bare chips take the chips growth channel; every other basket code the systems channel."""
    assert codes.channel("854231") == "chips"
    assert {codes.channel(c) for c in codes.SYSTEMS} == {"systems"}
    assert {codes.channel(c) for c in codes.BASKET} == set(codes.NONAI_GROWTH)


def test_years_since_mid2022_is_centred_between_june_and_july_2022():
    """Mid-2022 lies half a month before mid-July; twelve months add exactly one year."""
    assert codes.years_since_mid2022("2022-07") == pytest.approx(0.5 / 12)
    assert codes.years_since_mid2022("2022-06") == pytest.approx(-0.5 / 12)
    assert codes.years_since_mid2022("2024-07") - codes.years_since_mid2022("2023-07") == pytest.approx(1.0)


def test_basket_is_systems_plus_chips_with_a_content_share_per_code():
    """The basket partitions into systems and chips, machines are systems, and every code has a content share."""
    assert codes.BASKET == codes.SYSTEMS + codes.CHIPS
    assert not set(codes.SYSTEMS) & set(codes.CHIPS)
    assert set(codes.SERVER_CODES) < set(codes.SYSTEMS)
    assert set(codes.CHIP_CONTENT) == set(codes.BASKET)
    assert all(0 < v <= 1 for v in codes.CHIP_CONTENT.values())
    assert codes.quarter(str(pd.Period(codes.AI_LO, freq="Q").asfreq("M", "start"))) == codes.AI_LO
