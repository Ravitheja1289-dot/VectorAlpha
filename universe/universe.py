"""
Static universe selection for early-stage portfolio mechanics testing.

Purpose:
- Provide a fixed, liquid, tech-heavy list of US large caps.
- Avoid dynamic selection and alpha assumptions; focus on mechanics.

Notes:
- These symbols are widely traded and suitable for simulations.
- No data fetching or screening logic is included at this stage.
"""

from typing import List

__all__ = ["get_universe", "DEFAULT_UNIVERSE"]


# 15 highly liquid, predominantly tech/large-cap US equities
DEFAULT_UNIVERSE: List[str] = [
	"AAPL",  # Apple
	"MSFT",  # Microsoft
	"AMZN",  # Amazon
	"GOOGL", # Alphabet Class A
	"META",  # Meta Platforms
	"NVDA",  # NVIDIA
	"TSLA",  # Tesla
	"NFLX",  # Netflix
	"ORCL",  # Oracle
	"CRM",   # Salesforce
	"ADBE",  # Adobe
	"INTC",  # Intel
	"AMD",   # Advanced Micro Devices
	"QCOM",  # Qualcomm
	"CSCO",  # Cisco
]


def get_universe() -> List[str]:
	"""
	Return a copy of the static universe.

	This function intentionally avoids any dynamic logic (e.g., liquidity
	screens, sector filters, or alpha models). It is designed for
	validating portfolio accounting, execution, and risk mechanics.
	"""
	return DEFAULT_UNIVERSE.copy()

