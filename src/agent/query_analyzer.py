import re
from typing import Any, Dict, List


def analyze_query(user_query: str) -> Dict[str, Any]:
    """Extract entities, periods, and simple intent from the user query."""
    entities: List[str] = []
    periods: List[str] = []

    if "revenue" in user_query.lower():
        entities.append("revenue")
    if "profit" in user_query.lower():
        entities.append("profit")
    if "margin" in user_query.lower():
        entities.append("margin")

    year_match = re.findall(r"\b(19|20)\d{2}\b", user_query)
    periods.extend(year_match)

    return {
        "entities": entities,
        "periods": periods,
        "raw_query": user_query,
    }