def money(value: float) -> str:
    return f"${value:,.0f}"


def number(value: float, digits: int = 2) -> str:
    return f"{value:,.{digits}f}"


def percent(value: float, digits: int = 1) -> str:
    return f"{value * 100:,.{digits}f}%"


def p_value(value: float) -> str:
    if value < 0.001:
        return "<0.001"
    return f"{value:.3f}"
