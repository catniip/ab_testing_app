from __future__ import annotations


def validate_constraints(min_line: int, max_line: int, increment: int) -> list[str]:
    errors = []
    if min_line >= max_line:
        errors.append("Minimum Credit Line must be lower than Maximum Credit Line.")
    if increment <= 0:
        errors.append("Allowed Credit Line Increment must be greater than zero.")
    return errors


def validate_test_lines(lines: list[int], min_line: int, max_line: int, increment: int) -> list[str]:
    errors = []
    if not lines:
        return ["At least one test line is required."]
    if len(lines) != len(set(lines)):
        errors.append("Duplicate credit lines are not allowed.")
    for line in lines:
        if line < min_line or line > max_line:
            errors.append(f"{line:,.0f} is outside the allowed credit-line range.")
        if (line - min_line) % increment != 0:
            errors.append(f"{line:,.0f} does not follow the allowed increment.")
    return errors
