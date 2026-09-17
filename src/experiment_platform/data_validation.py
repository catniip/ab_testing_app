from __future__ import annotations

import pandas as pd
import warnings


def deduplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    seen: dict[str, int] = {}
    names = []
    for column in df.columns:
        name = str(column).strip()
        count = seen.get(name, 0)
        names.append(name if count == 0 else f"{name}_{count + 1}")
        seen[name] = count + 1
    df.columns = names
    return df


def first_series(df: pd.DataFrame, column: str | None) -> pd.Series | None:
    if column is None or column not in df.columns:
        return None
    selected = df[column]
    if isinstance(selected, pd.DataFrame):
        return selected.iloc[:, 0]
    return selected


def safe_numeric_series(df: pd.DataFrame, column: str | None) -> pd.Series:
    series = first_series(df, column)
    if series is None:
        return pd.Series(dtype="float64")
    return pd.to_numeric(series, errors="coerce")


def safe_datetime_series(df: pd.DataFrame, column: str | None) -> pd.Series:
    series = first_series(df, column)
    if series is None:
        return pd.Series(dtype="datetime64[ns]")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return pd.to_datetime(series, errors="coerce")


def date_like_columns(df: pd.DataFrame, minimum_valid_share: float = 0.8) -> list[str]:
    columns = []
    for column in df.columns:
        series = first_series(df, column)
        if series is None or pd.api.types.is_numeric_dtype(series):
            continue
        non_null = series.dropna()
        if not non_null.empty and safe_datetime_series(df, column).dropna().size / len(non_null) >= minimum_valid_share:
            columns.append(column)
    return columns


def normalize_uploaded_dataset(df: pd.DataFrame) -> pd.DataFrame:
    return deduplicate_columns(df)


def infer_column(columns: list[str], candidates: list[str], fallback: str = "") -> str:
    lowered = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return fallback or (columns[0] if columns else "")


def infer_mob_column(df: pd.DataFrame) -> str:
    columns = list(df.columns)
    named = infer_column(columns, ["mob", "months_on_book", "month_on_book"])
    if named:
        return named
    best = ""
    best_valid = -1
    for column in columns:
        values = safe_numeric_series(df, column)
        valid = values.dropna()
        if valid.empty:
            continue
        integer_like = ((valid % 1).abs() < 1e-9).mean()
        in_range = valid.between(0, 120).mean()
        score = int(len(valid) * integer_like * in_range)
        if score > best_valid:
            best = column
            best_valid = score
    return best


def infer_schema(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in df.columns:
        series = first_series(df, column)
        numeric = safe_numeric_series(df, column)
        parsed_numeric = int(numeric.notna().sum())
        parsed_date = int(safe_datetime_series(df, column).notna().sum())
        non_null = int(series.notna().sum()) if series is not None else 0
        if non_null and parsed_numeric / non_null > 0.9:
            suggested = "Numeric"
        elif non_null and parsed_date / non_null > 0.8:
            suggested = "Date"
        else:
            suggested = "Categorical"
        unique = int(series.nunique(dropna=True)) if series is not None else 0
        rows.append(
            {
                "Column": column,
                "Detected Type": str(series.dtype) if series is not None else "unknown",
                "Non-Null Rows": non_null,
                "Unique Values": unique,
                "Suggested Type": suggested,
            }
        )
    return pd.DataFrame(rows)


def data_quality_warnings(df: pd.DataFrame, unit_col: str, cpc_col: str, mob_col: str) -> list[str]:
    warnings = []
    if unit_col not in df.columns:
        warnings.append("Experimental Unit ID mapping is required.")
    if cpc_col and cpc_col not in df.columns:
        warnings.append("The selected population segment column is not present.")
    mob = safe_numeric_series(df, mob_col)
    if mob.empty or mob.notna().sum() == 0:
        warnings.append("We could not interpret the selected MOB column as numeric. Review the column mapping.")
    elif mob.isna().sum() > 0:
        warnings.append(f"{mob.isna().sum():,.0f} rows have missing or invalid MOB values.")
    return warnings
