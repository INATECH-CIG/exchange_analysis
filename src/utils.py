"""
Project: European Electricity Exchange Analysis
Author: Tiernan Buckley
Year: 2026
License: Creative Commons Attribution 4.0 International (CC BY 4.0)
Source: https://github.com/INATECH-CIG/exchange_analysis

Description:
Manages robust database and CSV file I/O operations, handles system logging,
and executes heuristics-based gap filling for missing or anomalous time-series
data.
"""

import time
import pandas as pd
import numpy as np
import sys
import logging
from pathlib import Path
from typing import Dict, Optional, List, Tuple, Any, Callable, Union
from sqlalchemy import text, inspect

from config import get_db_engine

logger = logging.getLogger(__name__)

# ==========================================
# GAP AUDITING HELPERS
# ==========================================
def _record_gap_method(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, method: str, col_name: str = "ROW") -> None:
    """Appends the specified imputation methodology to the metadata audit trail for a given temporal range."""
    if "gap_filling_method" not in df.columns:
        df["gap_filling_method"] = "None"
        
    mask = (df.index >= start) & (df.index <= end)
    tagged_method = f"[{col_name}] {method}"
    
    none_mask = mask & (df["gap_filling_method"] == "None")
    df.loc[none_mask, "gap_filling_method"] = tagged_method
    
    exist_mask = mask & (df["gap_filling_method"] != "None")
    
    def append_if_missing(current: str) -> str:
        return current if tagged_method in str(current) else f"{current}, {tagged_method}"
        
    df.loc[exist_mask, "gap_filling_method"] = df.loc[exist_mask, "gap_filling_method"].apply(append_if_missing)

def _merge_gap_methods(df_target: pd.DataFrame, df_source: pd.DataFrame) -> None:
    """Consolidates metadata strings when combining parallel datasets to maintain a unified audit trail."""
    if "gap_filling_method" not in df_source.columns: return
    if "gap_filling_method" not in df_target.columns:
        df_target["gap_filling_method"] = "None"
        
    valid_methods = df_source.loc[(df_source["gap_filling_method"] != "None") & df_source["gap_filling_method"].notna(), "gap_filling_method"]
    
    for t, method in valid_methods.items():
        if t in df_target.index:
            curr = df_target.at[t, "gap_filling_method"]
            if curr == "None":
                df_target.at[t, "gap_filling_method"] = method
            elif method not in str(curr):
                df_target.at[t, "gap_filling_method"] = f"{curr}, {method}"

# ==========================================
# DATA I/O HANDLER
# ==========================================
class DataIO:
    """
    Centralized Data Input/Output Handler.
    Orchestrates dual-writing to local flat CSV files and a configured database instance.
    """
    def __init__(self, config: Any) -> None:
        self.save_db = getattr(config, 'save_db', False)
        self.load_source = getattr(config, 'load_source', 'csv')

        if self.save_db or self.load_source == 'db':
            self.engine = get_db_engine()
        else:
            self.engine = None
            logger.info("[IO] Running in CSV-only mode. No database engine initialized.")

    def save(self, df: Optional[Union[pd.DataFrame, pd.Series]], filepath: Path, table_name: str, config: Any, bz: Optional[str] = None) -> None:
        """Persists structural arrays to defined storage mediums, handling schema evolution."""
        if df is None or df.empty: return

        df_out = df.to_frame() if isinstance(df, pd.Series) else df.copy()
        
        if bz is not None:
            df_out["bidding_zone"] = bz

        # Route metadata structures based on whether the data is raw extraction or downstream analysis
        is_result_table = table_name.startswith(("analysis_", "tracing_", "pool_", "annual_", "processed_"))
        
        if is_result_table:
            date_val = getattr(config, 'analysis_source_date', pd.Timestamp.utcnow().strftime('%Y-%m-%d'))
            df_out["source_download_date"] = date_val
            meta_cols = ["gap_filling_method", "bidding_zone", "source_download_date"]
        else:
            df_out["download_timestamp"] = pd.Timestamp.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
            meta_cols = ["gap_filling_method", "bidding_zone", "download_timestamp"]

        # Enforce column order to maintain tabular consistency
        data_cols = [c for c in df_out.columns if c not in meta_cols]
        present_meta = [c for c in meta_cols if c in df_out.columns]
        df_out = df_out[data_cols + present_meta]

        # 1. Execute local flat-file persistence
        if getattr(config, 'save_csv', True):
            filepath.parent.mkdir(parents=True, exist_ok=True)
            df_out.to_csv(filepath)

        # 2. Execute relational database persistence
        if getattr(config, 'save_db', True):
            clean_table = table_name.lower().replace("-", "_").replace(" ", "_")[:63]

            if bz is not None:
                if isinstance(df.index, pd.DatetimeIndex):
                    min_time = df.index.min().strftime('%Y-%m-%d %H:%M:%S%z')
                    max_time = df.index.max().strftime('%Y-%m-%d %H:%M:%S%z')
                    index_col = df.index.name or 'index'

                    delete_query = text(f"""
                        DELETE FROM {clean_table}
                        WHERE bidding_zone = '{bz}'
                        AND "{index_col}" >= '{min_time}'
                        AND "{index_col}" <= '{max_time}'
                    """)
                else:
                    delete_query = text(f"""
                        DELETE FROM {clean_table}
                        WHERE bidding_zone = '{bz}'
                    """)

                with self.engine.begin() as conn:
                    try:
                        conn.execute(delete_query)
                    except Exception:
                        pass 

            # Evaluate and apply dynamic schema evolution
            try:
                inspector = inspect(self.engine)
                if inspector.has_table(clean_table):
                    existing_cols = [col['name'] for col in inspector.get_columns(clean_table)]
                    new_cols = [c for c in df_out.columns if c not in existing_cols]

                    if new_cols:
                        with self.engine.begin() as conn:
                            for c in new_cols:
                                col_type = "TEXT" if c in meta_cols else "DOUBLE PRECISION"
                                conn.execute(text(f'ALTER TABLE {clean_table} ADD COLUMN "{c}" {col_type}'))
            except Exception as e:
                logger.warning(f"[DB Schema Warning] Could not auto-evolve schema for {clean_table}: {e}")

            try:
                df_out.to_sql(clean_table, self.engine, if_exists="append")
            except Exception as e:
                logger.error(f"[DB Error] Failed to save {clean_table} to database: {e}")

    def load(self, filepath: Path, table_name: str, config: Any, bz: Optional[str] = None) -> Optional[pd.DataFrame]:
        """Retrieves stored datasets based on configured storage preference (CSV vs DB)."""
        source = getattr(config, 'load_source', 'csv')
        start_str = config.start.strftime('%Y-%m-%d %H:%M:%S%z')
        end_str = config.end.strftime('%Y-%m-%d %H:%M:%S%z')

        if source == 'db':
            clean_table = table_name.lower().replace("-", "_").replace(" ", "_")[:63]
            try:
                base_query = f'SELECT * FROM {clean_table} WHERE "index" >= \'{start_str}\' AND "index" <= \'{end_str}\''
                query = f"{base_query} AND bidding_zone = '{bz}'" if bz is not None else base_query

                df = pd.read_sql(text(query), self.engine)
                if df.empty:
                    raise ValueError(f"No data found in DB for {clean_table} (bz={bz})")

                index_col = str(df.columns[0])
                df.set_index(index_col, inplace=True)
                df.index = pd.to_datetime(df.index, utc=True)
                df.index.name = None

                if bz is not None and "bidding_zone" in df.columns:
                    df = df.drop(columns=["bidding_zone"])

                df.dropna(axis=1, how='all', inplace=True)
                return df

            except Exception as e:
                logger.warning(f"[DB Warning] Falling back to CSV for {clean_table} (bz={bz}). Reason: {e}")

        if filepath.exists():
            df = pd.read_csv(filepath, index_col=0)
            df.index = pd.to_datetime(df.index, utc=True)
            mask = (df.index >= config.start) & (df.index <= config.end)
            return df.loc[mask]

        return None

# ==========================================
# LOGGING & API UTILS 
# ==========================================
def setup_logging(log_file_path: Path, log_level_str: str, debug_mode: bool) -> None:
    """Configures standard output and file-based logging streams with conditional formatting."""
    numeric_level = getattr(logging, log_level_str.upper(), logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s | %(levelname)-8s | %(name)s | %(message)s')
    console_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s' if not debug_mode 
        else '%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s'
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    if root_logger.hasHandlers(): root_logger.handlers.clear()

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    root_logger.info(f"Logging initialized. Level: {log_level_str} | Debug Mode: {debug_mode}")

def safe_query(func: Callable, max_retries: int = 3, delay: int = 2, context: Optional[str] = None, **kwargs: Any) -> Any:
    """Executes API queries with exponential backoff and localized error handling to mitigate transient failures."""
    for attempt in range(max_retries):
        try:
            return func(**kwargs)
        except Exception as e:
            error_msg = str(e)
            if not error_msg.strip(): error_msg = repr(e)
            if hasattr(e, 'response') and e.response is not None:
                error_msg += f" | API Response: {e.response.text}"

            msg = f"[Attempt {attempt + 1}/{max_retries}] Failed"
            if context: msg += f" for {context}"
            msg += f": {error_msg}"
            logger.warning(msg)

            if "NoMatchingDataError" in error_msg: 
                logger.warning(f"Data gap detected for {context}: Source returned empty.")
                return None

            if attempt < max_retries - 1:
                time.sleep(delay)
            else:
                logger.error(f"CRITICAL FAILURE: Skipping {context if context else 'query'} after max retries.", exc_info=True)
                return None
    return None

# ==========================================
# GAP FILLING ENGINE
# ==========================================
def default_rules(series: pd.Series, gaps: pd.DataFrame, inferred_freq: pd.Timedelta) -> None:
    """
    Establishes baseline heuristics for time-series imputation based on gap duration and temporal location.
    Applies to both standard missing data ('nan') and filtered outliers ('invalid_data').
    """
    gaps["method"] = "ZERO"
    MAX_WEEK_BEFORE = pd.Timedelta(weeks=1)
    MAX_LINEAR = pd.Timedelta(hours=3)
    
    target_types = ["nan", "invalid_data"]

    gaps.loc[
        (gaps["type"].isin(target_types)) & (gaps["duration"] * inferred_freq <= MAX_WEEK_BEFORE) &
        (gaps["start"] - series.index[0] >= MAX_WEEK_BEFORE), "method",
    ] = "WEEK_BEFORE"

    gaps.loc[
        (gaps["type"].isin(target_types)) & (gaps["duration"] * inferred_freq <= MAX_LINEAR) &
        (gaps["start"] > series.index[0]) & (gaps["end"] < series.index[-1]), "method",
    ] = "LINEAR"

    gaps.loc[
        (gaps["type"].isin(target_types)) & (gaps["duration"] * inferred_freq <= MAX_LINEAR) &
        (gaps["start"] > series.index[0]) & (gaps["end"] == series.index[-1]), "method",
    ] = "FORWARD_FILL"

    gaps.loc[
        (gaps["type"].isin(target_types)) & (gaps["duration"] * inferred_freq <= MAX_LINEAR) &
        (gaps["start"] == series.index[0]) & (gaps["end"] < series.index[-1]), "method",
    ] = "BACKWARD_FILL"

    mask_invalid = gaps["type"] == "invalid_data"
    if mask_invalid.any():
        gaps.loc[mask_invalid, "method"] = "FILTERED_OUTLIER_" + gaps.loc[mask_invalid, "method"]

def fill_gaps_series(series: pd.Series, gaps: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
    """Applies targeted imputation arrays to identified temporal gaps within a continuous 1D series."""
    gaps["success"] = False
    gaps["filled_values"] = 0
    gaps["filled_quantity"] = 0.0

    for i, gap in gaps.iterrows():
        start, end, duration, method = gap["start"], gap["end"], gap["duration"], gap["method"]
        if method == "ZERO":
            series.loc[start:end] = 0
        elif method == "LINEAR":
            pos_start = series.index.get_loc(start)
            series.loc[start:end] = np.linspace(series.iloc[pos_start - 1], series.iloc[pos_start + duration], duration + 2)[1:-1]
        elif method == "FORWARD_FILL":
            series.loc[start:end] = series.iloc[series.index.get_loc(start) - 1]
        elif method == "BACKWARD_FILL":
            series.loc[start:end] = series.iloc[series.index.get_loc(start) + duration]
        elif method == "WEEK_BEFORE":
            one_week = pd.Timedelta(weeks=1)
            series.loc[start:end] = series.loc[(start - one_week):(end - one_week)].values

        gaps.loc[i, "success"] = series.loc[start:end].count() > 0
        gaps.loc[i, "filled_values"] = series.loc[start:end].count()
        gaps.loc[i, "filled_quantity"] = series.loc[start:end].sum()

    return series, gaps

def find_gaps_series(
    series: pd.Series,
    output_dict: Optional[Dict[str, pd.DataFrame]] = None,
    check_negatives: bool = False,
    allow_negatives: Optional[List[str]] = None,
    fill_gaps: bool = False,
    gap_filling_rules: Optional[Callable] = None
) -> pd.Series:
    """Scans a continuous series to isolate, measure, and classify missing or invalid temporal blocks."""
    # Ignore non-numerical metadata features
    if not pd.api.types.is_numeric_dtype(series):
        return series
    
    if allow_negatives is None: allow_negatives = []

    # Identify structural NaNs and physical outliers distinctly
    is_invalid = series >= 100000
    is_nan = series.isna()

    series = series.mask(is_invalid, np.nan)

    def extract_blocks(mask: pd.Series, gap_type: str) -> pd.DataFrame:
        starts = mask & (~mask.shift(1, fill_value=False))
        ends = mask & (~mask.shift(-1, fill_value=False))
        df = pd.DataFrame({"start": series[starts].index, "end": series[ends].index})
        if not df.empty:
            df["duration"] = df.apply(lambda row: mask[row["start"] : row["end"]].sum(), axis=1).astype(int)
        else:
            df["duration"] = pd.Series(dtype=int)
        df["value"] = np.nan
        df["type"] = gap_type
        return df

    gaps = pd.concat([
        extract_blocks(is_nan, "nan"),
        extract_blocks(is_invalid, "invalid_data")
    ], ignore_index=True)

    if check_negatives and (str(series.name) not in allow_negatives):
        is_neg = series < 0
        negs = extract_blocks(is_neg, "negative")
        if not negs.empty:
            negs["value"] = negs.apply(lambda row: series[row["start"] : row["end"]].sum(), axis=1)
        gaps = pd.concat([gaps, negs], ignore_index=True)

    gaps = gaps.sort_values(by="start").reset_index(drop=True)

    inferred_freq = pd.infer_freq(series.index[:3])
    if (inferred_freq is not None) and (len(inferred_freq) == 1): inferred_freq = "1" + inferred_freq
    freq_td = pd.to_timedelta(inferred_freq) if inferred_freq else pd.Timedelta(hours=1)
    gaps["method"] = "UNDEFINED"

    if gap_filling_rules is not None: gap_filling_rules(series, gaps, freq_td)
    if fill_gaps: series, gaps = fill_gaps_series(series, gaps)
    if output_dict is not None: output_dict[str(series.name)] = gaps
    
    return series

def find_gaps(
    df: pd.DataFrame,
    check_negatives: bool = False,
    allow_negatives: Optional[List[str]] = None,
    fill_gaps: bool = False,
    gap_filling_rules: Callable = default_rules
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Iterates row-level gap scanning across a primary DataFrame matrix."""
    if allow_negatives is None: allow_negatives = []
    output_dict: Dict[str, pd.DataFrame] = {}
    df_result = df.apply(find_gaps_series, axis=0, output_dict=output_dict, check_negatives=check_negatives,
                         allow_negatives=allow_negatives, fill_gaps=fill_gaps, gap_filling_rules=gap_filling_rules)
    return df_result, output_dict

def patch_gaps_with_dayahead(
    flow_df: pd.DataFrame,
    gap_dict: Dict[str, pd.DataFrame],
    bz: str,
    neighbour: str,
    config: Any, 
    io: DataIO,
    min_gap_length: pd.Timedelta = pd.Timedelta(weeks=1)
) -> pd.DataFrame:
    """Leverages day-ahead commercial schedules as a physical proxy to impute extended missing flow blocks."""
    long_gaps: List[Tuple[str, pd.Timestamp, pd.Timestamp]] = []
    for col in [f"{bz}_{neighbour}", f"{neighbour}_{bz}"]:
        if col in gap_dict:
            for _, row in gap_dict[col].iterrows():
                if (row["end"] - row["start"]) > min_gap_length:
                    long_gaps.append((col, row["start"], row["end"]))

    if not long_gaps:
        return flow_df

    folder = "comm_flow_dayahead_bidding_zones"
    filename = f"{bz}_comm_flow_dayahead_bidding_zones.csv"
    path = config.get_output_path(folder) / filename
    table_name = "processed_commercial_flows_da"

    da_df = io.load(path, table_name, config, bz=bz)

    if da_df is None or da_df.empty:
        return flow_df

    patched_count = 0
    for col, start, end in long_gaps:
        if col in da_df.columns:
            replacement = da_df.loc[start:end, col]

            if not (replacement.empty or replacement.isna().all()):
                flow_df.loc[start:end, col] = replacement
                patched_count += 1
                _record_gap_method(flow_df, start, end, "DAYAHEAD_PROXY", col_name=col)

    if patched_count > 0:
        logger.info(f"   -> [Patch] Used {table_name} to fill {patched_count} long-duration gaps for {bz}.")

    return flow_df

def fill_gaps_wrapper(
    df: pd.DataFrame,
    gaps_dir: Optional[Path],
    prefix: str,
    config: Optional[Any] = None,
    io: DataIO = None,
    bz: Optional[str] = None,
    flow_type: Optional[str] = None,
    dayahead: bool = False
) -> pd.DataFrame:
    """Orchestrates the detection, rule assignment, and execution of the gap-filling sequence."""
    if df.empty: return df
    
    if "gap_filling_method" not in df.columns:
        df["gap_filling_method"] = "None"
        
    _, gaps_dict = find_gaps(df, check_negatives=False, fill_gaps=False)

    if config and bz and (flow_type == "commercial") and (not dayahead):
        if hasattr(config, 'neighbours_map') and bz in config.neighbours_map:
            for neighbour in [n for n in config.neighbours_map[bz] if f"{bz}_{n}" in df.columns]:
                df = patch_gaps_with_dayahead(df, gaps_dict, bz, neighbour, config, io)

    df_filled, new_gaps_dict = find_gaps(df, check_negatives=False, fill_gaps=True, gap_filling_rules=default_rules)

    for col_name, gap_df in new_gaps_dict.items():
        if gap_df.empty: continue
        for _, row in gap_df.iterrows():
            if row.get("success", True):
                _record_gap_method(df_filled, row["start"], row["end"], row["method"], col_name=str(col_name))

    if gaps_dir:
        for key, gap_df in new_gaps_dict.items():
            file_path = gaps_dir / f"{prefix}_{str(key).replace('/', '_').replace(' ', '_')}_gaps.csv"
            if not gap_df.empty:
                gap_df.to_csv(file_path)
            else:
                if file_path.exists():
                    file_path.unlink()

    return df_filled

def correct_zero_values(
    df: pd.DataFrame, 
    gaps_dir: Path, 
    bz: str, 
    config: Any, 
    flow_type: str = "commercial", 
    reference_df: Optional[pd.DataFrame] = None,
    is_post_imputation: bool = False,
    record_zeros: bool = True
) -> pd.DataFrame:
    """
    Identifies and patches anomalous zero-values in time-series flow data.
    
    Uses Commercial Proxy Cross-Validation universally on all borders to diagnose 
    telemetry failures vs. legitimate physical maintenance. 
    
    If is_post_imputation=True, it operates strictly as an auditor and sanitizes 
    the final dataframe so no NaNs remain.
    """
    df = df.copy()
    bidding_zones = getattr(config, 'zones', [])
    valid_zero_zones = getattr(config, 'valid_zero_zones', [])
    checked_cols = set()
    zero_timestamps = set()

    def apply_patch(mask: pd.Series, cols: List[str], method: str):
        # 1. Punch the NaNs (ONLY in Pass 1. In Pass 2, we leave them as 0s)
        if not is_post_imputation:
            df.loc[mask, cols] = np.nan
            
        # 2. Apply audit tags strictly to the masked rows (preventing date-range bleeding)
        if "gap_filling_method" not in df.columns:
            df["gap_filling_method"] = "None"
            
        for col in cols:
            tagged_method = f"[{col}] {method}"
            
            # Tag rows that currently have 'None'
            none_mask = mask & (df["gap_filling_method"] == "None")
            df.loc[none_mask, "gap_filling_method"] = tagged_method
            
            # Append to rows that already have other tags
            append_mask = mask & (df["gap_filling_method"] != "None") & (~df["gap_filling_method"].str.contains(tagged_method, regex=False, na=False))
            df.loc[append_mask, "gap_filling_method"] += f", {tagged_method}"

    # 1. Generation & Load Auditing
    if "Total Generation" in df.columns and "Total Load" in df.columns:
        gen_zero_mask = (df["Total Generation"] == 0) & (df["Total Load"] == 0)
        if gen_zero_mask.any():
            apply_patch(gen_zero_mask, ["Total Generation", "Total Load"], "GEN_LOAD_BILATERAL_ZERO")

    # 2. Bilateral Border Flow Auditing (STRICTLY FOR PHYSICAL FLOWS)
    if flow_type == "physical":
        neighbours = getattr(config, 'neighbours_map', {}).get(bz, [])
        for target in neighbours:
            if target in bidding_zones:
                col_out = f"{bz}_{target}"
                col_in = f"{target}_{bz}"

                if col_out in df.columns and col_in in df.columns:
                    checked_cols.update([col_out, col_in])
                    
                    # Explicit Check: Are these zones structurally allowed to have zero flow?
                    if bz in valid_zero_zones or target in valid_zero_zones:
                        continue

                    # Unified Telemetry Check: Both physical directions drop to 0.0
                    phys_dead_mask = (df[col_out] == 0) & (df[col_in] == 0)
                    
                    if phys_dead_mask.any():
                        if reference_df is not None and col_out in reference_df.columns and col_in in reference_df.columns:
                            # DIAGNOSTIC: Cross-Validate with Market Data
                            faulty_mask_out = phys_dead_mask & (reference_df[col_out].abs() > 10)
                            if faulty_mask_out.any():
                                apply_patch(faulty_mask_out, [col_out], f"TELEMETRY_DROPOUT_[{target}]")

                            faulty_mask_in = phys_dead_mask & (reference_df[col_in].abs() > 10)
                            if faulty_mask_in.any():
                                apply_patch(faulty_mask_in, [col_in], f"TELEMETRY_DROPOUT_[{target}]")
                        else:
                            # Fallback if no commercial proxy is available
                            apply_patch(phys_dead_mask, [col_out, col_in], f"PHYS_BILATERAL_ZERO_[{target}]")

        # ======================================================================
        # Complete Zone Blackout Audit (Symmetry Tie-Breaker Logic)
        # ======================================================================
        bilateral_cols = []
        for target in neighbours:
            out_c = f"{bz}_{target}"
            in_c = f"{target}_{bz}"
            if out_c in df.columns:
                bilateral_cols.append(out_c)
            if in_c in df.columns:
                bilateral_cols.append(in_c)

        bilateral_cols = list(dict.fromkeys(bilateral_cols))
        if bilateral_cols and record_zeros:
            mask_all_zero = (df[bilateral_cols] == 0).all(axis=1)
            zero_timestamps.update(df.loc[mask_all_zero].index)

    # 3. Persist local audit log (Safely Appends for Two-Pass execution)
    if zero_timestamps:
        new_zeros = pd.DataFrame(index=sorted(list(zero_timestamps)))
        gaps_dir.mkdir(parents=True, exist_ok=True)
        csv_path = gaps_dir / f"{bz}_zeros.csv"
        
        new_zeros.to_csv(csv_path)

    # 4. Final Data Sanitization (Pass 2 Only: Erase all remaining NaNs)
    if is_post_imputation:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(0.0)

    # 5. Deterministic Recalculation
    net_export_cols = [c for c in df.columns if c.endswith("_net_export")]
    if net_export_cols:
        for net_col in net_export_cols:
            col_out = net_col.replace("_net_export", "")
            target = col_out.replace(f"{bz}_", "") if col_out.startswith(f"{bz}_") else col_out.replace(f"_{bz}", "")
            col_in = f"{target}_{bz}"
            
            out_val = df[col_out] if col_out in df.columns else 0.0
            in_val = df[col_in] if col_in in df.columns else 0.0
            
            df[net_col] = out_val - in_val

    return df