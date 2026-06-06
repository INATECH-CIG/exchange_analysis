"""
Project: European Electricity Exchange Analysis
Description: Generates multi-year statistical comparisons and plots 
for commercial and physical flows. Refactored for Pipeline Integration.
"""

import pandas as pd
import numpy as np
import logging
import re
import os
from pathlib import Path
from typing import List, Optional
from config import PipelineConfig
from utils import DataIO

logger = logging.getLogger(__name__)

def save_chunked_files(df: pd.DataFrame, base_path: Path, prefix: str, entities_per_file: int = 6):
    """
    Dynamically chunks a formatted DataFrame into multiple CSV files.
    Safely groups by the blanked 'Zone' column so entities are never split across files.
    """
    # 1. Save the complete master file
    df.to_csv(base_path / f"top_{prefix}_all_years.csv", index=False)
    
    # 2. Find row indices where a new Zone/Border begins
    start_indices = df.index[df['Zone'] != ''].tolist()
    start_indices.append(len(df)) # Cap with the total length
    
    # 3. Slice into safe chunks and export
    for i in range(0, len(start_indices)-1, entities_per_file):
        start_row = start_indices[i]
        end_idx = min(i + entities_per_file, len(start_indices)-1)
        end_row = start_indices[end_idx]
        
        chunk = df.iloc[start_row:end_row]
        file_idx = (i // entities_per_file) + 1
        chunk.to_csv(base_path / f"top_{prefix}_all_years_{file_idx}.csv", index=False)

def get_unique_directional_borders(columns: List[str]) -> List[str]:
    """
    Filters a list of border columns to ensure bidirectional borders (A-B and B-A)
    are only represented once, preventing double-counting in volume aggregations.
    """
    seen = set()
    keep = []
    for col in columns:
        parts = col.split('-')
        if len(parts) == 2:
            canonical = tuple(sorted(parts))
            if canonical not in seen:
                seen.add(canonical)
                keep.append(col)
        else:
            keep.append(col)
    return keep

def perform_statistical_analysis(config: PipelineConfig, io: DataIO, target_years: Optional[List[str]] = None):
    """
    Executes cross-year statistical analysis on flow data.
    Temporarily overrides config temporal boundaries to allow historical loading.
    """
    logger.info("=== STARTING STATISTICAL ANALYSIS ===")

    # Default to 2021-2025 if no specific years are passed
    if target_years is None:
        target_years = ["2021", "2022", "2023", "2024", "2025"]

    # Save original config states to restore later
    original_year = config.year
    original_start = config.start
    original_end = config.end
    original_time_index = config.time_index

    # 1. Dynamically load topology from config
    bidding_zones = config.zones
    
    # Initialize global structures
    comm_exchange_dfs_yearly = {}
    comm_exchange_dayahead_dfs_yearly = {}
    sdac_net_positions_dfs_yearly = {}
    flow_dfs_yearly = {}
    gen_load_dfs_yearly = {}
    gen_load_net_position_dfs_yearly = {}

    try:
        # ==========================================
        # 1. MULTI-YEAR DATA LOADING & MASKING
        # ==========================================
        for year_str in target_years:
            year = int(year_str)
            logger.info(f"[Stats] Loading and masking data for {year}...")
            
            # Temporarily patch config bounds for DataIO filtering
            config.year = year
            config.start = pd.Timestamp(f"{year}-01-01 00:00", tz="UTC")
            config.end = pd.Timestamp(f"{year}-12-31 23:59", tz="UTC")
            config.time_index = pd.date_range(start=config.start, end=config.end, freq="1h")
            
            # Initialize year dictionaries
            comm_exchange_dfs_yearly[year_str] = {}
            comm_exchange_dayahead_dfs_yearly[year_str] = {}
            sdac_net_positions_dfs_yearly[year_str] = pd.DataFrame(index=config.time_index)
            flow_dfs_yearly[year_str] = {}
            gen_load_dfs_yearly[year_str] = {}
            gen_load_net_position_dfs_yearly[year_str] = pd.DataFrame(index=config.time_index)

            # Helper function for targeted purity filtering
            def apply_targeted_purity_filter(df, bz_country):
                if "gap_filling_method" in df.columns:
                    # 1. Create mask: not None, not NaN, AND does not contain "CLIPPED_NEGATIVE"
                    dirty_mask = (
                        (df["gap_filling_method"] != "None") & 
                        df["gap_filling_method"].notna() 
                    )
                    
                    if dirty_mask.any():
                        # 2. Always "punish" the Net Export row entirely (sum is structurally invalid)
                        if "Net Export" in df.columns:
                            df.loc[dirty_mask, "Net Export"] = np.nan
                            
                        # 3. Surgically strike individual borders based on the matching column name strings
                        num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != "Net Export"]
                        for col in num_cols:
                            base_border = col.replace("_net_export", "")
                            
                            if base_border.startswith(f"{bz_country}_"):
                                neighbour = base_border[len(f"{bz_country}_"):]
                                dir1 = f"{bz_country}_{neighbour}"
                                dir2 = f"{neighbour}_{bz_country}"
                                
                                # Check if EITHER direction name permutation is found inside the text trace
                                col_specific_dirty_mask = dirty_mask & (
                                    df["gap_filling_method"].astype(str).str.contains(dir1, na=False, regex=False) |
                                    df["gap_filling_method"].astype(str).str.contains(dir2, na=False, regex=False)
                                )
                                df.loc[col_specific_dirty_mask, col] = np.nan
                            else:
                                df.loc[dirty_mask, col] = np.nan
                return df

            # --- A. Commercial Flow Total (CFT) ---
            cft_dir = config.get_output_path("comm_flow_total_bidding_zones")
            for country in sorted(bidding_zones):
                df = io.load(cft_dir / f"{country}_comm_flow_total_bidding_zones.csv", "processed_commercial_flows", config, bz=country)
                if df is None or df.empty: continue
                df = apply_targeted_purity_filter(df, country)
                comm_exchange_dfs_yearly[year_str][country] = df.copy()

            # --- IT specific adjustments (CFT) ---
            if "IT_CSUD" in comm_exchange_dfs_yearly[year_str]:
                comm_exchange_dfs_yearly[year_str]["IT_CSUD"] = comm_exchange_dfs_yearly[year_str]["IT_CSUD"][abs(comm_exchange_dfs_yearly[year_str]["IT_CSUD"]["IT_CNOR_IT_CSUD"]) <= 8000]
            if "IT_CNOR" in comm_exchange_dfs_yearly[year_str]:
                comm_exchange_dfs_yearly[year_str]["IT_CNOR"] = comm_exchange_dfs_yearly[year_str]["IT_CNOR"][abs(comm_exchange_dfs_yearly[year_str]["IT_CNOR"]["IT_CNOR_IT_CSUD"]) <= 8000]
                comm_exchange_dfs_yearly[year_str]["IT_CNOR"] = comm_exchange_dfs_yearly[year_str]["IT_CNOR"][abs(comm_exchange_dfs_yearly[year_str]["IT_CNOR"]["IT_NORD_IT_CNOR"]) <= 8000]
            if "IT_SARD" in comm_exchange_dfs_yearly[year_str]:
                comm_exchange_dfs_yearly[year_str]["IT_SARD"] = comm_exchange_dfs_yearly[year_str]["IT_SARD"][abs(comm_exchange_dfs_yearly[year_str]["IT_SARD"]["IT_CSUD_IT_SARD"]) <= 8000]
            comm_exchange_dfs_yearly[year_str]["IT_CSUD"] = comm_exchange_dfs_yearly[year_str]["IT_CSUD"][abs(comm_exchange_dfs_yearly[year_str]["IT_CSUD"]["IT_CSUD_IT_SARD"]) <= 8000]
            if "IT_NORD" in comm_exchange_dfs_yearly[year_str]:
                comm_exchange_dfs_yearly[year_str]["IT_NORD"] = comm_exchange_dfs_yearly[year_str]["IT_NORD"][abs(comm_exchange_dfs_yearly[year_str]["IT_NORD"]["IT_NORD_IT_CNOR"]) <= 8000]

            # --- B. Dayahead Comm Exchange (CFD) ---
            cfd_dir = config.get_output_path("comm_flow_dayahead_bidding_zones")
            for country in sorted(bidding_zones):
                df = io.load(cfd_dir / f"{country}_comm_flow_dayahead_bidding_zones.csv", "processed_commercial_flows_da", config, bz=country)
                if df is None or df.empty: continue
                df = apply_targeted_purity_filter(df, country)
                comm_exchange_dayahead_dfs_yearly[year_str][country] = df.copy()

            # --- C. SDAC Net Positions ---
            sdac_dir = config.get_output_path("net_positions_dayahead")
            sdac_gaps_dir = config.get_output_path("net_positions_dayahead")
            for country in bidding_zones:
                df = io.load(sdac_dir / f"{country}_net_positions_dayahead.csv", "raw_net_positions_dayahead", config, bz=country)
                if df is not None and not df.empty:
                    times_to_drop = set()
                    gap_path = sdac_gaps_dir / f"{country}_gaps.csv"
                    
                    if gap_path.exists():
                        gap_df = pd.read_csv(gap_path)
                        for _, row in gap_df.iterrows():
                            if pd.to_datetime(row["start"], utc=True).year == year:
                                times_to_drop.update(pd.date_range(start=pd.to_datetime(row["start"], utc=True), end=pd.to_datetime(row["end"], utc=True), freq="1h"))
                    
                    value_series = df["Value"].copy()
                    if times_to_drop:
                        value_series.loc[value_series.index.isin(times_to_drop)] = np.nan
                        
                    sdac_net_positions_dfs_yearly[year_str][country] = value_series

            # --- D. Physical flows ---
            phys_dir = config.get_output_path("physical_flow_data_bidding_zones")
            for country in bidding_zones:
                df = io.load(phys_dir / f"{country}_physical_flow_data_bidding_zones.csv", "processed_physical_flows", config, bz=country)
                if df is None or df.empty: continue
                df = apply_targeted_purity_filter(df, country)
                flow_dfs_yearly[year_str][country] = df.copy()

            # --- E. Generation & Load ---
            gen_dir = config.get_output_path("generation_demand_data_bidding_zones")
            gen_bidding_zones = [z for z in bidding_zones if z != "GB"] 

            for country in sorted(gen_bidding_zones):
                df = io.load(gen_dir / f"{country}_generation_demand_data_bidding_zones.csv", "processed_generation", config, bz=country)
                if df is not None and not df.empty:
                    # Domain logic: Invalid demand ratio masking
                    if "Demand" in df.columns:
                        invalid_demand_mask = (df["Total Generation"] / df["Demand"] <= 0.33) | df["Demand"].isna()
                        num_cols = df.select_dtypes(include=[np.number]).columns
                        df.loc[invalid_demand_mask, num_cols] = np.nan
                        
                    # STATISTICAL PURITY FILTER
                    if "gap_filling_method" in df.columns:
                        dirty_mask = (
                        (df["gap_filling_method"] != "None") & 
                        df["gap_filling_method"].notna() &
                        ~df["gap_filling_method"].astype(str).str.contains("CLIPPED_NEGATIVE", na=False)
                        )
                        
                        if dirty_mask.any():
                            # 2. Always "punish" the Net Export row entirely (sum is structurally invalid)
                            if "Net Export" in df.columns:
                                df.loc[dirty_mask, "Net Export"] = np.nan
                            num_cols = df.select_dtypes(include=[np.number]).columns
                            df.loc[dirty_mask, num_cols] = np.nan

                    gen_load_dfs_yearly[year_str][country] = df.copy()
                    if "Net Export" in df.columns:
                        gen_load_net_position_dfs_yearly[year_str][country] = df["Net Export"]


        # ==========================================
        # 2. CROSS-YEAR AGGREGATION & MATRICES
        # ==========================================
        logger.info("[Stats] Compiling Structural Flow Matrices and Enforcing Symmetry...")

        phys_individual_flows_df_yearly = {}
        comm_individual_flows_df_yearly = {}
        comm_individual_dayahead_flows_df_yearly = {}

        comm_individual_dayahead_flows_df_abs_yearly = {}
        comm_individual_flows_df_abs_yearly = {}
        phys_individual_flows_df_abs_yearly = {}

        for year_str in target_years:
            config.year = int(year_str)
            time_range = pd.date_range(start=pd.Timestamp(f"{year_str}-01-01 00:00", tz="UTC"), 
                                       end=pd.Timestamp(f"{year_str}-12-31 23:59", tz="UTC"), freq="1h")
            
            p_ind = pd.DataFrame(index=time_range)
            c_ind = pd.DataFrame(index=time_range)
            da_ind = pd.DataFrame(index=time_range)

            for country in bidding_zones:
                for country_again in bidding_zones:
                    border_key = f"{country}-{country_again}"
                    col_key = f"{country}_{country_again}"
                    
                    if country in flow_dfs_yearly[year_str] and col_key in flow_dfs_yearly[year_str][country].columns:
                        if border_key not in p_ind.columns:
                            p_ind[border_key] = flow_dfs_yearly[year_str][country][f"{col_key}_net_export"]
                            
                    if country in comm_exchange_dfs_yearly[year_str] and col_key in comm_exchange_dfs_yearly[year_str][country].columns:
                        if border_key not in c_ind.columns:
                            c_ind[border_key] = comm_exchange_dfs_yearly[year_str][country][f"{col_key}_net_export"]

                    if country in comm_exchange_dayahead_dfs_yearly[year_str] and col_key in comm_exchange_dayahead_dfs_yearly[year_str][country].columns:
                        if border_key not in da_ind.columns:
                            da_ind[border_key] = comm_exchange_dayahead_dfs_yearly[year_str][country][f"{col_key}_net_export"]

            # ------------------------------------------
            # SYMMETRY RECONCILIATION: Enforce mutual NaNs
            # ------------------------------------------
            for df_matrix in [p_ind, c_ind, da_ind]:
                for col in df_matrix.columns:
                    parts = col.split('-')
                    if len(parts) == 2:
                        rev_col = f"{parts[1]}-{parts[0]}"
                        if rev_col in df_matrix.columns:
                            combined_mask = df_matrix[col].isna() | df_matrix[rev_col].isna()
                            df_matrix.loc[combined_mask, col] = np.nan
                            df_matrix.loc[combined_mask, rev_col] = np.nan

            phys_individual_flows_df_yearly[year_str] = p_ind
            comm_individual_flows_df_yearly[year_str] = c_ind
            comm_individual_dayahead_flows_df_yearly[year_str] = da_ind

            # Deduplicate borders prior to absolute summation to prevent double-counting volumes
            phys_unique = get_unique_directional_borders(p_ind.columns)
            comm_unique = get_unique_directional_borders(c_ind.columns)
            da_unique = get_unique_directional_borders(da_ind.columns)

            phys_individual_flows_df_abs_yearly[year_str] = abs(p_ind[phys_unique]).sum(axis=1)
            comm_individual_flows_df_abs_yearly[year_str] = abs(c_ind[comm_unique]).sum(axis=1)
            comm_individual_dayahead_flows_df_abs_yearly[year_str] = abs(da_ind[da_unique]).sum(axis=1)


        # ==========================================
        # 3. DETAILED STATISTICAL METRICS (Medians, Quantiles)
        # ==========================================
        logger.info("[Stats] Computing detailed summary statistics and deltas...")

        flow_comparison_df_yearly = {}
        delta_individual_flows_comparison_df_yearly = {}
        net_position_comparison_df_yearly = {}
        net_position_deltas_comparison_df_yearly = {}
        
        flow_comparison_df_yearly_comm_flow_total = {}
        flow_comparison_df_yearly_phys_flow = {}
        
        net_position_comparison_df_yearly_comm_flow_total = {}
        net_position_comparison_df_yearly_phys_flow = {}

        for year_str in target_years:
            config.year = int(year_str)
            flow_comp_dir = config.output_dir / "flow_type_comparison" / year_str
            delta_comp_dir = flow_comp_dir / "deltas" / "individual_flows"
            flow_comp_dir.mkdir(parents=True, exist_ok=True)
            delta_comp_dir.mkdir(parents=True, exist_ok=True)

            p_ind = phys_individual_flows_df_yearly[year_str]
            c_ind = comm_individual_flows_df_yearly[year_str]
            da_ind = comm_individual_dayahead_flows_df_yearly[year_str]

            # --- A. Border Flow Comparison (flow_comparison_df_yearly) ---
            if not p_ind.empty:
                border_index = np.array([val for val in p_ind.columns for _ in range(3)])
                type_index = np.array(["CFD", "CFT", "Phys."] * len(p_ind.columns))
                
                flow_comp_df = pd.DataFrame(
                    index=pd.MultiIndex.from_arrays([border_index, type_index], names=["border", "type"]),
                    columns=["Median", "Lower Q.", "Upper Q.", "Std. Dev.", 
                             "Min Val.", "Max Val.", "Neg. Val. Share",
                             "Pos. Val. Share", "Zero Val. Share", "Missing Share", 
                             "Neg. Val. Median", "Pos. Val. Median", "Abs. Val. Median", 
                             "Corr. CFD", "Corr. CFT", "Corr. Phys."]
                )

                for border in p_ind.columns:
                    for flow_type, df in [("CFD", da_ind), ("CFT", c_ind), ("Phys.", p_ind)]:
                        if border not in df.columns: continue
                        
                        s = df[border]
                        total_len = len(s)
                        if total_len == 0: continue

                        flow_comp_df.at[(border, flow_type), "Median"] = s.median()
                        flow_comp_df.at[(border, flow_type), "Lower Q."] = s.quantile(0.25)
                        flow_comp_df.at[(border, flow_type), "Upper Q."] = s.quantile(0.75)
                        flow_comp_df.at[(border, flow_type), "Std. Dev."] = s.std()
                        flow_comp_df.at[(border, flow_type), "Min Val."] = s.min()
                        flow_comp_df.at[(border, flow_type), "Max Val."] = s.max()
                        
                        flow_comp_df.at[(border, flow_type), "Neg. Val. Share"] = len(s[s < 0]) / total_len
                        flow_comp_df.at[(border, flow_type), "Pos. Val. Share"] = len(s[s > 0]) / total_len
                        flow_comp_df.at[(border, flow_type), "Zero Val. Share"] = len(s[s == 0]) / total_len
                        flow_comp_df.at[(border, flow_type), "Missing Share"] = s.isna().sum() / total_len
                        
                        flow_comp_df.at[(border, flow_type), "Neg. Val. Median"] = s[s < 0].median()
                        flow_comp_df.at[(border, flow_type), "Pos. Val. Median"] = s[s > 0].median()
                        flow_comp_df.at[(border, flow_type), "Abs. Val. Median"] = s.abs().median()
                        
                        if border in da_ind.columns:
                            flow_comp_df.at[(border, flow_type), "Corr. CFD"] = s.corr(da_ind[border])
                        if border in c_ind.columns:
                            flow_comp_df.at[(border, flow_type), "Corr. CFT"] = s.corr(c_ind[border])
                        if border in p_ind.columns:
                            flow_comp_df.at[(border, flow_type), "Corr. Phys."] = s.corr(p_ind[border])

                flow_comp_df.to_csv(flow_comp_dir / f"border_flow_stats_{year_str}.csv")
                flow_comparison_df_yearly[year_str] = flow_comp_df
                flow_comparison_df_yearly_comm_flow_total[year_str] = flow_comp_df.loc[[x for x in flow_comp_df.index if "CFT" in x[1]]]
                flow_comparison_df_yearly_phys_flow[year_str] = flow_comp_df.loc[[x for x in flow_comp_df.index if "Phys." in x[1]]]

                # Output Top 20 Subsets
                # Ensure we only pick one direction per border to avoid duplicate pairs in rankings
                unique_borders = get_unique_directional_borders(flow_comp_df.index.get_level_values(0).drop_duplicates().tolist())
                unique_flow_comp_df = flow_comp_df.loc[flow_comp_df.index.get_level_values(0).isin(unique_borders)]
                
                cft_subset = unique_flow_comp_df.loc[[x for x in unique_flow_comp_df.index if "CFT" in x[1]]]
                
                busiest_borders = unique_flow_comp_df.loc[[x for x in unique_flow_comp_df.index if "CFD" not in x[1]]].sort_values(by="Abs. Val. Median", ascending=False)[:20].apply(pd.to_numeric, errors='coerce').round(2)
                most_correlated = cft_subset.sort_values(by="Corr. Phys.", ascending=False)[:20].apply(pd.to_numeric, errors='coerce').round(2)
                least_correlated = cft_subset.sort_values(by="Corr. Phys.")[:20].apply(pd.to_numeric, errors='coerce').round(2)
                
                busiest_borders.to_csv(flow_comp_dir / f"busiest_borders_{year_str}.csv")
                most_correlated.to_csv(flow_comp_dir / f"most_correlated_borders_{year_str}.csv")
                least_correlated.to_csv(flow_comp_dir / f"least_correlated_borders_{year_str}.csv")

            # --- B. Border Flow Deltas (delta_individual_flows_comparison) ---
            if not p_ind.empty:
                delta_index = np.array([val for val in p_ind.columns for _ in range(2)])
                delta_type = np.array(["CFD-CFT", "CFT-Phys."] * len(p_ind.columns))
                
                delta_comp_df = pd.DataFrame(
                    index=pd.MultiIndex.from_arrays([delta_index, delta_type], names=["bidding_zone", "delta"]),
                    columns=["Median", "Lower Q.", "Upper Q.", "Std. Dev.", 
                             "Min Val.", "Max Val.", "Neg. Val. Share",
                             "Pos. Val. Share", "Zero Val. Share", "Missing Share", 
                             "Neg. Val. Median", "Pos. Val. Median", "Abs. Val. Median"]
                )

                delta_dfs = {
                    "CFD-CFT": da_ind - c_ind,
                    "CFT-Phys.": c_ind - p_ind
                }

                delta_dfs["CFD-CFT"].to_csv(delta_comp_dir / "comm_flow_dayahead_comm_flow_total_delta.csv")
                delta_dfs["CFT-Phys."].to_csv(delta_comp_dir / "comm_flow_total_phys_flow_delta.csv")

                for border in p_ind.columns:
                    for d_name, d_df in delta_dfs.items():
                        if border not in d_df.columns: continue
                        s = d_df[border]
                        total_len = len(s)
                        if total_len == 0: continue

                        delta_comp_df.at[(border, d_name), "Median"] = s.median()
                        delta_comp_df.at[(border, d_name), "Lower Q."] = s.quantile(0.25)
                        delta_comp_df.at[(border, d_name), "Upper Q."] = s.quantile(0.75)
                        delta_comp_df.at[(border, d_name), "Std. Dev."] = s.std()
                        delta_comp_df.at[(border, d_name), "Min Val."] = s.min()
                        delta_comp_df.at[(border, d_name), "Max Val."] = s.max()
                        
                        delta_comp_df.at[(border, d_name), "Neg. Val. Share"] = len(s[s < 0]) / total_len
                        delta_comp_df.at[(border, d_name), "Pos. Val. Share"] = len(s[s > 0]) / total_len
                        delta_comp_df.at[(border, d_name), "Zero Val. Share"] = len(s[s == 0]) / total_len
                        delta_comp_df.at[(border, d_name), "Missing Share"] = s.isna().sum() / total_len
                        
                        delta_comp_df.at[(border, d_name), "Neg. Val. Median"] = s[s < 0].median()
                        delta_comp_df.at[(border, d_name), "Pos. Val. Median"] = s[s > 0].median()
                        delta_comp_df.at[(border, d_name), "Abs. Val. Median"] = s.abs().median()

                delta_comp_df.to_csv(delta_comp_dir / f"border_flow_deltas_stats_{year_str}.csv")
                delta_individual_flows_comparison_df_yearly[year_str] = delta_comp_df
                
                unique_borders_delta = get_unique_directional_borders(delta_comp_df.index.get_level_values(0).drop_duplicates().tolist())
                unique_delta_comp_df = delta_comp_df.loc[delta_comp_df.index.get_level_values(0).isin(unique_borders_delta)]
                
                biggest_deltas = unique_delta_comp_df.sort_values(by="Abs. Val. Median", ascending=False)[:20].apply(pd.to_numeric, errors='coerce').round(2)
                biggest_deltas.to_csv(delta_comp_dir / f"biggest_delta_individual_flows_{year_str}.csv")

            # --- C. Net Position Stats (net_position_comparison) ---
            sdac_np = sdac_net_positions_dfs_yearly[year_str]
            cfd_np = pd.DataFrame({bz: comm_exchange_dayahead_dfs_yearly[year_str][bz]["Net Export"] for bz in bidding_zones if bz in comm_exchange_dayahead_dfs_yearly[year_str] and "Net Export" in comm_exchange_dayahead_dfs_yearly[year_str][bz].columns})
            cft_np = pd.DataFrame({bz: comm_exchange_dfs_yearly[year_str][bz]["Net Export"] for bz in bidding_zones if bz in comm_exchange_dfs_yearly[year_str] and "Net Export" in comm_exchange_dfs_yearly[year_str][bz].columns})
            phys_np = pd.DataFrame({bz: flow_dfs_yearly[year_str][bz]["Net Export"] for bz in bidding_zones if bz in flow_dfs_yearly[year_str] and "Net Export" in flow_dfs_yearly[year_str][bz].columns})
            gen_np = pd.DataFrame({bz: gen_load_dfs_yearly[year_str][bz]["Net Export"] for bz in bidding_zones if bz in gen_load_dfs_yearly[year_str] and "Net Export" in gen_load_dfs_yearly[year_str][bz].columns})

            np_index = np.array([val for val in bidding_zones for _ in range(5)])
            np_type = np.array(["SDAC", "CFD", "CFT", "Phys.", "Gen_Load"] * len(bidding_zones))

            np_comp_df = pd.DataFrame(
                index=pd.MultiIndex.from_arrays([np_index, np_type], names=["bidding_zone", "type"]),
                columns=["Median", "Lower Q.", "Upper Q.", "Std. Dev.", 
                         "Min Val.", "Max Val.", "Neg. Val. Share",
                         "Pos. Val. Share", "Zero Val. Share", "Missing Share", 
                         "Neg. Val. Median", "Pos. Val. Median", "Abs. Val. Median"]
            )

            for bz in bidding_zones:
                for flow_type, df in [("SDAC", sdac_np), ("CFD", cfd_np), ("CFT", cft_np), ("Phys.", phys_np), ("Gen_Load", gen_np)]:
                    if bz not in df.columns: continue
                    s = df[bz]
                    total_len = len(s)
                    if total_len == 0: continue

                    np_comp_df.at[(bz, flow_type), "Median"] = s.median()
                    np_comp_df.at[(bz, flow_type), "Lower Q."] = s.quantile(0.25)
                    np_comp_df.at[(bz, flow_type), "Upper Q."] = s.quantile(0.75)
                    np_comp_df.at[(bz, flow_type), "Std. Dev."] = s.std()
                    np_comp_df.at[(bz, flow_type), "Min Val."] = s.min()
                    np_comp_df.at[(bz, flow_type), "Max Val."] = s.max()
                    
                    np_comp_df.at[(bz, flow_type), "Neg. Val. Share"] = len(s[s < 0]) / total_len
                    np_comp_df.at[(bz, flow_type), "Pos. Val. Share"] = len(s[s > 0]) / total_len
                    np_comp_df.at[(bz, flow_type), "Zero Val. Share"] = len(s[s == 0]) / total_len
                    np_comp_df.at[(bz, flow_type), "Missing Share"] = s.isna().sum() / total_len
                    
                    np_comp_df.at[(bz, flow_type), "Neg. Val. Median"] = s[s < 0].median()
                    np_comp_df.at[(bz, flow_type), "Pos. Val. Median"] = s[s > 0].median()
                    np_comp_df.at[(bz, flow_type), "Abs. Val. Median"] = s.abs().median()

            np_comp_df.to_csv(flow_comp_dir / f"net_positions_stats_{year_str}.csv")
            net_position_comparison_df_yearly[year_str] = np_comp_df
            net_position_comparison_df_yearly_comm_flow_total[year_str] = np_comp_df.loc[[x for x in np_comp_df.index if "CFT" in x[1]]]
            net_position_comparison_df_yearly_phys_flow[year_str] = np_comp_df.loc[[x for x in np_comp_df.index if "Phys." in x[1]]]

            np_subset = np_comp_df.loc[[x for x in np_comp_df.index if "CFT" in x[1] or "Phys." in x[1]]]
            most_negative = np_subset.sort_values(by="Neg. Val. Median")[:20].apply(pd.to_numeric, errors='coerce').round(2)
            most_positive = np_subset.sort_values(by="Pos. Val. Median", ascending=False)[:20].apply(pd.to_numeric, errors='coerce').round(2)
            
            most_negative.to_csv(flow_comp_dir / f"most_negative_net_positions_{year_str}.csv")
            most_positive.to_csv(flow_comp_dir / f"most_positive_net_positions_{year_str}.csv")

            # --- D. Net Position Deltas (net_position_deltas_comparison) ---
            sdac_cfd_delta = sdac_np.subtract(cfd_np)
            cfd_cft_delta = cfd_np.subtract(cft_np)
            cft_phys_delta = cft_np.subtract(phys_np)
            phys_gen_delta = phys_np.subtract(gen_np)

            delta_dfs_map = {
                "SDAC-CFD": sdac_cfd_delta,
                "CFD-CFT": cfd_cft_delta,
                "CFT-Phys.": cft_phys_delta,
                "Phys.-Gen_Load": phys_gen_delta
            }

            np_delta_types = list(delta_dfs_map.keys())
            np_delta_index = np.repeat(bidding_zones, len(np_delta_types))
            np_delta_type_index = np.tile(np_delta_types, len(bidding_zones))
            
            np_delta_comp_df = pd.DataFrame(
                index=pd.MultiIndex.from_arrays([np_delta_index, np_delta_type_index], names=["bidding_zone", "delta"]),
                columns=["Median", "Lower Q.", "Upper Q.", "Std. Dev.", 
                         "Min Val.", "Max Val.", "Neg. Val. Share",
                         "Pos. Val. Share", "Zero Val. Share", "Missing Share", 
                         "Neg. Val. Median", "Pos. Val. Median", "Abs. Val. Median"]
            )

            for bz in bidding_zones:
                for d_name, d_df in delta_dfs_map.items():
                    if bz in d_df.columns:
                        s = d_df[bz]
                        total_len = len(s)
                        if total_len > 0:
                            np_delta_comp_df.at[(bz, d_name), "Median"] = s.median()
                            np_delta_comp_df.at[(bz, d_name), "Lower Q."] = s.quantile(0.25)
                            np_delta_comp_df.at[(bz, d_name), "Upper Q."] = s.quantile(0.75)
                            np_delta_comp_df.at[(bz, d_name), "Std. Dev."] = s.std()
                            np_delta_comp_df.at[(bz, d_name), "Min Val."] = s.min()
                            np_delta_comp_df.at[(bz, d_name), "Max Val."] = s.max()
                            
                            np_delta_comp_df.at[(bz, d_name), "Neg. Val. Share"] = len(s[s < 0]) / total_len
                            np_delta_comp_df.at[(bz, d_name), "Pos. Val. Share"] = len(s[s > 0]) / total_len
                            np_delta_comp_df.at[(bz, d_name), "Zero Val. Share"] = len(s[s == 0]) / total_len
                            np_delta_comp_df.at[(bz, d_name), "Missing Share"] = s.isna().sum() / total_len
                            
                            np_delta_comp_df.at[(bz, d_name), "Neg. Val. Median"] = s[s < 0].median()
                            np_delta_comp_df.at[(bz, d_name), "Pos. Val. Median"] = s[s > 0].median()
                            np_delta_comp_df.at[(bz, d_name), "Abs. Val. Median"] = s.abs().median()

            np_delta_comp_df.dropna(how='all').to_csv(flow_comp_dir / f"net_position_deltas_stats_{year_str}.csv")
            net_position_deltas_comparison_df_yearly[year_str] = np_delta_comp_df

            # get unique direction borders (to avoid double counting pairs if any, though for net position it's just zones)
            # note: the index level 0 here is 'bidding_zone' not borders like 'A-B'. But applying get_unique_directional_borders
            # on simple strings like 'FR' just returns them unchanged, which is safe.
            unique_np_delta = get_unique_directional_borders(np_delta_comp_df.index.get_level_values(0).drop_duplicates().tolist())
            unique_np_delta_comp_df = np_delta_comp_df.loc[np_delta_comp_df.index.get_level_values(0).isin(unique_np_delta)]
                
            top_net_position_deltas = unique_np_delta_comp_df.sort_values(by="Abs. Val. Median", ascending=False)[:20]

            top_net_position_deltas = top_net_position_deltas.apply(lambda col: pd.to_numeric(col, errors='coerce').round(2) if col.dtype == 'object' or pd.api.types.is_numeric_dtype(col) else col)
            top_net_position_deltas = top_net_position_deltas.round(2)

            top_net_position_deltas.to_csv(os.path.join(delta_comp_dir, f"top_net_position_deltas_{year_str}.csv"))


        # ==========================================
        # 4. CROSS-YEAR COMPILATIONS (ALL YEARS SUMMARY)
        # ==========================================
        logger.info("[Stats] Compiling 'All Years' Summary Tables...")
        flow_comp_main_dir = config.output_dir / "flow_type_comparison"
        flow_comp_main_dir.mkdir(parents=True, exist_ok=True)
        
        base_year = target_years[-1]
        
        # Explicit Paper Borders (Strictly Deduplicated - No Reverse Pairs)
        thesis_borders = [
            "FR-DE_LU", "CZ-PL", "FR-CH", "AT-DE_LU", "BE-NL", "NL-DE_LU", "BE-FR", 
            "SE_2-SE_3", "SE_3-SE_4", "NO_1-NO_3", "NO_1-NO_5", "NO_4-NO_3", "NO_5-NO_3", "NO_3-SE_2", 
            "FR-GB", "FR-IT_NORD", "FR-ES", "IT_NORD-IT_CNOR", "AT-IT_NORD", "RS-MK", 
            "RS-ME", "BE-GB", "NL-DK_1", "EE-FI", "DK_2-DE_LU", "GB-DK_1", "NO_2-DE_LU"
        ]

        thesis_zones = [
            "DE_LU", "FR", "GB", "BE", "CH", "AT", "NL", "NO_1", "NO_2", "NO_3", 
            "NO_4", "NO_5", "SE_2", "SE_3", "SE_4", "DK_1", "DK_2", "FI", "EE",
            "IT_NORD", "IT_CNOR", "ES", "CZ", "PL", "RS", "MK", "ME"
        ]

        # --- A1. All Years Border Commercial Flow Total ---
        if base_year in flow_comparison_df_yearly_comm_flow_total:
            borders_list = list(set([b for year in target_years for b, _ in flow_comparison_df_yearly_comm_flow_total[year].index]))
            
            all_years_border_comm_flow_total = pd.DataFrame(
                index=pd.MultiIndex(levels=[[],[]], codes=[[],[]], names=[u'border', u'year']),
                columns=flow_comparison_df_yearly_comm_flow_total[base_year].columns
            )

            for border in borders_list:
                for year in target_years:
                    if border in flow_comparison_df_yearly_comm_flow_total[year].index.get_level_values('border'):
                        for col in flow_comparison_df_yearly_comm_flow_total[base_year].columns:
                            all_years_border_comm_flow_total.at[(border,year), col] = flow_comparison_df_yearly_comm_flow_total[year].at[(border, "CFT"), col]
                            
                        prev_year = str(int(year) - 1)
                        if prev_year in target_years and border in flow_comparison_df_yearly_comm_flow_total[prev_year].index.get_level_values('border'):
                            all_years_border_comm_flow_total.at[(border,year), "Delta Median Year Prior"] = flow_comparison_df_yearly_comm_flow_total[year].at[(border, "CFT"), "Median"] - flow_comparison_df_yearly_comm_flow_total[prev_year].at[(border, "CFT"), "Median"]
                            
                        first_year = target_years[0]
                        if border in flow_comparison_df_yearly_comm_flow_total[first_year].index.get_level_values('border'):
                            all_years_border_comm_flow_total.at[(border,year), f"Delta Median {first_year}"] = flow_comparison_df_yearly_comm_flow_total[year].at[(border, "CFT"), "Median"] - flow_comparison_df_yearly_comm_flow_total[first_year].at[(border, "CFT"), "Median"]

            # Expand thesis_borders to capture BOTH directions for internal parsing before deduplication
            expanded_thesis_borders = set(thesis_borders)
            for b in thesis_borders:
                parts = b.split('-')
                if len(parts) == 2:
                    expanded_thesis_borders.add(f"{parts[1]}-{parts[0]}")

            subset_index = [x for x in all_years_border_comm_flow_total.index if x[0] in expanded_thesis_borders or "FR" in x[0] or "DE_LU" in x[0]]
            all_years_subset = all_years_border_comm_flow_total.loc[subset_index]
            
            # Use get_unique_directional_borders to guarantee no straggler reverse-pairs from the "FR" or "DE_LU" catch-all
            unique_subset_borders = get_unique_directional_borders([x[0] for x in all_years_subset.index])
            all_years_subset = all_years_subset[all_years_subset.index.get_level_values('border').isin(unique_subset_borders)]
            all_years_subset = all_years_subset[~all_years_subset.index.duplicated(keep='first')]

            all_years_subset = all_years_subset.apply(lambda col: pd.to_numeric(col, errors='coerce').round(2) if col.dtype == 'object' or pd.api.types.is_numeric_dtype(col) else col).round(2)
            if "Corr. CFT" in all_years_subset.columns:
                all_years_subset = all_years_subset.drop(columns=["Corr. CFT"])

            arrays = [np.array([x[0] for x in all_years_subset.index]), np.array([x[1] for x in all_years_subset.index])]
            all_years_subset.index = pd.MultiIndex.from_arrays(arrays, names=("Zone", "Year"))
            all_years_subset = all_years_subset.reset_index()
            all_years_subset['Zone'] = all_years_subset['Zone'].mask(all_years_subset['Zone'].duplicated(), '')

            all_years_border_comm_flow_total.to_csv(flow_comp_main_dir / "border_flows_all_years.csv")
            save_chunked_files(all_years_subset, flow_comp_main_dir, "border_flows")

        # --- A2. All Years Border Physical Flow ---
        if base_year in flow_comparison_df_yearly_phys_flow:
            borders_list = list(set([b for year in target_years for b, _ in flow_comparison_df_yearly_phys_flow[year].index]))
            
            all_years_border_phys_flow = pd.DataFrame(
                index=pd.MultiIndex(levels=[[],[]], codes=[[],[]], names=[u'border', u'year']),
                columns=flow_comparison_df_yearly_phys_flow[base_year].columns
            )

            for border in borders_list:
                for year in target_years:
                    if border in flow_comparison_df_yearly_phys_flow[year].index.get_level_values('border'):
                        for col in flow_comparison_df_yearly_phys_flow[base_year].columns:
                            all_years_border_phys_flow.at[(border,year), col] = flow_comparison_df_yearly_phys_flow[year].at[(border, "Phys."), col]
                            
                        prev_year = str(int(year) - 1)
                        if prev_year in target_years and border in flow_comparison_df_yearly_phys_flow[prev_year].index.get_level_values('border'):
                            all_years_border_phys_flow.at[(border,year), "Delta Median Year Prior"] = flow_comparison_df_yearly_phys_flow[year].at[(border, "Phys."), "Median"] - flow_comparison_df_yearly_phys_flow[prev_year].at[(border, "Phys."), "Median"]
                            
                        first_year = target_years[0]
                        if border in flow_comparison_df_yearly_phys_flow[first_year].index.get_level_values('border'):
                            all_years_border_phys_flow.at[(border,year), f"Delta Median {first_year}"] = flow_comparison_df_yearly_phys_flow[year].at[(border, "Phys."), "Median"] - flow_comparison_df_yearly_phys_flow[first_year].at[(border, "Phys."), "Median"]

            subset_index_phys = [x for x in all_years_border_phys_flow.index if x[0] in expanded_thesis_borders or "FR" in x[0] or "DE_LU" in x[0]]
            all_years_subset_phys = all_years_border_phys_flow.loc[subset_index_phys]
            
            # Apply deduplication logic to Physical flows
            unique_subset_borders_phys = get_unique_directional_borders([x[0] for x in all_years_subset_phys.index])
            all_years_subset_phys = all_years_subset_phys[all_years_subset_phys.index.get_level_values('border').isin(unique_subset_borders_phys)]
            all_years_subset_phys = all_years_subset_phys[~all_years_subset_phys.index.duplicated(keep='first')]

            all_years_subset_phys = all_years_subset_phys.apply(lambda col: pd.to_numeric(col, errors='coerce').round(2) if col.dtype == 'object' or pd.api.types.is_numeric_dtype(col) else col).round(2)

            arrays_phys = [np.array([x[0] for x in all_years_subset_phys.index]), np.array([x[1] for x in all_years_subset_phys.index])]
            all_years_subset_phys.index = pd.MultiIndex.from_arrays(arrays_phys, names=("Zone", "Year"))
            all_years_subset_phys = all_years_subset_phys.reset_index()
            all_years_subset_phys['Zone'] = all_years_subset_phys['Zone'].mask(all_years_subset_phys['Zone'].duplicated(), '')

            all_years_border_phys_flow.to_csv(flow_comp_main_dir / "border_phys_flows_all_years.csv")
            save_chunked_files(all_years_subset_phys, flow_comp_main_dir, "border_phys_flows")

        # --- B1. All Years Net Positions Commercial Flow Total ---
        if base_year in net_position_comparison_df_yearly_comm_flow_total:
            all_years_np = pd.DataFrame(
                index=pd.MultiIndex(levels=[[],[]], codes=[[],[]], names=[u'bz', u'year']),
                columns=net_position_comparison_df_yearly_comm_flow_total[base_year].columns
            )

            for country in bidding_zones:
                for year in target_years:
                    if country in net_position_comparison_df_yearly_comm_flow_total[year].index.get_level_values('bidding_zone'):
                        for col in net_position_comparison_df_yearly_comm_flow_total[base_year].columns:
                            all_years_np.at[(country,year), col] = net_position_comparison_df_yearly_comm_flow_total[year].at[(country, "CFT"), col]
                            
                        prev_year = str(int(year) - 1)
                        if prev_year in target_years and country in net_position_comparison_df_yearly_comm_flow_total[prev_year].index.get_level_values('bidding_zone'):
                            all_years_np.at[(country,year), "Delta Median Year Prior"] = net_position_comparison_df_yearly_comm_flow_total[year].at[(country, "CFT"), "Median"] - net_position_comparison_df_yearly_comm_flow_total[prev_year].at[(country, "CFT"), "Median"]
                        
                        first_year = target_years[0]
                        if country in net_position_comparison_df_yearly_comm_flow_total[first_year].index.get_level_values('bidding_zone'):
                            all_years_np.at[(country,year), f"Delta Median {first_year}"] = net_position_comparison_df_yearly_comm_flow_total[year].at[(country, "CFT"), "Median"] - net_position_comparison_df_yearly_comm_flow_total[first_year].at[(country, "CFT"), "Median"]

            np_subset = all_years_np.loc[[x for x in all_years_np.index if x[0] in thesis_zones]]
            np_subset = np_subset.apply(lambda col: pd.to_numeric(col, errors='coerce').round(2) if col.dtype == 'object' or pd.api.types.is_numeric_dtype(col) else col).round(2)
            
            arrays = [np.array([x[0] for x in np_subset.index]), np.array([x[1] for x in np_subset.index])]
            np_subset.index = pd.MultiIndex.from_arrays(arrays, names=("Zone", "Year"))
            np_subset = np_subset.reset_index()
            np_subset['Zone'] = np_subset['Zone'].mask(np_subset['Zone'].duplicated(), '')

            save_chunked_files(np_subset, flow_comp_main_dir, "net_positions")

        # --- B2. All Years Net Positions Physical Flow ---
        if base_year in net_position_comparison_df_yearly_phys_flow:
            all_years_np_phys = pd.DataFrame(
                index=pd.MultiIndex(levels=[[],[]], codes=[[],[]], names=[u'bz', u'year']),
                columns=net_position_comparison_df_yearly_phys_flow[base_year].columns
            )

            for country in bidding_zones:
                for year in target_years:
                    if country in net_position_comparison_df_yearly_phys_flow[year].index.get_level_values('bidding_zone'):
                        for col in net_position_comparison_df_yearly_phys_flow[base_year].columns:
                            all_years_np_phys.at[(country,year), col] = net_position_comparison_df_yearly_phys_flow[year].at[(country, "Phys."), col]
                            
                        prev_year = str(int(year) - 1)
                        if prev_year in target_years and country in net_position_comparison_df_yearly_phys_flow[prev_year].index.get_level_values('bidding_zone'):
                            all_years_np_phys.at[(country,year), "Delta Median Year Prior"] = net_position_comparison_df_yearly_phys_flow[year].at[(country, "Phys."), "Median"] - net_position_comparison_df_yearly_phys_flow[prev_year].at[(country, "Phys."), "Median"]
                        
                        first_year = target_years[0]
                        if country in net_position_comparison_df_yearly_phys_flow[first_year].index.get_level_values('bidding_zone'):
                            all_years_np_phys.at[(country,year), f"Delta Median {first_year}"] = net_position_comparison_df_yearly_phys_flow[year].at[(country, "Phys."), "Median"] - net_position_comparison_df_yearly_phys_flow[first_year].at[(country, "Phys."), "Median"]

            np_subset_phys = all_years_np_phys.loc[[x for x in all_years_np_phys.index if x[0] in thesis_zones]]
            np_subset_phys = np_subset_phys.apply(lambda col: pd.to_numeric(col, errors='coerce').round(2) if col.dtype == 'object' or pd.api.types.is_numeric_dtype(col) else col).round(2)
            
            arrays_np_phys = [np.array([x[0] for x in np_subset_phys.index]), np.array([x[1] for x in np_subset_phys.index])]
            np_subset_phys.index = pd.MultiIndex.from_arrays(arrays_np_phys, names=("Zone", "Year"))
            np_subset_phys = np_subset_phys.reset_index()
            np_subset_phys['Zone'] = np_subset_phys['Zone'].mask(np_subset_phys['Zone'].duplicated(), '')

            all_years_np_phys.to_csv(flow_comp_main_dir / "net_positions_phys_all_years.csv")
            save_chunked_files(np_subset_phys, flow_comp_main_dir, "net_positions_phys")

        # --- C. All Years Border Flow Deltas (CFT-Phys.) ---
        if base_year in delta_individual_flows_comparison_df_yearly:
            borders_list = list(set([b for year in target_years for b, _ in delta_individual_flows_comparison_df_yearly[year].index]))
            
            all_years_border_deltas = pd.DataFrame(
                index=pd.MultiIndex(levels=[[],[]], codes=[[],[]], names=[u'border', u'year']),
                columns=delta_individual_flows_comparison_df_yearly[base_year].columns
            )

            for border in borders_list:
                for year in target_years:
                    if border in delta_individual_flows_comparison_df_yearly[year].index.get_level_values('bidding_zone'):
                        for col in delta_individual_flows_comparison_df_yearly[base_year].columns:
                            all_years_border_deltas.at[(border,year), col] = delta_individual_flows_comparison_df_yearly[year].at[(border, "CFT-Phys."), col]
                            
                        prev_year = str(int(year) - 1)
                        if prev_year in target_years and border in delta_individual_flows_comparison_df_yearly[prev_year].index.get_level_values('bidding_zone'):
                            all_years_border_deltas.at[(border,year), "Delta Median Year Prior"] = delta_individual_flows_comparison_df_yearly[year].at[(border, "CFT-Phys."), "Median"] - delta_individual_flows_comparison_df_yearly[prev_year].at[(border, "CFT-Phys."), "Median"]
                            
                        first_year = target_years[0]
                        if border in delta_individual_flows_comparison_df_yearly[first_year].index.get_level_values('bidding_zone'):
                            all_years_border_deltas.at[(border,year), f"Delta Median {first_year}"] = delta_individual_flows_comparison_df_yearly[year].at[(border, "CFT-Phys."), "Median"] - delta_individual_flows_comparison_df_yearly[first_year].at[(border, "CFT-Phys."), "Median"]

            subset_index_deltas = [x for x in all_years_border_deltas.index if x[0] in expanded_thesis_borders or "FR" in x[0] or "DE_LU" in x[0]]
            all_years_subset_deltas = all_years_border_deltas.loc[subset_index_deltas]
            
            # SMART DEDUPLICATION: Find direction that is positive in the base year
            base_deltas = delta_individual_flows_comparison_df_yearly[base_year].xs("CFT-Phys.", level="delta")
            seen_pairs = set()
            smart_unique_borders = []
            
            for b in [x[0] for x in all_years_subset_deltas.index]:
                parts = b.split('-')
                if len(parts) == 2:
                    canonical = frozenset(parts)
                    if canonical not in seen_pairs:
                        seen_pairs.add(canonical)
                        dir1, dir2 = f"{parts[0]}-{parts[1]}", f"{parts[1]}-{parts[0]}"
                        
                        val1 = base_deltas.loc[dir1, "Median"] if dir1 in base_deltas.index else -np.inf
                        val2 = base_deltas.loc[dir2, "Median"] if dir2 in base_deltas.index else -np.inf
                        
                        val1 = val1 if pd.notna(val1) else -np.inf
                        val2 = val2 if pd.notna(val2) else -np.inf

                        smart_unique_borders.append(dir2 if val2 > val1 else dir1)
                else:
                    smart_unique_borders.append(b)

            all_years_subset_deltas = all_years_subset_deltas[all_years_subset_deltas.index.get_level_values('border').isin(smart_unique_borders)]
            all_years_subset_deltas = all_years_subset_deltas[~all_years_subset_deltas.index.duplicated(keep='first')]

            all_years_subset_deltas = all_years_subset_deltas.apply(lambda col: pd.to_numeric(col, errors='coerce').round(2) if col.dtype == 'object' or pd.api.types.is_numeric_dtype(col) else col).round(2)
            
            arrays = [np.array([x[0] for x in all_years_subset_deltas.index]), np.array([x[1] for x in all_years_subset_deltas.index])]
            all_years_subset_deltas.index = pd.MultiIndex.from_arrays(arrays, names=("Zone", "Year"))
            all_years_subset_deltas = all_years_subset_deltas.reset_index()
            all_years_subset_deltas['Zone'] = all_years_subset_deltas['Zone'].mask(all_years_subset_deltas['Zone'].duplicated(), '')

            all_years_border_deltas.to_csv(flow_comp_main_dir / "border_flow_deltas_all_years.csv")
            save_chunked_files(all_years_subset_deltas, flow_comp_main_dir, "border_flow_deltas")

        # --- D. All Years Net Position Deltas (CFT-Phys.) ---
        if base_year in net_position_deltas_comparison_df_yearly:
            all_years_np_deltas = pd.DataFrame(
                index=pd.MultiIndex(levels=[[],[]], codes=[[],[]], names=[u'bz', u'year']),
                columns=net_position_deltas_comparison_df_yearly[base_year].columns
            )

            for country in bidding_zones:
                for year in target_years:
                    if country in net_position_deltas_comparison_df_yearly[year].index.get_level_values('bidding_zone'):
                        for col in net_position_deltas_comparison_df_yearly[base_year].columns:
                            all_years_np_deltas.at[(country,year), col] = net_position_deltas_comparison_df_yearly[year].at[(country, "CFT-Phys."), col]
                            
                        prev_year = str(int(year) - 1)
                        if prev_year in target_years and country in net_position_deltas_comparison_df_yearly[prev_year].index.get_level_values('bidding_zone'):
                            all_years_np_deltas.at[(country,year), "Delta Median Year Prior"] = net_position_deltas_comparison_df_yearly[year].at[(country, "CFT-Phys."), "Median"] - net_position_deltas_comparison_df_yearly[prev_year].at[(country, "CFT-Phys."), "Median"]
                        
                        first_year = target_years[0]
                        if country in net_position_deltas_comparison_df_yearly[first_year].index.get_level_values('bidding_zone'):
                            all_years_np_deltas.at[(country,year), f"Delta Median {first_year}"] = net_position_deltas_comparison_df_yearly[year].at[(country, "CFT-Phys."), "Median"] - net_position_deltas_comparison_df_yearly[first_year].at[(country, "CFT-Phys."), "Median"]

            np_delta_subset = all_years_np_deltas.loc[[x for x in all_years_np_deltas.index if x[0] in thesis_zones]]
            np_delta_subset = np_delta_subset.apply(lambda col: pd.to_numeric(col, errors='coerce').round(2) if col.dtype == 'object' or pd.api.types.is_numeric_dtype(col) else col).round(2)
            
            arrays = [np.array([x[0] for x in np_delta_subset.index]), np.array([x[1] for x in np_delta_subset.index])]
            np_delta_subset.index = pd.MultiIndex.from_arrays(arrays, names=("Zone", "Year"))
            np_delta_subset = np_delta_subset.reset_index()
            np_delta_subset['Zone'] = np_delta_subset['Zone'].mask(np_delta_subset['Zone'].duplicated(), '')

            save_chunked_files(np_delta_subset, flow_comp_main_dir, "net_position_deltas")


        # ==========================================
        # 5. TARGETED TIME SLICE EXTRACTIONS (e.g. DE_LU Deltas)
        # ==========================================
        target_year = target_years[-1]
        delta_comp_dir = config.output_dir / "flow_type_comparison" / target_year / "deltas" / "individual_flows"
        
        cft_phys_path = delta_comp_dir / "comm_flow_total_phys_flow_delta.csv"
        cfd_cft_path = delta_comp_dir / "comm_flow_dayahead_comm_flow_total_delta.csv"
        
        if cft_phys_path.exists() and cfd_cft_path.exists():
            net_position_deltas_extract = {
                "comm_flow_total_phys_flow_delta": pd.read_csv(cft_phys_path, index_col=0),
                "comm_flow_dayahead_comm_flow_total_delta": pd.read_csv(cfd_cft_path, index_col=0)
            }
            
            if "DE_LU" in net_position_deltas_extract["comm_flow_dayahead_comm_flow_total_delta"].columns:
                dates = list(net_position_deltas_extract["comm_flow_dayahead_comm_flow_total_delta"].sort_values(by=['DE_LU'], ascending=False).index[:100])
                dates += list(net_position_deltas_extract["comm_flow_dayahead_comm_flow_total_delta"].sort_values(by=['DE_LU']).index[:100])

                highest_delta_df = {}
                highest_delta_df["net_position_deltas"] = pd.DataFrame(index=dates)
                for d_name, d_df in net_position_deltas_extract.items():
                    highest_delta_df["net_position_deltas"][d_name] = d_df.loc[dates, "DE_LU"]


        # ==========================================
        # 6. TOTAL FLOW TEMPORAL AGGREGATIONS (TWh/Volume)
        # ==========================================
        logger.info("[Stats] Generating Flow Type Absolute Volume Comparisons...")
        flow_total_comp_dir = config.output_dir / "flow_type_comparison" / "total_flows"
        flow_total_comp_dir.mkdir(parents=True, exist_ok=True)

        flow_types_agg = {
            "comm_flow_dayahead": comm_individual_dayahead_flows_df_abs_yearly,
            "comm_flow_total": comm_individual_flows_df_abs_yearly,
            "phys_flow": phys_individual_flows_df_abs_yearly
        }

        for flow_type_agg, data_dict in flow_types_agg.items():
            agg_monthly_summed, agg_hourly_summed, agg_daily_summed = None, None, None
            agg_yearly = pd.Series(dtype=float)
            
            for year_str in target_years:
                df = data_dict[year_str]
                agg_yearly.loc[year_str] = df.sum().sum()
                
                monthly_year = df.groupby(pd.Grouper(freq='MS')).sum()
                monthly_year.index = monthly_year.index.month
                agg_monthly_summed = monthly_year if agg_monthly_summed is None else agg_monthly_summed + monthly_year
                    
                times = pd.DatetimeIndex(df.index)
                hourly_year = df.groupby([times.hour]).sum()
                agg_hourly_summed = hourly_year if agg_hourly_summed is None else agg_hourly_summed + hourly_year
                    
                daily_year = df.groupby([times.strftime('%w')]).sum()
                agg_daily_summed = daily_year if agg_daily_summed is None else agg_daily_summed + daily_year
                
            agg_daily_summed = agg_daily_summed.rename({"0": "7"}).sort_index()
            num_to_day = {"1":"Mon", "2":"Tue", "3":"Wed", "4":"Thu", "5":"Fri", "6":"Sat", "7":"Sun"}
            agg_daily_summed.index = [num_to_day[x] for x in agg_daily_summed.index]

            agg_yearly.to_csv(flow_total_comp_dir / f"{flow_type_agg}_agg_yearly.csv")
            agg_monthly_summed.to_csv(flow_total_comp_dir / f"{flow_type_agg}_agg_monthly.csv")
            agg_daily_summed.to_csv(flow_total_comp_dir / f"{flow_type_agg}_agg_daily.csv")
            agg_hourly_summed.to_csv(flow_total_comp_dir / f"{flow_type_agg}_agg_hourly.csv")


        # ==========================================
        # 7. NORDIC-SPECIFIC VOLUME TRENDS (PHYSICAL vs CFT)
        # ==========================================
        logger.info("[Stats] Generating Nordic-specific trade volume trends...")
        
        nordic_zones = [
            'NO_1', 'NO_2', 'NO_3', 'NO_4', 'NO_5', 
            'SE_1', 'SE_2', 'SE_3', 'SE_4', 
            'FI', 'DK_1', 'DK_2'
        ]
        
        nordic_trends = []

        for year_str in target_years:
            # --- 1. Nordic Internal Physical Border Flows ---
            p_ind = phys_individual_flows_df_yearly[year_str]
            phys_unique = get_unique_directional_borders(p_ind.columns) if not p_ind.empty else []
            nordic_internal_borders_phys = [
                b for b in phys_unique 
                if b.split('-')[0] in nordic_zones and b.split('-')[1] in nordic_zones
            ]
            
            if not p_ind.empty and len(nordic_internal_borders_phys) > 0:
                hourly_nordic_internal_flow_phys = abs(p_ind[nordic_internal_borders_phys]).sum(axis=1)
                nordic_phys_flow_median = hourly_nordic_internal_flow_phys.median()
                nordic_phys_flow_mean = hourly_nordic_internal_flow_phys.mean()
            else:
                nordic_phys_flow_median = np.nan
                nordic_phys_flow_mean = np.nan

            # --- 2. Nordic Internal CFT Border Flows ---
            c_ind = comm_individual_flows_df_yearly[year_str]
            comm_unique = get_unique_directional_borders(c_ind.columns) if not c_ind.empty else []
            nordic_internal_borders_cft = [
                b for b in comm_unique 
                if b.split('-')[0] in nordic_zones and b.split('-')[1] in nordic_zones
            ]
            
            if not c_ind.empty and len(nordic_internal_borders_cft) > 0:
                hourly_nordic_internal_flow_cft = abs(c_ind[nordic_internal_borders_cft]).sum(axis=1)
                nordic_cft_flow_median = hourly_nordic_internal_flow_cft.median()
                nordic_cft_flow_mean = hourly_nordic_internal_flow_cft.mean()
            else:
                nordic_cft_flow_median = np.nan
                nordic_cft_flow_mean = np.nan

            # --- 3. Nordic Physical Net Positions ---
            phys_np_dict = {
                bz: flow_dfs_yearly[year_str][bz]["Net Export"] 
                for bz in nordic_zones 
                if bz in flow_dfs_yearly[year_str] and "Net Export" in flow_dfs_yearly[year_str][bz].columns
            }
            phys_np_df = pd.DataFrame(phys_np_dict)
            
            if not phys_np_df.empty:
                hourly_nordic_np_vol_phys = (abs(phys_np_df).sum(axis=1)) / 2.0
                nordic_phys_np_median = hourly_nordic_np_vol_phys.median()
                nordic_phys_np_mean = hourly_nordic_np_vol_phys.mean()
            else:
                nordic_phys_np_median = np.nan
                nordic_phys_np_mean = np.nan

            # --- 4. Nordic CFT Net Positions ---
            cft_np_dict = {
                bz: comm_exchange_dfs_yearly[year_str][bz]["Net Export"] 
                for bz in nordic_zones 
                if bz in comm_exchange_dfs_yearly[year_str] and "Net Export" in comm_exchange_dfs_yearly[year_str][bz].columns
            }
            cft_np_df = pd.DataFrame(cft_np_dict)
            
            if not cft_np_df.empty:
                hourly_nordic_np_vol_cft = (abs(cft_np_df).sum(axis=1)) / 2.0
                nordic_cft_np_median = hourly_nordic_np_vol_cft.median()
                nordic_cft_np_mean = hourly_nordic_np_vol_cft.mean()
            else:
                nordic_cft_np_median = np.nan
                nordic_cft_np_mean = np.nan

            # Append to trends
            nordic_trends.append({
                "Year": year_str,
                "Nordic_Internal_Phys_Flow_Median_MW": round(nordic_phys_flow_median, 2),
                "Nordic_Internal_Phys_Flow_Mean_MW": round(nordic_phys_flow_mean, 2),
                "Nordic_Internal_CFT_Flow_Median_MW": round(nordic_cft_flow_median, 2),
                "Nordic_Internal_CFT_Flow_Mean_MW": round(nordic_cft_flow_mean, 2),
                "Nordic_Phys_Net_Position_Volume_Median_MW": round(nordic_phys_np_median, 2),
                "Nordic_Phys_Net_Position_Volume_Mean_MW": round(nordic_phys_np_mean, 2),
                "Nordic_CFT_Net_Position_Volume_Median_MW": round(nordic_cft_np_median, 2),
                "Nordic_CFT_Net_Position_Volume_Mean_MW": round(nordic_cft_np_mean, 2)
            })

        # Export to CSV
        df_nordic_trends = pd.DataFrame(nordic_trends)
        df_nordic_trends.to_csv(flow_total_comp_dir / "nordic_trade_volume_trends_all_years.csv", index=False)

        # ==========================================
        # 7b. GRANULAR NORDIC INDIVIDUAL BORDER STATS (PHYSICAL vs CFT)
        # ==========================================
        logger.info("[Stats] Generating granular individual Nordic border statistics...")
        
        nordic_individual_stats = []

        # Find all unique Nordic internal borders across the entire time series
        all_nordic_borders = set()
        for year_str in target_years:
            p_ind = phys_individual_flows_df_yearly[year_str]
            phys_unique = get_unique_directional_borders(p_ind.columns) if not p_ind.empty else []
            for b in phys_unique:
                parts = b.split('-')
                if parts[0] in nordic_zones and parts[1] in nordic_zones:
                    all_nordic_borders.add(b)
        
        # Sort borders alphabetically for clear reference layout
        sorted_nordic_borders = sorted(list(all_nordic_borders))

        for border in sorted_nordic_borders:
            for year_str in target_years:
                p_ind = phys_individual_flows_df_yearly[year_str]
                c_ind = comm_individual_flows_df_yearly[year_str]
                
                # --- Extract Physical Metrics for this specific border ---
                if not p_ind.empty and border in p_ind.columns:
                    s_phys = p_ind[border].dropna()
                    phys_median = s_phys.median()
                    phys_mean = s_phys.mean()
                    phys_std = s_phys.std()
                    phys_min = s_phys.min()
                    phys_max = s_phys.max()
                else:
                    phys_median, phys_mean, phys_std, phys_min, phys_max = [np.nan] * 5

                # --- Extract Commercial (CFT) Metrics for this specific border ---
                if not c_ind.empty and border in c_ind.columns:
                    s_cft = c_ind[border].dropna()
                    cft_median = s_cft.median()
                    cft_mean = s_cft.mean()
                    cft_std = s_cft.std()
                    cft_min = s_cft.min()
                    cft_max = s_cft.max()
                else:
                    cft_median, cft_mean, cft_std, cft_min, cft_max = [np.nan] * 5
                
                # Calculate the absolute difference between the medians as a metric of distortion
                median_absolute_divergence = abs(cft_median - phys_median) if not (np.isnan(cft_median) or np.isnan(phys_median)) else np.nan

                nordic_individual_stats.append({
                    "Border": border,
                    "Year": year_str,
                    "Phys_Flow_Median_MW": round(phys_median, 2) if not np.isnan(phys_median) else np.nan,
                    "Phys_Flow_Mean_MW": round(phys_mean, 2) if not np.isnan(phys_mean) else np.nan,
                    "Phys_Flow_Std_Dev_MW": round(phys_std, 2) if not np.isnan(phys_std) else np.nan,
                    "Phys_Flow_Min_MW": round(phys_min, 2) if not np.isnan(phys_min) else np.nan,
                    "Phys_Flow_Max_MW": round(phys_max, 2) if not np.isnan(phys_max) else np.nan,
                    "CFT_Flow_Median_MW": round(cft_median, 2) if not np.isnan(cft_median) else np.nan,
                    "CFT_Flow_Mean_MW": round(cft_mean, 2) if not np.isnan(cft_mean) else np.nan,
                    "CFT_Flow_Std_Dev_MW": round(cft_std, 2) if not np.isnan(cft_std) else np.nan,
                    "CFT_Flow_Min_MW": round(cft_min, 2) if not np.isnan(cft_min) else np.nan,
                    "CFT_Flow_Max_MW": round(cft_max, 2) if not np.isnan(cft_max) else np.nan,
                    "Median_Absolute_Divergence_MW": round(median_absolute_divergence, 2) if not np.isnan(median_absolute_divergence) else np.nan
                })

        # Export Granular Data to a separate CSV
        df_granular_nordic = pd.DataFrame(nordic_individual_stats)
        
        # Apply scannability layout formatting: blank out repeating border names
        df_granular_nordic_clean = df_granular_nordic.copy()
        df_granular_nordic_clean['Border'] = df_granular_nordic_clean['Border'].mask(df_granular_nordic_clean['Border'].duplicated(), '')
        
        df_granular_nordic_clean.to_csv(flow_total_comp_dir / "nordic_individual_border_stats_all_years.csv", index=False)
        
        # ==========================================
        # 8. SYSTEM-WIDE VOLUME TRENDS (PHYSICAL vs CFT)
        # ==========================================
        logger.info("[Stats] Generating System-wide trade volume trends...")
        
        system_trends = []

        for year_str in target_years:
            # --- 1. System-Wide Physical & CFT Border Flows ---
            p_ind = phys_individual_flows_df_yearly[year_str]
            phys_unique = get_unique_directional_borders(p_ind.columns) if not p_ind.empty else []
            
            if not p_ind.empty and len(phys_unique) > 0:
                hourly_sys_flow_phys = abs(p_ind[phys_unique]).sum(axis=1)
                sys_phys_flow_median = hourly_sys_flow_phys.median()
                sys_phys_flow_mean = hourly_sys_flow_phys.mean()
            else:
                sys_phys_flow_median, sys_phys_flow_mean = np.nan, np.nan

            c_ind = comm_individual_flows_df_yearly[year_str]
            comm_unique = get_unique_directional_borders(c_ind.columns) if not c_ind.empty else []
            
            if not c_ind.empty and len(comm_unique) > 0:
                hourly_sys_flow_cft = abs(c_ind[comm_unique]).sum(axis=1)
                sys_cft_flow_median = hourly_sys_flow_cft.median()
                sys_cft_flow_mean = hourly_sys_flow_cft.mean()
            else:
                sys_cft_flow_median, sys_cft_flow_mean = np.nan, np.nan

            # --- 2. System-Wide Physical Net Positions ---
            phys_np_dict_sys = {
                bz: flow_dfs_yearly[year_str][bz]["Net Export"] 
                for bz in thesis_zones 
                if bz in flow_dfs_yearly[year_str] and "Net Export" in flow_dfs_yearly[year_str][bz].columns
            }
            phys_np_df_sys = pd.DataFrame(phys_np_dict_sys)
            
            if not phys_np_df_sys.empty:
                hourly_sys_np_phys = (abs(phys_np_df_sys).sum(axis=1)) / 2.0
                sys_phys_np_median = hourly_sys_np_phys.median()
                sys_phys_np_mean = hourly_sys_np_phys.mean()
            else:
                sys_phys_np_median, sys_phys_np_mean = np.nan, np.nan

            # --- 3. System-Wide CFT Net Positions ---
            cft_np_dict_sys = {
                bz: comm_exchange_dfs_yearly[year_str][bz]["Net Export"] 
                for bz in thesis_zones 
                if bz in comm_exchange_dfs_yearly[year_str] and "Net Export" in comm_exchange_dfs_yearly[year_str][bz].columns
            }
            cft_np_df_sys = pd.DataFrame(cft_np_dict_sys)
            
            if not cft_np_df_sys.empty:
                hourly_sys_np_cft = (abs(cft_np_df_sys).sum(axis=1)) / 2.0
                sys_cft_np_median = hourly_sys_np_cft.median()
                sys_cft_np_mean = hourly_sys_np_cft.mean()
            else:
                sys_cft_np_median, sys_cft_np_mean = np.nan, np.nan

            system_trends.append({
                "Year": year_str,
                "System_Phys_Flow_Median_MW": round(sys_phys_flow_median, 2),
                "System_Phys_Flow_Mean_MW": round(sys_phys_flow_mean, 2),
                "System_CFT_Flow_Median_MW": round(sys_cft_flow_median, 2),
                "System_CFT_Flow_Mean_MW": round(sys_cft_flow_mean, 2),
                "System_Phys_Net_Pos_Median_MW": round(sys_phys_np_median, 2),
                "System_Phys_Net_Pos_Mean_MW": round(sys_phys_np_mean, 2),
                "System_CFT_Net_Pos_Median_MW": round(sys_cft_np_median, 2),
                "System_CFT_Net_Pos_Mean_MW": round(sys_cft_np_mean, 2)
            })

        df_system_trends = pd.DataFrame(system_trends)
        df_system_trends.to_csv(flow_total_comp_dir / "system_trade_volume_trends_all_years.csv", index=False)

        logger.info("=== STATISTICAL ANALYSIS COMPLETE ===")

    finally:
        # Guarantee restoration of configuration bounds to ensure downstream pipeline safety
        config.year = original_year
        config.start = original_start
        config.end = original_end
        config.time_index = original_time_index