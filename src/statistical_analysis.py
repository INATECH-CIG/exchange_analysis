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

            # --- A. Commercial Flow Total (CFT) ---
            cft_dir = config.get_output_path("comm_flow_total_bidding_zones")
            for country in sorted(bidding_zones):
                df = io.load(cft_dir / f"{country}_comm_flow_total_bidding_zones.csv", "processed_commercial_flows", config, bz=country)
                if df is None or df.empty: continue
                
                # STATISTICAL PURITY FILTER: Erase imputed/patched data using metadata
                if "gap_filling_method" in df.columns:
                    dirty_mask = (df["gap_filling_method"] != "None") & df["gap_filling_method"].notna()
                    num_cols = df.select_dtypes(include=[np.number]).columns
                    df.loc[dirty_mask, num_cols] = np.nan

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
                    
                # STATISTICAL PURITY FILTER
                if "gap_filling_method" in df.columns:
                    dirty_mask = (df["gap_filling_method"] != "None") & df["gap_filling_method"].notna()
                    num_cols = df.select_dtypes(include=[np.number]).columns
                    df.loc[dirty_mask, num_cols] = np.nan

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
                
                # STATISTICAL PURITY FILTER
                if "gap_filling_method" in df.columns:
                    dirty_mask = (df["gap_filling_method"] != "None") & df["gap_filling_method"].notna()
                    num_cols = df.select_dtypes(include=[np.number]).columns
                    df.loc[dirty_mask, num_cols] = np.nan

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
                        dirty_mask = (df["gap_filling_method"] != "None") & df["gap_filling_method"].notna()
                        num_cols = df.select_dtypes(include=[np.number]).columns
                        df.loc[dirty_mask, num_cols] = np.nan

                    gen_load_dfs_yearly[year_str][country] = df.copy()
                    if "Net Export" in df.columns:
                        gen_load_net_position_dfs_yearly[year_str][country] = df["Net Export"]


        # ==========================================
        # 2. CROSS-YEAR AGGREGATION & MATRICES
        # ==========================================
        logger.info("[Stats] Compiling Structural Flow Matrices...")

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
                    border_key = f"{country_again}-{country}"
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

            phys_individual_flows_df_yearly[year_str] = p_ind
            comm_individual_flows_df_yearly[year_str] = c_ind
            comm_individual_dayahead_flows_df_yearly[year_str] = da_ind

            phys_individual_flows_df_abs_yearly[year_str] = abs(p_ind).sum(axis=1)
            comm_individual_flows_df_abs_yearly[year_str] = abs(c_ind).sum(axis=1)
            comm_individual_dayahead_flows_df_abs_yearly[year_str] = abs(da_ind).sum(axis=1)


        # ==========================================
        # 3. DETAILED STATISTICAL METRICS (Medians, Quantiles)
        # ==========================================
        logger.info("[Stats] Computing detailed summary statistics and deltas...")

        flow_comparison_df_yearly = {}
        delta_individual_flows_comparison_df_yearly = {}
        net_position_comparison_df_yearly = {}
        
        flow_comparison_df_yearly_comm_flow_total = {}
        net_position_comparison_df_yearly_comm_flow_total = {}

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
                             "Pos. Val. Share", "Neg. Val. Median",
                             "Pos. Val. Median", "Abs. Val. Median", "Corr. CFD",
                             "Corr. CFT", "Corr. Phys."]
                )

                for border in p_ind.columns:
                    for flow_type, df in [("CFD", da_ind), ("CFT", c_ind), ("Phys.", p_ind)]:
                        if border not in df.columns: continue
                        
                        s = df[border]
                        total_len = len(df)
                        if total_len == 0: continue

                        flow_comp_df.at[(border, flow_type), "Median"] = s.median()
                        flow_comp_df.at[(border, flow_type), "Lower Q."] = s.quantile(0.25)
                        flow_comp_df.at[(border, flow_type), "Upper Q."] = s.quantile(0.75)
                        flow_comp_df.at[(border, flow_type), "Std. Dev."] = s.std()
                        flow_comp_df.at[(border, flow_type), "Min Val."] = s.min()
                        flow_comp_df.at[(border, flow_type), "Max Val."] = s.max()
                        flow_comp_df.at[(border, flow_type), "Neg. Val. Share"] = len(s[s < 0]) / total_len
                        flow_comp_df.at[(border, flow_type), "Pos. Val. Share"] = len(s[s >= 0]) / total_len
                        flow_comp_df.at[(border, flow_type), "Neg. Val. Median"] = s[s < 0].median()
                        flow_comp_df.at[(border, flow_type), "Pos. Val. Median"] = s[s >= 0].median()
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

                # Output Top 20 Subsets
                cft_subset = flow_comp_df.loc[[x for x in flow_comp_df.index if "CFT" in x[1]]]
                
                busiest_borders = flow_comp_df.loc[[x for x in flow_comp_df.index if "CFD" not in x[1]]].sort_values(by="Abs. Val. Median", ascending=False)[:20].apply(pd.to_numeric, errors='coerce').round(2)
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
                             "Pos. Val. Share", "Neg. Val. Median",
                             "Pos. Val. Median", "Abs. Val. Median"]
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
                        total_len = len(d_df)
                        if total_len == 0: continue

                        delta_comp_df.at[(border, d_name), "Median"] = s.median()
                        delta_comp_df.at[(border, d_name), "Lower Q."] = s.quantile(0.25)
                        delta_comp_df.at[(border, d_name), "Upper Q."] = s.quantile(0.75)
                        delta_comp_df.at[(border, d_name), "Std. Dev."] = s.std()
                        delta_comp_df.at[(border, d_name), "Min Val."] = s.min()
                        delta_comp_df.at[(border, d_name), "Max Val."] = s.max()
                        delta_comp_df.at[(border, d_name), "Neg. Val. Share"] = len(s[s < 0]) / total_len
                        delta_comp_df.at[(border, d_name), "Pos. Val. Share"] = len(s[s >= 0]) / total_len
                        delta_comp_df.at[(border, d_name), "Neg. Val. Median"] = s[s < 0].median()
                        delta_comp_df.at[(border, d_name), "Pos. Val. Median"] = s[s >= 0].median()
                        delta_comp_df.at[(border, d_name), "Abs. Val. Median"] = s.abs().median()

                delta_comp_df.to_csv(delta_comp_dir / f"border_flow_deltas_stats_{year_str}.csv")
                delta_individual_flows_comparison_df_yearly[year_str] = delta_comp_df
                
                biggest_deltas = delta_comp_df.sort_values(by="Abs. Val. Median", ascending=False)[:20].apply(pd.to_numeric, errors='coerce').round(2)
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
                         "Pos. Val. Share", "Neg. Val. Median",
                         "Pos. Val. Median", "Abs. Val. Median"]
            )

            for bz in bidding_zones:
                for flow_type, df in [("SDAC", sdac_np), ("CFD", cfd_np), ("CFT", cft_np), ("Phys.", phys_np), ("Gen_Load", gen_np)]:
                    if bz not in df.columns: continue
                    s = df[bz]
                    total_len = len(df)
                    if total_len == 0: continue

                    np_comp_df.at[(bz, flow_type), "Median"] = s.median()
                    np_comp_df.at[(bz, flow_type), "Lower Q."] = s.quantile(0.25)
                    np_comp_df.at[(bz, flow_type), "Upper Q."] = s.quantile(0.75)
                    np_comp_df.at[(bz, flow_type), "Std. Dev."] = s.std()
                    np_comp_df.at[(bz, flow_type), "Min Val."] = s.min()
                    np_comp_df.at[(bz, flow_type), "Max Val."] = s.max()
                    np_comp_df.at[(bz, flow_type), "Neg. Val. Share"] = len(s[s < 0]) / total_len
                    np_comp_df.at[(bz, flow_type), "Pos. Val. Share"] = len(s[s >= 0]) / total_len
                    np_comp_df.at[(bz, flow_type), "Neg. Val. Median"] = s[s < 0].median()
                    np_comp_df.at[(bz, flow_type), "Pos. Val. Median"] = s[s >= 0].median()
                    np_comp_df.at[(bz, flow_type), "Abs. Val. Median"] = s.abs().median()

            np_comp_df.to_csv(flow_comp_dir / f"net_positions_stats_{year_str}.csv")
            net_position_comparison_df_yearly[year_str] = np_comp_df
            net_position_comparison_df_yearly_comm_flow_total[year_str] = np_comp_df.loc[[x for x in np_comp_df.index if "CFT" in x[1]]]

            np_subset = np_comp_df.loc[[x for x in np_comp_df.index if "CFT" in x[1] or "Phys." in x[1]]]
            most_negative = np_subset.sort_values(by="Neg. Val. Median")[:20].apply(pd.to_numeric, errors='coerce').round(2)
            most_positive = np_subset.sort_values(by="Pos. Val. Median", ascending=False)[:20].apply(pd.to_numeric, errors='coerce').round(2)
            
            most_negative.to_csv(flow_comp_dir / f"most_negative_net_positions_{year_str}.csv")
            most_positive.to_csv(flow_comp_dir / f"most_positive_net_positions_{year_str}.csv")


        # ==========================================
        # 4. CROSS-YEAR COMPILATIONS (ALL YEARS SUMMARY)
        # ==========================================
        logger.info("[Stats] Compiling 'All Years' Summary Tables...")
        flow_comp_main_dir = config.output_dir / "flow_type_comparison"
        flow_comp_main_dir.mkdir(parents=True, exist_ok=True)
        
        base_year = target_years[-1]

        # --- A. All Years Border Commercial Flow Total ---
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
                                
                                # Delta compared to first year in target_years (e.g. 2021)
                                first_year = target_years[0]
                                if border in flow_comparison_df_yearly_comm_flow_total[first_year].index.get_level_values('border'):
                                    all_years_border_comm_flow_total.at[(border,year), f"Delta Median {first_year}"] = flow_comparison_df_yearly_comm_flow_total[year].at[(border, "CFT"), "Median"] - flow_comparison_df_yearly_comm_flow_total[first_year].at[(border, "CFT"), "Median"]

            # ==========================================
            # GUARANTEE EXPLICIT PAPER BORDERS
            # ==========================================
            paper_borders = [
                # Historical / Loop Flow Corridors (Table 1)
                "FR-DE_LU",
                "CZ-PL", 
                "FR-CH", 
                "AT-DE_LU",
                
                # Benelux Squeeze (Table 1)
                "BE-NL", 
                "NL-DE_LU", 
                "BE-FR", 
                
                # Nordics (Internal & Cross-Border)
                "SE_2-SE_3", 
                "SE_3-SE_4", 
                "NO_1-NO_3", 
                "NO_4-NO_3", 
                "NO_5-NO_3", 
                "NO_3-SE_2", 
                
                # France's other borders
                "FR-GB", 
                "FR-IT_NORD", 
                "FR-ES", 
                
                # Italy & South
                "IT_NORD-IT_CNOR", 
                "AT-IT_NORD", 
                
                # Balkans (Least Correlated)
                "RS-MK", 
                "RS-ME", 
                
                # Highly Correlated / HVDC
                "BE-GB", 
                "NL-DK_1", 
                "EE-FI", 
                "DK_2-DE_LU", 
                "GB-DK_1", 
                "NO_2-DE_LU"
            ]

            # Grab explicit borders + keep broader general ones for robust files
            subset_index = [
                x for x in all_years_border_comm_flow_total.index 
                if x[0] in paper_borders or "FR" in x[0] or "DE_LU" in x[0]
            ]
            all_years_subset = all_years_border_comm_flow_total.loc[subset_index]
            
            # Prevent accidental duplicates
            all_years_subset = all_years_subset[~all_years_subset.index.duplicated(keep='first')]

            all_years_subset = all_years_subset.apply(lambda col: pd.to_numeric(col, errors='coerce').round(2) if col.dtype == 'object' or pd.api.types.is_numeric_dtype(col) else col).round(2)
            
            if "Corr. CFT" in all_years_subset.columns:
                all_years_subset = all_years_subset.drop(columns=["Corr. CFT"])

            arrays = [np.array([x[0] for x in all_years_subset.index]), np.array([x[1] for x in all_years_subset.index])]
            all_years_subset.index = pd.MultiIndex.from_arrays(arrays, names=("Zone", "Year"))
            all_years_subset = all_years_subset.reset_index()
            all_years_subset['Zone'] = all_years_subset['Zone'].mask(all_years_subset['Zone'].duplicated(), '')

            all_years_border_comm_flow_total.to_csv(flow_comp_main_dir / "border_flows_all_years.csv")
            all_years_subset.to_csv(flow_comp_main_dir / "top_border_flows_all_years.csv")
            all_years_subset.iloc[:30].to_csv(flow_comp_main_dir / "top_border_flows_all_years_1.csv")
            all_years_subset.iloc[30:60].to_csv(flow_comp_main_dir / "top_border_flows_all_years_2.csv")
            all_years_subset.iloc[60:90].to_csv(flow_comp_main_dir / "top_border_flows_all_years_3.csv")


        # --- B. All Years Net Positions Commercial Flow Total ---
        if base_year in net_position_comparison_df_yearly_comm_flow_total:
            all_years_np = pd.DataFrame(
                index=pd.MultiIndex(levels=[[],[]], codes=[[],[]], names=[u'bz', u'year']),
                columns=net_position_comparison_df_yearly_comm_flow_total[base_year].columns
            )

            for country in bidding_zones:
                for year in target_years:
                    for col in net_position_comparison_df_yearly_comm_flow_total[base_year].columns:
                        if country in net_position_comparison_df_yearly_comm_flow_total[year].index.get_level_values('bidding_zone'):
                            all_years_np.at[(country,year), col] = net_position_comparison_df_yearly_comm_flow_total[year].at[(country, "CFT"), col]
                            
                            prev_year = str(int(year) - 1)
                            if prev_year in target_years:
                                all_years_np.at[(country,year), "Delta Median Year Prior"] = net_position_comparison_df_yearly_comm_flow_total[year].at[(country, "CFT"), "Median"] - net_position_comparison_df_yearly_comm_flow_total[prev_year].at[(country, "CFT"), "Median"]
                                first_year = target_years[0]
                                all_years_np.at[(country,year), f"Delta Median {first_year}"] = net_position_comparison_df_yearly_comm_flow_total[year].at[(country, "CFT"), "Median"] - net_position_comparison_df_yearly_comm_flow_total[first_year].at[(country, "CFT"), "Median"]

            # ==========================================
            # GUARANTEE EXPLICIT PAPER ZONES
            # ==========================================
            thesis_zones = [
                # Core & Western Europe
                "DE_LU", "FR", "GB", "BE", "CH", "AT", "NL",
                
                # Nordics & Baltics
                "NO_1", "NO_2", "NO_3", "NO_4", "NO_5", 
                "SE_2", "SE_3", "SE_4",
                "DK_1", "DK_2", "FI", "EE",
                
                # South / Italy / Iberia
                "IT_NORD", "IT_CNOR", "ES",
                
                # East / Balkans
                "CZ", "PL", "RS", "MK", "ME"
            ]
            
            np_subset = all_years_np.loc[[x for x in all_years_np.index if x[0] in thesis_zones]]

            np_subset = np_subset.apply(lambda col: pd.to_numeric(col, errors='coerce').round(2) if col.dtype == 'object' or pd.api.types.is_numeric_dtype(col) else col).round(2)
            
            arrays = [np.array([x[0] for x in np_subset.index]), np.array([x[1] for x in np_subset.index])]
            np_subset.index = pd.MultiIndex.from_arrays(arrays, names=("Zone", "Year"))
            np_subset = np_subset.reset_index()
            np_subset['Zone'] = np_subset['Zone'].mask(np_subset['Zone'].duplicated(), '')

            np_subset.to_csv(flow_comp_main_dir / "top_net_positions_all_years.csv")
            np_subset.iloc[:30].to_csv(flow_comp_main_dir / "top_net_positions_all_years_1.csv")
            np_subset.iloc[30:60].to_csv(flow_comp_main_dir / "top_net_positions_all_years_2.csv")


        # ==========================================
        # 5. TARGETED TIME SLICE EXTRACTIONS (e.g. DE_LU Deltas)
        # ==========================================
        target_year = target_years[-1]
        delta_comp_dir = config.output_dir / "flow_type_comparison" / target_year / "deltas"
        
        cft_phys_path = delta_comp_dir / "comm_flow_total_phys_flow_delta.csv"
        cfd_cft_path = delta_comp_dir / "comm_flow_dayahead_comm_flow_total_delta.csv"
        
        if cft_phys_path.exists() and cfd_cft_path.exists():
            net_position_deltas = {
                "comm_flow_total_phys_flow_delta": pd.read_csv(cft_phys_path, index_col=0),
                "comm_flow_dayahead_comm_flow_total_delta": pd.read_csv(cfd_cft_path, index_col=0)
            }
            
            if "DE_LU" in net_position_deltas["comm_flow_dayahead_comm_flow_total_delta"].columns:
                dates = list(net_position_deltas["comm_flow_dayahead_comm_flow_total_delta"].sort_values(by=['DE_LU'], ascending=False).index[:100])
                dates += list(net_position_deltas["comm_flow_dayahead_comm_flow_total_delta"].sort_values(by=['DE_LU']).index[:100])

                highest_delta_df = {}
                highest_delta_df["net_position_deltas"] = pd.DataFrame(index=dates)
                for d_name, d_df in net_position_deltas.items():
                    highest_delta_df["net_position_deltas"][d_name] = d_df.loc[dates, "DE_LU"]


        # ==========================================
        # 6. TOTAL FLOW TEMPORAL AGGREGATIONS (TWh/Volume)
        # ==========================================
        logger.info("[Stats] Generating Flow Type Absolute Volume Comparisons...")
        flow_total_comp_dir = config.output_dir / "flow_type_comparison" / "total_flows"
        flow_total_comp_dir.mkdir(parents=True, exist_ok=True)

        flow_types = {
            "comm_flow_dayahead": comm_individual_dayahead_flows_df_abs_yearly,
            "comm_flow_total": comm_individual_flows_df_abs_yearly,
            "phys_flow": phys_individual_flows_df_abs_yearly
        }

        for flow_type, data_dict in flow_types.items():
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

            agg_yearly.to_csv(flow_total_comp_dir / f"{flow_type}_agg_yearly.csv")
            agg_monthly_summed.to_csv(flow_total_comp_dir / f"{flow_type}_agg_monthly.csv")
            agg_daily_summed.to_csv(flow_total_comp_dir / f"{flow_type}_agg_daily.csv")
            agg_hourly_summed.to_csv(flow_total_comp_dir / f"{flow_type}_agg_hourly.csv")

        logger.info("=== STATISTICAL ANALYSIS COMPLETE ===")

    finally:
        # Guarantee restoration of configuration bounds to ensure downstream pipeline safety
        config.year = original_year
        config.start = original_start
        config.end = original_end
        config.time_index = original_time_index