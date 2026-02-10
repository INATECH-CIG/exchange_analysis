import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from matplotlib import colors
from config import PipelineConfig

def generate_plots(config: PipelineConfig):
    print("Starting Plotting Phase...")
    
    # Paths
    year_str = str(config.year)
    totals_base = config.output_dir / "annual_totals_per_method" / year_str
    
    # Input Dirs (Where analysis.py saved the results)
    dirs = {
        "imp_bz": totals_base / "import/per_bidding_zone",
        "exp_bz": totals_base / "export/per_bidding_zone",
        "imp_type": totals_base / "import/per_type",
        "exp_type": totals_base / "export/per_type",
        "imp_agg": totals_base / "import/per_agg_type"
    }
    
    # Plot Output Dirs
    plot_dirs = {
        "imp_bz": totals_base / "import/per_bidding_zone/Plots",
        "exp_bz": totals_base / "export/per_bidding_zone/Plots",
        "imp_type": totals_base / "import/per_type/Plots",
        "exp_type": totals_base / "export/per_type/Plots",
        "imp_agg": totals_base / "import/per_agg_type/Plots",
        "main_imp": totals_base / "main_importers/Plots",
        "main_exp": totals_base / "main_exporters/Plots",
    }
    for p in plot_dirs.values(): p.mkdir(parents=True, exist_ok=True)

    # Load Data
    data = {k: {} for k in dirs.keys()}
    for k, path in dirs.items():
        for bz in config.zones:
            p = path / f"{bz}_annual_totals_import_per_bidding_zone_{year_str}.csv"
            if "export" in k: p = path / f"{bz}_annual_totals_export_per_bidding_zone_{year_str}.csv"
            if "per_type" in str(path) and "agg" not in str(path): 
                p = path / f"{bz}_annual_totals_{'import' if 'import' in str(path) else 'export'}_per_type_{year_str}.csv"
            if "agg" in str(path):
                p = path / f"{bz}_annual_totals_import_per_agg_type_{year_str}.csv"
                
            if p.exists():
                # Transpose back: CSVs are saved as [Category x Method], we want [Method x Category]
                data[k][bz] = pd.read_csv(p, index_col=0).T

    # --- 1. Per-Country Generic Plots ---
    for bz in config.zones:
        _plot_generic(data["imp_bz"].get(bz), "TWh", f"{bz} Import", plot_dirs["imp_bz"] / f"{bz}_import_country.png")
        _plot_generic(data["exp_bz"].get(bz), "TWh", f"{bz} Export", plot_dirs["exp_bz"] / f"{bz}_export_country.png")
        _plot_generic(data["imp_type"].get(bz), "TWh", f"{bz} Import Type", plot_dirs["imp_type"] / f"{bz}_import_type.png")
        _plot_generic(data["exp_type"].get(bz), "TWh", f"{bz} Export Type", plot_dirs["exp_type"] / f"{bz}_export_type.png")
        _plot_generic(data["imp_agg"].get(bz), "TWh", f"{bz} Import Agg Type", plot_dirs["imp_agg"] / f"{bz}_import_agg.png")

    # --- 2. Main Summary Plots ---
    ref = "Netted CFT"
    
    # Helper to get sum
    def get_sum(d, bz):
        if bz not in d: return 0
        df = d[bz]
        return df.loc[ref].sum() if ref in df.index else 0

    top_imp = sorted(data["imp_bz"].keys(), key=lambda x: get_sum(data["imp_bz"], x), reverse=True)[:6]
    top_exp = sorted(data["exp_bz"].keys(), key=lambda x: get_sum(data["exp_bz"], x), reverse=True)[:6]
    
    subset = ["Netted CFT", "Pooled Net CFT", "DC Flow Tracing", "AC Flow Tracing"]
    colors_list = list(colors.TABLEAU_COLORS.values())
    cmap_methods = dict(zip(subset, colors_list[6:10]))

    # Totals
    _plot_totals(top_imp, subset, data["imp_bz"], cmap_methods, plot_dirs["main_imp"] / f"Main_Importers_Totals_{year_str}.png")
    _plot_totals(top_exp, subset, data["exp_bz"], cmap_methods, plot_dirs["main_exp"] / f"Main_Exporters_Totals_{year_str}.png")

    # Stacked
    _plot_stacked(top_imp, subset, data["imp_bz"], plot_dirs["main_imp"] / f"Main_Importers_Country_{year_str}.png", mode="country", top_items=top_exp)
    _plot_stacked(top_imp, subset, data["imp_agg"], plot_dirs["main_imp"] / f"Main_Importers_Type_{year_str}.png", mode="type")
    _plot_stacked(top_exp, subset, data["exp_bz"], plot_dirs["main_exp"] / f"Main_Exporters_Country_{year_str}.png", mode="country", top_items=top_imp)

    print("Plotting Complete.")

# --- HELPERS ---
def _plot_generic(df, unit, title, path):
    if df is None or df.empty: return
    df = df.loc[:, (df.sum() > 0.01)]
    if df.empty: return
    df = df.reindex(df.sum().sort_values(ascending=False).index, axis=1)
    
    x = np.arange(len(df.columns))
    width = 0.8/len(df.index)
    fig, ax = plt.subplots(figsize=(15, 8))
    for i, m in enumerate(df.index):
        ax.bar(x - 0.4 + (width*i) + width/2, df.loc[m], width, label=m)
    
    ax.set_ylabel(unit)
    ax.set_title(title)
    ax.set_xticks(x, df.columns, rotation=45)
    ax.legend()
    plt.tight_layout()
    fig.savefig(path)
    plt.close()

def _plot_totals(countries, methods, data_dict, cmap, path):
    x = np.arange(len(countries))
    width = 0.2
    fig, ax = plt.subplots(figsize=(18, 10))
    for i, m in enumerate(methods):
        vals = [data_dict[c].loc[m].sum() if c in data_dict and m in data_dict[c].index else 0 for c in countries]
        rects = ax.bar(x + i*width, np.around(vals, 2), width, label=m, color=cmap.get(m, 'grey'), edgecolor='black')
        ax.bar_label(rects, padding=3)
    
    ax.set_ylabel('TWh', fontsize=18)
    ax.set_xticks(x + 1.5*width, countries, fontsize=18)
    ax.legend(fontsize=18)
    fig.savefig(path)
    plt.close()

def _plot_stacked(countries, methods, data_dict, path, mode="country", top_items=None):
    fig, ax = plt.subplots(figsize=(18, 15))
    width = 0.2
    
    # Colors
    if mode == "type":
        cmap = {"Solar": "yellow", "Biomass": "tab:green", "Hydro": "tab:blue", "Hard coal": "black", "Lignite": "tab:brown", 
                "Nuclear": "tab:red", "Storage": "white", "Wind Onshore": "skyblue", "Wind Offshore": "darkblue", "Gas": "orange", "Other": "grey"}
        all_cols = set()
        for c in countries: 
            if c in data_dict: all_cols.update(data_dict[c].columns)
        for c in all_cols: 
            if c not in cmap: cmap[c] = 'lightgrey'
    else:
        colors_list = list(colors.TABLEAU_COLORS.values()) + ["salmon", "maroon", "gold"]
        all_cols = set()
        for c in countries: 
            if c in data_dict: all_cols.update(data_dict[c].columns)
        
        legend_items = (top_items if top_items else []) + list(all_cols)
        # Dedupe preserving order
        seen = set()
        legend_items = [x for x in legend_items if not (x in seen or seen.add(x))]
        
        full_colors = colors_list * (len(legend_items)//len(colors_list) + 1)
        cmap = dict(zip(legend_items, full_colors))
        cmap["Other"] = "lightgrey"

    multiplier = 0
    x_ticks = []
    
    for bz in countries:
        if bz not in data_dict: multiplier += len(methods) + 1; continue
        df = data_dict[bz]
        
        for m in methods:
            if m not in df.index: multiplier += 1; continue
            bottom = 0
            row = df.loc[m]
            
            # Draw bars
            if mode == "country" and top_items:
                for p in top_items:
                    if p in row.index:
                        val = row[p]
                        if val > 0.01:
                            ax.bar(multiplier, val, width, bottom=bottom, color=cmap.get(p, 'grey'), edgecolor='black', label=p)
                            bottom += val
                other = row.sum() - bottom
                if other > 0.01:
                    ax.bar(multiplier, other, width, bottom=bottom, color="lightgrey", edgecolor='black', label="Other")
            else:
                for c in row.index:
                    val = row[c]
                    if val > 0.01:
                        ax.bar(multiplier, val, width, bottom=bottom, color=cmap.get(c, 'grey'), edgecolor='black', label=c)
                        bottom += val
            multiplier += 1
        
        x_ticks.append(multiplier - len(methods)/2.0 - 0.5)
        multiplier += 1

    ax.set_xticks(x_ticks, countries, fontsize=18)
    
    # Legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    
    if mode == "country" and top_items:
        keys = top_items + ["Other"]
        # Filter keys that actually exist
        valid_keys = [k for k in keys if k in by_label]
        valid_handles = [by_label[k] for k in valid_keys]
        ax.legend(valid_handles, valid_keys, loc='upper right', ncol=4, fontsize=14)
    else:
        ax.legend(by_label.values(), by_label.keys(), loc='upper right', ncol=4, fontsize=14)

    fig.savefig(path, bbox_inches='tight')
    plt.close()