Column Name,Origin,Description & Calculation Logic
"[Generation Type] (e.g., Biomass, Wind Onshore, Nuclear)",ENTSO-E / Elexon BMRS,"Raw generation data per technology. Negative values are clipped to 0.0. Resampled to 1h averages, and missing values are filled via imputation heuristics."
Storage Charge,Calculated,"Absolute sum of all negative values across storage components (e.g., ""Hydro Pumped Storage"", ""Energy storage""). Represents power pulled from the grid by storage assets."
Storage Discharge,Calculated,Sum of all positive values across storage components. Represents power injected into the grid by storage assets.
Generation,Calculated,The sum of all active physical generation columns at a given timestep (excluding storage components).
Total Generation,Calculated,Generation + Storage Discharge. The total physical power injected into the zonal grid.
Demand (or Actual Load),ENTSO-E / Elexon BMRS,"Raw electrical load data, resampled to 1h averages with missing gaps filled."
Total Load,Calculated,Demand + Storage Charge. The total physical power consumed within the zone.
Net Export,Calculated,"Total Generation - Total Load. The structural net border position. Positive = Net Exporter, Negative = Net Importer."
gap_filling_method,Metadata,Tracks the exact gap-filling algorithms (or negative clipping) applied to specific timestamps.
download_timestamp,Metadata,The localized temporal vintage of the raw data.
