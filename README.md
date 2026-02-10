# Exchange Analysis

This project analyses data available through the ENTSO-E API using multiple methods (e.g. flow tracing, pooling), in order to determine the import sources and export sinks in the European electricity market on a per bidding zone and per type basis, for each bidding zone in the network. Import/export results for the following methods can be calculated:

1. **Commercial Flows Total (CFT):** Take incoming/outgoing line (for import/export) as the exchange value from/to a neighbouring bidding zone.
2. **Netted Commercial Flows Total (Netted CFT):** Net over incoming and outgoing line as the exchange value from/to a neighbouring bidding zone.
3. **Pooled Net CFT:** A bidding zone's net import is proportionally supplied by all net exporters in the network at each timepoint, a "copper plate" grid model without transmission constraints. Vice versa for net export.
4. **Pooled Net Phys.:** Similar to Pooled Net CFT, however uses physical flow net position values instead of CFT values.
5. **Direct Coupling (DC) Flow Tracing:** Takes into account the potential for flows to continue beyond immediate neighbours and for transit flows. Each zone's generation, load and exchanges with neighbours are elements of the network.
6. **Aggregated Coupling (AC) Flow Tracing:** Similar to Direct Coupling, however the net position of a zone is now its contribution to the network instead of both its generation and load.

## 🛠️ Prerequisites

* **Python:** 3.10 or higher recommended.
* **Anaconda** (Recommended) or standard Python installation.
* **ENTSO-E API Key:** Required for downloading data. You can obtain a free API key by registering on the [ENTSO-E Transparency Platform](https://transparency.entsoe.eu/) and requesting "Restful API Access" in your account settings.

---

## 🚀 Setup Instructions

### 1. Clone or Download the Project

### 2. Create a Virtual Environment

It is highly recommended to use a virtual environment to avoid conflicts.

**Option A: Using Conda (Recommended)**

```bash
# Create the environment
conda create -n exchange-analysis python=3.10

# Activate it
conda activate exchange-analysis

```

**Option B: Using venv (Standard Python)**

```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate

```

### 3. Install Dependencies

Run this command inside your activated environment:

```bash
pip install -r requirements.txt

```

---

## 🔑 Configuration

You must create a `keys.yaml` file in the **root directory** of the project to store your API key. This file is ignored by Git to keep your secrets safe.

1. Create a file named `keys.yaml`.
2. Paste the following content into it:

```yaml
entsoe-key: "YOUR-UUID-API-KEY-HERE"

```

> **Note:** You can obtain a free API key by registering on the [ENTSO-E Transparency Platform](https://transparency.entsoe.eu/) and requesting "Restful API Access" in your account settings.

---

## ▶️ Usage

### 1. Configure the Run

Open `main.py` to adjust the **Control Panel** section:

* **Run Flags:** Set steps to `True` or `False` (e.g., set `download: False` if you already have the data).
* **Period:** Set your start and end dates (e.g., `"2024-01-01 00:00"`).
* **Subsets:** Uncomment `selected_bzs`/`target_zones` if you only want to download data for specific bidding zones (e.g., `["DE_LU", "FR"]`), and `selected_data_types`/`data_types` to download specific types of data (generation and load, commercial flow total etc.).


### 2. Run the Pipeline

Ensure your environment is activated (`conda activate energy-analysis`) and run:

```bash
python main.py

```

### 3. Check the Logs

Real-time logs will appear in your terminal. Detailed logs are saved to:
`logs/log_{TIMESTAMP}.log`

---

## 📂 Outputs

Import results are saved on a per bidding zone, per type, and "per type per bidding zone" basis as time series CSVs in the `outputs/` directory, organized by year. Only export totals are saved, as export time series would be essentially a duplicate of import time series results. The following folders can be generated:

* **`generation_demand_data.../`**: Hourly generation and load used for determining per type mixes of import/export, and used in direct flow tracing.
* **`comm_flow_total.../`**: Hourly commercial flow total exchanges, determines import results using in-coming line values or by netting over both directions, used as input to pooling approach.
* **`physical_flow_data.../`**: Physical cross-border flows, used as input to pooling approach and flow tracing.
* **`import_flow_tracing.../`**: Import source values as a result of flow tracing analysis.
* **`pooling/`**: Import source values as a result of flow tracing analysis.
* **`annual_totals_per_method/`**: Final aggregated TWh import and export totals.

---

