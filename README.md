# Exchange Analysis

This project analyses data available through the ENTSO-E API using multiple methods (e.g. flow tracing, pooling), in order to determine the import sources and export sinks in the European electricity market on a per bidding zone and per type basis, for each bidding zone in the network.

## 🛠️ Prerequisites

* **Python:** 3.10 or higher recommended.
* **Anaconda** (Recommended) or standard Python installation.
* **ENTSO-E API Key:** Required for downloading data.

---

## 🚀 Setup Instructions

### 1. Clone or Download the Project

Ensure your project folder looks like this:

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

## 🔑 Configuration (Important!)

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

Results are saved as time series in the `outputs/` directory, organized by year:

* **`generation_demand_data.../`**: Cleaned hourly generation and load data.
* **`comm_flow_total.../`**: Commercial exchange schedules (Netted and Gross).
* **`physical_flow_data.../`**: Physical cross-border flows.
* **`import_flow_tracing.../`**: Import source as a result of flow tracing analysis.
* **`pooling/`**: Hypothetical European mix calculations.
* **`annual_totals_per_method/`**: Final aggregated TWh CSVs for visualization.

---

