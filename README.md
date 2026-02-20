# OCx25 Dashboard

An interactive web dashboard for exploring and analyzing  data from OCx25 experiments, featuring HER (Hydrogen Evolution Reaction) and CO2RR (CO2 Reduction Reaction) performance data, as well as XRD (X-ray Diffraction) data.

This project is part of the OCx25 research: https://ai.meta.com/blog/open-catalyst-simulations-experiments/.

## Features

- **Interactive Data Filtering**: Filter data by source, batch, reaction type, element count, and voltage/current density ranges
- **Data Aggregation**: Average data across multiple replicates with statistical analysis
- **Interactive Plots**: Dedicated plotting interfaces for HER and CO2RR performance visualization
- **XRD Analysis Dashboard**: Comprehensive X-ray diffraction data analysis with phase identification and fitting
- **Data Export**: Export filtered or averaged data to CSV format
- **Composition Analysis**: Support for both XRF and target composition data
- **Voltage Conversion**: Customizable voltage conversion parameters for electrochemical measurements

## Data Structure

### Electrochemical Data
The dashboard works with electrochemical experimental data including:
- Elemental compositions (XRF and target)
- Voltage and current density measurements
- Faradaic efficiency measurements
- Batch and source tracking
- Multiple measurement replicates

### XRD Data Structure
The XRD Analysis Dashboard accesses data stored in a **Cloudflare R2 bucket** via a Cloudflare Worker API. The data structure includes:

#### Data Storage
- **Location**: Cloudflare R2 bucket (cloud storage)
- **Public Link**: https://pub-6ab47103d4af4add87fd1289ebeb7d64.r2.dev
- **Format**: JSON files containing XRD measurement data
- **Access**: Via Cloudflare Worker API endpoint
- **Organization**: Data organized by datasets and sample identifiers

## Usage

### Local Development
```bash
python OCx25_dashboard.py
```

### Production Deployment
The application automatically detects production environments and adjusts its behavior accordingly.

### Accessing Dashboards
- **Main Dashboard**: `/` - Overview and data filtering interface
- **HER Analysis**: `/her/` - Hydrogen Evolution Reaction plotting and analysis
- **CO2RR Analysis**: `/co2/` - CO2 Reduction Reaction plotting and analysis  
- **XRD Analysis**: `/xrd/` - X-ray Diffraction data analysis and phase identification

### XRD Dashboard Access
The XRD Analysis Dashboard requires:
1. **Cloudflare Worker API**: Must be deployed and accessible
2. **R2 Bucket Access**: XRD data stored in Cloudflare R2 bucket
3. **Network Connectivity**: Dashboard fetches data via API calls

**Note**: If XRD data fails to load, it may indicate:
- Sample doesn't have XRD data available
- Data stored in different location
- API endpoint connectivity issues

## Dependencies

See `requirements.txt` for a complete list of Python dependencies including:
- Flask for web framework
- Pandas for data manipulation
- Plotly for interactive visualizations
- Matplotlib for static plots
- NumPy and SciPy for numerical operations

## File Structure

- `OCx25_dashboard.py` - Main Flask application file
- `her_plot_blueprint.py` - HER-specific plotting blueprint
- `co2_plot_blueprint.py` - CO2RR-specific plotting blueprint  
- `xrd_plot_blueprint.py` - XRD Analysis Dashboard blueprint
- `templates/dashboard.html` - Main dashboard HTML template
- `Data/` - Local data files directory
- `environment.yml` - Conda environment specification
- `requirements.txt` - Python dependencies

## Acknowledgments

Developed through vibe coding with [Cursor](https://cursor.sh), [Warp](https://warp.dev), and me.


