# OCx24 Dashboard

An interactive web dashboard for exploring and analyzing electrochemical data from OCx24 experiments, featuring HER (Hydrogen Evolution Reaction) and CO2RR (CO2 Reduction Reaction) performance data.

## Features

- **Interactive Data Filtering**: Filter data by source, batch, reaction type, element count, and voltage/current density ranges
- **Data Aggregation**: Average data across multiple replicates with statistical analysis
- **Interactive Plots**: Dedicated plotting interfaces for HER and CO2RR performance visualization
- **Data Export**: Export filtered or averaged data to CSV format
- **Composition Analysis**: Support for both XRF and target composition data

## Data Structure

The dashboard works with electrochemical experimental data including:
- Elemental compositions (XRF and target)
- Voltage and current density measurements
- Faradaic efficiency measurements
- Batch and source tracking
- Multiple measurement replicates

## Usage

### Local Development
```bash
python OCx24_dashboard.py
```

### Production Deployment
The application automatically detects production environments and adjusts its behavior accordingly.

## Dependencies

See `requirements.txt` for a complete list of Python dependencies including:
- Flask for web framework
- Pandas for data manipulation
- Plotly for interactive visualizations
- Matplotlib for static plots
- NumPy and SciPy for numerical operations

## File Structure

- `OCx24_dashboard.py` - Main application file
- `interactive_plot_HER.py` - HER-specific plotting interface
- `interactive_plot_CO2.py` - CO2RR-specific plotting interface
- `templates/dashboard.html` - Frontend HTML template
- `Data/` - Data files directory
- `environment.yml` - Conda environment specification

## License

This project is part of the OCx24 research: https://ai.meta.com/blog/open-catalyst-simulations-experiments/.
