import pandas as pd
import matplotlib.pyplot as plt
from flask import Blueprint, render_template_string, request, jsonify, Response
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
from scipy.stats import linregress
import json
import glob

# Create Blueprint
co2_plot_bp = Blueprint('co2_plot', __name__, url_prefix='/co2')

# Atomic weights for conversion between atomic and weight fractions
ATOMIC_WEIGHTS = {
    'Ag': 107.8682, 'Au': 196.966569, 'Cd': 112.411, 'Cu': 63.546, 'Ga': 69.723,
    'Hg': 200.59, 'In': 114.818, 'Mn': 54.938044, 'Mo': 95.96, 'Nb': 92.90637,
    'Ni': 58.6934, 'Pd': 106.42, 'Pt': 195.084, 'Rh': 102.90550, 'Sn': 118.710,
    'Tl': 204.38, 'W': 183.84, 'Zn': 65.38
}

def convert_atomic_to_weight_fraction(df, element_columns):
    """
    Convert atomic fraction to weight fraction for elemental compositions.
    """
    df_converted = df.copy()
    
    for col in element_columns:
        if col in df_converted.columns and col in ATOMIC_WEIGHTS:
            df_converted[col] = df_converted[col] * ATOMIC_WEIGHTS[col]
    
    # Normalize to get weight fractions (0-1 scale)
    for idx, row in df_converted.iterrows():
        total_weight = sum(row[col] for col in element_columns if col in df_converted.columns and col in ATOMIC_WEIGHTS)
        if total_weight > 0:
            for col in element_columns:
                if col in df_converted.columns and col in ATOMIC_WEIGHTS:
                    df_converted.at[idx, col] = row[col] / total_weight
    
    return df_converted

def filter_df_by_current_density(df, target_current_density, tolerance=10):
    """
    Filter dataframe to get data close to a specific current density value.
    """
    # Filter data within tolerance of target current density
    filtered_df = df[abs(df['current density'] - target_current_density) <= tolerance].copy()
    return filtered_df

def generate_df_at_voltage(df, voltage_col='voltage_mean', cd_col='current density', fe_prefix='fe_', group_cols=None, target_voltage=3.0):
    """
    Generate a dataframe interpolated at a specific voltage value.
    """
    if group_cols is None:
        # Default: group by 'source' and all columns containing 'xrf' in their name
        xrf_cols = [col for col in df.columns if 'xrf' in col]
        group_cols = ['source'] + xrf_cols

    # Handle both _mean/_std suffix format and simple format
    fe_mean_cols = [col for col in df.columns if col.startswith(fe_prefix) and col.endswith('_mean')]
    fe_std_cols = [col for col in df.columns if col.startswith(fe_prefix) and col.endswith('_std')]
    
    # If no _mean columns found, look for simple fe_ columns (without _mean suffix)
    if not fe_mean_cols:
        fe_mean_cols = [col for col in df.columns if col.startswith(fe_prefix) and not col.endswith('_std')]
    
    # Get elemental composition columns (Ag, Au, Cu, etc.)
    # Handle both voltage_mean and voltage column names
    voltage_cols_to_exclude = ['voltage_mean', 'voltage_std', 'voltage']
    composition_col = 'xrf composition' if 'xrf composition' in df.columns else 'target composition'
    element_cols = [col for col in df.columns if col not in ['source', 'current density', composition_col, 'rep'] + voltage_cols_to_exclude and not col.startswith('fe_') and not col.endswith('std')]

    results = []

    for group_keys, group_df in df.groupby(group_cols):
        if not isinstance(group_keys, tuple):
            group_keys = (group_keys,)
        
        # Current density fit
        log_cds = np.log(group_df[cd_col].replace(0, np.nan).dropna().values)
        valid_idx = group_df[cd_col].replace(0, np.nan).dropna().index
        voltages_for_fit = group_df.loc[valid_idx, voltage_col].values

        if len(voltages_for_fit) >= 2:
            slope, intercept, _, _, _ = linregress(voltages_for_fit, log_cds)
            pred_log_cd = slope * target_voltage + intercept
            pred_cd = np.exp(pred_log_cd)
        else:
            pred_cd = np.nan

        fe_pred_dict = {}
        # MEAN
        for fe_col in fe_mean_cols:
            fe_vals = group_df[fe_col].values
            mask = ~np.isnan(fe_vals)
            if np.sum(mask) >= 2:
                slope_fe, intercept_fe, _, _, _ = linregress(group_df[voltage_col].values[mask], fe_vals[mask])
                pred_fe = slope_fe * target_voltage + intercept_fe
            else:
                pred_fe = np.nan
            fe_pred_dict[fe_col] = pred_fe
        
        # STD
        for fe_col in fe_std_cols: 
            fe_vals = group_df[fe_col].values
            mask = ~np.isnan(fe_vals)
            if np.sum(mask) >= 2:
                slope_fe, intercept_fe, _, _, _ = linregress(group_df[voltage_col].values[mask], fe_vals[mask])
                pred_fe = slope_fe * target_voltage + intercept_fe
            else:
                pred_fe = np.nan
            fe_pred_dict[fe_col] = pred_fe

        # Get elemental composition values (these don't change with voltage, so take the first value)
        element_dict = {}
        for element_col in element_cols:
            element_vals = group_df[element_col].dropna()
            if not element_vals.empty:
                element_dict[element_col] = element_vals.iloc[0]
            else:
                element_dict[element_col] = np.nan

        row = dict(zip(group_cols, group_keys))
        row['current density'] = pred_cd
        row.update(fe_pred_dict)
        row.update(element_dict)  # Add elemental composition columns
        results.append(row)

    return pd.DataFrame(results)

def load_xrd_data(sample_id, data_type="raw"):
    """
    Load XRD data for a specific sample ID from Data/XRD directory.
    Args:
        sample_id: The sample ID to load
        data_type: Either "raw" (.xy files) or "normalized" (.csv files)
    Returns the XRD data as a list of [x, y] pairs or None if not found.
    """
    try:
        # Construct the file path based on data type
        if data_type == "raw":
            xrd_file_path = f"Data/XRD/raw/{sample_id}.xy"
        elif data_type == "normalized":
            xrd_file_path = f"Data/XRD/normalized/{sample_id}.csv"
        else:
            print(f"Invalid data type: {data_type}")
            return None
            
        print(f"DEBUG: Looking for XRD file: {xrd_file_path}")
        print(f"DEBUG: Current working directory: {os.getcwd()}")
        print(f"DEBUG: File exists: {os.path.exists(xrd_file_path)}")
        
        # Check if file exists
        if not os.path.exists(xrd_file_path):
            print(f"XRD file not found: {xrd_file_path}")
            return None
        
        # Read the file
        data = []
        with open(xrd_file_path, 'r') as f:
            lines = f.readlines()
            # Skip the first line (header)
            for line_num, line in enumerate(lines[1:], 2):  # Start from line 2
                line = line.strip()
                if line and not line.startswith('#'):  # Skip empty lines and comments
                    try:
                        # Handle different separators (space, tab, comma)
                        parts = line.replace(',', ' ').split()
                        if len(parts) >= 2:
                            x_val = float(parts[0])
                            y_val = float(parts[1])
                            data.append([x_val, y_val])
                    except ValueError:
                        # Skip lines that can't be parsed as numbers
                        if line_num <= 10:  # Only log first few errors to avoid spam
                            print(f"Warning: Could not parse line {line_num} in {xrd_file_path}: {line}")
                        continue
        
        if not data:
            print(f"No valid data found in XRD file: {xrd_file_path}")
            return None
        
        print(f"Loaded XRD data for sample {sample_id} ({data_type}): {len(data)} data points")
        return data
        
    except Exception as e:
        print(f"Error loading XRD data: {e}")
        return None

def load_original_data():
    """Load the original data from CSV file or current data from dashboard"""
    try:
        # First try to load current data from dashboard
        current_data_file = "Data/current_data_co2.json"
        if os.path.exists(current_data_file):
            with open(current_data_file, 'r') as f:
                saved_data = json.load(f)
            
            if isinstance(saved_data, dict) and 'data' in saved_data and 'columns' in saved_data:
                current_data = saved_data['data']
                column_order = saved_data['columns']
                df = pd.DataFrame(current_data, columns=column_order)
            elif isinstance(saved_data, list):
                df = pd.DataFrame(saved_data)
            else:
                df = pd.DataFrame(saved_data)
            
            # Filter for CO2R reaction if reaction column exists
            if 'reaction' in df.columns:
                df = df[df['reaction'] == 'CO2R'].copy()
                df = df.drop('reaction', axis=1)
            
            print(f"DEBUG: Available columns after loading CO2R data: {list(df.columns)}")
            print(f"DEBUG: Data shape: {df.shape}")
            print(f"DEBUG: Voltage columns present: {[col for col in df.columns if 'voltage' in col.lower()]}")
            
            return df
    except Exception as e:
        print(f"Could not load current data: {e}")
    
    # Fallback to original CSV data
    try:
        df = pd.read_csv("Data/DashboardData.csv")
        if 'reaction' in df.columns:
            df = df[df['reaction'] == 'CO2R'].copy()
            df = df.drop('reaction', axis=1)
        return df
    except Exception as e:
        print(f"Could not load CSV data: {e}")
        return pd.DataFrame()

def calculate_pca_components(df):
    """Calculate PCA components from elemental composition data."""
    if df.empty or len(df) < 2:
        df['PCA1'] = np.nan
        df['PCA2'] = np.nan
        return df
    
    # Get only elemental composition columns
    voltage_cols_to_exclude = ['voltage_mean', 'voltage_std', 'voltage']
    composition_col = 'xrf composition' if 'xrf composition' in df.columns else 'target composition'
    element_cols = [col for col in df.columns if col not in ['sample id', 'source', 'batch number', 'batch date', 'current density', composition_col, 'target composition', 'xrf composition', 'rep'] + voltage_cols_to_exclude and not col.startswith('fe_') and not col.endswith('std')]
    
    # Filter out non-numeric columns
    numeric_element_cols = []
    for col in element_cols:
        try:
            if pd.to_numeric(df[col], errors='coerce').notna().sum() >= 2:
                numeric_element_cols.append(col)
        except:
            continue
    
    if len(numeric_element_cols) < 2:
        df['PCA1'] = np.nan
        df['PCA2'] = np.nan
        return df
    
    # Prepare data for PCA
    pca_data = df[numeric_element_cols].copy()
    for col in pca_data.columns:
        pca_data[col] = pd.to_numeric(pca_data[col], errors='coerce')
    pca_data = pca_data.fillna(0)
    
    if pca_data.sum().sum() == 0:
        df['PCA1'] = 0
        df['PCA2'] = 0
        return df
    
    try:
        # Standardize and apply PCA
        scaler = StandardScaler()
        pca_data_scaled = scaler.fit_transform(pca_data)
        pca = PCA(n_components=2)
        pca_components = pca.fit_transform(pca_data_scaled)
        
        df['PCA1'] = pca_components[:, 0]
        df['PCA2'] = pca_components[:, 1]
    except Exception as e:
        print(f"PCA calculation failed: {e}")
        df['PCA1'] = np.nan
        df['PCA2'] = np.nan
    
    return df

def format_column_name(column_name):
    """Format column names to be more readable."""
    if column_name in ['voltage_mean', 'voltage']:
        return 'Full Cell Voltage (V)'
    elif column_name.startswith('fe_'):
        base_name = column_name.replace('fe_', '').replace('_mean', '')
        if base_name == 'h2':
            return 'Faradaic Efficiency H₂'
        elif base_name == 'co':
            return 'Faradaic Efficiency CO'
        elif base_name == 'ch4':
            return 'Faradaic Efficiency CH₄'
        elif base_name == 'c2h4':
            return 'Faradaic Efficiency C₂H₄'
        elif base_name == 'gas_total':
            return 'Faradaic Efficiency Gas Total'
        elif base_name == 'liquid':
            return 'Faradaic Efficiency Liquid'
        else:
            return 'Faradaic Efficiency ' + base_name.upper()
    elif column_name in ['PCA1', 'PCA2']:
        return column_name
    elif column_name in ['Ag', 'Au', 'Cd', 'Cu', 'Ga', 'Hg', 'In', 'Ni', 'Pd', 'Pt', 'Rh', 'Sn', 'Tl', 'Zn']:
        return column_name
    else:
        return column_name

@co2_plot_bp.route('/')
def co2_plot_main():
    """Main CO2R plot page"""
    # Load and process data
    current_df = load_original_data()
    
    if current_df.empty:
        return "<h2>Error: No CO2R data available</h2><p>Please ensure CO2R data is available in the main dashboard.</p>"
    
    # Calculate PCA components
    df_with_pca = calculate_pca_components(current_df)
    
    # Identify element columns
    voltage_cols_to_exclude = ['voltage_mean', 'voltage_std', 'voltage']
    composition_col = 'xrf composition' if 'xrf composition' in df_with_pca.columns else 'target composition'
    element_cols = [col for col in df_with_pca.columns if col not in ['sample id', 'source', 'batch number', 'batch date', 'current density', composition_col, 'target composition', 'xrf composition', 'PCA1', 'PCA2', 'rep'] + voltage_cols_to_exclude and not col.startswith('fe_') and not col.endswith('std')]
    
    # Add PCA1 as first option if available
    if 'PCA1' in df_with_pca.columns:
        element_cols.insert(0, 'PCA1')
    if 'Cu' in df_with_pca.columns and 'Cu' not in element_cols:
        element_cols.insert(0, 'Cu')
    
    # Identify FE columns for y-axis options
    fe_cols = [col for col in df_with_pca.columns if col.startswith('fe_') and col.endswith('_mean')]
    if not fe_cols:
        fe_cols = [col for col in df_with_pca.columns if col.startswith('fe_') and not col.endswith('_std')]
    
    # Default y-axis to CO FE if available
    default_y_col = 'fe_co_mean' if 'fe_co_mean' in fe_cols else (fe_cols[0] if fe_cols else 'voltage_mean')
    
    # Generate dropdown options
    element_options = ''.join([f'<option value="{col}">{format_column_name(col)}</option>' for col in element_cols])
    fe_options = ''.join([f'<option value="{col}" {"selected" if col == default_y_col else ""}>{format_column_name(col)}</option>' for col in fe_cols])
    
    # Add voltage options to y-axis
    if 'voltage_mean' in df_with_pca.columns:
        fe_options += f'<option value="voltage_mean">{format_column_name("voltage_mean")}</option>'
    if 'voltage' in df_with_pca.columns:
        fe_options += f'<option value="voltage">{format_column_name("voltage")}</option>'
    
    # Create comprehensive x-axis and z-axis options from original
    comprehensive_x_axis_options = element_cols.copy()
    if 'PCA2' in df_with_pca.columns:
        comprehensive_x_axis_options.append('PCA2')
    if 'voltage_mean' in df_with_pca.columns:
        comprehensive_x_axis_options.append('voltage_mean')
    elif 'voltage' in df_with_pca.columns:
        comprehensive_x_axis_options.append('voltage')
    comprehensive_x_axis_options.extend(fe_cols)
    
    # Y-axis options: same as x-axis options
    y_axis_options = comprehensive_x_axis_options.copy()
    
    # Create z-axis options (for color control) - same as y-axis options with "Default" as first option
    z_axis_options = ['default_colors']  # Default option for current blue/red/black coloring
    z_axis_options.extend(y_axis_options)  # Add all y-axis options
    
    # Generate all dropdown options
    x_axis_options_html = ''.join([f'<option value="{col}">{format_column_name(col)}</option>' for col in comprehensive_x_axis_options])
    y_axis_options_html = ''.join([f'<option value="{col}" {"selected" if col == default_y_col else ""}>{format_column_name(col)}</option>' for col in y_axis_options])
    z_axis_options_html = ''.join([f'<option value="{col}" {"selected" if col == "default_colors" else ""}>{"Default" if col == "default_colors" else format_column_name(col)}</option>' for col in z_axis_options])
    
    # Current density options
    current_density_options = [50, 100, 150, 200, 300]
    default_current_density = 100
    
    # Create the comprehensive HTML template from original interactive plot
    html_template = f'''
    <!DOCTYPE html>
    <html>
    <head>
        <title>OCx24 Dataset: CO₂RR Performance Interactive Plot</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {{
                font-family: 'Roboto', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                margin: 0;
                padding: 0;
                background: #fafafa;
                min-height: 100vh;
                color: #202124;
                overflow-x: hidden;
                line-height: 1.6;
            }}
            
            .back-link {{
                position: fixed;
                top: 20px;
                left: 20px;
                z-index: 1000;
                background: #4285f4;
                color: white;
                padding: 12px 20px;
                border-radius: 8px;
                text-decoration: none;
                font-weight: 500;
                font-size: 14px;
                transition: all 0.2s ease;
                box-shadow: 0 2px 8px rgba(66, 133, 244, 0.3);
            }}
            
            .back-link:hover {{
                background: #3367d6;
                transform: translateY(-1px);
                box-shadow: 0 4px 12px rgba(66, 133, 244, 0.4);
            }}
            
            .container {{
                max-width: 100%;
                margin: 0 auto;
                background: #ffffff;
                border-radius: 12px;
                box-shadow: 0 2px 6px rgba(0, 0, 0, 0.08);
                overflow: hidden;
                margin: 12px;
                border: 1px solid #e8eaed;
            }}
            
            h1 {{
                color: #202124;
                text-align: center;
                font-size: 2.4em;
                font-weight: 400;
                letter-spacing: -0.5px;
                margin: 0;
                padding: 32px 24px 16px 24px;
                background: #ffffff;
                border-bottom: 1px solid #e8eaed;
            }}
            
            h3 {{
                color: #5f6368;
                text-align: center;
                margin: 0;
                padding: 0 24px 20px 24px;
                background: #ffffff;
                font-size: 1.1em;
                font-weight: 400;
                letter-spacing: 0.2px;
            }}
            
            .controls {{
                background: #f8f9fa;
                padding: 24px;
                border-bottom: 1px solid #e8eaed;
                display: flex;
                gap: 24px;
                align-items: center;
                flex-wrap: wrap;
                justify-content: center;
            }}
            
            .control-group {{
                display: flex;
                flex-direction: column;
                gap: 8px;
                align-items: center;
            }}
            
            label {{
                font-weight: 500;
                color: #5f6368;
                font-size: 0.875em;
                text-transform: none;
                letter-spacing: 0.2px;
            }}
            
            select {{
                padding: 12px 16px;
                border: 1px solid #dadce0;
                border-radius: 8px;
                background: #ffffff;
                font-size: 14px;
                font-weight: 400;
                color: #202124;
                cursor: pointer;
                transition: all 0.2s ease;
                min-width: 160px;
                box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
            }}
            
            select:hover {{
                border-color: #4285f4;
                box-shadow: 0 2px 8px rgba(66, 133, 244, 0.15);
            }}
            
            select:focus {{
                outline: none;
                border-color: #4285f4;
                box-shadow: 0 0 0 2px rgba(66, 133, 244, 0.2);
            }}
            
            .slider-container {{
                display: flex;
                flex-direction: column;
                gap: 12px;
                min-width: 200px;
                align-items: center;
            }}
            
            .slider-value {{
                font-size: 1.2em;
                font-weight: 500;
                color: #4285f4;
                background: #e8f0fe;
                padding: 12px 20px;
                border-radius: 8px;
                border: 1px solid #d2e3fc;
                box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
            }}
            
            input[type="range"] {{
                width: 200px;
                height: 6px;
                border-radius: 3px;
                background: #e8eaed;
                outline: none;
                opacity: 1;
                transition: all 0.2s ease;
                cursor: pointer;
                -webkit-appearance: none;
            }}
            
            input[type="range"]::-webkit-slider-thumb {{
                -webkit-appearance: none;
                appearance: none;
                width: 20px;
                height: 20px;
                border-radius: 50%;
                background: #4285f4;
                cursor: pointer;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
                border: 2px solid #ffffff;
                transition: all 0.2s ease;
            }}
            
            input[type="range"]::-webkit-slider-thumb:hover {{
                transform: scale(1.1);
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
            }}
            
            input[type="range"]::-moz-range-thumb {{
                width: 20px;
                height: 20px;
                border-radius: 50%;
                background: #4285f4;
                cursor: pointer;
                border: 2px solid #ffffff;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
            }}
            
            .checkbox-container {{
                display: flex;
                align-items: center;
                gap: 12px;
                background: #ffffff;
                padding: 16px 20px;
                border-radius: 8px;
                border: 1px solid #e8eaed;
                box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
            }}
            
            input[type="checkbox"] {{
                width: 18px;
                height: 18px;
                accent-color: #4285f4;
                cursor: pointer;
            }}
            
            .checkbox-container label {{
                margin: 0;
                color: #5f6368;
                font-weight: 400;
            }}
            
            .export-btn {{
                background: #ffffff;
                color: #34a853;
                border: 1px solid #34a853;
                padding: 12px 16px;
                border-radius: 8px;
                font-size: 14px;
                font-weight: 500;
                cursor: pointer;
                transition: all 0.2s ease;
                box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
                text-transform: none;
                letter-spacing: 0.2px;
                min-width: 160px;
            }}
            
            .export-btn:hover {{
                background: #34a853;
                color: #ffffff;
                box-shadow: 0 2px 8px rgba(52, 168, 83, 0.15);
                transform: translateY(-1px);
            }}
            
            .export-btn:active {{
                transform: translateY(0);
            }}
            
            .plots-container {{
                display: flex;
                flex-direction: column;
                gap: 20px;
                padding: 32px;
                background: #fafafa;
                min-height: 1200px;
            }}
            
            .plots-row {{
                display: flex;
                gap: 20px;
                min-height: 600px;
            }}
            
            .plot-section {{
                flex: 1;
                background: #ffffff;
                border-radius: 8px;
                box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
                overflow: hidden;
                transition: all 0.2s ease;
                border: 1px solid #e8eaed;
            }}
            
            .plot-section:hover {{
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.12);
            }}
            
            .plot-section h3 {{
                margin: 0;
                padding: 20px 20px 16px 20px;
                background: #f8f9fa;
                color: #202124;
                font-size: 1.1em;
                font-weight: 500;
                letter-spacing: 0.2px;
                border-bottom: 1px solid #e8eaed;
            }}
            
            .plot-section .header-row {{
                background: #f8f9fa;
                padding: 20px 20px 16px 20px;
                border-bottom: 1px solid #e8eaed;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }}
            
            .plot-section .header-row h3 {{
                background: none;
                padding: 0;
                margin: 0;
                flex: 1;
                border-bottom: none;
                color: #202124;
            }}
            
            .header-controls {{
                display: flex;
                align-items: center;
                gap: 20px;
            }}
            
            .toggle-group {{
                display: flex;
                align-items: center;
                gap: 10px;
            }}
            
            .toggle-label {{
                font-size: 14px;
                color: #5f6368;
                font-weight: 500;
            }}
            
            .toggle-switch {{
                display: flex;
                background: #e8eaed;
                border-radius: 20px;
                padding: 2px;
                position: relative;
            }}
            
            .toggle-switch input[type="radio"] {{
                display: none;
            }}
            
            .toggle-switch label {{
                padding: 8px 16px;
                font-size: 13px;
                font-weight: 500;
                color: #5f6368;
                cursor: pointer;
                border-radius: 18px;
                transition: all 0.2s ease;
                position: relative;
                z-index: 1;
            }}
            
            .toggle-switch input[type="radio"]:checked + label {{
                background: #4285f4;
                color: white;
                box-shadow: 0 2px 4px rgba(66, 133, 244, 0.3);
            }}
            
            .plot-content {{
                padding: 24px;
                min-height: 600px;
                display: flex;
                align-items: center;
                justify-content: center;
                background: #ffffff;
            }}
            
            .reset-btn {{
                background: #ffffff;
                color: #ea4335;
                border: 1px solid #ea4335;
                padding: 12px 16px;
                border-radius: 8px;
                font-size: 14px;
                font-weight: 500;
                cursor: pointer;
                transition: all 0.2s ease;
                box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
                text-transform: none;
                letter-spacing: 0.2px;
                min-width: 120px;
            }}
            
            .reset-btn:hover {{
                background: #ea4335;
                color: #ffffff;
                box-shadow: 0 2px 8px rgba(234, 67, 53, 0.15);
                transform: translateY(-1px);
            }}
            
            .reset-btn:active {{
                transform: translateY(0);
            }}
            
            .info-panel {{
                margin: 16px;
                padding: 20px;
                background: #f8f9fa;
                border-radius: 8px;
                font-size: 14px;
                color: #5f6368;
                border: 1px solid #e8eaed;
                box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
            }}
            
            .info-panel strong {{
                color: #202124;
                font-weight: 500;
            }}
            
            .info-panel em {{
                color: #5f6368;
                font-style: italic;
            }}
            
            .loading {{
                text-align: center;
                color: #5f6368;
                font-style: normal;
                margin: 24px 0;
                font-size: 14px;
            }}
            
            /* Custom scrollbar */
            ::-webkit-scrollbar {{
                width: 6px;
            }}
            
            ::-webkit-scrollbar-track {{
                background: #f1f3f4;
                border-radius: 3px;
            }}
            
            ::-webkit-scrollbar-thumb {{
                background: #dadce0;
                border-radius: 3px;
            }}
            
            ::-webkit-scrollbar-thumb:hover {{
                background: #bdc1c6;
            }}
            
            @media (max-width: 1200px) {{
                .plots-row {{
                    flex-direction: column;
                }}
                .controls {{
                    flex-direction: column;
                    gap: 24px;
                }}
                .control-group {{
                    min-width: 200px;
                }}
                h1 {{
                    font-size: 2em;
                }}
            }}
            
            @media (max-width: 768px) {{
                .container {{
                    margin: 16px;
                    border-radius: 8px;
                }}
                h1 {{
                    padding: 32px 24px 20px 24px;
                    font-size: 1.8em;
                }}
                h3 {{
                    padding: 0 24px 24px 24px;
                }}
                .controls {{
                    padding: 24px;
                }}
                .plots-container {{
                    padding: 24px;
                }}
            }}
        </style>
    </head>
    <body>
        <a href="/" class="back-link">← Back to Dashboard</a>
        
        <div class="container">
            <h1><strong>OCx24 Dataset:</strong> CO₂RR Performance Data Visualization</h1>
            
            <div class="controls">
                <div class="control-group">
                    <label for="xAxis">X-Axis</label>
                    <select id="xAxis" onchange="updatePlot()">
                        {x_axis_options_html}
                    </select>
                </div>
                
                <div class="control-group">
                    <label for="yAxis">Y-Axis</label>
                    <select id="yAxis" onchange="updatePlot()">
                        {y_axis_options_html}
                    </select>
                </div>
                
                <div class="control-group">
                    <label for="zAxis">Z-Axis (Color)</label>
                    <select id="zAxis" onchange="updatePlot()">
                        {z_axis_options_html}
                    </select>
                </div>
                
                <div class="slider-container" id="currentDensitySliderContainer">
                    <label for="currentDensitySlider">Current Density</label>
                    <div class="slider-value" id="currentDensityValue">{default_current_density} mA/cm²</div>
                    <input type="range" id="currentDensitySlider" 
                           min="0" max="{len(current_density_options)-1}" 
                           step="1" value="{current_density_options.index(default_current_density)}" 
                           oninput="updateCurrentDensity(this.value)" 
                           onchange="updatePlot()">
                </div>
                
                <div class="checkbox-container">
                    <input type="checkbox" id="errorBars" checked onchange="updatePlot()">
                    <label for="errorBars">Show Error Bars</label>
                </div>
                
                <div class="control-group">
                    <button id="exportBtn" class="export-btn" onclick="exportData()">
                        ⊞ Export Data (CSV)
                    </button>
                </div>
            </div>
            
            <div class="plots-container">
                <div class="plots-row">
                    <div class="plot-section">
                        <div class="header-row">
                            <h3>Main Plot</h3>
                            <div class="header-controls">
                                <div class="toggle-group">
                                    <label class="toggle-label">Units:</label>
                                    <div class="toggle-switch">
                                        <input type="radio" id="unitAtomic" name="unitType" value="atomic" checked>
                                        <label for="unitAtomic">At. fraction</label>
                                        <input type="radio" id="unitWeight" name="unitType" value="weight">
                                        <label for="unitWeight">Wt. fraction</label>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div class="plot-content">
                            <div id="plot"></div>
                        </div>
                    </div>
                    
                    <div class="plot-section">
                        <div class="header-row">
                            <h3>Point Analysis</h3>
                            <button id="resetBtn" class="reset-btn" onclick="resetPointPlot()">
                                ⟳ Reset
                            </button>
                        </div>
                        <div class="plot-content">
                            <div id="pointPlotContent"></div>
                        </div>
                    </div>
                </div>
                
                <div class="plot-section">
                    <div class="header-row">
                        <h3>XRD Analysis</h3>
                        <div class="header-controls">
                            <div class="toggle-group">
                                <label class="toggle-label">Data Type:</label>
                                <div class="toggle-switch">
                                    <input type="radio" id="xrdRaw" name="xrdDataType" value="raw">
                                    <label for="xrdRaw">Raw</label>
                                    <input type="radio" id="xrdNormalized" name="xrdDataType" value="normalized" checked>
                                    <label for="xrdNormalized">Normalized</label>
                                </div>
                            </div>
                            <button id="resetXrdBtn" class="reset-btn" onclick="resetXrdPlot()">
                                ⟳ Reset
                            </button>
                        </div>
                    </div>
                    <div class="plot-content">
                        <div id="xrdPlotContent"></div>
                    </div>
                </div>
            </div>
            
            <div id="loading" class="loading" style="display: none;">Updating plot...</div>
            
            <div class="info-panel">
                <strong>Symbol Coding:</strong><br>
                <span style="font-weight: bold;">Circles</span>: Samples synthesized by Chemical Reduction (UofT)<br>
                <span style="font-weight: bold;">Diamonds</span>: Samples synthesized by Spark Ablation (VSP)<br>
                
                <br><strong>Default Color Coding:</strong><br>
                • <span style="color: #ef4444;">Red points</span>: Performance above Cu (UofT) threshold<br>
                • <span style="color: #3b82f6;">Blue points</span>: Performance above Cu (VSP) threshold<br>
                • <span style="color: #6b7280;">Black points</span>: Performance below both thresholds<br>
                <em>Note: The specific threshold values depend on the selected y-axis metric and are calculated as the mean performance for each source.</em>
                
                <br><br><strong>Analysis Modes:</strong><br>
                • <strong>Current Density:</strong> Filter data at specific current density values (50-300 mA/cm²)<br>
                
                <br><br><strong>Note on Error Bars:</strong><br>
                Error bars are shown only when averaging across identical compositions in this analysis.<br>
                • <strong>UofT (Chemical Reduction):</strong> Samples were first made as powders, XRF-measured once, then used to prepare 3 GDEs (Gas Diffusion Electrodes) for electrochemical testing. Since all GDEs came from the same powder vial (same composition), they were grouped together to calculate mean and standard deviation.<br>
                • <strong>VSP (Spark Ablation):</strong> Samples were deposited directly as 3 separate GDEs. Each had slightly different XRF compositions, so they could not be grouped. Their results are shown individually, without averaged error bars.
                
                <br><br><strong>XRD Analysis:</strong><br>
                Click on any point in the main plot to view the corresponding XRD pattern in the third window.<br>
                • XRD data is loaded from <code>/Data/XRD/raw/</code> directory<br>
                • Files are named using the sample ID (e.g., <code>sample_001.xy</code>)<br>
                • The plot shows 2θ (degrees) vs Intensity (counts)<br>
                • If no XRD data is found for a sample, an error message will be displayed
            </div>
        </div>
        
        <script>
            // Global variables
            let currentData = null;
            let originalData = null; // Store original atomic % data for calculations
            let currentMode = 'current_density';
            let currentDensityOptions = {current_density_options};
            let currentDensityIndex = {current_density_options.index(default_current_density)};
            let currentVoltage = 3.0;
            let clickedPointData = null; // Store the clicked point data globally
            let accumulatedPoints = []; // Store multiple clicked points for comparison
            let accumulatedXrdData = []; // Store multiple XRD datasets for comparison
            let clickedPoints = new Set(); // Track clicked points by sample ID
            
            // Initialize with data (filtered at default current density)
            const initialData = {json.dumps(filter_df_by_current_density(df_with_pca, default_current_density).to_dict('records'))};
            currentData = initialData;
            originalData = initialData;
            
            console.log('Loaded CO2R data:', currentData.length, 'rows');
            console.log('Available columns:', Object.keys(currentData[0] || {{}}));
            
            // Set default selections
            document.getElementById('xAxis').value = 'PCA1';
            document.getElementById('yAxis').value = 'PCA2';
            document.getElementById('zAxis').value = 'default_colors';
            
            // Add change event listeners to clear second plot when axes change
            document.getElementById('xAxis').addEventListener('change', function() {{
                if (clickedPointData) {{
                    console.log('X-axis changed, clearing second plot');
                    document.querySelector('.plot-section:nth-child(2) .header-row h3').textContent = 'Point Analysis';
                    document.getElementById('pointPlotContent').innerHTML = '';
                    clickedPointData = null;
                }}
            }});
            
            document.getElementById('yAxis').addEventListener('change', function() {{
                if (clickedPointData) {{
                    console.log('Y-axis changed, clearing second plot');
                    document.querySelector('.plot-section:nth-child(2) .header-row h3').textContent = 'Point Analysis';
                    document.getElementById('pointPlotContent').innerHTML = '';
                    clickedPointData = null;
                }}
            }});
            
            document.getElementById('zAxis').addEventListener('change', function() {{
                if (clickedPointData) {{
                    console.log('Z-axis changed, clearing second plot');
                    document.querySelector('.plot-section:nth-child(2) .header-row h3').textContent = 'Point Analysis';
                    document.getElementById('pointPlotContent').innerHTML = '';
                    clickedPointData = null;
                }}
            }});
            
            function updateCurrentDensity(value) {{
                currentDensityIndex = parseInt(value);
                const currentDensity = currentDensityOptions[currentDensityIndex];
                document.getElementById('currentDensityValue').textContent = currentDensity + ' mA/cm²';
            }}
            
            async function updatePlot() {{
                const loadingDiv = document.getElementById('loading');
                loadingDiv.style.display = 'block';
                
                try {{
                    // Get current selections
                    const xCol = document.getElementById('xAxis').value;
                    const yCol = document.getElementById('yAxis').value;
                    const zCol = document.getElementById('zAxis').value;
                    
                    // Get current unit type
                    const selectedUnit = document.querySelector('input[name="unitType"]:checked').value;
                    
                    // Prepare request data based on current mode
                    let requestData = {{
                        mode: currentMode,
                        xAxis: xCol,
                        yAxis: yCol,
                        zAxis: zCol,
                        unitType: selectedUnit
                    }};
                    
                    if (currentMode === 'current_density') {{
                        requestData.currentDensity = currentDensityOptions[currentDensityIndex];
                    }} else {{
                        requestData.voltage = currentVoltage;
                    }}
                    
                    // Fetch new data
                    const response = await fetch('/co2/update_data', {{
                        method: 'POST',
                        headers: {{
                            'Content-Type': 'application/json',
                        }},
                        body: JSON.stringify(requestData)
                    }});
                    
                    if (!response.ok) {{
                        throw new Error('Network response was not ok');
                    }}
                    
                    const result = await response.json();
                    currentData = result.data;
                    originalData = result.originalData || result.data; // Use original data for calculations
                    
                    // Update the plot with new data
                    createPlot(xCol, yCol, zCol, currentData, originalData);
                    
                    // Update the second plot if points were clicked
                    await updatePointPlot();
                    
                }} catch (error) {{
                    console.error('Error updating plot:', error);
                    loadingDiv.textContent = 'Error updating plot. Please try again.';
                }} finally {{
                    loadingDiv.style.display = 'none';
                }}
            }}
            
            function createPlot(xCol, yCol, zCol, data, originalDataForCalc = null) {{
                // Use original data for calculations if available, otherwise use display data
                const calcData = originalDataForCalc || data;
                
                console.log('Creating CO2R plot with:', xCol, 'vs', yCol, 'colored by', zCol);
                console.log('Data points:', data.length);
                
                if (!data || data.length === 0) {{
                    document.getElementById('plot').innerHTML = '<div style="text-align: center; padding: 50px;"><h3>No data available for plotting</h3></div>';
                    return;
                }}
                
                // Calculate Cu means for the selected y column using original data
                let cuMeanUoft = null;
                let cuMeanVsp = null;
                
                // Find the y-axis value where Cu=1.0 for each source
                for (let row of calcData) {{
                    if (Math.abs(row['Cu'] - 1.0) < 0.001 && row['source'] === 'uoft') {{
                        cuMeanUoft = row[yCol];
                    }}
                    if (Math.abs(row['Cu'] - 1.0) < 0.001 && row['source'] === 'vsp') {{
                        cuMeanVsp = row[yCol];
                    }}
                }}
                
                // Create separate traces for UOFT and VSP points using original data for color calculations
                const uoftData = calcData.filter(row => row['source'] === 'uoft');
                const vspData = calcData.filter(row => row['source'] === 'vsp');
                
                const traces = [];
                
                // Helper: build a clean XY array filtering invalid numbers
                function buildXY(rows, source) {{
                    const x = [];
                    const y = [];
                    const text = [];
                    const errorArray = [];
                    const customdata = [];
                    
                    rows.forEach(row => {{
                        const xv = Number(row[xCol]);
                        const yv = Number(row[yCol]);
                        if (Number.isFinite(xv) && Number.isFinite(yv)) {{
                            x.push(xv);
                            y.push(yv);
                            
                            // For error bars, check if corresponding _std column exists
                            let errorValue = 0;
                            if (yCol.includes('_mean')) {{
                                const stdCol = yCol.replace('_mean', '_std');
                                if (row[stdCol] !== undefined) {{
                                    errorValue = Number(row[stdCol]) || 0;
                                }}
                            }}
                            errorArray.push(errorValue);
                            
                            // Add sample ID to customdata for visual feedback
                            customdata.push(row['sample id'] || 'Unknown');
                            
                            text.push(
                                `Source: ${{row['source']}}<br>Sample ID: ${{row['sample id'] || 'N/A'}}<br>Batch: ${{row['batch number'] || 'N/A'}} (${{row['batch date'] || 'N/A'}})<br>Chemical Formula: ${{row['xrf composition'] || row['target composition'] || 'N/A'}}<br>${{currentMode === 'current_density' ? 'Current Density: ' + currentDensityOptions[currentDensityIndex] + ' mA/cm²' : 'Voltage: ' + currentVoltage.toFixed(2) + 'V'}}<br>X: ${{xv.toFixed(3)}}<br>Y: ${{yv.toFixed(3)}}`
                            );
                        }}
                    }});
                    return {{ x, y, text, errors: errorArray, customdata }};
                }}
                
                // UOFT points (circles)
                if (uoftData.length > 0) {{
                    if (zCol === 'default_colors') {{
                        // Group UOFT data by color for default coloring
                        const uoftByColor = {{}};
                        
                        uoftData.forEach((row, index) => {{
                            const yValue = row[yCol];
                            let color = '#6b7280';
                            
                            if (cuMeanUoft !== null && cuMeanVsp !== null) {{
                                if (cuMeanVsp > cuMeanUoft) {{
                                    if (yValue >= cuMeanVsp) color = '#3b82f6';
                                    else if (yValue >= cuMeanUoft) color = '#ef4444';
                                    else color = '#6b7280';
                                }} else if (cuMeanUoft > cuMeanVsp) {{
                                    if (yValue >= cuMeanUoft) color = '#ef4444';
                                    else if (yValue >= cuMeanVsp) color = '#3b82f6';
                                    else color = '#6b7280';
                                }} else {{
                                    color = '#3b82f6';
                                }}
                            }}
                            
                            if (!uoftByColor[color]) {{
                                uoftByColor[color] = [];
                            }}
                            uoftByColor[color].push(row);
                        }});
                        
                        // Create separate trace for each color
                        Object.keys(uoftByColor).forEach(color => {{
                            const colorData = uoftByColor[color];
                            // Find corresponding display data for this color group
                            const displayColorData = data.filter(displayRow => 
                                colorData.some(origRow => origRow['sample id'] === displayRow['sample id'])
                            );
                            
                            const d = buildXY(displayColorData);
                            const trace = {{
                                x: d.x,
                                y: d.y,
                                mode: 'markers',
                                type: 'scatter',
                                marker: {{
                                    size: 12,
                                    color: color,
                                    line: {{
                                        width: d.customdata.map(id => clickedPoints.has(id) ? 4 : 1.5),
                                        color: d.customdata.map(id => clickedPoints.has(id) ? '#00FF00' : 'rgba(0,0,0,0.3)')
                                    }},
                                    symbol: 'circle',
                                    opacity: 0.9
                                }},
                                text: d.text,
                                hoverinfo: 'text',
                                showlegend: true,
                                name: 'UofT (chemical reduction)',
                                customdata: d.customdata
                            }};
                            
                            // Add error bars for fe_ columns or voltage columns (only if checkbox is checked)
                            if ((yCol.startsWith('fe_') || yCol === 'voltage_mean' || yCol === 'voltage') && document.getElementById('errorBars').checked) {{
                                trace.error_y = {{
                                    type: 'data',
                                    array: d.errors,
                                    visible: true,
                                    color: color,
                                    thickness: 1.5,
                                    width: 2
                                }};
                            }}
                            
                            traces.push(trace);
                        }});
                    }} else {{
                        // Single trace with coloraxis for custom z-axis
                        const d = buildXY(uoftData);
                        const trace = {{
                            x: d.x,
                            y: d.y,
                            mode: 'markers',
                            type: 'scatter',
                            marker: {{
                                size: 12,
                                color: uoftData.map(row => row[zCol]),
                                line: {{
                                    width: d.customdata.map(id => clickedPoints.has(id) ? 4 : 1.5),
                                    color: d.customdata.map(id => clickedPoints.has(id) ? 'red' : 'rgba(0,0,0,0.3)')
                                }},
                                symbol: 'circle',
                                opacity: 0.9,
                                coloraxis: 'coloraxis'
                            }},
                            text: d.text,
                            hoverinfo: 'text',
                            showlegend: false,
                            name: 'UofT (chemical reduction)',
                            customdata: d.customdata
                        }};
                        
                        // Add error bars for fe_ columns or voltage columns (only if checkbox is checked)
                        if ((yCol.startsWith('fe_') || yCol === 'voltage_mean' || yCol === 'voltage') && document.getElementById('errorBars').checked) {{
                            trace.error_y = {{
                                type: 'data',
                                array: d.errors,
                                visible: true,
                                thickness: 1.5,
                                width: 2
                            }};
                        }}
                        
                        traces.push(trace);
                    }}
                }}
                
                // VSP points (diamonds) - similar logic as UOFT
                if (vspData.length > 0) {{
                    if (zCol === 'default_colors') {{
                        // Group VSP data by color for default coloring
                        const vspByColor = {{}};
                        
                        vspData.forEach((row, index) => {{
                            const yValue = row[yCol];
                            let color = '#6b7280';
                            
                            if (cuMeanUoft !== null && cuMeanVsp !== null) {{
                                if (cuMeanVsp > cuMeanUoft) {{
                                    if (yValue >= cuMeanVsp) color = '#3b82f6';
                                    else if (yValue >= cuMeanUoft) color = '#ef4444';
                                    else color = '#6b7280';
                                }} else if (cuMeanUoft > cuMeanVsp) {{
                                    if (yValue >= cuMeanUoft) color = '#ef4444';
                                    else if (yValue >= cuMeanVsp) color = '#3b82f6';
                                    else color = '#6b7280';
                                }} else {{
                                    color = '#3b82f6';
                                }}
                            }}
                            
                            if (!vspByColor[color]) {{
                                vspByColor[color] = [];
                            }}
                            vspByColor[color].push(row);
                        }});
                        
                        // Create separate trace for each color
                        Object.keys(vspByColor).forEach(color => {{
                            const colorData = vspByColor[color];
                            const displayColorData = data.filter(displayRow => 
                                colorData.some(origRow => origRow['sample id'] === displayRow['sample id'])
                            );
                            
                            const d = buildXY(displayColorData);
                            const trace = {{
                                x: d.x,
                                y: d.y,
                                mode: 'markers',
                                type: 'scatter',
                                marker: {{
                                    size: 12,
                                    color: color,
                                    line: {{
                                        width: d.customdata.map(id => clickedPoints.has(id) ? 4 : 1.5),
                                        color: d.customdata.map(id => clickedPoints.has(id) ? '#00FF00' : 'rgba(0,0,0,0.3)')
                                    }},
                                    symbol: 'diamond',
                                    opacity: 0.9
                                }},
                                text: d.text,
                                hoverinfo: 'text',
                                showlegend: true,
                                name: 'VSP (spark ablation)',
                                customdata: d.customdata
                            }};
                            
                            // Add error bars for fe_ columns or voltage columns (only if checkbox is checked)
                            if ((yCol.startsWith('fe_') || yCol === 'voltage_mean' || yCol === 'voltage') && document.getElementById('errorBars').checked && d.errors.some(e => e > 0)) {{
                                trace.error_y = {{
                                    type: 'data',
                                    array: d.errors,
                                    visible: true,
                                    color: color,
                                    thickness: 1.5,
                                    width: 2
                                }};
                            }}
                            
                            traces.push(trace);
                        }});
                    }} else {{
                        // Single trace with coloraxis for custom z-axis
                        const d = buildXY(vspData);
                        const trace = {{
                            x: d.x,
                            y: d.y,
                            mode: 'markers',
                            type: 'scatter',
                            marker: {{
                                size: 12,
                                color: vspData.map(row => row[zCol]),
                                line: {{
                                    width: d.customdata.map(id => clickedPoints.has(id) ? 4 : 1.5),
                                    color: d.customdata.map(id => clickedPoints.has(id) ? 'red' : 'rgba(0,0,0,0.3)')
                                }},
                                symbol: 'diamond',
                                opacity: 0.9,
                                coloraxis: 'coloraxis'
                            }},
                            text: d.text,
                            hoverinfo: 'text',
                            showlegend: false,
                            name: 'VSP (spark ablation)',
                            customdata: d.customdata
                        }};
                        
                        // Add error bars for fe_ columns or voltage columns (only if checkbox is checked)
                        if ((yCol.startsWith('fe_') || yCol === 'voltage_mean' || yCol === 'voltage') && document.getElementById('errorBars').checked && d.errors.some(e => e > 0)) {{
                            trace.error_y = {{
                                type: 'data',
                                array: d.errors,
                                visible: true,
                                thickness: 1.5,
                                width: 2
                            }};
                        }}
                        
                        traces.push(trace);
                    }}
                }}
                
                // Get units for axis labels
                const selectedUnit = document.querySelector('input[name="unitType"]:checked').value;
                const xAxisUnit = getColumnUnits(xCol, selectedUnit);
                const yAxisUnit = getColumnUnits(yCol, selectedUnit);
                
                // Format column names for display
                const xColFormatted = formatColumnName(xCol);
                const yColFormatted = formatColumnName(yCol);
                
                // Get units for z-axis label
                const zAxisUnit = getColumnUnits(zCol, selectedUnit);
                const zColFormatted = formatColumnName(zCol);
                
                const layout = {{
                    title: {{
                        text: `${{xColFormatted}} vs ${{yColFormatted}} ${{currentMode === 'current_density' ? 'at ' + currentDensityOptions[currentDensityIndex] + ' mA/cm²' : 'at ' + currentVoltage.toFixed(2) + 'V'}}`,
                        font: {{ size: 18, color: '#202124' }},
                        x: 0.5
                    }},
                    xaxis: {{ 
                        title: xColFormatted + xAxisUnit, 
                        showgrid: true, 
                        gridwidth: 1, 
                        gridcolor: 'lightgray',
                        zerolinecolor: '#ccc',
                        color: '#333',
                        titlefont: {{ size: 14, color: '#666' }},
                        tickfont: {{ size: 12, color: '#666' }}
                    }},
                    yaxis: {{ 
                        title: yColFormatted + yAxisUnit, 
                        showgrid: true, 
                        gridwidth: 1, 
                        gridcolor: 'lightgray',
                        zerolinecolor: '#ccc',
                        color: '#333',
                        titlefont: {{ size: 14, color: '#666' }},
                        tickfont: {{ size: 12, color: '#666' }}
                    }},
                    hovermode: 'closest',
                    template: 'plotly_white',
                    width: 800,
                    height: 600,
                    showlegend: zCol === 'default_colors',
                    margin: {{ l: 60, r: 30, t: 60, b: 60 }},
                    legend: {{
                        x: 1.02,
                        y: 1,
                        bgcolor: 'rgba(255,255,255,0.8)',
                        bordercolor: '#ccc',
                        borderwidth: 1
                    }}
                }};
                
                // Add color bar if using custom z-axis
                if (zCol !== 'default_colors') {{
                    // Calculate min and max values for color bar
                    const zValues = data.map(d => d[zCol]).filter(v => v !== null && v !== undefined);
                    const minZ = Math.min(...zValues);
                    const maxZ = Math.max(...zValues);
                    
                    layout.coloraxis = {{
                        colorscale: [[0, '#3b82f6'], [0.5, '#f59e0b'], [1, '#ef4444']],
                        cmin: minZ,
                        cmax: maxZ,
                        colorbar: {{
                            title: {{
                                text: zColFormatted + getColumnUnits(zCol, selectedUnit),
                                font: {{ size: 14, color: '#666' }}
                            }},
                            tickfont: {{ size: 12, color: '#666' }},
                            len: 0.8,
                            y: 0.5,
                            yanchor: 'middle',
                            x: 1.02,
                            xanchor: 'left'
                        }}
                    }};
                    
                    // Update margin to make room for color bar
                    layout.margin = {{ l: 60, r: 100, t: 60, b: 60 }};
                }}
                
                // Add reference lines with annotations if available
                const shapes = [];
                const annotations = [];
                
                if (cuMeanUoft !== null) {{
                    // Calculate x-axis range more robustly
                    const xValues = data.map(row => row[xCol]).filter(val => val !== null && !isNaN(val));
                    const xMin = Math.min(...xValues);
                    const xMax = Math.max(...xValues);
                    const xRange = xMax - xMin;
                    
                    shapes.push({{
                        type: 'line',
                        x0: xMin - xRange * 0.1,
                        x1: xMax + xRange * 0.1,
                        y0: cuMeanUoft,
                        y1: cuMeanUoft,
                        line: {{ color: '#ef4444', dash: 'dash', width: 3 }}
                    }});
                    
                    // Add annotation for UOFT line
                    annotations.push({{
                        x: xMax + xRange * 0.1,
                        y: cuMeanUoft,
                        text: `Cu (UofT)`,
                        showarrow: false,
                        xanchor: 'left',
                        yanchor: 'middle',
                        bgcolor: 'rgba(255,255,255,0.9)',
                        bordercolor: '#ef4444',
                        borderwidth: 2,
                        font: {{ color: '#ef4444', size: 14 }}
                    }});
                }}
                
                if (cuMeanVsp !== null) {{
                    // Calculate x-axis range more robustly
                    const xValues = data.map(row => row[xCol]).filter(val => val !== null && !isNaN(val));
                    const xMin = Math.min(...xValues);
                    const xMax = Math.max(...xValues);
                    const xRange = xMax - xMin;
                    
                    shapes.push({{
                        type: 'line',
                        x0: xMin - xRange * 0.1,
                        x1: xMax + xRange * 0.1,
                        y0: cuMeanVsp,
                        y1: cuMeanVsp,
                        line: {{ color: '#3b82f6', dash: 'dot', width: 3 }}
                    }});
                    
                    // Add annotation for VSP line
                    annotations.push({{
                        x: xMax + xRange * 0.1,
                        y: cuMeanVsp,
                        text: `Cu (VSP)`,
                        showarrow: false,
                        xanchor: 'left',
                        yanchor: 'middle',
                        bgcolor: 'rgba(255,255,255,0.9)',
                        bordercolor: '#3b82f6',
                        borderwidth: 2,
                        font: {{ color: '#3b82f6', size: 14 }}
                    }});
                }}
                
                if (shapes.length > 0) {{
                    layout.shapes = shapes;
                }}
                
                if (annotations.length > 0) {{
                    layout.annotations = annotations;
                }}
                
                Plotly.newPlot('plot', traces, layout);
                
                // Add click event to the plot
                document.getElementById('plot').on('plotly_click', function(data) {{
                    const point = data.points[0];
                    
                    // Get the current axis selections
                    const xCol = document.getElementById('xAxis').value;
                    const yCol = document.getElementById('yAxis').value;
                    
                    // Extract the clicked point information directly from the trace data
                    const clickedX = point.x;
                    const clickedY = point.y;
                    const clickedSource = point.data.name; // This will be 'UofT (chemical reduction)' or 'VSP (spark ablation)'
                    
                    // Determine the source from the trace name
                    let source = 'uoft';
                    if (clickedSource.includes('VSP')) {{
                        source = 'vsp';
                    }}
                    
                    // Use the hover text to get complete sample information
                    let xrfComposition = 'Unknown';
                    let sampleId = 'Unknown';
                    let batchNumber = 'Unknown';
                    let batchDate = 'Unknown';
                    
                    try {{
                        const hoverText = point.data.text[point.pointIndex];
                        if (hoverText) {{
                            // Extract composition (XRF or target)
                            if (hoverText.includes('Chemical Formula:')) {{
                                xrfComposition = hoverText.split('Chemical Formula: ')[1].split('<br>')[0];
                            }}
                            // Extract Sample ID
                            if (hoverText.includes('Sample ID:')) {{
                                sampleId = hoverText.split('Sample ID: ')[1].split('<br>')[0];
                            }}
                            // Extract Batch information
                            if (hoverText.includes('Batch:')) {{
                                const batchInfo = hoverText.split('Batch: ')[1].split('<br>')[0];
                                // Parse "B001 (2024-01-01)" format
                                if (batchInfo.includes(' (')) {{
                                    batchNumber = batchInfo.split(' (')[0];
                                    batchDate = batchInfo.split(' (')[1].replace(')', '');
                                }} else {{
                                    batchNumber = batchInfo;
                                }}
                            }}
                        }}
                    }} catch (e) {{
                        console.log('Could not parse hover text, using fallback method');
                    }}
                    
                    // If hover text parsing failed, try to find the point in currentData as fallback
                    if (xrfComposition === 'Unknown') {{
                        console.log('Trying fallback method to find point data...');
                        for (let i = 0; i < currentData.length; i++) {{
                            const dataPoint = currentData[i];
                            if (Math.abs(dataPoint[xCol] - clickedX) < 0.001 && 
                                Math.abs(dataPoint[yCol] - clickedY) < 0.001 &&
                                dataPoint.source === source) {{
                                xrfComposition = dataPoint['xrf composition'] || dataPoint['target composition'];
                                sampleId = dataPoint['sample id'] || 'Unknown';
                                batchNumber = dataPoint['batch number'] || 'Unknown';
                                batchDate = dataPoint['batch date'] || 'Unknown';
                                console.log('Found complete sample data via fallback:', {{xrfComposition, sampleId, batchNumber, batchDate}});
                                break;
                            }}
                        }}
                    }}
                    
                    // Store the clicked point data globally with complete sample information
                    clickedPointData = {{
                        source: source,
                        'xrf composition': xrfComposition,
                        'sample id': sampleId,
                        'batch number': batchNumber,
                        'batch date': batchDate,
                        x_col: xCol,
                        y_col: yCol,
                        clicked_x: clickedX,
                        clicked_y: clickedY
                    }};
                    
                    console.log('Clicked point data stored:', clickedPointData);
                    console.log('Sample ID extracted:', sampleId);
                    console.log('XRF Composition:', xrfComposition);
                    
                    // Check if this point is already selected
                    if (clickedPoints.has(sampleId)) {{
                        console.log('Point already selected, deselecting:', sampleId);
                        
                        // Remove from clicked points set
                        clickedPoints.delete(sampleId);
                        
                        // Update plot to remove green border
                        updateClickedPointVisual(sampleId);
                        
                        // Remove from accumulated points
                        removePointFromAccumulation(sampleId);
                        
                        // Remove XRD data for the clicked sample
                        removeXrdPlot(sampleId);
                        
                        // Update accumulated points display
                        showAccumulatedPoints(xCol, yCol);
                    }} else {{
                        console.log('Point not selected, selecting:', sampleId);
                        
                        // Add to clicked points set for visual feedback
                        clickedPoints.add(sampleId);
                        
                        // Update plot to show clicked point with green border
                        updateClickedPointVisual(sampleId);
                        
                        // Add to accumulated points if it's a new point
                        addPointToAccumulation(clickedPointData);
                        
                        // Show accumulated points in the second plot
                        showAccumulatedPoints(xCol, yCol);
                        
                        // Load XRD data for the clicked sample
                        console.log('About to load XRD for sample:', sampleId);
                        loadXrdPlot(sampleId);
                    }}
                    
                    console.log('Point clicked successfully!');
                }});
            }}
            
            // JavaScript function to get column units
            function getColumnUnits(columnName, unitType = 'atomic') {{
                if (columnName === 'voltage_mean' || columnName === 'voltage') {{
                    return ' (V)';
                }} else if (columnName === 'current density') {{
                    return ' (mA/cm²)';
                }} else if (columnName.startsWith('fe_')) {{
                    return ' (%)';
                }} else if (columnName === 'PCA1' || columnName === 'PCA2') {{
                    return ''; // No units for dimensionless PCA components
                }} else if (['Ag', 'Au', 'Cd', 'Cu', 'Ga', 'Hg', 'In', 'Ni', 'Pd', 'Pt', 'Rh', 'Sn', 'Tl', 'Zn'].includes(columnName)) {{
                    return unitType === 'weight' ? ' (wt. fraction)' : ' (at. fraction)';
                }} else {{
                    return '';
                }}
            }}
            
            // JavaScript function to format column names
            function formatColumnName(columnName) {{
                if (columnName === 'default_colors') {{
                    return 'Default';
                }} else if (columnName === 'voltage_mean' || columnName === 'voltage') {{
                    return 'Full Cell Voltage (V)';
                }} else if (columnName === 'current density') {{
                    return 'Current Density';
                }} else if (columnName.startsWith('fe_')) {{
                    // Convert fe_h2_mean to "Faradaic Efficiency H2"
                    const baseName = columnName.replace('fe_', '').replace('_mean', '');
                    if (baseName === 'h2') {{
                        return 'Faradaic Efficiency H₂';
                    }} else if (baseName === 'co') {{
                        return 'Faradaic Efficiency CO';
                    }} else if (baseName === 'ch4') {{
                        return 'Faradaic Efficiency CH₄';
                    }} else if (baseName === 'c2h4') {{
                        return 'Faradaic Efficiency C₂H₄';
                    }} else if (baseName === 'gas_total') {{
                        return 'Faradaic Efficiency Gas Total';
                    }} else if (baseName === 'liquid') {{
                        return 'Faradaic Efficiency Liquid';
                    }} else {{
                        return 'Faradaic Efficiency ' + baseName.toUpperCase();
                    }}
                }} else if (columnName === 'PCA1' || columnName === 'PCA2') {{
                    return columnName;
                }} else if (['Ag', 'Au', 'Cd', 'Cu', 'Ga', 'Hg', 'In', 'Ni', 'Pd', 'Pt', 'Rh', 'Sn', 'Tl', 'Zn'].includes(columnName)) {{
                    return columnName;
                }} else {{
                    return columnName;
                }}
            }}
            
            // Function to add a new point to accumulation
            function addPointToAccumulation(pointData) {{
                // Check if this point is already in accumulation
                const isDuplicate = accumulatedPoints.some(point => 
                    point.source === pointData.source && 
                    point['xrf composition'] === pointData['xrf composition']
                );
                
                if (!isDuplicate) {{
                    accumulatedPoints.push(pointData);
                    console.log('Point added to accumulation. Total points:', accumulatedPoints.length);
                }} else {{
                    console.log('Point already in accumulation, skipping duplicate');
                }}
            }}
            
            // Function to remove a point from accumulation
            function removePointFromAccumulation(sampleId) {{
                console.log('Removing point from accumulation for sample:', sampleId);
                
                // Find and remove the point by sample ID
                const index = accumulatedPoints.findIndex(point => point['sample id'] === sampleId);
                if (index !== -1) {{
                    accumulatedPoints.splice(index, 1);
                    console.log('Removed point from accumulation. Remaining points:', accumulatedPoints.length);
                }} else {{
                    console.log('Point not found in accumulation:', sampleId);
                }}
            }}
            
            // Function to remove XRD plot for a specific sample ID
            function removeXrdPlot(sampleId) {{
                console.log('Removing XRD plot for sample:', sampleId);
                
                // Remove from accumulation
                const index = accumulatedXrdData.findIndex(item => item.sampleId === sampleId);
                if (index !== -1) {{
                    accumulatedXrdData.splice(index, 1);
                    console.log('Removed XRD data from accumulation. Remaining samples:', accumulatedXrdData.length);
                    
                    // Update XRD plot display
                    showAccumulatedXrdPlots();
                }} else {{
                    console.log('Sample not found in XRD accumulation:', sampleId);
                }}
            }}
            
            // Function to show accumulated points in the second plot
            async function showAccumulatedPoints(xCol, yCol) {{
                if (accumulatedPoints.length === 0) {{
                    document.querySelector('.plot-section:nth-child(2) .header-row h3').textContent = 'Point Analysis';
                    document.getElementById('pointPlotContent').innerHTML = '';
                    return;
                }}
                
                // Update title to show multiple points
                document.querySelector('.plot-section:nth-child(2) .header-row h3').textContent = 'Point Analysis';
                
                // Fetch data for all accumulated points
                const allPointData = [];
                for (const point of accumulatedPoints) {{
                    try {{
                        const response = await fetch('/co2/get_point_data', {{
                            method: 'POST',
                            headers: {{ 'Content-Type': 'application/json' }},
                            body: JSON.stringify({{
                                source: point.source,
                                xrf_composition: point['xrf composition'],
                                x_col: xCol,
                                y_col: yCol,
                                mode: currentMode
                            }})
                        }});
                        
                        if (response.ok) {{
                            const result = await response.json();
                            if (result.success) {{
                                allPointData.push({{
                                    point: point,
                                    data: result.data
                                }});
                            }}
                        }}
                    }} catch (error) {{
                        console.error('Error fetching data for point:', point, error);
                    }}
                }}
                
                // Create the combined plot
                createAccumulatedPointPlot(allPointData, xCol, yCol);
            }}
            
            // Function to create the accumulated points plot
            function createAccumulatedPointPlot(allPointData, xCol, yCol) {{
                if (allPointData.length === 0) {{
                    document.getElementById('pointPlotContent').innerHTML = 
                        '<p style="text-align: center; color: #5f6368; margin-top: 50px; font-size: 1.1em;">No data available for accumulated points</p>';
                    return;
                }}
                
                const traces = [];
                const colors = ['#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6', '#ec4899', '#06b6d4', '#84cc16'];
                
                allPointData.forEach((pointData, index) => {{
                    const point = pointData.point;
                    const data = pointData.data;
                    
                    // Filter out any null values
                    const validData = data.filter(d => d.x_value !== null && d.y_value !== null);
                    
                    if (validData.length > 0) {{
                        const color = colors[index % colors.length];
                        // Use the same symbol logic as the main plot (circle for UOFT, diamond for VSP)
                        const symbol = point.source === 'uoft' ? 'circle' : 'diamond';
                        
                        const trace = {{
                            x: validData.map(d => currentMode === 'current_density' ? d.current_density : d.voltage),
                            y: validData.map(d => d.y_value),
                            mode: 'markers+lines',
                            type: 'scatter',
                            marker: {{
                                size: 10,
                                color: color,
                                symbol: symbol,
                                line: {{ width: 1, color: 'rgba(0,0,0,0.5)' }}
                            }},
                            line: {{
                                color: color,
                                width: 2
                            }},
                            text: validData.map(d => 'Source: ' + point['source'] + '<br>Chemical Formula: ' + point['xrf composition'] + '<br>Sample ID: ' + point['sample id'] + '<br>Batch: ' + point['batch number'] + ' (' + point['batch date'] + ')<br>' + (currentMode === 'current_density' ? 'Current Density: ' + d.current_density + ' mA/cm²' : 'Voltage: ' + d.voltage + 'V') + '<br>Y: ' + (d.y_value?.toFixed(3) || 'N/A')),
                            hoverinfo: 'text',
                            name: point.source + ' - ' + point['xrf composition']
                        }};
                        
                        traces.push(trace);
                    }}
                }});
                
                const layout = {{
                    title: {{
                        text: formatColumnName(yCol) + ' vs ' + (currentMode === 'current_density' ? 'Current Density' : 'Voltage') + ' - Multiple Points',
                        font: {{ size: 18, color: '#202124' }},
                        x: 0.5
                    }},
                    xaxis: {{ 
                        title: currentMode === 'current_density' ? 'Current Density (mA/cm²)' : 'Full Cell Voltage (V)', 
                        showgrid: true, 
                        gridwidth: 1, 
                        gridcolor: 'lightgray',
                        zerolinecolor: '#ccc',
                        color: '#333',
                        titlefont: {{ size: 14, color: '#666' }},
                        tickfont: {{ size: 12, color: '#666' }}
                    }},
                    yaxis: {{ 
                        title: formatColumnName(yCol) + getColumnUnits(yCol), 
                        showgrid: true, 
                        gridwidth: 1, 
                        gridcolor: 'lightgray',
                        zerolinecolor: '#ccc',
                        color: '#333',
                        titlefont: {{ size: 14, color: '#666' }},
                        tickfont: {{ size: 12, color: '#666' }}
                    }},
                    hovermode: 'closest',
                    template: 'plotly_white',
                    width: 700,
                    height: 500,
                    showlegend: true,
                    margin: {{ l: 60, r: 30, t: 60, b: 60 }},
                    legend: {{
                        x: 1.02,
                        y: 1,
                        bgcolor: 'rgba(255,255,255,0.8)',
                        bordercolor: '#ccc',
                        borderwidth: 1
                    }}
                }};
                
                Plotly.newPlot('pointPlotContent', traces, layout);
            }}
            
            // Function to reset the accumulated points plot
            function resetPointPlot() {{
                accumulatedPoints = [];
                clickedPointData = null;
                document.querySelector('.plot-section:nth-child(2) .header-row h3').textContent = 'Point Analysis';
                document.getElementById('pointPlotContent').innerHTML = '';
                
                // Also reset XRD data and clicked points
                accumulatedXrdData = [];
                clickedPoints.clear(); // Clear clicked points
                document.getElementById('xrdPlotContent').innerHTML = '';
                
                // Update plot to remove green borders
                const plotDiv = document.getElementById('plot');
                if (plotDiv && plotDiv.data) {{
                    plotDiv.data.forEach(trace => {{
                        if (trace.marker && trace.marker.line) {{
                            trace.marker.line.width = 1.5;
                            trace.marker.line.color = 'rgba(0,0,0,0.3)';
                        }}
                    }});
                    Plotly.redraw('plot');
                }}
                
                console.log('Point Analysis and XRD reset');
            }}
            
            // Function to update the second plot when mode or parameters change
            async function updatePointPlot() {{
                if (accumulatedPoints.length > 0) {{
                    const xCol = document.getElementById('xAxis').value;
                    const yCol = document.getElementById('yAxis').value;
                    
                    // Only update if the axes haven't changed
                    const firstPoint = accumulatedPoints[0];
                    if (xCol === firstPoint.x_col && yCol === firstPoint.y_col) {{
                        console.log('Updating second plot for same axes');
                        await showAccumulatedPoints(xCol, yCol);
                    }} else {{
                        console.log('Axes changed, clearing second plot');
                        // Clear the second plot when axes change
                        document.querySelector('.plot-section:nth-child(2) .header-row h3').textContent = 'Point Analysis';
                        document.getElementById('pointPlotContent').innerHTML = '';
                        accumulatedPoints = []; // Reset accumulated points when axes change
                        clickedPointData = null;
                    }}
                }}
            }}
            
            // Function to load XRD plot for a specific sample ID and add to accumulation
            async function loadXrdPlot(sampleId) {{
                console.log('DEBUG: Loading XRD plot for sample:', sampleId);
                
                try {{
                    // Get selected data type from toggle
                    const dataType = document.querySelector('input[name="xrdDataType"]:checked').value;
                    console.log('DEBUG: Selected data type:', dataType);
                    
                    // Fetch XRD data for the specific sample
                    const response = await fetch('/co2/get_xrd_data', {{
                        method: 'POST',
                        headers: {{
                            'Content-Type': 'application/json',
                        }},
                        body: JSON.stringify({{ 
                            sample_id: sampleId,
                            data_type: dataType
                        }})
                    }});
                    
                    console.log('DEBUG: Fetch response status:', response.status);
                    
                    if (!response.ok) {{
                        const errorData = await response.json();
                        throw new Error(errorData.error || 'Failed to fetch XRD data');
                    }}
                    
                    const result = await response.json();
                    
                    if (!result.success) {{
                        throw new Error(result.error || 'XRD data not found');
                    }}
                    
                    // Add XRD data to accumulation
                    addXrdToAccumulation(result.data, result.sample_id, result.data_points);
                    
                }} catch (error) {{
                    console.error('Error loading XRD data:', error);
                    // Show error in popup instead of replacing plot content
                    alert('Error loading XRD data:\\n\\n' + error.message);
                }}
            }}
            
            // Function to add XRD data to accumulation
            function addXrdToAccumulation(xrdData, sampleId, dataPoints) {{
                // Check if this sample is already in accumulation
                const existingIndex = accumulatedXrdData.findIndex(item => item.sampleId === sampleId);
                
                if (existingIndex !== -1) {{
                    console.log('Sample already in XRD accumulation:', sampleId);
                    return; // Don't add duplicates
                }}
                
                // Get source and XRF composition from the clicked point data
                const source = clickedPointData ? clickedPointData.source : 'Unknown';
                const xrfComposition = clickedPointData ? clickedPointData['xrf composition'] : 'Unknown';
                
                // Add to accumulation
                accumulatedXrdData.push({{
                    data: xrdData,
                    sampleId: sampleId,
                    dataPoints: dataPoints,
                    source: source,
                    xrfComposition: xrfComposition
                }});
                
                console.log('Added XRD data to accumulation. Total samples:', accumulatedXrdData.length);
                
                // Show accumulated XRD plots
                showAccumulatedXrdPlots();
            }}
            
            // Function to show all accumulated XRD plots
            function showAccumulatedXrdPlots() {{
                if (accumulatedXrdData.length === 0) {{
                    document.getElementById('xrdPlotContent').innerHTML = '';
                    return;
                }}
                
                // Create traces for all accumulated XRD data
                const traces = [];
                const colors = ['#4285f4', '#ea4335', '#34a853', '#fbbc04', '#ff6d01', '#9c27b0', '#00bcd4', '#795548'];
                
                accumulatedXrdData.forEach((xrdItem, index) => {{
                    const color = colors[index % colors.length];
                    
                    // Use stored source and XRF composition to match point analysis format
                    const source = xrdItem.source || 'Unknown';
                    const xrfComposition = xrdItem.xrfComposition || 'Unknown';
                    
                    const trace = {{
                        x: xrdItem.data.x,
                        y: xrdItem.data.y,
                        mode: 'lines',
                        type: 'scatter',
                        line: {{
                            color: color,
                            width: 2
                        }},
                        name: source + ' - ' + xrfComposition,
                        hovertemplate: '<br>2θ: %{{x:.2f}}°<br>Intensity: %{{y:.2f}}<extra></extra>'
                    }};
                    traces.push(trace);
                }});
                
                const layout = {{
                    xaxis: {{ 
                        title: '2θ (degrees)', 
                        showgrid: true, 
                        gridwidth: 1, 
                        gridcolor: 'lightgray',
                        zerolinecolor: '#ccc',
                        color: '#333',
                        titlefont: {{ size: 14, color: '#666' }},
                        tickfont: {{ size: 12, color: '#666' }}
                    }},
                    yaxis: {{ 
                        title: 'Intensity (counts)', 
                        showgrid: true, 
                        gridwidth: 1, 
                        gridcolor: 'lightgray',
                        zerolinecolor: '#ccc',
                        color: '#333',
                        titlefont: {{ size: 14, color: '#666' }},
                        tickfont: {{ size: 12, color: '#666' }}
                    }},
                    hovermode: 'closest',
                    template: 'plotly_white',
                    autosize: true,
                    showlegend: true,
                    legend: {{
                        x: 1.02,
                        y: 1,
                        bgcolor: 'rgba(255,255,255,0.8)',
                        bordercolor: '#ccc',
                        borderwidth: 1
                    }},
                    margin: {{ l: 60, r: 150, t: 20, b: 60 }},
                    width: 1500,
                    height: 400
                }};
                
                Plotly.newPlot('xrdPlotContent', traces, layout);
                
                console.log('DEBUG: Accumulated XRD plots created successfully. Total traces:', traces.length);
            }}
            
            // Function to reset XRD accumulation
            function resetXrdPlot() {{
                accumulatedXrdData = [];
                clickedPoints.clear(); // Clear clicked points
                document.getElementById('xrdPlotContent').innerHTML = '';
                
                // Also reset point analysis
                accumulatedPoints = [];
                clickedPointData = null;
                document.querySelector('.plot-section:nth-child(2) .header-row h3').textContent = 'Point Analysis';
                document.getElementById('pointPlotContent').innerHTML = '';
                
                // Update plot to remove green borders
                const plotDiv = document.getElementById('plot');
                if (plotDiv && plotDiv.data) {{
                    plotDiv.data.forEach(trace => {{
                        if (trace.marker && trace.marker.line) {{
                            trace.marker.line.width = 1.5;
                            trace.marker.line.color = 'rgba(0,0,0,0.3)';
                        }}
                    }});
                    Plotly.redraw('plot');
                }}
                
                console.log('XRD and Point Analysis reset');
            }}
            
            // Function to update visual appearance of clicked points
            function updateClickedPointVisual(sampleId) {{
                // Get current plot data
                const plotDiv = document.getElementById('plot');
                const plotData = plotDiv.data;
                
                // Update marker borders for clicked points
                plotData.forEach(trace => {{
                    if (trace.customdata) {{
                        trace.marker.line.width = trace.customdata.map(id => 
                            clickedPoints.has(id) ? 4 : 1.5
                        );
                    trace.marker.line.color = trace.customdata.map(id => 
                        clickedPoints.has(id) ? '#00FF00' : 'rgba(0,0,0,0.3)'
                    );
                    }}
                }});
                
                // Redraw the plot efficiently
                Plotly.redraw('plot');
            }}
            
            // Function to export data as CSV
            async function exportData() {{
                try {{
                    const exportBtn = document.getElementById('exportBtn');
                    exportBtn.textContent = '⊞ Exporting...';
                    exportBtn.disabled = true;
                    
                    // Send request to export CSV
                    const response = await fetch('/co2/export_csv', {{
                        method: 'POST',
                        headers: {{
                            'Content-Type': 'application/json',
                        }},
                        body: JSON.stringify({{
                            mode: currentMode,
                            currentDensity: currentMode === 'current_density' ? currentDensityOptions[currentDensityIndex] : null,
                            voltage: currentMode === 'voltage' ? currentVoltage : null
                        }})
                    }});
                    
                    if (!response.ok) {{
                        throw new Error('Export failed');
                    }}
                    
                    // Get the CSV data
                    const csvData = await response.text();
                    
                    // Create and download the file
                    const blob = new Blob([csvData], {{ type: 'text/csv' }});
                    const url = window.URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    const filename = currentMode === 'current_density' ? 
                        `co2_data_${{currentDensityOptions[currentDensityIndex]}}mA_cm2.csv` : 
                        `co2_data_${{currentVoltage.toFixed(1)}}V.csv`;
                    a.download = filename;
                    document.body.appendChild(a);
                    a.click();
                    window.URL.revokeObjectURL(url);
                    document.body.removeChild(a);
                    
                    // Reset button
                    exportBtn.textContent = '⊞ Export Data (CSV)';
                    exportBtn.disabled = false;
                    
                }} catch (error) {{
                    console.error('Export error:', error);
                    alert('Export failed. Please try again.');
                    
                    // Reset button
                    const exportBtn = document.getElementById('exportBtn');
                    exportBtn.textContent = '⊞ Export Data (CSV)';
                    exportBtn.disabled = false;
                }}
            }}
            
            // Initialize the plot
            updatePlot();
            
            // Initialize XRD plot (empty)
            document.getElementById('xrdPlotContent').innerHTML = '';
            
            // Add event listeners to XRD data type toggle
            document.querySelectorAll('input[name="xrdDataType"]').forEach(radio => {{
                radio.addEventListener('change', function() {{
                    console.log('XRD data type changed to:', this.value);
                    // Reload all accumulated XRD plots with new data type - simplified for now
                    if (accumulatedXrdData.length > 0) {{
                        console.log('Would reload XRD plots with new data type');
                    }}
                }});
            }});
            
            // Add event listeners to unit type toggle
            document.querySelectorAll('input[name="unitType"]').forEach(radio => {{
                radio.addEventListener('change', function() {{
                    console.log('Unit type changed to:', this.value);
                    updatePlot();
                }});
            }});
            
        </script>
    </body>
    </html>
    '''
    
    return html_template

@co2_plot_bp.route('/export_csv', methods=['POST'])
def export_csv():
    """Export CO2R data as CSV"""
    try:
        # Get current data and calculate PCA
        current_df = load_original_data()
        df_with_pca = calculate_pca_components(current_df)
        
        # Use the dataframe with PCA components
        csv_data = df_with_pca.to_csv(index=False)
        
        # Create response with CSV data
        response = Response(
            csv_data,
            mimetype='text/csv',
            headers={'Content-Disposition': 'attachment; filename=CO2R_data.csv'}
        )
        
        return response
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# Create a route to update data based on mode
@co2_plot_bp.route('/update_data', methods=['POST'])
def update_data():
    try:
        data = request.get_json()
        mode = data.get('mode', 'current_density')
        x_axis = data.get('xAxis', 'Cu')
        y_axis = data.get('yAxis', 'PCA2')
        z_axis = data.get('zAxis', 'default_colors')
        unit_type = data.get('unitType', 'atomic')  # Get the unit type
        
        # Get current data
        current_df = load_original_data()
        
        if mode == 'current_density':
            current_density = data.get('currentDensity', 100)
            # Filter dataframe at the specified current density
            new_df = filter_df_by_current_density(current_df, current_density)
        else:  # voltage mode
            voltage = data.get('voltage', 3.0)
            # Generate new dataframe at the specified voltage
            new_df = generate_df_at_voltage(current_df, target_voltage=voltage)
        
        new_df = calculate_pca_components(new_df)
        
        # Store original data for calculations (color mapping, reference lines)
        original_df = new_df.copy()
        
        # Apply unit conversion if needed (only for display)
        if unit_type == 'weight':
            # Convert atomic fraction to weight fraction
            element_columns = [col for col in new_df.columns if col in ATOMIC_WEIGHTS]
            new_df = convert_atomic_to_weight_fraction(new_df, element_columns)
        
        # Convert to JSON-serializable format
        df_dict = new_df.to_dict('records')
        original_df_dict = original_df.to_dict('records')
        
        return jsonify({
            'success': True, 
            'data': df_dict,
            'originalData': original_df_dict,
            'unitType': unit_type
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# Create a route to get data for a specific point across all current densities or voltages
@co2_plot_bp.route('/get_point_data', methods=['POST'])
def get_point_data():
    try:
        data = request.get_json()
        source = data.get('source')
        xrf_composition = data.get('xrf_composition')
        x_col = data.get('x_col')
        y_col = data.get('y_col')
        mode = data.get('mode', 'current_density')
        
        print(f"Looking for point: source={source}, xrf={xrf_composition}, x_col={x_col}, y_col={y_col}, mode={mode}")
        
        point_data = []
        
        # Get current data
        current_df = load_original_data()
        
        if mode == 'current_density':
            # Get data for this specific point across all current densities
            current_density_options = [50, 100, 150, 200, 300]
            
            for cd in current_density_options:
                # Filter data at this current density
                filtered_df = filter_df_by_current_density(current_df, cd)
                filtered_df = calculate_pca_components(filtered_df)
                
                # Find the specific point
                composition_col = 'xrf composition' if 'xrf composition' in filtered_df.columns else 'target composition'
                point_row = filtered_df[(filtered_df['source'] == source) & 
                                      (filtered_df[composition_col] == xrf_composition)]
                
                if not point_row.empty:
                    x_val = point_row[x_col].iloc[0] if x_col in point_row.columns else None
                    y_val = point_row[y_col].iloc[0] if y_col in point_row.columns else None
                    
                    point_data.append({
                        'current_density': cd,
                        'x_value': x_val,
                        'y_value': y_val
                    })
                    
                    print(f"Found data at {cd} mA/cm²: x={x_val}, y={y_val}")
                else:
                    print(f"No data found at {cd} mA/cm² for this point")
        else:  # voltage mode
            # Get data for this specific point across all voltages
            voltage_options = [2.0, 2.5, 3.0, 3.5, 4.0]
            
            for voltage in voltage_options:
                # Generate dataframe at this voltage
                filtered_df = generate_df_at_voltage(current_df, target_voltage=voltage)
                filtered_df = calculate_pca_components(filtered_df)
                
                # Find the specific point
                composition_col = 'xrf composition' if 'xrf composition' in filtered_df.columns else 'target composition'
                point_row = filtered_df[(filtered_df['source'] == source) & 
                                      (filtered_df[composition_col] == xrf_composition)]
                
                if not point_row.empty:
                    x_val = point_row[x_col].iloc[0] if x_col in point_row.columns else None
                    y_val = point_row[y_col].iloc[0] if y_col in point_row.columns else None
                    
                    point_data.append({
                        'voltage': voltage,
                        'x_value': x_val,
                        'y_value': y_val
                    })
                    
                    print(f"Found data at {voltage}V: x={x_val}, y={y_val}")
                else:
                    print(f"No data found at {voltage}V for this point")
        
        print(f"Total points found: {len(point_data)}")
        return jsonify({'success': True, 'data': point_data})
        
    except Exception as e:
        print(f"Error in get_point_data: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

# Create a route to get XRD data for a specific sample ID
@co2_plot_bp.route('/get_xrd_data', methods=['POST'])
def get_xrd_data():
    try:
        data = request.get_json()
        sample_id = data.get('sample_id')
        data_type = data.get('data_type', 'raw')  # Default to raw
        
        if not sample_id:
            return jsonify({'success': False, 'error': 'Sample ID is required'}), 400
        
        print(f"DEBUG: Requesting XRD data for sample: {sample_id}, type: {data_type}")
        print(f"DEBUG: Flask working directory: {os.getcwd()}")
        
        # Load XRD data with specified type
        xrd_data = load_xrd_data(sample_id, data_type)
        
        if xrd_data is None:
            return jsonify({
                'success': False, 
                'error': f'No XRD data found/collected for sample: {sample_id}',
                'sample_id': sample_id,
                'data_type': data_type
            }), 404
        
        # Convert to format suitable for Plotly
        x_values = [point[0] for point in xrd_data]
        y_values = [point[1] for point in xrd_data]
        
        return jsonify({
            'success': True,
            'sample_id': sample_id,
            'data_type': data_type,
            'data': {
                'x': x_values,
                'y': y_values
            },
            'data_points': len(xrd_data)
        })
        
    except Exception as e:
        print(f"Error in get_xrd_data: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500
