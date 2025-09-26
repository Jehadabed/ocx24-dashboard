import pandas as pd
import matplotlib.pyplot as plt
from flask import Flask, render_template_string, request, jsonify
import webbrowser
import threading
import time
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

# Configure Plotly to open plots in browser
import plotly.graph_objects as go
import plotly.offline as pyo

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
    
    Args:
        df: DataFrame with elemental compositions in atomic fraction
        element_columns: List of element column names
    
    Returns:
        DataFrame with compositions converted to weight fraction (0-1 scale)
    """
    df_converted = df.copy()
    
    for col in element_columns:
        if col in df_converted.columns and col in ATOMIC_WEIGHTS:
            # Convert at. fraction to wt. fraction: wt. fraction = (at. fraction * atomic_weight) / sum(at. fraction * atomic_weight)
            df_converted[col] = df_converted[col] * ATOMIC_WEIGHTS[col]
    
    # Normalize to get weight fractions (0-1 scale)
    for idx, row in df_converted.iterrows():
        total_weight = sum(row[col] for col in element_columns if col in df_converted.columns and col in ATOMIC_WEIGHTS)
        if total_weight > 0:
            for col in element_columns:
                if col in df_converted.columns and col in ATOMIC_WEIGHTS:
                    df_converted.at[idx, col] = row[col] / total_weight
    
    return df_converted

# Global variable to store the current data from dashboard
df_original = None

def load_original_data():
    """Load the original data from CSV file or current data from dashboard"""
    global df_original
    # Always check for new data (don't cache)
    df_original = None
    if df_original is None:
        # First try to load current data from dashboard
        try:
            import json
            import os
            current_data_file = "Data/current_data_her.json"
            if os.path.exists(current_data_file):
                with open(current_data_file, 'r') as f:
                    saved_data = json.load(f)
                
                # Handle both old format (just data) and new format (data + columns)
                if isinstance(saved_data, dict) and 'data' in saved_data and 'columns' in saved_data:
                    # New format with explicit column order
                    current_data = saved_data['data']
                    column_order = saved_data['columns']
                    df_original = pd.DataFrame(current_data, columns=column_order)
                    print(f"Loaded current data from dashboard: {len(df_original)} rows, columns: {list(df_original.columns)}")
                elif isinstance(saved_data, list):
                    # Old format - just data array
                    current_data = saved_data
                    if current_data and len(current_data) > 0:
                        column_order = list(current_data[0].keys())
                        df_original = pd.DataFrame(current_data, columns=column_order)
                    else:
                        df_original = pd.DataFrame(current_data)
                    print(f"Loaded current data from dashboard (old format): {len(df_original)} rows, columns: {list(df_original.columns)}")
                else:
                    df_original = pd.DataFrame(saved_data)
                    print(f"Loaded current data from dashboard: {len(df_original)} rows")
                
                # Filter for HER reaction if reaction column exists
                if 'reaction' in df_original.columns:
                    df_original = df_original[df_original['reaction'] == 'HER'].copy()
                    # Remove the reaction column
                    df_original = df_original.drop('reaction', axis=1)
                    print(f"Filtered for HER reaction: {len(df_original)} rows")
                
                return df_original
        except Exception as e:
            print(f"Could not load current data: {e}")
        
        # Fallback to original CSV data
        df_original = pd.read_csv("Data/DashboardData.csv")
        
        # Filter for HER reaction if reaction column exists
        if 'reaction' in df_original.columns:
            df_original = df_original[df_original['reaction'] == 'HER'].copy()
            # Remove the reaction column
            df_original = df_original.drop('reaction', axis=1)
    
    return df_original





def calculate_pca_components(df):
    """
    Calculate PCA components from elemental composition data.
    Returns dataframe with PCA1 and PCA2 columns added.
    """
    # Check if dataframe is empty or has insufficient data
    if df.empty or len(df) < 2:
        print("Warning: Insufficient data for PCA calculation")
        df['PCA1'] = np.nan
        df['PCA2'] = np.nan
        return df
    
    # Get only elemental composition columns (exclude metadata columns, source, composition columns, fe_ columns, etc.)
    # Handle both voltage_mean and voltage column names
    voltage_cols_to_exclude = ['voltage_mean', 'voltage_std', 'voltage']
    composition_col = 'xrf composition' if 'xrf composition' in df.columns else 'target composition'
    element_cols = [col for col in df.columns if col not in ['sample id', 'source', 'batch number', 'batch date', 'current density', composition_col, 'target composition', 'xrf composition', 'rep'] + voltage_cols_to_exclude and not col.startswith('fe_') and not col.endswith('std')]
    
    # Filter out non-numeric columns and ensure we have enough data
    numeric_element_cols = []
    for col in element_cols:
        try:
            if pd.to_numeric(df[col], errors='coerce').notna().sum() >= 2:
                numeric_element_cols.append(col)
        except:
            continue
    
    if len(numeric_element_cols) < 2:
        print("Warning: Insufficient numeric columns for PCA calculation")
        df['PCA1'] = np.nan
        df['PCA2'] = np.nan
        return df
    
    # Prepare data for PCA (only numeric columns)
    pca_data = df[numeric_element_cols].copy()
    
    # Handle any non-numeric data
    for col in pca_data.columns:
        pca_data[col] = pd.to_numeric(pca_data[col], errors='coerce')
    
    # Fill NaN values with 0 (or you could use mean/median)
    pca_data = pca_data.fillna(0)
    
    # Check if we have any non-zero data
    if pca_data.sum().sum() == 0:
        print("Warning: All PCA data is zero")
        df['PCA1'] = 0
        df['PCA2'] = 0
        return df
    
    try:
        # Standardize the data
        scaler = StandardScaler()
        pca_data_scaled = scaler.fit_transform(pca_data)
        
        # Apply PCA
        pca = PCA(n_components=2)
        pca_components = pca.fit_transform(pca_data_scaled)
        
        # Add PCA components to the dataframe
        df['PCA1'] = pca_components[:, 0]
        df['PCA2'] = pca_components[:, 1]
        
        
        
    except Exception as e:
        print(f"Warning: PCA calculation failed: {e}")
        df['PCA1'] = np.nan
        df['PCA2'] = np.nan
    
    return df


def format_column_name(column_name):
    """
    Format column names to be more readable.
    """
    if column_name in ['voltage_mean', 'voltage']:
        return 'Voltage'
    elif column_name.startswith('fe_'):
        # Convert fe_h2_mean to "Faradaic Efficiency H2"
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

def find_pd_mean_value(df, target_column, source):
    """
    Find the mean value of a target column for a specific source where Pd composition is 1.0
    """
    # Filter data for the specific source and Pd = 1.0
    filtered_data = df[(df['source'] == source) & (df['Pd'] == 1.0)]
    
    if filtered_data.empty:
        return None
    
    # Get the mean value of the target column
    mean_value = filtered_data[target_column].mean()
    return mean_value

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


def initiate_interactive_plot():
    """
    Creates a Flask web application with interactive dropdown widgets for dynamic plotting.
    """
    # Create Flask app
    app = Flask(__name__)
    
    # Load or get current data
    current_df = load_original_data()
    
    # Check if dataframe is empty
    if current_df.empty:
        print("ERROR: No data loaded! Cannot create interactive plot.")
        return
    
    # Calculate PCA components for the current dataframe
    df_with_pca = calculate_pca_components(current_df)
    
    
    # Identify element columns for x-axis (chemical elements + PCA1)
    # Handle both voltage_mean and voltage column names
    voltage_cols_to_exclude = ['voltage_mean', 'voltage_std', 'voltage']
    composition_col = 'xrf composition' if 'xrf composition' in df_with_pca.columns else 'target composition'
    element_cols = [col for col in df_with_pca.columns if col not in ['sample id', 'source', 'batch number', 'batch date', 'current density', composition_col, 'target composition', 'xrf composition', 'PCA1', 'PCA2', 'rep'] + voltage_cols_to_exclude and not col.startswith('fe_') and not col.endswith('std')]
    

    
    # Check if we have any element columns
    if len(element_cols) == 0:
        print("ERROR: No element columns found for x-axis! Cannot create interactive plot.")
        return
    
    # Add PCA1 as the first option
    if 'PCA1' in df_with_pca.columns:
        element_cols.insert(0, 'PCA1')
    if 'Cu' in df_with_pca.columns and 'Cu' not in element_cols:
        element_cols.insert(0, 'Cu')
    
    # Determine which voltage column to use
    if 'voltage_mean' in df_with_pca.columns:
        y_axis_column = 'voltage_mean'
    elif 'voltage' in df_with_pca.columns:
        y_axis_column = 'voltage'
    else:
        print("ERROR: No voltage column found! Need either 'voltage_mean' or 'voltage' column.")
        print("Available columns:", list(df_with_pca.columns))
        return
    
    # Check if required columns exist
    required_columns = ['source', y_axis_column, 'Pd']
    missing_columns = [col for col in required_columns if col not in df_with_pca.columns]
    if missing_columns:
        print(f"ERROR: Missing required columns: {missing_columns}")
        print("Available columns:", list(df_with_pca.columns))
        return
    
    print("All required columns found. Proceeding to create HTML template...")
    
    # Check if there's actual data
    if len(df_with_pca) == 0:
        print("ERROR: Dataframe is empty! No data to plot.")
        return
    
    print(f"Dataframe has {len(df_with_pca)} rows of data.")
    
    # Check if we have any data with the required values
    uoft_data = df_with_pca[df_with_pca['source'] == 'uoft']
    vsp_data = df_with_pca[df_with_pca['source'] == 'vsp']
    
    print(f"UOFT data: {len(uoft_data)} rows")
    print(f"VSP data: {len(vsp_data)} rows")
    
    if len(uoft_data) == 0 and len(vsp_data) == 0:
        print("ERROR: No data found for either UOFT or VSP sources!")
        return
    
    # Check if Pd column has valid data
    pd_data = df_with_pca[df_with_pca['Pd'].notna()]
    print(f"Rows with valid Pd data: {len(pd_data)}")
    
    if len(pd_data) == 0:
        print("ERROR: No valid Pd data found!")
        return
    
    # Check if voltage column has valid data
    voltage_data = df_with_pca[df_with_pca[y_axis_column].notna()]
    print(f"Rows with valid voltage data: {len(voltage_data)}")
    
    if len(voltage_data) == 0:
        print("ERROR: No valid voltage data found!")
        return
    
    # Create HTML template with dropdowns and Plotly
    html_template = f'''
    <!DOCTYPE html>
    <html>
    <head>
        <title>OCx24 Dataset: HER</title>
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
            
            #plot {{
                background: #ffffff;
                border-radius: 0;
                box-shadow: none;
                padding: 0;
                border: none;
                transition: none;
            }}
            
            #plot:hover {{
                box-shadow: none;
            }}

            .loading {{
                text-align: center;
                color: #5f6368;
                font-style: normal;
                margin: 24px 0;
                font-size: 14px;
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
            
            .plot-content {{
                padding: 24px;
                min-height: 400px;
                display: flex;
                align-items: center;
                justify-content: center;
                background: #ffffff;
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
                .plot-container {{
                    padding: 24px;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1> <strong> OCx24 Dataset:</strong> HER Performance Data Visualization</h1>
            
            <div class="controls">
                <div class="control-group">
                    <label for="xAxis">X-Axis</label>
                    <select id="xAxis" onchange="updatePlot()">
                        {chr(10).join([f'                        <option value="{col}">{format_column_name(col)}</option>' for col in element_cols])}
                    </select>
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
                <strong>Default Color Coding:</strong><br>
                • <span style="color: #ef4444;">Red points</span>: Performance below Pd (UofT) threshold<br>
                • <span style="color: #3b82f6;">Blue points</span>: Performance below Pd (VSP) threshold<br>
                • <span style="color: #6b7280;">Black points</span>: Performance above both thresholds<br>
                <em>Note: The specific threshold values depend on the selected y-axis metric and are calculated as the mean performance for each source.</em>
                
                <br><br><strong>Note on Error Bars:</strong><br>
                Error bars are shown only when averaging across identical XRF compositions in this analysis.<br>
                • <strong>UofT (Chemical Reduction):</strong> Samples were first made as powders, XRF-measured once, then used to prepare 3 GDEs (Gas Diffusion Electrodes) for electrochemical testing. Since all GDEs came from the same powder vial (same composition), they were grouped together to calculate mean and standard deviation.<br>
                • <strong>VSP (Spark Ablation):</strong> Samples were deposited directly as 3 separate GDEs. Each had slightly different XRF compositions, so they could not be grouped. Their results are shown individually, without averaged error bars.
                
                <br><br><strong>XRD Analysis:</strong><br>
                Click on any point in the main plot to view the corresponding XRD pattern in the XRD Analysis window.<br>
                • XRD data is loaded from <code>/Data/XRD/raw/</code> directory<br>
                • Files are named using the sample ID (e.g., <code>sample_001.xy</code>)<br>
                • The plot shows 2θ (degrees) vs Intensity (counts)<br>
                • If no XRD data is found for a sample, an error message will be displayed
            </div>
        </div>
        
        <script>
            // Global variables
            let currentData = null;
            let originalData = null;
            let currentUnitType = 'atomic';
            
            // Set default selections
            document.getElementById('xAxis').value = '{element_cols[0] if element_cols else "Cu"}';
            
            // Function to get column units based on unit type
            function getColumnUnits(columnName, unitType = 'atomic') {{
                if (columnName === 'PCA1' || columnName === 'PCA2') {{
                    return '';
                }} else if (['Ag', 'Au', 'Cd', 'Cu', 'Ga', 'Hg', 'In', 'Ni', 'Pd', 'Pt', 'Rh', 'Sn', 'Tl', 'Zn'].includes(columnName)) {{
                    return unitType === 'weight' ? ' (wt. fraction)' : ' (at. fraction)';
                }} else {{
                    return '';
                }}
            }}
            
            async function updatePlot() {{
                const loadingDiv = document.getElementById('loading');
                loadingDiv.style.display = 'block';
                
                try {{
                    // Get current selections
                    const xCol = document.getElementById('xAxis').value;
                    const yCol = '{y_axis_column}'; // Dynamic voltage column
                    
                    // Get selected unit type
                    const selectedUnit = document.querySelector('input[name="unitType"]:checked').value;
                    currentUnitType = selectedUnit;
                    
                    console.log('Updating plot with xCol:', xCol, 'yCol:', yCol, 'unitType:', selectedUnit);
                    
                    // Fetch data
                    const response = await fetch('/update_data', {{
                        method: 'POST',
                        headers: {{
                            'Content-Type': 'application/json',
                        }},
                        body: JSON.stringify({{
                            xAxis: xCol,
                            unitType: selectedUnit
                        }})
                    }});
                    
                    if (!response.ok) {{
                        throw new Error('Network response was not ok');
                    }}
                    
                    const result = await response.json();
                    currentData = result.data;
                    originalData = result.originalData;
                    
                    console.log('Received data:', result);
                    console.log('Data length:', currentData.length);
                    console.log('First few rows:', currentData.slice(0, 3));
                    
                    // Update the plot with new data
                    createPlot(xCol, yCol, currentData, originalData);
                    
                }} catch (error) {{
                    console.error('Error updating plot:', error);
                    loadingDiv.textContent = 'Error updating plot. Please try again.';
                }} finally {{
                    loadingDiv.style.display = 'none';
                }}
            }}
            
            function createPlot(xCol, yCol, data, originalDataForCalc = null) {{
                console.log('Creating plot with:', xCol, yCol, 'data length:', data.length);
                
                // Use original data for calculations if available, otherwise use current data
                const calcData = originalDataForCalc || data;
                
                // Calculate Pd means for the selected y column
                let pdMeanUoft = null;
                let pdMeanVsp = null;
                
                // Find the y-axis value where Pd=1.0 for each source (use original data for calculations)
                for (let row of calcData) {{
                    if (Math.abs(row['Pd'] - 1.0) < 0.001 && row['source'] === 'uoft') {{
                        pdMeanUoft = row[yCol];
                    }}
                    if (Math.abs(row['Pd'] - 1.0) < 0.001 && row['source'] === 'vsp') {{
                        pdMeanVsp = row[yCol];
                    }}
                }}
                
                console.log('Pd means - UofT:', pdMeanUoft, 'VSP:', pdMeanVsp);
                
                // Create separate traces for UOFT and VSP points
                const uoftData = data.filter(row => row['source'] === 'uoft');
                const vspData = data.filter(row => row['source'] === 'vsp');
                
                // Get corresponding calc data for color mapping
                const uoftCalcData = calcData.filter(row => row['source'] === 'uoft');
                const vspCalcData = calcData.filter(row => row['source'] === 'vsp');
                
                console.log('Filtered data - UOFT:', uoftData.length, 'VSP:', vspData.length);
                console.log('Sample UOFT data:', uoftData.slice(0, 2));
                console.log('Sample VSP data:', vspData.slice(0, 2));
                
                const traces = [];
                
                // UOFT points (circles) - separate traces by color for proper error bar colors
                if (uoftData.length > 0) {{
                    // Group UOFT data by color
                    const uoftByColor = {{}};
                    
                    uoftData.forEach((row, index) => {{
                        const calcRow = uoftCalcData[index];
                        const yValue = calcRow ? calcRow[yCol] : row[yCol];
                        let color = '#6b7280';
                        
                        if (pdMeanUoft !== null && pdMeanVsp !== null) {{
                            if (pdMeanVsp > pdMeanUoft) {{
                                // VSP threshold is higher (worse performance)
                                if (yValue > pdMeanVsp) color = '#6b7280';  // Above both thresholds
                                else if (yValue > pdMeanUoft) color = '#3b82f6';  // Between thresholds, color of larger line
                                else color = '#ef4444';  // Below UofT threshold
                            }} else if (pdMeanUoft > pdMeanVsp) {{
                                // UofT threshold is higher (worse performance)
                                if (yValue > pdMeanUoft) color = '#6b7280';  // Above both thresholds
                                else if (yValue > pdMeanVsp) color = '#ef4444';  // Between thresholds, color of larger line
                                else color = '#3b82f6';  // Below VSP threshold
                            }} else {{
                                // Both thresholds are the same
                                if (yValue > pdMeanUoft) color = '#6b7280';  // Above both thresholds
                                else color = '#ef4444';  // Below both thresholds (choose red)
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
                        const trace = {{
                            x: colorData.map(row => row[xCol]),
                            y: colorData.map(row => row[yCol]),
                            mode: 'markers',
                            type: 'scatter',
                            marker: {{
                                size: 12,
                                color: color,
                                line: {{ width: 1.5, color: 'rgba(0,0,0,0.3)' }},
                                symbol: 'circle',
                                opacity: 0.9
                            }},
                            text: colorData.map(row => `Sample ID: ${{row['sample id'] || 'N/A'}}<br>Source: ${{row['source']}}<br>Batch: ${{row['batch number'] || 'N/A'}} (${{row['batch date'] || 'N/A'}})<br>Chemical Formula: ${{row['xrf composition'] || row['target composition']}}<br>Current Density: ${{row['current density']}} mA/cm²<br>X: ${{row[xCol].toFixed(3)}}${{getColumnUnits(xCol, currentUnitType)}}<br>Y: ${{row[yCol].toFixed(3)}} V`),
                            hoverinfo: 'text',
                            showlegend: true,
                            name: 'UofT (chemical reduction)',
                            customdata: colorData.map(row => row['sample id']),
                            source: colorData.map(row => row['source']),
                            'xrf composition': colorData.map(row => row['xrf composition'] || row['target composition'])
                        }};
                        
                        // Add error bars for voltage columns (only if checkbox is checked)
                        if (yCol === 'voltage_mean') {{
                            // Only add error bars if checkbox is checked
                            if (document.getElementById('errorBars').checked) {{
                                trace.error_y = {{
                                    type: 'data',
                                    array: colorData.map(row => row['voltage_std'] || 0),
                                    visible: true,
                                    color: color,
                                    thickness: 1.5,
                                    width: 2
                                }};
                            }}
                        }} else if (yCol === 'voltage') {{
                            // For simple voltage column, no error bars (no _std column)
                            // Error bars not available for simple voltage column
                        }}
                        
                        traces.push(trace);
                    }});
                }}
                
                // VSP points (diamonds) - separate traces by color for proper error bar colors
                if (vspData.length > 0) {{
                    // Group VSP data by color
                    const vspByColor = {{}};
                    
                    vspData.forEach((row, index) => {{
                        const calcRow = vspCalcData[index];
                        const yValue = calcRow ? calcRow[yCol] : row[yCol];
                        let color = '#6b7280';
                        
                        if (pdMeanUoft !== null && pdMeanVsp !== null) {{
                            if (pdMeanVsp > pdMeanUoft) {{
                                // VSP threshold is higher (worse performance)
                                if (yValue > pdMeanVsp) color = '#6b7280';  // Above both thresholds
                                else if (yValue > pdMeanUoft) color = '#3b82f6';  // Between thresholds, color of larger line
                                else color = '#ef4444';  // Below UofT threshold
                            }} else if (pdMeanUoft > pdMeanVsp) {{
                                // UofT threshold is higher (worse performance)
                                if (yValue > pdMeanUoft) color = '#6b7280';  // Above both thresholds
                                else if (yValue > pdMeanVsp) color = '#ef4444';  // Between thresholds, color of larger line
                                else color = '#3b82f6';  // Below VSP threshold
                            }} else {{
                                // Both thresholds are the same
                                if (yValue > pdMeanUoft) color = '#6b7280';  // Above both thresholds
                                else color = '#ef4444';  // Below both thresholds (choose red)
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
                        const trace = {{
                            x: colorData.map(row => row[xCol]),
                            y: colorData.map(row => row[yCol]),
                            mode: 'markers',
                            type: 'scatter',
                            marker: {{
                                size: 12,
                                color: color,
                                line: {{ width: 1.5, color: 'rgba(0,0,0,0.3)' }},
                                symbol: 'diamond',
                                opacity: 0.9
                            }},
                            text: colorData.map(row => `Sample ID: ${{row['sample id'] || 'N/A'}}<br>Source: ${{row['source']}}<br>Batch: ${{row['batch number'] || 'N/A'}} (${{row['batch date'] || 'N/A'}})<br>Chemical Formula: ${{row['xrf composition'] || row['target composition']}}<br>Current Density: ${{row['current density']}} mA/cm²<br>X: ${{row[xCol].toFixed(3)}}${{getColumnUnits(xCol, currentUnitType)}}<br>Y: ${{row[yCol].toFixed(3)}} V`),
                            hoverinfo: 'text',
                            showlegend: true,
                            name: 'VSP (spark ablation)',
                            customdata: colorData.map(row => row['sample id']),
                            source: colorData.map(row => row['source']),
                            'xrf composition': colorData.map(row => row['xrf composition'] || row['target composition'])
                        }};
                        
                        // Add error bars for voltage columns (only if checkbox is checked)
                        if (yCol === 'voltage_mean' && colorData.length > 0) {{
                            // Only add error bars if checkbox is checked
                            if (document.getElementById('errorBars').checked) {{
                                trace.error_y = {{
                                    type: 'data',
                                    array: colorData.map(row => row['voltage_std'] || 0),
                                    visible: true,
                                    color: color,
                                    thickness: 1.5,
                                    width: 2
                                }};
                            }}
                        }} else if (yCol === 'voltage') {{
                            // For simple voltage column, no error bars (no _std column)
                            // Error bars not available for simple voltage column
                        }}
                        
                        traces.push(trace);
                    }});
                }}
                
                // Format column names for display
                const xColFormatted = formatColumnName(xCol);
                const yColFormatted = 'Voltage';
                
                // Get units based on current unit type
                const xAxisUnit = getColumnUnits(xCol, currentUnitType);
                const yAxisUnit = ' (V)'; // Fixed to voltage
                
                const layout = {{
                    title: {{
                        text: `${{xColFormatted}} vs ${{yColFormatted}}`,
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
                
                // Add reference lines with annotations if available
                const shapes = [];
                const annotations = [];
                
                if (pdMeanUoft !== null) {{
                    // Calculate x-axis range more robustly
                    const xValues = data.map(row => row[xCol]).filter(val => val !== null && !isNaN(val));
                    const xMin = Math.min(...xValues);
                    const xMax = Math.max(...xValues);
                    const xRange = xMax - xMin;
                    
                    shapes.push({{
                        type: 'line',
                        x0: xMin - xRange * 0.1,
                        x1: xMax + xRange * 0.1,
                        y0: pdMeanUoft,
                        y1: pdMeanUoft,
                        line: {{ color: '#ef4444', dash: 'dash', width: 3 }}
                    }});
                    
                    // Add annotation for UOFT line
                    annotations.push({{
                        x: xMax + xRange * 0.1,
                        y: pdMeanUoft,
                        text: `Pd (UofT)`,
                        showarrow: false,
                        xanchor: 'left',
                        yanchor: 'middle',
                        bgcolor: 'rgba(255,255,255,0.9)',
                        bordercolor: '#ef4444',
                        borderwidth: 2,
                        font: {{ color: '#ef4444', size: 14 }}
                    }});
                }}
                
                if (pdMeanVsp !== null) {{
                    // Calculate x-axis range more robustly
                    const xValues = data.map(row => row[xCol]).filter(val => val !== null && !isNaN(val));
                    const xMin = Math.min(...xValues);
                    const xMax = Math.max(...xValues);
                    const xRange = xMax - xMin;
                    
                    shapes.push({{
                        type: 'line',
                        x0: xMin - xRange * 0.1,
                        x1: xMax + xRange * 0.1,
                        y0: pdMeanVsp,
                        y1: pdMeanVsp,
                        line: {{ color: '#3b82f6', dash: 'dot', width: 3 }}
                    }});
                    
                    // Add annotation for VSP line
                    annotations.push({{
                        x: xMax + xRange * 0.1,
                        y: pdMeanVsp,
                        text: `Pd (VSP)`,
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
                
                console.log('Final traces:', traces);
                console.log('Final layout:', layout);
                console.log('Plotting to element with id "plot"');
                
                Plotly.newPlot('plot', traces, layout);
                
                // Add click event handler for XRD loading
                document.getElementById('plot').on('plotly_click', function(data) {{
                    console.log('Plot clicked:', data);
                    
                    if (data.points && data.points.length > 0) {{
                        const point = data.points[0];
                        const pointData = point.data;
                        const pointIndex = point.pointIndex;
                        
                        // Get the sample ID from the clicked point
                        const sampleId = pointData.customdata ? pointData.customdata[pointIndex] : null;
                        
                        if (sampleId) {{
                            console.log('Sample ID:', sampleId);
                            
                            // Store clicked point data for XRD legend
                            clickedPointData = {{
                                source: pointData.source ? pointData.source[pointIndex] : 'Unknown',
                                'xrf composition': pointData['xrf composition'] ? pointData['xrf composition'][pointIndex] : 'Unknown'
                            }};
                            
                            // Load XRD data for the clicked sample
                            loadXrdPlot(sampleId);
                        }} else {{
                            console.log('No sample ID found for clicked point');
                        }}
                    }}
                }});
                
                console.log('Plot created successfully');
            }}
            
            // JavaScript function to get column units
            
            // JavaScript function to format column names
            function formatColumnName(columnName) {{
                if (columnName === 'voltage_mean' || columnName === 'voltage') {{
                    return 'Voltage';
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
            
            // Function to export data as CSV
            async function exportData() {{
                try {{
                    const exportBtn = document.getElementById('exportBtn');
                    exportBtn.textContent = '⊞ Exporting...';
                    exportBtn.disabled = true;
                    
                    // Send request to export CSV
                    const response = await fetch('/export_csv', {{
                        method: 'POST',
                        headers: {{
                            'Content-Type': 'application/json',
                        }},
                        body: JSON.stringify({{}})
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
                    a.download = 'HER_data.csv';
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
            
            // XRD functionality
            let accumulatedPoints = [];
            let accumulatedXrdData = [];
            let clickedPointData = null;
            
            // Initialize XRD plot (empty)
            document.getElementById('xrdPlotContent').innerHTML = '';
            
            // Add event listeners to XRD data type toggle
            document.querySelectorAll('input[name="xrdDataType"]').forEach(radio => {{
                radio.addEventListener('change', function() {{
                    console.log('XRD data type changed to:', this.value);
                    // Reload all accumulated XRD plots with new data type
                    reloadAccumulatedXrdPlots();
                }});
            }});
            
            // Add event listeners to unit type toggle
            document.querySelectorAll('input[name="unitType"]').forEach(radio => {{
                radio.addEventListener('change', function() {{
                    console.log('Unit type changed to:', this.value);
                    currentUnitType = this.value;
                    updatePlot();
                }});
            }});
            
            // Function to load XRD plot for a specific sample ID and add to accumulation
            async function loadXrdPlot(sampleId) {{
                try {{
                    // Get selected data type from toggle
                    const dataType = document.querySelector('input[name="xrdDataType"]:checked').value;
                    
                    // Fetch XRD data for the specific sample
                    const response = await fetch('/get_xrd_data', {{
                        method: 'POST',
                        headers: {{
                            'Content-Type': 'application/json',
                        }},
                        body: JSON.stringify({{ 
                            sample_id: sampleId,
                            data_type: dataType
                        }})
                    }});
                    
                    if (!response.ok) {{
                        const errorData = await response.json();
                        throw new Error(errorData.error || 'Failed to fetch XRD data');
                    }}
                    
                    const result = await response.json();
                    
                    if (!result.success) {{
                        throw new Error(result.error || 'XRD data not found');
                    }}
                    
                    // Add to accumulation
                    addXrdToAccumulation(result.data, result.sample_id, result.data_points);
                    
                }} catch (error) {{
                    console.error('Error loading XRD data:', error);
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
            
            // Function to show accumulated XRD plots
            function showAccumulatedXrdPlots() {{
                if (accumulatedXrdData.length === 0) {{
                    document.getElementById('xrdPlotContent').innerHTML = ''; // Clear content if no data
                    return;
                }}
                
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
                    title: '',
                    xaxis: {{
                        title: '2θ (degrees)',
                        showgrid: true,
                        gridcolor: '#e0e0e0'
                    }},
                    yaxis: {{
                        title: 'Intensity',
                        showgrid: true,
                        gridcolor: '#e0e0e0',
                        zeroline: false
                    }},
                    showlegend: true,
                    legend: {{
                        x: 1.02,
                        y: 1,
                        xanchor: 'left',
                        yanchor: 'top'
                    }},
                    margin: {{ l: 60, r: 150, t: 20, b: 60 }},
                    width: 1500,
                    height: 400
                }};
                
                Plotly.newPlot('xrdPlotContent', traces, layout);
                
                console.log('Accumulated XRD plots created successfully. Total traces:', traces.length);
            }}
            
            // Function to reset XRD accumulation
            function resetXrdPlot() {{
                accumulatedXrdData = [];
                document.getElementById('xrdPlotContent').innerHTML = '';
                console.log('XRD accumulation reset');
            }}
            
            
            // Function to reload all accumulated XRD plots with current data type
            async function reloadAccumulatedXrdPlots() {{
                if (accumulatedXrdData.length === 0) {{
                    return; // Nothing to reload
                }}
                
                console.log('Reloading accumulated XRD plots with new data type');
                
                // Store current accumulated data with metadata
                const currentSamples = [...accumulatedXrdData];
                
                // Clear current accumulation
                accumulatedXrdData = [];
                
                // Reload each sample with new data type while preserving metadata
                for (const xrdItem of currentSamples) {{
                    await reloadSingleXrdPlot(xrdItem.sampleId, xrdItem.source, xrdItem.xrfComposition);
                }}
            }}
            
            // Function to reload a single XRD plot while preserving metadata
            async function reloadSingleXrdPlot(sampleId, source, xrfComposition) {{
                try {{
                    // Get selected data type from toggle
                    const dataType = document.querySelector('input[name="xrdDataType"]:checked').value;
                    
                    // Fetch XRD data for the specific sample
                    const response = await fetch('/get_xrd_data', {{
                        method: 'POST',
                        headers: {{
                            'Content-Type': 'application/json',
                        }},
                        body: JSON.stringify({{ 
                            sample_id: sampleId,
                            data_type: dataType
                        }})
                    }});
                    
                    if (!response.ok) {{
                        const errorData = await response.json();
                        throw new Error(errorData.error || 'Failed to fetch XRD data');
                    }}
                    
                    const result = await response.json();
                    
                    if (!result.success) {{
                        throw new Error(result.error || 'XRD data not found');
                    }}
                    
                    // Add to accumulation with preserved metadata
                    addXrdToAccumulationWithMetadata(result.data, sampleId, result.data_points, source, xrfComposition);
                    
                }} catch (error) {{
                    console.error('Error reloading XRD data:', error);
                    alert('Error reloading XRD data:\\n\\n' + error.message);
                }}
            }}
            
            // Function to add XRD data to accumulation with explicit metadata
            function addXrdToAccumulationWithMetadata(xrdData, sampleId, dataPoints, source, xrfComposition) {{
                // Check if this sample is already in accumulation
                const existingIndex = accumulatedXrdData.findIndex(item => item.sampleId === sampleId);
                
                if (existingIndex !== -1) {{
                    console.log('Sample already in XRD accumulation:', sampleId);
                    return; // Don't add duplicates
                }}
                
                // Add to accumulation with explicit metadata
                accumulatedXrdData.push({{
                    data: xrdData,
                    sampleId: sampleId,
                    dataPoints: dataPoints,
                    source: source,
                    xrfComposition: xrfComposition
                }});
                
                console.log('Added XRD data to accumulation with metadata. Total samples:', accumulatedXrdData.length);
                
                // Show accumulated XRD plots
                showAccumulatedXrdPlots();
            }}
            
        </script>
    </body>
    </html>
    '''
    
    # Create a route to serve the HTML
    @app.route('/')
    def index():
        return html_template
    

    
    # Create a route to update data
    @app.route('/update_data', methods=['POST'])
    def update_data():
        try:
            data = request.get_json()
            x_axis = data.get('xAxis', 'Cu')
            y_axis = data.get('yAxis', y_axis_column)
            unit_type = data.get('unitType', 'atomic')
            
            # Get current data and calculate PCA
            current_df = load_original_data()
            new_df = calculate_pca_components(current_df)
            
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
                'originalData': original_df_dict
            })
            
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500
    
    # Create a route to export data as CSV
    @app.route('/export_csv', methods=['POST'])
    def export_csv():
        try:
            # Get current data and calculate PCA
            current_df = load_original_data()
            df_with_pca = calculate_pca_components(current_df)
            
            # Use the dataframe with PCA components
            csv_data = df_with_pca.to_csv(index=False)
            
            # Create response with CSV data
            from flask import Response
            response = Response(
                csv_data,
                mimetype='text/csv',
                headers={'Content-Disposition': 'attachment; filename=HER_data.csv'}
            )
            
            return response
            
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500
    
    # Create a route to get XRD data for a specific sample ID
    @app.route('/get_xrd_data', methods=['POST'])
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
                    'error': f'No XRD data found for sample {sample_id} ({data_type})',
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
    
    # Start the Flask app in a separate thread
    def run_app():
        try:
            app.run(debug=False, use_reloader=False, port=8083, host='0.0.0.0')
        except Exception as e:
            print(f"Error starting server: {e}")
            print("Trying alternative port...")
            try:
                app.run(debug=False, use_reloader=False, port=8084, host='0.0.0.0')
            except Exception as e2:
                print(f"Error with alternative port: {e2}")
                return
    
    thread = threading.Thread(target=run_app)
    thread.daemon = True
    thread.start()
    
    # Wait a moment for the server to start
    time.sleep(2)
    
    # Server started - dashboard will handle opening the browser
    print("HER Performance Plot server started!")
    print("Dashboard will open the plot automatically.")
    print("Manual access: http://localhost:8083 or http://localhost:8084")
    
    print("The Flask server is running. Close this terminal or press Ctrl+C to stop the server when done.")
    print("This interface shows voltage vs selected x-axis with Pd-based thresholds.")
    
    # Keep the main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\\nShutting down server...")

# Call the interactive plot function
if __name__ == "__main__":
    initiate_interactive_plot()
