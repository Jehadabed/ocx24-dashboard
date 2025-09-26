import pandas as pd
import matplotlib.pyplot as plt
from flask import Blueprint, render_template_string, request, jsonify, Response
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import json

# Create Blueprint
her_plot_bp = Blueprint('her_plot', __name__, url_prefix='/her')

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

def load_original_data():
    """Load the original data from CSV file or current data from dashboard"""
    try:
        # First try to load current data from dashboard
        current_data_file = "Data/current_data_her.json"
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
            
            # Filter for HER reaction if reaction column exists
            if 'reaction' in df.columns:
                df = df[df['reaction'] == 'HER'].copy()
                df = df.drop('reaction', axis=1)
            
            print(f"DEBUG: Available columns after loading HER data: {list(df.columns)}")
            print(f"DEBUG: Data shape: {df.shape}")
            print(f"DEBUG: Voltage columns present: {[col for col in df.columns if 'voltage' in col.lower()]}")
            
            return df
    except Exception as e:
        print(f"Could not load current data: {e}")
    
    # Fallback to original CSV data
    try:
        df = pd.read_csv("Data/DashboardData.csv")
        if 'reaction' in df.columns:
            df = df[df['reaction'] == 'HER'].copy()
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
        return 'Voltage'
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

def find_pd_mean_value(df, target_column, source):
    """Find the mean value of a target column for a specific source where Pd composition is 1.0"""
    # Filter data for the specific source and Pd = 1.0
    filtered_data = df[(df['source'] == source) & (df['Pd'] == 1.0)]
    
    if filtered_data.empty:
        return None
    
    # Get the mean value of the target column
    mean_value = filtered_data[target_column].mean()
    return mean_value

def load_xrd_data(sample_id, data_type="raw"):
    """Load XRD data for a specific sample ID from Data/XRD directory."""
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

@her_plot_bp.route('/')
def her_plot_main():
    """Main HER plot page"""
    # Load and process data
    current_df = load_original_data()
    
    if current_df.empty:
        return "<h2>Error: No HER data available</h2><p>Please ensure HER data is available in the main dashboard.</p>"
    
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
    
    # Determine voltage column
    if 'voltage_mean' in df_with_pca.columns:
        y_axis_column = 'voltage_mean'
    elif 'voltage' in df_with_pca.columns:
        y_axis_column = 'voltage'
    else:
        return "<h2>Error: No voltage data available</h2><p>Required voltage column not found.</p>"
    
    # Generate element options for dropdown
    element_options = ''.join([f'<option value="{col}">{format_column_name(col)}</option>' for col in element_cols])
    
    # Create the HTML template exactly matching the original
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
            <h1><strong>OCx24 Dataset:</strong> HER Performance Data Visualization</h1>
            
            <div class="controls">
                <div class="control-group">
                    <label for="xAxis">X-Axis</label>
                    <select id="xAxis" onchange="updatePlot()">
                        {element_options}
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
            // Store data globally
            let currentData = null;
            let originalData = null;
            let currentUnitType = 'atomic';
            let currentYColumn = '{y_axis_column}'; // Dynamic voltage column
            
            // Initialize with data
            const initialData = {json.dumps(df_with_pca.to_dict('records'))};
            currentData = initialData;
            originalData = initialData;
            
            console.log('Loaded HER data:', currentData.length, 'rows');
            console.log('Available columns:', Object.keys(currentData[0] || {{}}));
            console.log('Y-axis column detected:', currentYColumn);
            
            // Detect actual voltage column from data
            if (currentData && currentData.length > 0) {{
                const firstRow = currentData[0];
                if (firstRow.hasOwnProperty('voltage_mean')) {{
                    currentYColumn = 'voltage_mean';
                }} else if (firstRow.hasOwnProperty('voltage')) {{
                    currentYColumn = 'voltage';
                }} else {{
                    console.warn('No voltage column found, using fallback:', currentYColumn);
                }}
                console.log('Final Y-axis column:', currentYColumn);
            }}
            
            // Set default x-axis
            const xAxisSelect = document.getElementById('xAxis');
            if (xAxisSelect && xAxisSelect.options.length > 0) {{
                xAxisSelect.value = xAxisSelect.options[0].value;
            }}
            
            function getColumnUnits(columnName, unitType = 'atomic') {{
                if (columnName === 'PCA1' || columnName === 'PCA2') {{
                    return '';
                }} else if (['Ag', 'Au', 'Cd', 'Cu', 'Ga', 'Hg', 'In', 'Ni', 'Pd', 'Pt', 'Rh', 'Sn', 'Tl', 'Zn'].includes(columnName)) {{
                    return unitType === 'weight' ? ' (wt. fraction)' : ' (at. fraction)';
                }} else {{
                    return '';
                }}
            }}
            
            function formatColumnName(columnName) {{
                const formatMap = {{
                    'voltage_mean': 'Voltage',
                    'voltage': 'Voltage',
                    'PCA1': 'PCA1',
                    'PCA2': 'PCA2'
                }};
                return formatMap[columnName] || columnName;
            }}
            
            async function updatePlot() {{
                const xCol = document.getElementById('xAxis').value;
                const selectedUnit = document.querySelector('input[name="unitType"]:checked').value;
                currentUnitType = selectedUnit;
                
                try {{
                    // Update data with unit conversion if needed
                    const response = await fetch('/her/update_data', {{
                        method: 'POST',
                        headers: {{'Content-Type': 'application/json'}},
                        body: JSON.stringify({{xAxis: xCol, unitType: selectedUnit}})
                    }});
                    
                    if (response.ok) {{
                        const result = await response.json();
                        currentData = result.data;
                        originalData = result.originalData;
                    }}
                }} catch (error) {{
                    console.log('Using local data due to fetch error:', error);
                }}
                
                createPlot(xCol, currentYColumn, currentData, originalData);
            }}
            
            function createPlot(xCol, yCol, data, originalDataForCalc = null) {{
                console.log('Creating plot with:', xCol, 'vs', yCol);
                console.log('Data points:', data.length);
                
                if (!data || data.length === 0) {{
                    document.getElementById('plot').innerHTML = '<div style="text-align: center; padding: 50px;"><h3>No data available for plotting</h3></div>';
                    return;
                }}
                
                // Check if required columns exist
                if (!data[0].hasOwnProperty(xCol)) {{
                    console.error('X-axis column not found:', xCol);
                    document.getElementById('plot').innerHTML = '<div style="text-align: center; padding: 50px;"><h3>X-axis column "' + xCol + '" not found in data</h3></div>';
                    return;
                }}
                if (!data[0].hasOwnProperty(yCol)) {{
                    console.error('Y-axis column not found:', yCol);
                    document.getElementById('plot').innerHTML = '<div style="text-align: center; padding: 50px;"><h3>Y-axis column "' + yCol + '" not found in data</h3></div>';
                    return;
                }}
                
                const calcData = originalDataForCalc || data;
                
                // Calculate Pd means for threshold lines
                let pdMeanUoft = null;
                let pdMeanVsp = null;
                
                for (let row of calcData) {{
                    if (row['Pd'] && Math.abs(parseFloat(row['Pd']) - 1.0) < 0.001) {{
                        if (row['source'] === 'uoft') pdMeanUoft = parseFloat(row[yCol]);
                        if (row['source'] === 'vsp') pdMeanVsp = parseFloat(row[yCol]);
                    }}
                }}
                
                console.log('Pd thresholds - UofT:', pdMeanUoft, 'VSP:', pdMeanVsp);
                
                // Create traces for each source and color combination
                const traces = [];
                const uoftData = data.filter(row => row['source'] === 'uoft');
                const vspData = data.filter(row => row['source'] === 'vsp');
                
                // Color code points based on performance thresholds
                function getPointColor(yValue, pdMeanUoft, pdMeanVsp) {{
                    if (pdMeanUoft !== null && pdMeanVsp !== null) {{
                        if (pdMeanVsp > pdMeanUoft) {{
                            if (yValue > pdMeanVsp) return '#6b7280';
                            else if (yValue > pdMeanUoft) return '#3b82f6';
                            else return '#ef4444';
                        }} else if (pdMeanUoft > pdMeanVsp) {{
                            if (yValue > pdMeanUoft) return '#6b7280';
                            else if (yValue > pdMeanVsp) return '#ef4444';
                            else return '#3b82f6';
                        }} else {{
                            return yValue > pdMeanUoft ? '#6b7280' : '#ef4444';
                        }}
                    }}
                    return '#6b7280';
                }}
                
                // Group and create traces
                const uoftByColor = {{}};
                uoftData.forEach(row => {{
                    const color = getPointColor(row[yCol], pdMeanUoft, pdMeanVsp);
                    if (!uoftByColor[color]) uoftByColor[color] = [];
                    uoftByColor[color].push(row);
                }});
                
                const vspByColor = {{}};
                vspData.forEach(row => {{
                    const color = getPointColor(row[yCol], pdMeanUoft, pdMeanVsp);
                    if (!vspByColor[color]) vspByColor[color] = [];
                    vspByColor[color].push(row);
                }});
                
                // Create traces for UofT data (circles)
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
                            line: {{width: 1.5, color: 'rgba(0,0,0,0.3)'}},
                            symbol: 'circle',
                            opacity: 0.9
                        }},
                        text: colorData.map(row => 
                            `Sample: ${{row['sample id'] || 'N/A'}}<br>` +
                            `Source: ${{row['source']}}<br>` +
                            `Batch: ${{row['batch number'] || 'N/A'}}<br>` +
                            `Formula: ${{row['xrf composition'] || row['target composition'] || 'N/A'}}<br>` +
                            `Current Density: ${{row['current density']}} mA/cm²<br>` +
                            `X: ${{row[xCol].toFixed(3)}}${{getColumnUnits(xCol, currentUnitType)}}<br>` +
                            `Y: ${{row[yCol].toFixed(3)}} V`
                        ),
                        hoverinfo: 'text',
                        name: 'UofT (chemical reduction)',
                        showlegend: Object.keys(uoftByColor).indexOf(color) === 0,
                        customdata: colorData.map(row => row['sample id']),
                        source: colorData.map(row => row['source']),
                        'xrf composition': colorData.map(row => row['xrf composition'] || row['target composition'])
                    }};
                    
                    // Add error bars if enabled and available
                    if (document.getElementById('errorBars').checked && yCol === 'voltage_mean' && colorData[0] && colorData[0]['voltage_std'] !== undefined) {{
                        trace.error_y = {{
                            type: 'data',
                            array: colorData.map(row => row['voltage_std'] || 0),
                            color: color,
                            thickness: 1.5,
                            width: 2
                        }};
                    }}
                    
                    traces.push(trace);
                }});
                
                // Create traces for VSP data (diamonds)
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
                            line: {{width: 1.5, color: 'rgba(0,0,0,0.3)'}},
                            symbol: 'diamond',
                            opacity: 0.9
                        }},
                        text: colorData.map(row => 
                            `Sample: ${{row['sample id'] || 'N/A'}}<br>` +
                            `Source: ${{row['source']}}<br>` +
                            `Batch: ${{row['batch number'] || 'N/A'}}<br>` +
                            `Formula: ${{row['xrf composition'] || row['target composition'] || 'N/A'}}<br>` +
                            `Current Density: ${{row['current density']}} mA/cm²<br>` +
                            `X: ${{row[xCol].toFixed(3)}}${{getColumnUnits(xCol, currentUnitType)}}<br>` +
                            `Y: ${{row[yCol].toFixed(3)}} V`
                        ),
                        hoverinfo: 'text',
                        name: 'VSP (spark ablation)',
                        showlegend: Object.keys(vspByColor).indexOf(color) === 0,
                        customdata: colorData.map(row => row['sample id']),
                        source: colorData.map(row => row['source']),
                        'xrf composition': colorData.map(row => row['xrf composition'] || row['target composition'])
                    }};
                    
                    if (document.getElementById('errorBars').checked && yCol === 'voltage_mean' && colorData.length > 0 && colorData[0]['voltage_std'] !== undefined) {{
                        trace.error_y = {{
                            type: 'data',
                            array: colorData.map(row => row['voltage_std'] || 0),
                            color: color,
                            thickness: 1.5,
                            width: 2
                        }};
                    }}
                    
                    traces.push(trace);
                }});
                
                const layout = {{
                    title: {{
                        text: `${{formatColumnName(xCol)}} vs Voltage`,
                        font: {{size: 18}},
                        x: 0.5
                    }},
                    xaxis: {{
                        title: formatColumnName(xCol) + getColumnUnits(xCol, currentUnitType),
                        showgrid: true,
                        gridcolor: '#e8e8e8'
                    }},
                    yaxis: {{
                        title: 'Voltage (V)',
                        showgrid: true,
                        gridcolor: '#e8e8e8'
                    }},
                    hovermode: 'closest',
                    template: 'plotly_white',
                    height: 600,
                    margin: {{l: 60, r: 30, t: 80, b: 60}},
                    showlegend: true,
                    legend: {{
                        x: 1.02,
                        y: 1,
                        bgcolor: 'rgba(255,255,255,0.9)'
                    }}
                }};
                
                // Add reference lines
                const shapes = [];
                if (pdMeanUoft !== null) {{
                    const xValues = data.map(row => row[xCol]);
                    const xMin = Math.min(...xValues);
                    const xMax = Math.max(...xValues);
                    
                    const xRange = xMax - xMin;
                    shapes.push({{
                        type: 'line',
                        x0: xMin - xRange * 0.1,
                        x1: xMax + xRange * 0.1,
                        y0: pdMeanUoft,
                        y1: pdMeanUoft,
                        line: {{color: '#ef4444', dash: 'dash', width: 3}}
                    }});
                }}
                
                if (pdMeanVsp !== null) {{
                    const xValues = data.map(row => row[xCol]);
                    const xMin = Math.min(...xValues);
                    const xMax = Math.max(...xValues);
                    
                    const xRange = xMax - xMin;
                    shapes.push({{
                        type: 'line',
                        x0: xMin - xRange * 0.1,
                        x1: xMax + xRange * 0.1,
                        y0: pdMeanVsp,
                        y1: pdMeanVsp,
                        line: {{color: '#3b82f6', dash: 'dot', width: 3}}
                    }});
                }}
                
                // Add annotations for reference lines
                const annotations = [];
                
                if (pdMeanUoft !== null) {{
                    const xValues = data.map(row => row[xCol]);
                    const xMax = Math.max(...xValues);
                    const xRange = Math.max(...xValues) - Math.min(...xValues);
                    
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
                        font: {{color: '#ef4444', size: 14}}
                    }});
                }}
                
                if (pdMeanVsp !== null) {{
                    const xValues = data.map(row => row[xCol]);
                    const xMax = Math.max(...xValues);
                    const xRange = Math.max(...xValues) - Math.min(...xValues);
                    
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
                        font: {{color: '#3b82f6', size: 14}}
                    }});
                }}
                
                if (shapes.length > 0) {{
                    layout.shapes = shapes;
                }}
                
                if (annotations.length > 0) {{
                    layout.annotations = annotations;
                }}
                
                Plotly.newPlot('plot', traces, layout, {{responsive: true}});
                
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
            
            // Function to export data as CSV
            async function exportData() {{
                try {{
                    const exportBtn = document.getElementById('exportBtn');
                    exportBtn.textContent = '⊞ Exporting...';
                    exportBtn.disabled = true;
                    
                    // Send request to export CSV
                    const response = await fetch('/her/export_csv', {{
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
            
            // Initialize plot
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
                    const response = await fetch('/her/get_xrd_data', {{
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
                    const response = await fetch('/her/get_xrd_data', {{
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
    
    return html_template

@her_plot_bp.route('/update_data', methods=['POST'])
def update_data():
    """Handle AJAX requests to update plot data with unit conversions"""
    try:
        data = request.get_json()
        x_axis = data.get('xAxis', 'Cu')
        unit_type = data.get('unitType', 'atomic')
        
        # Load fresh data
        df = load_original_data()
        
        if df.empty:
            return jsonify({'error': 'No data available'}), 400
        
        # Calculate PCA if needed
        df = calculate_pca_components(df)
        
        # Store original data for calculations (color mapping, reference lines)
        original_df = df.copy()
        
        # Apply unit conversion if weight fraction is selected (only for display)
        if unit_type == 'weight':
            element_cols = [col for col in df.columns if col in ATOMIC_WEIGHTS]
            if element_cols:
                df = convert_atomic_to_weight_fraction(df, element_cols)
        
        return jsonify({
            'success': True,
            'data': df.to_dict('records'),
            'originalData': original_df.to_dict('records')
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@her_plot_bp.route('/export_csv', methods=['POST'])
def export_csv():
    """Export HER data as CSV"""
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
            headers={'Content-Disposition': 'attachment; filename=HER_data.csv'}
        )
        
        return response
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@her_plot_bp.route('/get_xrd_data', methods=['POST'])
def get_xrd_data():
    """Get XRD data for a specific sample ID"""
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
