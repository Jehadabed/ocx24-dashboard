import pandas as pd
import matplotlib.pyplot as plt
from flask import Blueprint, render_template_string, request, jsonify, Response
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
from scipy.stats import linregress
import json

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
    
    # Create the HTML template for CO2R plot
    html_template = f'''
    <!DOCTYPE html>
    <html>
    <head>
        <title>OCx24 Dataset: CO2R</title>
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
            
            .plot-content {{
                padding: 24px;
                min-height: 600px;
                display: flex;
                align-items: center;
                justify-content: center;
                background: #ffffff;
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
            
            .export-btn:hover {
                background: #34a853;
                color: #ffffff;
                box-shadow: 0 2px 8px rgba(52, 168, 83, 0.15);
                transform: translateY(-1px);
            }
            
            .checkbox-container {
                display: flex;
                align-items: center;
                gap: 12px;
                background: #ffffff;
                padding: 16px 20px;
                border-radius: 8px;
                border: 1px solid #e8eaed;
                box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
            }
            
            input[type="checkbox"] {
                width: 18px;
                height: 18px;
                accent-color: #4285f4;
                cursor: pointer;
            }
            
            .checkbox-container label {
                margin: 0;
                color: #5f6368;
                font-weight: 400;
            }
        </style>
    </head>
    <body>
        <a href="/" class="back-link">← Back to Dashboard</a>
        
        <div class="container">
            <h1><strong>OCx24 Dataset:</strong> CO2R Performance Data Visualization</h1>
            
            <div class="controls">
                <div class="control-group">
                    <label for="xAxis">X-Axis</label>
                    <select id="xAxis" onchange="updatePlot()">
                        {element_options}
                    </select>
                </div>
                
                <div class="control-group">
                    <label for="yAxis">Y-Axis</label>
                    <select id="yAxis" onchange="updatePlot()">
                        {fe_options}
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
            
            <div class="plot-content">
                <div id="plot"></div>
            </div>
        </div>
        
        <script>
            // Store data globally
            let currentData = null;
            
            // Initialize with data
            const initialData = {json.dumps(df_with_pca.to_dict('records'))};
            currentData = initialData;
            
            console.log('Loaded CO2R data:', currentData.length, 'rows');
            console.log('Available columns:', Object.keys(currentData[0] || {{}}));
            
            // Set default axes
            const xAxisSelect = document.getElementById('xAxis');
            if (xAxisSelect && xAxisSelect.options.length > 0) {{
                xAxisSelect.value = xAxisSelect.options[0].value;
            }}
            
            function formatColumnName(columnName) {{
                const formatMap = {{
                    'voltage_mean': 'Voltage',
                    'voltage': 'Voltage',
                    'fe_co_mean': 'FE CO (%)',
                    'fe_h2_mean': 'FE H₂ (%)',
                    'fe_ch4_mean': 'FE CH₄ (%)',
                    'fe_c2h4_mean': 'FE C₂H₄ (%)',
                    'PCA1': 'PCA1',
                    'PCA2': 'PCA2'
                }};
                return formatMap[columnName] || columnName;
            }}
            
            function updatePlot() {{
                const xCol = document.getElementById('xAxis').value;
                const yCol = document.getElementById('yAxis').value;
                createPlot(xCol, yCol, currentData);
            }}
            
            function createPlot(xCol, yCol, data) {{
                console.log('Creating CO2R plot with:', xCol, 'vs', yCol);
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
                
                // Create traces by source
                const uoftData = data.filter(row => row['source'] === 'uoft');
                const vspData = data.filter(row => row['source'] === 'vsp');
                
                const traces = [];
                
                // Helper: build a clean XY array filtering invalid numbers
                function buildXY(rows, source) {
                    const x = [];
                    const y = [];
                    const text = [];
                    const errorArray = [];
                    
                    rows.forEach(row => {
                        const xv = Number(row[xCol]);
                        const yv = Number(row[yCol]);
                        if (Number.isFinite(xv) && Number.isFinite(yv)) {
                            x.push(xv);
                            y.push(yv);
                            
                            // For error bars, check if corresponding _std column exists
                            let errorValue = 0;
                            if (yCol.includes('_mean')) {
                                const stdCol = yCol.replace('_mean', '_std');
                                if (row[stdCol] !== undefined) {
                                    errorValue = Number(row[stdCol]) || 0;
                                }
                            }
                            errorArray.push(errorValue);
                            
                            text.push(
                                `Sample: ${row['sample id'] || 'N/A'}<br>` +
                                `Source: ${row['source']}<br>` +
                                `Batch: ${row['batch number'] || 'N/A'}<br>` +
                                `Formula: ${row['xrf composition'] || row['target composition'] || 'N/A'}<br>` +
                                `Current Density: ${row['current density']} mA/cm²<br>` +
                                `X: ${xv.toFixed(3)}<br>` +
                                `Y: ${yv.toFixed(3)}${yCol.startsWith('fe_') ? ' %' : ''}${errorValue > 0 ? ' ± ' + errorValue.toFixed(3) + (yCol.startsWith('fe_') ? ' %' : '') : ''}`
                            );
                        }
                    });
                    return { x, y, text, errors: errorArray };
                }
                
                // UofT trace (circles)
                if (uoftData.length > 0) {
                    const d = buildXY(uoftData);
                    const trace = {
                        x: d.x,
                        y: d.y,
                        mode: 'markers',
                        type: 'scatter',
                        marker: {
                            size: 12,
                            color: '#4285f4',
                            opacity: 0.9,
                            symbol: 'circle',
                            line: { width: 1.5, color: 'rgba(0,0,0,0.25)' }
                        },
                        name: 'UofT (chemical reduction)',
                        text: d.text,
                        hovertemplate: '%{text}<extra></extra>'
                    };
                    
                    // Add error bars if enabled and available
                    if (document.getElementById('errorBars').checked && d.errors.some(e => e > 0)) {
                        trace.error_y = {
                            type: 'data',
                            array: d.errors,
                            color: '#4285f4',
                            thickness: 1.5,
                            width: 3
                        };
                    }
                    
                    traces.push(trace);
                }
                
                // VSP trace (diamonds)
                if (vspData.length > 0) {
                    const d = buildXY(vspData);
                    const trace = {
                        x: d.x,
                        y: d.y,
                        mode: 'markers',
                        type: 'scatter',
                        marker: {
                            size: 12,
                            color: '#ea4335',
                            opacity: 0.9,
                            symbol: 'diamond',
                            line: { width: 1.5, color: 'rgba(0,0,0,0.25)' }
                        },
                        name: 'VSP (spark ablation)',
                        text: d.text,
                        hovertemplate: '%{text}<extra></extra>'
                    };
                    
                    // Add error bars if enabled and available
                    if (document.getElementById('errorBars').checked && d.errors.some(e => e > 0)) {
                        trace.error_y = {
                            type: 'data',
                            array: d.errors,
                            color: '#ea4335',
                            thickness: 1.5,
                            width: 3
                        };
                    }
                    
                    traces.push(trace);
                }
                
                const layout = {
                    title: {
                        text: `${formatColumnName(xCol)} vs ${formatColumnName(yCol)}`,
                        font: {size: 18},
                        x: 0.5
                    },
                    xaxis: {
                        title: formatColumnName(xCol),
                        showgrid: true,
                        gridcolor: '#e8e8e8'
                    },
                    yaxis: {
                        title: formatColumnName(yCol),
                        showgrid: true,
                        gridcolor: '#e8e8e8'
                    },
                    hovermode: 'closest',
                    template: 'plotly_white',
                    height: 700,
                    width: 1000,
                    margin: {l: 80, r: 200, t: 80, b: 60},
                    showlegend: true,
                    legend: {
                        x: 1.02,
                        y: 1,
                        bgcolor: 'rgba(255,255,255,0.9)'
                    }
                };
                
                Plotly.newPlot('plot', traces, layout, {{responsive: true}});
                console.log('CO2R plot created successfully');
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
                    a.download = 'CO2R_data.csv';
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
