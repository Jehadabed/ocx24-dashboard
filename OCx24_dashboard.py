import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, render_template_string, request, jsonify
import webbrowser
import threading
import time
import os
import re
import requests
import psutil

# Import our blueprints
from her_plot_blueprint import her_plot_bp
from co2_plot_blueprint import co2_plot_bp
from xrd_plot_blueprint import xrd_plot_bp

# Global cache for uploaded data
uploaded_data_cache = None

def add_partial_current_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add partial current density columns for every column starting with 'fe_'.
    For raw data: creates 'partial_current_<name>' = df['fe_<name>'] * df['current density'].
    For averaged data where FE columns are suffixed with '_mean': creates
    'partial_current_<name>_mean' = df['fe_<name>_mean'] * df['current density'].
    """
    try:
        df_out = df.copy()
        # Determine whether FE columns are averaged (have _mean suffix)
        fe_mean_cols = [c for c in df_out.columns if c.startswith('fe_') and c.endswith('_mean')]
        fe_raw_cols = [c for c in df_out.columns if c.startswith('fe_') and not c.endswith('_mean') and not c.endswith('_std')]

        # Use current density column present in the dataframe
        current_density_col = 'current density' if 'current density' in df_out.columns else None
        if current_density_col is None:
            return df_out

        # Averaged FE columns -> create partial_current_<x>_mean
        for col in fe_mean_cols:
            species = col[len('fe_'):-len('_mean')]  # between fe_ and _mean
            new_col = f"partial_current_{species}_mean"
            try:
                df_out[new_col] = (df_out[col] * df_out[current_density_col]) / 100.0
            except Exception:
                pass

        # Raw FE columns -> create partial_current_<x>
        for col in fe_raw_cols:
            # Skip std columns (already excluded) and keep all fe_ including totals per spec
            species = col[len('fe_'):]
            new_col = f"partial_current_{species}"
            try:
                df_out[new_col] = (df_out[col] * df_out[current_density_col]) / 100.0
            except Exception:
                pass

        return df_out
    except Exception:
        return df

def add_max_partial_current_all(df: pd.DataFrame, averaged: bool = False) -> pd.DataFrame:
    """Add max partial current per species per sample id for CO2R rows; HER rows get 0.
    If averaged=True, use partial_current_*_mean and output max_partial_current_*_mean; otherwise use raw partial_current_*.
    """
    try:
        df_out = df.copy()
        if 'reaction' not in df_out.columns or 'sample id' not in df_out.columns:
            return df_out

        # Collect partial current columns depending on mode
        if averaged:
            value_cols = [c for c in df_out.columns if c.startswith('partial_current_') and c.endswith('_mean')]
        else:
            value_cols = [c for c in df_out.columns if c.startswith('partial_current_') and not c.endswith('_mean') and not c.endswith('_std')]

        if not value_cols:
            return df_out

        co2r_mask = df_out['reaction'] == 'CO2R'
        co2r_df = df_out[co2r_mask]

        for value_col in value_cols:
            # Derive species name
            species = value_col[len('partial_current_'):-len('_mean')] if averaged else value_col[len('partial_current_'):]
            target_col = f"max_partial_current_{species}_mean" if averaged else f"max_partial_current_{species}"

            try:
                if not co2r_df.empty and value_col in co2r_df.columns:
                    max_per_sample = co2r_df.groupby('sample id')[value_col].max()
                    df_out[target_col] = 0.0
                    df_out.loc[co2r_mask, target_col] = df_out.loc[co2r_mask, 'sample id'].map(max_per_sample).fillna(0.0)
                else:
                    df_out[target_col] = 0.0
            except Exception:
                # Ensure column exists even if computation failed
                df_out[target_col] = 0.0

        return df_out
    except Exception:
        return df

# Load the DataFrame
try:
    main_df = pd.read_csv("Data/DashboardData.csv")
    print(f"Data loaded successfully. Shape: {main_df.shape}")
    print(f"Columns: {list(main_df.columns)}")

    # Add partial current density columns before capturing original column order
    main_df = add_partial_current_columns(main_df)
    # Add max partial current per species per sample id (CO2R rows only)
    main_df = add_max_partial_current_all(main_df, averaged=False)

    # Store the original column order (including newly added partial current columns)
    original_columns = list(main_df.columns)
    print(f"Original column order preserved: {len(original_columns)} columns")
    
except FileNotFoundError:
    print("ERROR: CSV file 'ExpDataDump_241125-clean.csv' not found!")
    print("Please check if the file exists in the current directory.")
    print("Current working directory:", os.getcwd())
    main_df = pd.DataFrame()  # Empty dataframe as fallback
    original_columns = []
except Exception as e:
    print(f"ERROR loading CSV file: {e}")
    main_df = pd.DataFrame()  # Empty dataframe as fallback
    original_columns = []

# Server tracking
running_servers = {
    'her': None,  # Will store the process object
    'co2': None
}

def is_server_running(port):
    """Check if a server is running on the specified port"""
    try:
        response = requests.get(f'http://localhost:{port}', timeout=2)
        return response.status_code == 200
    except:
        return False

def kill_server_on_port(port):
    """Kill any process running on the specified port"""
    try:
        for proc in psutil.process_iter(['pid', 'name', 'connections']):
            try:
                for conn in proc.info['connections'] or []:
                    if conn.laddr.port == port:
                        print(f"Killing process {proc.info['pid']} on port {port}")
                        proc.kill()
                        return True
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
        return False
    except Exception as e:
        print(f"Error killing server on port {port}: {e}")
        return False

def parse_target_composition(formula):
    """
    Parse target composition formula like 'Au-0.6-Ag-0.4' and return a dictionary
    of element: fraction pairs.
    """
    if pd.isna(formula) or not formula:
        return {}
    
    try:
        # Split by '-' and process pairs
        parts = str(formula).split('-')
        composition = {}
        
        for i in range(0, len(parts), 2):
            if i + 1 < len(parts):
                element = parts[i].strip()
                fraction = float(parts[i + 1].strip())
                composition[element] = fraction
        
        return composition
    except Exception as e:
        print(f"Error parsing target composition '{formula}': {e}")
        return {}

def parse_xrf_composition(formula):
    """
    Parse XRF composition formula like 'Au-0.6-Ag-0.4' and return a dictionary
    of element: fraction pairs.
    """
    if pd.isna(formula) or not formula:
        return {}
    
    try:
        # Split by '-' and process pairs
        parts = str(formula).split('-')
        composition = {}
        
        for i in range(0, len(parts), 2):
            if i + 1 < len(parts):
                element = parts[i].strip()
                fraction = float(parts[i + 1].strip())
                composition[element] = fraction
        
        return composition
    except Exception as e:
        print(f"Error parsing XRF composition '{formula}': {e}")
        return {}

def compositions_within_tolerance(comp1, comp2, tolerance):
    """
    Check if two compositions are within the specified tolerance.
    Returns True if all elements in both compositions are within tolerance.
    """
    if not comp1 or not comp2:
        return False
    
    # Get all unique elements from both compositions
    all_elements = set(comp1.keys()) | set(comp2.keys())
    
    for element in all_elements:
        val1 = comp1.get(element, 0.0)
        val2 = comp2.get(element, 0.0)
        
        if abs(val1 - val2) > tolerance:
            return False
    
    return True

def group_compositions_with_tolerance(df, composition_column, tolerance):
    """
    Group compositions within tolerance and return a mapping from original composition to grouped composition.
    """
    if composition_column not in df.columns:
        return {}
    
    # Get unique compositions
    unique_compositions = df[composition_column].dropna().unique()
    
    # Create mapping from original composition to grouped composition
    composition_groups = {}
    grouped_compositions = []
    
    for comp in unique_compositions:
        if pd.isna(comp) or not comp:
            continue
            
        # Parse the composition
        parsed_comp = parse_xrf_composition(comp) if composition_column == 'xrf composition' else parse_target_composition(comp)
        
        # Find if this composition matches any existing group
        matched_group = None
        for group_comp in grouped_compositions:
            group_parsed = parse_xrf_composition(group_comp) if composition_column == 'xrf composition' else parse_target_composition(group_comp)
            if compositions_within_tolerance(parsed_comp, group_parsed, tolerance):
                matched_group = group_comp
                break
        
        if matched_group:
            composition_groups[comp] = matched_group
        else:
            # Create new group
            grouped_compositions.append(comp)
            composition_groups[comp] = comp
    
    return composition_groups

def create_filter_dashboard():
    """
    Creates a Flask web application with interactive filters for DataFrame exploration.
    """
    # Create Flask app
    app = Flask(__name__)
    
    # Register blueprints
    app.register_blueprint(her_plot_bp)
    app.register_blueprint(co2_plot_bp)
    app.register_blueprint(xrd_plot_bp)
    
    @app.route('/upload_data', methods=['POST'])
    def upload_data():
        """Handle CSV file upload and validate data"""
        try:
            if 'file' not in request.files:
                return jsonify({'success': False, 'error': 'No file provided'}), 400
            
            file = request.files['file']
            if file.filename == '':
                return jsonify({'success': False, 'error': 'No file selected'}), 400
            
            if not file.filename.lower().endswith('.csv'):
                return jsonify({'success': False, 'error': 'File must be a CSV'}), 400
            
            # Read the CSV file
            try:
                df = pd.read_csv(file)
            except Exception as e:
                return jsonify({'success': False, 'error': f'Failed to read CSV: {str(e)}'}), 400
            
            if df.empty:
                return jsonify({'success': False, 'error': 'CSV file is empty'}), 400
            
            # Basic validation - check for some expected columns
            required_columns = ['sample id']
            missing_required = [col for col in required_columns if col not in df.columns]
            if missing_required:
                return jsonify({
                    'success': False, 
                    'error': f'Missing required columns: {", ".join(missing_required)}. Please ensure your CSV contains at least: {", ".join(required_columns)}'
                }), 400
            
            # Add partial current columns to uploaded data
            df = add_partial_current_columns(df)
            # Add max partial current per species per sample id (CO2R rows only)
            df = add_max_partial_current_all(df, averaged=False)

            # Store the original column order for the uploaded dataset
            uploaded_columns = list(df.columns)
            
            # Clean the data
            # Replace infinite values with NaN
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            
            # Convert to records for JSON serialization
            data_records = df.to_dict('records')
            
            # Store uploaded data globally (in a real application, you'd use a database or session storage)
            global uploaded_data_cache
            uploaded_data_cache = {
                'data': data_records,
                'columns': uploaded_columns,
                'original_filename': file.filename
            }
            
            return jsonify({
                'success': True,
                'data': data_records,
                'column_order': uploaded_columns,
                'rows': len(data_records),
                'columns': len(uploaded_columns),
                'filename': file.filename
            })
            
        except Exception as e:
            return jsonify({'success': False, 'error': f'Upload failed: {str(e)}'}), 500
    
    @app.route('/save_current_data', methods=['POST'])
    def save_current_data():
        """Save current filtered/averaged data to a file for plot servers to read"""
        try:
            data = request.get_json()
            if data and 'plot_type' in data:
                import json
                import os
                
                # Get the data from the request
                if 'data' in data and data['data']:
                    current_data = data['data']
                    
                    # Use the original column order from the dashboard, but only include columns that exist in the data
                    available_columns = []
                    for col in original_columns:
                        if any(col in row for row in current_data):
                            available_columns.append(col)
                    
                    # Add any remaining columns that weren't in the original order
                    df = pd.DataFrame(current_data)
                    for col in df.columns:
                        if col not in available_columns:
                            available_columns.append(col)
                    
                    # Create a temporary file with the current data and column order
                    filename = f"Data/current_data_{data['plot_type']}.json"
                    save_data = {
                        'data': current_data,
                        'columns': available_columns
                    }
                    with open(filename, 'w') as f:
                        json.dump(save_data, f)
                    
                    print(f"Saved current data for {data['plot_type']} plot: {len(current_data)} rows, columns: {available_columns}")
                    return jsonify({'success': True, 'message': 'Data saved successfully'})
                else:
                    return jsonify({'success': False, 'error': 'No data available'}), 400
            else:
                return jsonify({'success': False, 'error': 'Invalid data format'}), 400
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/upload_xrd_data', methods=['POST'])
    def upload_xrd_data():
        """Handle XRD folder upload and store files"""
        try:
            if 'xrd_files' not in request.files:
                return jsonify({'success': False, 'error': 'No XRD files provided'}), 400
            
            files = request.files.getlist('xrd_files')
            if not files:
                return jsonify({'success': False, 'error': 'No files selected'}), 400
            
            # Create custom XRD directories if they don't exist
            custom_xrd_dir = "Data/CustomXRD"
            raw_dir = os.path.join(custom_xrd_dir, "raw")
            normalized_dir = os.path.join(custom_xrd_dir, "normalized")
            
            os.makedirs(raw_dir, exist_ok=True)
            os.makedirs(normalized_dir, exist_ok=True)
            
            uploaded_files = []
            raw_files = 0
            normalized_files = 0
            
            for file in files:
                if file.filename == '':
                    continue
                
                # Extract just the filename without any path components
                # This handles cases where browser sends full folder structure like "XRD/raw/filename.xy"
                original_filename = file.filename
                base_filename = os.path.basename(original_filename)
                filename_lower = base_filename.lower()
                
                if not (filename_lower.endswith('.xy') or filename_lower.endswith('.csv')):
                    continue
                
                # Determine file type and destination based on file extension
                if filename_lower.endswith('.xy'):
                    dest_dir = raw_dir
                    raw_files += 1
                else:  # .csv files
                    dest_dir = normalized_dir
                    normalized_files += 1
                
                # Save the file using just the base filename (no path)
                file_path = os.path.join(dest_dir, base_filename)
                file.save(file_path)
                uploaded_files.append(base_filename)
                print(f"Saved XRD file: {file_path}")
            
            if not uploaded_files:
                return jsonify({'success': False, 'error': 'No valid XRD files (.xy or .csv) found'}), 400
            
            return jsonify({
                'success': True,
                'files_uploaded': len(uploaded_files),
                'raw_files': raw_files,
                'normalized_files': normalized_files,
                'files': uploaded_files
            })
            
        except Exception as e:
            print(f"XRD upload error: {e}")
            return jsonify({'success': False, 'error': f'XRD upload failed: {str(e)}'}), 500
    
    @app.route('/reset_xrd_data', methods=['POST'])
    def reset_xrd_data():
        """Reset XRD data to original by removing custom XRD directory"""
        try:
            import shutil
            
            custom_xrd_dir = "Data/CustomXRD"
            if os.path.exists(custom_xrd_dir):
                shutil.rmtree(custom_xrd_dir)
                print(f"Removed custom XRD directory: {custom_xrd_dir}")
            
            return jsonify({
                'success': True,
                'message': 'XRD data reset to original'
            })
            
        except Exception as e:
            print(f"XRD reset error: {e}")
            return jsonify({'success': False, 'error': f'XRD reset failed: {str(e)}'}), 500
    
    # Check if dataframe is empty
    if main_df.empty:
        print("ERROR: No data loaded! Cannot create filter dashboard.")
        return
    
    # Get unique values for filter options
    def get_filter_options():
        options = {}
        for col in main_df.columns:
            if main_df[col].dtype == 'object':  # Categorical columns
                unique_vals = main_df[col].dropna().unique()
                options[col] = sorted(unique_vals.tolist())
            else:  # Numeric columns
                options[col] = {
                    'min': float(main_df[col].min()) if not main_df[col].isna().all() else 0,
                    'max': float(main_df[col].max()) if not main_df[col].isna().all() else 1
                }
        return options
    
    filter_options = get_filter_options()
    
    def apply_default_sorting(df):
        """
        Apply default sorting: source, batch number, batch date, sample id, 
        reaction (HER first), then current density ascending
        """
        try:
            # Define sorting columns in order of priority
            sort_columns = []
            sort_orders = []
            
            # 1. Source
            if 'source' in df.columns:
                sort_columns.append('source')
                sort_orders.append(True)  # ascending
            
            # 2. Batch number
            if 'batch number' in df.columns:
                sort_columns.append('batch number')
                sort_orders.append(True)  # ascending
            
            # 3. Batch date
            if 'batch date' in df.columns:
                sort_columns.append('batch date')
                sort_orders.append(True)  # ascending
            
            # 4. Sample ID
            if 'sample id' in df.columns:
                sort_columns.append('sample id')
                sort_orders.append(True)  # ascending
            
            # 5. Reaction (HER first)
            if 'reaction' in df.columns:
                # Create a custom sort key to put HER first
                df['_reaction_sort'] = df['reaction'].apply(
                    lambda x: 0 if str(x).upper() == 'HER' else 1
                )
                sort_columns.append('_reaction_sort')
                sort_orders.append(True)  # ascending
                sort_columns.append('reaction')
                sort_orders.append(True)  # ascending
            
            # 6. Current density
            if 'current density' in df.columns:
                sort_columns.append('current density')
                sort_orders.append(True)  # ascending
            
            if sort_columns:
                df_sorted = df.sort_values(by=sort_columns, ascending=sort_orders)
                # Remove the temporary sorting column
                if '_reaction_sort' in df_sorted.columns:
                    df_sorted = df_sorted.drop('_reaction_sort', axis=1)
                return df_sorted
            else:
                return df
                
        except Exception as e:
            print(f"Warning: Sorting failed: {e}")
            return df
    
    # Create routes
    @app.route('/')
    def index():
        return render_template_string(open('templates/dashboard.html').read())
    
    @app.route('/get_data')
    def get_data():
        try:
            # Ensure columns are in original order
            df_ordered = main_df[original_columns] if original_columns else main_df
            
            # Apply default sorting
            df_sorted = apply_default_sorting(df_ordered)
            
            df_dict = df_sorted.to_dict('records')
            return jsonify({
                'success': True, 
                'data': df_dict,
                'columns': list(df_sorted.columns)  # Send column order explicitly
            })
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/filter_data', methods=['POST'])
    def filter_data():
        try:
            request_data = request.get_json()
            filters = request_data.get('filters', {})
            current_data = request_data.get('currentData', [])
            
            print(f"Received filters: {filters}")
            print(f"Current data length: {len(current_data)}")
            
            # Check if we're working with averaged data by looking for _mean or _std columns
            is_averaged_data = len(current_data) > 0 and any('_mean' in str(key) or '_std' in str(key) for row in current_data for key in row.keys())
            
            if is_averaged_data and len(current_data) > 0:
                # Working with averaged data - filter from the current data (which is the original averaged data)
                print("Detected averaged data - filtering from original averaged data")
                filtered_df = pd.DataFrame(current_data)
            else:
                # Working with original data - always start from main_df to avoid cascading filters
                print("Using original data for filtering (avoiding cascading filters)")
                filtered_df = main_df.copy()
            
            print(f"Using data shape: {filtered_df.shape}")
            
            # Apply source filter
            if filters.get('source') and len(filters['source']) > 0:
                print(f"Applying source filter: {filters['source']}")
                print(f"DEBUG: Source filter type: {type(filters['source'])}")
                print(f"DEBUG: Source filter content: {filters['source']}")
                print(f"DEBUG: 'both' in filters['source']: {'both' in filters['source']}")
                print(f"DEBUG: 'both_target' in filters['source']: {'both_target' in filters['source']}")
                print(f"DEBUG: filters['source'] == ['both_target']: {filters['source'] == ['both_target']}")
                print(f"DEBUG: filters['source'][0] == 'both_target': {filters['source'][0] == 'both_target' if filters['source'] else False}")
                print(f"DEBUG: About to check condition...")
                print(f"DEBUG: filters['source'] = {filters['source']}")
                print(f"DEBUG: 'both_target' in filters['source'] = {'both_target' in filters['source']}")
                before_count = len(filtered_df)
                
                # Handle special "both" options for samples made by both UofT and VSP
                if 'both' in filters['source']:
                    print(f"DEBUG: Processing 'both' filter")
                    
                    # Get all unique XRF compositions that exist in both UofT and VSP sources
                    both_samples = []
                    
                    # Get all unique XRF compositions from the full dataset
                    compositions = filtered_df['xrf composition'].unique()
                    print(f"DEBUG: Found {len(compositions)} unique XRF compositions")
                    print(f"DEBUG: XRF compositions: {compositions[:5]}...")  # Show first 5
                    
                    # Check what sources are available
                    available_sources = filtered_df['source'].unique()
                    print(f"DEBUG: Available sources: {available_sources}")
                    
                    for composition in compositions:
                        # Check if this composition exists in both UofT and VSP sources
                        uoft_samples = filtered_df[(filtered_df['xrf composition'] == composition) & (filtered_df['source'] == 'uoft')]
                        vsp_samples = filtered_df[(filtered_df['xrf composition'] == composition) & (filtered_df['source'] == 'vsp')]
                        
                        if len(uoft_samples) > 0 and len(vsp_samples) > 0:
                            print(f"DEBUG: Found XRF composition '{composition}' in both sources - UofT: {len(uoft_samples)} samples, VSP: {len(vsp_samples)} samples")
                            # This composition exists in both sources, include ALL samples with this composition
                            both_samples.extend(uoft_samples.index.tolist())
                            both_samples.extend(vsp_samples.index.tolist())
                    
                    print(f"DEBUG: Total samples found in both sources: {len(both_samples)}")
                    
                    # Filter to only include samples that exist in both sources
                    if both_samples:
                        filtered_df = filtered_df.loc[both_samples]
                        print(f"DEBUG: Successfully filtered to {len(filtered_df)} rows")
                    else:
                        # No samples found in both sources
                        print("DEBUG: No samples found in both sources, creating empty dataframe")
                        filtered_df = filtered_df.iloc[0:0]  # Create empty dataframe
                
                elif 'both_target' in filters['source']:
                    print(f"DEBUG: Processing 'both_target' filter")
                    
                    # Get all unique target compositions that exist in both UofT and VSP sources
                    both_samples = []
                    
                    # Get all unique target compositions from the full dataset
                    compositions = filtered_df['target composition'].unique()
                    print(f"DEBUG: Found {len(compositions)} unique target compositions")
                    print(f"DEBUG: Target compositions: {compositions[:5]}...")  # Show first 5
                    
                    # Check what sources are available
                    available_sources = filtered_df['source'].unique()
                    print(f"DEBUG: Available sources: {available_sources}")
                    
                    for composition in compositions:
                        # Check if this composition exists in both UofT and VSP sources
                        uoft_samples = filtered_df[(filtered_df['target composition'] == composition) & (filtered_df['source'] == 'uoft')]
                        vsp_samples = filtered_df[(filtered_df['target composition'] == composition) & (filtered_df['source'] == 'vsp')]
                        
                        if len(uoft_samples) > 0 and len(vsp_samples) > 0:
                            print(f"DEBUG: Found target composition '{composition}' in both sources - UofT: {len(uoft_samples)} samples, VSP: {len(vsp_samples)} samples")
                            # This composition exists in both sources, include ALL samples with this composition
                            both_samples.extend(uoft_samples.index.tolist())
                            both_samples.extend(vsp_samples.index.tolist())
                    
                    print(f"DEBUG: Total samples found in both sources: {len(both_samples)}")
                    
                    # Filter to only include samples that exist in both sources
                    if both_samples:
                        filtered_df = filtered_df.loc[both_samples]
                        print(f"DEBUG: Successfully filtered to {len(filtered_df)} rows")
                    else:
                        # No samples found in both sources
                        print("DEBUG: No samples found in both sources, creating empty dataframe")
                        filtered_df = filtered_df.iloc[0:0]  # Create empty dataframe
                else:
                    # Regular source filtering (exclude special 'both' options from the filter list)
                    source_filters = [s for s in filters['source'] if s not in ['both', 'both_target']]
                    if source_filters:
                        filtered_df = filtered_df[filtered_df['source'].isin(source_filters)]
                    else:
                        # Only special 'both' options were selected but we already handled them above
                        pass
                
                after_count = len(filtered_df)
                print(f"Source filter: {before_count} -> {after_count} rows")
            
            # Apply batch filter (simple batch numbers)
            if 'batch' in filters:
                if filters['batch'] and len(filters['batch']) > 0:
                    # Convert filter values to integers to match the data type
                    batch_values = [int(b) for b in filters['batch'] if str(b).isdigit()]
                    print(f"Applying batch filter: {batch_values}")
                    before_count = len(filtered_df)
                    filtered_df = filtered_df[filtered_df['batch number'].isin(batch_values)]
                    after_count = len(filtered_df)
                    print(f"Batch filter: {before_count} -> {after_count} rows")
                else:
                    # No batches selected - show no data
                    print("No batches selected - showing empty table")
                    filtered_df = filtered_df.iloc[0:0]  # Create empty dataframe with same structure
            
            # Apply reaction filter
            if 'reaction' in filters:
                if filters['reaction'] and len(filters['reaction']) > 0:
                    print(f"Applying reaction filter: {filters['reaction']}")
                    before_count = len(filtered_df)
                    filtered_df = filtered_df[filtered_df['reaction'].isin(filters['reaction'])]
                    after_count = len(filtered_df)
                    print(f"Reaction filter: {before_count} -> {after_count} rows")
                else:
                    # No reactions selected - show no data
                    print("No reactions selected - showing empty table")
                    filtered_df = filtered_df.iloc[0:0]  # Create empty dataframe with same structure
            
            # Apply voltage range filter (check for both voltage and voltage_mean columns)
            if filters.get('voltageMin'):
                if 'voltage_mean' in filtered_df.columns:
                    filtered_df = filtered_df[filtered_df['voltage_mean'] >= float(filters['voltageMin'])]
                elif 'voltage' in filtered_df.columns:
                    filtered_df = filtered_df[filtered_df['voltage'] >= float(filters['voltageMin'])]
            if filters.get('voltageMax'):
                if 'voltage_mean' in filtered_df.columns:
                    filtered_df = filtered_df[filtered_df['voltage_mean'] <= float(filters['voltageMax'])]
                elif 'voltage' in filtered_df.columns:
                    filtered_df = filtered_df[filtered_df['voltage'] <= float(filters['voltageMax'])]
            
            # Apply current density range filter
            if filters.get('currentDensityMin'):
                filtered_df = filtered_df[filtered_df['current density'] >= float(filters['currentDensityMin'])]
            if filters.get('currentDensityMax'):
                filtered_df = filtered_df[filtered_df['current density'] <= float(filters['currentDensityMax'])]
            
            # Apply element count filter
            if 'elementCount' in filters:
                element_counts = filters.get('elementCount', [])
                print(f"Element count filter values: {element_counts}")
                
                # If no element counts are selected, show all data (don't apply filter)
                if len(element_counts) == 0:
                    print("No element counts selected - showing all data")
                else:
                    print(f"Applying element count filter: {element_counts}")
                    before_count = len(filtered_df)
                    
                    # Define elemental columns (excluding non-elemental columns)
                    elemental_columns = ['Ag', 'Au', 'Cd', 'Cu', 'Ga', 'Hg', 'In', 'Mn', 'Mo', 'Nb', 'Ni', 'Pd', 'Pt', 'Rh', 'Sn', 'Tl', 'W', 'Zn']
                    
                    # Filter to only include columns that exist in the dataframe
                    available_elemental_columns = [col for col in elemental_columns if col in filtered_df.columns]
                    
                    if available_elemental_columns:
                        # Create a mask for rows that have the specified number of non-zero elements
                        def count_non_zero_elements(row):
                            return sum(1 for col in available_elemental_columns if row[col] > 0)
                        
                        # Apply the filter
                        element_count_mask = filtered_df.apply(count_non_zero_elements, axis=1).isin(element_counts)
                        filtered_df = filtered_df[element_count_mask]
                        
                        after_count = len(filtered_df)
                        print(f"Element count filter: {before_count} -> {after_count} rows")
                    else:
                        print("No elemental columns found for element count filtering")
            
            # Reorder columns to match the original dataframe structure
            reordered_columns = []
            for col in original_columns:
                if col in filtered_df.columns:
                    # Original column exists, add it
                    reordered_columns.append(col)
                elif col + '_mean' in filtered_df.columns:
                    # Original column was renamed to _mean, add the _mean version
                    reordered_columns.append(col + '_mean')
                    # Also add the corresponding _std column if it exists
                    if col + '_std' in filtered_df.columns:
                        reordered_columns.append(col + '_std')
                # If neither exists, skip it (e.g., 'rep' column)
            
            # Add any remaining columns that weren't in the original order
            for col in filtered_df.columns:
                if col not in reordered_columns:
                    reordered_columns.append(col)
            
            df_ordered = filtered_df[reordered_columns] if reordered_columns else filtered_df
            
            # Apply default sorting
            df_sorted = apply_default_sorting(df_ordered)
            
            df_dict = df_sorted.to_dict('records')
            return jsonify({
                'success': True, 
                'data': df_dict,
                'columns': list(df_sorted.columns)  # Send column order explicitly
            })
            
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/average_data', methods=['POST'])
    def average_data():
        try:
            # Get grouping columns from request
            request_data = request.get_json() or {}
            user_group_columns = request_data.get('groupingColumns', [])
            composition_tolerance = request_data.get('compositionTolerance', 0.01)
            current_data_list = request_data.get('currentData')
            
            # Use provided current data or fall back to main_df
            if current_data_list:
                work_df = pd.DataFrame(current_data_list)
                print(f"AGGREGATION: Received {len(work_df)} rows of uploaded data")
            else:
                work_df = main_df.copy()
                print(f"AGGREGATION: Using original {len(work_df)} rows")
            
            # Validate composition tolerance
            try:
                composition_tolerance = float(composition_tolerance)
                if composition_tolerance < 0 or composition_tolerance > 1:
                    return jsonify({'success': False, 'error': 'Composition tolerance must be between 0 and 1'}), 400
            except (ValueError, TypeError):
                return jsonify({'success': False, 'error': 'Invalid composition tolerance value'}), 400
            
            # Default grouping columns (mandatory)
            default_group_columns = ['reaction', 'current density']
            
            # Combine mandatory and user-selected columns
            group_columns = default_group_columns.copy()
            for col in user_group_columns:
                if col not in group_columns:  # Avoid duplicates
                    group_columns.append(col)
            
            print(f"User selected grouping columns: {user_group_columns}")
            print(f"Final grouping columns: {group_columns}")
            print(f"Composition tolerance: {composition_tolerance}")
            
            # Check which grouping columns exist in the dataframe
            available_group_columns = [col for col in group_columns if col in work_df.columns]
            
            if not available_group_columns:
                return jsonify({'success': False, 'error': 'No grouping columns found'}), 400
            
            # Handle composition tolerance for XRF and target composition columns
            df_for_grouping = work_df.copy()
            composition_mappings = {}
            
            if 'xrf composition' in available_group_columns and composition_tolerance > 0:
                print("Applying XRF composition tolerance grouping...")
                xrf_mapping = group_compositions_with_tolerance(df_for_grouping, 'xrf composition', composition_tolerance)
                composition_mappings['xrf composition'] = xrf_mapping
                df_for_grouping['xrf composition'] = df_for_grouping['xrf composition'].map(xrf_mapping).fillna(df_for_grouping['xrf composition'])
                print(f"XRF composition groups: {len(set(xrf_mapping.values()))} groups from {len(xrf_mapping)} compositions")
            
            if 'target composition' in available_group_columns and composition_tolerance > 0:
                print("Applying target composition tolerance grouping...")
                target_mapping = group_compositions_with_tolerance(df_for_grouping, 'target composition', composition_tolerance)
                composition_mappings['target composition'] = target_mapping
                df_for_grouping['target composition'] = df_for_grouping['target composition'].map(target_mapping).fillna(df_for_grouping['target composition'])
                print(f"Target composition groups: {len(set(target_mapping.values()))} groups from {len(target_mapping)} compositions")
            
            # Identify only voltage and fe_ columns for averaging
            voltage_fe_columns = [col for col in work_df.columns 
                                if (col.startswith('voltage') or col.startswith('fe_')) 
                                and col not in available_group_columns]
            
            # All other columns (excluding grouping columns and rep) should be preserved
            preserve_columns = [col for col in work_df.columns 
                              if col not in available_group_columns 
                              and col not in voltage_fe_columns 
                              and col != 'rep']
            
            # Group and aggregate
            if voltage_fe_columns:
                # Create a copy of the dataframe and replace NaN with 0 for voltage and fe_ columns
                df_for_averaging = df_for_grouping.copy()
                for col in voltage_fe_columns:
                    df_for_averaging[col] = df_for_averaging[col].fillna(0)
                
                # Create aggregation dictionary
                agg_dict = {}
                
                # Only average voltage and fe_ columns (with NaN replaced by 0)
                for col in voltage_fe_columns:
                    agg_dict[col] = 'mean'
                
                # For all other columns, take the first value to preserve them
                for col in preserve_columns:
                    agg_dict[col] = 'first'
                
                averaged_df = df_for_averaging.groupby(available_group_columns).agg(agg_dict).reset_index()
                
                # Add sample count as a separate column if sample id column exists
                if 'sample id' in df_for_averaging.columns:
                    sample_counts = df_for_averaging.groupby(available_group_columns)['sample id'].nunique().reset_index()
                    sample_counts = sample_counts.rename(columns={'sample id': 'sample_count'})
                    averaged_df = averaged_df.merge(sample_counts, on=available_group_columns, how='left')
                    print(f"Added sample_count column with unique sample ID counts")
                
                # Add calculated standard deviation columns for each averaged column
                for col in voltage_fe_columns:
                    std_col_name = col + '_std'
                    try:
                        # Calculate standard deviation for each group
                        std_values = df_for_averaging.groupby(available_group_columns)[col].std()
                        # Merge the std values back to the averaged dataframe
                        averaged_df = averaged_df.merge(
                            std_values.rename(std_col_name).reset_index(), 
                            on=available_group_columns, 
                            how='left'
                        )
                        # Fill NaN values (where std calculation failed) with 0
                        averaged_df[std_col_name] = averaged_df[std_col_name].fillna(0.0)
                    except Exception as e:
                        print(f"Warning: Failed to calculate std for {col}: {e}")
                        # If calculation fails, fill with zeros
                        averaged_df[std_col_name] = 0.0
            else:
                # If no voltage/fe_ columns, just group and take first values for all columns
                all_other_cols = [col for col in work_df.columns 
                                if col not in available_group_columns and col != 'rep']
                agg_dict = {col: 'first' for col in all_other_cols}
                
                averaged_df = df_for_grouping.groupby(available_group_columns).agg(agg_dict).reset_index()
                
                # Add sample count as a separate column if sample id column exists
                if 'sample id' in df_for_grouping.columns:
                    sample_counts = df_for_grouping.groupby(available_group_columns)['sample id'].nunique().reset_index()
                    sample_counts = sample_counts.rename(columns={'sample id': 'sample_count'})
                    averaged_df = averaged_df.merge(sample_counts, on=available_group_columns, how='left')
                    print(f"Added sample_count column with unique sample ID counts")
            
            # Rename voltage and fe_ columns to add "_mean" suffix BEFORE reordering
            # But exclude _std columns from this renaming
            column_mapping = {}
            for col in averaged_df.columns:
                if (col.startswith('voltage') or col.startswith('fe_')) and not col.endswith('_std'):
                    column_mapping[col] = col + '_mean'
            
            if column_mapping:
                averaged_df = averaged_df.rename(columns=column_mapping)
                print(f"Renamed columns after averaging: {column_mapping}")
            
            # Add partial current columns for averaged data (creates partial_current_*_mean)
            averaged_df = add_partial_current_columns(averaged_df)

            # Create partial current std columns using FE stds
            try:
                fe_std_cols = [c for c in averaged_df.columns if c.startswith('fe_') and c.endswith('_std')]
                if 'current density' in averaged_df.columns:
                    for col in fe_std_cols:
                        species = col[len('fe_'):-len('_std')]
                        pc_std_col = f"partial_current_{species}_std"
                        try:
                            averaged_df[pc_std_col] = (averaged_df[col] * averaged_df['current density']) / 100.0
                        except Exception:
                            pass
            except Exception:
                pass

            # Add max partial current for averaged data (per sample id within CO2R rows)
            averaged_df = add_max_partial_current_all(averaged_df, averaged=True)

            # Remove raw partial_current_* columns from averaged output, keep only *_mean and *_std
            try:
                raw_pc_cols = [c for c in averaged_df.columns if c.startswith('partial_current_') and not (c.endswith('_mean') or c.endswith('_std'))]
                if raw_pc_cols:
                    averaged_df = averaged_df.drop(columns=raw_pc_cols)
            except Exception:
                pass

            # Now reorder columns to maintain original positions with _mean suffixes
            reordered_columns = []
            for col in original_columns:
                if col in averaged_df.columns:
                    # Original column exists, add it
                    reordered_columns.append(col)
                elif col + '_mean' in averaged_df.columns:
                    # Original column was renamed to _mean, add the _mean version
                    reordered_columns.append(col + '_mean')
                    # Also add the corresponding _std column if it exists
                    if col + '_std' in averaged_df.columns:
                        reordered_columns.append(col + '_std')
                # If neither exists, skip it (e.g., 'rep' column)
            
            # Add any remaining columns that weren't in the original order
            for col in averaged_df.columns:
                if col not in reordered_columns:
                    reordered_columns.append(col)
            
            df_ordered = averaged_df[reordered_columns]
            
            # If target composition was used for grouping, parse it into elemental columns
            if 'target composition' in available_group_columns and 'target composition' in df_ordered.columns:
                print("Target composition detected in grouping. Parsing into elemental columns...")
                for idx, row in df_ordered.iterrows():
                    target_formula = row['target composition']
                    if pd.notna(target_formula) and target_formula:
                        parsed_composition = parse_target_composition(target_formula)
                        for element, fraction in parsed_composition.items():
                            if element in df_ordered.columns:
                                df_ordered.at[idx, element] = fraction
                
                # Remove xrf composition column when target composition is used
                if 'xrf composition' in df_ordered.columns:
                    df_ordered = df_ordered.drop('xrf composition', axis=1)
                    print("Removed 'xrf composition' column as 'target composition' is being used")
            
            # Apply default sorting
            df_sorted = apply_default_sorting(df_ordered)
            
            # Convert to dictionary for JSON response
            df_dict = df_sorted.to_dict('records')
            return jsonify({
                'success': True, 
                'data': df_dict,
                'columns': list(df_sorted.columns)  # Send column order explicitly
            })
            
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/export_csv', methods=['POST'])
    def export_csv():
        try:
            # Get the current filtered/averaged data from the request
            data = request.get_json()
            
            if data and 'data' in data and 'columns' in data:
                # Export the current filtered/averaged data
                import pandas as pd
                df = pd.DataFrame(data['data'])
                
                # Use the column order from the frontend (which includes averaged columns)
                if data['columns']:
                    # Ensure all columns from the frontend are present
                    available_columns = [col for col in data['columns'] if col in df.columns]
                    if available_columns:
                        df = df[available_columns]
                
                csv_data = df.to_csv(index=False)
                
                # Determine filename based on whether data appears to be averaged
                has_mean_columns = any('_mean' in col or '_std' in col for col in df.columns)
                filename = 'averaged_data.csv' if has_mean_columns else 'filtered_data.csv'
                
            else:
                # Fallback to original data if no current data provided
                df_ordered = main_df[original_columns] if original_columns else main_df
                csv_data = df_ordered.to_csv(index=False)
                filename = 'all_data.csv'
            
            from flask import Response
            response = Response(
                csv_data,
                mimetype='text/csv',
                headers={'Content-Disposition': f'attachment; filename={filename}'}
            )
            
            return response
            
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/her_plot')
    def her_plot():
        """Route to serve the HER interactive plot"""
        try:
            # Start the HER plot server in a separate thread if not already running
            import subprocess
            import sys
            import os
            
            her_port = 8083
            
            # Check if HER plot server is already running
            if is_server_running(her_port):
                
                return '''
                <!DOCTYPE html>
                <html>
                <head>
                    <title>HER Performance Plot</title>
                    <meta http-equiv="refresh" content="0; url=http://localhost:8083">
                </head>
                <body>
                    <p>Opening HER Performance Plot...</p>
                    <p>If the plot doesn't open automatically, <a href="http://localhost:8083">click here</a></p>
                </body>
                </html>
                '''
            
            # Start the HER plot server
            def start_her_server():
                try:
                    proc = subprocess.Popen([sys.executable, 'interactive_plot_HER.py'], 
                                          cwd=os.getcwd(), 
                                          stdout=subprocess.DEVNULL, 
                                          stderr=subprocess.DEVNULL)
                    running_servers['her'] = proc
                    time.sleep(5)  # Give it more time to start
                    

                except Exception as e:
                    print(f"Error starting HER server: {e}")
            
            # Start server in background
            her_thread = threading.Thread(target=start_her_server)
            her_thread.daemon = True
            her_thread.start()
            
            # Return redirect page with delay
            return '''
            <!DOCTYPE html>
            <html>
            <head>
                <title>HER Performance Plot</title>
                <meta http-equiv="refresh" content="6; url=http://localhost:8083">
            </head>
            <body>
                <p>Starting HER Performance Plot server...</p>
                <p>Please wait a moment for the plot to load.</p>
                <p>If the plot doesn't open automatically, <a href="http://localhost:8083">click here</a></p>
            </body>
            </html>
            '''
            
        except Exception as e:
            return f"Error loading HER plot: {str(e)}", 500
    
    @app.route('/close_her_plot')
    def close_her_plot():
        """Route to close the HER interactive plot server"""
        try:
            her_port = 8083
            success = False
            
            # Try to kill the tracked process first
            if running_servers['her'] and running_servers['her'].poll() is None:
                try:
                    running_servers['her'].terminate()
                    running_servers['her'].wait(timeout=5)
                    running_servers['her'] = None
                    success = True
                    print("HER server terminated via tracked process")
                except Exception as e:
                    print(f"Error terminating tracked HER server: {e}")
            
            # If that didn't work, kill by port
            if not success:
                success = kill_server_on_port(her_port)
                if success:
                    print("HER server terminated via port kill")
            
            if success:
                return jsonify({'success': True, 'message': 'HER plot server closed successfully'})
            else:
                return jsonify({'success': False, 'message': 'No HER server was running'})
                
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/her_server_status')
    def her_server_status():
        """Check if HER server is running"""
        try:
            her_port = 8083
            is_running = is_server_running(her_port)
            return jsonify({'running': is_running})
        except Exception as e:
            return jsonify({'running': False, 'error': str(e)}), 500
    
    @app.route('/co2_plot')
    def co2_plot():
        """Route to redirect to the CO2RR blueprint plot"""
        return '''
        <!DOCTYPE html>
        <html>
        <head>
            <title>CO2RR Performance Plot</title>
            <meta http-equiv="refresh" content="0; url=/co2">
        </head>
        <body>
            <p>Opening CO2RR Performance Plot...</p>
            <p>If the plot doesn't open automatically, <a href="/co2">click here</a></p>
        </body>
        </html>
        '''
    
    @app.route('/close_co2_plot')
    def close_co2_plot():
        """Route to close the CO2RR interactive plot server (for compatibility with older UI)"""
        # This is now just a stub for compatibility with the dashboard UI
        # since we're using a blueprint instead of a separate server
        return jsonify({'success': True, 'message': 'CO2RR plot is integrated - no need to close'})        
    
    @app.route('/co2_server_status')
    def co2_server_status():
        """Check if CO2RR server is available (for compatibility with older UI)"""
        # Always return true since we're using a blueprint instead of a separate server
        return jsonify({'running': True})
    
    @app.route('/get_xrd_count')
    def get_xrd_count():
        """Get the count of XRD normalized files"""
        try:
            xrd_normalized_path = "Data/XRD/normalized"
            if os.path.exists(xrd_normalized_path):
                # Count CSV files in the normalized directory
                csv_files = [f for f in os.listdir(xrd_normalized_path) if f.endswith('.csv')]
                count = len(csv_files)
                return jsonify({'success': True, 'count': count})
            else:
                return jsonify({'success': False, 'error': 'XRD normalized directory not found'}), 404
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500
    
    # New embedded plotting routes
    @app.route('/plot/her')
    def embedded_her_plot():
        """Embedded HER plot using current dashboard data"""
        try:
            import plotly.graph_objects as go
            import plotly.offline as pyo
            import json
            
            # Load current HER data
            current_data_file = "Data/current_data_her.json"
            if os.path.exists(current_data_file):
                with open(current_data_file, 'r') as f:
                    saved_data = json.load(f)
                
                if isinstance(saved_data, dict) and 'data' in saved_data:
                    df_plot = pd.DataFrame(saved_data['data'])
                else:
                    df_plot = pd.DataFrame(saved_data)
            else:
                # Fallback to filtered main data for HER
                df_plot = main_df[main_df['reaction'] == 'HER'].copy() if 'reaction' in main_df.columns else main_df.copy()
            
            if df_plot.empty:
                return "<h2>No HER data available for plotting</h2><p>Please filter for HER data in the main dashboard first.</p>"
            
            # Create a simple scatter plot
            fig = go.Figure()
            
            # Plot voltage vs current density if available
            if 'voltage_mean' in df_plot.columns or 'voltage' in df_plot.columns:
                voltage_col = 'voltage_mean' if 'voltage_mean' in df_plot.columns else 'voltage'
                fig.add_trace(go.Scatter(
                    x=df_plot[voltage_col],
                    y=df_plot['current density'],
                    mode='markers',
                    text=[f"Sample: {row.get('sample id', 'Unknown')}<br>Source: {row.get('source', 'Unknown')}" for _, row in df_plot.iterrows()],
                    marker=dict(size=8, opacity=0.7)
                ))
                
                fig.update_layout(
                    title='HER Performance: Voltage vs Current Density',
                    xaxis_title='Voltage (V)',
                    yaxis_title='Current Density (mA/cm²)',
                    height=600,
                    template='plotly_white'
                )
            else:
                return "<h2>Insufficient data for HER plotting</h2><p>Voltage data not found.</p>"
            
            # Convert to HTML
            plot_html = pyo.plot(fig, output_type='div', include_plotlyjs=True)
            
            return f'''
            <!DOCTYPE html>
            <html>
            <head>
                <title>HER Performance Plot</title>
                <meta charset="utf-8">
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .back-link {{ margin-bottom: 20px; }}
                    .back-link a {{ text-decoration: none; background: #007bff; color: white; padding: 10px 20px; border-radius: 5px; }}
                </style>
            </head>
            <body>
                <div class="back-link">
                    <a href="/">← Back to Dashboard</a>
                </div>
                <h1>HER Performance Analysis</h1>
                {plot_html}
            </body>
            </html>
            '''
            
        except Exception as e:
            return f"<h2>Error loading HER plot: {str(e)}</h2><p>Please ensure HER data is available in the dashboard.</p>"
    
    @app.route('/plot/co2')
    def embedded_co2_plot():
        """Embedded CO2 plot using current dashboard data"""
        try:
            import plotly.graph_objects as go
            import plotly.offline as pyo
            import json
            
            # Load current CO2 data
            current_data_file = "Data/current_data_co2.json"
            if os.path.exists(current_data_file):
                with open(current_data_file, 'r') as f:
                    saved_data = json.load(f)
                
                if isinstance(saved_data, dict) and 'data' in saved_data:
                    df_plot = pd.DataFrame(saved_data['data'])
                else:
                    df_plot = pd.DataFrame(saved_data)
            else:
                # Fallback to filtered main data for CO2R
                df_plot = main_df[main_df['reaction'] == 'CO2R'].copy() if 'reaction' in main_df.columns else main_df.copy()
            
            if df_plot.empty:
                return "<h2>No CO2R data available for plotting</h2><p>Please filter for CO2R data in the main dashboard first.</p>"
            
            # Create a simple scatter plot
            fig = go.Figure()
            
            # Plot voltage vs current density if available
            if 'voltage_mean' in df_plot.columns or 'voltage' in df_plot.columns:
                voltage_col = 'voltage_mean' if 'voltage_mean' in df_plot.columns else 'voltage'
                fig.add_trace(go.Scatter(
                    x=df_plot[voltage_col],
                    y=df_plot['current density'],
                    mode='markers',
                    text=[f"Sample: {row.get('sample id', 'Unknown')}<br>Source: {row.get('source', 'Unknown')}" for _, row in df_plot.iterrows()],
                    marker=dict(size=8, opacity=0.7, color='red')
                ))
                
                fig.update_layout(
                    title='CO2RR Performance: Voltage vs Current Density',
                    xaxis_title='Voltage (V)',
                    yaxis_title='Current Density (mA/cm²)',
                    height=600,
                    template='plotly_white'
                )
            else:
                return "<h2>Insufficient data for CO2R plotting</h2><p>Voltage data not found.</p>"
            
            # Convert to HTML
            plot_html = pyo.plot(fig, output_type='div', include_plotlyjs=True)
            
            return f'''
            <!DOCTYPE html>
            <html>
            <head>
                <title>CO2RR Performance Plot</title>
                <meta charset="utf-8">
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .back-link {{ margin-bottom: 20px; }}
                    .back-link a {{ text-decoration: none; background: #007bff; color: white; padding: 10px 20px; border-radius: 5px; }}
                </style>
            </head>
            <body>
                <div class="back-link">
                    <a href="/">← Back to Dashboard</a>
                </div>
                <h1>CO2RR Performance Analysis</h1>
                {plot_html}
            </body>
            </html>
            '''
            
        except Exception as e:
            return f"<h2>Error loading CO2R plot: {str(e)}</h2><p>Please ensure CO2R data is available in the dashboard.</p>"
    
    # Get port from environment variable (for cloud deployment) or use default
    import os
    port = int(os.environ.get('PORT', 8085))
    
    # Start the Flask app in a separate thread
    def run_app():
        try:
            app.run(debug=False, use_reloader=False, port=port, host='0.0.0.0')
        except Exception as e:
            print(f"Error starting server on port {port}: {e}")
            # Try alternative port only in development
            if port == 8085:  # Only try alternative if using default port
                try:
                    app.run(debug=False, use_reloader=False, port=8086, host='0.0.0.0')
                except Exception as e2:
                    print(f"Error with alternative port: {e2}")
                    return
            else:
                return
    
    # Check if running in production (when PORT env var is set by cloud provider)
    is_production = 'PORT' in os.environ
    
    if is_production:
        # Production mode: run directly without threading or browser opening
        print(f"Starting production server on port {port}")
        app.run(debug=False, use_reloader=False, port=port, host='0.0.0.0')
    else:
        # Development mode: use threading and open browser
        thread = threading.Thread(target=run_app)
        thread.daemon = True
        thread.start()
        
        # Wait a moment for the server to start
        time.sleep(2)
        
        # Try to open the browser with different URLs
        urls_to_try = [
            f'http://localhost:{port}',
            f'http://127.0.0.1:{port}',
            'http://localhost:8086',
            'http://127.0.0.1:8086'
        ]
        
        browser_opened = False
        for url in urls_to_try:
            try:
                webbrowser.open(url)
                browser_opened = True
                print(f"Filter dashboard opened in your browser at {url}!")
                break
            except Exception as e:
                print(f"Could not open {url}: {e}")
                continue
        
        if not browser_opened:
            print("Could not automatically open browser. Please manually navigate to:")
            print(f"http://localhost:{port} or http://localhost:8086")
        
        print("The Flask server is running. Close this terminal or press Ctrl+C to stop the server when done.")
        print("This interface allows you to filter and explore the OCx24 dataset.")
        
        # Keep the main thread alive
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down server...")

    return app  # Return the app for production WSGI servers

# Call the filter dashboard function
if __name__ == "__main__":
    create_filter_dashboard()
