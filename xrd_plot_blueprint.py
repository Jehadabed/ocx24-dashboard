import pandas as pd
import matplotlib.pyplot as plt
from flask import Blueprint, render_template_string, request, jsonify, Response, send_file
import os
import json

# Create Blueprint
xrd_plot_bp = Blueprint('xrd_plot', __name__, url_prefix='/xrd')

@xrd_plot_bp.route('/')
def xrd_plot_main():
    """Main XRD plot page"""
    
    # Create the HTML template exactly matching the original XRD dashboard
    html_template = '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>OCx24 XRD Analysis Dashboard</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
            }
            
            .back-link {
                position: fixed;
                top: 20px;
                left: 20px;
                z-index: 1000;
                background: #1a73e8;
                color: white;
                padding: 12px 20px;
                border-radius: 8px;
                text-decoration: none;
                font-weight: 500;
                font-size: 14px;
                transition: all 0.2s ease;
                box-shadow: 0 2px 8px rgba(26, 115, 232, 0.3);
            }
            
            .back-link:hover {
                background: #1557b0;
                transform: translateY(-1px);
                box-shadow: 0 4px 12px rgba(26, 115, 232, 0.4);
            }
            
            .container {
                max-width: 1400px;
                margin: 0 auto;
                background-color: white;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                padding: 30px;
            }
            
            .header {
                text-align: center;
                margin-bottom: 30px;
                border-bottom: 2px solid #e0e0e0;
                padding-bottom: 20px;
            }
            
            .header h1 {
                color: #333; 
                margin: 0;
                font-size: 2.5em;
            }
            
            .header p {
                color: #666;
                margin: 10px 0 0 0;
                font-size: 1.1em;
            }
            
            .controls {
                display: flex;
                gap: 20px;
                margin-bottom: 30px;
                align-items: center;
                flex-wrap: wrap;
            }
            
            .control-group {
                display: flex;
                flex-direction: column;
                gap: 5px;
            }
            
            .control-group label {
                font-weight: 600;
                color: #555;
                font-size: 0.9em;
            }
            
            .control-group select,
            .control-group input {
                padding: 10px 15px;
                border: 2px solid #ddd;
                border-radius: 5px;
                font-size: 1em;
                background-color: white;
                min-width: 200px;
                transition: border-color 0.3s;
            }
            
            .control-group select:focus,
            .control-group input:focus {
                outline: none;
                border-color: #4CAF50;
            }
            
            .control-group input.invalid {
                border-color: #f44336;
                background-color: #ffebee;
            }
            
            .control-group input.valid {
                border-color: #4CAF50;
                background-color: #e8f5e8;
            }
            
            .plot-container {
                background-color: white;
                border-radius: 8px;
                padding: 20px;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            }
            
            .metadata-panel {
                background-color: white;
                border-radius: 8px;
                padding: 20px;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
                margin-bottom: 20px;
            }
            
            .metadata-panel h3 {
                margin-top: 0;
                color: #333;
                border-bottom: 2px solid #4CAF50;
                padding-bottom: 10px;
            }
            
            .metadata-section {
                margin-bottom: 20px;
            }
            
            .metadata-section h4 {
                color: #555;
                margin-bottom: 10px;
                font-size: 1.1em;
            }
            
            .metadata-item {
                display: flex;
                justify-content: space-between;
                padding: 5px 0;
                border-bottom: 1px solid #eee;
            }
            
            .metadata-item:last-child {
                border-bottom: none;
            }
            
            .metadata-label {
                font-weight: 600;
                color: #666;
            }
            
            .metadata-value {
                color: #333;
            }
            
            .iteration-list {
                max-height: 200px;
                overflow-y: auto;
                border: 1px solid #ddd;
                border-radius: 4px;
                padding: 10px;
            }
            
            .iteration-item {
                padding: 8px;
                margin: 5px 0;
                border-radius: 4px;
                cursor: pointer;
                transition: background-color 0.2s;
            }
            
            .iteration-item:hover {
                background-color: #f5f5f5;
            }
            
            .iteration-item.selected {
                background-color: #e8f5e8;
                border: 1px solid #4CAF50;
            }
            
            .rwp-score {
                font-weight: bold;
            }
            
            .rwp-good { color: #4CAF50; }
            .rwp-medium { color: #FF9800; }
            .rwp-poor { color: #f44336; }
            
            .iterations-table-panel {
                background-color: white; 
                border-radius: 8px;
                padding: 20px;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
                margin-bottom: 20px;
            }
            
            .iterations-table-panel h3 {
                margin-top: 0;
                color: #333;
                border-bottom: 2px solid #4CAF50;
                padding-bottom: 10px;
            }
            
            .table-controls {
                margin-bottom: 15px;
                display: flex;
                gap: 10px;
                flex-wrap: wrap;
                align-items: center;
            }
            
            .rwp-filter {
                display: flex;
                align-items: center;
                gap: 5px;
                background-color: #f8f9fa;
                padding: 8px 12px;
                border-radius: 4px;
                border: 1px solid #ddd;
            }
            
            .rwp-filter label {
                font-size: 0.9em;
                font-weight: 600;
                color: #555;
                margin: 0;
            }
            
            .rwp-filter input {
                width: 80px;
                padding: 4px 8px;
                border: 1px solid #ccc;
                border-radius: 3px;
                font-size: 0.9em;
            }
            
            .rwp-filter input:focus {
                outline: none;
                border-color: #4CAF50;
            }
            
            .weight-filter {
                display: flex;
                align-items: center;
                gap: 5px;
                background-color: #f8f9fa;
                padding: 8px 12px;
                border-radius: 4px;
                border: 1px solid #ddd;
            }
            
            .weight-filter label {
                font-size: 0.9em;
                font-weight: 600;
                color: #555;
                margin: 0;
            }
            
            .weight-filter input {
                width: 80px;
                padding: 4px 8px;
                border: 1px solid #ccc;
                border-radius: 3px;
                font-size: 0.9em;
            }
            
            .weight-filter input:focus {
                outline: none;
                border-color: #4CAF50;
            }
            
            .sort-btn, .filter-btn {
                padding: 8px 12px;
                border: 1px solid #ddd;
                border-radius: 4px;
                background-color: #f8f9fa;
                cursor: pointer;
                font-size: 0.9em;
                transition: all 0.2s;
            }
            
            .sort-btn:hover, .filter-btn:hover {
                background-color: #e9ecef;
                border-color: #4CAF50;
            }
            
            .sort-btn.active {
                background-color: #4CAF50;
                color: white;
                border-color: #4CAF50;
            }
            
            .filter-btn.active {
                background-color: #2196F3;
                color: white;
                border-color: #2196F3;
            }
            
            .table-container {
                overflow-x: auto;
                border: 1px solid #ddd;
                border-radius: 4px;
            }
            
            #iterationsTable {
                width: 100%;
                border-collapse: collapse;
                font-size: 0.9em;
            }
            
            #iterationsTable th {
                background-color: #f8f9fa;
                padding: 12px 8px;
                text-align: left;
                border-bottom: 2px solid #ddd;
                font-weight: 600;
                color: #555;
                white-space: normal;
                word-wrap: break-word;
                max-width: 150px;
            }
            
            #iterationsTable th.sortable {
                cursor: pointer;
                user-select: none;
                position: relative;
            }
            
            #iterationsTable th.sortable:hover {
                background-color: #e9ecef;
            }
            
            #iterationsTable th.sortable::after {
                content: ' ↕';
                opacity: 0.5;
            }
            
            #iterationsTable th.sort-asc::after {
                content: ' ↑';
                opacity: 1;
                color: #4CAF50;
            }
            
            #iterationsTable th.sort-desc::after {
                content: ' ↓';
                opacity: 1;
                color: #4CAF50;
            }
            
            #iterationsTable td {
                padding: 10px 8px;
                border-bottom: 1px solid #eee;
                vertical-align: top;
            }
            
            #iterationsTable td:has(.phase-visual) {
                padding-top: 10px;
                padding-bottom: 10px;
            }
            
            #iterationsTable tr:hover {
                background-color: #f8f9fa;
            }
            
            #iterationsTable tr.selected {
                background-color: #e8f5e8;
            }
            
            .phase-names {
                max-width: 200px;
                font-size: 0.8em;
                line-height: 1.3;
            }
            
            .weights {
                font-family: monospace;
                font-size: 0.8em;
            }
            
            .action-btn {
                padding: 4px 8px;
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 3px;
                cursor: pointer;
                font-size: 0.8em;
            }
            
            .action-btn:hover {
                background-color: #45a049;
            }
            
            .status-pass { color: #4CAF50; font-weight: bold; }
            .status-fail { color: #f44336; font-weight: bold; }
            .status-yes { color: #4CAF50; font-weight: bold; }
            .status-no { color: #666; }
            
            .phase-visual {
                width: 120px;
                height: 40px;
                position: relative;
                border: 1px solid #ddd;
                border-radius: 4px;
                overflow: visible;
                cursor: pointer;
                transition: transform 0.2s;
            }
            
            .phase-visual:hover {
                transform: scale(1.05);
                border-color: #4CAF50;
            }
            
            .phase-bar {
                height: 100%;
                float: left;
                position: relative;
                transition: all 0.2s;
                cursor: pointer;
            }
            
            .phase-bar:hover {
                opacity: 0.8;
                transform: scaleY(1.1);
            }
            
            .phase-label {
                position: absolute;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                font-size: 0.7em;
                font-weight: bold;
                color: white;
                text-shadow: 1px 1px 1px rgba(0,0,0,0.5);
                pointer-events: none;
            }
            
            .phase-tooltip {
                position: fixed;
                background-color: rgba(0,0,0,0.95);
                color: white;
                padding: 8px 12px;
                border-radius: 6px;
                font-size: 0.8em;
                z-index: 9999;
                pointer-events: none;
                opacity: 0;
                transition: opacity 0.3s;
                white-space: nowrap;
                max-width: 300px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.4);
                border: 1px solid rgba(255,255,255,0.1);
            }
            
            .pie-chart-container {
                width: 40px;
                height: 40px;
                position: relative;
                cursor: pointer;
            }
            
            .pie-slice {
                position: absolute;
                width: 100%;
                height: 100%;
                border-radius: 50%;
                clip-path: polygon(50% 50%, 50% 0%, 100% 0%, 100% 100%, 0% 100%, 0% 0%);
                transform-origin: 50% 50%;
            }
            
            #plotly-chart {
                width: 100%;
                height: 600px;
            }
            
            .info-panel {
                margin-top: 20px;
                padding: 15px;
                background-color: #f8f9fa;
                border-radius: 5px;
                border-left: 4px solid #4CAF50;
            }
            
            .info-panel h3 {
                margin: 0 0 10px 0;
                color: #333;
            }
            
            .info-panel p {
                margin: 5px 0;
                color: #666;
            }
            
            .loading {
                text-align: center;
                padding: 40px;
                color: #666;
                font-style: italic;
            }
            
            /* Animated loading spinner */
            .loading-spinner {
                display: inline-block;
                width: 20px;
                height: 20px;
                border: 3px solid #f3f3f3;
                border-top: 3px solid #1976d2;
                border-radius: 50%;
                animation: spin 1s linear infinite;
                margin-right: 10px;
            }
            
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            
            .loading-animated {
                text-align: center;
                padding: 40px;
                color: #1976d2;
                background-color: #e3f2fd;
                border: 1px solid #bbdefb;
                border-radius: 8px;
                margin: 20px;
                font-size: 16px;
                display: flex;
                align-items: center;
                justify-content: center;
            }
            
            .error {
                text-align: center;
                padding: 40px;
                color: #f44336;
                background-color: #ffebee;
                border: 1px solid #f5c6cb;
                border-radius: 8px;
                margin: 20px;
            }
            
            .loading {
                text-align: center;
                padding: 40px;
                color: #1976d2;
                background-color: #e3f2fd;
                border: 1px solid #bbdefb;
                border-radius: 8px;
                margin: 20px;
            }
            
            @media (max-width: 768px) {
                .controls {
                    flex-direction: column;
                    align-items: stretch;
                }
                
                .control-group {
                    width: 100%;
                }
                
                .control-group select,
                .control-group input {
                    min-width: auto;
                    width: 100%;
                }
                
                .table-controls {
                    flex-direction: column;
                    align-items: stretch;
                }
            }
        </style>
    </head>
    <body>
        <a href="/" class="back-link">← Back to Dashboard</a>
        
        <div class="container">
            <div class="header">
                <h1>OCx24 XRD Analysis Dashboard</h1>
            </div>
            
            <div class="controls">
                <div class="control-group">
                    <label for="datasetSelect">Dataset:</label>
                    <select id="datasetSelect" onchange="updateDataset()">
                        <option value="">Select dataset...</option>
                    </select>
                </div>
                
                <div class="control-group">
                    <label for="sampleInput">Sample ID:</label>
                    <input type="text" id="sampleInput" list="sampleList" placeholder="Type or select sample ID..." onchange="updateSample()" oninput="updateSample()" onclick="showAllSamples()">
                    <datalist id="sampleList">
                        <option value="">Loading samples...</option>
                    </datalist>
                </div>
                
                <div class="control-group">
                    <label for="fittingSelect">Data:</label>
                    <select id="fittingSelect" onchange="updatePlot()">
                        <option value="original">Original Data</option>
                    </select>
                </div>
            </div>
            
            <div class="plot-container">
                <div id="plotly-chart">
                </div>
            </div>
            
            <div class="metadata-panel">
                <h3>Sample Metadata</h3>
                <div id="metadataContent">
                    <div class="metadata-section">
                        <div id="basicInfo">
                            <div class="loading">Select a sample to view metadata</div>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="iterations-table-panel">
                <h3>Table overview of all fitted XRD fitting solutions</h3>
                <div class="table-controls">
                    <div class="rwp-filter">
                        <label>RWP Max:</label>
                        <input type="number" id="rwpMaxInput" placeholder="e.g., 50" onkeypress="if(event.key==='Enter') applyRwpFilter()">
                        <button class="filter-btn" onclick="applyRwpFilter()">Apply</button>
                    </div>
                    <div class="weight-filter">
                        <label>Major Phase Weight Min:</label>
                        <input type="number" id="weightMinInput" placeholder="e.g., 0.5" step="0.1" min="0" max="1" onkeypress="if(event.key==='Enter') applyWeightFilter()">
                        <button class="filter-btn" onclick="applyWeightFilter()">Apply</button>
                    </div>
                    <button class="filter-btn" onclick="checkPhaseMassBalance()" title="Filter iterations where phase weights match XRF composition within tolerance">Check Phase Mass Balance with XRF</button>
                    <button class="filter-btn" onclick="clearFilters()">Clear Filters</button>
                </div>
                <div class="table-container">
                    <table id="iterationsTable">
                        <thead>
                            <tr>
                                <th onclick="sortTable('xrd_fit_number')" class="sortable">#</th>
                                <th onclick="sortTable('rwp_score')" class="sortable">RWP Score</th>
                                <th onclick="sortTable('number_of_phases_matched')" class="sortable"># Phases</th>
                                <th>Phase Names</th>
                                <th>Weights</th>
                                <th>Visual</th>
                                <th>Action</th>
                            </tr>
                        </thead>
                        <tbody id="iterationsTableBody">
                            <tr><td colspan="7" style="text-align: center; padding: 20px;">Select a dataset and sample to view iterations</td></tr>
                        </tbody>
                    </table>
                </div>
            </div>
        </div>

        <script>
            // Global variables
            let allDatasets = {};
            let currentDataset = '';
            let currentSample = '';
            let currentFitting = 'original';
            let currentTableData = [];
            let currentFilters = {};
            let currentSortColumn = 'rwp_score';
            let currentSortDirection = 'asc';

            // Available datasets - these are the actual filenames in the extracted data folder
            const datasets = [
                'uoft4_240405', 'uoft5_240507', 'uoft6_240624', 'uoft7_240716', 'uoft8_241025', 'uoft9_241025',
                'vsp1_240503', 'vsp10_240726', 'vsp11_240802', 'vsp12_240816', 'vsp13_240823',
                'vsp14_240906', 'vsp15_240809', 'vsp2_240718', 'vsp20_241011', 'vsp23_241025', 'vsp26_241004',
                'vsp27_241030', 'vsp28_240927', 'vsp29_241004', 'vsp3_240718', 'vsp4_240726', 'vsp5_241011',
                'vsp6_240718', 'vsp7_241025', 'vsp8_240802', 'vsp9_240726'
            ];

            // Load all datasets from Cloudflare Worker API
            async function loadAllDatasets() {
                console.log('🔄 Loading ocx24 xrd datasets via Cloudflare Worker API...');
                
                try {
                    // Load metadata via Worker API
                    console.log('📋 Loading metadata via Worker API...');
                    const metadataResponse = await fetch('https://xrd-dashboard-api.ocx24-xrd.workers.dev/api/metadata', {
                        mode: 'cors',
                        headers: {
                            'Accept': 'application/json',
                        }
                    });
                    
                    if (!metadataResponse.ok) {
                        throw new Error(`Failed to load metadata via Worker API: ${metadataResponse.status}`);
                    }
                    
                    // Use streaming JSON parsing for metadata as well
                    const reader = metadataResponse.body.getReader();
                    const decoder = new TextDecoder();
                    let jsonString = '';
                    
                    console.log('🔄 Streaming metadata JSON...');
                    
                    while (true) {
                        const { done, value } = await reader.read();
                        if (done) break;
                        
                        jsonString += decoder.decode(value, { stream: true });
                    }
                    
                    const metadataData = JSON.parse(jsonString);
                    console.log('✅ Metadata loaded successfully via Worker API using streaming');
                    
                    // Initialize datasets with metadata
                    allDatasets = {};
                    Object.keys(metadataData).forEach(dataset => {
                        allDatasets[dataset] = {
                            plotly: null, // Will load on demand
                            metadata: metadataData[dataset]
                        };
                    });
                    
                    console.log(`🎉 Initialized ${Object.keys(allDatasets).length} datasets with metadata`);
                    populateDatasetDropdown();
                    
                    // Show success message
                    document.getElementById('plotly-chart').innerHTML = 
                        '<div class="loading">' +
                        Object.keys(allDatasets).length + ' datasets available<br><br>' +
                        'Select a dataset from the dropdown to load XRD data...</div>';
                    
                } catch (error) {
                    console.error('❌ Error loading data via Worker API:', error);
                    document.getElementById('plotly-chart').innerHTML = 
                        '<div class="error">Error loading data via Worker API<br><br>' +
                        'Please ensure:<br>' +
                        '1. Cloudflare Worker is deployed<br>' +
                        '2. Worker has access to R2 bucket<br>' +
                        '3. API endpoints are working<br><br>' +
                        'Error details: ' + error.message + '</div>';
                }
            }
            
            // Load plotly data for specific dataset on demand using JSON streaming
            async function loadDatasetData(dataset) {
                if (allDatasets[dataset] && allDatasets[dataset].plotly) {
                    return allDatasets[dataset].plotly; // Already loaded
                }
                
                console.log(`📊 Loading plotly data for ${dataset} using JSON streaming...`);
                try {
                    const dataResponse = await fetch(`https://xrd-dashboard-api.ocx24-xrd.workers.dev/api/data/${dataset}`, {
                        mode: 'cors',
                        headers: {
                            'Accept': 'application/json',
                        }
                    });
                    
                    if (!dataResponse.ok) {
                        throw new Error(`Failed to load data for ${dataset}: ${dataResponse.status}`);
                    }
                    
                    // Use streaming JSON parsing for all datasets regardless of size
                    const reader = dataResponse.body.getReader();
                    const decoder = new TextDecoder();
                    let jsonString = '';
                    
                    console.log(`🔄 Streaming JSON data for ${dataset}...`);
                    
                    while (true) {
                        const { done, value } = await reader.read();
                        if (done) break;
                        
                        jsonString += decoder.decode(value, { stream: true });
                    }
                    
                    // Parse the complete JSON string
                    const plotlyData = JSON.parse(jsonString);
                    
                    console.log(`✅ Streamed JSON data loaded for ${dataset}`);
                    
                    // Cache the data
                    if (allDatasets[dataset]) {
                        allDatasets[dataset].plotly = plotlyData;
                    }
                    
                    return plotlyData;
                    
                } catch (error) {
                    console.error(`❌ Error loading streamed data for ${dataset}:`, error);
                    throw error;
                }
            }

            // Populate dataset dropdown
            function populateDatasetDropdown() {
                const datasetSelect = document.getElementById('datasetSelect');
                datasetSelect.innerHTML = '<option value="">Select dataset...</option>';
                
                // Sort datasets properly (numerical order for numbers)
                const sortedDatasets = Object.keys(allDatasets).sort((a, b) => {
                    // Extract the numeric part for proper sorting
                    const aMatch = a.match(/^(uoft|vsp)(\d+)/);
                    const bMatch = b.match(/^(uoft|vsp)(\d+)/);
                    
                    if (aMatch && bMatch) {
                        const aPrefix = aMatch[1];
                        const bPrefix = bMatch[1];
                        const aNum = parseInt(aMatch[2]);
                        const bNum = parseInt(bMatch[2]);
                        
                        // First sort by prefix (uoft before vsp)
                        if (aPrefix !== bPrefix) {
                            return aPrefix.localeCompare(bPrefix);
                        }
                        
                        // Then sort by number
                        return aNum - bNum;
                    }
                    
                    // Fallback to regular string sorting
                    return a.localeCompare(b);
                });
                
                sortedDatasets.forEach(dataset => {
                    const option = document.createElement('option');
                    option.value = dataset;
                    option.textContent = dataset;
                    datasetSelect.appendChild(option);
                });
            }

            // Update dataset selection
            async function updateDataset() {
                const datasetSelect = document.getElementById('datasetSelect');
                currentDataset = datasetSelect.value;
                
                if (currentDataset && allDatasets[currentDataset]) {
                    // Populate sample dropdown and auto-load first sample's Original Data
                    populateSampleDropdown();
                    // populateSampleDropdown() will automatically call updateSample() for the first sample
                    // which will trigger updatePlot() and updateMetadata()
                } else {
                    // Clear everything
                    document.getElementById('sampleList').innerHTML = '<option value="">No dataset selected</option>';
                    document.getElementById('sampleInput').value = '';
                    document.getElementById('fittingSelect').innerHTML = '<option value="original">Original Data</option>';
                    document.getElementById('plotly-chart').innerHTML = '<div class="loading">Select a dataset to view data</div>';
                    document.getElementById('basicInfo').innerHTML = '<div class="loading">Select a dataset and sample to view metadata</div>';
                    document.getElementById('iterationsTableBody').innerHTML = '<tr><td colspan="7" style="text-align: center; padding: 20px;">Select a dataset and sample to view iterations</td></tr>';
                }
            }

            // Populate the sample dropdown using metadata
            function populateSampleDropdown() {
                const sampleList = document.getElementById('sampleList');
                sampleList.innerHTML = '';
                
                if (!currentDataset || !allDatasets[currentDataset] || !allDatasets[currentDataset].metadata) {
                    return;
                }
                
                // Get sample IDs from metadata instead of plotly data
                const samples = Object.keys(allDatasets[currentDataset].metadata).sort();
                
                samples.forEach(sampleId => {
                    const option = document.createElement('option');
                    option.value = sampleId;
                    sampleList.appendChild(option);
                });
                
                if (samples.length > 0) {
                    document.getElementById('sampleInput').value = samples[0];
                    updateSample();
                }
            }

            // Show all samples when dropdown arrow is clicked
            function showAllSamples() {
                const sampleInput = document.getElementById('sampleInput');
                const currentValue = sampleInput.value;
                
                // Temporarily clear the input to show all options
                sampleInput.value = '';
                
                // Restore the value after a short delay to allow the dropdown to appear
                setTimeout(() => {
                    sampleInput.value = currentValue;
                }, 10);
            }

            // Update sample selection
            function updateSample() {
                const sampleInput = document.getElementById('sampleInput');
                const fittingSelect = document.getElementById('fittingSelect');
                
                currentSample = sampleInput.value.trim();
                
                // Validate sample ID
                if (!currentSample) {
                    return;
                }

                // Check if sample exists in current dataset metadata
                if (!currentDataset || !allDatasets[currentDataset] || !allDatasets[currentDataset].metadata || !allDatasets[currentDataset].metadata[currentSample]) {
                    sampleInput.classList.add('invalid');
                    sampleInput.classList.remove('valid');
                    return;
                }

                sampleInput.classList.add('valid');
                sampleInput.classList.remove('invalid');

                // Update fitting options - start with Original Data, load others on demand
                fittingSelect.innerHTML = '<option value="original">Original Data</option>';
                currentFitting = 'original';
                
                // Auto-load and plot the Original Data immediately
                updatePlot();
                updateMetadata();
            }

            // Update the plot
            async function updatePlot() {
                const fittingSelect = document.getElementById('fittingSelect');
                currentFitting = fittingSelect.value;

                if (!currentDataset || !allDatasets[currentDataset] || !currentSample) {
                    document.getElementById('plotly-chart').innerHTML = '<div class="loading">Select a dataset and sample to view plot</div>';
                    return;
                }

                // Load plotly data only when needed for plotting
                if (!allDatasets[currentDataset].plotly) {
                    console.log(`🔄 Loading plotly data for ${currentDataset} (needed for plotting)...`);
                    try {
                        document.getElementById('plotly-chart').innerHTML = '<div class="loading-animated"><div class="loading-spinner"></div>Loading XRD data for plotting...</div>';
                        const plotlyData = await loadDatasetData(currentDataset);
                        allDatasets[currentDataset].plotly = plotlyData;
                        console.log(`✅ Plotly data loaded for ${currentDataset}`);
                        
                        // Populate fitting options now that data is loaded
                        const plotlyDataForSample = plotlyData[currentSample];
                        if (plotlyDataForSample && plotlyDataForSample.files && plotlyDataForSample.files.iterations) {
                            Object.keys(plotlyDataForSample.files.iterations).sort((a, b) => parseInt(a) - parseInt(b)).forEach(iterNum => {
                                const option = document.createElement('option');
                                option.value = iterNum;
                                option.textContent = `Fitted XRD ${iterNum}`;
                                fittingSelect.appendChild(option);
                            });
                        }
                    } catch (error) {
                        console.error(`❌ Error loading plotly data for ${currentDataset}:`, error);
                        document.getElementById('plotly-chart').innerHTML = 
                            '<div class="error">Error loading XRD data for plotting<br><br>' +
                            'Error details: ' + error.message + '</div>';
                        return;
                    }
                }

                const plotlyData = allDatasets[currentDataset].plotly[currentSample];
                let traces = [];
                let layout = {
                    title: `${currentSample} - ${currentFitting === 'original' ? 'Original Data' : `Fitted XRD ${currentFitting}`}`,
                    xaxis: { title: '2θ (degrees)' },
                    yaxis: { title: 'Intensity' },
                    showlegend: true,
                    legend: { x: 1.02, y: 1 },
                    margin: { r: 150 }
                };

                if (currentFitting === 'original') {
                    // Show original XRD data
                    const xrdData = plotlyData.files.xrd;
                    traces = xrdData.traces.map(trace => {
                        const traceName = trace.name ? trace.name.toLowerCase() : '';
                        const isNormalized = traceName.includes('normalized');
                        const isExtractedPeaks = traceName.includes('peak') || traceName.includes('extracted');
                        
                        const plotlyTrace = {
                            x: trace.x || [],
                            y: trace.y || [],
                            type: isExtractedPeaks ? 'bar' : 'scatter',
                            mode: isExtractedPeaks ? undefined : 'lines',
                            name: trace.name || 'unnamed',
                            visible: isNormalized || isExtractedPeaks ? true : 'legendonly'
                        };
                        
                        // Add styling based on trace type
                        if (isExtractedPeaks) {
                            plotlyTrace.marker = { size: 4 };
                        } else {
                            plotlyTrace.line = { width: 2 };
                        }
                        
                        return plotlyTrace;
                    });
                    
                } else {
                    // Show normalized original + iteration data
                    const xrdData = plotlyData.files.xrd;
                    const iterData = plotlyData.files.iterations[currentFitting];
                    
                    // Find normalized trace from original data by name
                    const normalizedTrace = xrdData.traces.find(trace => 
                        trace.name && trace.name.toLowerCase().includes('normalized')
                    );
                    
                    if (normalizedTrace) {
                        traces.push({
                            x: normalizedTrace.x,
                            y: normalizedTrace.y,
                            type: 'scatter',
                            mode: 'lines',
                            name: 'normalized xrd',
                            line: { width: 2, color: '#1f77b4' },
                            visible: true
                        });
                    }
                    
                    // Add iteration traces
                    iterData.traces.forEach(trace => {
                        const traceName = trace.name ? trace.name.toLowerCase() : '';
                        const isCalc = traceName === 'calc';
                        const isBackground = traceName.includes('background');
                        const isNormalized = traceName.includes('normalized');
                        const hasZeroWeight = traceName.includes('wt=0.0');
                        
                        // Determine trace type and visibility
                        let traceType = 'scatter';
                        let traceMode = 'lines';
                        let isVisible = true;
                        
                        if (isBackground) {
                            isVisible = 'legendonly'; // Background invisible by default
                        } else if (hasZeroWeight) {
                            isVisible = 'legendonly'; // Zero weight traces invisible by default
                        } else if (!isCalc && !isNormalized) {
                            traceType = 'bar'; // Everything else as bars
                            traceMode = undefined;
                        }
                        
                        const plotlyTrace = {
                            x: trace.x || [],
                            y: trace.y || [],
                            type: traceType,
                            mode: traceMode,
                            name: trace.name || 'unnamed',
                            visible: isVisible
                        };
                        
                        // Add appropriate styling
                        if (traceType === 'bar') {
                            plotlyTrace.marker = { size: 4 };
                        } else {
                            plotlyTrace.line = { width: 2 };
                        }
                        
                        traces.push(plotlyTrace);
                    });
                }

                // Clear any loading animation before rendering the plot
                document.getElementById('plotly-chart').innerHTML = '';
                Plotly.newPlot('plotly-chart', traces, layout, {
                    responsive: true,
                    toImageButtonOptions: {
                        format: 'png',
                        filename: 'xrd_analysis_plot',
                        height: 800,
                        width: 1200,
                        scale: 3
                    }
                });
            }

            // Update metadata display
            function updateMetadata() {
                if (!currentDataset || !allDatasets[currentDataset] || !allDatasets[currentDataset].metadata || !currentSample) {
                    document.getElementById('basicInfo').innerHTML = '<div class="loading">Select a dataset and sample to view metadata</div>';
                    return;
                }

                const sampleMetadata = allDatasets[currentDataset].metadata[currentSample];
                if (!sampleMetadata) {
                    document.getElementById('basicInfo').innerHTML = '<div class="loading">No metadata available for this sample</div>';
                    return;
                }

                updateBasicInfo(sampleMetadata);
                updateIterationsTable(sampleMetadata);
            }

            // Update basic info section
            function updateBasicInfo(sampleMetadata) {
                const basicInfo = document.getElementById('basicInfo');
                
                basicInfo.innerHTML = `
                    <div class="metadata-item">
                        <span class="metadata-label">Target Composition:</span>
                        <span class="metadata-value">${sampleMetadata.target_composition || 'N/A'}</span>
                    </div>
                    <div class="metadata-item">
                        <span class="metadata-label">XRF Composition:</span>
                        <span class="metadata-value">${sampleMetadata.xrf_composition || 'N/A'}</span>
                    </div>
                `;
            }

            // Update iterations table
            function updateIterationsTable(sampleMetadata) {
                if (!sampleMetadata.iterations || sampleMetadata.iterations.length === 0) {
                    document.getElementById('iterationsTableBody').innerHTML = 
                        '<tr><td colspan="7" style="text-align: center; padding: 20px;">No data available</td></tr>';
                    return;
                }
                
                currentTableData = [...sampleMetadata.iterations];
                
                // Set default sort by RWP score (ascending - best scores first)
                currentSortColumn = 'rwp_score';
                currentSortDirection = 'asc';
                
                renderTable();
            }

            // Render the iterations table
            function renderTable() {
                const tbody = document.getElementById('iterationsTableBody');
                let filteredData = [...currentTableData];
                
                // Apply filters
                filteredData = filteredData.filter(iteration => {
                    for (const [key, value] of Object.entries(currentFilters)) {
                        if (key === 'rwp_max') {
                            // Special handling for RWP maximum filter
                            if (iteration.rwp_score !== undefined && iteration.rwp_score > value) {
                                return false;
                            }
                        } else if (key === 'weight_min') {
                            // Special handling for weight minimum filter (major phase detection)
                            if (iteration.weights && iteration.weights.length > 0) {
                                const hasMajorPhase = iteration.weights.some(weight => weight >= value);
                                if (!hasMajorPhase) {
                                    return false;
                                }
                            } else {
                                // No weights data, filter out
                                return false;
                            }
                        } else {
                            // Standard filter handling
                            if (iteration[key] !== value) {
                                return false;
                            }
                        }
                    }
                    return true;
                });
                
                // Apply sorting
                if (currentSortColumn) {
                    filteredData.sort((a, b) => {
                        const aVal = a[currentSortColumn];
                        const bVal = b[currentSortColumn];
                        
                        if (aVal === undefined && bVal === undefined) return 0;
                        if (aVal === undefined) return 1;
                        if (bVal === undefined) return -1;
                        
                        if (typeof aVal === 'number' && typeof bVal === 'number') {
                            return currentSortDirection === 'asc' ? aVal - bVal : bVal - aVal;
                        }
                        
                        const aStr = String(aVal).toLowerCase();
                        const bStr = String(bVal).toLowerCase();
                        return currentSortDirection === 'asc' ? aStr.localeCompare(bStr) : bStr.localeCompare(aStr);
                    });
                }
                
                // Generate HTML
                let html = '';
                if (filteredData.length === 0) {
                    html = '<tr><td colspan="7" style="text-align: center; padding: 20px;">No iterations match the current filters</td></tr>';
                } else {
                    filteredData.forEach(iteration => {
                        const isSelected = iteration.xrd_fit_number == currentFitting;
                        html += `
                            <tr class="${isSelected ? 'selected' : ''}" data-iteration="${iteration.xrd_fit_number}">
                                <td>${iteration.xrd_fit_number || 'N/A'}</td>
                                <td class="rwp-score ${getRwpClass(iteration.rwp_score)}">${iteration.rwp_score || 'N/A'}</td>
                                <td>${iteration.number_of_phases_matched || 'N/A'}</td>
                                <td class="phase-names">${iteration.phases_matched ? iteration.phases_matched.join(', ') : 'N/A'}</td>
                                <td class="weights">${iteration.weights ? '[' + iteration.weights.join(', ') + ']' : 'N/A'}</td>
                                <td>${generatePhaseVisual(iteration)}</td>
                                <td><button class="action-btn" onclick="selectIteration(${iteration.xrd_fit_number})">Plot</button></td>
                            </tr>
                        `;
                    });
                }
                
                tbody.innerHTML = html;
            }

            // Get RWP class for styling
            function getRwpClass(rwpScore) {
                if (rwpScore === 'N/A' || rwpScore === undefined) return '';
                if (rwpScore <= 50) return 'rwp-good';
                if (rwpScore <= 100) return 'rwp-medium';
                return 'rwp-poor';
            }

            // Generate phase visual
            function generatePhaseVisual(iteration) {
                if (!iteration.phases_matched || !iteration.weights || 
                    iteration.phases_matched.length === 0 || iteration.weights.length === 0) {
                    return '<div class="phase-visual" style="background-color: #f0f0f0; display: flex; align-items: center; justify-content: center; font-size: 0.8em; color: #666;">No Data</div>';
                }
                
                // Generate colors for phases
                const colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9'];
                
                let visualHtml = '<div class="phase-visual">';
                
                // Calculate total weight for normalization
                const totalWeight = iteration.weights.reduce((sum, weight) => sum + Math.abs(weight), 0);
                
                if (totalWeight === 0) {
                    visualHtml += '<div style="width: 100%; height: 100%; background-color: #f0f0f0; display: flex; align-items: center; justify-content: center; font-size: 0.8em; color: #666;">Zero Weights</div>';
                } else {
                    iteration.phases_matched.forEach((phase, index) => {
                        const weight = Math.abs(iteration.weights[index] || 0);
                        const percentage = (weight / totalWeight) * 100;
                        const color = colors[index % colors.length];
                        
                        if (percentage > 0) {
                            visualHtml += `
                                <div class="phase-bar" 
                                     style="width: ${percentage}%; background-color: ${color};"
                                     onmouseenter="showTooltip(event, '${phase}', '${percentage.toFixed(1)}%')"
                                     onmouseleave="hideTooltip()">
                                    ${percentage > 15 ? `<div class="phase-label">${(percentage).toFixed(0)}%</div>` : ''}
                                </div>
                            `;
                        }
                    });
                }
                
                visualHtml += '</div>';
                return visualHtml;
            }

            // Sort table
            function sortTable(column) {
                if (currentSortColumn === column) {
                    currentSortDirection = currentSortDirection === 'asc' ? 'desc' : 'asc';
                } else {
                    currentSortColumn = column;
                    currentSortDirection = 'asc';
                }
                renderTable();
            }

            // Apply RWP filter
            function applyRwpFilter() {
                const rwpMaxInput = document.getElementById('rwpMaxInput');
                const value = parseFloat(rwpMaxInput.value);
                
                if (!isNaN(value) && value > 0) {
                    currentFilters.rwp_max = value;
                } else {
                    delete currentFilters.rwp_max;
                }
                
                renderTable();
            }

            // Apply weight filter
            function applyWeightFilter() {
                const weightMinInput = document.getElementById('weightMinInput');
                const value = parseFloat(weightMinInput.value);
                
                if (!isNaN(value) && value >= 0 && value <= 1) {
                    currentFilters.weight_min = value;
                } else {
                    delete currentFilters.weight_min;
                }
                
                renderTable();
            }

            // Check phase mass balance
            function checkPhaseMassBalance() {
                currentFilters.xrf_mass_balance_check = true;
                renderTable();
            }

            // Clear all filters
            function clearFilters() {
                currentFilters = {};
                document.getElementById('rwpMaxInput').value = '';
                document.getElementById('weightMinInput').value = '';
                renderTable();
            }

            // Select iteration for plotting
            function selectIteration(iterationNumber) {
                const fittingSelect = document.getElementById('fittingSelect');
                fittingSelect.value = iterationNumber;
                currentFitting = iterationNumber;
                updatePlot();
            }

            // Tooltip functions for phase visuals
            function showTooltip(event, phase, percentage) {
                // Remove any existing tooltip
                hideTooltip();
                
                // Extract short form of phase name (first part before underscore)
                const shortPhase = phase.split('_')[0];
                
                // Create tooltip element
                const tooltip = document.createElement('div');
                tooltip.className = 'phase-tooltip';
                tooltip.innerHTML = `${shortPhase}<br>${percentage}`;
                document.body.appendChild(tooltip);
                
                // Position tooltip relative to mouse position
                const rect = event.target.getBoundingClientRect();
                const tooltipRect = tooltip.getBoundingClientRect();
                
                let left = rect.right + 8;
                let top = rect.top + (rect.height / 2) - (tooltipRect.height / 2);
                
                // Adjust if tooltip would go off screen
                if (left + tooltipRect.width > window.innerWidth) {
                    left = rect.left - tooltipRect.width - 8;
                }
                if (top < 0) {
                    top = 8;
                }
                if (top + tooltipRect.height > window.innerHeight) {
                    top = window.innerHeight - tooltipRect.height - 8;
                }
                
                tooltip.style.left = left + 'px';
                tooltip.style.top = top + 'px';
                tooltip.style.opacity = '1';
            }
            
            function hideTooltip() {
                const existingTooltip = document.querySelector('.phase-tooltip');
                if (existingTooltip) {
                    existingTooltip.remove();
                }
            }

            // Initialize the dashboard
            document.addEventListener('DOMContentLoaded', async function() {
                try {
                    await loadAllDatasets();
                    
                    // Check for URL parameters
                    const urlParams = new URLSearchParams(window.location.search);
                    const datasetParam = urlParams.get('dataset');
                    const sampleParam = urlParams.get('sample');
                    
                    if (datasetParam && sampleParam) {
                        // Set dataset selection
                        const datasetSelect = document.getElementById('datasetSelect');
                        datasetSelect.value = datasetParam;
                        
                        // Update dataset and load samples
                        await updateDataset();
                        
                        // Set sample input
                        const sampleInput = document.getElementById('sampleInput');
                        sampleInput.value = sampleParam;
                        
                        // Load the specific sample
                        await updateSample();
                    }
                } catch (error) {
                    console.error('Error initializing dashboard:', error);
                    alert('Error loading datasets: ' + error.message);
                }
            });
        </script>
    </body>
    </html>
    '''
    
    return html_template
