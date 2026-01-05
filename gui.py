import streamlit as st
import tempfile
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
import plotly.graph_objects as go
from scipy.interpolate import griddata, Rbf
from scipy.ndimage import gaussian_filter
import os
import requests
import io
import zipfile
import tempfile
import urllib.request
from datetime import datetime
import xarray as xr
import math
import time
from numba import jit, prange

# ==============================
# App Configuration & Header
# ==============================
st.set_page_config(page_title="Geoid Corrections Pro", layout="wide")

# Initialize session states
if 'show_help' not in st.session_state:
    st.session_state.show_help = False
if 'current_section' not in st.session_state:
    st.session_state.current_section = "Data Acquisition"
if 'interpolated_data' not in st.session_state:
    st.session_state.interpolated_data = {}
if 'geoid_correction_results' not in st.session_state:
    st.session_state.geoid_correction_results = {}
if 'df_crust' not in st.session_state:
    st.session_state.df_crust = None
if 'df_sed' not in st.session_state:
    st.session_state.df_sed = None
if 'df_topo' not in st.session_state:
    st.session_state.df_topo = None
if 'df_geoid' not in st.session_state:
    st.session_state.df_geoid = None

# ==============================
# SIDEBAR NAVIGATION (Petrel-like interface)
# ==============================
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; margin-bottom: 1rem;">
        <img src="app/static/geoid_header.png" alt="Geoid Model"
             style="width: 100%; max-height: 120px; object-fit: cover;
                    border-radius: 8px; margin-bottom: 0.5rem;">
        <h2 style="margin: 0.5rem 0;">🌍 Geoid Corrections</h2>
        <p style="font-size: 0.9rem; color: #666; margin: 0;">Developed by Rajkumar Mondal</p>
    </div>
    """, unsafe_allow_html=True)

    
    # Main navigation buttons
    nav_options = {
        "📥 Data Acquisition": "Data Acquisition",
        "📁 Data Upload": "Data Upload", 
        "📊 Data Analysis": "Data Analysis",
        "🗺️ Data Visualization": "Data Visualization",
        "🔧 Geoid Corrections": "Geoid Corrections",
        "📈 Draw Profiling Line": "Draw Profiling Line"
        #"⚖️ Uncertainty Analysis": "Uncertainty Analysis"
    }
    
    for icon_label, section in nav_options.items():
        if st.button(icon_label, use_container_width=True, 
                    type="primary" if st.session_state.current_section == section else "secondary"):
            st.session_state.current_section = section
            st.rerun()
    
    st.markdown("---")
    
    # Help section
    if st.button("❓ Help Guide", use_container_width=True):
        st.session_state.show_help = not st.session_state.show_help
        st.rerun()
    
    # Status indicators
    st.markdown("### 📊 Data Status")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Crustal Data", "✅" if st.session_state.df_crust is not None else "❌")
        st.metric("Topography", "✅" if st.session_state.df_topo is not None else "❌")
    with col2:
        st.metric("Sedimentary", "✅" if st.session_state.df_sed is not None else "❌")
        st.metric("Geoid", "✅" if st.session_state.df_geoid is not None else "❌")
    
    st.metric("Interpolated Sets", len(st.session_state.interpolated_data))
    st.metric("Corrections", len(st.session_state.geoid_correction_results))

# ==============================
# HELP PAGE
# ==============================
if st.session_state.show_help:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 10px; margin-bottom: 2rem;">
        <h1 style="color: white; margin: 0; text-align: center;">❓ Comprehensive Help Guide</h1>
        <p style="color: #f0f0f0; text-align: center; font-size: 1.1rem; margin-top: 0.5rem;">
            Complete User Manual for Geoid Data Corrections Interface
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Back to Main App Button
    if st.button("⬅ Back to Main Application", type="primary"):
        st.session_state.show_help = False
        st.rerun()

    st.markdown("""
    ## 📖 Table of Contents

    1. [Application Overview](#application-overview)
    2. [Quick Start Guide](#quick-start-guide)
    3. [Data Acquisition](#data-acquisition)
    4. [Data Upload](#data-upload)
    5. [Data Distribution Analysis](#data-distribution-analysis)
    6. [Data Visualization](#data-visualization)
    7. [Geoid Corrections](#geoid-corrections)
    8. [Troubleshooting](#troubleshooting)
    9. [Best Practices](#best-practices)

    ---

    ## 🎯 Application Overview

    The **Geoid Data Corrections Interface** is a comprehensive web application designed for geophysicists, geodesists, and researchers working with gravity field and geoid data. This tool enables you to:

    - **Upload and process** multiple geospatial data formats
    - **Analyze data distributions** and detect outliers
    - **Visualize data** with advanced interpolation and plotting
    - **Apply geoid corrections** using tesseroid methods
    - **Generate publication-quality** maps and figures

    ### Key Features:
    - 📥 **Multi-format support**: CSV, GeoTIFF, NetCDF, GDF
    - 📊 **Statistical analysis**: Outlier detection, distribution analysis
    - 🗺️ **Advanced visualization**: Interactive plots with contour lines
    - 🔧 **Physical corrections**: Topographic, crustal, sedimentary corrections
    - 💾 **Export capabilities**: High-resolution images and data files

    ---

    ## 🚀 Quick Start Guide

    ### Step-by-Step Workflow:

    1. **Data Preparation** (5-10 minutes)
       - Download required data from official sources
       - Ensure proper format: CSV with longitude, latitude, value columns
       - Verify units consistency (meters for elevation, kilometers for thickness)

    2. **Data Upload** (2-5 minutes)
       - Upload your four main datasets:
         - Crustal Thickness
         - Sedimentary Thickness  
         - Topographic Data
         - Geoid Data

    3. **Quality Control** (5-15 minutes)
       - Use Data Distribution Analysis to check data quality
       - Identify and remove outliers
       - Verify spatial coverage

    4. **Visualization** (3-8 minutes)
       - Generate maps of each dataset
       - Adjust interpolation settings
       - Customize plot appearance

    5. **Correction Computation** (10-60 minutes)
       - Select correction type
       - Set physical parameters
       - Run computation
       - Review results

    ---

    ## 📥 Data Acquisition

    ### Required Datasets:

    #### 1. Crustal Thickness Data
    - **Source**: IRIS CRUST1.0
    - **Resolution**: 1°×1° global
    - **Format**: ASCII/text files
    - **Parameters**: Moho depth, crustal layers
    - **Download**: [IRIS Database](https://www.earthcrustmodel1.com/)

    #### 2. Sedimentary Thickness Data  
    - **Source**: GlobSed Model
    - **Resolution**: 5 arc-minutes
    - **Format**: NetCDF, ASCII grid
    - **Parameters**: Total sediment thickness
    - **Download**: [NOAA NGDC](https://www.ngdc.noaa.gov/mgg/sedthick/)

    #### 3. Topographic Data
    - **Source**: ETOPO1
    - **Resolution**: 1 arc-minute
    - **Format**: GeoTIFF, NetCDF
    - **Parameters**: Elevation/bathymetry
    - **Download**: [NOAA ETOPO1](https://www.ngdc.noaa.gov/mgg/global/)

    #### 4. Geoid Data
    - **Source**: EGM2008
    - **Resolution**: 2.5 arc-minutes
    - **Format**: GDF, ASCII grid
    - **Parameters**: Geoid height
    - **Download**: [ICGEM](http://icgem.gfz-potsdam.de/EGM2008)

    ### Data Format Requirements:
    ```
    longitude,latitude,value
    -120.5,45.2,35.6
    -120.0,45.2,34.8
    -119.5,45.2,33.9
    ```

    ---

    ## 📁 Data Upload

    ### Supported File Formats:

    | Format | Extensions | Best For |
    |--------|------------|----------|
    | CSV | `.csv` | Point data, easy editing |
    | GeoTIFF | `.tif`, `.tiff` | Raster data, GIS compatibility |
    | NetCDF | `.nc` | Gridded data, metadata rich |
    | ASCII Grid | `.grd`, `.asc` | Simple grid formats |
    | GDF | `.gdf` | Geodetic data formats |

    ### Upload Tips:
       **crustal thickness:csv file**\n
       **sedimentary thickness:csv file**\n
       **Topographic thickness:NetCDF(.nc) file**\n
       **Geoid: .gdf file**

    - **Column Names**: The app automatically detects `longitude`, `latitude`, and `value` columns
    - **Units**: 
      - Thickness data: kilometers (will auto-convert if needed)
      - Elevation/Geoid: meters
    - **Coordinate System**: WGS84 (longitude: -180 to 180, latitude: -90 to 90)
    - **Data Quality**: Remove NaN values before upload for best results

    ### Common Upload Issues:

    ❌ **Problem**: "No numeric columns found"
    ✅ **Solution**: Ensure your CSV has numeric values in the data columns

    ❌ **Problem**: "Could not auto-detect coordinate columns"  
    ✅ **Solution**: Rename columns to include 'lon', 'lat', 'x', 'y' keywords

    ❌ **Problem**: File format not supported
    ✅ **Solution**: Convert to CSV using QGIS, GDAL, or Python

    ---

    ## 📊 Data Distribution Analysis

    This section helps you understand your data quality and identify potential issues.

    ### Key Features:

    #### 1. Outlier Detection Methods:
    - **Standard Deviation**: Simple ±nσ method
    - **Isolation Forest**: Machine learning approach
    - **Minimum Covariance Determinant**: Robust statistical method
    - **Local Outlier Factor**: Density-based detection
    - **One-Class SVM**: Advanced ML technique

    #### 2. Visualization Options:
    - **Histograms**: With KDE curves
    - **Violin Plots**: Distribution shape and quartiles
    - **Box Plots**: Traditional statistical view
    - **Spatial Maps**: Before/after comparison

    ### Recommended Workflow:

    1. **Start with Standard Deviation** method (3σ)
    2. **Check spatial distribution** of outliers
    3. **Compare multiple methods** if uncertain
    4. **Document removed points** for reproducibility

    ### Interpretation Guide:

    - **Normal Distribution**: Data is well-behaved
    - **Heavy Tails**: Consider robust methods
    - **Spatial Clustering**: May indicate regional features
    - **Random Outliers**: Likely measurement errors

    ---

    ## 🗺️ Data Visualization

    Create professional-quality maps for analysis and publication.

    ### Interpolation Methods:

    | Method | Speed | Accuracy | Best For |
    |--------|-------|----------|----------|
    | Nearest | ⚡ Fast | 🟡 Medium | Large datasets, categorical data |
    | Linear | ⚡⚡ Fast | 🟢 Good | Smooth continuous fields |
    | Cubic | ⚡⚡⚡ Medium | 🟢🟢 Better | High-quality visualization |
    | RBF | ⚡⚡⚡⚡ Slow | 🟢🟢🟢 Best | Irregular spacing, precision |

    ### Plot Customization:

    #### Color Schemes:
    - **Jet**: Classic geophysics colors
    - **Viridis**: Modern, perceptually uniform
    - **RdBu**: Diverging for corrections
    - **Terrain**: Topographic data

    #### Enhancement Features:
    - **Percentile Clipping**: Remove extremes for better color range
    - **Smoothing**: Gaussian filter for cleaner appearance
    - **Contour Lines**: Add isolines for precise reading
    - **Data Points**: Overlay original measurements

    ### Publication Tips:

    - Use **300+ DPI** for journal submissions
    - **Black boundary boxes** meet most journal requirements
    - **Vector formats (SVG)** for best quality
    - Include **scale bar and north arrow** in final figures

    ---

    ## 🔧 Geoid Corrections

    Apply physical corrections to geoid data using tesseroid methods.

    ### Available Corrections:

    1. **Topographic Correction Only**
       - Accounts for topography and bathymetry
       - Uses rock/water density contrast

    2. **Crustal Thickness Correction Only**  
       - Corrects for Moho depth variations
       - Uses crust-mantle density contrast

    3. **Sedimentary Correction Only**
       - Accounts for sediment basin effects
       - Uses sediment-crust density contrast

    4. **Combined Correction (All Three)**
       - Comprehensive correction
       - Most physically complete

    5. **Residual Geoid**
       - Original minus all corrections
       - Reveals deep Earth structure

    ### Physical Parameters:

    #### Density Values (kg/m³):
    - **Rock Density**: 2600-2800 (default: 2670)
    - **Water Density**: 1000-1100 (default: 1030)  
    - **Crust Density**: 2800-3100 (default: 3000)
    - **Mantle Density**: 3200-3400 (default: 3300)

    #### Reference Values:
    - **Crustal Thickness**: 30-50 km (default: 43 km)
    - **Angular Cutoff**: 1-20° (affects computation speed)

    ### Computation Details:

    #### Tesseroid Method:
    - **3D integration** using spherical prisms
    - **Numerical optimization** for speed
    - **Parallel processing** for large datasets

    #### Performance Tips:
    - Start with **coarser grid** for testing
    - Use **larger angular cutoff** for faster computation
    - **Batch processing** for memory management

    ### Result Interpretation:

    - **Positive Correction**: Mass deficit in the Earth
    - **Negative Correction**: Mass excess in the Earth  
    - **Large Corrections**: Significant density contrasts
    - **Small Corrections**: Homogeneous structure

    ---

    ## 🔧 Troubleshooting

    ### Common Issues and Solutions:

    ❌ **Problem**: "No interpolated data found"
    ✅ **Solution**: Generate plots in Data Visualization section first

    ❌ **Problem**: Computation takes too long
    ✅ **Solution**: Reduce grid resolution, increase angular cutoff

    ❌ **Problem**: Memory errors with large datasets
    ✅ **Solution**: Use smaller regions, increase batch size

    ❌ **Problem**: Colors don't show variation
    ✅ **Solution**: Enable percentile clipping, adjust color range

    ❌ **Problem**: Contour lines not visible
    ✅ **Solution**: Increase contour step, check data range

    ### Performance Optimization:

    - **Dataset Size**: < 1 million points recommended
    - **Grid Resolution**: 100-500 pixels per axis
    - **Memory**: 8GB+ RAM for large computations
    - **Computation Time**: 5-60 minutes depending on settings

    ---

    ## 💡 Best Practices

    ### Data Management:

    1. **Backup Original Data**: Always keep raw data files
    2. **Document Parameters**: Record all correction settings
    3. **Version Control**: Save different correction results
    4. **Metadata**: Document data sources and processing steps

    ### Scientific Workflow:

    1. **Quality Control** before correction
    2. **Start Simple** with single corrections
    3. **Validate Results** with known benchmarks
    4. **Sensitivity Analysis** test parameter variations

    ### Publication Ready:

    - Use **consistent color schemes** across figures
    - Include **error estimates** where possible
    - Provide **scale information** for all maps
    - Document **computation parameters** in methods

    ### Advanced Tips:

    - **Regional Studies**: Use higher resolution data
    - **Global Studies**: Ensure data coverage
    - **Comparative Studies**: Use identical parameters
    - **Method Validation**: Compare with independent methods

    ---

    ## 📞 Support and Contact

    For technical support, bug reports, or feature requests:

    - **Developer**: Rajkumar Mondal, Chandra Prakash Dubey
    - **Laboratory**: LithoSphereX Lab  
    - **Contact**: rajkumarmondal691@gmail.com, p.dubey48@gmail.com
    - **Version**: 1.0 (November 2025)

    ### How to Cite:

    If you use this application in your research, please cite:

    ```
    Mondal, R. (2025). Geoid Data Corrections Interface [Computer software]. 
    LithoSphereX Lab. https://github.com/your-repository
    ```

    ---

    ## 🎓 Tutorial Videos

    Coming soon:
    - Basic workflow demonstration
    - Advanced correction techniques  
    - Publication figure preparation
    - Troubleshooting common issues

    *Check back regularly for updates and new features!*
    """)

    # Back to Top Button
    st.markdown("[⬆ Back to Top](#comprehensive-help-guide)")

    

    # Don't show the main app when help is active
    st.stop()
   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

# ==============================
# MAIN APPLICATION HEADER
# ==============================
st.markdown(f"""
<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.5rem; border-radius: 10px; margin-bottom: 1rem;">
    <h1 style="color: white; margin: 0; text-align: center;">🌍 {st.session_state.current_section}</h1>
    <p style="color: #f0f0f0; text-align: center; font-size: 1rem; margin-top: 0.5rem;">
        Geoid Data Corrections Interface; Developed by Rajkumar Mondal (2025)
    </p>
</div>
""", unsafe_allow_html=True)



# ==============================
# SECTION 1: DATA ACQUISITION
# ==============================
if st.session_state.current_section == "Data Acquisition":
    st.header("📥 Data Acquisition Guide")
    st.markdown("""
    Comprehensive guidance for obtaining crustal thickness, sedimentary thickness, topography, and geoid data from official global databases.
    Download the required data files and upload them in the Data Upload section.
    """)

    with st.expander("🔍 Data Acquisition Help & Resources", expanded=True):
        # [Keep all the data acquisition content from original...]
        st.markdown("### 🌐 Official Data Sources & Download Guidelines")
        
        # Crustal Thickness Section
        st.markdown("---")
        st.markdown("#### 🏔️ **Crustal Thickness Data**")
        
        col_crust1, col_crust2 = st.columns([2, 1])
        
        with col_crust1:
            st.markdown("""
            **Primary Source:** IRIS (Incorporated Research Institutions for Seismology)  
            **Model:** CRUST1.0 - 1°×1° global crustal model  
            **Format:** Text files (ASCII), NetCDF  
            **Parameters:** Moho depth, crustal layers thickness  
            
            **📥 Download Steps:**
            1. Visit [IRIS CRUST1.0 Database](https://www.earthcrustmodel1.com/)
            2. Download the global model or regional extracts
            3. Look for files: Total Thickness - Earth Crustal Model 1 (ECM1) 
            
            **File Format Example (CSV):**
            ```
            longitude,latitude,crustal_thickness_km
            -120.5,45.2,35.6
            -120.0,45.2,34.8
            ...
            ```
            """)
        
        with col_crust2:
            st.markdown("""
            **Quick Links:**
            - [🌐 IRIS EMC](https://ds.iris.edu/ds/products/emc/)
            - [📁 CRUST1.0 Download](https://ds.iris.edu/ds/products/emc-crust1.0/)
            - [📚 Documentation](https://doi.org/10.17611/DP/9991885)
            - [🔧 Data Tools](https://ds.iris.edu/ds/products/emc-tools/)
        
            **Alternative Sources:**
            - [USGS Crustal Models](https://www.usgs.gov/natural-hazards/earthquake-hazards/crustal-model)
            - [EUROPEAN CRUST](https://www.ecrust.org/)
            """)
    
    # Sedimentary Thickness Section
    st.markdown("---")
    st.markdown("#### 🏗️ **Sedimentary Thickness Data (GlobSed Model)**")
    
    col_sed1, col_sed2 = st.columns([2, 1])
    
    with col_sed1:
        st.markdown("""
        **Primary Source:** NOAA National Centers for Environmental Information  
        **Model:** GlobSed - Global sediment thickness  
        **Format:** Text files (ASCII), Grid formats  
        **Parameters:** Total sediment thickness in meters
        
        **📥 Download Steps:**
        1. Visit https://www.earthcrustmodel1.com/
        2. Download the global sediment thickness grid
        3. Available formats: txt
        4. 
        
        **File Format Example (CSV):**
        ```
        longitude,latitude,sedimentary_thickness_km
        -120.5,45.2,1.2
        -120.0,45.2,0.8
        ...
        ```
        """)
    
    with col_sed2:
        st.markdown("""
        **Quick Links:**
        - [🌐 GlobSed Main Page](https://www.ngdc.noaa.gov/mgg/sedthick/)
        - [📁 Direct Download](https://www.ngdc.noaa.gov/mgg/sedthick/data/version3/GlobSed_v3.nc)
        - [📚 Publication](https://doi.org/10.1029/2018GC008115)
        - [🔧 Data Viewer](https://www.ngdc.noaa.gov/mgg/sedthick/sedthick.html)
        
        **Alternative Sources:**
        - [Total Sediment Thickness of the World's Oceans](https://www.ngdc.noaa.gov/mgg/sedthick/)
        - [CRUST1.0 Sediment Layer](http://ds.iris.edu/ds/products/emc-crust1.0/)
        """)
    
    # Topography Section
    st.markdown("---")
    st.markdown("#### ⛰️ **Topographic/Bathymetric Data (ETOPO1)**")
    
    col_topo1, col_topo2 = st.columns([2, 1])
    
    with col_topo1:
        st.markdown("""
        **Primary Source:** NOAA National Centers for Environmental Information  
        **Model:** ETOPO1 - 1 arc-minute global relief model  
        **Format:** GeoTIFF, NetCDF, ASCII Grid, XYZ  
        **Parameters:** Elevation (m) relative to sea level
        
        **📥 Download Steps:**
        1. Visit [NOAA ETOPO1 Portal](https://www.ngdc.noaa.gov/mgg/global/)
        2. Choose regional or global download
        3. Select format: NetCDF (.nc) recommended for this GUI
        4. For this app: Convert to CSV with lon, lat, elevation
        
        **Supported Formats:**
        - NetCDF (.nc)
        - ASCII Grid (.asc)
        - XYZ text files (.xyz)
        """)
    
    with col_topo2:
        st.markdown("""
        **Quick Links:**
        - [🌐 ETOPO1 Main Page](https://www.ngdc.noaa.gov/mgg/global/)
        - [📁 Direct Download](https://www.ngdc.noaa.gov/mgg/global/relief/ETOPO1/data/)
        - [🗺️ Interactive Map](https://maps.ngdc.noaa.gov/viewers/etopo/)
        - [🔧 Data Tools](https://www.ngdc.noaa.gov/mgg/global/gridtools.html)
        
        **Higher Resolution:**
        - [ETOPO2022](https://www.ncei.noaa.gov/products/etopo-global-relief-model) - 15 arc-second
        - [SRTM](https://www2.jpl.nasa.gov/srtm/) - 30m resolution
        - [GEBCO](https://www.gebco.net/) - Ocean bathymetry
        """)
    
    # Geoid Section
    st.markdown("---")
    st.markdown("#### 🌊 **Geoid Height Data (EGM2008)**")
    
    col_geoid1, col_geoid2 = st.columns([2, 1])
    
    with col_geoid1:
        st.markdown("""
        **Primary Source:** NASA/NGA Earth Gravitational Model  
        **Model:** EGM2008 - 2.5' resolution global geoid  
        **Format:** GDF, ASCII Grid, GeoTIFF  
        **Parameters:** Geoid height (m) relative to reference ellipsoid
        
        **📥 Download Steps:**
        1. Visit [NASA EGM2008](https://earth-info.nga.mil/) or mirror sites
        2. Download global or regional grids
        3. Supported formats: GDF (primary), GeoTIFF, ASCII
        4. Convert to CSV for this application
        
        **File Formats Supported:**
        - GDF (Geodetic Data Format)
        - ASCII Grid (.asc, .grd)
        - NetCDF (.nc)
        """)
    
    with col_geoid2:
        st.markdown("""
        **Quick Links:**
        - [🌐 NASA/NGA Portal](https://earth-info.nga.mil/)
        - [📁 ICGEM Download](http://icgem.gfz-potsdam.de/EGM2008)
        - [🔧 Online Calculator](http://icgem.gfz-potsdam.de/calc)
        - [📚 Model Details](https://earth-info.nga.mil/GandG/wgs84/gravitymod/egm2008/)
        
        **Alternative Sources:**
        - [ICGEM](http://icgem.gfz-potsdam.de/) - International Centre
        - [EGM96](https://earth-info.nga.mil/GandG/wgs84/gravitymod/egm96/egm96.html)
        - [XGM2019e](http://icgem.gfz-potsdam.de/tom_relnotes)
        """)
    
    

    
    # Quick Access Panel
    st.markdown("---")
    st.markdown("#### 🚀 **Quick Data Access Panel**")
    
    col_quick1, col_quick2, col_quick3, col_quick4 = st.columns(4)
    
    with col_quick1:
        if st.button("🌐 CRUST1.0", use_container_width=True):
            st.markdown("[Open CRUST1.0 Database](https://www.earthcrustmodel1.com/)")
    
    with col_quick2:
        if st.button("🗻 GlobSed", use_container_width=True):
            st.markdown("[Open GlobSed Database](https://www.earthcrustmodel1.com/)")
    
    with col_quick3:
        if st.button("⛰️ ETOPO1", use_container_width=True):
            st.markdown("[Open ETOPO1 Database](https://www.ngdc.noaa.gov/mgg/global/)")
    
    with col_quick4:
        if st.button("🌊 EGM2008", use_container_width=True):
            st.markdown("[Open EGM2008 Database](https://icgem.gfz-potsdam.de/calcgrid?modeltype=longtime&modelid=c50128797a9cb62e936337c890e4425f03f0461d7329b09a8cc8561504465340)")
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

# ==============================
# SECTION 2: DATA UPLOAD
# ==============================
elif st.session_state.current_section == "Data Upload":
    st.header("📁 Data Upload Section")
    st.markdown("Upload your CSV files containing crustal and sedimentary thickness data and upload Topography data as NetCDF(.nc) and Geoid data as (.gdf)")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.subheader("1. Crustal Thickness")
        uploaded_crust = st.file_uploader("Upload crustal thickness (CSV)", key="crust")
        
    with col2:
        st.subheader("2. Sedimentary Thickness")
        uploaded_sed = st.file_uploader("Upload sedimentary thickness (CSV)", key="sed")

    with col3:
        st.subheader("3. Topographic Data")
        uploaded_topo = st.file_uploader("Upload topographic data (CSV/GeoTIFF/NetCDF)", 
                                        type=['csv', 'nc', 'grd'], 
                                        key="topo")

    with col4:
        st.subheader("4. Geoid Data")
        uploaded_geoid = st.file_uploader("Upload geoid data (CSV/NetCDF/GDF)", 
                                         type=['csv', 'nc', 'grd', 'gdf'], 
                                         key="geoid")

    # File reading function (same as original)
    def read_geospatial_file(uploaded_file):
        """Read various geospatial formats and return a DataFrame with lon, lat, value"""
        

        # [Keep the same implementation as original...]
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            file_ext = os.path.splitext(uploaded_file.name)[1].lower()
            
            if file_ext in ['.csv']:
                df = pd.read_csv(tmp_path, sep=None, engine='python', skipinitialspace=True)
                df.columns = df.columns.str.strip()
                
            elif file_ext in ['.tif', '.tiff']:
                # [Keep TIFF reading implementation...]
                try:
                    import rioxarray
                    ds = rioxarray.open_rasterio(tmp_path)
                    df = ds.squeeze().to_dataframe(name='value').reset_index()
                    
                    if 'x' in df.columns and 'y' in df.columns:
                        df = df.rename(columns={'x': 'longitude', 'y': 'latitude'})
                    elif 'lon' in df.columns and 'lat' in df.columns:
                        df = df.rename(columns={'lon': 'longitude', 'lat': 'latitude'})
                        
                except ImportError:
                    # [Fallback implementation...]
                    pass
                    
            elif file_ext in ['.nc', '.grd', '.gdf']:
                # [Keep NetCDF reading implementation...]
                try:
                    engines_to_try = ['netcdf4', 'scipy']
                    df = None
                    
                    for engine in engines_to_try:
                        try:
                            with xr.open_dataset(tmp_path, engine=engine) as ds:
                                df = ds.to_dataframe().reset_index()
                            break
                        except Exception as e:
                            continue
                    
                    if df is None:
                        try:
                            with xr.open_dataset(tmp_path) as ds:
                                df = ds.to_dataframe().reset_index()
                        except Exception as e:
                            if file_ext == '.gdf':
                                try:
                                    df = pd.read_csv(tmp_path, delim_whitespace=True, header=None, 
                                                   names=['longitude', 'latitude', 'value'])
                                except:
                                    return None
                            else:
                                return None
                    
                    if df is not None:
                        coord_cols = [col for col in df.columns if col.lower() in ['x', 'y', 'lon', 'long', 'longitude', 'lat', 'latitude']]
                        value_cols = [col for col in df.columns if col not in coord_cols and col != 'band' and col != 'spatial_ref']
                        
                        if len(coord_cols) >= 2 and len(value_cols) >= 1:
                            df = df[coord_cols[:2] + value_cols[:1]]
                            rename_dict = {}
                            for col in coord_cols[:2]:
                                col_lower = col.lower()
                                if any(x in col_lower for x in ['x', 'lon', 'long']):
                                    rename_dict[col] = 'longitude'
                                elif any(x in col_lower for x in ['y', 'lat']):
                                    rename_dict[col] = 'latitude'
                            
                            for col in value_cols[:1]:
                                rename_dict[col] = 'value'
                            
                            df = df.rename(columns=rename_dict)
                        
                except Exception as e:
                    return None
                    
            else:
                st.error(f"Unsupported file format: {file_ext}")
                return None
                
            required_cols = ['longitude', 'latitude', 'value']
            if not all(col in df.columns for col in required_cols):
                col_mapping = {}
                for req_col in required_cols:
                    for actual_col in df.columns:
                        if req_col in actual_col.lower():
                            col_mapping[req_col] = actual_col
                            break
                
                if len(col_mapping) == 3:
                    df = df.rename(columns=col_mapping)
                else:
                    st.warning(f"Could not automatically identify required columns. Found: {list(df.columns)}")
                    return df
            
            return df
            
        except Exception as e:
            st.error(f"Error processing file: {e}")
            return None
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except Exception as e:
                    pass

    # Process uploaded files
    if uploaded_crust is not None:
        st.session_state.df_crust = pd.read_csv(
            uploaded_crust, 
            sep=None,
            engine='python',
            skipinitialspace=True,
            skip_blank_lines=True
        )
        st.session_state.df_crust.columns = st.session_state.df_crust.columns.str.strip()
        st.success(f"✅ Crustal thickness CSV loaded successfully! Found {len(st.session_state.df_crust.columns)} columns.")

    if uploaded_sed is not None:
        st.session_state.df_sed = pd.read_csv(
            uploaded_sed,
            sep=None,
            engine='python',
            skipinitialspace=True,
            skip_blank_lines=True
        )
        st.session_state.df_sed.columns = st.session_state.df_sed.columns.str.strip()
        st.success(f"✅ Sedimentary thickness CSV loaded successfully! Found {len(st.session_state.df_sed.columns)} columns.")

    if uploaded_topo is not None:
        st.session_state.df_topo = read_geospatial_file(uploaded_topo)
        if st.session_state.df_topo is not None:
            st.success(f"✅ Topographic data loaded successfully! Found {len(st.session_state.df_topo.columns)} columns.")

    if uploaded_geoid is not None:
        st.session_state.df_geoid = read_geospatial_file(uploaded_geoid)
        if st.session_state.df_geoid is not None:
            st.success(f"✅ Geoid data loaded successfully! Found {len(st.session_state.df_geoid.columns)} columns.")

    # Show data previews
    if any([st.session_state.df_crust is not None, st.session_state.df_sed is not None, 
            st.session_state.df_topo is not None, st.session_state.df_geoid is not None]):
        st.markdown("### 📋 Data Previews")
        
        preview_cols = st.columns(4)
        datasets = [
            ("Crustal", st.session_state.df_crust),
            ("Sedimentary", st.session_state.df_sed),
            ("Topography", st.session_state.df_topo),
            ("Geoid", st.session_state.df_geoid)
        ]
        
        for i, (name, df) in enumerate(datasets):
            with preview_cols[i]:
                if df is not None:
                    st.metric(f"{name} Data", f"{len(df)} rows")
                    st.dataframe(df.head(3), use_container_width=True)
                else:
                    st.info(f"No {name} data")


    
   
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    

# ==============================
# SECTION 3: DATA ANALYSIS
# ==============================
elif st.session_state.current_section == "Data Analysis":
    st.header("📊 Data Distribution Analysis")
    
    # Create a mapping of available datasets
    available_datasets = {}
    if st.session_state.df_crust is not None:
        available_datasets["Crustal Thickness"] = st.session_state.df_crust
    if st.session_state.df_sed is not None:
        available_datasets["Sedimentary Thickness"] = st.session_state.df_sed
    if st.session_state.df_topo is not None:
        available_datasets["Topographic Data"] = st.session_state.df_topo
    if st.session_state.df_geoid is not None:
        available_datasets["Geoid Data"] = st.session_state.df_geoid
    
    if not available_datasets:
        st.error("❌ Please upload at least one dataset first to analyze distributions.")
        st.info("💡 Go to 'Data Upload' section to upload your data files")
    else:
        # Dataset selection
        selected_dataset_name = st.selectbox(
            "Choose dataset to analyze:",
            options=list(available_datasets.keys()),
            key="dist_dataset_select"
        )
        
        df_selected_dist = available_datasets[selected_dataset_name]
        
        st.success(f"✅ Selected: **{selected_dataset_name}** with {len(df_selected_dist)} rows and {len(df_selected_dist.columns)} columns")
        
        # Column selection for analysis
        st.markdown("#### 🎯 Select Column for Analysis")
        
        numeric_columns = df_selected_dist.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_columns:
            st.error("❌ No numeric columns found in the selected dataset.")
        else:
            # Auto-detect value column based on common names
            def find_value_column(columns, dataset_name):
                col_lower = [str(col).lower() for col in columns]
                if "thickness" in dataset_name.lower():
                    for i, col in enumerate(col_lower):
                        if any(keyword in col for keyword in ['thick', 'depth', 'value', 'z']):
                            return columns[i]
                elif "topo" in dataset_name.lower() or "elev" in dataset_name.lower():
                    for i, col in enumerate(col_lower):
                        if any(keyword in col for keyword in ['elev', 'topo', 'height', 'altitude', 'value', 'z']):
                            return columns[i]
                elif "geoid" in dataset_name.lower():
                    for i, col in enumerate(col_lower):
                        if any(keyword in col for keyword in ['geoid', 'height', 'undulation', 'value', 'h']):
                            return columns[i]
                elif "corrected" in dataset_name.lower():
                    for i, col in enumerate(col_lower):
                        if any(keyword in col for keyword in ['corrected', 'geoid', 'value']):
                            return columns[i]
                elif "residual" in dataset_name.lower():
                    for i, col in enumerate(col_lower):
                        if any(keyword in col for keyword in ['residual', 'geoid', 'value']):
                            return columns[i]
                return columns[0]  # fallback to first column
            
            default_value_col = find_value_column(numeric_columns, selected_dataset_name)
            
            value_column = st.selectbox(
                "Select value column for analysis:",
                options=numeric_columns,
                index=numeric_columns.index(default_value_col) if default_value_col in numeric_columns else 0,
                key="dist_value_column"
            )
            
            # Get the data
            values_original = df_selected_dist[value_column].dropna()
            
            if len(values_original) == 0:
                st.error("❌ No valid numeric data found in the selected column.")
            else:
                st.info(f"📊 Analyzing **{value_column}** - {len(values_original)} valid values")
                
                # Outlier Detection Section
                st.markdown("#### 🎯 Outlier Detection & Removal")
                
                outlier_method = st.selectbox(
                    "Select outlier detection method:",
                    [
                        "No Outlier Removal",
                        "Standard Deviation Method",
                        "Isolation Forest", 
                        "Minimum Covariance Determinant",
                        "Local Outlier Factor",
                        "One-Class SVM"
                    ],
                    index=0,
                    key="outlier_method"
                )
                
                values_clean = values_original.copy()
                outlier_mask = np.zeros(len(values_original), dtype=bool)
                outlier_info = {}
                
                if outlier_method != "No Outlier Removal":
                    col_param1, col_param2 = st.columns(2)
                    
                    with col_param1:
                        if outlier_method == "Standard Deviation Method":
                            n_std = st.slider(
                                "Number of standard deviations",
                                min_value=1.0,
                                max_value=5.0,
                                value=3.0,
                                step=0.1,
                                key="n_std"
                            )
                            threshold = n_std * values_original.std()
                            mean_val = values_original.mean()
                            outlier_mask = (values_original < mean_val - threshold) | (values_original > mean_val + threshold)
                            outlier_info = {
                                "method": "Standard Deviation",
                                "parameters": f"±{n_std}σ from mean",
                                "threshold_low": mean_val - threshold,
                                "threshold_high": mean_val + threshold
                            }
                        
                        elif outlier_method == "Isolation Forest":
                            contamination = st.slider(
                                "Contamination ratio",
                                min_value=0.01,
                                max_value=0.5,
                                value=0.1,
                                step=0.01,
                                key="if_contamination"
                            )
                            try:
                                from sklearn.ensemble import IsolationForest
                                iso_forest = IsolationForest(contamination=contamination, random_state=42)
                                preds = iso_forest.fit_predict(values_original.values.reshape(-1, 1))
                                outlier_mask = preds == -1
                                outlier_info = {
                                    "method": "Isolation Forest",
                                    "parameters": f"contamination={contamination}",
                                    "contamination": contamination
                                }
                            except Exception as e:
                                st.error(f"Error in Isolation Forest: {e}")
                                outlier_mask = np.zeros(len(values_original), dtype=bool)
                        
                        elif outlier_method == "Minimum Covariance Determinant":
                            contamination = st.slider(
                                "Contamination ratio",
                                min_value=0.01,
                                max_value=0.5,
                                value=0.1,
                                step=0.01,
                                key="mcd_contamination"
                            )
                            try:
                                from sklearn.covariance import EllipticEnvelope
                                envelope = EllipticEnvelope(contamination=contamination, random_state=42)
                                preds = envelope.fit_predict(values_original.values.reshape(-1, 1))
                                outlier_mask = preds == -1
                                outlier_info = {
                                    "method": "Minimum Covariance Determinant",
                                    "parameters": f"contamination={contamination}",
                                    "contamination": contamination
                                }
                            except Exception as e:
                                st.error(f"Error in Minimum Covariance: {e}")
                                outlier_mask = np.zeros(len(values_original), dtype=bool)
                    
                    with col_param2:
                        if outlier_method == "Local Outlier Factor":
                            n_neighbors = st.slider(
                                "Number of neighbors",
                                min_value=5,
                                max_value=50,
                                value=20,
                                step=5,
                                key="lof_neighbors"
                            )
                            contamination = st.slider(
                                "Contamination ratio",
                                min_value=0.01,
                                max_value=0.5,
                                value=0.1,
                                step=0.01,
                                key="lof_contamination"
                            )
                            try:
                                from sklearn.neighbors import LocalOutlierFactor
                                lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
                                preds = lof.fit_predict(values_original.values.reshape(-1, 1))
                                outlier_mask = preds == -1
                                outlier_info = {
                                    "method": "Local Outlier Factor",
                                    "parameters": f"n_neighbors={n_neighbors}, contamination={contamination}",
                                    "n_neighbors": n_neighbors,
                                    "contamination": contamination
                                }
                            except Exception as e:
                                st.error(f"Error in Local Outlier Factor: {e}")
                                outlier_mask = np.zeros(len(values_original), dtype=bool)
                        
                        elif outlier_method == "One-Class SVM":
                            nu = st.slider(
                                "Nu parameter",
                                min_value=0.01,
                                max_value=0.5,
                                value=0.1,
                                step=0.01,
                                key="svm_nu"
                            )
                            kernel = st.selectbox(
                                "Kernel type",
                                ["rbf", "linear", "poly", "sigmoid"],
                                index=0,
                                key="svm_kernel"
                            )
                            try:
                                from sklearn.svm import OneClassSVM
                                oc_svm = OneClassSVM(kernel=kernel, nu=nu)
                                preds = oc_svm.fit_predict(values_original.values.reshape(-1, 1))
                                outlier_mask = preds == -1
                                outlier_info = {
                                    "method": "One-Class SVM",
                                    "parameters": f"kernel={kernel}, nu={nu}",
                                    "kernel": kernel,
                                    "nu": nu
                                }
                            except Exception as e:
                                st.error(f"Error in One-Class SVM: {e}")
                                outlier_mask = np.zeros(len(values_original), dtype=bool)
                    
                    # Apply outlier removal
                    values_clean = values_original[~outlier_mask]
                    
                    # Display outlier statistics
                    st.markdown("##### 📊 Outlier Detection Results")
                    col_out1, col_out2, col_out3, col_out4 = st.columns(4)
                    
                    with col_out1:
                        st.metric("Original Count", f"{len(values_original):,}")
                    with col_out2:
                        st.metric("Outliers Detected", f"{outlier_mask.sum():,}")
                    with col_out3:
                        st.metric("Outlier Percentage", f"{(outlier_mask.sum()/len(values_original)*100):.1f}%")
                    with col_out4:
                        st.metric("Clean Data Count", f"{len(values_clean):,}")
                
                # Plot type selection
                st.markdown("#### 📊 Choose Plot Type")
                
                plot_type = st.radio(
                    "Select distribution plot type:",
                    ["Histogram", "Violin Plot", "Box Plot"],
                    horizontal=True,
                    key="dist_plot_type"
                )
                
                # Create tabs for different visualizations
                dist_tab1, dist_tab2, dist_tab3 = st.tabs(["🎨 Plot Customization", "📈 Before/After Comparison", "📊 Statistics"])
                
                with dist_tab1:
                    st.markdown("##### 🎨 Plot Customization Options")
                    
                    if plot_type == "Histogram":
                        col_hist1, col_hist2, col_hist3 = st.columns(3)
                        
                        with col_hist1:
                            hist_bins = st.slider(
                                "Number of bins",
                                min_value=10,
                                max_value=100,
                                value=30,
                                key="hist_bins_dist"
                            )
                            hist_color_before = st.color_picker(
                                "Before color",
                                value="#1f77b4",
                                key="hist_color_before"
                            )
                            hist_color_after = st.color_picker(
                                "After color", 
                                value="#ff7f0e",
                                key="hist_color_after"
                            )
                        
                        with col_hist2:
                            edge_color = st.color_picker(
                                "Edge color",
                                value="#000000",
                                key="edge_color_dist"
                            )
                            edge_width = st.slider(
                                "Edge width",
                                min_value=0.0,
                                max_value=2.0,
                                value=1.0,
                                step=0.1,
                                key="edge_width_dist"
                            )
                            alpha_val = st.slider(
                                "Transparency",
                                min_value=0.1,
                                max_value=1.0,
                                value=0.7,
                                step=0.1,
                                key="alpha_hist"
                            )
                        
                        with col_hist3:
                            show_kde = st.checkbox(
                                "Show KDE curve",
                                value=True,
                                key="show_kde_dist"
                            )
                            kde_color_before = st.color_picker(
                                "KDE before color",
                                value="#1f77b4",
                                key="kde_color_before"
                            ) if show_kde else "#1f77b4"
                            kde_color_after = st.color_picker(
                                "KDE after color",
                                value="#ff7f0e", 
                                key="kde_color_after"
                            ) if show_kde else "#ff7f0e"
                    
                    elif plot_type == "Violin Plot":
                        col_violin1, col_violin2 = st.columns(2)
                        
                        with col_violin1:
                            violin_color_before = st.color_picker(
                                "Before color",
                                value="#1f77b4",
                                key="violin_color_before"
                            )
                            violin_color_after = st.color_picker(
                                "After color",
                                value="#ff7f0e",
                                key="violin_color_after"
                            )
                            show_points = st.checkbox(
                                "Show data points",
                                value=False,
                                key="violin_points_dist"
                            )
                        
                        with col_violin2:
                            show_quartiles = st.checkbox(
                                "Show quartiles",
                                value=True,
                                key="violin_quartiles"
                            )
                            alpha_val = st.slider(
                                "Transparency",
                                min_value=0.1,
                                max_value=1.0,
                                value=0.7,
                                step=0.1,
                                key="alpha_violin"
                            )
                    
                    elif plot_type == "Box Plot":
                        col_box1, col_box2 = st.columns(2)
                        
                        with col_box1:
                            box_color_before = st.color_picker(
                                "Before color",
                                value="#1f77b4",
                                key="box_color_before"
                            )
                            box_color_after = st.color_picker(
                                "After color",
                                value="#ff7f0e",
                                key="box_color_after"
                            )
                            show_outliers = st.checkbox(
                                "Show outliers",
                                value=True,
                                key="box_outliers_dist"
                            )
                        
                        with col_box2:
                            notch_box = st.checkbox(
                                "Show confidence intervals",
                                value=False,
                                key="box_notch_dist"
                            )
                            alpha_val = st.slider(
                                "Transparency",
                                min_value=0.1,
                                max_value=1.0,
                                value=0.7,
                                step=0.1,
                                key="alpha_box"
                            )
                    
                    # General styling options
                    st.markdown("##### 🎭 General Styling")
                    
                    col_style1, col_style2, col_style3 = st.columns(3)
                    
                    with col_style1:
                        fig_width = st.slider(
                            "Figure width",
                            min_value=8,
                            max_value=20,
                            value=14,
                            key="fig_width_dist"
                        )
                    
                    with col_style2:
                        fig_height = st.slider(
                            "Figure height",
                            min_value=6,
                            max_value=15,
                            value=8,
                            key="fig_height_dist"
                        )
                    
                    with col_style3:
                        font_size = st.slider(
                            "Font size",
                            min_value=10,
                            max_value=20,
                            value=14,
                            key="font_size_dist"
                        )
                
                with dist_tab2:
                    st.markdown("##### 📈 Before vs After Outlier Removal")
                    
                    # Create comparison plot
                    fig_comparison, (ax1, ax2) = plt.subplots(1, 2, figsize=(fig_width, fig_height))
                    
                    if plot_type == "Histogram":
                        # Before outlier removal
                        n_before, bins_before, patches_before = ax1.hist(
                            values_original, 
                            bins=hist_bins, 
                            color=hist_color_before, 
                            edgecolor=edge_color, 
                            linewidth=edge_width,
                            alpha=alpha_val,
                            density=show_kde,
                            label='Before'
                        )
                        
                        # After outlier removal
                        n_after, bins_after, patches_after = ax2.hist(
                            values_clean, 
                            bins=hist_bins, 
                            color=hist_color_after, 
                            edgecolor=edge_color, 
                            linewidth=edge_width,
                            alpha=alpha_val,
                            density=show_kde,
                            label='After'
                        )
                        
                        # Add KDE if requested
                        if show_kde:
                            from scipy.stats import gaussian_kde
                            try:
                                # Before KDE
                                kde_before = gaussian_kde(values_original)
                                x_kde_before = np.linspace(values_original.min(), values_original.max(), 200)
                                y_kde_before = kde_before(x_kde_before)
                                ax1.plot(x_kde_before, y_kde_before, color=kde_color_before, linewidth=2, label='KDE')
                                
                                # After KDE
                                kde_after = gaussian_kde(values_clean)
                                x_kde_after = np.linspace(values_clean.min(), values_clean.max(), 200)
                                y_kde_after = kde_after(x_kde_after)
                                ax2.plot(x_kde_after, y_kde_after, color=kde_color_after, linewidth=2, label='KDE')
                            except Exception as e:
                                st.warning(f"Could not compute KDE: {e}")
                        
                        ax1.legend(fontsize=font_size-2)
                        ax2.legend(fontsize=font_size-2)
                        
                        ax1.set_xlabel(value_column, fontsize=font_size)
                        ax2.set_xlabel(value_column, fontsize=font_size)
                        ax1.set_ylabel('Density' if show_kde else 'Frequency', fontsize=font_size)
                    
                    elif plot_type == "Violin Plot":
                        # Before outlier removal
                        violin_parts_before = ax1.violinplot(
                            values_original, 
                            showmeans=True, 
                            showmedians=show_quartiles,
                            showextrema=True
                        )
                        
                        # Customize before violin
                        for pc in violin_parts_before['bodies']:
                            pc.set_facecolor(violin_color_before)
                            pc.set_alpha(alpha_val)
                            pc.set_edgecolor('black')
                        
                        violin_parts_before['cmeans'].set_color('red')
                        violin_parts_before['cmedians'].set_color('blue')
                        
                        # After outlier removal
                        violin_parts_after = ax2.violinplot(
                            values_clean, 
                            showmeans=True, 
                            showmedians=show_quartiles,
                            showextrema=True
                        )
                        
                        # Customize after violin
                        for pc in violin_parts_after['bodies']:
                            pc.set_facecolor(violin_color_after)
                            pc.set_alpha(alpha_val)
                            pc.set_edgecolor('black')
                        
                        violin_parts_after['cmeans'].set_color('red')
                        violin_parts_after['cmedians'].set_color('blue')
                        
                        # Add data points if requested
                        if show_points:
                            x_jitter_before = np.random.normal(1, 0.05, len(values_original))
                            x_jitter_after = np.random.normal(1, 0.05, len(values_clean))
                            ax1.scatter(x_jitter_before, values_original, alpha=0.3, color='black', s=20)
                            ax2.scatter(x_jitter_after, values_clean, alpha=0.3, color='black', s=20)
                        
                        ax1.set_ylabel(value_column, fontsize=font_size)
                        ax2.set_ylabel(value_column, fontsize=font_size)
                        ax1.set_xticks([1])
                        ax2.set_xticks([1])
                        ax1.set_xticklabels(['Before'])
                        ax2.set_xticklabels(['After'])
                    
                    elif plot_type == "Box Plot":
                        # Before outlier removal
                        box_plot_before = ax1.boxplot(
                            values_original,
                            patch_artist=True,
                            showfliers=show_outliers,
                            notch=notch_box
                        )
                        
                        # Customize before box
                        box_plot_before['boxes'][0].set_facecolor(box_color_before)
                        box_plot_before['boxes'][0].set_alpha(alpha_val)
                        box_plot_before['medians'][0].set_color('red')
                        
                        # After outlier removal
                        box_plot_after = ax2.boxplot(
                            values_clean,
                            patch_artist=True,
                            showfliers=show_outliers,
                            notch=notch_box
                        )
                        
                        # Customize after box
                        box_plot_after['boxes'][0].set_facecolor(box_color_after)
                        box_plot_after['boxes'][0].set_alpha(alpha_val)
                        box_plot_after['medians'][0].set_color('red')
                        
                        ax1.set_ylabel(value_column, fontsize=font_size)
                        ax2.set_ylabel(value_column, fontsize=font_size)
                        ax1.set_xticks([1])
                        ax2.set_xticks([1])
                        ax1.set_xticklabels(['Before'])
                        ax2.set_xticklabels(['After'])
                    
                    # Common styling for both subplots
                    ax1.set_title(f'Before Outlier Removal\n({len(values_original):,} points)', 
                                fontsize=font_size+1, fontweight='bold')
                    ax2.set_title(f'After Outlier Removal\n({len(values_clean):,} points)', 
                                fontsize=font_size+1, fontweight='bold')
                    
                    for ax in [ax1, ax2]:
                        ax.tick_params(axis='both', which='major', labelsize=font_size-2)
                        ax.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    st.pyplot(fig_comparison)
                    
                    # Spatial Distribution Maps (Before and After) - UPDATED VERSION
                    if outlier_method != "No Outlier Removal":
                        st.markdown("##### 🗺️ Spatial Distribution Before vs After")
                        
                        # Find latitude and longitude columns
                        def find_coordinate_columns(df):
                            lat_candidates = [col for col in df.columns if any(name in col.lower() for name in ['lat', 'y'])]
                            lon_candidates = [col for col in df.columns if any(name in col.lower() for name in ['lon', 'long', 'x'])]
                            return lat_candidates[0] if lat_candidates else None, lon_candidates[0] if lon_candidates else None
                        
                        lat_col, lon_col = find_coordinate_columns(df_selected_dist)
                        
                        if lat_col and lon_col:
                            # Prepare data for mapping
                            df_original = df_selected_dist[[lat_col, lon_col, value_column]].dropna()
                            df_clean = df_selected_dist[~outlier_mask][[lat_col, lon_col, value_column]].dropna()
                            
                            # Interpolation controls
                            st.markdown("###### 🔧 Interpolation Settings")
                            col_interp1, col_interp2, col_interp3 = st.columns(3)
                            
                            with col_interp1:
                                interp_method = st.selectbox(
                                    "Interpolation method:",
                                    ["linear", "cubic", "nearest"],
                                    index=0,
                                    key="interp_method"
                                )
                            
                            with col_interp2:
                                grid_resolution = st.slider(
                                    "Grid resolution:",
                                    min_value=100,
                                    max_value=1000,
                                    value=300,
                                    step=50,
                                    key="grid_res"
                                )
                            
                            with col_interp3:
                                interp_color_map = st.selectbox(
                                    "Color map:",
                                    ["viridis", "plasma", "inferno", "magma", "coolwarm", "rainbow", "jet"],
                                    index=0,
                                    key="interp_cmap"
                                )
                            
                            # Create THREE subplots: Before, After Scatter, After Interpolated
                            fig_maps, (ax_map1, ax_map2, ax_map3) = plt.subplots(1, 3, figsize=(fig_width * 1.5, fig_height-2))
                            
                            # Plot 1: Before outlier removal (scatter)
                            sc1 = ax_map1.scatter(df_original[lon_col], df_original[lat_col], 
                                                c=df_original[value_column], cmap='viridis', 
                                                s=15, alpha=0.7)
                            ax_map1.set_title(f'Before: {value_column}\n({len(df_original):,} points)', 
                                            fontsize=font_size, fontweight='bold')
                            ax_map1.set_xlabel('Longitude', fontsize=font_size-1)
                            ax_map1.set_ylabel('Latitude', fontsize=font_size-1)
                            plt.colorbar(sc1, ax=ax_map1, label=value_column)
                            
                            # Plot 2: After outlier removal (scatter)
                            sc2 = ax_map2.scatter(df_clean[lon_col], df_clean[lat_col], 
                                                c=df_clean[value_column], cmap='viridis', 
                                                s=15, alpha=0.7)
                            ax_map2.set_title(f'After: {value_column}\n({len(df_clean):,} points)', 
                                            fontsize=font_size, fontweight='bold')
                            ax_map2.set_xlabel('Longitude', fontsize=font_size-1)
                            ax_map2.set_ylabel('Latitude', fontsize=font_size-1)
                            plt.colorbar(sc2, ax=ax_map2, label=value_column)
                            
                            # Plot 3: After outlier removal (interpolated)
                            try:
                                # Create grid for interpolation
                                x_min, x_max = df_clean[lon_col].min(), df_clean[lon_col].max()
                                y_min, y_max = df_clean[lat_col].min(), df_clean[lat_col].max()
                                
                                # Create grid coordinates
                                xi = np.linspace(x_min, x_max, grid_resolution)
                                yi = np.linspace(y_min, y_max, grid_resolution)
                                xi, yi = np.meshgrid(xi, yi)
                                
                                # Perform interpolation
                                from scipy.interpolate import griddata
                                zi = griddata(
                                    (df_clean[lon_col], df_clean[lat_col]), 
                                    df_clean[value_column], 
                                    (xi, yi), 
                                    method=interp_method,
                                    fill_value=np.nan  # Fill areas with no data with NaN
                                )
                                
                                # Plot interpolated data
                                im = ax_map3.contourf(xi, yi, zi, levels=50, cmap=interp_color_map, alpha=0.8)
                                
                                # Add contour lines for better visualization
                                contour_lines = ax_map3.contour(xi, yi, zi, levels=10, colors='black', linewidths=0.5, alpha=0.5)
                                ax_map3.clabel(contour_lines, inline=True, fontsize=8, fmt='%.1f')
                                
                                # Overlay original points for reference
                                ax_map3.scatter(df_clean[lon_col], df_clean[lat_col], 
                                              c='k', s=5, alpha=0.3, label='Data points')
                                
                                ax_map3.set_title(f'Interpolated: {value_column}\n({interp_method} method)', 
                                                fontsize=font_size, fontweight='bold')
                                ax_map3.set_xlabel('Longitude', fontsize=font_size-1)
                                ax_map3.set_ylabel('Latitude', fontsize=font_size-1)
                                plt.colorbar(im, ax=ax_map3, label=value_column)
                                
                                # Add legend for data points
                                ax_map3.legend(loc='upper right', fontsize=font_size-3)
                                
                                st.success(f"✅ Interpolation completed using {interp_method} method on {grid_resolution}x{grid_resolution} grid")
                                
                            except Exception as e:
                                st.error(f"❌ Interpolation failed: {e}")
                                # Fallback: show empty subplot with error message
                                ax_map3.text(0.5, 0.5, f'Interpolation\nFailed\n{str(e)[:50]}...', 
                                           ha='center', va='center', transform=ax_map3.transAxes, fontsize=12)
                                ax_map3.set_title(f'Interpolation Failed', fontsize=font_size, fontweight='bold')
                            
                            # Set consistent axis limits for all three plots
                            x_min = min(df_original[lon_col].min(), df_clean[lon_col].min())
                            x_max = max(df_original[lon_col].max(), df_clean[lon_col].max())
                            y_min = min(df_original[lat_col].min(), df_clean[lat_col].min())
                            y_max = max(df_original[lat_col].max(), df_clean[lat_col].max())
                            
                            for ax in [ax_map1, ax_map2, ax_map3]:
                                ax.set_xlim(x_min, x_max)
                                ax.set_ylim(y_min, y_max)
                                ax.tick_params(axis='both', which='major', labelsize=font_size-2)
                                ax.grid(True, alpha=0.3)
                            
                            plt.tight_layout()
                            st.pyplot(fig_maps)
                            
                            # Additional interpolation statistics
                            if 'zi' in locals() and not np.isnan(zi).all():
                                st.markdown("###### 📊 Interpolation Statistics")
                                col_interp_stats1, col_interp_stats2, col_interp_stats3 = st.columns(3)
                                
                                with col_interp_stats1:
                                    st.metric("Grid Resolution", f"{grid_resolution}×{grid_resolution}")
                                    st.metric("Valid Grid Points", f"{np.sum(~np.isnan(zi)):,}")
                                
                                with col_interp_stats2:
                                    st.metric("Interpolation Method", interp_method)
                                    st.metric("Data Coverage", f"{(np.sum(~np.isnan(zi)) / (grid_resolution**2) * 100):.1f}%")
                                
                                with col_interp_stats3:
                                    if not np.isnan(zi).all():
                                        st.metric("Min Interpolated", f"{np.nanmin(zi):.3f}")
                                        st.metric("Max Interpolated", f"{np.nanmax(zi):.3f}")
                        else:
                            st.warning("⚠️ Could not auto-detect latitude/longitude columns for spatial mapping")
                    
                    # Download option for the figures - UPDATED
                    col_dl1, col_dl2, col_dl3 = st.columns(3)

                    with col_dl1:
                        # Save comparison figure to buffer
                        buf_comparison = io.BytesIO()
                        fig_comparison.savefig(buf_comparison, format='png', dpi=300, bbox_inches='tight')
                        buf_comparison.seek(0)
                        
                        st.download_button(
                            label="📥 Download Comparison Plot (PNG)",
                            data=buf_comparison,
                            file_name=f"{selected_dataset_name}_{value_column}_comparison.png",
                            mime="image/png"
                        )

                    with col_dl2:
                        if outlier_method != "No Outlier Removal" and lat_col and lon_col:
                            # Save maps figure to buffer
                            buf_maps = io.BytesIO()
                            fig_maps.savefig(buf_maps, format='png', dpi=300, bbox_inches='tight')
                            buf_maps.seek(0)
                            
                            st.download_button(
                                label="📥 Download Maps (PNG)",
                                data=buf_maps,
                                file_name=f"{selected_dataset_name}_{value_column}_maps.png",
                                mime="image/png"
                            )

                    with col_dl3:
                        if outlier_method != "No Outlier Removal" and lat_col and lon_col and 'zi' in locals():
                            # Save interpolation data as CSV
                            try:
                                # Create flattened arrays for download
                                interp_data = []
                                for i in range(grid_resolution):
                                    for j in range(grid_resolution):
                                        if not np.isnan(zi[i, j]):
                                            interp_data.append({
                                                'longitude': xi[i, j],
                                                'latitude': yi[i, j], 
                                                value_column: zi[i, j]
                                            })
                                
                                df_interp = pd.DataFrame(interp_data)
                                csv_interp = df_interp.to_csv(index=False)
                                
                                st.download_button(
                                    label="📥 Download Interpolated Data (CSV)",
                                    data=csv_interp,
                                    file_name=f"{selected_dataset_name}_{value_column}_interpolated_{interp_method}.csv",
                                    mime="text/csv"
                                )
                            except Exception as e:
                                st.error(f"Could not prepare interpolation data for download: {e}")
                
                with dist_tab3:
                    st.markdown("##### 📊 Detailed Statistics")
                    
                    # Statistics comparison
                    st.markdown("###### 📈 Statistics Comparison")
                    
                    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                    
                    with col_stat1:
                        st.metric("Count Before", f"{len(values_original):,}")
                        st.metric("Mean Before", f"{values_original.mean():.3f}")
                        if outlier_method != "No Outlier Removal":
                            st.metric("Count After", f"{len(values_clean):,}", 
                                    delta=f"{-outlier_mask.sum()}")
                            st.metric("Mean After", f"{values_clean.mean():.3f}", 
                                    delta=f"{(values_clean.mean() - values_original.mean()):.3f}")
                    
                    with col_stat2:
                        st.metric("Std Before", f"{values_original.std():.3f}")
                        st.metric("Min Before", f"{values_original.min():.3f}")
                        if outlier_method != "No Outlier Removal":
                            st.metric("Std After", f"{values_clean.std():.3f}", 
                                    delta=f"{(values_clean.std() - values_original.std()):.3f}")
                            st.metric("Min After", f"{values_clean.min():.3f}", 
                                    delta=f"{(values_clean.min() - values_original.min()):.3f}")
                    
                    with col_stat3:
                        st.metric("25% Before", f"{values_original.quantile(0.25):.3f}")
                        st.metric("Median Before", f"{values_original.median():.3f}")
                        if outlier_method != "No Outlier Removal":
                            st.metric("25% After", f"{values_clean.quantile(0.25):.3f}", 
                                    delta=f"{(values_clean.quantile(0.25) - values_original.quantile(0.25)):.3f}")
                            st.metric("Median After", f"{values_clean.median():.3f}", 
                                    delta=f"{(values_clean.median() - values_original.median()):.3f}")
                    
                    with col_stat4:
                        st.metric("75% Before", f"{values_original.quantile(0.75):.3f}")
                        st.metric("Max Before", f"{values_original.max():.3f}")
                        if outlier_method != "No Outlier Removal":
                            st.metric("75% After", f"{values_clean.quantile(0.75):.3f}", 
                                    delta=f"{(values_clean.quantile(0.75) - values_original.quantile(0.75)):.3f}")
                            st.metric("Max After", f"{values_clean.max():.3f}", 
                                    delta=f"{(values_clean.max() - values_original.max()):.3f}")
                    
                    # Outlier details if applicable
                    if outlier_method != "No Outlier Removal":
                        st.markdown("###### 🎯 Outlier Analysis")
                        
                        col_out_anal1, col_out_anal2 = st.columns(2)
                        
                        with col_out_anal1:
                            st.write("**Outlier Detection Details:**")
                            st.write(f"- Method: {outlier_info.get('method', 'N/A')}")
                            st.write(f"- Parameters: {outlier_info.get('parameters', 'N/A')}")
                            st.write(f"- Outliers removed: {outlier_mask.sum():,}")
                            st.write(f"- Percentage removed: {(outlier_mask.sum()/len(values_original)*100):.2f}%")
                        
                        with col_out_anal2:
                            st.write("**Outlier Values Summary:**")
                            outlier_values = values_original[outlier_mask]
                            if len(outlier_values) > 0:
                                st.write(f"- Min outlier: {outlier_values.min():.3f}")
                                st.write(f"- Max outlier: {outlier_values.max():.3f}")
                                st.write(f"- Mean outlier: {outlier_values.mean():.3f}")
                                st.write(f"- Most extreme: {outlier_values.iloc[np.argmax(np.abs(outlier_values - values_original.mean()))]:.3f}")
                            else:
                                st.write("No outliers detected")
                    
                    # Data summary for download
                    st.markdown("##### 💾 Download Statistics")
                    
                    # Prepare comprehensive statistics
                    stats_comparison = {
                        'Metric': [
                            'Count', 'Mean', 'Standard Deviation', 'Minimum', 
                            '25th Percentile', 'Median', '75th Percentile', 'Maximum',
                            'Variance', 'Skewness', 'Kurtosis', 'IQR'
                        ],
                        'Before': [
                            len(values_original), values_original.mean(), values_original.std(), values_original.min(),
                            values_original.quantile(0.25), values_original.median(), values_original.quantile(0.75), values_original.max(),
                            values_original.var(), values_original.skew(), values_original.kurtosis(),
                            values_original.quantile(0.75) - values_original.quantile(0.25)
                        ]
                    }
                    
                    if outlier_method != "No Outlier Removal":
                        stats_comparison['After'] = [
                            len(values_clean), values_clean.mean(), values_clean.std(), values_clean.min(),
                            values_clean.quantile(0.25), values_clean.median(), values_clean.quantile(0.75), values_clean.max(),
                            values_clean.var(), values_clean.skew(), values_clean.kurtosis(),
                            values_clean.quantile(0.75) - values_clean.quantile(0.25)
                        ]
                        stats_comparison['Difference'] = [
                            len(values_clean) - len(values_original),
                            values_clean.mean() - values_original.mean(),
                            values_clean.std() - values_original.std(),
                            values_clean.min() - values_original.min(),
                            values_clean.quantile(0.25) - values_original.quantile(0.25),
                            values_clean.median() - values_original.median(),
                            values_clean.quantile(0.75) - values_original.quantile(0.75),
                            values_clean.max() - values_original.max(),
                            values_clean.var() - values_original.var(),
                            values_clean.skew() - values_original.skew(),
                            values_clean.kurtosis() - values_original.kurtosis(),
                            (values_clean.quantile(0.75) - values_clean.quantile(0.25)) - (values_original.quantile(0.75) - values_original.quantile(0.25))
                        ]
                    
                    df_stats_comparison = pd.DataFrame(stats_comparison)
                    csv_stats = df_stats_comparison.to_csv(index=False)
                    
                    st.download_button(
                        label="📥 Download Statistics Comparison (CSV)",
                        data=csv_stats,
                        file_name=f"{selected_dataset_name}_{value_column}_statistics_comparison.csv",
                        mime="text/csv"
                    )
        
        
        
        
        
        
        
        
        
        
        
        

# ==============================
# SECTION 4: DATA VISUALIZATION
# ==============================
elif st.session_state.current_section == "Data Visualization":
    st.header("📊 Data Visualization & Interpolation")
    
    # Check if data is available
    if all(df is None for df in [st.session_state.df_crust, st.session_state.df_sed, 
                                st.session_state.df_topo, st.session_state.df_geoid]):
        st.error("❌ No data available for visualization. Please upload data files first.")
        st.info("💡 Go to 'Data Upload' section to upload your data files")
    else:
        # Data type selection
        option = st.selectbox(
            "Select data type to plot:",
            ["Sedimentary thickness", "Crustal thickness", "Topographic thickness", "Geoid data"],
            key="plot_option"
        )

        df_map = {
            "Sedimentary thickness": st.session_state.df_sed,
            "Crustal thickness": st.session_state.df_crust,
            "Topographic thickness": st.session_state.df_topo,
            "Geoid data": st.session_state.df_geoid
        }
        
        df_selected = df_map.get(option)

        if df_selected is None:
            st.warning(f"⚠️ Please upload the file for **{option}** first in the Data Upload section.")
        else:
            st.markdown(f"### Selected: **{option}**")
            st.info(f"📋 Available columns in this dataset: {', '.join([str(c) for c in df_selected.columns])}")
            raw_cols = list(df_selected.columns)
            
            # strip whitespace for display purposes
            stripped = [c.strip() if isinstance(c, str) else str(c) for c in raw_cols]

            # build display names and map to original column indices
            display_names = []
            name_counts = {}
            display_to_index = {}
            for idx, name in enumerate(stripped):
                # increment count
                cnt = name_counts.get(name, 0) + 1
                name_counts[name] = cnt
                if cnt == 1:
                    disp = name
                else:
                    disp = f"{name} ({cnt})"
                display_names.append(disp)
                display_to_index[disp] = idx  # map display string -> original column index

            # helper to pick defaults by keyword (searches stripped names)
            def find_default_index(keywords, names_list, fallback=0):
                for i, n in enumerate(names_list):
                    nl = n.lower()
                    for kw in keywords:
                        if kw in nl:
                            return i
                return fallback if names_list else 0

            # Improved default detection for all data types
            if option == "Crustal thickness":
                lat_default_disp_idx = find_default_index(["latit", "lat", "latitude", "y"], stripped, 0)
                lon_default_disp_idx = find_default_index(["long", "lon", "longitude", "x"], stripped, 1 if len(stripped) > 1 else 0)
                val_default_disp_idx = find_default_index(["thk", "thick", "thickness", "value", "depth"], stripped, 2 if len(stripped) > 2 else 0)
            elif option == "Sedimentary thickness":
                lat_default_disp_idx = find_default_index(["lat", "latitude", "y"], stripped, 1 if len(stripped) > 1 else 0)
                lon_default_disp_idx = find_default_index(["long", "lon", "longitude", "x"], stripped, 0)
                val_default_disp_idx = find_default_index(["sed_thick", "sed", "sediment", "thick", "thickness", "value"], stripped, 2 if len(stripped) > 2 else 0)
            elif option == "Topographic thickness":
                lat_default_disp_idx = find_default_index(["lat", "latitude", "y"], stripped, 1 if len(stripped) > 1 else 0)
                lon_default_disp_idx = find_default_index(["long", "lon", "longitude", "x"], stripped, 0)
                val_default_disp_idx = find_default_index(["elev", "topo", "topography", "height", "altitude", "value", "z"], stripped, 2 if len(stripped) > 2 else 0)
            else:  # Geoid data
                lat_default_disp_idx = find_default_index(["lat", "latitude", "y"], stripped, 1 if len(stripped) > 1 else 0)
                lon_default_disp_idx = find_default_index(["long", "lon", "longitude", "x"], stripped, 0)
                val_default_disp_idx = find_default_index(["geoid", "height", "undulation", "value", "h"], stripped, 2 if len(stripped) > 2 else 0)

            st.markdown("#### 📍 Column Selection for Current Dataset")
            st.markdown(f"**Dataset:** {option}")
            
            # Create columns for better layout
            col_select1, col_select2, col_select3 = st.columns(3)
            
            with col_select1:
                st.markdown("**Latitude Column**")
                st.caption(f"Choose from {len(display_names)} available columns")
                lat_disp = st.selectbox(
                    "Select latitude column", 
                    display_names, 
                    index=lat_default_disp_idx, 
                    key=f"lat_disp_{option}",
                    label_visibility="collapsed"
                )
            
            with col_select2:
                st.markdown("**Longitude Column**")
                # For longitude, show all columns from current dataset
                lon_options = display_names.copy()
                # decide a default index for lon_options
                lon_idx_default = lon_default_disp_idx if lon_default_disp_idx < len(lon_options) else 0
                st.caption(f"Choose from {len(lon_options)} available columns")
                lon_disp = st.selectbox(
                    "Select longitude column", 
                    lon_options, 
                    index=lon_idx_default, 
                    key=f"lon_disp_{option}",
                    label_visibility="collapsed"
                )
            
            with col_select3:
                st.markdown("**Value/Thickness Column**")
                # Value column shows all columns from current dataset
                val_options = display_names.copy()
                # choose default for value column based on earlier guess
                val_idx_default = val_default_disp_idx if val_default_disp_idx < len(val_options) else 0
                st.caption(f"Choose from {len(val_options)} available columns")
                val_disp = st.selectbox(
                    f"Select {option.lower()} column", 
                    val_options, 
                    index=val_idx_default, 
                    key=f"val_disp_{option}",
                    label_visibility="collapsed"
                )

            # Map the chosen display names back to original column indices and original names
            lat_idx = display_to_index[lat_disp]
            lon_idx = display_to_index[lon_disp]
            val_idx = display_to_index[val_disp]
            # original column labels (may still have whitespace) for clarity
            lat_col_label = raw_cols[lat_idx]
            lon_col_label = raw_cols[lon_idx]
            val_col_label = raw_cols[val_idx]

            st.markdown("---")
            st.markdown("#### ⚙️ Interpolation Settings")
            
            col_interp1, col_interp2 = st.columns(2)
            
            with col_interp1:
                interp_method = st.radio(
                    "Interpolation method", 
                    ["Nearest", "Linear", "Cubic", "RBF"],
                    index=1, 
                    key=f"interp_method_{option}",
                    horizontal=True
                )
            
            with col_interp2:
                grid_res = st.slider(
                    "Grid resolution (max pixels on longer axis)",
                    min_value=50, max_value=1000, value=300, step=10, 
                    key=f"grid_res_{option}"
                )
            
            # Get data bounds for boundary suggestions
            lats_data = df_selected.iloc[:, lat_idx].to_numpy()
            lons_data = df_selected.iloc[:, lon_idx].to_numpy()
            vals_data = df_selected.iloc[:, val_idx].to_numpy()
            ok = (~np.isnan(lats_data)) & (~np.isnan(lons_data)) & (~np.isnan(vals_data))
            lats_clean = lats_data[ok]
            lons_clean = lons_data[ok]
            
            if len(lats_clean) > 0 and len(lons_clean) > 0:
                data_lon_min, data_lon_max = np.min(lons_clean), np.max(lons_clean)
                data_lat_min, data_lat_max = np.min(lats_clean), np.max(lats_clean)
                
                # Add small padding for default boundaries
                lon_pad = (data_lon_max - data_lon_min) * 0.02
                lat_pad = (data_lat_max - data_lat_min) * 0.02
                default_lon_min = max(-360.0, data_lon_min - lon_pad)
                default_lon_max = min(360.0, data_lon_max + lon_pad)
                default_lat_min = max(-90.0, data_lat_min - lat_pad)
                default_lat_max = min(90.0, data_lat_max + lat_pad)
            else:
                default_lon_min, default_lon_max = -180, 180
                default_lat_min, default_lat_max = -90, 90

            # MAP STYLE BOUNDARIES - Only show actual data extent
            st.markdown("---")
            st.markdown("#### 🗺️ Map Boundaries (Data Extent)")
            
            col_bound1, col_bound2, col_bound3, col_bound4 = st.columns(4)
            
            with col_bound1:
                lon_min = st.number_input(
                    "Longitude Min",
                    min_value=-360.0,
                    max_value=360.0,
                    value=float(default_lon_min),
                    step=0.1,
                    key=f"lon_min_{option}",
                    help="Minimum longitude of your data extent"
                )
            
            with col_bound2:
                lon_max = st.number_input(
                    "Longitude Max", 
                    min_value=-360.0,
                    max_value=360.0,
                    value=float(default_lon_max),
                    step=0.1,
                    key=f"lon_max_{option}",
                    help="Maximum longitude of your data extent"
                )
            
            with col_bound3:
                lat_min = st.number_input(
                    "Latitude Min",
                    min_value=-90.0,
                    max_value=90.0,
                    value=float(default_lat_min),
                    step=0.1,
                    key=f"lat_min_{option}",
                    help="Minimum latitude of your data extent"
                )
            
            with col_bound4:
                lat_max = st.number_input(
                    "Latitude Max",
                    min_value=-90.0,
                    max_value=90.0,
                    value=float(default_lat_max),
                    step=0.1,
                    key=f"lat_max_{option}",
                    help="Maximum latitude of your data extent"
                )
            
            # Validate boundaries
            if lon_min >= lon_max:
                st.error("❌ Longitude Min must be less than Longitude Max")
            if lat_min >= lat_max:
                st.error("❌ Latitude Min must be less than Latitude Max")
            
            # Show data extent information
            st.info(f"📍 **Data Extent:** Longitude: {data_lon_min:.2f}° to {data_lon_max:.2f}° | Latitude: {data_lat_min:.2f}° to {data_lat_max:.2f}°")

            st.markdown("#### 🎨 Plot Styling")
            
            col_style1, col_style2, col_style3, col_style4 = st.columns(4)
            
            with col_style1:
                # Geoscience-focused color schemes
                geo_colormaps = {
                    "Jet (Classic Geophysics)": "Jet",
                    "Rainbow": "Rainbow",
                    "Turbo (Enhanced Jet)": "Turbo",
                    "RdYlBu (Diverging)": "RdYlBu",
                    "RdBu (Red-Blue)": "RdBu",
                    "Spectral": "Spectral",
                    "Earth (Topography)": "Earth",
                    "Portland": "Portland",
                    "Picnic": "Picnic",
                    "Viridis": "Viridis",
                    "Plasma": "Plasma",
                    "Hot": "Hot",
                    "Cool": "Cool"
                }
                colorscale_name = st.selectbox(
                    "Colorbar scheme",
                    list(geo_colormaps.keys()),
                    index=0,
                    key=f"colorscale_{option}"
                )
                colorscale = geo_colormaps[colorscale_name]
            
            with col_style2:
                show_points = st.checkbox(
                    "Show data points", 
                    value=False, 
                    key=f"show_points_{option}"
                )
                show_contours = st.checkbox(
                    "Show contour lines",
                    value=True,
                    key=f"show_contours_{option}"
                )
            
            with col_style3:
                font_size = st.slider(
                    "Font size",
                    min_value=8, max_value=20, value=12, step=1,
                    key=f"font_size_{option}"
                )
                colorbar_thickness = st.slider(
                    "Colorbar thickness",
                    min_value=10, max_value=40, value=20, step=2,
                    key=f"colorbar_thick_{option}"
                )
            
            with col_style4:
                # Add figure size control
                fig_width = st.slider(
                    "Figure width",
                    min_value=600, max_value=1200, value=900, step=50,
                    key=f"fig_width_{option}"
                )
                fig_height = st.slider(
                    "Figure height", 
                    min_value=400, max_value=800, value=500, step=50,
                    key=f"fig_height_{option}"
                )
                
                # Add unit input for colorbar - customized for different data types
                if "thickness" in option.lower():
                    default_unit = "km"
                elif "topo" in option.lower():
                    default_unit = "m"
                elif "geoid" in option.lower():
                    default_unit = "m"
                else:
                    default_unit = "units"
                    
                colorbar_unit = st.text_input(
                    "Colorbar unit",
                    value=default_unit,
                    help="Unit to display on colorbar (e.g., km, m, mGal)",
                    key=f"unit_{option}"
                )   
                colorbar_name = st.text_input(
                    "Colorbar name",
                    help="Name to display on colorbar (e.g., km, m, mGal)",
                    key=f"Title_{option}"
                )
            
            # Advanced color scaling options for better variation
            st.markdown("#### 🔧 Color Scale Enhancement")
            col_scale1, col_scale2, col_scale3, col_scale4 = st.columns(4)
            
            with col_scale1:
                use_percentile = st.checkbox(
                    "Use percentile clipping",
                    value=True,
                    help="Remove extreme values to enhance color variation",
                    key=f"use_percentile_{option}"
                )
            
            with col_scale2:
                if use_percentile:
                    vmin_percentile = st.slider(
                        "Min percentile (%)",
                        min_value=0.0, max_value=10.0, value=2.0, step=0.5,
                        key=f"vmin_pct_{option}"
                    )
            
            with col_scale3:
                if use_percentile:
                    vmax_percentile = st.slider(
                        "Max percentile (%)",
                        min_value=90.0, max_value=100.0, value=98.0, step=0.5,
                        key=f"vmax_pct_{option}"
                    )
            
            with col_scale4:
                smooth_sigma = st.slider(
                    "Smoothing factor",
                    min_value=0.0, max_value=5.0, value=1.0, step=0.5,
                    help="Higher values = smoother interpolation",
                    key=f"smooth_{option}"
                )

            # Confirm user selections
            st.success(
                f"✅ **Current Selection Summary:**\n\n"
                f"- **Dataset:** {option}\n"
                f"- **Latitude:** {lat_disp} (column: `{lat_col_label}`)\n"
                f"- **Longitude:** {lon_disp} (column: `{lon_col_label}`)\n"
                f"- **Value:** {val_disp} (column: `{val_col_label}`)\n"
                f"- **Interpolation:** {interp_method}\n"
                f"- **Grid Resolution:** {grid_res}\n"
                f"- **Map Boundaries:** Lon({lon_min:.2f}° to {lon_max:.2f}°), Lat({lat_min:.2f}° to {lat_max:.2f}°)\n"
                f"- **Colorscale:** {colorscale}"
            )

            # Generate plot button
            if st.button("🎨 Generate Plot", key=f"genplot_{option}", type="primary"):
                with st.spinner("Running interpolation and generating plot..."):
                    # read arrays via iloc (works even with duplicate labels)
                    lats = df_selected.iloc[:, lat_idx].to_numpy()
                    lons = df_selected.iloc[:, lon_idx].to_numpy()
                    vals = df_selected.iloc[:, val_idx].to_numpy()
                    ok = (~np.isnan(lats)) & (~np.isnan(lons)) & (~np.isnan(vals))
                    lats = lats[ok]; lons = lons[ok]; vals = vals[ok]

                    if len(vals) < 3:
                        st.error("❌ Need at least 3 valid points to interpolate.")
                    else:
                        st.success(f"✅ Data ready for plotting: {len(vals)} points")

                        # Calculate grid dimensions maintaining aspect ratio
                        lon_span = lon_max - lon_min if lon_max != lon_min else 1.0
                        lat_span = lat_max - lat_min if lat_max != lat_min else 1.0
                        
                        if lon_span >= lat_span:
                            nx = grid_res
                            ny = max(10, int(np.round(grid_res * (lat_span / lon_span))))
                        else:
                            ny = grid_res
                            nx = max(10, int(np.round(grid_res * (lon_span / lat_span))))

                        xi = np.linspace(lon_min, lon_max, nx)
                        yi = np.linspace(lat_min, lat_max, ny)
                        XI, YI = np.meshgrid(xi, yi)

                        # Interpolate
                        method = interp_method.lower()
                        if method == "rbf":
                            from scipy.interpolate import Rbf
                            rbf = Rbf(lons, lats, vals, function='multiquadric')
                            ZI = rbf(XI, YI)
                        else:
                            try:
                                ZI = griddata((lons, lats), vals, (XI, YI), method=method)
                            except Exception as e:
                                st.warning(f"griddata failed ({e}), falling back to 'nearest'")
                                ZI = griddata((lons, lats), vals, (XI, YI), method='nearest')

                        # Apply smoothing with user-controlled sigma
                        if smooth_sigma > 0:
                            try:
                                ZI_s = gaussian_filter(ZI, sigma=smooth_sigma)
                            except Exception:
                                ZI_s = ZI
                        else:
                            ZI_s = ZI
                        
                        # Apply percentile clipping for MUCH better color variation
                        if use_percentile:
                            # Remove NaN for percentile calculation
                            valid_data = ZI_s[~np.isnan(ZI_s)]
                            if len(valid_data) > 0:
                                vmin = np.percentile(valid_data, vmin_percentile)
                                vmax = np.percentile(valid_data, vmax_percentile)
                                # Apply clipping to enhance variation
                                ZI_display = np.clip(ZI_s, vmin, vmax)
                                st.info(f"📊 Color range: {vmin:.2f} to {vmax:.2f} (enhancing {vmax_percentile-vmin_percentile:.1f}% of data)")
                            else:
                                ZI_display = ZI_s
                                vmin = np.nanmin(ZI_s)
                                vmax = np.nanmax(ZI_s)
                        else:
                            ZI_display = ZI_s
                            vmin = np.nanmin(ZI_s)
                            vmax = np.nanmax(ZI_s)

                        # Handle cases where vmin == vmax (constant data)
                        if vmin == vmax:
                            vmax = vmin + 1e-6
                            st.warning("⚠️ Data appears to be constant. Adding small range for visualization.")

                        # Calculate contour step safely
                        contour_step = (vmax - vmin) / 10
                        if np.isnan(contour_step) or contour_step <= 0:
                            contour_step = 1.0

                        # CREATE THE MAP-STYLE FIGURE WITH LAT/LON AXES LIKE YOUR EXAMPLE
                        fig = go.Figure()
                        
                        
                        fig.update_layout(
                            width=600,   
                            height=700,
                            
                        )

                        # Main heatmap
                        fig.add_trace(
                            go.Heatmap(
                                z=ZI_display,
                                x=xi,
                                y=yi,
                                colorscale=colorscale,
                                zmin=vmin,
                                zmax=vmax,
                                colorbar=dict(
                                    title=dict(
                                        text = f"{colorbar_name}, {colorbar_unit}",
                                        font=dict(size=font_size+1, family="Arial, sans-serif", color="black"),
                                        side="right"
                                    ),
                                    thickness=colorbar_thickness,
                                    len=0.8,
                                    x=1.02,
                                    xanchor='left',
                                    y=0.5,
                                    yanchor='middle',
                                    tickfont=dict(size=font_size, family="Arial, sans-serif", color="black"),
                                    outlinewidth=1,
                                    outlinecolor='black',
                                    tickmode='array',
                                    tickvals=np.linspace(vmin, vmax, 6),
                                    ticktext=[f"{x:.1f}" for x in np.linspace(vmin, vmax, 6)],
                                    ticks="outside",
                                    
                                ),
                                hovertemplate=f"Lon: %{{x:.2f}}°<br>Lat: %{{y:.2f}}°<br>{option}: %{{z:.2f}} {colorbar_unit}<extra></extra>"
                            )
                        )
                        
                        # Add contour lines if requested
                        if show_contours:
                            try:
                                # Generate contour levels
                                contour_levels = np.linspace(vmin, vmax, 8)
                                
                                fig.add_trace(
                                    go.Contour(
                                        z=ZI_display,
                                        x=xi,
                                        y=yi,
                                        colorscale=[[0, 'rgba(0,0,0,0)'], [1, 'rgba(0,0,0,0)']],
                                        showscale=False,
                                        contours=dict(
                                            showlabels=True,
                                            labelfont=dict(
                                                size=font_size, 
                                                color='black', 
                                                family="Arial, sans-serif"
                                            ),
                                            start=vmin,
                                            end=vmax,
                                            size=contour_step
                                        ),
                                        line=dict(color='black', width=1),
                                        hoverinfo='skip'
                                    )
                                )
                            except Exception as e:
                                st.warning(f"Could not draw contour lines: {e}")
                        
                        # Add data points if requested
                        if show_points:
                            fig.add_trace(
                                go.Scatter(
                                    x=lons, y=lats, mode='markers',
                                    marker=dict(
                                        size=4, 
                                        color='white', 
                                        line=dict(width=1, color='black'),
                                        opacity=0.8
                                    ),
                                    name='Data points',
                                    hovertemplate=f"Lon: %{{x:.2f}}°<br>Lat: %{{y:.2f}}°<br>{option}: %{{text}} {colorbar_unit}<extra></extra>",
                                    text=[f"{v:.2f}" for v in vals],
                                    showlegend=False
                                )
                            )
                        
                        
                        
                        # Display the plot
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Statistics display
                        st.markdown("---")
                        st.markdown("#### 📈 Data Statistics")
                        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                        
                        with col_stat1:
                            st.metric("Min Value", f"{np.nanmin(vals):.2f} {colorbar_unit}")
                        with col_stat2:
                            st.metric("Max Value", f"{np.nanmax(vals):.2f} {colorbar_unit}")
                        with col_stat3:
                            st.metric("Mean Value", f"{np.nanmean(vals):.2f} {colorbar_unit}")
                        with col_stat4:
                            st.metric("Std Dev", f"{np.nanstd(vals):.2f} {colorbar_unit}")
                        
                        # Store interpolated data in session state
                        interpolation_key = f"{option}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                        
                        st.session_state.interpolated_data[interpolation_key] = {
                            'data_type': option,
                            'XI': XI,
                            'YI': YI, 
                            'ZI': ZI_display,
                            'lon_min': lon_min,
                            'lon_max': lon_max,
                            'lat_min': lat_min,
                            'lat_max': lat_max,
                            'grid_res': grid_res,
                            'interp_method': interp_method,
                            'timestamp': datetime.now(),
                            'raw_data_info': {
                                'lat_col': lat_col_label,
                                'lon_col': lon_col_label,
                                'val_col': val_col_label,
                                'unit': colorbar_unit
                            }
                        }
                        
                        st.success(f"✅ Plot generated and interpolated data stored! (Key: {interpolation_key})")
                        
                        # Show stored datasets
                        st.markdown("#### 💾 Stored Interpolated Datasets")
                        stored_datasets = st.session_state.interpolated_data
                        if stored_datasets:
                            for key, data in stored_datasets.items():
                                col_store1, col_store2, col_store3 = st.columns([3, 2, 1])
                                with col_store1:
                                    st.write(f"**{data['data_type']}** - {data['timestamp'].strftime('%H:%M:%S')}")
                                with col_store2:
                                    st.write(f"Grid: {data['XI'].shape[1]}×{data['YI'].shape[0]}")
                                with col_store3:
                                    if st.button("🗑️", key=f"del_{key}"):
                                        del st.session_state.interpolated_data[key]
                                        st.rerun()
                        else:
                            st.info("No stored interpolated datasets yet.")
                        
                        # Add download option for high-resolution figure
                        st.markdown("---")
                        st.markdown("#### 💾 Export Options")
                        col_exp1, col_exp2 = st.columns(2)
                        with col_exp1:
                            dpi_export = st.slider("Export DPI (for publications)", 150, 600, 300, 50, key=f"dpi_{option}")
                        with col_exp2:
                            st.info("📥 Use the Plotly toolbar (camera icon) to download as PNG/SVG")
                        
                        st.caption("💡 **Tip:** For journal submissions, use 300+ DPI and export as SVG for vector graphics. The clean map style and enhanced color variation meet most journal requirements.")
    

        
        




# ==============================
# SECTION 5: GEOID CORRECTIONS - CORRECTED VERSION
# ==============================
elif st.session_state.current_section == "Geoid Corrections":
    st.header("🔧 Geoid Corrections")
    
    # Check if we have stored interpolated data
    stored_datasets = st.session_state.interpolated_data
    
    if not stored_datasets:
        st.error("""
        ❌ **No interpolated data found!**
        
        Please generate plots in the **Data Visualization** section first. 
        The geoid correction will use the interpolated data from your plots.
        """)
        st.info("💡 Go to 'Data Visualization' section to create interpolated datasets")
    else:
        st.success(f"✅ Found {len(stored_datasets)} stored interpolated dataset(s)")
        
        # Show available datasets
        st.markdown("#### 📋 Available Interpolated Datasets")
        
        # Group datasets by type
        crustal_sets = {k: v for k, v in stored_datasets.items() if v['data_type'] == 'Crustal thickness'}
        geoid_sets = {k: v for k, v in stored_datasets.items() if v['data_type'] == 'Geoid data'}
        topo_sets = {k: v for k, v in stored_datasets.items() if v['data_type'] == 'Topographic thickness'}
        sed_sets = {k: v for k, v in stored_datasets.items() if v['data_type'] == 'Sedimentary thickness'}
        
        col_avail1, col_avail2, col_avail3, col_avail4 = st.columns(4)
        
        with col_avail1:
            st.metric("Crustal Data", len(crustal_sets))
            for key, data in list(crustal_sets.items())[:2]:
                st.caption(f"📍 {data['timestamp'].strftime('%H:%M')} - {data['XI'].shape[1]}×{data['YI'].shape[0]}")
        
        with col_avail2:
            st.metric("Geoid Data", len(geoid_sets))
            for key, data in list(geoid_sets.items())[:2]:
                st.caption(f"📍 {data['timestamp'].strftime('%H:%M')} - {data['XI'].shape[1]}×{data['YI'].shape[0]}")
        
        with col_avail3:
            st.metric("Topography", len(topo_sets))
            for key, data in list(topo_sets.items())[:2]:
                st.caption(f"📍 {data['timestamp'].strftime('%H:%M')} - {data['XI'].shape[1]}×{data['YI'].shape[0]}")
        
        with col_avail4:
            st.metric("Sedimentary", len(sed_sets))
            for key, data in list(sed_sets.items())[:2]:
                st.caption(f"📍 {data['timestamp'].strftime('%H:%M')} - {data['XI'].shape[1]}×{data['YI'].shape[0]}")
        
        # Correction Type Selection - FIXED: Use different key for session state
        st.markdown("#### 🎯 Select Correction Type")
        
        correction_type = st.radio(
            "Choose correction type:",
            [
                "1. Topographic Correction Only",
                "2. Crustal Thickness Correction Only", 
                "3. Sedimentary Correction Only",
                "4. Combined Correction (All Three)",
                "5. Residual Geoid (Original - All Corrections)"
            ],
            index=1,  # Default to Crustal Thickness
            key="correction_type_radio"  # Changed key to avoid conflict
        )
        
        # Dataset selection based on correction type
        st.markdown("#### 📁 Select Datasets for Correction")
        
        # OPTION 1: Topographic Correction Only
        if correction_type == "1. Topographic Correction Only":
            col1, col2 = st.columns(2)
            with col1:
                if geoid_sets:
                    selected_geoid = st.selectbox(
                        "Geoid Dataset",
                        options=list(geoid_sets.keys()),
                        format_func=lambda x: f"{geoid_sets[x]['timestamp'].strftime('%H:%M:%S')} ({geoid_sets[x]['XI'].shape[1]}×{geoid_sets[x]['YI'].shape[0]})",
                        key="select_geoid_topo"
                    )
                else:
                    st.error("No geoid data available")
                    selected_geoid = None
            
            with col2:
                if topo_sets:
                    selected_topo = st.selectbox(
                        "Topography Dataset",
                        options=list(topo_sets.keys()),
                        format_func=lambda x: f"{topo_sets[x]['timestamp'].strftime('%H:%M:%S')} ({topo_sets[x]['XI'].shape[1]}×{topo_sets[x]['YI'].shape[0]})",
                        key="select_topo_only"
                    )
                else:
                    st.error("No topography data available")
                    selected_topo = None
            
            selected_crust = None
            selected_sed = None
            
        # OPTION 2: Crustal Thickness Correction Only
        elif correction_type == "2. Crustal Thickness Correction Only":
            col1, col2 = st.columns(2)
            with col1:
                if geoid_sets:
                    selected_geoid = st.selectbox(
                        "Geoid Dataset",
                        options=list(geoid_sets.keys()),
                        format_func=lambda x: f"{geoid_sets[x]['timestamp'].strftime('%H:%M:%S')} ({geoid_sets[x]['XI'].shape[1]}×{geoid_sets[x]['YI'].shape[0]})",
                        key="select_geoid_crust"
                    )
                else:
                    st.error("No geoid data available")
                    selected_geoid = None
            
            with col2:
                if crustal_sets:
                    selected_crust = st.selectbox(
                        "Crustal Thickness Dataset",
                        options=list(crustal_sets.keys()),
                        format_func=lambda x: f"{crustal_sets[x]['timestamp'].strftime('%H:%M:%S')} ({crustal_sets[x]['XI'].shape[1]}×{crustal_sets[x]['YI'].shape[0]})",
                        key="select_crust_only"
                    )
                else:
                    st.error("No crustal data available")
                    selected_crust = None
            
            selected_topo = None
            selected_sed = None
        
        # OPTION 3: Sedimentary Correction Only  
        elif correction_type == "3. Sedimentary Correction Only":
            col1, col2 = st.columns(2)
            with col1:
                if geoid_sets:
                    selected_geoid = st.selectbox(
                        "Geoid Dataset",
                        options=list(geoid_sets.keys()),
                        format_func=lambda x: f"{geoid_sets[x]['timestamp'].strftime('%H:%M:%S')} ({geoid_sets[x]['XI'].shape[1]}×{geoid_sets[x]['YI'].shape[0]})",
                        key="select_geoid_sed"
                    )
                else:
                    st.error("No geoid data available")
                    selected_geoid = None
            
            with col2:
                if sed_sets:
                    selected_sed = st.selectbox(
                        "Sedimentary Dataset",
                        options=list(sed_sets.keys()),
                        format_func=lambda x: f"{sed_sets[x]['timestamp'].strftime('%H:%M:%S')} ({sed_sets[x]['XI'].shape[1]}×{sed_sets[x]['YI'].shape[0]})",
                        key="select_sed_only"
                    )
                else:
                    st.error("No sedimentary data available")
                    selected_sed = None
            
            selected_topo = None
            selected_crust = None
        
        # OPTION 4: Combined Correction (All Three)
        elif correction_type == "4. Combined Correction (All Three)":
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if geoid_sets:
                    selected_geoid = st.selectbox(
                        "Geoid Dataset",
                        options=list(geoid_sets.keys()),
                        format_func=lambda x: f"{geoid_sets[x]['timestamp'].strftime('%H:%M:%S')} ({geoid_sets[x]['XI'].shape[1]}×{geoid_sets[x]['YI'].shape[0]})",
                        key="select_geoid_combined"
                    )
                else:
                    st.error("No geoid data")
                    selected_geoid = None
            
            with col2:
                if topo_sets:
                    selected_topo = st.selectbox(
                        "Topography Dataset",
                        options=list(topo_sets.keys()),
                        format_func=lambda x: f"{topo_sets[x]['timestamp'].strftime('%H:%M:%S')} ({topo_sets[x]['XI'].shape[1]}×{topo_sets[x]['YI'].shape[0]})",
                        key="select_topo_combined"
                    )
                else:
                    st.error("No topography data")
                    selected_topo = None
            
            with col3:
                if crustal_sets:
                    selected_crust = st.selectbox(
                        "Crustal Thickness Dataset",
                        options=list(crustal_sets.keys()),
                        format_func=lambda x: f"{crustal_sets[x]['timestamp'].strftime('%H:%M:%S')} ({crustal_sets[x]['XI'].shape[1]}×{crustal_sets[x]['YI'].shape[0]})",
                        key="select_crust_combined"
                    )
                else:
                    st.error("No crustal data")
                    selected_crust = None
            
            with col4:
                if sed_sets:
                    selected_sed = st.selectbox(
                        "Sedimentary Dataset",
                        options=list(sed_sets.keys()),
                        format_func=lambda x: f"{sed_sets[x]['timestamp'].strftime('%H:%M:%S')} ({sed_sets[x]['XI'].shape[1]}×{sed_sets[x]['YI'].shape[0]})",
                        key="select_sed_combined"
                    )
                else:
                    st.info("No sedimentary data")
                    selected_sed = None
        
        # OPTION 5: Residual Geoid
        elif correction_type == "5. Residual Geoid (Original - All Corrections)":
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if geoid_sets:
                    selected_geoid = st.selectbox(
                        "Geoid Dataset",
                        options=list(geoid_sets.keys()),
                        format_func=lambda x: f"{geoid_sets[x]['timestamp'].strftime('%H:%M:%S')} ({geoid_sets[x]['XI'].shape[1]}×{geoid_sets[x]['YI'].shape[0]})",
                        key="select_geoid_residual"
                    )
                else:
                    st.error("No geoid data")
                    selected_geoid = None
            
            with col2:
                if topo_sets:
                    selected_topo = st.selectbox(
                        "Topography Dataset",
                        options=list(topo_sets.keys()),
                        format_func=lambda x: f"{topo_sets[x]['timestamp'].strftime('%H:%M:%S')} ({topo_sets[x]['XI'].shape[1]}×{topo_sets[x]['YI'].shape[0]})",
                        key="select_topo_residual"
                    )
                else:
                    st.error("No topography data")
                    selected_topo = None
            
            with col3:
                if crustal_sets:
                    selected_crust = st.selectbox(
                        "Crustal Thickness Dataset",
                        options=list(crustal_sets.keys()),
                        format_func=lambda x: f"{crustal_sets[x]['timestamp'].strftime('%H:%M:%S')} ({crustal_sets[x]['XI'].shape[1]}×{crustal_sets[x]['YI'].shape[0]})",
                        key="select_crust_residual"
                    )
                else:
                    st.error("No crustal data")
                    selected_crust = None
            
            with col4:
                if sed_sets:
                    selected_sed = st.selectbox(
                        "Sedimentary Dataset",
                        options=list(sed_sets.keys()),
                        format_func=lambda x: f"{sed_sets[x]['timestamp'].strftime('%H:%M:%S')} ({sed_sets[x]['XI'].shape[1]}×{sed_sets[x]['YI'].shape[0]})",
                        key="select_sed_residual"
                    )
                else:
                    st.info("No sedimentary data")
                    selected_sed = None
        
        # Check if required datasets are selected
        required_datasets = {
            "1. Topographic Correction Only": [selected_geoid, selected_topo],
            "2. Crustal Thickness Correction Only": [selected_geoid, selected_crust],
            "3. Sedimentary Correction Only": [selected_geoid, selected_sed],
            "4. Combined Correction (All Three)": [selected_geoid, selected_topo, selected_crust],
            "5. Residual Geoid (Original - All Corrections)": [selected_geoid, selected_topo, selected_crust]
        }
        
        required = required_datasets.get(correction_type, [])
        
        if not all(required):
            st.error(f"❌ Please select all required datasets for {correction_type}")
        else:
            st.success("✅ All required datasets selected!")
            
            # Physical constants
            G = 6.67430e-11
            a_ell = 6378137.0
            b_ell = 6356752.314245
            e2 = 1.0 - (b_ell / a_ell) ** 2
            
            # Physical parameters expander
            st.markdown("#### 🔧 Physical Parameters")
            
            # TOPOGRAPHIC CORRECTION PARAMETERS
            if correction_type in ["1. Topographic Correction Only", "4. Combined Correction (All Three)", "5. Residual Geoid (Original - All Corrections)"]:
                with st.expander("⛰️ Topographic Correction Parameters"):
                    st.markdown("##### Density Parameters (kg/m³)")
                    
                    col_topo1, col_topo2, col_topo3 = st.columns(3)
                    
                    with col_topo1:
                        rho_rock = st.number_input(
                            "Rock Density",
                            min_value=2000,
                            max_value=3000,
                            value=2670,
                            step=10,
                            help="Surface rock density for topographic correction"
                        )
                    
                    with col_topo2:
                        rho_water = st.number_input(
                            "Water Density",
                            min_value=1000,
                            max_value=2000,
                            value=1030,
                            step=10,
                            help="Seawater density"
                        )
                    
                    with col_topo3:
                        topo_min_thickness = st.number_input(
                            "Minimum Topography (m)",
                            min_value=0.01,
                            max_value=100.0,
                            value=0.01,
                            step=0.01,
                            help="Ignore topography variations below this value"
                        )
                    
                    st.markdown("##### Computation Parameters")
                    col_topo4, col_topo5 = st.columns(2)
                    
                    with col_topo4:
                        cutoff_deg_topo = st.number_input(
                            "Angular Cutoff (degrees)",
                            min_value=1.0,
                            max_value=20.0,
                            value=12.0,
                            step=1.0,
                            help="Ignore sources beyond this angular distance"
                        )
                    
                    with col_topo5:
                        batch_size_topo = st.number_input(
                            "Batch Size",
                            min_value=1000,
                            max_value=10000,
                            value=5000,
                            step=1000,
                            help="Number of observations per batch"
                        )
            
            # CRUSTAL CORRECTION PARAMETERS
            if correction_type in ["2. Crustal Thickness Correction Only", "4. Combined Correction (All Three)", "5. Residual Geoid (Original - All Corrections)"]:
                with st.expander("🏔️ Crustal Correction Parameters"):
                    st.markdown("##### Density Parameters (kg/m³)")
                    
                    col_crust1, col_crust2, col_crust3 = st.columns(3)
                    
                    with col_crust1:
                        rho_crust = st.number_input(
                            "Crust Density",
                            min_value=2500,
                            max_value=3100,
                            value=3000,
                            step=10,
                            help="Typical continental crust density"
                        )
                    
                    with col_crust2:
                        rho_mantle = st.number_input(
                            "Mantle Density", 
                            min_value=3100,
                            max_value=3500,
                            value=3300,
                            step=10,
                            help="Upper mantle density"
                        )
                    
                    with col_crust3:
                        reference_thickness = st.number_input(
                            "Reference Crustal Thickness (km)",
                            min_value=20.0,
                            max_value=100.0,
                            value=43.0,
                            step=0.1,
                            help="Global average crustal thickness"
                        )
                    
                    st.markdown("##### Computation Parameters")
                    col_crust4, col_crust5, col_crust6 = st.columns(3)
                    
                    with col_crust4:
                        crust_min_thickness = st.number_input(
                            "Minimum Thickness Anomaly (m)",
                            min_value=100.0,
                            max_value=5000.0,
                            value=1000.0,
                            step=100.0,
                            help="Ignore crustal thickness variations below this value"
                        )
                    
                    with col_crust5:
                        cutoff_deg_crust = st.number_input(
                            "Angular Cutoff (degrees)",
                            min_value=1.0,
                            max_value=20.0,
                            value=4.0,
                            step=0.5,
                            help="Ignore sources beyond this angular distance",
                            key="cutoff_crust"
                        )
                    
                    with col_crust6:
                        batch_size_crust = st.number_input(
                            "Batch Size",
                            min_value=1000,
                            max_value=10000,
                            value=5000,
                            step=1000,
                            help="Number of observations per batch",
                            key="batch_crust"
                        )
            
            # SEDIMENTARY CORRECTION PARAMETERS
            if correction_type in ["3. Sedimentary Correction Only", "4. Combined Correction (All Three)", "5. Residual Geoid (Original - All Corrections)"] and selected_sed:
                with st.expander("🏗️ Sedimentary Correction Parameters"):
                    st.markdown("##### Density Parameters (kg/m³)")
                    
                    col_sed1, col_sed2 = st.columns(2)
                    
                    with col_sed1:
                        rho_sediment_contrast = st.number_input(
                            "Sediment Density Contrast",
                            min_value=-500,
                            max_value=0,
                            value=-200,
                            step=50,
                            help="Density contrast: sediment - crust (typically negative)"
                        )
                    
                    with col_sed2:
                        sed_min_thickness = st.number_input(
                            "Minimum Sediment Thickness (m)",
                            min_value=0.1,
                            max_value=100.0,
                            value=0.5,
                            step=0.1,
                            help="Ignore sediment thickness below this value"
                        )
                    
                    st.markdown("##### Computation Parameters")
                    col_sed3, col_sed4 = st.columns(2)
                    
                    with col_sed3:
                        cutoff_deg_sed = st.number_input(
                            "Angular Cutoff (degrees)",
                            min_value=1.0,
                            max_value=20.0,
                            value=12.0,
                            step=1.0,
                            help="Ignore sources beyond this angular distance",
                            key="cutoff_sed"
                        )
                    
                    with col_sed4:
                        batch_size_sed = st.number_input(
                            "Batch Size",
                            min_value=1000,
                            max_value=10000,
                            value=4000,
                            step=1000,
                            help="Number of observations per batch",
                            key="batch_sed"
                        )
            
            # ==============================
            # HELPER FUNCTIONS
            # ==============================
            
            def ellipsoidal_radius(lat_rad):
                """Compute ellipsoidal radius at given latitude"""
                sin_lat = np.sin(lat_rad)
                return a_ell * np.sqrt(1 - e2) / np.sqrt(1 - e2 * sin_lat * sin_lat)
            
            def somigliana_gamma(lat_rad):
                """Compute normal gravity using Somigliana's formula"""
                gamma_e = 9.7803253359
                k = 0.00193185265241
                s2 = np.sin(lat_rad) ** 2
                return gamma_e * (1 + k * s2) / np.sqrt(1 - e2 * s2)
            
            # Define Numba JIT kernels
            @jit(nopython=True)
            def tesseroid_potential_contrib(lat_obs, lon_obs, r_obs, lat_t, lon_t, r1, r2, rho, dlat, dlon):
                """Single tesseroid potential contribution (Heck & Seitz series expansion)"""
                cos_psi = (math.sin(lat_obs) * math.sin(lat_t) + 
                          math.cos(lat_obs) * math.cos(lat_t) * math.cos(lon_obs - lon_t))
                if cos_psi > 1.0:
                    cos_psi = 1.0
                elif cos_psi < -1.0:
                    cos_psi = -1.0
                psi = math.acos(cos_psi)
                r_t = 0.5 * (r1 + r2)
                l0_sq = r_obs * r_obs + r_t * r_t - 2.0 * r_obs * r_t * cos_psi
                if l0_sq <= 0.0:
                    return 0.0
                l0 = math.sqrt(l0_sq)
                if l0 < 1e-12:
                    l0 = 1e-12
                K000 = 1.0 / l0
                K200 = (3.0 * (r_obs - r_t * cos_psi) ** 2 - l0_sq) / (l0_sq ** 2.5)
                sin_psi = math.sin(psi)
                if sin_psi > 1e-12:
                    K020 = (3.0 * (r_t * psi) ** 2 - l0_sq) / (l0_sq ** 2.5)
                    K002 = (3.0 * (r_t * sin_psi) ** 2 - l0_sq) / (l0_sq ** 2.5)
                else:
                    K020 = 0.0
                    K002 = 0.0
                dr = r2 - r1
                K = K000 + (dr * dr / 24.0) * K200 + (dlat * dlat / 24.0) * K020 + (dlon * dlon / 24.0) * K002
                cos_lat_t = math.cos(lat_t)
                if cos_lat_t < 0.0:
                    cos_lat_t = 0.0
                dV = r_t * r_t * cos_lat_t * dr * dlat * dlon
                return G * rho * dV * K
            
            @jit(nopython=True, parallel=True)
            def compute_potential_batch(obs_lats_rad, obs_lons_rad, obs_radii,
                                       src_lats_rad, src_lons_rad, src_r1, src_r2, src_rho,
                                       dlat, dlon, cos_cutoff, results):
                """Compute potential for a batch of observations"""
                n_obs = len(obs_lats_rad)
                n_src = len(src_lats_rad)
                for ii in prange(n_obs):
                    ro = obs_radii[ii]
                    if np.isnan(ro):
                        results[ii] = np.nan
                        continue
                    lat_o = obs_lats_rad[ii]
                    lon_o = obs_lons_rad[ii]
                    pot_sum = 0.0
                    for ss in range(n_src):
                        r1 = src_r1[ss]
                        r2 = src_r2[ss]
                        rho = src_rho[ss]
                        if np.isnan(r1) or np.isnan(r2):
                            continue
                        cos_psi = (math.sin(lat_o) * math.sin(src_lats_rad[ss]) +
                                  math.cos(lat_o) * math.cos(src_lats_rad[ss]) * math.cos(lon_o - src_lons_rad[ss]))
                        if cos_psi < cos_cutoff:
                            continue
                        pot_sum += tesseroid_potential_contrib(lat_o, lon_o, ro, src_lats_rad[ss], src_lons_rad[ss], 
                                                              r1, r2, rho, dlat, dlon)
                    results[ii] = pot_sum

            # ==============================
            # PLOTTING SETTINGS SECTION
            # ==============================
            st.markdown("---")
            st.markdown("#### 🎨 Plotting Settings")
            
            # Color map selection for publication quality
            col_cmap1, col_cmap2, col_cmap3 = st.columns(3)

            with col_cmap1:
                # Geoid color maps
                geoid_cmaps = {
                    "Viridis (Recommended)": "viridis",
                    "Plasma": "plasma", 
                    "Inferno": "inferno",
                    "Magma": "magma",
                    "Cividis": "cividis",
                    "Jet (Classic)": "jet",
                    "Rainbow": "rainbow"
                }
                geoid_cmap = st.selectbox(
                    "Geoid Color Map",
                    options=list(geoid_cmaps.keys()),
                    index=0,
                    key="geoid_cmap_select"
                )

            with col_cmap2:
                # Topography color maps
                topo_cmaps = {
                    "Terrain (Recommended)": "terrain",
                    "Topo": "gist_earth",
                    "GrayEarth": "gist_earth",
                    "Hypsometric": "gist_earth",
                    "Relief": "terrain"
                }
                topo_cmap = st.selectbox(
                    "Topography Color Map",
                    options=list(topo_cmaps.keys()),
                    index=0,
                    key="topo_cmap_select"
                )

            with col_cmap3:
                # Correction color maps (diverging)
                correction_cmaps = {
                    "RdBu_r (Recommended)": "RdBu_r",
                    "RdBu": "RdBu",
                    "PuOr_r": "PuOr_r", 
                    "BrBG_r": "BrBG_r",
                    "Spectral_r": "Spectral_r",
                    "CoolWarm": "coolwarm",
                    "Seismic": "seismic"
                }
                correction_cmap = st.selectbox(
                    "Correction Color Map",
                    options=list(correction_cmaps.keys()),
                    index=0,
                    key="correction_cmap_select"
                )

            # Additional styling options
            st.markdown("##### 📐 Plot Styling Options")

            col_style1, col_style2, col_style3 = st.columns(3)

            with col_style1:
                font_size = st.slider(
                    "Font Size",
                    min_value=8,
                    max_value=16,
                    value=10,
                    step=1,
                    help="Font size for all text elements"
                )
                
            with col_style2:
                dpi = st.slider(
                    "Resolution (DPI)",
                    min_value=150,
                    max_value=600,
                    value=300,
                    step=50,
                    help="DPI for publication quality"
                )
                
            with col_style3:
                plot_style = st.selectbox(
                    "Plot Style",
                    ["Nature", "Classic", "Seaborn", "ggplot"],
                    index=0,
                    help="Overall plot styling theme"
                )

            # Apply selected style
            if plot_style == "Nature":
                plt.style.use('default')
                # Nature style: clean, minimal, high contrast
                nature_params = {
                    'font.size': font_size,
                    'font.family': 'sans-serif',
                    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
                    'figure.dpi': dpi,
                    'savefig.dpi': dpi,
                    'figure.figsize': (7.2, 5.4),  # Nature single column width
                    'figure.facecolor': 'white',
                    'axes.facecolor': 'white',
                    'axes.edgecolor': 'black',
                    'axes.linewidth': 0.8,
                    'axes.grid': False,
                    'grid.alpha': 0.3,
                    'xtick.color': 'black',
                    'ytick.color': 'black',
                    'xtick.direction': 'out',
                    'ytick.direction': 'out',
                    'xtick.major.width': 0.8,
                    'ytick.major.width': 0.8,
                    'lines.linewidth': 1.0,
                    'patch.linewidth': 0.8,
                    'legend.frameon': True,
                    'legend.framealpha': 0.8,
                    'legend.edgecolor': 'black'
                }
                plt.rcParams.update(nature_params)
            elif plot_style == "Seaborn":
                plt.style.use('seaborn-v0_8-whitegrid')
            elif plot_style == "ggplot":
                plt.style.use('ggplot')
            else:
                plt.style.use('classic')

            # Get actual colormap values
            geoid_cmap_val = geoid_cmaps[geoid_cmap]
            topo_cmap_val = topo_cmaps[topo_cmap]
            correction_cmap_val = correction_cmaps[correction_cmap]

            # ==============================
            # COMPUTATION BUTTON
            # ==============================
            if st.button("🚀 Compute Selected Correction", type="primary"):
                with st.spinner(f"Computing {correction_type}..."):
                    try:
                        # Extract grid from selected geoid dataset
                        geoid_data = stored_datasets[selected_geoid]
                        lons = np.unique(geoid_data['XI'])
                        lats = np.unique(geoid_data['YI'])
                        nlons, nlats = len(lons), len(lats)
                        dx_deg = lons[1] - lons[0]
                        
                        grid_lons, grid_lats = np.meshgrid(lons, lats)
                        geoid_grid = geoid_data['ZI']
                        
                        st.info(f"📐 Using grid from selected dataset: {nlats}×{nlons} = {nlats*nlons} cells, resolution = {dx_deg:.4f}°")
                        
                        # Get topography if needed
                        elev_grid = None
                        if selected_topo:
                            topo_data = stored_datasets[selected_topo]
                            # Resample to geoid grid
                            elev_grid = griddata(
                                (topo_data['XI'].flatten(), topo_data['YI'].flatten()),
                                topo_data['ZI'].flatten(),
                                (grid_lons, grid_lats),
                                method='linear'
                            )
                            mask = np.isnan(elev_grid)
                            if mask.any():
                                elev_grid[mask] = griddata(
                                    (topo_data['XI'].flatten(), topo_data['YI'].flatten()),
                                    topo_data['ZI'].flatten(),
                                    (grid_lons, grid_lats),
                                    method='nearest'
                                )[mask]
                        
                        # Get crustal thickness if needed
                        crustal_grid = None
                        if selected_crust:
                            crust_data = stored_datasets[selected_crust]
                            crustal_grid = griddata(
                                (crust_data['XI'].flatten(), crust_data['YI'].flatten()),
                                crust_data['ZI'].flatten(),
                                (grid_lons, grid_lats),
                                method='linear'
                            )
                            mask = np.isnan(crustal_grid)
                            if mask.any():
                                crustal_grid[mask] = griddata(
                                    (crust_data['XI'].flatten(), crust_data['YI'].flatten()),
                                    crust_data['ZI'].flatten(),
                                    (grid_lons, grid_lats),
                                    method='nearest'
                                )[mask]
                            # Convert to meters if in km
                            if np.nanmax(np.abs(crustal_grid)) < 100:
                                crustal_grid = crustal_grid * 1000.0
                                st.info("📏 Converted crustal thickness from km to meters")
                        
                        # Get sedimentary thickness if needed
                        sedimentary_grid = None
                        if selected_sed:
                            sed_data = stored_datasets[selected_sed]
                            sedimentary_grid = griddata(
                                (sed_data['XI'].flatten(), sed_data['YI'].flatten()),
                                sed_data['ZI'].flatten(),
                                (grid_lons, grid_lats),
                                method='linear'
                            )
                            mask = np.isnan(sedimentary_grid)
                            if mask.any():
                                sedimentary_grid[mask] = griddata(
                                    (sed_data['XI'].flatten(), sed_data['YI'].flatten()),
                                    sed_data['ZI'].flatten(),
                                    (grid_lons, grid_lats),
                                    method='nearest'
                                )[mask]
                            # Convert to meters if in km
                            if np.nanmax(np.abs(sedimentary_grid)) < 50:
                                sedimentary_grid = sedimentary_grid * 1000.0
                                st.info("📏 Converted sedimentary thickness from km to meters")
                        
                        # Prepare observation geometry
                        lats_rad = np.radians(lats)
                        ell_radii = np.array([ellipsoidal_radius(lat) for lat in lats_rad])
                        gamma_vals = np.array([somigliana_gamma(lat) for lat in lats_rad])
                        gamma_grid = gamma_vals[:, np.newaxis]
                        
                        # Safe grids for radius computation
                        geoid_safe = np.where(np.isfinite(geoid_grid), geoid_grid, 0.0)
                        elev_safe = np.where(np.isfinite(elev_grid), elev_grid, 0.0) if elev_grid is not None else np.zeros_like(geoid_grid)
                        
                        # Observation radius (at geoid surface)
                        r_obs_grid = ell_radii[:, np.newaxis] + elev_safe + geoid_safe
                        valid_obs_mask = np.isfinite(geoid_grid)
                        if elev_grid is not None:
                            valid_obs_mask &= np.isfinite(elev_grid)
                        
                        # Initialize results
                        results = {
                            'original_geoid': geoid_grid,
                            'lons': lons,
                            'lats': lats,
                            'grid_lons': grid_lons,
                            'grid_lats': grid_lats
                        }
                        
                        correction_num = correction_type.split(".")[0]
                        
                        # ==============================
                        # TOPOGRAPHIC CORRECTION
                        # ==============================
                        if correction_num == "1" or correction_num in ["4", "5"]:
                            st.info("🏔️ Computing topographic correction...")
                            
                            # Build source tesseroids
                            src_rows, src_cols = np.where(np.isfinite(elev_grid) & (np.abs(elev_grid) > topo_min_thickness))
                            n_src = len(src_rows)
                            st.info(f"Building {n_src} topographic source tesseroids...")
                            
                            src_lat_rad = np.empty(n_src, dtype=np.float64)
                            src_lon_rad = np.empty(n_src, dtype=np.float64)
                            src_r1 = np.empty(n_src, dtype=np.float64)
                            src_r2 = np.empty(n_src, dtype=np.float64)
                            src_rho = np.empty(n_src, dtype=np.float64)
                            
                            for k, (i, j) in enumerate(zip(src_rows, src_cols)):
                                H = elev_grid[i, j]
                                lat = lats[i]
                                lon = lons[j]
                                src_lat_rad[k] = math.radians(lat)
                                src_lon_rad[k] = math.radians(lon)
                                
                                Nval = geoid_grid[i, j] if np.isfinite(geoid_grid[i, j]) else 0.0
                                r_top = ell_radii[i] + H + Nval
                                r_bottom = ell_radii[i]
                                
                                src_r1[k] = min(r_top, r_bottom)
                                src_r2[k] = max(r_top, r_bottom)
                                src_rho[k] = rho_rock if H >= 0 else rho_water
                            
                            # Compute potential
                            dlat_rad = math.radians(dx_deg)
                            dlon_rad = math.radians(dx_deg)
                            cutoff_rad = math.radians(cutoff_deg_topo)
                            cos_cutoff = math.cos(cutoff_rad)
                            
                            obs_lats_rad_flat = np.radians(np.repeat(lats, nlons))
                            obs_lons_rad_flat = np.radians(np.tile(lons, nlats))
                            r_obs_flat = r_obs_grid.flatten().copy()
                            r_obs_flat[~valid_obs_mask.flatten()] = np.nan
                            
                            n_obs = len(r_obs_flat)
                            potentials_flat = np.full(n_obs, np.nan, dtype=np.float64)
                            
                            n_batches = (n_obs + batch_size_topo - 1) // batch_size_topo
                            progress_bar = st.progress(0)
                            t0 = time.time()
                            
                            for b in range(n_batches):
                                s = b * batch_size_topo
                                e = min((b+1) * batch_size_topo, n_obs)
                                results_batch = np.full(e - s, np.nan, dtype=np.float64)
                                
                                compute_potential_batch(
                                    obs_lats_rad_flat[s:e], obs_lons_rad_flat[s:e], r_obs_flat[s:e],
                                    src_lat_rad, src_lon_rad, src_r1, src_r2, src_rho,
                                    dlat_rad, dlon_rad, cos_cutoff, results_batch
                                )
                                potentials_flat[s:e] = results_batch
                                progress_bar.progress((b + 1) / n_batches)
                            
                            t_elapsed = time.time() - t0
                            st.success(f"✅ Topographic potential computed in {t_elapsed:.1f} s")
                            
                            potential_grid = potentials_flat.reshape((nlats, nlons))
                            gamma_grid_safe = np.where(gamma_grid > 1e-8, gamma_grid, 1e-8)
                            deltaN_topo = potential_grid / gamma_grid_safe
                            deltaN_topo[~valid_obs_mask] = np.nan
                            
                            results['topography'] = elev_grid
                            results['topographic_correction'] = deltaN_topo
                        
                        # ==============================
                        # CRUSTAL CORRECTION
                        # ==============================
                        if correction_num == "2" or correction_num in ["4", "5"]:
                            st.info("🌍 Computing crustal thickness correction...")
                            
                            ref_thk_m = reference_thickness * 1000.0
                            delta_rho = rho_mantle - rho_crust
                            
                            # Build source tesseroids
                            src_rows, src_cols = np.where(np.isfinite(crustal_grid))
                            n_src_candidates = len(src_rows)
                            
                            src_lat_rad = []
                            src_lon_rad = []
                            src_r1 = []
                            src_r2 = []
                            src_rho = []
                            
                            for i, j in zip(src_rows, src_cols):
                                ct = crustal_grid[i, j]
                                delta_moho = ct - ref_thk_m
                                
                                if abs(delta_moho) < crust_min_thickness:
                                    continue
                                
                                H = elev_safe[i, j]
                                N = geoid_safe[i, j]
                                
                                # Moho depth from surface
                                r_m = (ell_radii[i] + H + N) - ct
                                r_ref = (ell_radii[i] + H + N) - ref_thk_m
                                
                                if r_m <= 0 or r_ref <= 0 or abs(r_ref - r_m) < 1e-6:
                                    continue
                                
                                src_lat_rad.append(math.radians(lats[i]))
                                src_lon_rad.append(math.radians(lons[j]))
                                src_r1.append(min(r_m, r_ref))
                                src_r2.append(max(r_m, r_ref))
                                # Positive where crust < reference, negative where crust > reference
                                src_rho.append(delta_rho if ct < ref_thk_m else -delta_rho)
                            
                            src_lat_rad = np.array(src_lat_rad, dtype=np.float64)
                            src_lon_rad = np.array(src_lon_rad, dtype=np.float64)
                            src_r1 = np.array(src_r1, dtype=np.float64)
                            src_r2 = np.array(src_r2, dtype=np.float64)
                            src_rho = np.array(src_rho, dtype=np.float64)
                            n_src = len(src_r1)
                            
                            st.info(f"Building {n_src} crustal source tesseroids...")
                            
                            # Compute potential
                            dlat_rad = math.radians(dx_deg)
                            dlon_rad = math.radians(dx_deg)
                            cutoff_rad = math.radians(cutoff_deg_crust)
                            cos_cutoff = math.cos(cutoff_rad)
                            
                            obs_lats_rad_flat = np.radians(np.repeat(lats, nlons))
                            obs_lons_rad_flat = np.radians(np.tile(lons, nlats))
                            r_obs_flat = r_obs_grid.flatten().copy()
                            r_obs_flat[~valid_obs_mask.flatten()] = np.nan
                            
                            n_obs = len(r_obs_flat)
                            potentials_flat = np.full(n_obs, np.nan, dtype=np.float64)
                            
                            n_batches = (n_obs + batch_size_crust - 1) // batch_size_crust
                            progress_bar = st.progress(0)
                            t0 = time.time()
                            
                            for b in range(n_batches):
                                s = b * batch_size_crust
                                e = min((b+1) * batch_size_crust, n_obs)
                                results_batch = np.full(e - s, np.nan, dtype=np.float64)
                                
                                compute_potential_batch(
                                    obs_lats_rad_flat[s:e], obs_lons_rad_flat[s:e], r_obs_flat[s:e],
                                    src_lat_rad, src_lon_rad, src_r1, src_r2, src_rho,
                                    dlat_rad, dlon_rad, cos_cutoff, results_batch
                                )
                                potentials_flat[s:e] = results_batch
                                progress_bar.progress((b + 1) / n_batches)
                            
                            t_elapsed = time.time() - t0
                            st.success(f"✅ Crustal potential computed in {t_elapsed:.1f} s")
                            
                            potential_grid = potentials_flat.reshape((nlats, nlons))
                            gamma_grid_safe = np.where(gamma_grid > 1e-8, gamma_grid, 1e-8)
                            deltaN_crust = potential_grid / gamma_grid_safe
                            deltaN_crust[~valid_obs_mask] = np.nan
                            
                            results['crustal_thickness'] = crustal_grid
                            results['crustal_correction'] = deltaN_crust
                        
                        # ==============================
                        # SEDIMENTARY CORRECTION
                        # ==============================
                        if (correction_num == "3" or correction_num in ["4", "5"]) and sedimentary_grid is not None:
                            st.info("🏗️ Computing sedimentary correction...")
                            
                            # Build source tesseroids
                            src_rows, src_cols = np.where(sedimentary_grid > sed_min_thickness)
                            n_src = len(src_rows)
                            
                            st.info(f"Building {n_src} sedimentary source tesseroids...")
                            
                            src_lat_rad = np.empty(n_src, dtype=np.float64)
                            src_lon_rad = np.empty(n_src, dtype=np.float64)
                            src_r1 = np.empty(n_src, dtype=np.float64)
                            src_r2 = np.empty(n_src, dtype=np.float64)
                            src_rho = np.empty(n_src, dtype=np.float64)
                            
                            for k, (i, j) in enumerate(zip(src_rows, src_cols)):
                                src_lat_rad[k] = math.radians(lats[i])
                                src_lon_rad[k] = math.radians(lons[j])
                                
                                H = elev_safe[i, j]
                                N = geoid_safe[i, j]
                                r_top = ell_radii[i] + H + N
                                r_bottom = r_top - sedimentary_grid[i, j]
                                
                                if r_bottom >= r_top:
                                    src_r1[k] = np.nan
                                    src_r2[k] = np.nan
                                    src_rho[k] = 0.0
                                else:
                                    src_r1[k] = r_bottom
                                    src_r2[k] = r_top
                                    src_rho[k] = rho_sediment_contrast
                            
                            # Remove invalid
                            valid_mask = np.isfinite(src_r1) & np.isfinite(src_r2)
                            src_lat_rad = src_lat_rad[valid_mask]
                            src_lon_rad = src_lon_rad[valid_mask]
                            src_r1 = src_r1[valid_mask]
                            src_r2 = src_r2[valid_mask]
                            src_rho = src_rho[valid_mask]
                            n_src = len(src_r1)
                            
                            st.info(f"Valid sedimentary tesseroids: {n_src}")
                            
                            # Compute potential
                            dlat_rad = math.radians(dx_deg)
                            dlon_rad = math.radians(dx_deg)
                            cutoff_rad = math.radians(cutoff_deg_sed)
                            cos_cutoff = math.cos(cutoff_rad)
                            
                            obs_lats_rad_flat = np.radians(np.repeat(lats, nlons))
                            obs_lons_rad_flat = np.radians(np.tile(lons, nlats))
                            r_obs_flat = r_obs_grid.flatten().copy()
                            r_obs_flat[~valid_obs_mask.flatten()] = np.nan
                            
                            n_obs = len(r_obs_flat)
                            potentials_flat = np.full(n_obs, np.nan, dtype=np.float64)
                            
                            n_batches = (n_obs + batch_size_sed - 1) // batch_size_sed
                            progress_bar = st.progress(0)
                            t0 = time.time()
                            
                            for b in range(n_batches):
                                s = b * batch_size_sed
                                e = min((b+1) * batch_size_sed, n_obs)
                                results_batch = np.full(e - s, np.nan, dtype=np.float64)
                                
                                compute_potential_batch(
                                    obs_lats_rad_flat[s:e], obs_lons_rad_flat[s:e], r_obs_flat[s:e],
                                    src_lat_rad, src_lon_rad, src_r1, src_r2, src_rho,
                                    dlat_rad, dlon_rad, cos_cutoff, results_batch
                                )
                                potentials_flat[s:e] = results_batch
                                progress_bar.progress((b + 1) / n_batches)
                            
                            t_elapsed = time.time() - t0
                            st.success(f"✅ Sedimentary potential computed in {t_elapsed:.1f} s")
                            
                            potential_grid = potentials_flat.reshape((nlats, nlons))
                            gamma_grid_safe = np.where(gamma_grid > 1e-8, gamma_grid, 1e-8)
                            deltaN_sed = potential_grid / gamma_grid_safe
                            deltaN_sed[~valid_obs_mask] = np.nan
                            
                            results['sedimentary_thickness'] = sedimentary_grid
                            results['sedimentary_correction'] = deltaN_sed
                        
                        # ==============================
                        # ASSEMBLE FINAL RESULTS
                        # ==============================
                        if correction_num == "1":
                            results['correction'] = results['topographic_correction']
                            results['corrected_geoid'] = geoid_grid - results['topographic_correction']
                            results['total_correction'] = results['topographic_correction']
                        
                        elif correction_num == "2":
                            results['correction'] = results['crustal_correction']
                            results['corrected_geoid'] = geoid_grid - results['crustal_correction']
                            results['total_correction'] = results['crustal_correction']
                        
                        elif correction_num == "3":
                            results['correction'] = results['sedimentary_correction']
                            results['corrected_geoid'] = geoid_grid - results['sedimentary_correction']
                            results['total_correction'] = results['sedimentary_correction']
                        
                        elif correction_num == "4":
                            total_corr = results['topographic_correction'] + results['crustal_correction']
                            if 'sedimentary_correction' in results:
                                total_corr += results['sedimentary_correction']
                            results['total_correction'] = total_corr
                            results['corrected_geoid'] = geoid_grid - total_corr
                        
                        elif correction_num == "5":
                            total_corr = results['topographic_correction'] + results['crustal_correction']
                            if 'sedimentary_correction' in results:
                                total_corr += results['sedimentary_correction']
                            results['total_correction'] = total_corr
                            results['residual_geoid'] = geoid_grid - total_corr

                        # ==============================
                        # CORRECTED PLOTTING SECTION
                        # ==============================
                        st.markdown("---")
                        st.markdown(f"#### 📊 {correction_type} Results - Publication Quality")

                        # Store results in session state for replotting - FIXED: Use different keys
                        st.session_state.correction_results = results
                        st.session_state.current_correction_num = correction_num  # Changed key
                        st.session_state.current_correction_type = correction_type  # Changed key

                    except Exception as e:
                        st.error(f"❌ Error during computation: {str(e)}")
                        st.exception(e)

            # ==============================
            # PLOTTING SECTION (USES SESSION STATE)
            # ==============================
            if 'correction_results' in st.session_state:
                results = st.session_state.correction_results
                correction_num = st.session_state.get('current_correction_num', '1')  # Use get with default
                correction_type = st.session_state.get('current_correction_type', '1. Topographic Correction Only')  # Use get with default

                # CORRECTED: Use proper extent
                lon_min, lon_max = results['lons'][0], results['lons'][-1]
                lat_min, lat_max = results['lats'][0], results['lats'][-1]

                # Ensure proper orientation for all plots
                def ensure_correct_orientation(data):
                    """Ensure data is oriented correctly for plotting"""
                    # If data shape matches grid shape, use as is
                    if data.shape == results['original_geoid'].shape:
                        return data
                    # If data needs to be transposed or reshaped, handle it
                    return data

                # Apply orientation correction
                original_geoid_corrected = ensure_correct_orientation(results['original_geoid'])
                
                # FIXED: Define axes variable for all cases
                axes = None
                
                if correction_num in ["1", "2", "3"]:
                    # Create figure with Nature journal dimensions
                    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
                    
                    # Plot 1: Original Geoid - CORRECTED
                    im1 = axes[0, 0].imshow(
                        original_geoid_corrected, 
                        extent=[lon_min, lon_max, lat_min, lat_max],
                        origin='lower',
                        cmap=geoid_cmap_val,
                        aspect='auto'
                    )
                    axes[0, 0].set_title('(a) Original Geoid', fontsize=font_size+1, fontweight='bold', pad=10)
                    axes[0, 0].set_xlabel('Longitude (°)', fontsize=font_size)
                    axes[0, 0].set_ylabel('Latitude (°)', fontsize=font_size)
                    cbar1 = plt.colorbar(im1, ax=axes[0, 0], shrink=0.8, pad=0.02)
                    cbar1.set_label('Geoid Height (m)', fontsize=font_size)
                    cbar1.ax.tick_params(labelsize=font_size-1)
                    
                    # Plot 2: Source data - CORRECTED (this was already correct)
                    if correction_num == "1":
                        source_data = ensure_correct_orientation(results['topography'])
                        im2 = axes[0, 1].imshow(
                            source_data, 
                            extent=[lon_min, lon_max, lat_min, lat_max],
                            origin='lower',
                            cmap=topo_cmap_val,
                            aspect='auto'
                        )
                        axes[0, 1].set_title('(b) Topography', fontsize=font_size+1, fontweight='bold', pad=10)
                        cbar_label = 'Elevation (m)'
                    elif correction_num == "2":
                        source_data = ensure_correct_orientation(results['crustal_thickness']/1000)
                        im2 = axes[0, 1].imshow(
                            source_data, 
                            extent=[lon_min, lon_max, lat_min, lat_max],
                            origin='lower',
                            cmap='plasma',
                            aspect='auto'
                        )
                        axes[0, 1].set_title('(b) Crustal Thickness', fontsize=font_size+1, fontweight='bold', pad=10)
                        cbar_label = 'Thickness (km)'
                    else:
                        source_data = ensure_correct_orientation(results['sedimentary_thickness']/1000)
                        im2 = axes[0, 1].imshow(
                            source_data, 
                            extent=[lon_min, lon_max, lat_min, lat_max],
                            origin='lower',
                            cmap='YlOrBr',
                            aspect='auto'
                        )
                        axes[0, 1].set_title('(b) Sedimentary Thickness', fontsize=font_size+1, fontweight='bold', pad=10)
                        cbar_label = 'Thickness (km)'
                    
                    axes[0, 1].set_xlabel('Longitude (°)', fontsize=font_size)
                    axes[0, 1].set_ylabel('Latitude (°)', fontsize=font_size)
                    cbar2 = plt.colorbar(im2, ax=axes[0, 1], shrink=0.8, pad=0.02)
                    cbar2.set_label(cbar_label, fontsize=font_size)
                    cbar2.ax.tick_params(labelsize=font_size-1)
                    
                    # Plot 3: Correction - CORRECTED
                    correction_data = ensure_correct_orientation(results['correction'])
                    vmax = max(np.nanmax(np.abs(correction_data)), 1e-6)
                    im3 = axes[1, 0].imshow(
                        correction_data, 
                        extent=[lon_min, lon_max, lat_min, lat_max],
                        origin='lower',
                        cmap=correction_cmap_val,
                        vmin=-vmax, 
                        vmax=vmax,
                        aspect='auto'
                    )
                    axes[1, 0].set_title('(c) Correction', fontsize=font_size+1, fontweight='bold', pad=10)
                    axes[1, 0].set_xlabel('Longitude (°)', fontsize=font_size)
                    axes[1, 0].set_ylabel('Latitude (°)', fontsize=font_size)
                    cbar3 = plt.colorbar(im3, ax=axes[1, 0], shrink=0.8, pad=0.02)
                    cbar3.set_label('ΔN (m)', fontsize=font_size)
                    cbar3.ax.tick_params(labelsize=font_size-1)
                    
                    # Plot 4: Corrected Geoid - CORRECTED
                    corrected_geoid_data = ensure_correct_orientation(results['corrected_geoid'])
                    im4 = axes[1, 1].imshow(
                        corrected_geoid_data, 
                        extent=[lon_min, lon_max, lat_min, lat_max],
                        origin='lower',
                        cmap=geoid_cmap_val,
                        aspect='auto'
                    )
                    axes[1, 1].set_title('(d) Corrected Geoid', fontsize=font_size+1, fontweight='bold', pad=10)
                    axes[1, 1].set_xlabel('Longitude (°)', fontsize=font_size)
                    axes[1, 1].set_ylabel('Latitude (°)', fontsize=font_size)
                    cbar4 = plt.colorbar(im4, ax=axes[1, 1], shrink=0.8, pad=0.02)
                    cbar4.set_label('Geoid Height (m)', fontsize=font_size)
                    cbar4.ax.tick_params(labelsize=font_size-1)

                elif correction_num in ["4", "5"]:
                    # For combined and residual corrections - create 3x2 grid
                    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
                    
                    # Plot 1: Original Geoid
                    im1 = axes[0, 0].imshow(
                        original_geoid_corrected, 
                        extent=[lon_min, lon_max, lat_min, lat_max],
                        origin='lower',
                        cmap=geoid_cmap_val,
                        aspect='auto'
                    )
                    axes[0, 0].set_title('(a) Original Geoid', fontsize=font_size+1, fontweight='bold', pad=10)
                    axes[0, 0].set_xlabel('Longitude (°)', fontsize=font_size)
                    axes[0, 0].set_ylabel('Latitude (°)', fontsize=font_size)
                    cbar1 = plt.colorbar(im1, ax=axes[0, 0], shrink=0.8, pad=0.02)
                    cbar1.set_label('Geoid Height (m)', fontsize=font_size)
                    cbar1.ax.tick_params(labelsize=font_size-1)
                    
                    # Plot 2: Topography
                    topo_data = ensure_correct_orientation(results['topography'])
                    im2 = axes[0, 1].imshow(
                        topo_data, 
                        extent=[lon_min, lon_max, lat_min, lat_max],
                        origin='lower',
                        cmap=topo_cmap_val,
                        aspect='auto'
                    )
                    axes[0, 1].set_title('(b) Topography', fontsize=font_size+1, fontweight='bold', pad=10)
                    axes[0, 1].set_xlabel('Longitude (°)', fontsize=font_size)
                    axes[0, 1].set_ylabel('Latitude (°)', fontsize=font_size)
                    cbar2 = plt.colorbar(im2, ax=axes[0, 1], shrink=0.8, pad=0.02)
                    cbar2.set_label('Elevation (m)', fontsize=font_size)
                    cbar2.ax.tick_params(labelsize=font_size-1)
                    
                    # Plot 3: Crustal Thickness
                    crust_data = ensure_correct_orientation(results['crustal_thickness']/1000)
                    im3 = axes[1, 0].imshow(
                        crust_data, 
                        extent=[lon_min, lon_max, lat_min, lat_max],
                        origin='lower',
                        cmap='plasma',
                        aspect='auto'
                    )
                    axes[1, 0].set_title('(c) Crustal Thickness', fontsize=font_size+1, fontweight='bold', pad=10)
                    axes[1, 0].set_xlabel('Longitude (°)', fontsize=font_size)
                    axes[1, 0].set_ylabel('Latitude (°)', fontsize=font_size)
                    cbar3 = plt.colorbar(im3, ax=axes[1, 0], shrink=0.8, pad=0.02)
                    cbar3.set_label('Thickness (km)', fontsize=font_size)
                    cbar3.ax.tick_params(labelsize=font_size-1)
                    
                    # Plot 4: Sedimentary Thickness (if available)
                    if 'sedimentary_thickness' in results:
                        sed_data = ensure_correct_orientation(results['sedimentary_thickness']/1000)
                        im4 = axes[1, 1].imshow(
                            sed_data, 
                            extent=[lon_min, lon_max, lat_min, lat_max],
                            origin='lower',
                            cmap='YlOrBr',
                            aspect='auto'
                        )
                        axes[1, 1].set_title('(d) Sedimentary Thickness', fontsize=font_size+1, fontweight='bold', pad=10)
                        cbar_label = 'Thickness (km)'
                    else:
                        # Placeholder if no sedimentary data
                        im4 = axes[1, 1].imshow(
                            np.zeros_like(original_geoid_corrected), 
                            extent=[lon_min, lon_max, lat_min, lat_max],
                            origin='lower',
                            cmap='gray',
                            aspect='auto'
                        )
                        axes[1, 1].set_title('(d) No Sedimentary Data', fontsize=font_size+1, fontweight='bold', pad=10)
                        cbar_label = 'N/A'
                    
                    axes[1, 1].set_xlabel('Longitude (°)', fontsize=font_size)
                    axes[1, 1].set_ylabel('Latitude (°)', fontsize=font_size)
                    cbar4 = plt.colorbar(im4, ax=axes[1, 1], shrink=0.8, pad=0.02)
                    cbar4.set_label(cbar_label, fontsize=font_size)
                    cbar4.ax.tick_params(labelsize=font_size-1)
                    
                    # Plot 5: Total Correction
                    total_corr_data = ensure_correct_orientation(results['total_correction'])
                    vmax = max(np.nanmax(np.abs(total_corr_data)), 1e-6)
                    im5 = axes[2, 0].imshow(
                        total_corr_data, 
                        extent=[lon_min, lon_max, lat_min, lat_max],
                        origin='lower',
                        cmap=correction_cmap_val,
                        vmin=-vmax, 
                        vmax=vmax,
                        aspect='auto'
                    )
                    axes[2, 0].set_title('(e) Total Geoid Correction', fontsize=font_size+1, fontweight='bold', pad=10)
                    axes[2, 0].set_xlabel('Longitude (°)', fontsize=font_size)
                    axes[2, 0].set_ylabel('Latitude (°)', fontsize=font_size)
                    cbar5 = plt.colorbar(im5, ax=axes[2, 0], shrink=0.8, pad=0.02)
                    cbar5.set_label('ΔN (m)', fontsize=font_size)
                    cbar5.ax.tick_params(labelsize=font_size-1)
                    
                    # Plot 6: Final Result
                    if correction_num == "4":
                        final_data = ensure_correct_orientation(results['corrected_geoid'])
                        title = '(f) Corrected Geoid'
                    else:  # correction_num == "5"
                        final_data = ensure_correct_orientation(results['residual_geoid'])
                        title = '(f) Residual Geoid'
                    
                    im6 = axes[2, 1].imshow(
                        final_data, 
                        extent=[lon_min, lon_max, lat_min, lat_max],
                        origin='lower',
                        cmap=geoid_cmap_val,
                        aspect='auto'
                    )
                    axes[2, 1].set_title(title, fontsize=font_size+1, fontweight='bold', pad=10)
                    axes[2, 1].set_xlabel('Longitude (°)', fontsize=font_size)
                    axes[2, 1].set_ylabel('Latitude (°)', fontsize=font_size)
                    cbar6 = plt.colorbar(im6, ax=axes[2, 1], shrink=0.8, pad=0.02)
                    cbar6.set_label('Geoid Height (m)', fontsize=font_size)
                    cbar6.ax.tick_params(labelsize=font_size-1)

                # FIXED: Only apply styling if axes is defined
                if axes is not None:
                    # Apply consistent styling to all subplots
                    if isinstance(axes, np.ndarray):
                        for ax in axes.flat:
                            ax.tick_params(axis='both', which='major', labelsize=font_size-1)
                            ax.grid(False)
                    else:
                        axes.tick_params(axis='both', which='major', labelsize=font_size-1)
                        axes.grid(False)

                    plt.tight_layout(pad=2.0, w_pad=1.5, h_pad=1.5)
                    st.pyplot(fig)

                    # Download options
                    st.markdown("##### 💾 Download Options")
                    
                    col_dl1, col_dl2, col_dl3 = st.columns(3)
                    
                    with col_dl1:
                        buf_png = io.BytesIO()
                        fig.savefig(buf_png, format='png', dpi=dpi, bbox_inches='tight')
                        buf_png.seek(0)
                        st.download_button(
                            label=f"📥 Download PNG ({dpi} DPI)",
                            data=buf_png,
                            file_name=f"geoid_correction_{correction_num}.png",
                            mime="image/png"
                        )
                    
                    with col_dl2:
                        buf_pdf = io.BytesIO()
                        fig.savefig(buf_pdf, format='pdf', bbox_inches='tight')
                        buf_pdf.seek(0)
                        st.download_button(
                            label="📥 Download PDF",
                            data=buf_pdf,
                            file_name=f"geoid_correction_{correction_num}.pdf",
                            mime="application/pdf"
                        )
                    
                    with col_dl3:
                        # Prepare CSV download
                        download_data = {
                            'Longitude': results['grid_lons'].flatten(),
                            'Latitude': results['grid_lats'].flatten(),
                            'Original_Geoid': results['original_geoid'].flatten()
                        }
                        
                        if 'corrected_geoid' in results:
                            download_data['Corrected_Geoid'] = results['corrected_geoid'].flatten()
                        if 'correction' in results:
                            download_data['Correction'] = results['correction'].flatten()
                        if 'total_correction' in results:
                            download_data['Total_Correction'] = results['total_correction'].flatten()
                        if 'residual_geoid' in results:
                            download_data['Residual_Geoid'] = results['residual_geoid'].flatten()
                        
                        df_download = pd.DataFrame(download_data)
                        csv_data = df_download.to_csv(index=False)
                        
                        st.download_button(
                            label="📥 Download CSV",
                            data=csv_data,
                            file_name=f"geoid_correction_{correction_num}.csv",
                            mime="text/csv"
                        )
                else:
                    st.error("❌ Plotting failed: axes not properly initialized")























# ==============================
# SECTION 6: DRAW PROFILING LINE - ENHANCED VERSION (NO GEOPY DEPENDENCY)
# ==============================
elif st.session_state.current_section == "Draw Profiling Line":
    st.header("📈 Draw Profiling Line - Enhanced Analysis")
    
    # Check if we have correction results
    if 'correction_results' not in st.session_state:
        st.error("""
        ❌ **No geoid correction results found!**
        
        Please compute geoid corrections first in the **Geoid Corrections** section.
        The profiling tool will work on the most recent correction results.
        """)
        st.info("💡 Go to 'Geoid Corrections' section to compute corrections first")
    else:
        st.success("✅ Found geoid correction results! You can now draw and analyze multiple profiles.")
        
        results = st.session_state.correction_results
        correction_num = st.session_state.get('current_correction_num', '1')
        correction_type = st.session_state.get('current_correction_type', '1. Topographic Correction Only')
        
        # Initialize session state for profiles
        if 'profiles' not in st.session_state:
            st.session_state.profiles = {}
        
        # Available data fields for profiling - COMPREHENSIVE LIST
        available_fields = {
            'Original Geoid': {
                'data': results['original_geoid'],
                'unit': 'm',
                'description': 'Original geoid height'
            },
            'Sedimentary Thickness': {
                'data': results.get('sedimentary_thickness', np.zeros_like(results['original_geoid'])),
                'unit': 'm',
                'description': 'Sedimentary thickness'
            },
            'Crustal Thickness': {
                'data': results.get('crustal_thickness', np.zeros_like(results['original_geoid'])),
                'unit': 'm',
                'description': 'Crustal thickness'
            },
            'Topography': {
                'data': results.get('topography', np.zeros_like(results['original_geoid'])),
                'unit': 'm',
                'description': 'Topographic elevation'
            },
            'Sedimentary Correction': {
                'data': results.get('sedimentary_correction', np.zeros_like(results['original_geoid'])),
                'unit': 'm',
                'description': 'Sedimentary correction to geoid'
            },
            'Crustal Correction': {
                'data': results.get('crustal_correction', np.zeros_like(results['original_geoid'])),
                'unit': 'm',
                'description': 'Crustal correction to geoid'
            },
            'Topographic Correction': {
                'data': results.get('topographic_correction', np.zeros_like(results['original_geoid'])),
                'unit': 'm',
                'description': 'Topographic correction to geoid'
            },
            'Total Correction': {
                'data': results.get('total_correction', np.zeros_like(results['original_geoid'])),
                'unit': 'm',
                'description': 'Total geoid correction (all corrections combined)'
            },
            'Corrected Geoid': {
                'data': results.get('corrected_geoid', np.zeros_like(results['original_geoid'])),
                'unit': 'm',
                'description': 'Corrected geoid height (original - total correction)'
            },
            'Residual Geoid': {
                'data': results.get('residual_geoid', np.zeros_like(results['original_geoid'])),
                'unit': 'm',
                'description': 'Residual geoid (original - all corrections)'
            }
        }
        
        # Remove fields that are all zeros (not computed) and handle unit conversions
        cleaned_fields = {}
        for k, v in available_fields.items():
            if not np.all(v['data'] == 0):
                # Convert thickness fields from meters to kilometers for better readability
                if 'Thickness' in k and np.nanmax(np.abs(v['data'])) > 1000:
                    cleaned_fields[k] = {
                        'data': v['data'] / 1000.0,  # Convert to km
                        'unit': 'km',
                        'description': v['description']
                    }
                else:
                    cleaned_fields[k] = v
        
        available_fields = cleaned_fields
        
        st.markdown("#### 🎯 Select Data Fields for Profiling")
        
        # Field selection with organized grouping
        col_field1, col_field2 = st.columns(2)
        
        with col_field1:
            # Organize fields by category for better UX
            field_categories = {
                "Geoid Fields": ['Original Geoid', 'Corrected Geoid', 'Residual Geoid'],
                "Thickness Fields": [f for f in available_fields.keys() if 'Thickness' in f],
                "Correction Fields": [f for f in available_fields.keys() if 'Correction' in f and f != 'Total Correction'],
                "Total Corrections": ['Total Correction']
            }
            
            st.markdown("**Available Field Categories:**")
            for category, fields in field_categories.items():
                if any(f in available_fields for f in fields):
                    st.caption(f"**{category}:** {', '.join([f for f in fields if f in available_fields])}")
        
        with col_field2:
            selected_fields = st.multiselect(
                "Data Fields to Profile",
                options=list(available_fields.keys()),
                default=[
                    'Original Geoid', 
                    'Sedimentary Thickness', 
                    'Crustal Thickness', 
                    'Topography',
                    'Total Correction',
                    'Corrected Geoid'
                ] if all(f in available_fields for f in ['Original Geoid', 'Sedimentary Thickness', 'Crustal Thickness', 'Topography', 'Total Correction', 'Corrected Geoid']) 
                else list(available_fields.keys())[:min(4, len(available_fields))],
                help="Select multiple fields to plot together. Thickness fields are automatically converted to km when appropriate."
            )
            
            # Display field information
            if selected_fields:
                st.markdown("**Selected Fields Info:**")
                for field in selected_fields:
                    info = available_fields[field]
                    data_min = np.nanmin(info['data'])
                    data_max = np.nanmax(info['data'])
                    st.caption(f"**{field}**: {info['description']} | Range: {data_min:.1f} to {data_max:.1f} {info['unit']}")
        
        st.markdown("---")
        st.markdown("#### 📐 Profile Line Management")
        
        # Haversine distance function (replacement for geopy)
        def haversine_distance(lon1, lat1, lon2, lat2):
            """
            Calculate the great-circle distance between two points 
            on the Earth (specified in decimal degrees)
            """
            # Convert decimal degrees to radians
            lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
            
            # Haversine formula
            dlon = lon2 - lon1
            dlat = lat2 - lat1
            a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
            c = 2 * math.asin(math.sqrt(a))
            r = 6371  # Radius of Earth in kilometers
            return c * r
        
        # Profile creation section
        with st.expander("➕ Add New Profile Line", expanded=True):
            col_profile1, col_profile2 = st.columns(2)
            
            with col_profile1:
                st.markdown("**Profile Definition**")
                profile_name = st.text_input(
                    "Profile Name",
                    value=f"Profile_{len(st.session_state.profiles) + 1}",
                    help="Give a meaningful name to this profile (e.g., 'A-A', 'CrossSection_1')"
                )
                
                st.markdown("**Start Point**")
                start_lon = st.number_input(
                    "Start Longitude (°)",
                    min_value=float(results['lons'][0]),
                    max_value=float(results['lons'][-1]),
                    value=float(results['lons'][0]),
                    step=0.1,
                    key="start_lon_new"
                )
                start_lat = st.number_input(
                    "Start Latitude (°)",
                    min_value=float(results['lats'][0]),
                    max_value=float(results['lats'][-1]),
                    value=float(results['lats'][0]),
                    step=0.1,
                    key="start_lat_new"
                )
            
            with col_profile2:
                st.markdown("**Profile Settings**")
                profile_color = st.color_picker(
                    "Profile Color",
                    value="#FF0000",  # Red
                    key=f"color_{profile_name}"
                )
                
                line_style = st.selectbox(
                    "Line Style",
                    ["solid", "dashed", "dotted", "dashdot"],
                    key=f"style_{profile_name}"
                )
                
                st.markdown("**End Point**")
                end_lon = st.number_input(
                    "End Longitude (°)",
                    min_value=float(results['lons'][0]),
                    max_value=float(results['lons'][-1]),
                    value=float(results['lons'][-1]),
                    step=0.1,
                    key="end_lon_new"
                )
                end_lat = st.number_input(
                    "End Latitude (°)",
                    min_value=float(results['lats'][0]),
                    max_value=float(results['lats'][-1]),
                    value=float(results['lats'][-1]),
                    step=0.1,
                    key="end_lat_new"
                )
            
            # Profile parameters
            col_param1, col_param2 = st.columns(2)
            
            with col_param1:
                num_points = st.slider(
                    "Number of profile points",
                    min_value=10,
                    max_value=500,
                    value=100,
                    step=10,
                    help="More points = smoother profile"
                )
            
            with col_param2:
                distance_units = st.selectbox(
                    "Distance units",
                    ["kilometers", "meters"],
                    index=0
                )
            
            if st.button("✅ Add Profile", type="primary"):
                if profile_name in st.session_state.profiles:
                    st.error(f"Profile name '{profile_name}' already exists. Please choose a different name.")
                else:
                    # Calculate profile line
                    profile_lons = np.linspace(start_lon, end_lon, num_points)
                    profile_lats = np.linspace(start_lat, end_lat, num_points)
                    
                    # Calculate distances using Haversine formula
                    distances = [0.0]  # Start with 0
                    for i in range(1, num_points):
                        dist_km = haversine_distance(
                            profile_lons[i-1], profile_lats[i-1],
                            profile_lons[i], profile_lats[i]
                        )
                        cumulative_dist = distances[i-1] + dist_km
                        distances.append(cumulative_dist)
                    
                    if distance_units == "meters":
                        distances = [d * 1000 for d in distances]
                    
                    # Store profile
                    st.session_state.profiles[profile_name] = {
                        'start_lon': start_lon,
                        'start_lat': start_lat,
                        'end_lon': end_lon,
                        'end_lat': end_lat,
                        'profile_lons': profile_lons,
                        'profile_lats': profile_lats,
                        'distances': distances,
                        'color': profile_color,
                        'line_style': line_style,
                        'num_points': num_points,
                        'distance_units': distance_units,
                        'values': {}  # Will store extracted values for each field
                    }
                    st.success(f"✅ Profile '{profile_name}' added successfully!")
                    st.rerun()
        
        # Display and manage existing profiles
        if st.session_state.profiles:
            st.markdown("#### 📋 Existing Profiles")
            
            # Create tabs for profile management and visualization
            tab_manage, tab_visualize, tab_compare = st.tabs(["🗂️ Manage Profiles", "📊 View Profiles", "📈 Compare Profiles"])
            
            with tab_manage:
                st.markdown("##### Manage Your Profiles")
                
                # Display profiles in a nice layout
                cols = st.columns(3)
                profile_names = list(st.session_state.profiles.keys())
                
                for i, profile_name in enumerate(profile_names):
                    with cols[i % 3]:
                        profile = st.session_state.profiles[profile_name]
                        
                        st.markdown(f"**{profile_name}**")
                        st.caption(f"Start: ({profile['start_lon']:.2f}°, {profile['start_lat']:.2f}°)")
                        st.caption(f"End: ({profile['end_lon']:.2f}°, {profile['end_lat']:.2f}°)")
                        st.caption(f"Length: {profile['distances'][-1]:.1f} {profile['distance_units']}")
                        st.caption(f"Points: {profile['num_points']}")
                        
                        # Color preview
                        st.markdown(
                            f"<div style='background-color: {profile['color']}; height: 20px; border-radius: 3px;'></div>",
                            unsafe_allow_html=True
                        )
                        
                        col_edit, col_del = st.columns(2)
                        with col_edit:
                            if st.button("✏️ Edit", key=f"edit_{profile_name}"):
                                # For simplicity, we'll just remove and let user recreate
                                st.session_state.profiles.pop(profile_name)
                                st.rerun()
                        with col_del:
                            if st.button("🗑️ Delete", key=f"del_{profile_name}"):
                                st.session_state.profiles.pop(profile_name)
                                st.rerun()
                
                # Clear all button
                if st.button("🗑️ Clear All Profiles", type="secondary"):
                    st.session_state.profiles = {}
                    st.rerun()
            
            with tab_visualize:
                st.markdown("##### Individual Profile Visualization")
                
                if selected_fields:
                    # Let user select which profile to visualize
                    selected_profile = st.selectbox(
                        "Select Profile to Visualize",
                        options=list(st.session_state.profiles.keys())
                    )
                    
                    if selected_profile:
                        profile = st.session_state.profiles[selected_profile]
                        
                        # Extract values for selected fields
                        for field in selected_fields:
                            if field not in profile['values']:
                                field_values = []
                                for i in range(profile['num_points']):
                                    lon_idx = np.argmin(np.abs(results['lons'] - profile['profile_lons'][i]))
                                    lat_idx = np.argmin(np.abs(results['lats'] - profile['profile_lats'][i]))
                                    field_values.append(available_fields[field]['data'][lat_idx, lon_idx])
                                profile['values'][field] = field_values
                        
                        # ==============================
                        # NEW LAYOUT: Map first, then profile plots with individual scales
                        # ==============================
                        
                        # Create map with profile line - SHOW THIS FIRST
                        st.markdown("#### 🗺️ Profile Location Map")
                        fig_map, ax_map = plt.subplots(figsize=(10, 6))
                        
                        # Plot the first selected field as background
                        first_field = selected_fields[0]
                        im = ax_map.imshow(
                            available_fields[first_field]['data'],
                            extent=[results['lons'][0], results['lons'][-1], 
                                   results['lats'][0], results['lats'][-1]],
                            origin='lower',
                            cmap='viridis',
                            aspect='auto'
                        )
                        
                        # Plot the profile line
                        ax_map.plot(profile['profile_lons'], profile['profile_lats'], 
                                  color=profile['color'], linewidth=3, 
                                  linestyle=profile['line_style'], label=selected_profile)
                        ax_map.plot([profile['start_lon']], [profile['start_lat']], 
                                  'o', color=profile['color'], markersize=8, label='Start')
                        ax_map.plot([profile['end_lon']], [profile['end_lat']], 
                                  's', color=profile['color'], markersize=8, label='End')
                        
                        ax_map.set_xlabel('Longitude (°)', fontsize=12)
                        ax_map.set_ylabel('Latitude (°)', fontsize=12)
                        ax_map.set_title(f'Profile Location - {first_field}', fontsize=14)
                        ax_map.legend()
                        plt.colorbar(im, ax=ax_map, label=f"{first_field} ({available_fields[first_field]['unit']})")
                        
                        plt.tight_layout()
                        st.pyplot(fig_map)
                        
                        # ==============================
                        # PLOT DOWNLOAD OPTIONS
                        # ==============================
                        st.markdown("#### 💾 Download Plot")
                        col_dpi1, col_dl1 = st.columns(2)
                        
                        with col_dpi1:
                            dpi_map = st.slider(
                                "Map DPI (Resolution)",
                                min_value=100,
                                max_value=600,
                                value=300,
                                key="dpi_map"
                            )
                        
                        with col_dl1:
                            # Create download buffer for map
                            buf_map = io.BytesIO()
                            fig_map.savefig(buf_map, format="png", dpi=dpi_map, bbox_inches='tight')
                            st.download_button(
                                label="📥 Download Map",
                                data=buf_map.getvalue(),
                                file_name=f"{selected_profile}_map.png",
                                mime="image/png"
                            )
                        
                        # ==============================
                        # TWO-COLUMN LAYOUT: Statistics and Profile Plots
                        # ==============================
                        col_stats, col_plots = st.columns([1, 2])
                        
                        with col_stats:
                            st.markdown("#### 📊 Profile Statistics")
                            
                            # Profile metadata
                            st.markdown("**Profile Information**")
                            st.write(f"**Name:** {selected_profile}")
                            st.write(f"**Start:** ({profile['start_lon']:.2f}°, {profile['start_lat']:.2f}°)")
                            st.write(f"**End:** ({profile['end_lon']:.2f}°, {profile['end_lat']:.2f}°)")
                            st.write(f"**Length:** {profile['distances'][-1]:.1f} {profile['distance_units']}")
                            st.write(f"**Points:** {profile['num_points']}")
                            
                            st.markdown("---")
                            st.markdown("**Field Statistics**")
                            
                            # Create statistics for each field
                            stats_data = []
                            for field in selected_fields:
                                values = profile['values'][field]
                                field_info = available_fields[field]
                                
                                stats_data.append({
                                    'Field': field,
                                    'Unit': field_info['unit'],
                                    'Min': f"{np.nanmin(values):.2f}",
                                    'Max': f"{np.nanmax(values):.2f}",
                                    'Mean': f"{np.nanmean(values):.2f}",
                                    'Std': f"{np.nanstd(values):.2f}",
                                    'Range': f"{np.nanmax(values) - np.nanmin(values):.2f}"
                                })
                            
                            # Display statistics in a clean format
                            for field in selected_fields:
                                values = profile['values'][field]
                                field_info = available_fields[field]
                                
                                with st.expander(f"📈 {field} Statistics", expanded=False):
                                    col_stat1, col_stat2 = st.columns(2)
                                    
                                    with col_stat1:
                                        st.metric("Minimum", f"{np.nanmin(values):.2f} {field_info['unit']}")
                                        st.metric("Maximum", f"{np.nanmax(values):.2f} {field_info['unit']}")
                                    
                                    with col_stat2:
                                        st.metric("Mean", f"{np.nanmean(values):.2f} {field_info['unit']}")
                                        st.metric("Std Dev", f"{np.nanstd(values):.2f} {field_info['unit']}")
                        
                        with col_plots:
                            st.markdown("#### 📈 Parameter Profiles with Symmetrical Layout")
                            
                            # ==============================
                            # MODIFIED: Create symmetrical grid layout for profile plots
                            # ==============================
                            
                            # Determine optimal grid layout
                            num_fields = len(selected_fields)
                            
                            # Calculate number of columns and rows for symmetrical layout
                            if num_fields <= 2:
                                ncols = 2
                                nrows = 1
                            elif num_fields <= 4:
                                ncols = 2
                                nrows = 2
                            elif num_fields <= 6:
                                ncols = 3
                                nrows = 2
                            elif num_fields <= 9:
                                ncols = 3
                                nrows = 3
                            else:
                                ncols = 4
                                nrows = (num_fields + 3) // 4  # Ceiling division
                            
                            # Adjust figure size based on grid size
                            fig_width = 6 * ncols
                            fig_height = 4 * nrows
                            
                            # Create subplots with symmetrical grid
                            fig_profiles, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height))
                            
                            # Handle single subplot case
                            if num_fields == 1:
                                axes = np.array([axes])
                            
                            # Flatten axes array for easy iteration
                            if nrows > 1 and ncols > 1:
                                axes_flat = axes.flatten()
                            else:
                                axes_flat = axes if isinstance(axes, np.ndarray) else [axes]
                            
                            # Plot each field in its own subplot
                            for idx, field in enumerate(selected_fields):
                                if idx < len(axes_flat):
                                    field_info = available_fields[field]
                                    values = profile['values'][field]
                                    
                                    # Plot the field
                                    axes_flat[idx].plot(profile['distances'], values, 
                                                      color=profile['color'], linewidth=2, 
                                                      linestyle=profile['line_style'])
                                    
                                    axes_flat[idx].set_ylabel(f'{field} ({field_info["unit"]})', fontsize=10)
                                    axes_flat[idx].grid(True, alpha=0.3)
                                    axes_flat[idx].set_title(f'{field} Profile', fontsize=12)
                                    
                                    # Add statistics to the plot
                                    stats_text = f'Mean: {np.nanmean(values):.2f} {field_info["unit"]}\n'
                                    stats_text += f'Range: {np.nanmax(values) - np.nanmin(values):.2f} {field_info["unit"]}'
                                    
                                    axes_flat[idx].text(0.02, 0.98, stats_text, transform=axes_flat[idx].transAxes,
                                                      verticalalignment='top', fontsize=8,
                                                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                            
                            # Hide any unused subplots
                            for idx in range(len(selected_fields), len(axes_flat)):
                                axes_flat[idx].set_visible(False)
                            
                            # Set common x-labels for bottom row only
                            if nrows > 1:
                                bottom_axes = axes_flat[-ncols:] if nrows > 1 else axes_flat
                            else:
                                bottom_axes = axes_flat
                            
                            for ax in bottom_axes:
                                if ax.get_visible():  # Only set label for visible axes
                                    ax.set_xlabel(f'Distance along profile ({profile["distance_units"]})', fontsize=10)
                            
                            plt.suptitle(f'Profile: {selected_profile}\n'
                                       f'Start: ({profile["start_lon"]:.2f}°, {profile["start_lat"]:.2f}°) → '
                                       f'End: ({profile["end_lon"]:.2f}°, {profile["end_lat"]:.2f}°)', 
                                       fontsize=14, y=0.98)
                            
                            plt.tight_layout()
                            st.pyplot(fig_profiles)
                            
                            # ==============================
                            # PLOT DOWNLOAD OPTIONS
                            # ==============================
                            col_dpi2, col_dl2 = st.columns(2)
                            
                            with col_dpi2:
                                dpi_profiles = st.slider(
                                    "Profiles DPI (Resolution)",
                                    min_value=100,
                                    max_value=600,
                                    value=300,
                                    key="dpi_profiles"
                                )
                            
                            with col_dl2:
                                # Create download buffer for profiles
                                buf_profiles = io.BytesIO()
                                fig_profiles.savefig(buf_profiles, format="png", dpi=dpi_profiles, bbox_inches='tight')
                                st.download_button(
                                    label="📥 Download Profile Plots",
                                    data=buf_profiles.getvalue(),
                                    file_name=f"{selected_profile}_profiles.png",
                                    mime="image/png"
                                )
                            
                            # ==============================
                            # ADDITIONAL: Combined plot for comparison (optional)
                            # ==============================
                            with st.expander("📊 Combined Profile View (All Fields)", expanded=False):
                                st.markdown("**All parameters on single plot (normalized)**")
                                
                                fig_combined, ax_combined = plt.subplots(figsize=(12, 6))
                                
                                # Plot normalized values for comparison
                                for field in selected_fields:
                                    field_info = available_fields[field]
                                    values = profile['values'][field]
                                    
                                    # Normalize values to 0-1 range for comparison
                                    if np.nanmax(values) != np.nanmin(values):
                                        normalized_values = (values - np.nanmin(values)) / (np.nanmax(values) - np.nanmin(values))
                                    else:
                                        normalized_values = np.zeros_like(values)
                                    
                                    ax_combined.plot(profile['distances'], normalized_values, 
                                                   linewidth=2, label=f"{field} ({field_info['unit']})")
                                
                                ax_combined.set_xlabel(f'Distance along profile ({profile["distance_units"]})', fontsize=12)
                                ax_combined.set_ylabel('Normalized Value (0-1)', fontsize=12)
                                ax_combined.grid(True, alpha=0.3)
                                ax_combined.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                                ax_combined.set_title(f'Normalized Profile Comparison: {selected_profile}', fontsize=14)
                                
                                plt.tight_layout()
                                st.pyplot(fig_combined)
                                
                                # Download option for combined plot
                                col_dpi3, col_dl3 = st.columns(2)
                                
                                with col_dpi3:
                                    dpi_combined = st.slider(
                                        "Combined Plot DPI",
                                        min_value=100,
                                        max_value=600,
                                        value=300,
                                        key="dpi_combined"
                                    )
                                
                                with col_dl3:
                                    buf_combined = io.BytesIO()
                                    fig_combined.savefig(buf_combined, format="png", dpi=dpi_combined, bbox_inches='tight')
                                    st.download_button(
                                        label="📥 Download Combined Plot",
                                        data=buf_combined.getvalue(),
                                        file_name=f"{selected_profile}_combined.png",
                                        mime="image/png"
                                    )
                        
                        # ==============================
                        # DATA DOWNLOAD FOR THIS PROFILE
                        # ==============================
                        st.markdown("---")
                        st.markdown("#### 💾 Download This Profile Data")
                        
                        # Create DataFrame for this specific profile
                        profile_data = {
                            'distance': profile['distances'],
                            'longitude': profile['profile_lons'],
                            'latitude': profile['profile_lats']
                        }
                        
                        for field in selected_fields:
                            profile_data[field] = profile['values'][field]
                        
                        df_profile = pd.DataFrame(profile_data)
                        csv_profile = df_profile.to_csv(index=False)
                        
                        col_dl1, col_dl2 = st.columns(2)
                        
                        with col_dl1:
                            st.download_button(
                                label="📥 Download Profile Data (CSV)",
                                data=csv_profile,
                                file_name=f"{selected_profile}_data.csv",
                                mime="text/csv",
                                key=f"dl_single_{selected_profile}"
                            )
                        
                        with col_dl2:
                            # Create summary statistics for download
                            summary_stats = []
                            for field in selected_fields:
                                values = profile['values'][field]
                                field_info = available_fields[field]
                                
                                summary_stats.append({
                                    'Profile': selected_profile,
                                    'Field': field,
                                    'Unit': field_info['unit'],
                                    'Min': np.nanmin(values),
                                    'Max': np.nanmax(values),
                                    'Mean': np.nanmean(values),
                                    'Std': np.nanstd(values),
                                    'Start_Lon': profile['start_lon'],
                                    'Start_Lat': profile['start_lat'],
                                    'End_Lon': profile['end_lon'],
                                    'End_Lat': profile['end_lat'],
                                    'Length': profile['distances'][-1],
                                    'Length_Units': profile['distance_units']
                                })
                            
                            df_summary = pd.DataFrame(summary_stats)
                            csv_summary = df_summary.to_csv(index=False)
                            
                            st.download_button(
                                label="📥 Download Statistics (CSV)",
                                data=csv_summary,
                                file_name=f"{selected_profile}_statistics.csv",
                                mime="text/csv",
                                key=f"dl_stats_{selected_profile}"
                            )
            
            with tab_compare:
                st.markdown("##### Compare Multiple Profiles")
                
                if len(st.session_state.profiles) > 1 and selected_fields:
                    # Select which profiles to compare
                    profiles_to_compare = st.multiselect(
                        "Select Profiles to Compare",
                        options=list(st.session_state.profiles.keys()),
                        default=list(st.session_state.profiles.keys())[:2]
                    )
                    
                    field_to_compare = st.selectbox(
                        "Field to Compare",
                        options=selected_fields
                    )
                    
                    if profiles_to_compare and field_to_compare:
                        # Create comparison plot
                        fig_compare, ax_compare = plt.subplots(figsize=(12, 6))
                        
                        for profile_name in profiles_to_compare:
                            profile = st.session_state.profiles[profile_name]
                            
                            # Ensure values are extracted
                            if field_to_compare not in profile['values']:
                                field_values = []
                                for i in range(profile['num_points']):
                                    lon_idx = np.argmin(np.abs(results['lons'] - profile['profile_lons'][i]))
                                    lat_idx = np.argmin(np.abs(results['lats'] - profile['profile_lats'][i]))
                                    field_values.append(available_fields[field_to_compare]['data'][lat_idx, lon_idx])
                                profile['values'][field_to_compare] = field_values
                            
                            # Normalize distances for comparison
                            normalized_distances = np.array(profile['distances']) / profile['distances'][-1]
                            
                            ax_compare.plot(normalized_distances, profile['values'][field_to_compare],
                                          color=profile['color'], linestyle=profile['line_style'],
                                          linewidth=2, label=profile_name)
                        
                        ax_compare.set_xlabel('Normalized Distance (0 = Start, 1 = End)', fontsize=12)
                        ax_compare.set_ylabel(f'{field_to_compare} ({available_fields[field_to_compare]["unit"]})', fontsize=12)
                        ax_compare.grid(True, alpha=0.3)
                        ax_compare.legend()
                        ax_compare.set_title(f'Comparison of {field_to_compare} across Profiles', fontsize=14)
                        
                        plt.tight_layout()
                        st.pyplot(fig_compare)
                        
                        # Download option for comparison plot
                        col_dpi_compare, col_dl_compare = st.columns(2)
                        
                        with col_dpi_compare:
                            dpi_compare = st.slider(
                                "Comparison Plot DPI",
                                min_value=100,
                                max_value=600,
                                value=300,
                                key="dpi_compare"
                            )
                        
                        with col_dl_compare:
                            buf_compare = io.BytesIO()
                            fig_compare.savefig(buf_compare, format="png", dpi=dpi_compare, bbox_inches='tight')
                            st.download_button(
                                label="📥 Download Comparison Plot",
                                data=buf_compare.getvalue(),
                                file_name="profile_comparison.png",
                                mime="image/png"
                            )
                        
                        # Comparison statistics table
                        st.markdown("##### 📈 Comparison Statistics")
                        
                        comparison_data = []
                        for profile_name in profiles_to_compare:
                            profile = st.session_state.profiles[profile_name]
                            values = profile['values'][field_to_compare]
                            
                            comparison_data.append({
                                'Profile': profile_name,
                                'Length': f"{profile['distances'][-1]:.1f} {profile['distance_units']}",
                                'Mean': f"{np.nanmean(values):.2f}",
                                'Min': f"{np.nanmin(values):.2f}",
                                'Max': f"{np.nanmax(values):.2f}",
                                'Std': f"{np.nanstd(values):.2f}",
                                'Range': f"{np.nanmax(values) - np.nanmin(values):.2f}"
                            })
                        
                        df_comparison = pd.DataFrame(comparison_data)
                        st.dataframe(df_comparison, use_container_width=True)
                
                else:
                    st.info("👆 Add at least 2 profiles and select fields to enable comparison")
        
        else:
            st.info("👆 Add your first profile line above to get started with profiling analysis!")
        
        # Download section - FIXED VERSION
        if st.session_state.profiles and selected_fields:
            st.markdown("---")
            st.markdown("#### 💾 Download All Results")
            
            col_dl1, col_dl2, col_dl3 = st.columns(3)
            
            with col_dl1:
                # Download all profile data as separate CSV files in a ZIP
                def create_all_profiles_zip():
                    zip_buffer = io.BytesIO()
                    with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
                        # Add individual profile data files
                        for profile_name, profile in st.session_state.profiles.items():
                            profile_data = {
                                'distance': profile['distances'],
                                'longitude': profile['profile_lons'],
                                'latitude': profile['profile_lats']
                            }
                            
                            for field in selected_fields:
                                if field in profile['values']:
                                    profile_data[field] = profile['values'][field]
                            
                            df_profile = pd.DataFrame(profile_data)
                            csv_data = df_profile.to_csv(index=False)
                            zip_file.writestr(f"{profile_name}_data.csv", csv_data)
                    
                    zip_buffer.seek(0)
                    return zip_buffer.getvalue()
                
                zip_data = create_all_profiles_zip()
                st.download_button(
                    label="📥 Download All Profiles (ZIP)",
                    data=zip_data,
                    file_name="all_profiles_data.zip",
                    mime="application/zip"
                )
            
            with col_dl2:
                # Download profile coordinates
                def create_coordinates_zip():
                    zip_buffer = io.BytesIO()
                    with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
                        for profile_name, profile in st.session_state.profiles.items():
                            coords_data = {
                                'distance': profile['distances'],
                                'longitude': profile['profile_lons'],
                                'latitude': profile['profile_lats']
                            }
                            df_coords = pd.DataFrame(coords_data)
                            csv_coords = df_coords.to_csv(index=False)
                            zip_file.writestr(f"{profile_name}_coordinates.csv", csv_coords)
                    
                    zip_buffer.seek(0)
                    return zip_buffer.getvalue()
                
                coords_zip = create_coordinates_zip()
                st.download_button(
                    label="📥 Download Coordinates (ZIP)",
                    data=coords_zip,
                    file_name="profile_coordinates.zip",
                    mime="application/zip"
                )
            
            with col_dl3:
                # Download summary statistics
                summary_data = []
                for profile_name, profile in st.session_state.profiles.items():
                    for field in selected_fields:
                        if field in profile['values']:
                            values = profile['values'][field]
                            summary_data.append({
                                'Profile': profile_name,
                                'Field': field,
                                'Length': f"{profile['distances'][-1]:.1f} {profile['distance_units']}",
                                'Mean': np.nanmean(values),
                                'Min': np.nanmin(values),
                                'Max': np.nanmax(values),
                                'Std': np.nanstd(values),
                                'Start_Lon': profile['start_lon'],
                                'Start_Lat': profile['start_lat'],
                                'End_Lon': profile['end_lon'],
                                'End_Lat': profile['end_lat']
                            })
                
                df_summary = pd.DataFrame(summary_data)
                csv_summary = df_summary.to_csv(index=False)
                
                st.download_button(
                    label="📥 Download Summary Statistics (CSV)",
                    data=csv_summary,
                    file_name="profile_statistics_summary.csv",
                    mime="text/csv"
                )
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            


























# ==============================
# FOOTER
# ==============================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem;">
    <p>🌍 <strong>Geoid Data Corrections Interface</strong> | Developed by RAJKUMAR MONDAL at LithoSphereX Lab</p>
    <p>📧 For support and queries: rajkumarmondal691@gmail.com / cpdubey@gg.iitkgp.ac.in / p.dubey48@gmail.com</p>
</div>
""", unsafe_allow_html=True)