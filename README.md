# Geoid
Apply the corrections on geoid data

GAPVIEW is a comprehensive Streamlit web application for geophysicists, geodesists, and researchers working with gravity field and geoid data. This powerful tool enables advanced geoid data corrections using tesseroid methods with an intuitive, Petrel-like interface.
Key Features
•	Multi-format Data Support: CSV, GeoTIFF, NetCDF, GDF
•	Advanced Data Analysis: Outlier detection, distribution analysis, statistical visualization
•	Interactive Visualization: High-quality maps with multiple interpolation methods (Liner, Cubic and RBF)
•	Physical Corrections: Topographic, crustal, sedimentary and combined corrections.
•	Profile Analysis: Draw and analyze multiple cross-sectional profiles
 Application Layout
The GUI follows:
1.	Data Acquisition
2.	Data Upload
3.	Data Analysis
4.	Data Visualization
5.	Geoid Corrections
6.	Draw Profiling Line

   
Required Python Libraries

streamlit==1.28.0, pandas==2.0.3, numpy==1.24.3, matplotlib==3.7.2, seaborn==0.12.2, scipy==1.11.2, plotly==5.16.1, xarray==2023.7.0, netCDF4==1.6.4, scikit-learn==1.3.0, numba==0.57.1, requests==2.31.0
How to Run
Navigate to the application directory (open in ternimal): cd path/to/gui file (py needs to be available in the system)
Run the following command: streamlit run gui.py





Data Attached
Data Type	Recommended Source	Format
Crustal Thickness	IRIS CRUST1.0	CSV
Sedimentary Thickness	CRUST1.0	CSV
Topography	ETOPO1 / ETOPO2022	NetCDF
Geoid	EGM2008	GDF / NetCDF

Data Analysis
The Data Analysis section focuses on assessing data quality and statistical distribution. Users can examine minimum, maximum, mean, and standard deviation values, visualize data distributions, and identify potential anomalies or outliers. This step is essential for ensuring that erroneous or extreme values do not bias interpolation or geoid correction results. The analysis tools support informed decision-making prior to spatial modeling.
Data Visualization
The Data Visualization module enables spatial interpolation and mapping of all uploaded datasets. Users can generate continuous grids using interpolation methods such as nearest-neighbor, linear, cubic, or radial basis functions. Additional visualization enhancements include Gaussian smoothing, contour overlays, and customizable color maps. Both static and interactive plots are supported, allowing detailed inspection of spatial patterns in geoid and correction fields.
Geoid Corrections
The Geoid Corrections section implements physically motivated corrections to observed geoid heights. Users can compute topographic corrections, crustal thickness corrections, sedimentary corrections, or a combined correction that accounts for all mass contributions. Physical parameters such as density contrasts, reference crustal thickness, and angular cutoffs are user-defined. The correction calculations are optimized using numerical acceleration techniques to handle large datasets efficiently.
Profiling Line Tool
The profiling tool allows users to draw a custom line across the study region and extract geoid or corrected geoid values along that line. This functionality is particularly useful for tectonic, lithospheric, and regional interpretation, where variations along transect provide insight into subsurface structure. The extracted profiles can be used for comparison with seismic or geological models.
Support & Contact
Developer: Rajkumar Mondal, Dr. Chandra Prakash Dubey
Laboratory: LithoSphereX Lab, Dept. Geology and Geophysics, IIT Kharagpur
Email: rajkumarmondal691@gmail.com, p.dubey48@gmail.com, cpdubey@gg.iitkgp.ac.in
