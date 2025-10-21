# Positioning Road-level Flood Sensing for Traffic Disruption Risk Assessment based on Uncertainties and Bayesian Networks

## Overview

This project implements a probabilistic framework for optimal road-level flood sensing placement in urban road networks experiencing flooding events. 

## Major Features
The major features of the program could be found in `main.py`.
### Bayesian Network Models

- **Flood Bayesian Network**: Models spatial dependencies between road flood states
- **Traffic Bayesian Network**: Captures traffic speed dependencies and flood-induced disruption patterns with bidirectional information propagation (downward/upward)

### Information-Theoretic Metrics

- **Disruption (D)**: Measures the divergence from normal traffic conditions
- **Unexpectedness (UoD)**: Captures the reduction in uncertainty about the system state (from prior to posterior)
- **Weighted VoI**: Combines disruption and unexpectedness to guide sensor placement

### Multi-Sensor Sequential Placement

- Iterative sensor placement considering existing observations in the greedy algorithm framework
- Scenario tree pruning for computational efficiency

## Project Structure

```
Major scripts
├── config.py              # Configuration parameters
├── main.py                # Main execution script for the features
├── val.py                 # Validating the implementation of Bayesian Networks
Models
├── data.py                # Handling and road speed data processing; defining the RoadData class
├── model.py               # Bayesian Network implementations; defining the TrafficBayesNetwork and 
├                            FloodBayesNetwork classes
Visualization
├── visualization.py       # Plotting and visualization functions
Data and output folders
├── data/                  # Data directory
│   ├── nyc_street_flooding_20250218.csv (download from 
│       https://data.cityofnewyork.us/Social-Services/Street-Flooding/wymi-u6i8/about_data 
│       and name it as shown here)
│   ├── nyc_roads_geometry_corrected.geojson (adapted from NYC OpenData)
│   └── nyc_boundary/ (shp file, only for visualizatin)
│   └── nyc_opendata_token.txt (must-have but not included in repo, please use your own NYCOpenData token)
├── cache/                 # Cached model instances; will be used when running
├── results/               # Bayesian Networks and Sensor placement results
└── figures/               # Save visualizations
```

## Dependencies

```
numpy
pandas
geopandas
networkx
pgmpy (0.1.26)
scikit-learn (1.6.1)
scipy
matplotlib
shapely
```

## Usage

### Major Workflow

1. **Check the configuration file**: Default parameters are recommended for illustration purposes
2. **Run main.py**: The major program
3. **Run val.py if needed**: It checks the implmentation of the Bayesian Network in `main.py` by comparing the incident-wise results with the ground truth (simialr but different from the major program). 
4. **Visualization**: Do visualization by uncommenting codes in `main.py` and `val.py`

### Configuration

Key parameters in `config.py`:

- `sensor_count`: Number of sensors to place (default: 9)
- `corr_thr`: Correlation threshold for network edges (default: 0.6)
- `weight_thr`: Weight threshold for flood network (default: 0.2)
- `weight_disruption`: Weight between disruption and unexpectedness objectives (default: 0.5)
- `prune_rate`: Scenario tree pruning threshold (default: 0.01)

### Main program
The main script will:
- Load and build road speed data. It may take long, as it pull data from NYC OpenData
- Construct flood and traffic Bayesian Networks
- Calculate sensor placement for multiple sensors
- Save results to the `results/` directory

### Validation program
Different from the main program which calculate the long-term flood and speed "observations" and calculate long-term VoI, the validation program
- Load and fit the traffic speed distribution in the non-flooding time. It may take long, as it pull data from NYC OpenData
- Flood "observations" and speed "observations" are directly obtained from the historical records during the incident
- The "observations" will propagate in the Bayesian Networks obtained from the `main.py`
- Historical traffic speed disruption and disruption unexpectedness at the propagated roads are used as the ground truth, and be compared with the estimated ones
- The expected result is that the estimation degree is proportional to the ground truth degree, which means the Bayesian Networks obtained from the `main.py` can be used to estimate the risk (although not neccessarily the actuall degree of disruption or unexpectedness)


## Contact
The paper of the program is under working. Email to Xiyu Pan (xyp@gatech.edu) for more information.
