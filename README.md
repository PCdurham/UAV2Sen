# UAV2Sen
 Fuzzy Classification of Sentinel 2 imagery from UAV ground data
 
## Description
The methods described here use low-altitude UAV orthoimagery in order to train fuzzy classification models of Sentinel 2 imagery.  The intended application context is river corridors and the models assume a simple 3 class division of the fluvial landscape: water, vegetation and dry sediment.  We include code to organise the data for machine learning, train (shallow) convolutional neural networks and finally to analyse the quality of predictions.  We also provide large scale deployment capabilities with a script that can deliver a geocoded fuzzy classification from a large geocoded tif image.  

## Dependencies
1. Tensorflow version 2.1 (GPU usage is desirable but not essential)
2. Scikit-Learn
3. Scikit-Image
4. Rasterio
5. Pandas
6. Seaborn

We recommend using the Anaconda distribution of Python 3 which installs all the needed libraries except rasterio and tensorflow.

## Usage

### Step 0: Data Preparation
*Super Resolved Sentinel 2 Imagery*
The algorithms work best with Sentinel 2 data where all bands have been super-resolved to 10m.  We used the super-resolution plugin for ESA SNAP. [Super-Resolution for SNAP](https://nicolas.brodu.net/recherche/superres/)

*GIS preparation*
Once the Sentinel 2 images have been super-resolved, they should be cropped to an area matching the UAV survey areas that wil be used to train the fuzzy classification.  We assume the use of 3 classes for water, sediment and vegetation. It also requires class rasters created manually (rasterised polygons) which define areas of the classes above. The process will therefore expect 3 files for each site: A cropped Sentinel 2 file ending in _S2.tif

### Model Training
UAV2SEN_FuzzyDNN, UAV2SEN_FuzzyRF, UAV2SEN_FuzzyCNN will train the appropriate model with the sample data.  Note that the model will not be saved so must be ketp in memory for the next steps.  

### Model Validation
UAV2SEN.FuzzyValidation will validate any of the trained predictors against grount truth data also included in the sample dataset.  

### Map Production
UAV2SEN_FuzzyCNN_QGIS is coded to run in the Python console of QGIS.  Scripts were tested under QGIS 3.4 long term release. Keras, Tensorflow and Scikit-Image must also be installed in the python 3 environment of QGIS.  This script will train the CNN model and calculate a fuzzy classification for a user-specified image.  The result will be 3 seperate raster outputs containing the membership % for classes of water, vegetation and dry sediment.
