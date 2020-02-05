# UAV2Sen
 Fuzzy Classification of Sentinel 2 imagery from UAV ground data
 
## Description
The methods described here use low-altitude UAV orthoimagery in order to train fuzzy classification models of Sentinel 2 imagery.  The intended application context is river corridors and the models assume a simple 3 class division of the fluvial landscape: water, vegetation and dry sediment.  We include code to organise the data for machine learning, train random forest, dense neural network and (shallow) convolutional neural networks and finally to analyse the quality of predictions.  We also provide GIS integration with a script that can train and run a fuzzy classification with PyQGIS in order to deliver mapping capabilities.  

## Dependencies
1. Tensorflow version 1.14 (GPU usage is desirable but not essential)
2. Keras
3. Feather
4. Scikit-Learn
5. Scikit-Image
6. Pandas
7. Seaborn

We recommend using the Anaconda distribution of Python 3 which installs all the needed libraries except Feather, Keras and Tensorflow.

## Usage

### Data Preparation
*Super Resolved Sentinel 2 Imagery*
We recommend the use of super-resolved sentinel 2
The dense neural network (DNN) and random forest (RF) classifiers are pixel-based and require data vectorised in tabular formats and stored as a Pandas dataframe.  We use Feather to reduce data volume of the tabular data.  The convolutional neural network (CNN) classifiers requires data to be tiled and stored as a tensor.  We provide a script for this UAV2SEN_MakeFuzzyTensor.py.  Current version of the scripts for DNN, RF and CNN classifier training scripts are setup to use sample data available here:

### Model Training
UAV2SEN_FuzzyDNN, UAV2SEN_FuzzyRF, UAV2SEN_FuzzyCNN will train the appropriate model with the sample data.  Note that the model will not be saved so must be ketp in memory for the next steps.  

### Model Validation
UAV2SEN.FuzzyValidation will validate any of the trained predictors against grount truth data also included in the sample dataset.  

### Map Production
UAV2SEN_FuzzyCNN_QGIS is coded to run in the Python console of QGIS.  Scripts were tested under QGIS 3.4 long term release. All dependencies must also be installed in the python 3 environment of QGIS.  This script will then train the CNN model
