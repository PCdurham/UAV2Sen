# UAV2Sen
 Fuzzy Classification of Sentinel 2 imagery from UAV ground data
 
## Description
The methods described here use low-altitude UAV orthoimagery in order to train fuzzy classification models of Sentinel 2 imagery.  The intended application context is river corridors and the models assume a simple 3 class division of the fluvial landscape: water, vegetation and dry sediment.  We include code to organise the data for machine learning, train (shallow) convolutional neural networks and finally to analyse the quality of predictions.  We also provide large scale deployment capabilities with a script that can deliver a geocoded fuzzy classification from a large geocoded tif image.  

## Dependencies
1. Tensorflow version 2.1 (GPU usage is desirable but not essential)
2. Scikit-Learn
3. Scikit-Image
4. Rasterio
5. GDAL
6. Pandas
7. Seaborn

We recommend using the Anaconda distribution of Python 3 which installs all the needed libraries except GDAL, rasterio and tensorflow.

## Usage

### Step 0: Data Preparation
*Super Resolved Sentinel 2 Imagery*
The algorithms work best with Sentinel 2 data where all bands have been super-resolved to 10m.  We used the super-resolution plugin for ESA SNAP. [Super-Resolution for SNAP](https://nicolas.brodu.net/recherche/superres/)

*GIS preparation*
Once the Sentinel 2 images have been super-resolved, they should be cropped to an area matching the UAV survey areas that wil be used to train the fuzzy classification.  We assume the use of 3 classes for water, sediment and vegetation. It also requires class rasters created manually (rasterised polygons) which define areas of the classes above. The process will therefore expect 3 files for each site: A cropped Sentinel 2 file ending in _S2.tif; a classified orthomosaic ending with _UAVCLS.tif and a rasterised polygon file ending in _dbPoly.tif. 

This data is controlled with a small csv file with fields: 'Site Name', 'Abbrev', 'Month', 'Year', 'Training', 'Validation'. The training and validation fields are filled with 0s or 1s and later used to select sites for training of validation. The script UAV2SEN_CheckFiles.py can be used to check that all the listed sites have the expected files in the folder.

### Step 1: Data Compilation
Before modelling the data must be organised into a set of tensors and associated dataframe with the classification labels and associated site details.  This process is controlled with the csv file.  First, run UAV2SEN_MakeCrispTensor.py.  This will run through all the listed sites, ignoring the 'Training' and 'Validation' fields.   Label will be compiled by converting the UAV classes into crisp classes and by using the polygon interpretations.  All the super-resolved sentinel 2 cropped images will be tiled and associated label data compiled.  The tensors will be saved as a large numpy array and the labels as a csv.  Second, run UAV2SEN_MakeFuzzyTensor.py.  This process is similar except that the label data will consist of fuzzy membership percentages for each pixel.

For each script, the user must carefully read and edit the 'User Inputs' section at the start of the script. The key variable at this stage is the size of individual tiles in the tensor.  Choose 3, 5 or 7.

### Step 1: Modelling
Once data have been compiled, modelling can begin.  The modelling process is controlled with the csv site list file.  To use a site for model training, add a 1 in the training column.  To use a site for model validation, add a 1 in the validation column.  For each model run, a minimum of 1 training site and minimum of 1 validation site are required.  It is not recommended to use a site for both training and validation, but this will not cause an exception.

UAV2SEN_CrispCNN will use crisp data and run a crisp classifier algorithm.  The routine has a model tuning option that allows the user to adjust training epochs and avoid overfitting. The validation data will be used to produce F1 scores for the validation predictions.  There is also an option to display the mapped output of the validation site predictions.  The trained model will be saved to disk.

UAV2SEN_FuzzyCNN will use the fuzzy membership data.  The routine has similar model tuning options.  The validation data will be used to examine mean and rms values of fuzzy prediction as well as the slope of predicted vs observed fuzzy predictions. There is also an option to display the mapped output of the validation site predictions. The trained model will be saved to disk.

UAV2SEN_Fuzzy2CrispCNN will train a fuzzy model but will use this model to predict crisp classes. Validation will be done with F1 scores and here the user can adjust the threshold of what constitutes a pure class. There is also an option to display the mapped output of the validation site predictions. The trained model will NOT be saved to disk. Use the fuzzyCNN script above for to produce saved models.

Each script has a long list of adjustable parameters detailed with comments at the start in the user inputs section.

### Step 2: Large Scale Deployment
For map production, UAV2SEN_fuzzyCNN_BigTif.py can take a large geocoded image and run a pre-trained CNN model in order to deliver fuzzy classifications at the scale of whole satellite tiles.  The outputs will be 3 rasters for vegetation membership, water membership and sediment membership.  Full geocoding information is embedded in the geotif metadata and the fuzzy class rasters will open in Python or any geospatial software package for further processing.  The rasters are coded from 0 to 100 thus directly giving the membership % for the given class of a given pixel.  EG, if a pixel of the water membership raster has a value of 56, the pixel is predicted to have 56% water.  For large images, this script will process the fuzzy classifications row-by-row. The result is a process which is less memory intensive and will run on most computers.  But it is considerably slower than simultaneous processing of the entire image.  For a tile size of 5 and all 12 bands, CNN classification would require the creation of a tensor of dimensions (Image.shape[0]*Image.shape[1],5,5,12).  Since a full Senitenl 2 tile is about 10 000 x 10 000, the required tensor would have over 1E10 elements.    The row-by-row process requires only tensors on the order of 1E6 elements and is more reliable even if Fuzzy Classification of a full Sentinel 2 tile will take aproximately 1 hour.


