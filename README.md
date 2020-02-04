# UAV2Sen
 Fuzzy Classification of Sentinel 2 imagery from UAV ground data
 
## Description
The methods described here use low-altitude UAV orthoimagery in order to train fuzzy classification models of Sentinel 2 imagery.  The intended application context is river corridors and the models assume a simple 3 class division of the fluvial landscape: water, vegetation and dry sediment.  We include code to organise the data for machine learning, train random forest, dense neural network and (shallow) convolutional neural networks and finally to analyse the quality of predictions.  We also provide GIS integration with a script that can train and run a fuzzy classification with PyQGIS in order to deliver mapping capabilities.  

## Dependencies
1. Tensorflow versio 1.14 (GPU usage is desirable)
2. Keras
3. Scikit-Learn
4. Scikit-Image
5. Pandas
6. Seaborn

We recommend using the Anaconda distribution of Python 3 which installs all the needed libraries except Keras and Tensorflow. 
