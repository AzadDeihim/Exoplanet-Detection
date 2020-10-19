# Exoplanet-Detection

Dataset:
https://www.kaggle.com/keplersmachines/kepler-labelled-time-series-data
  
To excecute the code, simply run main.py. This will produce plots, and print
results to the console for all of the models discussed in the report.
The estimated completion time of the code is 25-45 minutes. There are some parts of the code commented out, such as the tSNE
visualizations and the grid searchs. Uncomment if you wish to run them,
they take about an hour and 4 days to complete, respectively. 

You may get UndefinedMetricWarning when running main.py: this occurs when the number of exoplanets predicted correctly is 0,
as there are no examples that can be used to calcluate precision/recall/f1 score.

