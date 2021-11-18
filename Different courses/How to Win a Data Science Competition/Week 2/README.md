## Week №2

### Visualization tools
- [Seaborn](https://seaborn.pydata.org/)
- [Plotly](https://plotly.com/python/)
- [Bokeh](https://github.com/bokeh/bokeh)
- [ggplot](http://ggplot.yhathq.com/)
- [Graph visualization with NetworkX](https://networkx.org/)

### Others
- [Biclustering algorithms for sorting corrplots](https://scikit-learn.org/stable/auto_examples/bicluster/plot_spectral_biclustering.html)

## Validation strategies

This page contains information about main validation strategies (schemes): holdout, K-Fold, LOO.

The main rule you should know — **never** use data you train on to measure the quality of your model. The trick is to split all your data into training and validation parts. 

Below you will find several ways to validate a model.

**a) Holdout scheme:**

- Split train data into two parts: **partA** and **partB**.

- Fit the model on **partA**, predict for **partB**.

- Use predictions for partB for estimating model quality. Find such hyper-parameters, that quality on partB is maximized.

**b) K-Fold scheme:**

- Split train data into K folds. 

- Iterate though each fold: retrain the model on all folds except current fold, predict for the current fold.

- Use the predictions to calculate quality on each fold. Find such hyper-parameters, that quality on each fold is maximized. You can also estimate mean and variance of the loss. This is very helpful in order to understand significance of improvement.

**c) LOO (Leave-One-Out) scheme:**

- Iterate over samples: retrain the model on all samples except current sample, predict for the current sample. You will need to retrain the model N times (if N is the number of samples in the dataset).

- In the end you will get LOO predictions for every sample in the trainset and can calculate loss.

Notice, that these are *validation* schemes are supposed to be used to estimate quality of the model. When you found the right hyper-parameters and want to get test predictions don't forget to retrain your model using all training data.

- [Validation in Sklearn](https://scikit-learn.org/stable/modules/cross_validation.html)
- [Advices on validation in a competition](http://www.chioka.in/how-to-select-your-final-models-in-a-kaggle-competitio/)

- [Perfect score script by Oleg Trott](https://www.kaggle.com/olegtrott/the-perfect-score-script) -- used to probe leaderboard
- [Page about data leakages on Kaggle](https://www.kaggle.com/docs/competitions#leakage)
- [A​nother page about data leakages on Kaggle](https://www.kaggle.com/dansbecker/data-leakage)