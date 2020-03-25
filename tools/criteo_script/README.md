# Criteo Dataset #
The data is provided by CriteoLabs (http://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset/).
The original training set contains 45,840,617 examples.
Each example contains a label (1 if the ad was clicked, otherwise 0) and 39 features (13 integer features and 26 categorical features).
Because the original test set doesn't contain labels, it is not used.


# Train on HugeCTR #
To train a model with Criteo dataset on HugeCTR, it must be first preprocessed accordingly.
For the detailed instruction, refer to samples/{$sample-name}/README.md.
