# This python file describes how the model performance will be measured

# To be able to run this you need to install sklearn and numpy libraries
# You can do it via command "pip install sklearn numpy" on your terminal

# Import libraries
import numpy as np
from sklearn.metrics import average_precision_score

# Data example

# Coolers target is a list representing the label for a set of coolers
# 0 means is working, 1 means is failing
coolers_target = np.array([0, 0, 1, 0, 0, 1, 0])

# probabilities is a list representing the probability for a set of coolers to be failing
# more close to 1 means that the cooler is more probable to fail
probabilities = np.array([0.1093, 0.4231, 0.8992, 0.5121, 0.9016, 0.3521, 0.2775])

# Performance evaluation with Average Precision Score
# The result of this method represents the score of the precision-recall curve
pr_auc = average_precision_score(coolers_target, probabilities)
print(round(pr_auc, 4))