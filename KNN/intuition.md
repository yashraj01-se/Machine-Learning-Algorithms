# KNN is used for both Classification and Regression but mostly used for classification.

# Key idea:

“Similar points should have similar labels.”

# Core Idea:
Unlike Linear / Logistic Regression:
❌ No training phase
❌ No weights
❌ No loss function
❌ No gradient descent

KNN is a lazy learning algorithm.

# Distance Formula:
Euclidian Distance=sumation of root(x2-x1)^2 

# What is k?
K = number of neighbors considered.

If K = 1:
Extremely sensitive to noise
Overfitting

If K is large:
Smooth decision boundary
Can underfit

# KNN Produces Non-Linear Boundaries One that Logistic regression fals.

# One-Line Mental Model
KNN predicts by asking: “Who are your closest neighbors?”