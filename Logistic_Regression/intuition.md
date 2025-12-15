# So First of all Logistic regression is used for classification not regression
Typical use cases:
->Spam vs Not Spam
->Fraud vs Not Fraud
->Disease vs No Disease
->Pass vs Fail

# What if we use Linear regression for classification:
->output >1 or <0 is very common.
->No clear Boundaries.
->Sensitive to outliers.

# Solution:
Logistic reggression applies a sigmoid function over the linear model.
y=Wx+b
then sigmoid=1/1+e^-z

raw logits into probability distribution lying between 0 and 1.

# Loss function:
Instead of MSE loss we use Cross Entopy loss.
It penalizes the confident wrong prediction.
works well with probabilities.

# One-Line Mental Model
Logistic Regression learns a decision boundary by predicting probabilities using a sigmoid function.