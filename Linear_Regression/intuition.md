# Linear Regression Models the relationships between:
->Input variable, x
->Output variable, y
using a straight line:
            y=wx+b
where, w=Weight and b=Bias

The parameter w represents:
How strongly the feature x influences the output y.

What the Bias:
The bias term b controls the vertical position of the regression line.It represents the model’s predicted output when all input features are zero.(if x turns to be zero).

->b shifts the model up or down
->b gives flexibility to fit real data
->Without bias, the model must pass through the origin (unrealistic)

So the motive is to find the Best value of w and b so that the predicted value is closer to the actual value.

This is done by minimizing the loss function.

# Most real-world relationships are locally linear, meaning that:
Over small ranges, a line approximates the relationship well.
Linear models are simple, interpretable, and compute-efficient.

# Loss Function:(MSE)
We Measure how far is the predicted value from the actual Value.

# Gradient Descent:
We update parameters in the direction of decreasing loss.Basically taking Derivative of Loss with respect to weight.

## When to use Linear Regression:
1.When the relationship between x and y is approximately Linear.
2.If you want a fast interpretable model.
3.If the dataset is not that large.

## Limitations:
1.When Data is Non-linear.
2.When Outliers plays a major role in data Distribution.
3.When then data Varience is Very high.

### Key Oneline Takeaway:
Linear Regression help us to find Best-fit Straight Line by minimizing the MSE loss Between the actual value and predicted value. 

