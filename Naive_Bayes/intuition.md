# What problem Naive Bayes Solves:
Pick the class that has the highest posterior probability given the data.
Naive Bayes is a generative model, not discriminative.

# Mathematically:
p(y/x) is directly proportional to p(x/y).p(y)
This comes from the Bayes Theorem.

Core formula:
Bayes Theorem:P(y/x)=(p(x/y).p(y))/p(x)

where p(y/x)=Posterior(What we want to know)
p(x/y)=Likelihood
p(y)=prior
p(x)=Evidence(Same for all(Ignored))

# Why is it called Naive:
Because it makes a very strong Assumption that:
All Features are condiionally independent given the class.

# Types of Naive Bayes (Important)
Depending on feature type:
1.Gaussian Naive Bayes
1.1.Continuous features
1.2.Assumes normal distribution
2.Multinomial Naive Bayes
2.1.Counts (word frequencies)
2.2.NLP
3.Bernoulli Naive Bayes
3.1.Binary features (0/1)

# OneLine Explanation:
Naive Bayes is a probabilistic classifier that applies Bayes’ theorem with a strong independence assumption between features.                     
