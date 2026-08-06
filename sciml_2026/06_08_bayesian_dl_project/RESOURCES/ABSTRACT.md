# Bayesian Deep Learning Methods

### Comparing Bayesian deep learning methods on your own data

## Abstract

A neural network will give you a confident prediction for an input unlike anything
it was trained on. For a scientist deciding which compound to synthesise or which
experiment to run next, knowing that uncertainty is essential. 

In this project you implement four ways of getting a predictive distribution out of
a neural network: a deterministic baseline with a heteroscedastic likelihood, MC
Dropout, a deep ensemble, and Bayes by Backprop written from the ELBO, and
evaluate them through one shared set of metrics. You choose the domain: Type Ia
supernova brightnesses, molecular solubility, or blood-cell microscopy, each with a
distribution shift built into the held-out data. Alongside accuracy you measure
calibration, out-of-distribution detection, and the split between uncertainty that
more data would remove and uncertainty that it would not.

Move on to ASSIGNMENT.md