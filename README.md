# Expectile-based Distributional Reinforcement Learning in the Brain

This repository contains some code and notebooks from my exploration
of expectiles, distributional reinforcement learning, and the brains.

Highlights:

* `expectiles.py` for an implementation of an efficient method of
  calculating expectiles that seems to be much much much faster
  than using a scipy optimisation routine, and

* The first part of `Expectiles.ipynb` for a motivation of this
  method.

* `LearningRateStability.ipynb` for a tutorial on online computation
  of means and expectiles, including a hypothetical asymmetric rate
  setting scheme which is probably optimal and seems to match what
  mice neurons like to do:
  
  ![Learning rate stability scheme fits neural data](learning_rate_stability.png)

  *dat α so fit!*---I should learn how to do error bars and regression
  tests. But not today!

Made with :purple_heart: by Matt, but, I must say, I really don't
feel purple love for jupyter notebooks. More like red fiery rage.
