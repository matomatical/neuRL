# Expectile-based Distributional Reinforcement Learning in the Brain

This repository contains some code and notebooks from my exploration
of expectiles, distributional reinforcement learning, and the brain.

Made with :purple_heart: by Matt, but, I must say, I really don't
feel purple love for jupyter notebooks. More like red fiery rage.


## Highlights:

* `expectiles.py` for an implementation of an efficient method of
  calculating expectiles that seems to be much much much faster
  than using a scipy optimisation routine, and the notebook
  `CalculatingExpectiles.ipynb` for a motivation of this method.
  (WARNING: the github formula rendering doesn't work well).

* `LearningRateStability.ipynb` for a tutorial on online computation
  of means and expectiles, including a hypothetical asymmetric rate
  setting scheme which is probably optimal and seems to match what
  *mice neurons* like to do:
  
  ![Learning rate stability scheme fits neural data](plots/learning_rate_stability.png)

  *Dat α, so fit!*---I should learn how to do error bars and regression
  tests. But not today!
  
* `Imputation-Direct.ipynb` for an alternative to DeepMind's optimisation-
  based imputation method, which should be vastly more computationally
  efficient (see `Imputation-Optimisation.ipynb` for my reproduction of
  their method).


## Where to from here?

Some more questions:

* Can the direct imputation method lead to a nicer 'decoding' demonstration
  using the neural data?

* Can the updates from 'learning rate stability' lead to an efficient
  EDRL algorithm?

* The plots in 'aggregate RPEs' suggest some kind of (measurable?)
  nonlinear response from an ensemble of asymmetrically tuned
  dopamine neurons to rewards; is there any empirical evidence of this?
  Are there limits to the nonlinearity, say, at the lower end due to
  the negative signal being incodes as an inhibition of baseline response?

Other imputation strategies

* Apparently there's a DistRL algorithm based on mixtures of gaussians.
  Since, according to Richard, gaussians are common in neurobiology, this
  could be worth looking at on the basis of its biological plausibility.