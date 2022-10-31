# OFParticleFilter
Project 2 in EQ2801 Optimal Filtering HT2022

In estimators.py you'll find the characterization of each of the particle distribution generating functions.

In uniform_addition.py you'll find a method for generating a distribution for Z given a distribution for X,
done through resampling, to obtain a distribution with uniform weights, and then adding the uniform variable U.

In normal_sampling.py you'll find the implementation of the generation of a particle distribution for Z given
a particle distribution on X through the sampling of a proposal distribution and then the update of the weights.

In measurement_update.py you'll find the implementation for the weight update on Z given a measurement of Y.

In uniform_addition and normal_sampling, after the characterization of the particle distributions on Z generated,
the questions pertaining to the conditional distribution of Z on Y are also answered.