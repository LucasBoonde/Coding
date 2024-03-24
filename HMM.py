import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
initial_distributions = tfd.Categorical(probs=[0.8, 0.2])
transition_distributions = tfd.Categorical(probs=[0.7, 0.3])

observation_distributions = tfd.Normal(loc=[0.,15.],scale=[5.,10.])#loc er din mean, og scale er din std.

model = tfd.HiddenMarkModel(initial_distribution=initial_distributions, transition_distribution = transition_distributions, observation_distribution = observation_distributions, num_steps=7)
