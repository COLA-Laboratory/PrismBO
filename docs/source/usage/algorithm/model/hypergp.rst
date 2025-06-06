.. _hypergp:

HyperGP
=======
HyperGP comes from **Pre-trained Gaussian processes for Bayesian optimization**:cite:`Wang2021`

In this method, the surrogate model is built on a pre-trained GP with data from related tasks. This approach uses a KL divergence-based loss function to pre-train the GP, ensuring it captures similarities between the target function and past data. The pre-trained GP serves as the prior for BO, allowing the model to make better predictions with fewer observations by leveraging the pre-trained knowledge.

