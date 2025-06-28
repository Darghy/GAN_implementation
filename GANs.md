# What I learnt

* What GANs are:
  - a discriminator and generator model
  - a way to *generate* samples from a distribution of data, not only classify the samples.
* The underlying noise distribution DOES NOT need to be 28x28 pixels - it can be any random noise, and in practice o3 tells me "the MNIST data is captured in a 100-dimensional manifold", so I take it to be $\text{Unif}[-1,1]^{100}$.
* How LeakyReLU works: it's just like ReLU, but we have a small slope for $x<0$ as well, e.g. $f(x) = x$ for $x>0$ and $f(x) = 0.2x$ for $x < 0$. The internet tells me "This type of activation function is popular in tasks where we may suffer from sparse gradients, for example training generative adversarial networks."
* The reason why the original papers uses this weird FID distance is because there is no better natural loss metric to track in this case - there's no equivalent of "line go down" for the generator/discriminator loss, since they're trained adversarially against each other.
* Funny mistake: if I load models, but I've defined the optimizers above, then the loaded models won't point to those optimizers, learning nothing. Solution: always create optimizers AFTER my FINAL model instances.

I tried to write/modularize the code myself. 
This didn't work for steps 3 and 5 of the training loop, where Claude did them for me. I let Claude go wild after, but I rewrote the minimal loop myself in `training_loop_minimal.py`.

TODO:
* learn how `.detach()` works