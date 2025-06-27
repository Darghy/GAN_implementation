# What I learnt

* What GANs are: a discriminator and generator model
* The underlying noise distribution DOES NOT need to be 28x28 pixels - it can be any random noise, and in practice o3 tells me "the MNIST data is captured in a 100-dimensional manifold", so I take it to be $\text{Unif}[-1,1]^{100}$.
* How LeakyReLU works: it's just like ReLU, but we have a small slope for $x<0$ as well, e.g. $f(x) = x$ for $x>0$ and $f(x) = 0.2x$ for $x < 0$. The internet tells me "This type of activation function is popular in tasks where we may suffer from sparse gradients, for example training generative adversarial networks."