# GAN implementation

I read the original GAN paper and implemented it for MNIST. This works! After 40k batches of size 128 we get pretty decent results:

<img src="selected_images/progress_epoch_40000.png" width="70%" alt="Generated MNIST digits after 40k epochs">

Claude wrote all the functionality for proper logging, loading/saving features, and generating images during training, i.e. most of `training_loop.py`. I wrote the conceptual training loop (based on the pseudocode in the original paper) in `training_loop_minimal.py`.

