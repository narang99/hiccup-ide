The next experiment is simple. We take a much smaller version of inceptionv1 like architecture and train it on cifar.  
We do the same exercise for the final model to begin with to analyse 1 neuron.  
We would also do the exercise on different snapshots of the model during training.

Which snapshots to pick? We will graph the euclidean distance of the weights of the given neuron with the previous iteration (before the step) and plot it. We would checkpoint regularly based on this. and then find the places where the weights are changing significantly. This ends up giving us a snapshot at different points in time finally. Then we run the pipeline to find the clusters. easy. Along with the change in the weights.
