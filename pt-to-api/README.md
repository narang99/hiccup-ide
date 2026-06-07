Progress track

- Use hold-frac=0 in cosine annealing. Our MSE graphs spiked in the part where mulitiplier was 1. This keeps the MSE and weights loss stable across seeds. (Weights loss was observed to not change a lot)
- JAX is done and is working fine
- Tensorboard support added, working fine
- Threshold finding needs to be done during generation, its easier this way. GMMs can take time on a lot of data.

- I still have problems choosing the right model. It is quite hard. Everything looks like the right answer lol.
  - My older intention was to find the best losses and find clusters among them (among the best losses only)
  - Best losses do give similar results. The components where the best losses have best stability scores is the perfect one
  - now in 20 runs, i sadly did not get many good best losses. Quite unfortunate. I know now that 9 is the right answer though.
    - It has best loss similar to 10, but with less moving parts. Lets see if i can cluster the losses and find stability scores among them.

- Ohk, clustering is struggling too. We are clustering by adding everyone together and thats not nice.
  - We do component wise separately. And loss wise too (similar losses are clustered)
  - The easiest way right now is to find the loss differences manually, threshold them and be good. Is there something else i can do? I can cluster the losses for a component first, lets try that

# Model selection

For each n_component

- Run as many seeds as we can
- Cluster the losses (GMM, find BIC, get min -> best clustering)
- Find the stability score of the minimum loss cluster

Across n_components

- Plot the loss and min loss cluster stability score
- The main n_components are hovering around the minima of loss, find the components with the highest stability score there.

Chasing minimum MSE

- Sometimes, no seed reaches the minimum MSE, sucks. If none of them reach, there is no way to know if an MSE is lower than ours
- If the minimum loss cluster size << other loss cluster sizes (idk, how low, for now, we just do scatter plots for each components), then we need stronger attack on MSE, the annealing needs to favor MSE more.
- For now, this would also be subjective

## Steps

- Plot losses
- For each n-components
  - Cluster losses
  - if the min loss cluster is size 1, need more seeds which reach here
  - if not, find stability score of this cluster

- Find the maximum stability score number among the min losses across n-components (around the minima)
  - find the minimum loss one in the max stability score components
  - More samples might actually help, would be useful to check this out too.

The only problem now is getting good MSE (need to think a bit more about this now).

- Great, now drive is giving problems lol. The only good thing about drive seems to be the mounting now. S3 is hands down the best. I might need to do this later.
  - Drive fails in tensorboard, which is what i really mainly wanted it for `;_;`

The problem of local minima is quite annoying, we might need to do one cycle fit maybe? or cosine with warm restarts is also a good idea to try.
I'll also need to see if normal init works better than mine now.

Stuff to try finally:

- Warm restarts (although ive already tried them before, lets see)
- NoInitStrategy
- min_factor=10 was already tested, i dont see any differences from 1 (default)

Most of these might really not be the right answer though, ive tried them before. Although i didnt have such a clear picture before.
I might also reduce the number of epochs, the descent is already finished in 500 it seems, doing 1000 epochs might be beneficial.

## Stuff to try for loss chasing

- Hyperparameters vectorize
  - is it worth it?
  - nah no need to nitpick. lets do the runs again.
- No init
  - works
- Fatter encoder (easiest to do rn)
  - does not work

- Its best to start another layer, with no init. that works out well.
- Then we come back to layer 2, chan 5

Ohk, colab is cooking fine for now. We need to get onto:

- First layer
- graph building

Well, it seems this procedure might be useful for finding attribution cutoffs for a kernel:

- sort all values (do pos and neg separately)
- plot it. you ll find an elbow
- the elbow is the threshold. For a random kernel 7% of the data was above the elbow, which is not very bad
  - this is a lot more "convincing" than the otsu per column thing that i do right now
  - i think, im not sure though. this will need testing. for now, we do what have been doing

- What do i do next?
  - I need to see the graphs of losses for each seed for each component. Plot them. Pick the smallest one. See if we need more runs.
  - Then pick the models. It would be useful to do some stability analyses also, but im not sure if i need to right now.
  - Just seeing the models would be fine i guess (the ones which cluster towards lower losses)
  - we can gmm with mean 0 i think, should be fine.

Im thinking the problem with more components is simply that the model is not training better. Technically, the loss should keep going down, but it goes into a random basin and stays there. The problem with our loss function is that there are many basins.
Im not sure how i can make the landscape smoother. More data might actually help. Im not sure though.

The biggest problem is the weights settling down on something which does not have a lot of effect on the MSE. if this happens, it would have components which are low scoring, while having components which could have been decomposed further.
What are the ways I can improve on this?

- Jerk it out of the basin from time to time. This might require killing some components to 0 from time to time? im not sure.
  - we basically want to kill low scoring components in the MSE.
  - What is a low scoring component? Actually, i dont know if we should do the low scoring thing at all. Lets do random.
  - Randomly kill some ratio of components (make them 0).
    - Every some epochs.
    - Anyways the descent is quite fast, we should be fine.
    - Now should i make them 0? or should i make them non-zero?
      - all non-zero increase the weight loss
      - all zero decrease the weight loss (but hurt the mse)
      - tis hard to judge. Anyways, this is for another day. today i just select one of the minimum loss ones anyways

## Basic model selection procedure

- Find the n-components with least loss.
- pick the model with least loss xD.
- For sanity, check the stability score and clustering.

Nothing else seems to be more or less useful, its just more work in the end it seems. Its generally quite hard to know what is the right configuration. So lite max.

- For a random run, ive changed sigma-eps a bit, and it is giving not bad results (better loss). Changing the hyperparameters randomly might help, not doing it right now though.

Well, the model is not exactly scaling well with number of components. It is falling down to solving weights loss.  
For layers.0, we have very good loss results. this is not so for layers.2.  
For layers.2, we dont see an elbow, with loss progressively going down to the baseline as we increase components. The loss spikes up after one point. After this point, the weights loss is getting fulfilled. I'm assuming $\sigma_0$ is decreasing and getting us this effect. It would be instructive to now look at the gradients of good and bad runs, to see what works out best.
I do believe im close now though, lets hope this works out.

We have very good models trained for the first layer. Lets try them out first.
For graph building, ill start only with the first layer.

The layers.0 components are BEAUTIFUL. I need this good components for 2. After the graph.  
But amazing, this gives me some more hope :)

Well, the layers.0 comp is too good to be true lols.
It might just be dictionary learning output though at this point. Who knows. Well im going to ignore that for now :)
It might simply be a very low noise i guess.  
I'll have to think what went so well in this (other than the model learning extremely beautifully).

# Visualising the graph

Now, i need some way to visualise the graph. Directly seeing the activations as an image where each number is a vector represented by some column of some length is not working. Why?

- There is too much detail around "spatialness". The graph cares more about what it caught. Not where (although this is not very correct, the last conv layer's output do have some spatial awareness [the linear layer expects some spatialness]).
- but still, there might be an edge which was detected, and it would be side mei. not nearby.
- how do i fix this? now there is one problem technically, and that is that the spatial thing does matter in my network. But lets ignore that for a moment (the network kinda accepts a stem finder to find stems some place in the down part of the image, not somewhere up, but still lets forget that for a moment, bigger networks have lesser of these concepts [or atleast, they come in the very end]).
- What a kernel caught is important, simple. For now, not where (although again, tis wrong.).
- What is most important is that we have a "lit up" portion of the graph. And we want to see it.
- Consider the example of the big curve finder. A full semi-circle was getting detected in the network somewhere. It was mixing \,/,| together. How do we detect that finder?
- hmmmmmmmmmm, im stuck at this.

Yea definitely not very interpretable. What can we do?

- The graph is very hard to read. Why? I dont know which kernel is firing. i dont know whether two things are same or not. I see a line and i think that they are good, but that is the wrong way to think about this. the same feature occurring in both happens in the row, and can happen anywhere. Feature combinations come as lines. Which also might be interesting though (which patterns are combining together?). hmmm

I also dont know the boundaries, maybe mixing them wasn't the right idea? But what can we do about this hmm. Lets say i want to keep the image "spatialness", although this might be something about aadat? no. wait cant say that. hmmmmmmm
How do we keep the images spatialness and keep continuing?
A pixel can have "x" components. We would need to highlight which component lit up for which. We basically show the codes for each image for each kernel.

For a given kernel, for a given component, see the codes ka distribution on the image. simple.

Ohk, the kernels have not done the overlap support thing. Basically our good old recon loss overwhelmed the weights loss :).  
Gotta check that tomorrow. i need simpler alarms on when the losses are not behaving the way they should though.

Ohk, ive got a way of analysing what all is caught at a pixel. Let's try to track what each pixel means then.  
How do we do that? We create a graph first (of the activations).

Each node is actually n-components + 1 node (an AND node, and n-components label nodes). Or each node can also be considered a vector of codes, thats also okay.  
We give each a name, and we see what the final activations mean in the end.

Looking at the graph is not that easy hmm. But we do have better codes now i feel. let me try a bit more.

- We have a lot of overlap between the dimensions for n=9. Interesting.
- i would need a training cycle which does not do this bc.
- is dict learning the answer? hmmmmmmm i had problems with that too technically

Its generally always falling into a basin where the grads become 0 (for both recon and weights) for higher components.  
The problem simply is that of too many basins out there.

We need jerking. ill need to implement warm restarts then in learning rates i think then.

The loss function is getting quite complicated lol. Lets see if the correlation matrix approach works ;\_;

- it seems the whole sigma_eps thing was making my gradients very noisy. Now, there is only hyperparams i think
- so instead of the math thing, im gonna replace all hyperparameters simple.

- we use different ranges for alpha (still do annealing though, to keep gradients happy). SGD gradients can be very noisy. Ive to normalise it (im not happy about it but whatever). I might still do Adam.
- So we have alpha, beta, gamma and delta. simple
