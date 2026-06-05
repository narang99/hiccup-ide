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
