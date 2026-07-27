Good paper, although the math was slightly intimidating at first, after a bit of reading it is becoming clearer.  

They basically have two sets of models, the first is the basis vector `phi`, and the second is the sparse code vector `a`.  
They make a reasonable probability distribution argument about them.  
That would require some writing lol.  

I also now understand why there is a length constraint on the `phi` matrix.  


A lot of math is now becoming more intuitive. They are converting intuition to math distributions.  

The sparse coefficients basically should be generally 0, so they say that the distrbiution of these coefficients can be chosen from laplace/cauchy (these have 1 at 0, and go down sharply to 0 on either side).  
For the error term, everyone assumes gaussian.  
This error term gives the probability distribution of `y | inputs` to be gaussian (cuz inputs are fixed).  
So you bascially just descend on minimising this error distribution, leading to MSE many times.  

It would be fun to write my problem in this way. The first thing is maximum likelihood estimation.  
Lets say we have a gaussian curve, with mu,sigma. The probability distribution function gives the probability of observing samples `y[i]` given sigma and mu.   
We have the samples though, we we want to find the correct sigma and mu, which fit them.  
You basically want to say, how likely is some parameter `mu[i]`  and `sigma[i]` given dataset `y[i]`.  
This is `P(mu,sigma | y)` instead of `P(y|mu,sigma)`. This comes into bayesian rules now.  

```
P(A | B)P(B) = P(B | A)P(A)
```

`P(params | y)` = `P(y | params) P(params) / P(y)`.   
Now im not sure what to do xD.  

What is the `P(params)`?  What is P(y).  

# sparseness prior

L1 on coefficients directly, not very bad. problem is that L1 on direct codes pushes the coefficients down too much in value too.  
```
finetune epoch 4800 | recon_loss 2039.5404 code_loss 2040.9456
finetune epoch 5000 | recon_loss 1500.0206 code_loss 2109.8135
finetune epoch 5200 | recon_loss 1735.0452 code_loss 2122.0308
finetune epoch 5400 | recon_loss 1522.7332 code_loss 1954.7939
finetune epoch 5600 | recon_loss 1475.0591 code_loss 1969.1360
finetune epoch 5800 | recon_loss 1665.5251 code_loss 1994.0542
finetune epoch 6000 | recon_loss 1476.2241 code_loss 2106.4749
finetune epoch 6200 | recon_loss 1708.2336 code_loss 2045.3422
finetune epoch 6400 | recon_loss 1529.5114 code_loss 1899.9451
finetune epoch 6600 | recon_loss 1967.4382 code_loss 2023.3951
finetune epoch 6800 | recon_loss 1796.7842 code_loss 1921.0477
finetune epoch 7000 | recon_loss 1467.7522 code_loss 1913.7998
finetune epoch 7200 | recon_loss 1617.0540 code_loss 2138.6177
finetune epoch 7400 | recon_loss 3357.2083 code_loss 2272.3069
finetune epoch 7600 | recon_loss 2017.4265 code_loss 1944.9791
finetune epoch 7800 | recon_loss 1489.6775 code_loss 1900.6624
finetune epoch 8000 | recon_loss 1955.9692 code_loss 1881.5790
finetune epoch 8200 | recon_loss 1902.6034 code_loss 1862.1335
finetune epoch 8400 | recon_loss 1461.6171 code_loss 1827.7040
finetune epoch 8600 | recon_loss 1280.8079 code_loss 1861.6217
finetune epoch 8800 | recon_loss 1555.8617 code_loss 1863.3916
finetune epoch 9000 | recon_loss 1845.7153 code_loss 1893.6752
finetune epoch 9200 | recon_loss 1459.5586 code_loss 1825.2336
finetune epoch 9400 | recon_loss 1661.3124 code_loss 1845.3879
finetune epoch 9600 | recon_loss 1881.1093 code_loss 2140.8853
finetune epoch 9800 | recon_loss 1434.2057 code_loss 1754.8931
```

Interesting, cauchy prior on encoder is making it harder to do correct reconstruction.   
The codes are also not 0.  

Cauchy on encoder is in general weird.  
its log likelihood is log(1+(x/sigma)**2).  

for higher values in the loss, you would need a smaller sigma. that also corresponds to sparseness theoretically in cauchy.  
that however then, makes the gradients extremely small. (the grad would be 1 / (term inside log, whcih is huge)).    
Might be worth writing about.  

It might be the reason why my other prior on weights "flattens" out after one point, the grads are too low once the log is too high.  
The higher value of the log cuts the loss, making grad descent not want to play bad.  
I guess the sweet spot would be where both of them nullify each other? not sure.  I'll have to see if there is something in the math which supports nullification.   


There is the other problem that the dict is created from codes, amazing how grad descent makes it work lol.  

In any case, next todo things:
- coordinate descent. 
- try different alpha on olevetti

## next steps again

- i've got a reasonable grip on what to use as hyperparameters. the relation between them. this is reasonable progress.
- i need to try on olivetti again with the new params
- need to now also see how i can understand if two patterns are walking together almost all the time
  - that is, two components are always coming together, the decomposer overdid it.  

- which do i pick first?
  - oliveti is the least effort, hopefully it would be easier to do

## next steps 22/may

- olivetti worked beautifully. the cocktail party thing, and topic modeling are not working
- ica is not working on kernels, but our thing is. now its time to move on to that topic.
- the other part is understanding if something is broken more than needed
  - i also need to check if its going to work well in the synthetic case when we have one dim which is subset of another but is rare

## next steps 23/may
- ohk, random experiments are done now, its time to write.  
  - I'll write the draft blog with the math
  - basically all the details i want to add. without the show off part or making the introduction easy enough for people to skim and take notice of
  - the work with hyperparameter tuning is also useful
- this should be started tomorrow, else i'll keep procrastinating
- the other part is to also work on testing noisy scenarios, and specifically findings indicators of what "good decomposition" means
  - create another group of atoms which own a subset of existing group. fire them for some ratio of ds. see when decomposition considers it noise vs pattern
    - find what are useful indicators for it
  - simply using the elbow might be useful, although increasing the components requires me to increase the tightness around W, so not sure if this would be helpful
  - the most useful part might be trying to fit using coefficients of a single dim, to another dim, using only scale. check the residual
  - check how much variance of the data does an atom do? the signal that it covers?
  - these are the ones that come off the top of my head right now. 
  - what are the scenarios for breaking? Currently, im only testing on clean data. i need to introduce noise that i expect would come
    - a group of dimensions is really 3 groups, and require further decomposition
      - we want to know when decomposing further is useful. Even if we don't do it, is it possible to find out about this?
        - Lets say i did do 3 more comnponents for the next cycle
        - we basically do a sweep +5, each time
        - we can explain "80% of data had these three collapsed", etc.
          - but for a single collapse, we would like to know the amount of data erroring out
          - so we can assign an error metric of what if we collapse
      - i might be able to check the elbow by using reconstruction accuracy i thikn. the elbow might be the most reliable way i think.  
      - maybe seed stability might be quite useful i think

## next steps 24/may
I need to finish setting up the tests for synthetic data. the goal is to stress test the model.  


- Make two atoms overlap with each other. see what we get. increase one component and see if the overlap is taken out.  
- Make an atom a subset of another, and fire it randomly, at some ratio. see what is taken out. Loss should go down.  


- Things of importance
  - how much noise can the model handle? How do we spread the noise?
  - how much overlap can a model handle? This means: we overlap some ratio times and see how the model behaves. Increasing components helps the loss here.  
  - how stable is the model across seeds, very easy.  
- finally, we should be able to call the data gen function randomly, and determine num components directly from the model


If two atoms overlap, then it really is just 3 atoms, for now, this is fine.  


### noise

- global noise, added as a ratio of std dev, easy to test
- local noise, make an atom "misbehave" in its dimensions, some ratio number of times
- overlapping noise, make two atoms overlap randomly some ratio number of times

### stuff to keep track of

- svd initialisation gives deterministic outputs across seeds. It also gives good graphs of reconstruction error (gets closer to baseline at the correct n_components)
- using the correct sigma_eps is important. otherwise the training does not converge enough


- ohk, i need more structure and need to regroup. 
  - each experiment is the type of data we use
  - in this, we check how svd results look, how stable is the solution across 20 seeds, find the most stable atoms.   
  - compare results of svd + stable atoms + true atoms.  
  - compare graphs for runs (losses, gram error, etc). 
    - this would require me to use some sort of aggregate for each n_component for random seeds.  

## next steps 27/may

- its important that i finish the paper by tomorrow i feel. For today atleast, all tests need to be finished. For whatever I'm ready with
- lets first create the nesting of hyperparamters
  - log term add or not?
    - sigma_eps vs sigma_0 vs sigma_s
    - initialisations
      - svd or not
      - with the correct sigmas or not?
      - it does make sense to keep $w$ in the same range as sigma_0 simply because alpha depends on sigma_0

- the easiest thing to do first is to recognise whether the distributions of weights are even finally at sigma_0 or not.   
  - normally people do weights normalisation.  
- lets do this first. see what ranges the model comes up with in different tests.   
  - i only first do for the simple baseline, unregularised
- we do follow curriculum learning though

- since there are so many hyperparameters, it makes sense to go through them one by one and prove the logic.  
  - we will do this for many tests i guess?   
  - for now, we have a set, lets do it on them simple enough

- Now i need to run every test on small datasets. and see how it works. then keep it running in the night for bigger ones
  - for that, i need faithful saving and loading from the small dataset
  - i will need to save the data im getting somewhere, so that it is perfectly reproducible. then load it somewhere to see the metrics separately
  - the easiest way is to simply dump the whole thing using pickle, for each run, includign inputs and everything
  - then reload in another notebook to see them
- each run needs:
  - alias for the "type of test"
  - alias for method used for descending
  - all different configs.
    - now all different configs can be stored as json i think, which can be used to retrieve the files

## next 28/may

Good, we have prelimnary results on ideal data now. We'll also run this from colab for more data.   later.  
A bigger correlation seems to be for data size compared to atoms and dims, instead of the other params i think now.  

- Convergence seems to mainly depend on the sample set's ratio.   
- `ln` does not matter. We'll keep it.  
  - Needs checking though, will think about this.  
- sigma_s = sigma_0 = sigma_x gives good performance


It would be useful to do experiments in colab, but we wait for answers from different init strategies.   
They are running. I've got time. But its best not to hurry for now. We have stable results. wait for it. do fastai.  

Note that now, MSE is per number (we take the full mean). For lower dimensions, the MSE is of the order `0.009072` where we get good similarity (1%).  
For failing dimensions, the MSE is 0.131610 (13%) per number which is quite high.  

We need to see if different initialisation strategies help now.    


For now, we have two things
- Data size (how many samples compared to dims and atoms)
- sigma_s = sigma_0 (this is settled)

- ive to merge the dicts of the warmups done after not discarding the optimiser.  


- the main thing we wanna test is how the sample size affects similarity.  
  - It might be useful now, to decrease sigma_eps further for cases with less data to try and get better reconstructions (increase the force of the reconstruction loss).  

For colab, we'll not do the sigma_s=sigma_0 experiment at all. We'll see if there is any benefit to removing ln in sparse data, so its useful to also do it for pure case i guess


For fastai, im finishing up the convolutions lecture also, ive to cover two notebooks, collaborative filtering and convolutions, along with the tests on the article.  



### Analysis of the runs

- Add that we will use sigma_s/0 equal case only
- We in general have a problem of needing more data. How do i plot it? There are 4 variables
  - number of dimensions
  - number of atoms
  - similarity (y value)
  - number of samples
  - we generally see that more dimensions / more atoms -> need more samples to get good similarity.  
  - This is generally hard to see hmmm.  
  - what might be useful is to see how many samples we need to reach 90%, but we dont see that graph for 100 cases
  - simplest: for each n sample ratio and atom ratio pair, see what similarity we get across dims.  

- There is in general a problem of setting the correct hyperparameters. its too damn finicky.  
  - The whole sigma thing
  - its best not to think too much about "perfect" data. we assume a sigma_eps of 1e-4 and continue
  - using baseline is working for now, no problem
    - gives us reasonable sigma_eps
- amazing how annoying this ended up being, it takes a lot of work to make a method "good".  
  - I should also start working on noise
  - the simple strategy is this:
    - we now know which hyperparams are good, they are the default right now
    - for noise, we simply start with gaussian noise, similar setting to current work. 
    - We basically want to know how well the model works out given some global gaussian noise
    - it would again have correlation to the sample set size.  
- I would also want to run this notebook in colab. without extra bt.  
  - for the 1000 case.  
  - run with 100 epochs, finishes fast. then see if everything works, then let it run after using runtime.unassign. easy.  
  - i would like to know about 
  - we do need proof that sigma_s=sigma_0=sigma_x/root works. so we would need to run the cell.  
  - for log, there is no need. for svd, i should check it out.  


So what are the important hyperparameters?
  - init strategy
  - ln term
  - sigma_s/x
  - for now, do i want to think a lot? i can just run them.  
  - but ive wasted time like this.  
  - hmm i need clean result. lets first run this thing for 1000 in colab i guess. we need to use drive for correct shelve usage


- ohk, have run in colab, for pure case, init strategies.  
  - for 1000 and 10,000 dims
  - it seems some babysitting is always needed lol
  - the best way might be what i had in mind, clustering and then finding the most similar ones, the algorithm is simple:
    - run i for some k runs
      - for each component, calculate how MSE changes with it.  
      - cluster the components, starting with maximum MSE changes.  
      - pick the top MSE changer, with some threshold on the number of repetations required (do not do clusters with 1 object at all)
      - now, the dimensions it explains are good to go, we work without them. 
      - repeat. break when
        - we get zero clusters (many dead atoms compared to previously picked atoms).  
        - The MSE does not change a lot (by some tol).  
        - n_components have been picked.  
  - technically this is not very difficult. The only problem is that runs can take time. Am I willing to wait?  
    - This is worse for higher dimensions. We are running `k*n_components` times for every test.  
    - Is there something I can do to make this faster?
      - The obvious speed up is to take out multiple disjoint atoms from each run.  
      - So after discarding bad clusters, we have a set of good clusters
      - Each has MSE, we can simply say "if they explain variance within a threshold", like cluster1 is the winner with 0.1 and cluster2 is second, with 0.05, and the variance threshold is 0.05, we can use cluster2.  
      - The best way is to cut up the clusters in disjoint cluster sets. From each disjoint set, pick the winner. Thats the easiest.
      - How do i cut by disjoint set? We calculate the support overlap. If someone has support overlap > thres with someone else, they are in the same cluster, we can do this by taking `1-support-overlap`. This algorithm would need testing though. Testing on two cases might be good, Gaussian noise, and perturbations. For 100 dims, 20 atoms is fine.  
      - this is a fair strategy.  
      - but needs testing, would take the whole day. Should do fastai with it too. But is this worth the write up? It is if it gives consistently good mean similarity.  
      - Testing would be the bulk of the time.  
      - We do algo0 first (one cluster per run).   
      - then we see if algo1 is good. But for that to work, I'll not do the automatic selection right now, I'll check out the results everytime.  
  
- well, ive been running the loss on the W, i should run it on WS and see if it works.  
  - for each vector `s` we have `n` coefficients, which are all multiplied to their corresponding basis vector.  
  - The column regularizer is on this intermediate matrix.  

- ill nmove on now, for tomorrow
  - finish collaborative learning notebooks, taht finishes part1 of the course. i can either do this or skip and do the next lecture (much more interesting). might do that
  - use ica and the new thing to finish one kernel in MNIST.




# ICA / PCA / SVD

All give the same results with overlap though.  Can this be non-overlapping at all?  
Who knows imma needs to check.   

For us, the noise is a lot though.  
It is interesting. I can make out some atoms which are coming together. What can I do?  



Problem:

- noise, makes it less interpretable. I need a metric of whenever a new component branches off.   
- I can start with 2 components. make one. freeze it. train more. get one more. freeze. then one more. freeze. and so on.   
  - This might give us the top level components
  - Then it is time to see which ones can be decomposed further.  
  - We unfreeze one, copy it in another atom. Train those two. See if the MSE changes for the better. If not, then that does not need further decomposition.  
  - We do this for every atom. Which atom is good for decomposition? the one with the highest MSE in its dimensions (in its dimensions is the key).  
  - We continue until we feel we can decompose, then we bail.   
- The code technically makes the first atom catch all. thats okay i think. Maybe. Lets see.  

# June 1 notes

- fastai collab filtering is done, very easy. Need to recap a bit of 9 and start 10.  

- Lecture 10 fastai
- Move forward with decomposition somehow lol
  - I can either tweak it a bit more
    - run on more data to see the stability
    - run multiple seeds and find the stable ones



The problem right now is truly that I dont know what decompositions are "correct".  
- A decomposition happens: is that worth it? We can find it by finding the MSE difference from that decomposition.  
In this case, I first need to figure out what decomposed into what. We can do this manually for now, would be interesting.  
- interesting to see the number of samples a pattern was positive for. 


Main problem is I myself don't know a good decomposition criteria, I should work on that first.  


huh, i think the recon annealing thing might just give similar results across runs.  


- Get all the original components that were extracted by doing multiple runs for the same seed.
- find the unique ones
- now they would have overlaps. we want to get the best combination
- if i try to train with all the components added, it might add new components, this would remove the purpose
- an interesting thing i can do, is keep the weight loss, keep the codes loss, BUT freeze the weights. the codes would find the best combination, might be interesting to check this out.   
  - finally something good clicked hehe.   


- how do i find whether the stuff is working though?  
  - finally i would have a set of coefficients, i would like to know the "groups".  
  - A group is a set of overlapping dimensions whcih dont come together. overlap is soft and hard to calculate. 
  - so we find the people whose codes dont come with each other (im a code, i wanna know others who were always near 0, we can find mean and shit for this, idk. it does seem like correlation matrix)
  - wait, top level first
  - hmmm, grad descent can btw cheat also, it might find components which do overlap for differnet cases, hmmmmmmmmmmmmmmmm
  - the best case would be to just find the components whose `s` would be very low generally, we prune them out.   


- we still have slight stability problems, BUT. we know that some of them are extremely common and come up all the time.  
- I'll simply cluster the values across seeds then, it should work, good feeling about it yessssssss
- We will be pragmatic, we only pick the stable ones


Ohk, looking at the outputs, I can now guess which ones are most "stable". The question is on whether we simply use them or not.  
And how do i quantify it?   

A small problem with clustering is that it can get weird quickly.  


- No more of coincidence programming, im losing discipline in my hurry to do more experiments, in my hurry to not waste nights whcih could have been useful computing, it just gives more pain later. Make it work first on a smaller scale to perfection, then run the model on a larger scale.  



- The thing is working. Now what are the next steps.  

- First is cleaning this up. And running this for all the others.  
- So i need to get the stuff working


What were the steps until now?

- Collect patches (run the whole model on the full DS and get the dataset)
  - First go through the kernel's contribution graphs, see which POIs are useful, then use them.  
  - I've to do POI by POI which is a problem of my attribution method (i dont know if i can take from all the points and threshold correctly [since the POIs have different thresholds])
  - This is for now okay, I'll also need to try more reliable methods for getting patches (which would be basically all methods which make sense to me for now, I also dont mind doing neg and pos separately to maintain the signs and all).  
  - Select thresholds
  - Run the model again, get the internal activations this time for points, with thresholding
- Save the full samples.  
- Take a random sample, save it (this is the training sample, important)
- Train a model on that random sample, across components, across seeds
- Each model has important scaler values that it was trained on, we would want to save them with the model. So we save the run and the scaler itself
- Find the elbow
- Pick some epoch. use its min loss val (i would like to also not use very low score stuff, but that is an internal detail)


It might be useful to keep a separate notebook to show GMM thresholds are good.  

# June 3

- Runs for 4 kernels are done i think. Im analysing one to see if there are any problems. if yes, im going to run them again. But first we will verify
- The trend has changed a bit, its interesting. The loss now goes to a minima and then shoots up. It seems its because of dead atoms for now.  
- It is important to reiterate that every intermediate step has a set of choices. How to threshold attributions? Which algorithm to use for attribution? How to threshold coefficients? How to calculate scores?
  - For now, I will need to keep track of everything. And keep moving with whatever I have.
  - Debugging later would be painful, but I cant really fixate on everything
  - At this point, the codebase is also reasonably big. With complicated pieces.  
  - Each has been tested, with its own proofs and all. But I dont have perfect proofs. 
    - Although, model run strategies do have enough proof i think (I need to defend cosine annealing and not using log term)
  - Ive fixed the thresholding for coefficients also
  - Need to make sure i dont forget the bigger picture

- The scores are pretty reliable and stable now. They dont change or do any weird stuff.  

# June 4
- JAX is ready. Need to test on colab. Also need to test if model can be retrieved correctly. Will be done in colab now.  
- We use the older structure (directly saving SingleRun), it now contains encoder and decoder separately as np arrays, for compatibility between torch and JAX.   
  - Its the easiest way to get stuff done rn, JAX stores checkpoints directories and all. not interested right now at all.  

next steps?
- Test on colab. Check seed wise perf
  - Add support for git in colab (im getting auth problems) ✅
  - uv install pt-to-api in the colab notebook now ✅
  - run across 1 seed and multiple seeds to see if we are actually getting a boost 
  - then retreive the model to see if its working correctly end to end ✅
  - run for one kernel tonight at least.  
