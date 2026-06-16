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

Im not able to actually find a lot from simply looking at the graph. its now time to annotate every pixel. and get meanings out.  
Its basically simple, each pixel has some set of tags. That set of tags define what the pixel is doing. For now, this is fine.

Did i make a mistake by not using dictionary learning here? Cuz, it would give me more exact stuff. hmmm nvm. For now, lets move on.
Actually, one last time trying dict learning :)

What was the problem with the overlapping thing? Im not sure now lols. Odewa, aaho.

Ohk, i might go kwazy :)  
Like real kwazy :)  
Using patches is quite good, but it takes an extreme number of components. i dont know if the network is THAT non-linear.  
And it wont scale :)  
So lets go with pw only. Its only about finding a reasonable one which helps me make correct deductions.

# Layers.0 seems to be good

- Auto assigning is mostly done for layers.0.
- It has given me the scale of the problem lol. Good humbling thing. Now what are the next steps?
  - Many components for each kernel might map to simpler concepts (which are more umbrella).
  - I can use them to make the meanings ratio smaller. but thats okay i guess.
- Now, lets say i know what everyone does (or not). thats okay
- Ab, next layer ka dekhna hoga. in that case, i will then look for a single output pixel, and see its input combinations. We do one kernel tomorrow morning, check its graph. others we'll show 0s only. (somethin empty lol).
  - phir check out each pixel, and its input combinations. make a library of that small piece of graph.
  - add support to recognise that graph.
  - I might actually find that graph in many places, we can take a small piece and think of doing that.
  - After one point, graphs become more "subjective" in what they do. at least for higher level concepts i think.
  - Will think tomorrow now

- mere paas ab hai pixel wise meaning for first layer outputs.
- ab kya karu?
- mera final goal is to mark ke har pixek kya karra hai, for that, it would be useful to start with this layer atleast first, konsi category kya karri hai

For now, patches are good. we gonna use that now.

Ive again reached the problem i had with normal dictionary learning. it does not give the correct representation for more dimensions (you need it to give only 1 active coefficient but it goes wonk).

- things i can try in dict learning itself
  - our weight loss (in the JAX model maybe with a sparsity constraint?)
  - maybe correlation matrix weight loss is also worth trying (cheap and easy)

- hmm hmm hmmmmmmmmmmmm, hehe

- When a thing is done, it can use dictionary learning to increase the components and handle the dims for that component. We only run for reduced dimensions. Now the question is, how do we select n-components?

- For the case of layers.0, dictionary learning would give some 20 components (which are the right answer, they are interpretable).
  - our thing would degenerate into weights having one weight active

- For the case of layers.2
  - dict learning would descent to very less loss, with a lot of components and a lot of support overlap. I guess support overlap is the thing. we check dict learnings results and see if latents have support overlap, ez.

Also, tis JAX model is taking too long for patches weirdly, and not on pointwise, interesting.

- Anyways, we would want to find which components belong together, in this case then, we can simply let dict learning handle the dimensions for us (no need to even initialise it with our weights. simply use our model to find the reduced dimensions)

# I honestly dont know now

- Dictionary learning gives me weird representations (it tries to go over everything)
- so does our algorithm :)
- tis too much now ;\_;

Lets still try out the mix of our algo and dict learning. we basically want to find the lower dimensional major structure, on top of which dict learning finds the bigger structure.  
How can we do that? lower dimensional structure = PCA or something. I've noticed that the output of PCA is similar to what we get.

hmmm, the optimisation problem is hard for dict learning. DJL is killing off too many activations.

Time to think, what are my options now.

Dictionary learning has very good optimisation performance, there is something these people are doing which is making it work that well.
Normal SAE does not work as well as the dict learning thing for me. Interesting.

## Two things

- try dictionayr learning wiht more L1 penalty
- try the JAX model thing again. we currently have this problems
  - the dimensions are too strong (it works too well.).
  - so first: only take the samples where this atom was active
  - second problem is that we dont have good dimension split.

- try our loss on the "codes". This is a way for sparse codes i think. L1 has a kink. we dont. lets hope it works out :)

# Kmeans

yes. back to this. this sucks hard.  
Test the hypothesis that kmeans is not a bad option. Dont overthink it.  
Just do it for all layers, tis gonna be much faster. Then see if the graph theory is right.

- for a pixel, find its graph
- Given a set of graphs, find equal ones and see if they have the same sample

Guess imma very stupid. stupiddddddd.

Ohk. Kmeans done everywhere. Now time to look at the graphs. First check out layers.0. See if the points are making sense. Then do layers.2

- For layers.2, ill first check the output points in isolation. then we do something about creating the graph
- We want the points which are "included" in the centroid. That is, the points which the centroid cares about. Otherwise ill just end up with the whole layers.0 points, and that might not be very useful. for now though, we do that only. Then see if i can remove some points which are closer to 0
  - technically most of the points we have should be "interesting" to at least one kernel
  - we basically want to know the points which are getting closer to 0 (the kernel ignores them, we assume this here).
    - this should be a simpler proxy then. im not sure how i would have a reliable way of doing this though
    - the point is useless if the dot product with the kernel is very low compared to its original value. abs value.
    - we'll simply threshold here for now :) (10% in the pointwise mult).

- lets find the best patterns graph.

The process needs to be simple and repeatable. like GPT.
Easy. hopefully. lols.

The more pressing problem rn is not getting the exact graph though, ill need to check it out.

Ohk, the graph is too fine. What are my options?

- Lesser components, are we getting too many patterns? Maybe
- More threshold on attribution graph for extracting patches might work too.

oh god. ive got one threshold per input pixel. each kernel in a layer would have a different attribution threshold ;\_;  
but wait, i only get the inputs attribution. i would need it wrt some kernel no?
we have it wrt kernel lol.

Ohk, the first thing i do is the simplest one, tone down the number fo components. l2 norm. ill also save the max ones.

Cool, we have the attribution calculation going now.  
It might be interesting to now do the crawl in the network while checking out what each component is doing.

The basic stuff is working:

- For a single kernel, find the integrated gradients to get important pixels. For each pixel of importance, find the integrated gradients. This is the first step. It tells us what each kernel is picking up for a given input.
- Now for every pixel that we found the integrated gradients, find the input pixels which were there for this kernel. Take the integrated gradients of those pixels.
  - At this point, you have
    - Layer name
    - Neuron selector.

Now, we are interested in getting the tensor positions a position is dependent on.

Ohk, patches are done. for racecar and cat. Now for clustering.

- collect all the patches to disk. get the weight also.
- get pws. learn kmeans on that. simple. now we just have a basic channel datafetcher, we have the infra for this.
  - although, very less data so lets see. If it does not work out extremely well, i can get more data.
  - For testing, ill need to get the neuron attribution for each label later.
  - First kmeans

Turns out, the cluster stability in kmeans was due to using kmeans++ initialisation lol.  
Onn random, it suffers, takes an extremely big hit.

ohk, tis clustering. now what? We assign labels to new stuff, and find that place's gradeints, save them. and we good

- Run Layer attribution for input with correct target class
- Find all attribution values above threshold
- get corresponding input patches
- get label for each patch
- put stuff in layer_name/channel/label/ directory. for now, tis fine (although, it would be useful to also add the actual answer the network gave (the classification id)). We can use webdatasets for this too, easy i think. Shard it and good (it takes care of keeping stuff in order well enough, i can add the input key also, easy peasy lemon squeezy).

- Ohk, so, abi ek hi example karega mai hehe. :O
- hehehhehehehhehehehehehe

- im very unproductive right now. Ive done clustering. now i need to test the cluster.
  - Go through all the attributoin patches, assign labels to patches
  - for now, we do multiple runs on the patches, each run finds a single label, simple
  - problems right now are that we have less data, very less data for this (the upper thresholds have very less data actually).
    - i might decrease the threshold later
    - we have 250 components for 5000 data (20 comps per center, thats too high).
    - (i also think it can improve a bit more, maybe we only include the activations with high attribution in the calculation?).

KMeans is working on mixed5e:55. We good. The first big problem is Integrated gradients takes a long time. We try the same thing with deeplift.  
While we do that, we'll also download more data from imagenet. We take 100 images per sample, sample. No more. Anyways there are so many classes per type of animal also, so we should be good. These are baseline images though, lets see if we can get better

- Do the analysis on another neuron
  - layer inception5a, neuron 233 [reference](https://www.alignmentforum.org/posts/eDicGjD9yte6FLSie/interpreting-neural-networks-through-the-polytope-lens)
- Best download 100 images from each category for now.
  - Easy enough. actually, lets do 50? hmmmmmm, CONFUSION. When in doubt, do more hehe, 100
- i dont actually need to upload to s3 if im doing only 100 per image. We put in dir, then shard them and upload to s3 for now.
  - I simply hate the drive throttling

- i might need a better runtime than colab let me see the options

- ohk, its there for uploading, ill use paperspace gradient for my work now. But before that, lets test deeplift
  - with kaggle and correct s3 support, ill have a lot of parallel running notebooks.
  - paperspace + colab + kaggle == 6 notebooks at least in parallel. that should be enough.
  - today however, we test deeplift. if that works, a lot of work is simplified.
  - and check out the mixed5a:233 neuron
  - well everyone says deeplift is good, lets see
- Iske alava, i can use the summit application to get the neurons to check.
  - it is simply for prioritising and nothing else.
  - in the end, we will now run with all image classes, so it should be fine.

- i overanalysed again. ese to kahi aage hi nai badhega bc
  - So first, we do deeplift test. see if it gives reasonably similar results
    - at every point: attribution threshold, clustering, visualisation
- Then do another neuron
- Then check if dictionary learning gives better results (im assuming that the existence of a dimension is useful indicator, some basis which is coming again and again is useful, in this case, we would want to check out by active dimensions for a given input).

- I keep wanting to do dict learning lol, but well, lets wait.
- first we would also like to do deeplift, it is a useful test
  - this is definitely faster, but ill need to do something about the disk thing.
  - activation capture is the problem now lol
  - its the size of the disk write ;\_;. I honestly dont know what to do about it.
  - Claude suggests LZ4 compression. We dont do float16. LZ4 needs testing and verification
  - Now for parallel reads and writes, it seems simple enough, ill think about it. LZ4 first though.
  - Wait, why am i getting all activations? I think i should simply do it online, and only get the ones which are high for a given attribution, i think that would be cheaper.
    - So basically, while reading attribution, ill simply run the model for activation again, there is no need to store all of them. This saves most of the headache.
    - Fair, ill need to test this out too. I was stupid in not doing that.

- colab is also extremely painful when it comes to disconnections. ill need a better option, i might even do aws lol. but it would be cheaper to do paperspace. later. for tonight maybe, we ll do the other neuron first though.
- and the deeplift thing too.
- A ridiculous pain between torch tensor and numpy arrays in the datasets. (which format are we using?). We will exclusively use torch tensors for storing now hmpf.

- Note that in deeplift, the disk speed is the biggest concern now. it seems reading attribution shards from non-local storage is just going to be extremely painful. i think pushing attributions on the disk should be cheap enough (although going forward there would be a lot more examples). I would need more disk space i think lets see.

- It seems some clusters are genuinely hard to do. Specifically the racecar ones. Its a lot of data coming from everywhere. I have some options. The easiest to get a clustering which assigns -1 to noise. This might be useful. Easier than dict learning.
  - we ll try HDBScan now
  - In the end, we do dictionary learning, but the implications are hard to define, how do i say what is there? Maybe ill simply show images with the active component.
  - the visualisations would be more complicated though, currently we assume we return a single label. Now we would return multiple (whatever is non-zero).
  - well hdbscan assigns a shit load of -1s.
- I need to step back and prioritise now. KMeans is the simplest, but the quality is not that nice. HDBScan is pushing a lot to -1.

- pehle pura code ke baare mei socho
  - we call transform for the trained model, which returns the medoid in the case of hdbscan, the cluster center in case of kmeans, the reconstruction in case of dict learning.
  - for HDBScan in particular, the number of clusters it assigns -1 is quite high (rightly so i feel right now). It is worth checking what the non -1s are
  - lets first create a report for hdbscan
  - it might just work out of the box, since -1 is a valid directory name lol.

- final code takes in a trained model, calls it for each patch with high attribution, gets the label, and saves the neuron attribution in the corresponding label directory (along with activation). now we would like to save it for multiple labels thats it. no problem.
  - tis wont be that bad
  - now for the actual cluster center, there is none, in dict learning, there is the basis vector which we can show (there would be a fourth component in dict learning, where we show the full thing, this sounds good).
  - currently we dont even show the cluster center.
  - its fine though, itna code change karne ki zarurat lag nai rahi hai
  - for each label, you want the cluster center also (or the basis vector, whatever). we can add that in the end. for just testing its not very needed.
    - the main change now is to return a list of labels instead of 1.

- if deeplift works, last step would be faster so we should be fine (faster experimentation).
- Although im changing stuff a bit, but by tonight, I need the final answers for the second neuron.

- most of the code is running now, i think i can go for a chakkar, but i need to save stuff in drive

- deeplift experiment took too long. drive is a bitch. In general colab is painful.
  - in any case, i do need hdbscan and deeplift's results. so we ll wait for this one. if both work, to chaandi hi chaandi hai
  - after that, ill have to move to s3 instead of drive (although ill do the next neuron tonight on normal colab and drive only)
  - we do that if we want to scale later :)

Currently running experiments:

- KMeans assigns a lot of noise to good clusters. trying hdbscan to see if it gives better clusters (and separates out noise)
  - If HDBScan does not work, ill need to find a version of kmeans which caters to noise, cuz kmeans does cluster well.
- Deeplift with KMeans, does it give similar results to Integrated gradients? 100x speedup if this works.
- Dictionary learning: can we use it for better reconstructions? The problem is more than one labels coming
  - you basically want to verify if there are some "major" basis vectors, and all the others are simply catering to noise
  - You see the distribution of coefficients of each basis vector, find the threshold for saying something is activated by thresholding at 5% its max value.
  - Plot the number of samples they are activated on. if only some of them activate a lot, then they are the strong basis vectors
  - if that is equal to number of non-noisy clusters, then its clustering.
  - The problem now is checking if we can confirm that these features dont come together (same sample should not have 2 major basis vectors, its entangled otherwise).
  - If the gram for the major basis vectors is relatively empty, this is fine and works.
  - But would it better than normal clustering? yet to be seen
- Why do we have noise?
  - A theory is similar to a previous experience.
  - in mnist, we had a diagonal detector, its diagonals were being used downstream by another kernel (say a curve detector).
  - now the diagonal detector emits a pattern, which looked like a diagonal edge, simple
  - for the next kernel, only this pattern matters, what was before the older diagonal detector does not.
  - now, diagonals are not very clean in the world, you might have garbled pixels along the way.
  - A diagonal detector might emit a non-zero activation for those garbled pixels.
  - So for a diagonal, it emits a diagonal edge, where one of the pixels is garbled. say they all have the same sign
  - then for the later kernel, this garbled pixel cluster is a diagonal only.
  - So it does not matter
  - So, the garbled pixels get high attribution.
  - can we verify this? What if a noisy label consistently has a good cluster counterpart in the same photo (and they are making a pattern for a downstream detector?). We get grass as noise. Maybe we also get good clusters of them. After that its a simple question of

# Speedup and estimates

- Download input to machine. for 50k examples, no sharding needed
- Collect attributions to find thresholds. (takes time)
  - Technically, has to be done once per layer/input pair for the total dataset ever. That should be fine. We can test how long integrated gradients takes for this, we would have attributions after this
- Collect all activations. This is also fine

All of the above mainly have a problem of taking up insane storage. BUT, we can do layer level at once, so its not that bad.

- After activation collection, find the patches with high attribution
- Run HDBScan

Now technically, the whole high activation thing is easy and fast. i just need to do one forward pass, collect the activations in a hook thresholded. This should be easy i think. We can make it blazingly fast infact, its crawling right now.

- We don't even need to do it inside the hook, simply do one loop and collect, no need to do anything fancy. I can even do attribution calculation here (no need to store attributions also now, although for getting threshold i do need to store them but we can use huge shard size and do with gdrive/s3 depending on what works (and is cheaper)). Im inclined towards S3 but gdrive still is much cheaper and easier (I never learn it seems).

Now, we have the high activations and attributions for the whole layer. Run HDBScan on each channel. Save the components. Simple, easy.

This obviously would take some time, assuming hdbscan takes a minute, we would have some 1000 minutes for a single layer. in this case, i would like checkpointing.

So we need to store the patches we are going to use in remote storage. This is the first important thing.
Then, we train the hdbscan model after downloading the corresponding patches package. This is quite simple now, hopefully i can also do this in parallel somehow, lets see. still not that big a headache.

We would like to store the HDBScan objects in remote storage. With some done marking or something.

This is done. Now HTML generation is quite slow and costly. Since we have to run it for all prospective attributions. BUT, again we have a lot of noise, so we will add a filter for noise.
We can also have a filter like `max_samples_per_label`. This would help do early stopping (make sure you go through classes interleaved to make this happen. We would still do a lot of passes if we do it on the whole dataset).

- Currently, we save all the attributions and activations at the beginning, and we simply loop through them (after thresholding).
- This is also fair, i might do attribution calculation and activation calculation here again too (although, this is technically a once in a lifetime activity).
  - but still, it is painful too. Hmmmmm. Ill measure how long it takes to do that.
  - For each layer, we simply need to do the activations of its previous layer. And attributions of the current layer. And we good.

Ese technically, the whole current architecture is not bad then. It simply goes through all inputs. Calculates and persists their layer attribution with previous layer activations. Trains HDBScan. Then simply runs inference on all stored activations and attributions.  
Question is, do we save anything by doing this? We can technically generate the attrribution the fly, its sinmply 2 passes. Neuron gradient calculation in fact requires a lot more passes since we go through all high attribution channels (but we would also remove older things). There is downloading and uploading and all also.

Hmmmmmmmmmmmmmmmmmmmmmmm.

Which approach should i take? Lets go through the current approach again

- Input is sharded
- Activations are captured and sharded
- Attributions are captured and sharded

One experiment i should do is see if i:

- do activation capture
- and attribution capture separately, do we get correct outputs again?

If that is the case, we just store the attributions and activations of the whole model in one go only. in S3, no problem. $0.023 per GB. I should calculate how much all the activations are total for each sample lol. thats like 23$ per TB, which should be fine for now. its only gonna stay a month (although, its still high).

This moves our main cost to simply running the model for neuron gradients. in this case, early stopping should be fine.

It does look realistic for me to train on the entire model now. Simply because:

- Layer attribution is for the full layer, not for a single neuron, so we should be good.

If deeplift works well, all the better bro lol.

Overall, the current architecture also works out i guess.
All i need now is automatic thresholding.  
HDBScan is blissfully automatic. Automatic thresholding should be easy to i think.

We can try the full automated pipeline for our neuron again in some time. we use S3 though, gdrive is very annoying. But gdrive is free lol. But it does not scale. Aaaaaaaaaaaa
S3 it is. im not wasting time now hmpf.

ill simply check what the size of the full activation of one sample is. if it is manageable, its gonna be fine.

I might do the last neuron gradient thing locally, or in an aws machine, where i can run the script for multiple thins at the same time.

Its not that bad though, took 30 minutes for 2k files (ive got 50k though lol)

I also need to get my shit together on what im storing, numpy or tensors? (always tensors please).

All pain goes away with better disk actually, let me see how much colab has. Aah, all the activagtions and attributoins are costly to store hmmm. do i need to store them? Honestly? It might be useful to run a test to see how long it takes to generate them.
Yes, i gotta calculates first how much space it takes.
All the copying right now is actually taking time. Simpler architecture if this iss fast:

- run simple layer attribution for the full layer. store it on disk only. find threhsold. upload the photo of the graph. upload compressed attributions (10x smaller is easy, simply pick every 10 steps after sorting). This is one time exercise for threshold finding which can be done for ALL channels in a layer in one go.
  - find layer attributions. save them for all input samples
  - plot graph. save and push to remote. Save threshold for each channel.

- huh, we couldnt do low with deeplift..... Tis a ded

uh, tis weird, tis no works.
weird. very weird. i thought it wqas working before. tis sucks 😭
aaaaaaaaaa integrated gradients run karna padega kya itna saara 😭

last thing we have to check if dict learning (cuz it does better reconstruction)

Ohk, fair. integrated gradients gave me the cat thiung. it seems we are stuck with them. Fair.
Two three things now require testing

- How much space does the activation of a single sample take? Thats also the space of our attributions btw.
  - Lets check this first
- its a lot, 35MB per sample.

We have 50k samples. That would be 1TB lol. Thats a lot of data. Not sure i would want that. It changes layer by layer though. Capturing activations is easy. Capturing attributions is painful. But theres not much i can do about that (fucking deeplift did not work ;\_; whyyyyyy, naazeeeeeeeeee).

Is high attribution really that important though? Hmmmmmmmm.
nvm, we have a system that works for now, use integ grads for the other example too.
mixed4d is 500K, so 2 are 1MB. so we have 25k mb, thats 25 GB of data, which is not all that bad.

Note that this is mainly for fucking attributions because it is very painful to calculate them.

Shoudl i do the test for deeplift again? it doesnt make sense that its not working

For one thing, there were very few patches from deeplift (the other one had a lot more)

Question is. do i do this for the whole 50k samples. it would be crazy high the values, 25GB of activations which are not very useful since im gonna ignore most of them. It makes sense to generate the activations on the go. The other option is only doing all the dog breeds, hmmmmmmm.
I do need to test whether this works out for a new neuron as fast as i can.

IG to time lene vala hai, to karna to padega.
uske alava, kya kar sakte hai? Activations storing kal dekha jayega i think, ill see if it is easy to do that thing on the go instead of storing it (we simply take a snapshot on runtime, its not very costly).
We only store the attribution for now.

My code does need s3 support bc. hmpf. Would it work out with cloudpathlib?

Ohk, need to think a bit before doing wojnnky shit now.

- 50k samples. Easy. We can technically download all to local but its not needed its fine.
- Integ gradients is also btw not as slow as i thought it would be lols.
- We use S3 now, for sure. no drive, allows me to use otgher services
  - webdatasets is a kinda bitch on this btw, so im thinking kya kar sakte
  - lets try cloudpathlib first

- 1 layers attribution calculation and upload took some 5 hours, with upload. not bad. I might be able to do multiple layers after all (theres technically only some 25 or something, if i can somehow get 7-8 notebooks, it should be easy).

- Ohk, now we want to train the model
- For this, first is threhsold calculation. A problem is the huge dataset, idk how to calculate the threshold.
- 1000 classes, 50 per class, 500KB one sample. so thats 25k mb, 25GB of data, thats gonna crash and burn the RAM. We do channel wise.
  - the simplest way is to batch the channels. But first, we do one channel with the kneed algorithm.
  - For batching the channels, you essentially take a slice out of the thing. Simply take a set of channels from everyone as a batch, keep the start channel number. Easy.

WAITUUUUU, 2.6MB, 50k files, oh yea thats sa lot yes

Hag diya hai maine pura code. phir bi chalte jaara hu, not nice at all.

Lets fix stuff now lol. Its not possible for me to keep going like this.

Dir structure:

```
image_shards_dir
  imagenet_label
    000000.tar
    ...
hiccup-workdir
  attributions
    # TODO: change this
    # it should be: layer_name/imagenet_label since thats the normal way for pulling and all
    imagenet_label
      layer_name
        000000.tar
        ...
  thresholds
    layer_name
      out-channel (or kernel number, used in neuron selector)
        thresholds.json
        thresholds.jpeg
  patches
    layer_name
      imagenet_label
        000000.tar
```

Did some refactoring, lots of changes. for now its fine. im hoping it does not conk out.
We are almost ready to do lots of neurons together. until hdbscan btw. not neuron integrated gradients (these still take too long and are a big bottleneck. we can test if deeplift gives similar results here, it would be at least a 10x speedup if the accuracy does not drop terribly).

But the first step right now, is to actually finish until attribution threshold calculation with the new code.

2-3 problems. webdataset caching is quite sad. it uses a single dir.since all my shards are named 00000.tar tis sucks.
it will keep querying s3. that is bad. for this reason it makes snese to download the attributions in one go i think.
and images. for now. a better way is to pass a root cache dir. hmmmm.
webdatasets is too simple and painful lol.

Problems of performance now:

- webdatasets caching is shit. It uses one flat dir, without any CAS, just uses plain file name. We'll need to maintain the cache intelligently (keep a local cache dir, and pass the nested path to it). Im assuming there wont be any cache cleanup but for now its okay. We'll manage that in the end (we have enough disk fo that).
- Right now, im downloading once using aws s3 sync, then uploading lol

Well, ive worked hard to do mass stuff. but kya mai gadha hu? cuz it might be possible that ab chalega nai 😭
I should have only downloaded useful classes, tis all happened cuz i took the full val set 😭  
I always assume stuff will work out and do a large scale experiment aaaaaaaaaaa

ohk, the thing is done now. threshold calc, hopefully its not totally wrong.

Good, now we need filtered activations lol.

Well, ive made many changes, im hoping i didnt screw things up :)
Only time will tell xD.

I can batchify activation finder also btw, for the whole layer, that would also be nice, speeds things up crazy.
Basically, every op until now was cahnnel level, its layer level now.

caching was a mistake btw, im hoping i didnt screw up any steps because of that, it was implicitly using the same cache file for all shards with the same name, quite shit i would say hmpffffffffff.  
It would be better to add functions to download before running and all.

Ohk, hdbscan itself takes quite a long time, it also does not show progress lol. Technically its only 14k samples but it is still taking a while. I'll have to try cuml later. In the end, everything most likely will go on GPU.

As more categories come, our visualisation would get more painful. It would be useful for each cluster label to have:

- some 50 representative visualisations from different places
- then we simply want collapsed visualisations for other tags. It should have double collapse, show tag wise collapse and then each tags collapse.

# Performance of different steps

- Layer attribution. 5 hours per layer on GPU. not a problem
- Threshold calculation. 15 minutes per layer. Negligible
- Patch collection for clustering. tested on CPU 1 hour for a neuron. But i was generating activations for all neurons in a layer, should be batchable. Maximum 1 hour per layer.
- HDBScan. On CPU: 15 minutes per neuron
- Apply clustering and do neuron attribution on the whole validation set. This is the costliest. 1 hour per neuron.

The last two steps need speedup. We can try cuML HDBScan (RAPIDS). It does it on GPU. said to be 100x faster. if we see that speedup, clustering should be a breeze (although we do need to collect all the patches and all, so for each neuron, there is some data capturing overhead also, ill have to see if all can be taken in one go). Or at least, if we can batch (technically, we can do a queue like architecture, one io thread, and one clustering thread.).

- HDBScan hopefully should be doable. Requirements: CUML is fast. And we can do the IO thing.

Clustering application on the whole dataset:

- This is by far the costliest operation. We extract the patches again (although this is repetation, if we can somehow tag the data collected for training the cluster, then it can be ignored.). Although, i can do this again if im simply collecting the patches for the whole layer in one go (thats not a bad approach). The same code as the training data gatherer (although, i again feel we can do something good about this, the problem is patches are per input, so we would have a large number of files, we would need to tar them, still, not too bad). If we do that, we dont do patch collection again.
  - Then its about running the clustering algorithm first (this also takes time)
  - and then neuron attribution.
  - We can basically save the state of clustering algorithm and the patches. And keep neuron attribution a separate step.
  - To decrease neuron attribution time, we simply do random sampling (although since we have only 50 examples per category, im not sure how well the random sampling would work out). I should just do the "representative" set. Easy. Sample 50 for each cluster label, run and generate report. Generating report tag-wise can wait later.

The goal for neuron attribution step should be 15 minutes on GPU (although this is also quite high but i dont think i can do better).

Two main steps which work at neuron level are the costly ones. And would require a lot of compute.

Assume that at best, i take 30 minutes for a single neuron. For a layer of 256 neurons, its approximately 100 hours of compute (GPU). I'll need 10 servers (thats gonna be costly).  
I dont see any other way though.
If i use colab pro, i get 3 concurrent servers with 100 compute credits. That would trake 30 hours (1 day). Add kaggle. one more notebook. And then add paperspace i guess, thats 7 notebooks.

100 compute units is 20 dollars. This layer would then be 20$.

Can I do better? Im really not sure lol.

So there are 14 layers. 6 hours each for gathering attribution data. Thats fine. if we have 7 notebooks, its approximately 1 day of data gathering.

The other thing to test is if we can do neuron attribution using deeplift for the last step. if it gives similar results, we would decrease the time a lot.

there are a lot of things to do lol. A problem with sklearn is that there is no easy way to serialise the hdbscan model in a reliable manner.

Ohk, neuron integrated gradients are taking an hour in the last case. Tis too annoying. Ill need to think now.
although, it seems its run on very few cases, in this case, ill need to see if i can decouple this only (i think the older cluster in one go on the whole patches set is not bad).

hmmmmmmmm. tis might turn out annoying enough. well, i do have a fair bit of work to do to get it to be fast.
The best way is to definitely sample or something, ill have to think about this.
One thing though, i definitely can run a lot of workers for hdbscan predictions on the final thing though. (note that if cuml works we good but for bulk prediction it might not be the right answer, its better to do on a cpu machine with many hdbscans done parallely).
I might just use aws lol. CPU machines are cheaper and the parallelism is definitely useful.

CuML is blazingly fast. It finishes before i even blink. As fast as threhsold selection infact. So it makes hdbscan trivial if it works that well.

This is becoming more real :)

Ohk. cuml is giving problems, will need further testing.

- Still, it working is important. it makes doing the whole network possible (forget the reports for now).

# Priorities 15/June

- Make CuML hdbscan work
  - try more approximate params in normal hdbscan also
- hyperparameter tuning of hdbscan
- Extract patches of the whole layer in one go. Come up with standard directory structure.
- Put stuff in package. Test on one layer.

Weird, cpu one is also not working now lol. Is the data right? I'll need to verify everything 😭

- Tests to do:
  - cluster with higher attribution threshold
  - cluster again just in cpu, without importing cuml, is it intercepting? thats unlikely.
  - check if float32 has duplicate values (hdbscan can get confused in this)

- well everything is definitely quite slow for experimentation, its quite annoying.
- at least the last step (report generation) is painful. it will take 15 minutes to just do clustering i think of all the points, its doing the whole patch gathering again, turns out, that really is painful. i need to speed it up
- After that we sample points and run neuron integ grads for report.
- What can we do though?

- Ohk, in general, the cluster at 0 will have lesser members it seems. The split is quite biased.
  - ill need to verify what is happening, it is marking a lot of stuff as noise btw
  - how do i get fast integ grads?
  - hmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
  - i just need to run it for some 15-20 samples.
    - if i know what categories to query its much easier.
    - Actually, i do have the attribution right? Ill just pick some of them
  - what is a faster algorithm for report generation?
    - first we already have attributions and patches. this itself takes quite a while,

- perfect. the problem with cuml was simply bad input cached data for some reason. removing functools.lru-cache fixed it.
- Deeplift for neuron attribution is working. we have all the ingredients now (other than hyperparameter selection for hdbscan).

Stage 1 (Layer level) ✅

- Get the layer attribution for all input samples
- From this, get the threshold using elbow, save in S3

Stage 2 (Patch collection)

- Run model inference. get the patches which are above threshold for attribution
  - this is quite slow right now. we need to add batching
  - Currently it is running at neuron level, we want to collect patches for all neurons in a single example in one go
  - We'll save the indices and patch in a shard (like we do for attribution and inputs)
  - In the end, also create a single pt file for each channel, containing data to cluster on

Stage 3 (HDBScan clustering)

- download the cluster training data. run hdbscan
- We might have hyperparameter selection later, ill need to test it out. ✅

Stage 4 (optional)

- Once we have all the neurons of a layer, do the cluster prediction on the stage 2 patches
- We get the labels from this, along with indices. use those to do report generation (optional)

Stage 5

- use the cluster predicts on the whole network on a single training example.

# Tests remaining

- Deeplift test again with new setup. full deeplift instead of integrated gradients
  - im testing this again lol.
  - tis okay for now, its just basic compute usage, it would speed up stuff if it works.
  - pre pulling before running is helping, i get 7 minutes instead of 1 hour lol
  - pre pulling needs to be tested with IG too then. i guess vo bi phir 2 hour mei ho jayega.
- hyperparameter selection for hdbscan. we have the speed to do multiple clusterings to decide which is good

Tonight, i need to run code for some layers at least. (attribution calculation minimum).
For that, we first need some standardization in our code (needs to go to lib).

There is something weird about deeplift. the results are def different from integrated gradients i think.

Interestingly, it does give new labels, its not honestly bad. There is one big difference, it has human faces. while we did not have them in integ grads (although, i have not checked out label 2 [the later options] of integ gradients, ill need to see it).
Its honestly hard to see the difference, ill need to read the paper to judge.

These are the last smaller problems to tackle, overall the method seems to work.

There are human faces also coming up. The clusters have two extra minor categories from deeplift. These are okay, not too bad.

I should keep these reports that ive generated, they are important for the blog.

We'll have to test a set of neurons i think in two layers. its okay to let the first test run on multiple attribution calculations.  
Deeplift is also extremely cheap (7 minutes). So its not very painful.

- There is one minor hitch with deeplift, but its so much faster lol.

Now, about clustering stability lol. I need to define what a good cluster is definitely.

# DBCV paper

- Defines a distance, similar to how HDBSCan defines it
  - It has its own quirks which are hard to reason about for me
  - add more here
- Then it creates a Minimum spanning tree for all clusters (1 for each cluster). The MST is basically the graph connecting all the nodes inside the cluster, with minimum total edge weight. This is also similar to HDBScan, where they also created an MST.
  - intuitively this makes sense, it kinda approximates "minimum distance of a point to the next for each point"
- Then they define the "sparseness" of the cluster. Its the maximum edge weight inside each cluster's MST essentially (it only uses points "inside" the cluster, not the points on the cluster edge, these are basically points which have only 1 edge).
  - they call these points inner points, the edges connecting inner points are inner edges
  - They use the max edge length of the inner edges to define the sparsity of the cluster
  - This is called "Density Sparseness of Cluster, or DSC"
- now they want to measure how close two different clusters are. They simply use the minimum reachability distance between any 2 inner points in these clusters (basically find the distance between the closest points).
  - this is called "Density Separation of a Pair of Clusters", or DSPC

Now we define validity index of a single cluster, called `Vc(c[i])`:

- for every cluster:
  - minimum of the DSPC with every other cluster
  - basically the distance to the closest cluster
  - subtract the DSC inside our cluster
  - So this is
    - The distance to the closest cluster
    - minus the maximum distance inside our cluster between two neighbors
    - Since these are distances between two points always, these are comparable, smart
- divided by max of the above two values (instead of minus).

Negative value = bad cluster
Positive = good cluster

It creates a metric for each of these clusters.

Now the final index:

- we take a weighted sum of each `Vc(c[i])`. each of the individual indices are multiplied by `number of points in the cluster / number of noise points`. The noise points is also smart, more the noise, smaller the index gets.
- A big problem is the MST construction, that is going to be quite painful and slow. Can we directly use the MST used by hdbscan?
  - this is what the hdbscan package also does. Im concerned that ill need to implement it for the fast hdbscan lol
- ohk, these people are saying that zero code fallback thing should work. i need to mostly make my version compatible with hdbscan, we should have initial results soon (mostly the version mismatch with hdbscan is whats hurting our current construction).

# Patch collection

- currently, we collect per channel, we want to collect for everyone in a layer in one go, cheaper and simpler.

There are two places where we want the patches:

- clustering: this just needs a 2d vector. first dim is batch, second is flattened patch
  - it needs it per channel
- report generation: currently we generate patches for every attribution again.
  - this can also work per channel, it can also work at layer level
  - There are two bottlenecks here. one main bottleneck is that patches generation takes a long time for all inputs. im not sure why.
  - we would benefit from saving all the patches i think. but then, they need to be identifiable with the corresponding input, along with the index from where it originates. which is all very painful. but not that painful if we use tarwriter correctly.
  - all the final code needs is indices actually. we can go about this in a way that works for us yea. simply do the standard writer interface.
  - we write a shard for patches, each shard contains patches for that input example.
  - for channel level clustering, we'll add another piece of code which takes this shard and writes to a simple tar (flat without keys), this can be done for all channels in one go.

Ohk, this is done now. Now each shard for a layer contains all the channel's patches.
For clustering, ill simply do one more step where the data is duplicated (or actually just extract from the channel).

Ohk, activations stored all together is too big, 105GB for this layer. its best to store it channel wise only. and let the function take a batch of channels as input (channel-start and channel-end).
The other thing i can do is use S3 directly, which does make sense i feel.

Cool, first thing to do is store it channel wise.

Ohk, storing all activation patches of a layer takes a lot of space, it is taking 55GB on my computer to store all the attributions of a single layer.  
To create reports also, we'll need to download all of them so its quite painful.

I've made multiple mistakes at this point which are biting me back:

- Everything is around labels, thats wrong. No need for that.
- I should store everything in batches in one go only, no need for imagenet labels. this way, i can use much larger shards.
- although there is one problem, each layer has a different size of its attribution vectors. im currently handling one of the smallest ones. in this case, a shard can still get huge. but wait, it still wont get terribly huge for at least attributions. it can for patches. The core problem is that its hard for me to guess what the size of all patches would be. the other thing is, i might be running for too many inputs (do i need 50 per class? although lets forget that). Lets say i dont change a lot of stuff now, its too late anyways. i should just store the patches channel wise and call it a day. if we still end up with storage problems, we ll use s3 directly, or decrease the channel width size.

The whole drama is also for reports work. when we end up trying to get reports, we would want to query the channel data of an input. so it makes sense to actually make this input key wise instead of shard wise.

- We store as:
  - layer
    - channel
      - input-key
        - patches.pth

Although this would end up giving a lot of files. In this case, it would be easy for report generator to query this too.  
For training data, we simply read through all the pth files and stack the patches.

Ohk, tis getting more complicated than it should.
We want to keep disk usage in check. Before this, i was writing the patches for the whole layer per shard.
Now i can write it per channel, thats one way.

So, actually, does it matter what shard size i use? if i use input keys correctly?

Ohk, the problem is simple. if i do too many different files, its gonna get slower. the io is killing it.  
Storing everything together makes it much much faster, it takes 20 minutes for the whole layer. the only problem is disk usage.  
Note that we would need all of this data. but still we have a problem anyways.  
It would be useful to simply create batches per "channel range". again.

We could keep it fixed, like 32 or something.
Hopefully it does not gobble up all the disks. i would however, like the flexibility to choose the size of this.

Again, the problem is batch size. if it is small enough, there are too many writes.
Y

Ohk, patch storing is a pain. and the whole downstream report usage is hindering work.  
we were storing this before using simple pth files. ohk older code is doing shard wise for each channel, its not that bad

shard + channel + label is good enough combination for keeping data in check on disk. we take the slowness.
For now, we simply store per shard, with keys and indices

Aaah, i wont even change this now, just use s3 combo. we split it into training data mid epochs, and then upload and then delete the directories. easy.
Or I can just push to s3 directly for now.  
There are many options. Lets first try pushing directly to s3.

Or, the whole do on disk and push to s3 from time to time is also fine (it does use aws s3 syncs though).

This is only useful for report. which is actually quite painful lol.  
aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.

still very bad 😭

There is no win-win.

I either keep shard length by storing full attributions. Or i keep channel wise. or band wise xD
band wise is more complication, first do shard storing channel wise.

Obviously, this would also be quite small for some

ohk, each shard is maximum 50 images. tis bad. we do channel banding fuck it. Also, note that this can be done in one go easily. its simple enough. after every label epochs, we push to s3 and then clear the current dir.  
actually, first let me try simple channel wise shards.

Actually. we only need, channel wise data right? It does not need to be related to label in any way for now.  
We screw report generation, ill do it for multiple neurons at a time to make it easier.

So we create top level channel pushing shards, and we keep writing data to it.  
The only thing is finishing, should we just write to s3? how do we keep a

Ohk, the tar itslef is going to be huge compared to the data inside it. Its best to now use a different shard writer per channel, and put all the data there.  
Thats it. simple easy. we push labels inside the main loop. We cant keep it in sync with inputs right now.

Then, i cant do a whole lot of checkpointing i think. bummer fuck it.

Ohk, so we take a list of channels for a layer. for those channels, we use shard writer with some batch size.
so we have

```
patches
  layer_name
    channel
      000000.tar
      ...
...
```

The code is too spread out. im getting confused.
first thing, we need a class, passing the params all the time is painful

ohk, tis working. much faster now. next scene is doing hdbscan in a loop with auto best cluster selection.
This is also easy i hope.

Other than this, we need to now finish up putting everything in the lib, and make the notebook leaner. with one full test for everyone.
