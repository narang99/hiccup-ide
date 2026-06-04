# Remember the goal
- TRACE ONE FULL CIRCUIT. Illuminated. Then we go to the next question. Thats it.  

# Tool for convnet interpretation

The goal is to be as minimal as possible, with all the tools i use.  
The first major concern is to get started with the minimum number of tools I need, the amount I use for analysis right now atleast using jupyter notebooks. I think I can do pretty fast development.   

- I have a coordinate, it returns a graph to my code, we first create the backend for generating the graph and returning it

# Subgraph visualisation
- this is done. The full circuit is not giving the full story right now atleast.  
- i get higher activations from each slice, i got those. i would also need to see the conv kernel though
- We need more information
  - I have, conv 2d out -> input slice directly
  - for the input slice, i would want the kernel, and the output slice added too
  - so we need to create a composite node with contains those
  - the frontend shows layer node with these

- Each point has a receptive field. in the earlier layers, it seems all patterns are fully captured or something. idk if this is relevant though
- For our poi, we have multiple points picking up the diagonal left part of 4. They are picking the full breadth


# TODO

- in many places, the kernel is giving contribs to stuff caught which wasnt what it was catching
  - as an example, locally the patch is a vertical line, but we can see its part of a larger diagonal line. This seems like an important detail which keeps coming up
  - So a larger pattern also needs to be added, along with the actual value
  - We will have breaks in them (like the larger pattern is diagonal, but the kernel is not aligning perfectly, for now its best to add them as notes). Then analyse the data to add more context
- It is important to note the larger pattern it seems with convolutions

- I dont think i still have the most clear picture.
  - Why do some contribs look spurious? So many of them?
  - Some have extremely weird detection patterns, which are kind of easy to understand when you look at the trend across many inputs
  - Some have bad alignment
  - I feel spurious and bad alignment might be because the final layers have specific configurations which say "9". So anything that puts something there, no matter how spurious, adds a point for 9, kinda
    - currently the last layer is extremely simple so this is plausible
- Next questions of importance?
  - Check my assumption about final layer having the same kinda configuration for all inputs for a specific class
  - Check how each meaning looks across different inputs
- I might need to add differentiation to my contrib calculation too 
  - Local differentiation infact (some values have a higher impact on contrib, we might wanna have a formula for that)
- I still dont know how im going to compose my meanings lol

- Do a big bunch of 4s tomorrow. Ill upload 100 images of 4 only and do 10 of them atleast. I believe i might see better pattern clustering for POIs on single classes maybe
  - Also check if the final activations look similar for all 4s
- Somehow visualise what an output pixel is showing, this is going to be slightly harder
  - I could show the poi in earlier layer affecting it, then poi in the layer before it affecting it, basically show the maximal part of its receptive field to understand its story, might be useful


- Ive made graph making facilities in backend. lets now see what we can do about this.  
- Ive made a mistake also lol, i made the edges from parents to children
  -  But that is not working
- Well most of the graph making code from claude has been quite weird.  
  - Ill need to write this myself lol

- My requirements are also kinda more
  - While building the graph, i would want to from time to time, support filtering
  -            aaaaaaaaaaaaaaaaaaaaaaaaa
  - Im down. not sure what im even doing now ;_;
  - im not sure if this project will work out at all at this point.  

- what do i want?
  - I want to see the story of a single poi
  - Looking at the full story is great, but its very costly to render.
  - Note that a story can be decomposed into substories, and they can be looked at in isolation. So subgraphs in isolation are known to provide good results.  
- The first thing im going to do is continue what i thought. see a subgraph of a single poi if it is present in the graph (it is significant that is).   
  - Create the graph building algorithm first
  - Then do the UI tfms

- How to tfm?
  - Walk through nodes
  - If Conv2dSlice -> [Conv2dInput]
    - Return Conv2dPatch
    - Attach all parents of Conv2dInput to Conv2dPatch
  - now claude has done the thing right where a node turns to ui node
    - If it is Conv2dInput node -> it turns to Conv2dPatch
    - If it is Conv2dSlice node -> it also turns to Conv2dPatch
    - All ancestors to Conv2dSlice should point to Conv2dPatch (it would then have edges to its own)
- What is the other way?
  - Can i write a recursive function?

# Pre training some model and using embeddings for clustering
- Can I pretrain a model on internal activations or something?
  - Like make it predict the next thing or something?
- The main goal is to get clustering? or no? Actually, ill think a bit more about this later. I think i can manually look at the illuminated path with the correct tools and guess what is happening


# Progress
Good progress now
- full visualisation on main screen
  - global vs local color scaling
  - dark vs light mode 
  - slider to see the brighter pixels of each layer (top K pixels which make sum to X% [X taken from slider])
  - auto layouting
  - on click, open the slice wise contribs for conv layer blocks
- conv layer block slices view
  - slider again for seeing bigger contribs
  - on click -> go to kernel analysis view
- kernel analysis view
  - show weight, input, output, and saliency
  - on hover in saliency map, show the input receptive field
- graph pruning
  - workflow to prune saliency map
  - first prune last layer, recalculate for earlier layers
  - see the recalculated part, prune again, and so on
  - Gives a fully pruned graph
- Adding notes on each POI. Checking then in the final view how something was made
  - Labelling each POI (a cluster label for the top level pattern)
  - the actual meaning and extra notes
  - keep saving in backend
- landing page
- showing other input activations on the kernel 
- add support to mark a poi as spurious
- Different view for high activated poi examples for a kernel
- indicator for "done"


# Solution to pixels=0 being important and worth considering
Problem:
- A single activation can be simply a result of 9 input activations. Do we throw away most of the activations?
- Also im pretty sure that having a single pixel being red is also a signal, which im ignoring right now.  
  - or actually, it being zero is also a signal.
  - the signal is lost due in our method, it would make more sense to also use the gradients as auxiliary contribs. Calculate both types separately, rescale them individually, then add them
- For this, I need rescaling at each layer of raw contribs themselves. But this is fine i guess for now.   
- Sounds simply like changing the contrib calculating algorithm, will be done later.  




# Pruning

## Alg1
Current pruning strat:
- Start with last layer, threshold
  - backprop contribs again with these new values
- then do next layer, threshold
  - backprop contribs again with these new values
- repeat

- downstream layers are not affected in subsequent backprops

**NOTE** This can turn actually negative final contribs to positive contribs (since we are removing all negative contribs anyways after pruning)
For now, this is okay for circuit analysis in the end, i would need to see if there are better algos though (or i could keep the max reds, basically thresholding with reds also). Lets see, for now this is fine

# Coordinating a model
I've done this without thinking about it for long enough now. How do you identify unique positions in a model? I'm going to use discriminated unions for this.   
I also now want to correctly put the "params" in the model schema, its again hurting my ocd.  
I dont know where ill start my refactor but fuck it.  

- There is one thing i can do, simply not think too much about nesting and create flat structures.  
  - is this going to help though?
  - like the types are flat and dumb, should be okay i feel


- layer name
- layer type

## type:conv2d

We can have:
- inputs
  - In this case, we need in channel number, y, x
- output of a single kernel
  - out channel number, y, x
- slice
  - out channel number, in channel number, y, x


## type:relu
in channels = out channels = channels
- inputs and outputs both
  - channel number, y, x

## type:input layer
- channel number, y, x

For now, we are not really dealing with linear and flatten layers (This is only there for initial contrib calculation and will remain so for some time).  
It might be best to actually not keep these in the codebase only for now bc.  


# Model schema
Again, we have layer name, layer type
Dependening on layer type, you have different params
this is all the metadata related to a layer

## type:conv2d
- input channels
- output channels
- kernel size
- padding
- dilation
- stride

## type:relu
- channels

## type:input
- channels

# Where to analyse?
One problem with my analysis is that it simply uses horizontal slices as the data that i look at. But convolution is 3d, i can check a vertical slice also (along channels).  
For 1 kernel size convolutions, thats about the only thing i can actually look at (and its not even an image).  

This is a big problem for me lol. My analysis might be just wrong at its core.  
But we do see interesting patterns for some reason, but it might be a coincidence, maybe the patterns are also interesting along the vertical axis, i will need to check this.  
Or I could maybe do the analysis on both sides, or on the cube itself, and see if there is a correlation. This is useful I think, I would need to do statistical analysis on the cube, then on our interpretation slices, to see if one "kind" has won.  

The output of a conv layer is stacked in slices. The next layer is created by stacking individual outputs.  
There is no reason for a layer to actually not use the horizontal display at all.  

I would need a metric to see if a POI has a "horizontal" bias, and only analyse those for now.   
How can I do it?  

Cluster the POI using the 2d slice data for a specific slice.  
Cluster the POI using the 3d slice data for a specific slice.  

Find the similarity between the two generated set of clusters.  


## looking at the 3d view

A POI is in the end a bunch of pointwise multiplications. simple here.   
In my contrib framework, I analyse the highs of the pois and the lows. For a specific POI, I can look at the contribs of the individual pixels, note that this is simply pointwise mults scaled down   
In this case, i do want to pick the maximum highs and the maximum lows? To see where the strongest signal is coming from.  
If the strongest signals lay on a single or a bunch of slices, then there is logic to getting features spatially from the channel.  
I can technically find correlations in the input cube that come together for POIs. Now these input activations are definitively different.  
I see them side by side for a POI, along with their contributions.  
It would be useful to cluster the pointwise mults to see which kind of inputs come together.  
Now we simply think of each thing as an actual meaning carrying pixel. This kernel just carries the meaning forward.  

This essentially means that only the first layer is creating the first meaning, others are just carrying it forward in specific combinations.   
We want to know if different meanings come together.  

Is it correlated to the way the input slices look though?   

If we actually get stuff from a horizontal slice, then there is a spatial property of that channel being extracted.  
Across channels, lets say two pixels are firing together almost always and have a high contribution. Then the kernel is correlating these two (technically since I follow meanings, I can kinda see if two meanings are coming together in a kernel).  
There is a chance that the first layers are interested in finding meaning spatially in the same channel (some of them atleast, the ones with color wont).  

First question, how do you identify them? 

### cluster the slice output and the cube output separately
- For a single POI, across examples
  - cluster the slice POI activations
  - And the total POI activations
  - And see if the two clusterings are similar. I would need a score for this.  


### What other way?
- For a single POI
  - cluster the whole thing, we already know what high contrib ones are present in that POI
  - Abi if a single image is causing that thing to come, then we should see repetation of only one slice in a single cluster, all the other slices would be less interested.  
  - This seems like a decomposition question, if i decompose my pointwise data for a poi, and the decomposition comes up with 
  - For a cluster, i essentially want to find the set of points which are connected for this POI.  

Consider a single cube.  
- My current contrib method basically relies on finding the high values.  
  - Woudl the extra data be just noise?  
  - The pointwise mults wont be noise. i think i can cluster the pointwise mults.  
  - Now the problem is that of dimensionality. Instead of 9, ive got 72 dimensions on one kernel.  
  - Again, im quite bad at this sort of clustering.  
- So we can do kernel analysis by relying on getting good clustering results only at this point.  
  - This is now paramount lol.  
  - The cluster mean might be also useful here (hm hmm hmmmmmm)

The point of assigning important is important lol, but i will think of a better algorithm later :).  
The best way would be to first do a summary analysis of the many existing methods and see which one seems more useful. I currently have more hopes from the using mean as baseline in my method thing.  

PCA seems to be useful for dimensionality reduction, i had unnecessarily ignored it before. It should tell me which variables are interesting in our 3d plot.  
Autoencoders or something might be interesting in this, i need to thnk of a good loss function. I want the remaining points to have maximal variance, that is remove the points which are not useful hmmmmm.  

I have it wrong again, this is simply similar to autoencoders i feel. Lite max. So it is only useful in decreasing the dimension for clustering.  
I can basically cluster the whole thing, and see if there is a correlation in a single cluster, how do we find this correation?  A pearson correlation matrix might be useful here. Simple.  Lets start then.  


A lot of what im doing is rediscovering at this point, Distill has done a lot of what Im doing already, and more it seems.  
I need to find a simpler problem. A more restricted one to start with.  

The first thing that might be useful is to test if we see better clustering behavior inside a kernel for higher contribs.  
This is simply a comparative study right now. Lets do this with our network.  

Well, i see shit clustering results for the whole kernel actually, i see good only for some points.  
Lets see them separately ;_;

I dont have anything to show now, this is quite bad. Im not sure what ill do.   
The realisation that the kernel should not be decomposed to slice level stuff is hard on me lol.   
distill.pub has done a lot of work already, should i just start from where the last article finished?  

What are my options now?
- Find pois in network where i can reliably cluster an assign labels during runtime, in a way tracing the nature of a circuit automatically.  
  - How do i cluster though? There is a lot of noise.   
  - Simple pointwise mults have a lot of noise, what we are looking for is a subspace with the same pattern coming up over and over.  
  - my initial thought was clustering after dimensionality reduction (but there is a lot of noise already).  
    - now now, there are ridiculous number of activations lighting up, why? is it because i scaled it? maybe, maybe. lets do scaling after mult.  
    - removing scaling helps a bit yes. Now what though.  Although scaling seems like the wrong idea, lets do without
- Im doing hokum right now, what to do hmpf.  


So i got good clustering because i was specifically looking at that spatial slice's high activation.   
I can just simply only consider the higher contribs of the input activations.  
Since this is what is anyways what LRP is doing, lets do that and see if a pattern emerges.  

On a separate note, an interesting experiment is to try to generate an input patch from a given output pixel using a small network, we have the labelled data (input activation and the output pixel). This basically says whether the meaning is carried forward in just one pixel or not. Then we can successively add pixels to it to see if reconstruction accuracy gets better. A patch of 6 pixel's maximal activations might be simply generated using 2 pixels (like 2 pixels which form a diagonal. or 3 the left part and the right, who knows). This gives a measure of how densely information is packed, and where it is packed. This would be an interesting exercise.  
Assuming I'm able to crack which examples are useful (for now im assuming it is in high contrib path).   


A single kernel can extract multiple features. Many times, for the same POI too.  
In this case, how do i find correlation between components coming together? Should i just do a pearson correlation matrix? Of highs coming together?  
Now different parts of the kernel would have different patterns being recognised.   

What can I do?
- brute force:  I can try clustering with a random subset each time, hoping to hit jackpot


Dict learning is working reasonably well, i need more data though.  

# Dict learning

It is working beautifully on the 2d case, ill have to test it on the 3d case. One problem is me not being able to get a good `n_components` reasonably automatically, ive tried some stuff:
- Model error using a small model with a penalty on the size of error
  - the hope was that the loss would elbow out fast, but error constrained to 5% the size of the reference vector is not giving me like amazing results lol
  - It might be useful to just give claude the data and ask it to give us a good metric lol
- There is still a problem of me wanting to recognize whether we are actually able to model the input without looking at it.  
  - The error component did help in this, we want the error to not use its budge consistently, when it does not, the model might be converging? 
  - Its still not the best way i think, too much hokum, need to get something better
  - I need to see some bad examples for it too maybe?
- component stability is another thing that can be useful
  - basically, using different seeds, we should get the same kernels.  
- For the 2d case, loss never went below 1 btw (root MSE).   


## Measuring what is good reconstruction

- For now visual inspection is fine. We just skim through reconstructed patches to see everything tracks. For now, making the vector sparse and tracking loss has made loss less opaque, its not that bad.  
- Now for interpretability. What do the basis mean?
  - Each basis is an arbitrary combination of multiple basis. 
  - Find the basis most similar to current basis, project it on an input (the patch where it is actually not bad) and see what difference exists. (we can generate the patches too, although, i can simply draw which portions both are detecting in inputs everytime, that would be quite useful).    
  - so instead of rects, we highlight the stuff we reconstructed (highlight the things which matter).  
  - We have the positions, we have the numbers
  - I can try doing this just for one basis vector first.  
  - With overlapping dict representations, sparseness is more important.  

I now have a long list of checklist, which i dont really understand myself. need to get started with one of them.  
- The first and the most interesting is seeing if patterns are coherent.
  - That is, i can find what meaning came out
  - The easiest is to first find an input where a single component fired alone.  
    - In this case, i would like to see the pattern the component caught
    - basically make rects around all the vector components where the component was non-zero

Its actually quite clear what patterns are going through now.  
They are entangled, but they are categorized in a simpler manner.  
- Lets now scale this to all the spatial POIs for a kernel. I would ignore the boundaries for now? Actually no, it should be fine.  
  - There is a chance that the number of points per POI can be different, we want to pick a correct number of random samples (i am concerned about the model overfitting for each)
  - The problem is, someone might have genuinely more samples. Now we would need to check the variance in data.  
    - low variance? lower number of samples, will need to check this empirically.  
  - writing coherently about each experiment is useful, i think i should start that.  
  - kinda like blogging, it keeps me sane, helps me remember the results and what im doing.  
  - its just, a lot of pain lol.   
    - There would be a lot ill have to write about. But for now, lets start with the dict learning things.  


# Reconstructing input from output pixels
- What if, the set of output pixels required for reconstructing an input pixel from a kernel is dependent on the next layer's kernel? That is, the next layer's kernel might have some component which aligns with a specific representation of the previous kernel?
  - Like a diagonal filter in front of a filter which does diagonal+vertial+horizontal
  - It is recognising one of the patterns
  - In a patch of the next kernel which passes through a set of features, the task then becomes to see if which patterns from the older kernel can be regenerated from the patch which was passed through. Might be useful.  


# Prime numbers in 2^x system

2 1
4 2
8 3
16 4
32 5
64 6
128 7
256 8
1024 9

3*3 = 9 
this is (2^3)^3 (the operation is done twice, so the log should be 2)
2^3^2 = 6 (not prime)  

So for the exponent space too, there would be some natural log version which says that exponents occur this many times.  
In this space, `e` is 2^{our e}. log wrt this. actually since its proportional, it does not matter.  

prime numbers in this space, what are they in our space?   

3*3 = 9
log<3>(9) = 2  (number of times 3 is used)

but this is to base 3, and not to base

2 4 8 32 128


it makes weird sense, there seems to be a connection



hmm so if we use multiplication as the core operation with lets say 2 as the base, instead of addition, to make a number system, then log<that number system> is our number system



then we take a log again, we get prime numbers hmmm.

so the operation that brings down from a system defined by multiplication, when used, also brings down addition to the core operators of multiplication in the addition numner system state.



so there should be an analogue in multiplication. there would be an operator acts like how multiply works in ours (its the opeartor "raised to" in addition spacE?)

that would be basically counting the number of times the exponentiation operator in our space is used. 



log2 of those primes, is pretty much our primes.

now ln<our numbers> then should be inside a space where addition is defined. 

what operation when combined becomes addition. 


This is a problem of multiplication being a simple copmosition of addition.  


fastai, part 1 is done. im also quite fluent with most of the stuff now since ive been working on interpretability.  
The other thing is that the algorithm we have is simply giving us disjointness over ICA, a bit better than ICA, nothing else. It might be useful to proving that disjointness might be the thing for our kernel?  
If ICA and disjoint equations give similar results, its a useful direction.   




see how the samples change when we increase n_components actually.  
- with cycled weight loss also (not cycled but like starting from rand rows)
- check if we can make sigma eps learnable with the full MSE loss, to see if noise can be catered to like this.  


Good, we notice components successively decomposing.  
- The thing is, its hard to know which components have been decomposed from what.  
- If I know what the decompositions are, I can find the MSE change in those dims
- Actually, I can find MSE change per dim, and print the top, 4 or something, and see if it is the same the decomposed dims.  
  - A dim whose loss changes significantly has definitely decomposed