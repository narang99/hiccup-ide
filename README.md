# Tool for convnet interpretation

The goal is to be as minimal as possible, with all the tools i use.  
The first major concern is to get started with the minimum number of tools I need, the amount I use for analysis right now atleast using jupyter notebooks. I think I can do pretty fast development.   

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

