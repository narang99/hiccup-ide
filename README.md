# Tool for convnet interpretation

The goal is to be as minimal as possible, with all the tools i use.  
The first major concern is to get started with the minimum number of tools I need, the amount I use for analysis right now atleast using jupyter notebooks. I think I can do pretty fast development.   

# TODO



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