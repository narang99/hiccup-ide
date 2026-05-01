# Tool for convnet interpretation

The goal is to be as minimal as possible, with all the tools i use.  
The first major concern is to get started with the minimum number of tools I need, the amount I use for analysis right now atleast using jupyter notebooks. I think I can do pretty fast development.   


Minimal features needed:
- Visualising activations for convolution operator ✅
  - full activation of the whole 3D kernel
  - slice wise activation, input and output
- Visualise convolution operator kernel, only slice-wise ✅
- Visualise full network activations ✅
- Visualise contribs 🏗️
  - Backend work left
  - Frontend work should be quite easy comparatively (almost no changes i think)
  - first generate the contribs as json, but in a format that the frontend understands
- Contrib minimisation with strategies at each step
  - We basically want a graph pruning strategy
  - Basically user picks the highest contribs, the frontend notes them down in the backend
  - Each prune operation results in a new graph of usefulness
  - We keep pruning by each layer to find the final pruned graph
- Analysis of each POI
  - Clicking on a POI shows the input patch along with the whole data
  - We 'note down' what the POI means locally
  - This noting down associates the analysis to the POI and to the kernel itself (the kernel meaning is attached to the network)
  - I would also want other images from other activations for this

Finally we would like to see the meaning (meaning can be propagated at each step).   



## Cutting the graph

- Graph pruning can be straightforward. I need the full POI view to actually prune the graph.    
- The biggest blocker now is not having a backend.  

- In any case, you keep a layer view of the saliency map. in that view, you can basically get the contribs using a global limits in color map
- You get a graph of contribs, play with it, find the points of interest. Let the frontend then push those point of interests to backend
- Then with these POIs, you have a subet of POIs in the previous layers, you can again go through and get the ones you are only interested in, and we should be good.   
  - This aggressively prunes the graph.   
  - We would have inputs to POIs which won't be fully assigned a meaning and that is okay for now.  
  - Technically, I should prune the graph for each POI, but that might be excessive  


- Until now my workflow is
  - Find all significant POI, find why they are POI, note down what is happening basically.  
- We want to change this workflow. We first take a pruned up graph
  - This will start by first pruning all POIs in the last conv layer
  - Then for each POI, we find the most significant POIs of it, its a kinda of depth first search.  
  - We create a tree, and at point in the tree, we cut branches which are not useful.  
  - Then we merge them to find the most useful graph.  

- Good now that the saliency map view is done, it is time to think about how i can go about creating a graph out of my contributions.  
  - This graph is important. Its not technically just contributions, its the graph of all input activations which affect an output activation (its the whole model graph actually, but instead of operations, we track each activation)
  - This might be huge in terms of size i guess, ill have to think about what i can do. but its okay. lets first try to generate the graph


### different methods of pruning

#### Alg1
- Find the POIs in the last layer
  - For each POI, keep the activation shares in its inputs, only consider those. 
  - Repeat

#### Alg2
- For each layer, find the all the interesting POIs
  - Create the graph containing all POIs connected to these POIs


In essence, i do need to build a graph out of my contribs. So lets have claude do that first
For now, we only do part one of this.  

I would need full illumination later though i think.   


#### Solution to pixels=0 being important and worth considering
Problem:
- A single activation can be simply a result of 9 input activations. Do we throw away most of the activations?
- Also im pretty sure that having a single pixel being red is also a signal, which im ignoring right now.  
  - or actually, it being zero is also a signal.
  - the signal is lost due in our method, it would make more sense to also use the gradients as auxiliary contribs. Calculate both types separately, rescale them individually, then add them
- For this, I need rescaling at each layer of raw contribs themselves. But this is fine i guess for now.   
- Sounds simply like changing the contrib calculating algorithm, will be done later.  


