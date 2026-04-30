# Tool for convnet interpretation

The goal is to be as minimal as possible, with all the tools i use.  
The first major concern is to get started with the minimum number of tools I need, the amount I use for analysis right now atleast using jupyter notebooks. I think I can do pretty fast development.   


Minimal features needed:
- Visualising activations for convolution operator ✅
  - full activation of the whole 3D kernel
  - slice wise activation, input and output
- Visualise convolution operator kernel, only slice-wise
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