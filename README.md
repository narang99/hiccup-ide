# Tool for convnet interpretation

The goal is to be as minimal as possible, with all the tools i use.  
The first major concern is to get started with the minimum number of tools I need, the amount I use for analysis right now atleast using jupyter notebooks. I think I can do pretty fast development.   


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

Next steps:
- model alias, work alias, input alias should not be hard-coded
  - change in BE main code
  - change in FE (urls)
  - change in load_data
    - figure out the format of load_data to directly take a model and a set of inputs, for which it can load data (type = SimpleMNIST works for setting model)
    - each input needs a label too
    - a folder
      - model.pt
      - inputs
        - ip1.pt
        - ip2.pt
      - meta.json
    - meta.json contains each inputs label

- I would also like a view to show all the inputs of the kernel with high contribs 
  - this would need some work in load data only right now (im not going to put a lot of interactivity in this)
  - basically load all inputs, with their contribs
    - then in kernel analysis view, we show one more layer with high contrib inputs (with patch drawn). (we dont need to show the output for now or anything)


# old roadmap
Minimal features needed:
- Visualising activations for convolution operator ✅
  - full activation of the whole 3D kernel
  - slice wise activation, input and output
- Visualise convolution operator kernel, only slice-wise ✅
- Visualise full network activations ✅
- Visualise contribs ✅
  - Backend work left
  - Frontend work should be quite easy comparatively (almost no changes i think)
  - first generate the contribs as json, but in a format that the frontend understands
- Contrib minimisation with strategies at each step
  - Basically user picks the highest contribs, the frontend notes them down in the backend ✅
  - Each prune operation results in a new graph of usefulness 
  - We keep pruning by each layer to find the final pruned graph
- Analysis of each POI
  - Each poi in a view opens up another which digs a bit deeper. and allows for thresholding too
  - Clicking on a POI shows the input patch along with the whole data
  - We 'note down' what the POI means locally
  - This noting down associates the analysis to the POI and to the kernel itself (the kernel meaning is attached to the network)
  - I would also want other images from other activations for this

Finally we would like to see the meaning (meaning can be propagated at each step).   

- Now we want to re calculate the contribs given an older layers contribs.  
  - This requires me to add the package to the backend
  - i now need to come up with a good workflow for this.  
- Contrib calculation requires the model, requires the definition of the model too i think
  - model definition json calc requires actual model definition for torch.fx i think (i might be wrong)
- but we do need to run the model for calculation of contribs.  
- in the most basic thing, i want to call the contrib calc with the inputs and a new contrib vector
  - this is the easiest workflow



## POI analysis workflow
- We select the POI from the main graph layer view. 
- opens a new view with slice wise contribs, we threshold again, save them
- Then select individual slice


- next step: open slice contribs view only

## Easy cutting
- Cut in the last layer, then go to next.
- We keep a pruning session, simple
- You prune and then fix the graph.  
- After a graph is pruned, the user can poll the pruned session itself

- While cutting you cant go back, you can only go forward.  
  - I can later add esa ke if you go back, the progress in the mids is lost
- We need a table to manage a Pruned Graph. 
  - One table for all saliency maps


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



# Workflow

- Generally working through this requires following a specific workflow.  
- You cut the graph, find subgraph of interest. go through all of the points in the subgraph of interest, and annotate them
- So the first part is cutting the graph. 
  - Go layer by layer
  - Find threshold for cutting
  - Save it
- This then requires a saliency map view specifically for cutting the graph
  - This can be the main view right now (I dont really care a lot about seeing activations)
- Basically graph cutting does require us to have a different view i think (or is it simply the panel changes?). Who knows. I'll think about this later
  - For the top level, creating the graph is the same, no matter if you want a cursory view, or if you want a cutted view.  
  - Stuff is just dimmed out. Later, we would add support to simply have a sort of swift reporting which allows me to quickly go through all POIs in the layer and assign them meaning
- Once this is done, we would like to see the illuminated graph. This is vague in my head, each pixel has a graph illumination, which we are interested in.  
- still, the first thing is to create a cutting workflow. for now we only do simple thresholding.  
- The biggest problem with doing something like this is "defining" what the pattern is. The pattern can only be inferred across invocations of POIs and clustering them i think.  This is the last step.  This is the step which has given me problems until now. 
  - The best way is to simply look at all the POIs across examples for a kernel to categorize the patterns myself. Easy.   
- So, cut the graph. then swift report. then illuminate. It would be useful to simply swift report for a single circuit for now.  We pick the polysematic neurons which the distill team is confused about.  
  - a single circuit is good enough.  
- This seems like the easiest option.  

## Cutting view

- Add each layer to its own subflow
- We dont need edges now
- Show saliency maps
- We now change the main view to simply a saliency map view for now

## AI handicap

- I get handicapped by token usage. I also dont understand the FE code that well, idm using AI majorly for CSS, but i do need to now understand the code better and do it in a more engineered fashion
  - First step is to create a separate component for creating the subgraph of convolution layer
  - The general workflow is passing a static graph to react flow, but once it is done, we can't "refresh" that easily i think
- For now, im going to use react context only, the zustand store seems complicated lol
  - I dont see a need for zustant right now, lets do context yes
- Ohk, so the main problem is that nodes are siblings. There is literally no way for me to set a heirarchy, i can only say: put this node's position relative to this parent nodes position. This sucks.  
  - The amount of re renders on global context would be crazy
  - I would now need a global store. but i dont want re triggering happening for everyone too.  

- Creating subflows is quite painful it seems. what should we do.  
  - We could create a simple node besides the of a layer for analysing them.  

- Tis shit is more complicated than i thought it would be. claude is basically doing random shit now. Its like changing 1 CSS line and having everything go to shit.   

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