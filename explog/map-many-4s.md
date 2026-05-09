I need to map a lot of 4s. What are the things I should look out for?

- Do the last layer contribs look the same? Are there patterns?
- Do all kernels behave the same? I believe I might be able to get the specific pattern that a kernel is looking for if I do a bunch of different inputs and compare. I would need a way to compare later.  


I've also just loaded 4s only in the backend now, lets see how highly activated pois look now.  



There are many reasons why I dont understand how the network is working:
- Same kernel having inconsistent behavior across inputs. It is useful now to see pattern across inputs to see what the kernel is doing
- Why something is POI? is it the spatial position? Is it that others are killed?


- About spurious activations
  - It seems the final activations that result in high contribs do follow a larger pattern. Like finding points across a diagonal.
  - Sometimes the kernel does not actually find good alignment for actual diagonal detection, but by luck (gradient descent though i guess) it is able to find alignment.
    - This seems a recurring theme in edge detectors, they are not perfect edge detectors, they luckily find varieties. A `/` detector flows across to a `|` detector sometimes (it gives activations for both) to cover a wide range of patterns. 


# Map 5 fours
- Today I have to map 5 4s. Then I decide how to analyse the data.   
- If you forget about your thought thread, look at the notes and questions above.  


# Thoughts

- We see many spurious / lucky activations coming when alignment is not good. 
  - Say a kernel detects an L and gives a weaker activation for just a horizontal line
  - I see many times that when it does not align with the L intersection, it gives a weaker activation just beside the intersection L
  - What is remarkable is that the network considers this a high contribution

## Visualization of the graph
- one 4 is done, can i see the path somehow in the ui?
  - The path for a single POI gives its meaning
  - It would be good to see it visually first
  - But kya dekhu?


- Conv out pixel
  - See all the slices of interest in the back
  - Then for each slice, show the patches in input
- Then for each patch, we would want to see how the input came for that one
  - This might include the ones which are accounted themselves, or it might not
  - I would need to still see the POIs which are accounted

- For a patch, we show all the inputs which had high contribs. 
  - This can be a relu behind. for now its a conv only (relu is pass through)
  - For each input pixel, we again have the slice thing


So we have:
- For each output pixel
  - The input slices of interest (with only that pixel on)
  - These form edges to this input slice
  - Then for each of these, we have one input slice with the patches highlighted
  - This is one group
- Each group has ancestors on their own things
  - For the scale that i have right now, this is fine
- I might not need the labels anyways (though they do have value in the end, for final graph building [they allow us to do the graph analysis automatically, the thing that im doing with raw eyes right now]).  
  - But this view should be interesting.  
  - Lets try it out

- How can i do this though?
  - Having a graph here is useful (instead of manually finding its receptive field i guess)
  - This is really just a graph visualisation technically
- Do i first build the graph? or do i start visualisation by recursively doing it in the frontend?
  - hmmmmm
  - I can recursively build the graph quite easily in the frontned. in any case, before drawing it, we would want to actually see how we can build it
- I need to come up with an algorithm







```
[show patch] (L2 o12 i1 [2,3])  (L2 o12 i3 [2,3])  (L2 o12 i5 [2,3])  (L2 o12 i7 [2,3]),
[show one pixel highlighted]  (L2 o12 [2, 3])
```

Each from the patch is an output pixel itself.  
- So i start with one output pixel (separate node)
- Then i show all the slices of it, the patches (these are separate nodes in the graph)
- For the 9 pixels, if any are in high contribs in the input slice, we show it (as output pixel itself)
  - This is not the best because currently i only show using LRP, i might need to add support for the differentiation scores also later, to see the full story


So building the graph
- Pick one output pixel, its a node of type "output-channel", with coordinate "(x,y)", layer_type "Conv2d"
- Find nodes of type "input-output-channel", coordinate "(x,y)", layer type "Conv2d" -> node of type input-output-channel
  - These now have a receptive field which tells us the patch of interest in the back
  - For each coordinate in the receptive field, we find if it is a poi, if yes, we create a new node for it as layer-type=relu node-type=output-channel node "(x,y)", and set it as a parent of this node
- for relu, its simply the input activation highlighted for "(x,y)", so now the parent is (x,y), L0, type=output-channel, then the story repeats
