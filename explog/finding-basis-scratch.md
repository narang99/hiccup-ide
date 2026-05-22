What is the final vision? I would like each input pixel that came to a kernel to represent a high contrib patch in the input space (yes LRP is problematic and a liar, later, I have an approach to work with that).      

Multiple input activations which form a group of atomic patterns are passed through together. We would like to see the input patches which were passed through in that group. Its one operation that the kernel is doing i think. The hypothesis is proved if we see similar patterns in the majority of our examples.  

If, a pixel has the SAME labels across examples, then it should actually point to similar patterns in the image.  
The labels form a grpah, we would need find the most similar graphs among a group of graphs for a given meaning. This works if we are able to disentangle it though. So, this would come later.  

What about the current work? I don't have disentangled inputs, this is an incomplete tagging system, but it still is a tagging system (technically, we tag the combination of labels, which is what the graph would have in the end too). I think, the hypothesis does not change, for the same graph of labels, we should have similar patterns in the input.  

This is essentially trying to say "if the circuit is the same, we damn sure its the same pattern". Different circuits can differ in graph, the individual labels assigned at each node are the biggest variance.   


So, I might actually be able to say, these two are the same circuits now.  With what I have atleast.  
What value would disentanglement give? Im not sure now, after writing this.  

Technically, you can just look at the saliency map of each input pixel, the saliency map of each output pixel, and see what is happening.  
This is easy, known, other than the fact that saliency maps can lie, its a good starting point though.  

With our tagging system, we are basically marking each node in the pathway.   

For a given disentangled atom representation, i can technically just see how a single group combines in my picture, instead of the whole group, this tells us the separate kinds of operations a convolutional kernel allows.  
The major problem is that the local pattern itself is quite hard to interpret.  
Although it is simply a linear transform, it still is a huge linear transform, in 3-d. So i can basically take a look at all the elements separately.  


Now, we are in a way, explaining the graph. At each point, explaining what is happening. I need a consistent story there, which i dont have right now.  


# Problems

I picked contribs from all pixels and otsu-thresholded them. But its not working.  
The weights are zero-ing out. We get zero answers. We also get some answers.  

Time to analyse by looking at the worst losses   