I was assuming that the number of features caught by a channel would be limited, but it doesn't look like it is true (or atleast for the whole 3d channel, i have yet to see evidence on slice level).  
I see three different features already in 4 inputs.
- A long diagonal line (above a horizontal line?) in left side
- A long diagonal line (above a horizontal line?) in right side
- Absence of the curve joining the two lines.   

- This already breaks my assumption that a single channel gives similar results across inputs (although, there is still symmetry, i need to look at more data).  
  - Look at 12 inputs i guess


- si1, 4qo, 0ve, ayd, otm, hnm, lud, c2z, wxq, 4pc, 9ci

- interesting, [c2z](http://localhost:5173/models/simple_mnist_v1/4-C2z/v1/) did not detect ANYTHING. it has vertical lines, this kernel did not detect them.   (very interesting)
- wxq however worked, it has straight line though lol
  - same for 4pc


- There are a set of behaviors that i see in this kernel, i need to document them all i guess.  
  - I should also look at slice level behavior i feel and see if that gives us something, although the final output does not see slice level behavior at all.  
  - For mass checking the graphs, i would need to programmatically find which ones have one slice's high pois and open their graphs, should be okay
  - For this i first need to prune a bunch of graphs, i would need indicators in the landing page though


- OutChannel_12, InChannel_6, layers.2
  - This is the negative channel. Lets first look at all the inputs which are there in this channel maybe?
  - I would need to see through the graphs of inputs where this was lit up
  - Verify if this is always trying to find whether there is "empty space" between the two four dande


```python
In [6]: def has_positive(d):
   ...:     for r in d:
   ...:         for c in r:
   ...:             if c > 0:
   ...:                 return True
   ...:     return False
   ...: 

In [7]: has_positive(d)
Out[7]: True

In [8]: wsms = WorkSaliencyMap.objects.filter(coordinate="layers.2.out_12.in_6")

In [9]: wsms.count()
Out[9]: 101

In [10]: res = [wsm for wsm in wsms if has_positive(wsm.data)]

In [11]: 

In [11]: len(res)
Out[11]: 89


TEMPLATE = http://localhost:5173/models/simple_mnist_v1/{INPUT_ALIAS}/{WORK_ALIAS}/ui-graph/?type=Conv2dOutputCoordinate&layer_name=layers.2&channel=12&y=2&x=3&layer_type=conv2d&coordinate_type=output
```

So we have 101 total inputs, 89 of these have a high activation in this channel.  
- Later I'll have to also see if this channel has this behavior across others which want to find empty space.  


I'll also need the coordinates, we ll focus on the middle ones first, so 0 < y < 6, 0 < x < 6 (not the first and the last coord)


It seems, (4,2), when it has a contrib > 0, it is following a pattern. It seems to have an empty space, between two 4s.  
So where the pixel is coming, is also important.  
And what value is also important i think. In my case, there was no need to cluster the outputs, the network seems to get positive contribs only when we have (4,2) being a wedge.  
It would be useful to see for all 4s now, instead of just the ones i have.  
I need to verify this, can i do it in notebook? that would be very painful.  

So, there is another thing. A pixel location lighting up, can have a meaning. The network makes a composite meaning there.  
The first symmetry I've found is, the location [layers.2, out=12, in=6, (2,4)] has a non zero contrib only when the kernel has detected empty space between 2 vertical bars for the images of 4s.   
- I need to analyse this more.  


Instead of starting from beginning and saying why the answer is "something", start backwards and start from "something" and prove the possible inputs, that can give that something.   
I should be able to cluster the outputs of a kernel, and somehow reason about the inputs basically.  
But how much does clustering need to see to be able to cluster?  Should it see the layout? Should it see only that one pixel? should it see a patch? I need to first find that out myself.  


So im back to first splitting at slice, picking the top, then moving on.  
So you split the pixel you want at slices, but then dont prune, you follow everyone. for conv, you still have to follow all pixels

# detour: better path light

- problem: 0s at high negative kernel weight is also information and we would like to see them.  
The current algorithm does not account for these things. The current algorithm does:
- Maximum positive values get higher weight

The problem is that for a kernel, where the positive value comes is also important.  
Or where the negative value comes.  

When there is a high negative component in the weight, the corresponding input activation being empty is important.  
- Now, it can be empty because the feature before it was killed, or it can mean something. 
  - That can only be seen by looking at the pixel itself. So the first thing is assigning something to it
  - The problem with ReLU is that it kills all negatives, so it makes sense to assign some "unit" weight to all negatives (it does not matter how negative something is, its the same to the next layer).  
  - In any case, we would like to investigate why something is 0 generally when it being positive hurts.  
  - It is the absence of positive that is helping the activation be high.  

Ohk, we have a better function now, its time to use test it.  

For higher things, the pointwise mult is the component, w*x   
For lower activations for higher reds,  w*x will be closer to 0 (lower input activation).  
So we do (w * (x - mean-of-that-activation))  (its contribution would be).   

For 0, the contribution then becomes positive for negative weight, else it becomes negative, which is lesser than w*x, so for positive inputs near 0 and negaative kernels, this is fine.  

Although this is also becoming kinda complicated lol.  I'll need to do a literature survey finally lol.  

Lets first do the deeplift thing, where we use reference / mean. Assuming that a kernel gives near 0 inputs everywhere, we can assign scores to the deviation from the mean.  


| x |  w | mean | formula |
|---|----|----|----|
| positive near 0 | positive | haha | hehe |
