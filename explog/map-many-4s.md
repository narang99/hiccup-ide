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