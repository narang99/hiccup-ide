Good paper, although the math was slightly intimidating at first, after a bit of reading it is becoming clearer.  

They basically have two sets of models, the first is the basis vector `phi`, and the second is the sparse code vector `a`.  
They make a reasonable probability distribution argument about them.  
That would require some writing lol.  

I also now understand why there is a length constraint on the `phi` matrix.  


A lot of math is now becoming more intuitive. They are converting intuition to math distributions.  

The sparse coefficients basically should be generally 0, so they say that the distrbiution of these coefficients can be chosen from laplace/cauchy (these have 1 at 0, and go down sharply to 0 on either side).  
For the error term, everyone assumes gaussian.  
This error term gives the probability distribution of `y | inputs` to be gaussian (cuz inputs are fixed).  
So you bascially just descend on minimising this error distribution, leading to MSE many times.  

It would be fun to write my problem in this way. The first thing is maximum likelihood estimation.  
Lets say we have a gaussian curve, with mu,sigma. The probability distribution function gives the probability of observing samples `y[i]` given sigma and mu.   
We have the samples though, we we want to find the correct sigma and mu, which fit them.  
You basically want to say, how likely is some parameter `mu[i]`  and `sigma[i]` given dataset `y[i]`.  
This is `P(mu,sigma | y)` instead of `P(y|mu,sigma)`. This comes into bayesian rules now.  

```
P(A | B)P(B) = P(B | A)P(A)
```

`P(params | y)` = `P(y | params) P(params) / P(y)`.   
Now im not sure what to do xD.  

What is the `P(params)`?  What is P(y).  

# sparseness prior

L1 on coefficients directly, not very bad. problem is that L1 on direct codes pushes the coefficients down too much in value too.  
```
finetune epoch 4800 | recon_loss 2039.5404 code_loss 2040.9456
finetune epoch 5000 | recon_loss 1500.0206 code_loss 2109.8135
finetune epoch 5200 | recon_loss 1735.0452 code_loss 2122.0308
finetune epoch 5400 | recon_loss 1522.7332 code_loss 1954.7939
finetune epoch 5600 | recon_loss 1475.0591 code_loss 1969.1360
finetune epoch 5800 | recon_loss 1665.5251 code_loss 1994.0542
finetune epoch 6000 | recon_loss 1476.2241 code_loss 2106.4749
finetune epoch 6200 | recon_loss 1708.2336 code_loss 2045.3422
finetune epoch 6400 | recon_loss 1529.5114 code_loss 1899.9451
finetune epoch 6600 | recon_loss 1967.4382 code_loss 2023.3951
finetune epoch 6800 | recon_loss 1796.7842 code_loss 1921.0477
finetune epoch 7000 | recon_loss 1467.7522 code_loss 1913.7998
finetune epoch 7200 | recon_loss 1617.0540 code_loss 2138.6177
finetune epoch 7400 | recon_loss 3357.2083 code_loss 2272.3069
finetune epoch 7600 | recon_loss 2017.4265 code_loss 1944.9791
finetune epoch 7800 | recon_loss 1489.6775 code_loss 1900.6624
finetune epoch 8000 | recon_loss 1955.9692 code_loss 1881.5790
finetune epoch 8200 | recon_loss 1902.6034 code_loss 1862.1335
finetune epoch 8400 | recon_loss 1461.6171 code_loss 1827.7040
finetune epoch 8600 | recon_loss 1280.8079 code_loss 1861.6217
finetune epoch 8800 | recon_loss 1555.8617 code_loss 1863.3916
finetune epoch 9000 | recon_loss 1845.7153 code_loss 1893.6752
finetune epoch 9200 | recon_loss 1459.5586 code_loss 1825.2336
finetune epoch 9400 | recon_loss 1661.3124 code_loss 1845.3879
finetune epoch 9600 | recon_loss 1881.1093 code_loss 2140.8853
finetune epoch 9800 | recon_loss 1434.2057 code_loss 1754.8931
```

Interesting, cauchy prior on encoder is making it harder to do correct reconstruction.   
The codes are also not 0.  

Cauchy on encoder is in general weird.  
its log likelihood is log(1+(x/sigma)**2).  

for higher values in the loss, you would need a smaller sigma. that also corresponds to sparseness theoretically in cauchy.  
that however then, makes the gradients extremely small. (the grad would be 1 / (term inside log, whcih is huge)).    
Might be worth writing about.  

It might be the reason why my other prior on weights "flattens" out after one point, the grads are too low once the log is too high.  
The higher value of the log cuts the loss, making grad descent not want to play bad.  
I guess the sweet spot would be where both of them nullify each other? not sure.  I'll have to see if there is something in the math which supports nullification.   


There is the other problem that the dict is created from codes, amazing how grad descent makes it work lol.  

In any case, next todo things:
- coordinate descent. 
- try different alpha on olevetti

## next steps again

- i've got a reasonable grip on what to use as hyperparameters. the relation between them. this is reasonable progress.
- i need to try on olivetti again with the new params
- need to now also see how i can understand if two patterns are walking together almost all the time
  - that is, two components are always coming together, the decomposer overdid it.  

- which do i pick first?
  - oliveti is the least effort, hopefully it would be easier to do