Directory structure

Input shards

```
input-root-dir
  imagenet-label
    000000.tar
```

Attributions

```
attribution-root-dir
  layer_name
    imagenet-label
      000000.tar
```

Patches

```
patches-root-dir
  layer_name
    channel
      000000.tar
```

Models

```
model-dir
  layer_name
    channel
      model.pkl
      meta.json
```

Thike, the trained model is working. First thing ill do is train all the models for a single layer :).  
This is not the best scaling method (it might be better to split it into stages later), but its fine.
I anyways need to put the model training script in the library.

After this, ill tackle the report problem, ill need to design something better. Its best to also think about how we would also instrument all the neurons in the end for generating reports, so it might be useful to come with an intermediate representation.

My S3 bill is already $22 lol. This is because of the repeated push and pulls ive done. It might be best to let it run end to end for every layer now i think. 22$ for one layer is brutal.

Technically, i can just put only the models. without the data, if something fails, we regenerate the data. we would also like the thresholds though :)

Ohk, the cluster is too noisy definitely. DBCV is not working out as a great metric it seems. There is one thing though, i definitely need higher min-cluster-sizes, small sizes are stupid

I definitely need a higher cluster size.

Auto clustering is definitely difficult

This might be an artifact of deeplift also btw, deeplift ends up giving more clusters generally.
I might just try integrated gradients and get some random reports to see if they are any good actually.
Its not that bad, 2 hours for a layer.
I would need to run attributions collection in a loop at night basically.

Colab disconnecting is a pain. we do have sagemaker though. i can use 30 hours of kaggle gpus also. i should start that too. but need cuml support, along with cpu serialisation and testing.

Ohk, so deeplift has a tendency of giving one-off results (with single examples sending a lot of labels. we want to disable them).
So before generating the report, the IR structure:

We keep a flat csv with the following columns

```json
{
  "cluster_label": "...",
  "imagenet_label": "...",
  "layer_name": "...",
  "channel": "...",
  "input_image_key": "...", // image shard key
  "location": "..." // [h,w]
}
```

For a single neuron we query the layer_name and channel combo. If it only contains one imagenet example, we flag it. easy.

Ohk, now for report generation, we take a layer by channel by clusterer dict.
Call layer attribution for all existing layers.  
Go through each channel clusterer present, get those attributions, call our code on the input data, then store the combination in the dataset.

Report generator in the end would go through all image shards, for every input_image_key, it finds all cluster labels and positions. each cluster label also has a marker "valid" label
During this report generation, we would also like to mark some labels as "bad" for a clusterer (the ones which have only single example). We silence them by default (in the report, they should come in the end, collapsed). The report should also has some sort of red marking to do that.
The neuron attribution generator now picks the set of locations it wants to check out, goes through the input again, calls neuron deeplift for each location, puts the output in the corresponding directory.

The reporter then simply goes through these, prepares report by overlaying. and we done.

We need sampling of input data, hdbscan can go wonk. it stores the existing data for future predictions, so it can get huge. tis very bad.

Ohk, RF is not bad. But the first thing i need is reports. its hard to do much without them. Some things to consider:

- keep a list of labels which are not good. (add a floor on the number of examples a label should have)
- train distilled RF. Keep it running on the floor only
- Report generation should run both RF and HDBScan, it should flag the outputs where they disagree for each label.
  (RF said this, HDB said that kinda thing).

Good, now ive got a report, no need to run the model again, we simply need to run neuron attribution.

```
chan55
full data threshold: {"positive": 1.4402424312720541e-05, "negative": -1.2457826414902229e-05}
only cat and car data: {"positive": 3.2870642030502495e-07, "negative": -3.212472279301437e-07}
```

The problem is evident now, we dont see these occurring. deepdream dreams up stuff which maximises the activation of this kernel, but those maximising things dont have high attribution to the final output it seems.  
There is only one natural next step, have thresholds per class.
This will be major code changes i think.

# Check label wise thresholds are fine

plot the distribution of values above threshold below:

```
from pathlib import Path
import json

base = Path("./workdir/thresholds/mixed4e_1x1_pre_relu_conv")
with open(base / "55" / "thresholds.json") as f:
  content = json.load(f)

apts, ants = [], []
for chan in range(256):
    with open(base / str(chan) / "thresholds.json", "r") as f:
      content = json.load(f)
      pts = [c["pos_above_elbow_ratio"] for c in content.values()]
      apts.extend(pts)
      nts = [c["neg_above_elbow_ratio"] for c in content.values()]
      ants.extend(nts)

plt.hist(apts)
plt.show()
plt.hist(ants)
plt.show()
```

This gives reasonable results, maximum 0.08. nothing crazy. label 281 for 55 (cat for mixed4e-pre-relu-1x1 neuron 255) has 0.03 (3%). its fine for now. good to test first.

# Restrict data for speed

I also would like to see proof about how many cats ended up being in the face cluster (281 label? how many were in faces cluster?).  
Simply download the report to laptop, check in pandas how many unique keys are present in that cluster.

Ohk, now i need proof that it works. Have downloaded the report. not bad for now.  
The main problem is that now we have a huge number of objects. should i halve it? Actually lets wait for the results.

Ohk, the data is in gbs now, this way, i would need a day to train a layer lol. i dont have that kind of time. but this is also necessary. We need at least 10 samples per category i think.
The main problem is the sheer amount of noise that can't be clustered.
Yea i dont see a way out of this, its fine. first we need the initial test results to see if clustering is even possible.

I will restrict number of samples to 16 i think. thats not a bad number (approx 1/3rd the original data).

# Two things

- now there is a lot of data. we want to analyse it to see if clustering is good.
  - first see how the cat classes have clustered.
  - Then see how the car classes have clustered
  - hopefully they are separate

- Ohk, we will now analyse this neuron first. then do sampled training and see how it works.
  - for now, i do have the clustering numbers from sampled training (the split of labels, hopefully it matches)

Well we have a problem now. same image can be in multiple imagenet labels. ;\_;  
And this decreases my dataset size. tis annoying. very annoying.

What can we do? This is a property of the dataset and is hard to reconcile with.

Its time to write about our work. do i need qmd file? or ipynb? I might want to write code later too, for showing csv files. other than that, i dont expect to write a lot of code though.
What to do? why tis already got complicated lol.

Its best to do ipynb for now, then we'll see if we want more stuff.

Actually, lets do `qmd` only.

Ohk, some things to note.

- We would need better final reports.
- For clustering details, I should link the older reports.

More things to test for the blog:

- get the count of images in each cluster label from the csv file, currently we are using 50 images each.
- we would like to decrease the dataset after one point, but we do want 50 images right now

# im running very confused

What do i need to test more? One is the polysemantic car neuron these people are talking about which goes onto dogs. Does it only do cars?
We would like report on 50 images, for the article.

First thing, i need cleaner final reports for my article. for this, i would like to have images which have only single label.  
That is what we'll do for the final test. for that, i would need new image shards, new attribution shards, etc.

That needs to run in a separate notebook.

Now, do we run on 16 images or 50 images?

We have the results. running on 16 images is fast and i would like to do that. before that however, do we need a result for 50 images? For the report itself, we would need both 50 and 16 image reports.

That is the first task. 50 -> less than 50 cuz some might not be nice.

The other thing is clustering. Im using leaf with a high in cluster size, lets try a smaller size and see the report at least. This is interesting and will be done right now.

Ohk, mai actually report analysis mei aalas karra hu. We want to prove that a lot of cats are coming. But before that, I would like to show the report itself.

Abi im doing aalas in two things.

1. Doing circuit analysis and finding what input pixels we depend on.
2. redo-ing everything from scratch for cleaner data.

Kaafi chizo mei mai lite lera hu. the main problem is samples having same name, we gotta fix that.

For work to be interesting also, it would make sense to have a way to trace back on major contributions of a given label. Can we use random forests?

ohk, data seems nice, we dont need to change it. the last thing we might need to do is just use 16 images maybe?
Do i need to do the whole thing again? Well yea i should i think. just need to run attribution analysis on them for now. 16 is useful cuz i can do more experiments rapidly also.
But, having the 50 directory is also useful, you would like to do experiments with more data when you want. easy. so no need for that right now. we skip.

So what are the next steps? First is making another leaf report for cluster size 5? to see what is caught. the problem is the hyperparameter generally. but, we are not really interested in bulk running for now. the goal is much smaller, to check out neurons one at a time.
It would be useful to provide correct image categories using deepdream, but in general get attributions and patches will take more than an hour, which is quite painful to work with. even though all of that can be automated easily, ive not done that. cuz of the amount of data involved.

So even if i do one neuron at a time, its gonna take a long time cuz ive to zip through the whole data. can i do something about this? the first thing is using 16 images, which would definitely decrease the time we go through the dataset, and is quite useful. for 16 cases though, we'll need a different work dir. Otherwise we'll mess things up.

Generally though, im gonna work with 16 elements only, helpful for both disk, and attribution work.  
So we first do that for our simple case. Also, its best if i remove the duplciate images anyways.

After this, we do analysis one by one for neurons which are supposed to be polysemantic.

We consistently see leaf model being better, now there is only the problem of stability of the leaf clusterer. it can change wildly on min-cluster-size lol.
I would be okay if it was splitting more, but the noise handling has changed, which is a problem

So, fuck doing mass clustering. its not that easy or nice. i need to look at results normally to see what is happening lol.

So clustering now needs to me more of a experimenting thing instead of the thing we have where we write stuff to fs, so ill use the basic clustering code directly.

# Have clean data

I need to get less ambitious and put something out first.

- We see differences in what feature visualisation catches, and what we catch.
  - we are catching a superset generally.
  - we would like to see the differences
- a clustering workflow
  - test eom vs leaf on neuron255
  - see different reports. decide what you want to use. im leaning towards leaf
  - once that is done, the only way to actually verify how the clusters look is through manually checking them out for now
  - one thing i should do is closely look at leaf-5 size and leaf-20 size. are the noise points similar? in this case, leaf-5 is simply doing a bigger superset of leaf-20 with finer clustering. GMMs won't work though (the data does not follow the long tail thing we have noticed before).

- im thinking that for leaf mode, DBCV would keep increasing, can we do elbow? test this.
  - dbcv likes sparse micro clusters. lets see how the value looks when we keep increasing it.
