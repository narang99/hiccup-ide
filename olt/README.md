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
