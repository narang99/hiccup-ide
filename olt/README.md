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
