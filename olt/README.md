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
