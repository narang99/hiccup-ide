import torch


class TrainedModel:
    def __init__(self, data_type, normaliser, components):
        self.components = components
        self.data_type = data_type
        self.normaliser = normaliser

    def transform(self, X):
        raise NotImplementedError


class TrainedDictLearningModel(TrainedModel):
    def __init__(self, dl, data_type, normaliser):
        super().__init__(data_type, normaliser, dl.components_)
        self.dl = dl

    def transform(self, X):
        codes = self.dl.transform(X)
        return codes, codes @ self.components


class TrainedKMeansModel(TrainedModel):
    def __init__(self, cluster, data_type, normaliser):
        super().__init__(data_type, normaliser, cluster.cluster_centers_)
        self.cluster = cluster

    def transform(self, X):
        labels = self.cluster.predict(X)
        return labels, self.cluster.cluster_centers_[labels]


def save_model(layer_data_dir, model: TrainedModel):
    norm_cls = model.normaliser.__class__.__name__
    n_comps = model.components.shape[0]
    alg_name = model.__class__.__name__
    save_dir = layer_data_dir / alg_name / str(n_comps) / model.data_type / norm_cls
    dest = save_dir / "model.pt"
    save_dir.mkdir(parents=True, exist_ok=True)
    print("saving to", dest)
    torch.save(model, dest)


class TrainedKMedoidsModel(TrainedModel):
    def __init__(self, cluster, data_type, normaliser):
        super().__init__(data_type, normaliser, cluster.cluster_centers_)
        self.cluster = cluster

    def transform(self, X):
        from sklearn.metrics.pairwise import pairwise_distances_argmin

        labels = pairwise_distances_argmin(
            X, Y=self.cluster.cluster_centers_, metric="euclidean"
        )
        recon = self.cluster.cluster_centers_[labels]
        return labels, recon


# 10k for 72: thats approx 100 per dim
# I have 512 dims, so 51,200 atleast, lets do 1 lakh then.
