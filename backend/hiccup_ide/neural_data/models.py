from django.db import models

class Model(models.Model):
    alias = models.CharField(max_length=100, unique=True, db_index=True)
    name = models.CharField(max_length=200)
    definition = models.JSONField()
    pt_file = models.FileField(upload_to='models/pt/')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.alias} - {self.name}"

    def load_pt_file(self):
        import torch
        if not self.pt_file:
            return None
        return torch.load(self.pt_file.path)

    class Meta:
        db_table = "models"

class Input(models.Model):
    alias = models.CharField(max_length=100, db_index=True)
    model = models.ForeignKey(Model, on_delete=models.CASCADE, related_name='inputs')
    name = models.CharField(max_length=200)
    data_path = models.CharField(max_length=500)
    pt_file = models.FileField(upload_to='inputs/pt/')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.model.alias}/{self.alias} - {self.name}"

    def load_pt_file(self):
        import torch
        if not self.pt_file:
            return None
        return torch.load(self.pt_file.path, weights_only=False)

    class Meta:
        db_table = "inputs"
        unique_together = ['model', 'alias']

class Activation(models.Model):
    input = models.ForeignKey(Input, on_delete=models.CASCADE, related_name='activations')
    coordinate = models.CharField(max_length=200, db_index=True)
    layer_name = models.CharField(max_length=200, db_index=True)
    data = models.JSONField()
    shape = models.JSONField()
    layer_type = models.CharField(max_length=100)
    coordinate_type = models.CharField(max_length=100)
    output_channel = models.IntegerField(null=True, blank=True)
    input_channel = models.IntegerField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.input} - {self.coordinate}"

    class Meta:
        db_table = "activations"
        unique_together = ['input', 'coordinate']


class SaliencyMapData(models.Model):
    layer_name = models.CharField(max_length=200, db_index=True)
    data = models.JSONField()
    shape = models.JSONField()
    coordinate_type = models.CharField(max_length=100)
    data_type = models.CharField(max_length=100)
    output_channel = models.IntegerField(null=True, blank=True)
    input_channel = models.IntegerField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)


    class Meta:
        db_table = "saliency_maps"
        unique_together = ['input', 'coordinate']
        abstract = True

class SaliencyMap(SaliencyMapData):
    input = models.ForeignKey(Input, on_delete=models.CASCADE, related_name='saliency_maps')
    coordinate = models.CharField(max_length=200, db_index=True)

    def __str__(self):
        return f"{self.input} - {self.coordinate} (saliency)"

    class Meta:
        db_table = "saliency_maps"
        unique_together = ['input', 'coordinate']

class Weight(models.Model):
    model = models.ForeignKey(Model, on_delete=models.CASCADE, related_name='weights')
    coordinate = models.CharField(max_length=200, db_index=True)
    layer_name = models.CharField(max_length=200, db_index=True)
    data = models.JSONField()
    shape = models.JSONField()
    layer_type = models.CharField(max_length=100)
    coordinate_type = models.CharField(max_length=100)
    data_type = models.CharField(max_length=100)  # "weights" or "bias"
    output_channel = models.IntegerField(null=True, blank=True)
    input_channel = models.IntegerField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.model.alias} - {self.coordinate} ({self.data_type})"

    class Meta:
        db_table = "weights"
        unique_together = ['model', 'coordinate']

class Work(models.Model):
    input = models.ForeignKey(Input, on_delete=models.CASCADE, related_name='works')
    name = models.CharField(max_length=100, db_index=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.input} - {self.name}"

    class Meta:
        db_table = "works"
        unique_together = ['input', 'name']

class WorkGraph(models.Model):
    work = models.OneToOneField(Work, on_delete=models.CASCADE, related_name='graph')

    class Meta:
        db_table = "work_graphs"


class WorkSaliencyMap(SaliencyMapData):
    input = models.ForeignKey(Input, on_delete=models.CASCADE, related_name='work_saliency_maps')
    coordinate = models.CharField(max_length=200, db_index=True)
    graph = models.ForeignKey(WorkGraph, on_delete=models.CASCADE)
    is_done = models.BooleanField(default=False)

    def __str__(self):
        return f"{self.input} - {self.coordinate} (saliency)"

    class Meta:
        unique_together = ['coordinate', 'graph']

class TempPruneSaliencyMap(SaliencyMapData):
    input = models.ForeignKey(Input, on_delete=models.CASCADE, related_name='temp_saliency_maps')
    coordinate = models.CharField(max_length=200, db_index=True)
    graph = models.ForeignKey(WorkGraph, on_delete=models.CASCADE)
    is_modified = models.BooleanField(default=False)

    def __str__(self):
        return f"{self.input} - {self.coordinate} (temp saliency)"

    class Meta:
        unique_together = ['input', 'coordinate', 'graph']


class POI(models.Model):
    work = models.ForeignKey(Work, on_delete=models.CASCADE)
    weight = models.ForeignKey(Weight, on_delete=models.CASCADE)

    x = models.IntegerField()
    y = models.IntegerField()

    note = models.TextField()
    label = models.CharField(max_length=20)

    class Meta:
        db_table = "pois"
        unique_together = ['work', 'weight', 'x', 'y']


class KernelLabels(models.Model):
    weight = models.OneToOneField(Weight, on_delete=models.CASCADE)
    labels = models.JSONField(default=list)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Labels for {self.weight.coordinate}: {self.labels}"

    class Meta:
        db_table = "kernel_labels"
