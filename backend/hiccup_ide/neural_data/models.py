from django.db import models

class Model(models.Model):
    alias = models.CharField(max_length=100, unique=True, db_index=True)
    name = models.CharField(max_length=200)
    definition = models.JSONField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.alias} - {self.name}"

    class Meta:
        db_table = "models"

class Input(models.Model):
    alias = models.CharField(max_length=100, db_index=True)
    model = models.ForeignKey(Model, on_delete=models.CASCADE, related_name='inputs')
    name = models.CharField(max_length=200)
    data_path = models.CharField(max_length=500)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.model.alias}/{self.alias} - {self.name}"

    class Meta:
        db_table = "inputs"
        unique_together = ['model', 'alias']

class Activation(models.Model):
    input = models.ForeignKey(Input, on_delete=models.CASCADE, related_name='activations')
    coordinate = models.CharField(max_length=200, db_index=True)
    data = models.JSONField()
    shape = models.JSONField()
    layer_type = models.CharField(max_length=100)
    coordinate_type = models.CharField(max_length=100)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.input} - {self.coordinate}"

    class Meta:
        db_table = "activations"
        unique_together = ['input', 'coordinate']

class SaliencyMap(models.Model):
    input = models.ForeignKey(Input, on_delete=models.CASCADE, related_name='saliency_maps')
    coordinate = models.CharField(max_length=200, db_index=True)
    data = models.JSONField()
    shape = models.JSONField()
    coordinate_type = models.CharField(max_length=100)
    data_type = models.CharField(max_length=100)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.input} - {self.coordinate} (saliency)"

    class Meta:
        db_table = "saliency_maps"
        unique_together = ['input', 'coordinate']

class Weight(models.Model):
    model = models.ForeignKey(Model, on_delete=models.CASCADE, related_name='weights')
    coordinate = models.CharField(max_length=200, db_index=True)
    data = models.JSONField()
    shape = models.JSONField()
    layer_type = models.CharField(max_length=100)
    coordinate_type = models.CharField(max_length=100)
    data_type = models.CharField(max_length=100)  # "weights" or "bias"
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.model.alias} - {self.coordinate} ({self.data_type})"

    class Meta:
        db_table = "weights"
        unique_together = ['model', 'coordinate']