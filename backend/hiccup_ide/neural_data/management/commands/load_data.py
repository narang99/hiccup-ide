import json
from pathlib import Path
from django.core.management.base import BaseCommand
from neural_data.models import Model, Input, Activation, SaliencyMap, Weight


class Command(BaseCommand):
    help = "Load data from frontend public folder into database"

    def handle(self, *args, **options):
        public_folder = Path(
            "/Users/hariomnarang/Desktop/personal/hiccup-ide/frontend/neural-viz/public"
        )

        if not public_folder.exists():
            self.stdout.write(
                self.style.ERROR(f"Public folder not found: {public_folder}")
            )
            return

        try:
            # Load model definition
            model_file = public_folder / "example-model.json"
            if not model_file.exists():
                self.stdout.write(
                    self.style.ERROR(f"Model file not found: {model_file}")
                )
                return

            with open(model_file, "r") as f:
                model_data = json.load(f)

            # Create or get model
            model, created = Model.objects.get_or_create(
                alias="example-model",
                defaults={"name": "Example Neural Network", "definition": model_data},
            )

            if created:
                self.stdout.write(f"Created new model: {model.alias}")
            else:
                self.stdout.write(f"Using existing model: {model.alias}")
                # Update definition
                model.definition = model_data
                model.save()

            # Create or get input
            input_obj, created = Input.objects.get_or_create(
                model=model,
                alias="first-input",
                defaults={"name": "First Input", "data_path": "/data/first-input.json"},
            )

            if created:
                self.stdout.write(f"Created new input: {input_obj.alias}")
            else:
                self.stdout.write(f"Using existing input: {input_obj.alias}")

            # Load activations
            activations_folder = public_folder / "activations"
            activation_count = 0
            if activations_folder.exists():
                for activation_file in activations_folder.glob("*.json"):
                    coordinate = activation_file.stem
                    try:
                        with open(activation_file, "r") as f:
                            activation_data = json.load(f)

                        layer_name = activation_data.get("layer_name")

                        activation, created = Activation.objects.get_or_create(
                            input=input_obj,
                            coordinate=coordinate,
                            defaults={
                                "layer_name": layer_name,
                                "data": activation_data["data"],
                                "shape": activation_data["shape"],
                                "layer_type": activation_data["layer_type"],
                                "coordinate_type": activation_data["coordinate_type"],
                            },
                        )

                        if not created:
                            # Update existing activation
                            activation.layer_name = layer_name
                            activation.data = activation_data["data"]
                            activation.shape = activation_data["shape"]
                            activation.layer_type = activation_data["layer_type"]
                            activation.coordinate_type = activation_data[
                                "coordinate_type"
                            ]
                            activation.save()

                        activation_count += 1

                    except Exception as e:
                        self.stdout.write(
                            self.style.WARNING(
                                f"Error loading activation {activation_file}: {e}"
                            )
                        )

            # Load saliency maps
            saliency_folder = public_folder / "saliency_map"
            saliency_count = 0
            if saliency_folder.exists():
                for saliency_file in saliency_folder.glob("*.json"):
                    coordinate = saliency_file.stem
                    try:
                        with open(saliency_file, "r") as f:
                            saliency_data = json.load(f)

                        # Extract layer_name from data as requested
                        layer_name = saliency_data.get("layer_name")

                        saliency_map, created = SaliencyMap.objects.get_or_create(
                            input=input_obj,
                            coordinate=coordinate,
                            defaults={
                                "layer_name": layer_name,
                                "data": saliency_data["data"],
                                "shape": saliency_data["shape"],
                                "coordinate_type": saliency_data["coordinate_type"],
                                "data_type": saliency_data.get("data_type", "contrib"),
                            },
                        )

                        if not created:
                            # Update existing saliency map
                            saliency_map.layer_name = layer_name
                            saliency_map.data = saliency_data["data"]
                            saliency_map.shape = saliency_data["shape"]
                            saliency_map.coordinate_type = saliency_data[
                                "coordinate_type"
                            ]
                            saliency_map.data_type = saliency_data.get(
                                "data_type", "contrib"
                            )
                            saliency_map.save()

                        saliency_count += 1

                    except Exception as e:
                        self.stdout.write(
                            self.style.WARNING(
                                f"Error loading saliency map {saliency_file}: {e}"
                            )
                        )

            # Load weights
            weights_folder = public_folder / "weights"
            weights_count = 0
            if weights_folder.exists():
                for weights_file in weights_folder.glob("*.json"):
                    coordinate = weights_file.stem
                    try:
                        with open(weights_file, "r") as f:
                            weights_data = json.load(f)

                        layer_name = weights_data.get("layer_name")

                        weight, created = Weight.objects.get_or_create(
                            model=model,
                            coordinate=coordinate,
                            defaults={
                                "layer_name": layer_name,
                                "data": weights_data["data"],
                                "shape": weights_data["shape"],
                                "layer_type": weights_data["layer_type"],
                                "coordinate_type": weights_data["coordinate_type"],
                                "data_type": weights_data["data_type"],
                            },
                        )

                        if not created:
                            # Update existing weight
                            weight.layer_name = layer_name
                            weight.data = weights_data["data"]
                            weight.shape = weights_data["shape"]
                            weight.layer_type = weights_data["layer_type"]
                            weight.coordinate_type = weights_data["coordinate_type"]
                            weight.data_type = weights_data["data_type"]
                            weight.save()

                        weights_count += 1

                    except Exception as e:
                        self.stdout.write(
                            self.style.WARNING(
                                f"Error loading weight {weights_file}: {e}"
                            )
                        )

            self.stdout.write(
                self.style.SUCCESS(
                    f"Successfully loaded data:\n"
                    f"- Model: {model.alias}\n"
                    f"- Input: {input_obj.alias}\n"
                    f"- Activations: {activation_count}\n"
                    f"- Saliency Maps: {saliency_count}\n"
                    f"- Weights: {weights_count}"
                )
            )

        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Error loading data: {e}"))
