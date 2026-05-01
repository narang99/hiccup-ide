import json
from pathlib import Path
from django.core.management.base import BaseCommand
from neural_data.models import Model, Input, Activation, SaliencyMap


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

                        activation, created = Activation.objects.get_or_create(
                            input=input_obj,
                            coordinate=coordinate,
                            defaults={
                                "data": activation_data["data"],
                                "shape": activation_data["shape"],
                                "layer_type": activation_data["layer_type"],
                                "coordinate_type": activation_data["coordinate_type"],
                            },
                        )

                        if not created:
                            # Update existing activation
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
            saliency_folder = public_folder / "saliency_maps"
            saliency_count = 0
            if saliency_folder.exists():
                for saliency_file in saliency_folder.glob("*.json"):
                    coordinate = saliency_file.stem
                    try:
                        with open(saliency_file, "r") as f:
                            saliency_data = json.load(f)

                        saliency_map, created = SaliencyMap.objects.get_or_create(
                            input=input_obj,
                            coordinate=coordinate,
                            defaults={
                                "data": saliency_data["data"],
                                "shape": saliency_data["shape"],
                                "coordinate_type": saliency_data["coordinate_type"],
                                "data_type": saliency_data.get("data_type", "contrib"),
                            },
                        )

                        if not created:
                            # Update existing saliency map
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

            self.stdout.write(
                self.style.SUCCESS(
                    f"Successfully loaded data:\n"
                    f"- Model: {model.alias}\n"
                    f"- Input: {input_obj.alias}\n"
                    f"- Activations: {activation_count}\n"
                    f"- Saliency Maps: {saliency_count}"
                )
            )

        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Error loading data: {e}"))
