import json
from pathlib import Path
from django.core.management.base import BaseCommand, CommandError
from django.core.files import File
from neural_data.models import Model, Input, Activation, SaliencyMap, Weight
from neural_data.model_plugs import mnist

class Command(BaseCommand):
    help = "Load data from a folder containing meta.json into the database"

    def add_arguments(self, parser):
        parser.add_argument("folder_path", type=str, help="Path to the folder containing meta.json")

    def handle(self, *args, **options):
        folder_path = Path(options["folder_path"])
        if not folder_path.exists() or not folder_path.is_dir():
            raise CommandError(f"Folder not found: {folder_path}")

        meta_file = folder_path / "meta.json"
        if not meta_file.exists():
            raise CommandError(f"meta.json not found in {folder_path}")

        with open(meta_file, "r") as f:
            meta_data = json.load(f)

        model_info = meta_data.get("model")
        if not model_info:
            raise CommandError("meta.json missing 'model' information")

        model_name = model_info.get("name")
        model_path_rel = model_info.get("path")

        if not model_name:
            raise CommandError("Model 'name' is required in meta.json")

        model_pt_path = None
        if model_path_rel:
            model_pt_path = folder_path / model_path_rel
            if not model_pt_path.exists():
                raise CommandError(f"Model file not found: {model_pt_path}")

            self.stdout.write(f"Processing model: {model_name} from {model_pt_path}")
            model_schema, weight_coordinates = mnist.get_processed_model(str(model_pt_path))

            model_obj, created = Model.objects.get_or_create(
                alias=model_name,
                defaults={"name": model_name, "definition": model_schema},
            )

            if not created:
                model_obj.definition = model_schema
                model_obj.name = model_name
            
            # Save the pt file
            with open(model_pt_path, "rb") as f:
                model_obj.pt_file.save(model_pt_path.name, File(f), save=False)
            
            model_obj.save()

            # Process Weights
            self.stdout.write(f"Loading weights for model {model_name}...")
            weight_count = 0
            for coord, weight_data in weight_coordinates.items():
                Weight.objects.update_or_create(
                    model=model_obj,
                    coordinate=coord,
                    defaults={
                        "layer_name": weight_data.get("layer_name"),
                        "data": weight_data["data"],
                        "shape": weight_data["shape"],
                        "layer_type": weight_data["layer_type"],
                        "coordinate_type": weight_data["coordinate_type"],
                        "data_type": weight_data["data_type"],
                        "output_channel": weight_data.get("output_channel"),
                        "input_channel": weight_data.get("input_channel"),
                    }
                )
                weight_count += 1
            self.stdout.write(self.style.SUCCESS(f"Loaded {weight_count} weights"))
        else:
            # Check if model exists
            try:
                model_obj = Model.objects.get(alias=model_name)
                self.stdout.write(f"Using existing model: {model_name}")
                if not model_obj.pt_file:
                    raise CommandError(f"Model {model_name} exists but has no pt_file to process inputs")
                model_pt_path = Path(model_obj.pt_file.path)
            except Model.DoesNotExist:
                raise CommandError(f"Model {model_name} not found and no path provided to create it")

        # Process Inputs
        inputs_list = meta_data.get("inputs", [])
        for input_info in inputs_list:
            inp_path_rel = input_info.get("path")
            inp_label = input_info.get("label")

            if not inp_path_rel or inp_label is None:
                self.stdout.write(self.style.WARNING(f"Skipping invalid input: {input_info}"))
                continue

            inp_pt_path = folder_path / inp_path_rel
            if not inp_pt_path.exists():
                self.stdout.write(self.style.WARNING(f"Input file not found: {inp_pt_path}"))
                continue

            inp_alias = inp_pt_path.stem
            self.stdout.write(f"Processing input: {inp_alias} (label: {inp_label})")

            # Create/Update Input object
            category = str(inp_label)
            input_obj, created = Input.objects.get_or_create(
                model=model_obj,
                alias=inp_alias,
                defaults={"name": inp_alias, "data_path": str(inp_path_rel), "category": category},
            )
            
            if not created:
                input_obj.data_path = str(inp_path_rel)
                input_obj.name = inp_alias
                input_obj.category = category
            
            # Save pt file
            with open(inp_pt_path, "rb") as f:
                input_obj.pt_file.save(inp_pt_path.name, File(f), save=False)
            input_obj.save()

            # Process activations and saliency maps
            input_tensor = mnist.load_input_tensor(str(inp_pt_path))
            act_coords, contrib_coords = mnist.get_processed_activations_and_contribs(
                str(model_pt_path), input_tensor, inp_label
            )

            # Activations
            act_count = 0
            for coord, act_data in act_coords.items():
                Activation.objects.update_or_create(
                    input=input_obj,
                    coordinate=coord,
                    defaults={
                        "layer_name": act_data.get("layer_name"),
                        "data": act_data["data"],
                        "shape": act_data["shape"],
                        "layer_type": act_data["layer_type"],
                        "coordinate_type": act_data["coordinate_type"],
                        "output_channel": act_data.get("output_channel"),
                        "input_channel": act_data.get("input_channel"),
                    }
                )
                act_count += 1
            
            # Saliency Maps
            sal_count = 0
            for coord, sal_data in contrib_coords.items():
                SaliencyMap.objects.update_or_create(
                    input=input_obj,
                    coordinate=coord,
                    defaults={
                        "layer_name": sal_data.get("layer_name"),
                        "data": sal_data["data"],
                        "shape": sal_data["shape"],
                        "coordinate_type": sal_data["coordinate_type"],
                        "data_type": sal_data.get("data_type", "contrib"),
                        "output_channel": sal_data.get("output_channel"),
                        "input_channel": sal_data.get("input_channel"),
                    }
                )
                sal_count += 1
            
            self.stdout.write(self.style.SUCCESS(f"Loaded {act_count} activations and {sal_count} saliency maps for {inp_alias}"))

        self.stdout.write(self.style.SUCCESS("Folder loading complete!"))
