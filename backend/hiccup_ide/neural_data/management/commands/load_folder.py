from django.core.management.base import BaseCommand, CommandError
from neural_data.load_folder import load_folder_data

class Command(BaseCommand):
    help = "Load data from a folder containing meta.json into the database"

    def add_arguments(self, parser):
        parser.add_argument("folder_path", type=str, help="Path to the folder containing meta.json")

    def handle(self, *args, **options):
        folder_path = options["folder_path"]
        try:
            load_folder_data(folder_path, self.stdout)
        except Exception as e:
            raise CommandError(str(e))
