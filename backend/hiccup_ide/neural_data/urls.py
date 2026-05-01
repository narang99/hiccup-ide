from django.urls import path
from neural_data.api import router
from ninja import NinjaAPI


api = NinjaAPI()

api.add_router("", router)

urlpatterns = [
    path("api/", api.urls),
]
