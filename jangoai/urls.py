from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path("admin/", admin.site.urls),
    # bbs 앱의 URL을 루트("/")에 연결
    path("", include("bbs.urls")),
]
