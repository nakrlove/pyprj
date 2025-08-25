from django.contrib import admin

from django.urls import path,include
from bbs import views
from django.conf import settings
from django.conf.urls.static import static
# from bbs.views import Push

urlpatterns = [
    path("admin/", admin.site.urls),
    # bbs 앱의 URL을 루트("/")에 연결
    path("", include("bbs.urls")),

]
