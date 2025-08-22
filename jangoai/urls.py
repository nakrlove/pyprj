from django.contrib import admin
<<<<<<< HEAD
from django.urls import path,include
from bbs import views
from django.conf import settings
from django.conf.urls.static import static
# from bbs.views import Push


urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.index, name='index'),
    path('bbs/', include('bbs.urls')),
    # path('bbs/',include('bbs.urls')),
    # path('send_push/', Push.as_view(), name='send_push'), 
=======
from django.urls import path, include

urlpatterns = [
    path("admin/", admin.site.urls),
    # bbs 앱의 URL을 루트("/")에 연결
    path("", include("bbs.urls")),
>>>>>>> 7f26a98dffa9cf835af6ea79046f57e8a94510b8
]
