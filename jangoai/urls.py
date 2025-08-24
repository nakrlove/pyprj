from django.contrib import admin

from django.urls import path,include
from bbs import views
from django.conf import settings
from django.conf.urls.static import static
# from bbs.views import Push


urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.index, name='index'),
    path('bbs/', include('bbs.urls')),
  
]
