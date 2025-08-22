"""
URL configuration for jangoai project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

from django.contrib import admin
from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static

app_name = 'bbs'
urlpatterns = [
    # 첫 페이지를 MainPage 뷰로 연결
    path('', views.MainPage.as_view() , name="index"),
    
    # 게시판 목록 페이지는 별도 URL('list/')로 분리
    path('list/', views.BbsLV.as_view() , name="list"),
    
    # 기존 게시판 URL 패턴
    path('write/', views.BbsCreateView.as_view(), name='write'),
    path('<int:pk>/update/', views.BbsUpdateView.as_view(), name='update'),
    path('<int:pk>/', views.BbsDetailView.as_view(), name='detail'),
    path("deeplearning/", views.Deeplearing.as_view(), name="deeplearning"),

    # 새로운 페이지를 위한 URL 패턴
    path('monthly/', views.MonthlyForecastPage.as_view(), name='monthly_forecast'),
    path('district/', views.DistrictForecastPage.as_view(), name='district_forecast'),
    path('price-range/', views.PriceRangeForecastPage.as_view(), name='price_range_forecast'),
    path('deposit/', views.DepositForecastPage.as_view(), name='deposit_forecast'),
]

# 개발 환경에서 정적 파일을 서빙하기 위한 설정 추가
if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
    # 또한, 미디어 파일을 위한 설정도 추가합니다.
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)