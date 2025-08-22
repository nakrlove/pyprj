from django.urls import path
from . import views

app_name = "bbs"

urlpatterns = [
    # 메인 페이지
    path("", views.MainPage.as_view(), name="index"),

    # 게시판
    path("list/", views.BbsLV.as_view(), name="list"),
    path("write/", views.BbsCreateView.as_view(), name="write"),
    path("<int:pk>/update/", views.BbsUpdateView.as_view(), name="update"),
    path("<int:pk>/", views.BbsDetailView.as_view(), name="detail"),

    # 딥러닝 실행 페이지
    path("deeplearning/", views.Deeplearning.as_view(), name="deeplearning"),

    # 예측 관련 페이지
    path("monthly/", views.MonthlyForecastPage.as_view(), name="monthly_forecast"),
    path("district/", views.DistrictForecastPage.as_view(), name="district_forecast"),
    path("price-range/", views.PriceRangeForecastPage.as_view(), name="price_range_forecast"),
    path("deposit/", views.DepositForecastPage.as_view(), name="deposit_forecast"),
]
