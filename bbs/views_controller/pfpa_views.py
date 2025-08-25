# 2025년 10월 전용면적별 거래량 예측
import logging
from django.shortcuts import render
from django.urls import reverse_lazy
from django.views.generic import (
    ListView, DetailView, FormView, CreateView, UpdateView, TemplateView
)
from django.http import JsonResponse
from django.views import View
import pandas as pd
import numpy as np
import json
from bbs.biz.price_for_per_area_line import engine
class PriceForPerAreaPage(TemplateView):
    template_name = 'bbs/price_for_per_area.html'
    # def get_context_data(self, **kwargs):
    #     context = super().get_context_data(**kwargs)
        
    #     try:
    #         # resultData() 함수를 호출하여 분석 결과를 가져옵니다.
    #         analysis_result = engine()
    #         # context에 결과 딕셔너리 자체를 추가합니다. .items()는 제거합니다.
    #         context['result'] = analysis_result
    #     except Exception as e:
    #         # 오류가 발생하면 오류 메시지를 context에 추가합니다.
    #         context['result'] = {'ERROR': str(e)}
            
    #     return context
    

def convert(obj):
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    elif isinstance(obj, pd.Series):
        return obj.to_dict()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: convert(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert(v) for v in obj]
    else:
        return obj
    
class PriceForPerArea(View):
    template_name = 'bbs/price_for_per_area.html'
    def get(self, request, *args, **kwargs):
       
        try:
            # result = engine().to_dict(orient="records")
            result = engine()
            result = convert(result)
            # print(" ============================ ")
            # print(f" {result} ")
            return JsonResponse(result, safe=False)
        except Exception as e:
            # 오류가 발생하면 오류 메시지를 context에 추가합니다.
            # context['result'] = {'ERROR': str(e)}
            print( str(e))
            return JsonResponse({'ERROR': str(e)}, status=500)
        
    def post(self, request, *args, **kwargs):
        try:
            # 요청 JSON 읽기
            body = json.loads(request.body)
            print("받은 JSON:", body)

            # 예시: 파라미터 추출
            # area = body.get("area")
            # year = body.get("year")
            # month = body.get("month")
            eventType = body.get("TYPE")
            # engine 호출 (예시)
            result = engine(eventType)
            result = convert(result)

            return JsonResponse(result, safe=False)
        except Exception as e:
            return JsonResponse({"ERROR": str(e)}, status=500)
    # def post(self, request, *args, **kwargs):
    #     print(" post #####우리가 남이다 .......")
    #     try:
    #         result = engine()
    #         print(f" 우리가 남이다  {result}")
    #         return JsonResponse(result)
    #     except Exception as e:
    #         # 오류가 발생하면 오류 메시지를 context에 추가합니다.
    #         # context['result'] = {'ERROR': str(e)}
    #         return JsonResponse({'ERROR': str(e)}, status=500)
        
    