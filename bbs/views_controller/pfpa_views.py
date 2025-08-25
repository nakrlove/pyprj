# 2025년 10월 전용면적별 거래량 예측
import logging
from django.shortcuts import render
from django.urls import reverse_lazy
from django.views.generic import (
    ListView, DetailView, FormView, CreateView, UpdateView, TemplateView
)
from django.http import JsonResponse
from django.views import View

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
    


class PriceForPerArea(View):
    template_name = 'bbs/price_for_per_area.html'
    def get(self, request, *args, **kwargs):
        print(" get #####우리가 남이다 .......")
        try:
            analysis_result = engine()
            print(" 우리가 남이다 .......")
            return JsonResponse({'result': analysis_result})
        except Exception as e:
            # 오류가 발생하면 오류 메시지를 context에 추가합니다.
            # context['result'] = {'ERROR': str(e)}
            return JsonResponse({'ERROR': str(e)}, status=500)
        

    def post(self, request, *args, **kwargs):
        print(" post #####우리가 남이다 .......")
        try:
            analysis_result = engine()
            print(" 우리가 남이다 .......")
            return JsonResponse({'result': analysis_result})
        except Exception as e:
            # 오류가 발생하면 오류 메시지를 context에 추가합니다.
            # context['result'] = {'ERROR': str(e)}
            return JsonResponse({'ERROR': str(e)}, status=500)
        
    