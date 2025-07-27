# from bbs.dao.bbs_models import Bbs
# from django.db import models

# from django.db.models import F
from bbs.db.bbs_mysql import get_bbs_with_rownum,get_total_bbs_count
from django.views.generic import ListView
# from django.core.paginator import Paginator
# from bbs.models import Bbs


    # def get_queryset(self):
    #     # 1번 + 2번 조건: 댓글은 원글 밑에, 게시글은 최신순
    #     return Bbs.objects.filter(is_deleted=False).order_by(
    #         F('group_id').asc(nulls_last=True),  # 같은 그룹끼리
    #         'depth',                             # 댓글 구조 순서
    #         '-created_at' if self.request.GET.get('depth_sort') else 'created_at'  # 필요시 역순 depth
    #     )

    # def get_context_data(self, **kwargs):
    #     context = super().get_context_data(**kwargs)

    #     # 4번 조건: 전체 건수 기준으로 리스트 번호 계산
    #     page = context['page_obj']
    #     total = self.get_queryset().count()
    #     per_page = self.paginate_by
    #     current_page = page.number
    #     start_index = total - ((current_page - 1) * per_page)

    #     context['total'] = total
    #     context['start_index'] = start_index
    #     print(f"start_index={start_index}, total={total}, page={page}, per_page={per_page}")
    #     return context    
# class MySQLDB(ListView):    

#     def get_queryset(self):
#         # 페이지 번호 계산
#         page = self.request.GET.get('page')
#         if page is None:
#             page = 1
#         else:
#             try:
#                 page = int(page)
#             except ValueError:
#                 page = 1

#         offset = (page - 1) * self.paginate_by
#         limit = self.paginate_by

#         # get_bbs_with_rownum 호출해서 데이터 가져오기
#         return get_bbs_with_rownum(offset=offset, limit=limit)

#     def get_context_data(self, **kwargs):
#         context = super().get_context_data(**kwargs)

#         # 총 데이터 개수는 get_bbs_with_rownum에서 같이 가져오거나 별도 함수로 가져와야 함
#         # 예를 들어 get_total_bbs_count() 함수 따로 만들고 호출 가능

#         total = get_total_bbs_count()
#         page = context['page_obj']
#         per_page = self.paginate_by
#         current_page = page.number
#         start_index = total - ((current_page - 1) * per_page)

#         context['total'] = total
#         context['start_index'] = start_index
#         return context    