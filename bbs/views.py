from django.shortcuts import render
from bbs.dao.bbs_models import Bbs
# Create your views here.
from django.core.paginator import Paginator
from bbs.db.bbs_mysql import get_bbs_with_rownum,get_total_bbs_count
# from bbs.models import MySQLDB  
from django.views.generic import ListView
class BbsLV(ListView):
# class BbsLV(MySQLDB):    
    model = Bbs
    template_name = 'bbs/bbs_list.html'
    context_object_name = 'bbs_list'
    paginate_by = 20  # 3번 조건: 20건씩 가져오기    

 
    def get_queryset(self):
        # 페이지 번호 계산
        page = self.request.GET.get('page')
        if page is None:
            page = 1
        else:
            try:
                page = int(page)
            except ValueError:
                page = 1

        offset = (page - 1) * self.paginate_by
        limit = self.paginate_by

        # get_bbs_with_rownum 호출해서 데이터 가져오기
        return get_bbs_with_rownum(offset=offset, limit=limit)

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)

        # 총 데이터 개수는 get_bbs_with_rownum에서 같이 가져오거나 별도 함수로 가져와야 함
        # 예를 들어 get_total_bbs_count() 함수 따로 만들고 호출 가능
        # ----------------------------------------
        total = get_total_bbs_count()
        page = context['page_obj']
        per_page = self.paginate_by
        current_page = page.number
        start_index = total - ((current_page - 1) * per_page)

        context['total'] = total
        context['start_index'] = start_index
        print(f"start_index = {start_index}")
        # ----------------------------------------




        paginator = context['paginator']
        page_obj = context['page_obj']

        current_page = page_obj.number
        total_pages = paginator.num_pages
        page_range = paginator.page_range

        block_size = 3
        start_page = ((current_page - 1) // block_size) * block_size + 1
        end_page = start_page + block_size - 1
        if end_page > total_pages:
            end_page = total_pages

        context['start_page'] = start_page
        context['end_page'] = end_page
        context['has_prev_block'] = start_page > 1
        context['has_next_block'] = end_page < total_pages
        context['prev_block_page'] = start_page - 1
        context['next_block_page'] = end_page + 1
        context['page_range'] = range(start_page, end_page + 1)

        return context    


def bbs_list(request):
    post_list = Bbs.objects.all().order_by('-created_at')  # 최신순
    paginator = Paginator(post_list, 10)  # 한 페이지 10개
    page_number = request.GET.get('page', 1)
    page_obj = paginator.get_page(page_number)
    print("ddddddddd")
    # 전체 글 수
    total_count = paginator.count
    # 현재 페이지의 시작 순번 (전체 글 수에서 현재 페이지 시작 인덱스 빼기)
    start_index = total_count - (page_obj.number - 1) * paginator.per_page

    context = {
        'page_obj': page_obj,
        'start_index': start_index,
    }
    return render(request, 'bbs/bbs_list.html', context)