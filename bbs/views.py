from django.shortcuts import render
from bbs.dao.bbs_models import Bbs
# Create your views here.
from django.core.paginator import Paginator
from bbs.db.bbs_mysql import get_bbs_with_rownum,get_total_bbs_count
# from bbs.models import MySQLDB  
from django.views.generic import ListView,DetailView,TemplateView



from django.views.generic.edit import CreateView, UpdateView
from django.urls import reverse_lazy
# from .models import Bbs
from .forms import PostForm  # ModelForm
from django.db import transaction
from .models import Bbs
from bbs.dao.bbs_models import BbsFile 

#####################################
# 게시판 목록 조회
#####################################
class BbsLV(ListView):
# class BbsLV(MySQLDB):    
    model = Bbs
    template_name = 'bbs/bbs_list.html'
    context_object_name = 'bbs_list'
    paginate_by = 20  # 3번 조건: 20건씩 가져오기    

    #==================================================================================================
    # ListView는 기본적으로 GET 요청만 처리하도록 설계되어 있으며, POST 요청을 처리하려면 직접 오버라이드해서 구현해야 합니다
    #==================================================================================================
    #  순서 흐름 (GET 요청 기준):
    # as_view() 
    # → def dispatch(self, request, *args, **kwargs)
    # → def get(self, request, *args, **kwargs)
    # → def get_queryset(self)
    # → def get_context_data(self, **kwargs):
    # → def render_to_response(self, context, **response_kwargs)
    # → template 렌더링


    #  순서 흐름 (POST 요청 기준):
    # as_view() 
    # → dispatch(request, *args, **kwargs) 
    # → post(request, *args, **kwargs)   
    # → def form_valid(self, form)	  유효한 form 처리	HttpResponseRedirect, render_to_response()
    # → def form_invalid(self, form)  유효하지 않은 form 처리	render_to_response()
    # → render_to_response(context) 또는 HttpResponse 반환

    
    # GET 파라미터 접근	self.request.GET.get('키')
    # POST 파라미터 접근	self.request.POST.get('키')
    # 쿠키 접근	self.request.COOKIES.get('키')
    # 세션 접근	self.request.session.get('키')
    # 헤더 접근	self.request.headers.get('헤더이름')

    # GET 요청
    # ──────▶ as_view()
    #         └──▶ dispatch()
    #             └──▶ get()
    #                     ├──▶ get_queryset()
    #                     ├──▶ get_context_data()
    #                     └──▶ render_to_response()
    #                         └──▶ return HttpResponse

    # POST 요청
    # ──────▶ as_view()
    #         └──▶ dispatch()
    #             └──▶ post()
    #                     ├──▶ form_valid() / form_invalid()
    #                     └──▶ render_to_response() or redirect()
    #                         └──▶ return HttpResponse



    def dispatch(self, request, *args, **kwargs):
        print("🟠 dispatch 호출됨")
        return super().dispatch(request, *args, **kwargs)

    def get(self, request, *args, **kwargs):
        print("🔵 get 호출됨")
        return super().get(request, *args, **kwargs)


    
    # def render_to_response(self, context, **response_kwargs):
    #     print("######### render_to_response() ##########")


    def get_queryset(self):
        print(" ======== get_queryset ======")
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

      
        # for obj in Bbs.objects.all().values():
        #     print(obj)
        # get_bbs_with_rownum 호출해서 데이터 가져오기
        return get_bbs_with_rownum(offset=offset, limit=limit)

    def get_context_data(self, **kwargs):
        print(" ======== get_context_data ======")
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


#####################################
# 게시판 내용 신규등록
#####################################
class BbsCreateView(CreateView):
    model = Bbs
    form_class = PostForm
    template_name = 'bbs/bbs_writer.html'
    success_url = reverse_lazy('bbs:index')  # 작성 후 이동할 URL

    @transaction.atomic
    def form_valid(self, form):
        print(" form_valid. called #################")
        response = super().form_valid(form)
        print(f" 1 bbs_instance  ####---")
        bbs_instance = self.object
        print(f" 2 bbs_instance  ####---")
        # group_id에 id 복사
        if not bbs_instance.group_id:
            bbs_instance.group_id = bbs_instance.id
            bbs_instance.save()

        # 첨부파일 처리
        files = self.request.FILES.getlist('file')  # name="file"과 일치해야 함
        for f in files:
            BbsFile.objects.create(
                bbs=bbs_instance,
                file=f,
                orig_name=f.name,
            )
       
             
        return response

    def get_context_data(self, **kwargs):
        print(f" ---- get_context_data  #")
        context = super().get_context_data(**kwargs)
        context['is_edit'] = False
        return context


#####################################
# 게시판 내용 수정
#####################################
class BbsUpdateView(UpdateView):
    model = Bbs
    form_class = PostForm
    template_name = 'bbs/bbs_writer.html'
    success_url = reverse_lazy('bbs:index')  # 글 수정 후 이동할 URL


    @transaction.atomic
    def form_valid(self, form):
        print(" BbsUpdateView form_valid. called #################")
        response = super().form_valid(form)
        bbs_instance = self.object
        # group_id에 id 복사
        if not bbs_instance.group_id:
            bbs_instance.group_id = bbs_instance.id
            bbs_instance.save()

        # 첨부파일 처리
        files = self.request.FILES.getlist('file')  # name="file"과 일치해야 함
        for f in files:
            BbsFile.objects.create(
                bbs=bbs_instance,
                file=f,
                orig_name=f.name,
            )
        return response


    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['is_edit'] = True
        return context
    

#####################################
# 게시판 수정을 위한 상세조회
#####################################
class BbsDetailView(DetailView):
    model = Bbs
    template_name = 'bbs/bbs_detail.html'  # 상세보기 템플릿
    context_object_name = 'bbs'             # 템플릿에서 쓸 변수명
    print(" BbsDetailView called ================= ")

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['file_list'] = self.object.files.all()  # related_name='files' 이용
        print(f"file ==={context['file_list']}")
        for f in self.object.files.all():
         print(f.orig_name , f.file)
        return context
    

# from bbs.service.send_push import send_test_push

# class Push:

#     def get_context_data(self, **kwargs):
#         context = super().get_context_data(**kwargs)
#         print(" 호출 했다. ")
#         context['push_result'] = send_test_push(self.request)
#         return context


from bbs.biz.real_estate_price_forecast import engine
class Deeplearing(TemplateView):
    template_name = 'bbs/deeplearing.html'  # 상세보기 템플릿
    # context_object_name = 'bbs'             # 템플릿에서 쓸 변수명
    print(" Deeplearing called ================= ")

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
    
        # print(" Deeplearing Start == ")
        # engine()
        # print(" Deeplearing E == ")
        return context
    

    def post(self, request, *args, **kwargs):
        # POST 요청 처리 로직
        print("Deeplearing POST Start == ")
        retsult = engine()  # 예: 딥러닝 호출
        print("Deeplearing POST End == ")
        return render(request, self.template_name, {'result': retsult}) # 원하는 응답으로 바꿀 수 있음    