from django.db import models

# Create your models here.
class Bbs(models.Model):
    POST = 'POST'
    COMMENT = 'COMMENT'

    TYPE_CHOICES = [
        (POST, '게시글'),
        (COMMENT, '댓글'),
    ]

    type = models.CharField(
        max_length=10,
        choices=TYPE_CHOICES,
        default=POST,
        verbose_name='타입'
    )
    title = models.CharField(
        max_length=200,
        null=True,
        blank=True,
        verbose_name='제목'
    )
    
    content = models.TextField(verbose_name='본문')
    writer = models.CharField(max_length=100, verbose_name='작성자')
    created_at = models.DateTimeField(auto_now_add=True, verbose_name='작성일')
    updated_at = models.DateTimeField(auto_now=True, verbose_name='수정일')
    is_deleted = models.BooleanField(default=False, verbose_name='삭제 여부')

    group_id = models.IntegerField(
        null=True,
        blank=True,
        db_index=True,
        verbose_name='그룹 ID'
    )
    parent = models.ForeignKey(
        'self',
        null=True,
        blank=True,
        on_delete=models.CASCADE,
        related_name='replies',
        verbose_name='부모 댓글'
    )
    depth = models.PositiveIntegerField(default=0, verbose_name='댓글 깊이')

    # db_table	실제 DB에 생성될 테이블 이름 지정
    # ordering	기본 정렬 순서 지정 (예: ['-created_at'])
    # verbose_name	관리자 화면에서 모델의 단수 이름 지정
    # verbose_name_plural	복수 이름 지정
    # managed	True면 Django가 테이블 생성/삭제를 관리함, False면 관리 안 함 (기존 테이블 사용 시 필수)
    # app_label	이 모델이 어떤 앱에 속해 있는지 명시 (모델이 models.py 외부에 있을 경우 필수)
    class Meta:
        verbose_name = '게시글 및 댓글'
        verbose_name_plural = '게시글 및 댓글'
        ordering = ['group_id', 'created_at']
        app_label = 'bbs'           # 
        db_table = 'bbs_bbs'        # MySQL 테이블명과 일치하게
        managed = False             # 이미 MySQL에 존재한다면 꼭 필요

    def __str__(self):
        return f"[{self.get_type_display()}] {self.title or self.content[:30]}"

