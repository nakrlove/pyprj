from django import forms
from .models import Bbs

class PostForm(forms.ModelForm):
    class Meta:
        model = Bbs
        fields = ['title', 'content', 'writer']  # 사용하는 필드 지정