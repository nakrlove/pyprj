from bbs.bbsmodels import Bbs
# from django.db import models
from django.views.generic import ListView

class BbsLV(ListView):
    model = Bbs