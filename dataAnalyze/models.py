from django.db import models

# Create your models here.


from django.db import models


class StaticsDetail(models.Model):
    ClothCode = models.CharField(max_length=255, verbose_name="布匹编号", blank=True, default="000000")
    BatchNum = models.CharField(max_length=255, verbose_name="产品批号", blank=True, default="请上传说明文档")
    Specs = models.CharField(max_length=255, verbose_name="产品规格", blank=True, default="请上传说明文档")
    CountAll = models.BigIntegerField(verbose_name="总数", default=0)
    DefectCount = models.BigIntegerField(verbose_name="次品数", default=0)
    YW = models.BigIntegerField(verbose_name="油污", default=0)
    JB = models.BigIntegerField(verbose_name="浆斑", default=0)
    TCHJ = models.BigIntegerField(verbose_name="停车痕（紧）", default=0)
    TCHS = models.BigIntegerField(verbose_name="停车痕（松）", default=0)
    BW = models.BigIntegerField(verbose_name="并纬", default=0)
    CB = models.BigIntegerField(verbose_name="擦白", default=0)
    CS = models.BigIntegerField(verbose_name="擦伤", default=0)
    CW = models.BigIntegerField(verbose_name="糙纬", default=0)
    CH = models.BigIntegerField(verbose_name="错花", default=0)
    DJ1 = models.BigIntegerField(verbose_name="断经1", default=0)
    DJ2 = models.BigIntegerField(verbose_name="断经2", default=0)
    DW = models.BigIntegerField(verbose_name="断纬", default=0)
    JJ = models.BigIntegerField(verbose_name="尽机", default=0)
    JT = models.BigIntegerField(verbose_name="经条", default=0)
    KZ = models.BigIntegerField(verbose_name="空织", default=0)
    QJ = models.BigIntegerField(verbose_name="起机", default=0)
    QW1 = models.BigIntegerField(verbose_name="缺纬1", default=0)
    QW2 = models.BigIntegerField(verbose_name="缺纬2", default=0)
    SW = models.BigIntegerField(verbose_name="缩纬", default=0)
    ZF = models.BigIntegerField(verbose_name="折返", default=0)
    QT = models.BigIntegerField(verbose_name="其他", default=0)