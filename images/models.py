from django.db import models

# Create your models here.
from django.db import models

from wagtail.images.models import Image, AbstractImage, AbstractRendition
from wagtail.documents.models import AbstractDocument


class Cloth(models.Model):
    ClothCode = models.CharField(max_length=255, verbose_name="布匹编号", blank=True, default="000000")
    BatchNum = models.CharField(max_length=255, verbose_name="产品批号", blank=True, default="请上传说明文档")
    Specs = models.CharField(max_length=255, verbose_name="产品规格", blank=True, default="请上传说明文档")
    BatchNum = models.CharField(max_length=255, verbose_name="产品批号", blank=True, default="请上传说明文档")


class CustomImage(AbstractImage):
    # Add any extra fields to image here

    # eg. To add a caption field:
    # caption = models.CharField(max_length=255, blank=True)
    YW = "油污"
    JB = "浆斑"
    TCHJ = "停车痕（紧）"
    TCHS = "停车痕（松）"
    BW = "并纬"
    CB = "擦白"
    CS = "擦伤"
    CW = "糙纬"
    CH = "错花"
    DJ1 = "断经1"
    DJ2 = "断经2"
    DW = "断纬"
    JJ = "尽机"
    JT = "经条"
    KZ = "空织"
    QJ = "起机"
    QW1 = "缺纬1"
    QW2 = "缺纬2"
    SW = "缩纬"
    ZF = "折返"
    QT = "其他"


    DefectType_Choices = [
        (None, "无瑕疵"),
        (YW, "油污"),
        (JB, "浆斑"),
        (TCHJ, "停车痕（紧）"),
        (TCHS, "停车痕（松）"),
        (BW, "并纬"),
        (CB, "擦白"),
        (CS, "擦伤"),
        (CW, "糙纬"),
        (CH, "错花"),
        (DJ1, "断经1"),
        (DJ2, "断经2"),
        (DW, "断纬"),
        (JJ, "尽机"),
        (JT, "经条"),
        (KZ, "空织"),
        (QJ, "起机"),
        (QW1, "缺纬1"),
        (QW2, "缺纬2"),
        (SW, "缩纬"),
        (ZF, "折返"),
        (QT, "其他"),
    ]

    HasDefect = models.BooleanField(verbose_name="有无瑕疵", blank=True, default=False)
    DefectType = models.CharField(max_length=255, verbose_name="瑕疵类别", blank=True,
                                  choices=DefectType_Choices)
    ExtensionRatio = models.FloatField(verbose_name="伸长比", blank=True, default=1)
    IsDetect = models.BooleanField(verbose_name="是否检测", blank=True, default=False)
    ClothCode = models.CharField(max_length=255, verbose_name="布匹编号", blank=True, default="000000")
    BatchNum = models.CharField(max_length=255, verbose_name="产品批号", blank=True, default="请上传说明文档")
    Specs = models.CharField(max_length=255, verbose_name="产品规格", blank=True, default="请上传说明文档")

    admin_form_fields = Image.admin_form_fields + (
        # Then add the field names here to make them appear in the form:
        # 'caption',
        'BatchNum',
        'Specs',
        'HasDefect',
        'DefectType',
    )


class CustomRendition(AbstractRendition):
    image = models.ForeignKey(CustomImage, on_delete=models.CASCADE, related_name='renditions')

    class Meta:
        unique_together = (
            ('image', 'filter_spec', 'focal_point_key'),
        )


class CustomDocument(AbstractDocument):
    admin_form_fields = Image.admin_form_fields + (

    )


