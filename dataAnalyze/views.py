from io import *
import os

from django.shortcuts import render

# Create your views here.
from images.models import CustomImage
from django.http import HttpResponse
from xlwt import *
from django.utils.encoding import escape_uri_path


def excel_export(request):
    """
    导出excel表格
    """
    batchNum = request.GET['BatchNum']
    list_obj = CustomImage.objects.filter(BatchNum=batchNum).order_by("created_at")
    if list_obj:
        # 创建工作薄
        ws = Workbook(encoding='utf-8')
        w = ws.add_sheet(u"数据报表第一页")
        w.write(0, 0, u"图片名")
        w.write(0, 1, u"产品规格")
        w.write(0, 2, u"有无瑕疵")
        w.write(0, 3, u"瑕疵类型")
        w.write(0, 4, u"疵点坐标")
        # 写入数据
        excel_row = 1
        for obj in list_obj:
            location = '(' + str(obj.focal_point_x) + ',' + str(obj.focal_point_y) + ')'
            w.write(excel_row, 0, obj.title)
            w.write(excel_row, 1, obj.Specs)
            w.write(excel_row, 2, obj.HasDefect)
            w.write(excel_row, 3, obj.DefectType)
            w.write(excel_row, 4, location)
            excel_row += 1
        # 检测文件是够存在
        # 方框中代码是保存本地文件使用，如不需要请删除该代码
        fileName = '第' + str(batchNum) + '批布匹检测报表.xls'

        sio = BytesIO()
        ws.save(sio)
        sio.seek(0)
        response = HttpResponse(sio.getvalue(), content_type='application/vnd.ms-excel')
        response['Content-Disposition'] = "attachment; filename*=utf-8''{}".format(escape_uri_path(fileName))
        response.write(sio.getvalue())
        return response
