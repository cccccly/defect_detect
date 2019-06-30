from wagtail.contrib import forms
from wagtail.admin import widgets
from django import forms


class StatisticsForm(forms.Form):
    batchNum = forms.CharField(label='批次', max_length=100)
    dateTime = forms.DateTimeField(label="日期", widget=widgets.AdminDateTimeInput)
