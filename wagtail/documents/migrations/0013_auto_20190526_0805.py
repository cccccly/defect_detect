# Generated by Django 2.2.1 on 2019-05-26 08:05

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('wagtaildocs', '0012_auto_20190523_2132'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='document',
            name='GroupID',
        ),
        migrations.AddField(
            model_name='document',
            name='BatchNum',
            field=models.CharField(blank=True, default='请上传说明文档', max_length=255, verbose_name='产品批号'),
        ),
        migrations.AddField(
            model_name='document',
            name='ClothCode',
            field=models.CharField(blank=True, default='000000', max_length=255, verbose_name='布匹编号'),
        ),
        migrations.AddField(
            model_name='document',
            name='Specs',
            field=models.CharField(blank=True, default='请上传说明文档', max_length=255, verbose_name='产品规格'),
        ),
    ]
