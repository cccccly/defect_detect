from wagtail.contrib.modeladmin.options import (
    ModelAdmin, modeladmin_register)
from .models import CustomImage
from wagtail.core import hooks
from django.conf.urls import include, url
from django.urls import reverse
from wagtail.admin.menu import MenuItem,SubmenuMenuItem

# @hooks.register('register_admin_urls')
# def register_admin_urls():
#     return [
#         url(r'^images/', include(urls, namespace='detect')),
#     ]

#shangchuan
@hooks.register('register_admin_menu_item')
def register_images_menu_item():
    return MenuItem(
        ('开始检测'), reverse('index'),
        name='images', classnames='icon icon-search', order=300
    )


# class BookAdmin(ModelAdmin):
#     model = CustomImage
#     menu_label = 'Book'  # ditch this to use verbose_name_plural from model
#     menu_icon = 'pilcrow'  # change as required
#     menu_order = 700  # will put in 3rd place (000 being 1st, 100 2nd)
#     add_to_settings_menu = False  # or True to add your model to the Settings sub-menu
#     exclude_from_explorer = False # or True to exclude pages of this type from Wagtail's explorer view
#     # list_display = ('title',)
#     # list_filter = ('title',)
#     # search_fields = ('title', 'author')
#
# # Now you just need to register your customised ModelAdmin class with Wagtail
# modeladmin_register(BookAdmin)