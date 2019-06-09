import os

from django.core.exceptions import PermissionDenied
from django.http import HttpResponseBadRequest, JsonResponse
from django.shortcuts import get_object_or_404, render
from django.template.loader import render_to_string
from django.utils.encoding import force_text
from django.views.decorators.http import require_POST
from django.views.decorators.vary import vary_on_headers

from wagtail.admin.utils import PermissionPolicyChecker
from wagtail.core.models import Collection
from wagtail.search.backends import get_search_backends

from ..forms import get_document_form, get_document_multi_form
from ..models import get_document_model
from ..permissions import permission_policy

permission_checker = PermissionPolicyChecker(permission_policy)



@permission_checker.require('add')
@vary_on_headers('X-Requested-With')
def add(request):
    # 定义字典确定产品规格
    dic1 = {'L': "临时品种", 'X': "打样品种", 'C': "常规品种"}
    dic2 = {'N': "尼丝纺", 'T': "塔丝隆", 'P': "春亚纺", 'S': "桃皮绒",
            'J': "锦涤纺", 'R': "麂皮绒", 'D': "涤塔夫", 'Q': "其它品种"}
    dic3 = {'T': "平纹", 'W': "斜纹", 'B': "格子", 'S': "缎纹"}
    # 路径
    mediaPath = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))) + '/media/documents'
    Document = get_document_model()
    DocumentForm = get_document_form(Document)
    DocumentMultiForm = get_document_multi_form(Document)

    collections = permission_policy.collections_user_has_permission_for(request.user, 'add')
    if len(collections) > 1:
        collections_to_choose = Collection.order_for_display(collections)
    else:
        # no need to show a collections chooser
        collections_to_choose = None

    if request.method == 'POST':
        if not request.is_ajax():
            return HttpResponseBadRequest("Cannot POST to this view without AJAX")

        if not request.FILES:
            return HttpResponseBadRequest("Must upload a file")

        # Build a form for validation
        print(request.FILES['files[]'].name)

        form = DocumentForm({
            'title': request.FILES['files[]'].name,
            'collection': request.POST.get('collection'),
        }, {
            'file': request.FILES['files[]']
        }, user=request.user)

        if form.is_valid():
            # Save it
            print(333)
            doc = form.save(commit=False)
            doc.uploaded_by_user = request.user
            doc.file_size = doc.file.size

            # Set new document file hash
            doc.file.seek(0)
            doc._set_file_hash(doc.file.read())
            doc.file.seek(0)

            # set BatchNum,Specs

            doc.file.readline()
            doc.Info = str(doc.file.readline())
            line = doc.Info
            # 提取line中的产品规格及产品批号
            specs = dic1.get(line[2], "XXXX") + "--" + dic2.get(line[3], "XXXX") + "--" + dic3.get(line[4], "XXXX")
            batchNum = line[5:11]
            clothCode = str(request.FILES['files[]'])[0:6]
            doc.BatchNum = batchNum
            doc.Specs = specs
            doc.ClothCode = clothCode
            doc.save()


            # Success! Send back an edit form for this document to the user
            return JsonResponse({
                'success': True,
                'doc_id': int(doc.id),
                'form': render_to_string('wagtaildocs/multiple/edit_form.html', {
                    'doc': doc,
                    'form': DocumentMultiForm(
                        instance=doc, prefix='doc-%d' % doc.id, user=request.user
                    ),
                }, request=request),
            })
        else:
            # Validation error
            print(222)
            return JsonResponse({
                'success': False,

                # https://github.com/django/django/blob/stable/1.6.x/django/forms/util.py#L45
                'error_message': '\n'.join(['\n'.join([force_text(i) for i in v]) for k, v in form.errors.items()]),
            })


    else:
        form = DocumentForm(user=request.user)

    return render(request, 'wagtaildocs/multiple/add.html', {
        'help_text': form.fields['file'].help_text,
        'collections': collections_to_choose,
    })


@require_POST
def edit(request, doc_id, callback=None):

    Document = get_document_model()
    DocumentMultiForm = get_document_multi_form(Document)

    doc = get_object_or_404(Document, id=doc_id)

    if not request.is_ajax():
        return HttpResponseBadRequest("Cannot POST to this view without AJAX")

    if not permission_policy.user_has_permission_for_instance(request.user, 'change', doc):
        raise PermissionDenied

    form = DocumentMultiForm(
        request.POST, request.FILES, instance=doc, prefix='doc-' + doc_id, user=request.user
    )

    if form.is_valid():
        form.save()

        # Reindex the doc to make sure all tags are indexed
        for backend in get_search_backends():
            backend.add(doc)

        return JsonResponse({
            'success': True,
            'doc_id': int(doc_id),
        })
    else:
        return JsonResponse({
            'success': False,
            'doc_id': int(doc_id),
            'form': render_to_string('wagtaildocs/multiple/edit_form.html', {
                'doc': doc,
                'form': form,
            }, request=request),
        })


@require_POST
def delete(request, doc_id):
    Document = get_document_model()

    doc = get_object_or_404(Document, id=doc_id)

    if not request.is_ajax():
        return HttpResponseBadRequest("Cannot POST to this view without AJAX")

    if not permission_policy.user_has_permission_for_instance(request.user, 'delete', doc):
        raise PermissionDenied

    doc.delete()

    return JsonResponse({
        'success': True,
        'doc_id': int(doc_id),
    })
