from itertools import groupby

from django import forms
from django.contrib.auth.models import Group, Permission
from django.db import transaction
from django.template.loader import render_to_string
from django.utils.translation import ugettext as _

from wagtail.core.models import Collection, CollectionViewRestriction, GroupCollectionPermission

from .view_restrictions import BaseViewRestrictionForm


class CollectionViewRestrictionForm(BaseViewRestrictionForm):

    class Meta:
        model = CollectionViewRestriction
        fields = ('restriction_type', 'password', 'groups')


class CollectionForm(forms.ModelForm):
    class Meta:
        model = Collection
        fields = ('name',)


class BaseCollectionMemberForm(forms.ModelForm):
    """
    Abstract form handler for editing models that belong to a collection,
    such as documents and images. These total are (optionally) instantiated
    with a 'user' kwarg, and take care of populating the 'collection' field's
    choices with the collections the user has permission for, as well as
    hiding the field when only one collection is available.

    Subclasses must define a 'permission_policy' attribute.
    """
    def __init__(self, *args, **kwargs):
        user = kwargs.pop('user', None)

        super().__init__(*args, **kwargs)

        if user is None:
            self.collections = Collection.objects.all()
        else:
            self.collections = (
                self.permission_policy.collections_user_has_permission_for(user, 'add')
            )

        if self.instance.pk:
            # editing an existing document; ensure that the list of available collections
            # includes its current collection
            self.collections = (
                self.collections | Collection.objects.filter(id=self.instance.collection_id)
            )

        if len(self.collections) == 0:
            raise Exception(
                "Cannot construct %s for a user with no collection permissions" % type(self)
            )
        elif len(self.collections) == 1:
            # don't show collection field if only one collection is available
            del self.fields['collection']
        else:
            self.fields['collection'].queryset = self.collections

    def save(self, commit=True):
        if len(self.collections) == 1:
            # populate the instance's collection field with the one available collection
            self.instance.collection = self.collections[0]

        return super().save(commit=commit)


class BaseGroupCollectionMemberPermissionFormSet(forms.BaseFormSet):
    """
    A base formset class for managing GroupCollectionPermissions for a
    model with CollectionMember behaviour. Subclasses should provide attributes:
    permission_types - a list of (codename, short_label, long_label) tuples for the permissions
        being managed here
    permission_queryset - a queryset of Permission objects for the above permissions
    default_prefix - prefix to use on form fields if one is not specified in __init__
    template = template filename
    """
    def __init__(self, data=None, files=None, instance=None, prefix=None):
        if prefix is None:
            prefix = self.default_prefix

        if instance is None:
            instance = Group()

        self.instance = instance

        initial_data = []

        for collection, collection_permissions in groupby(
            instance.collection_permissions.filter(
                permission__in=self.permission_queryset
            ).select_related('permission__content_type', 'collection').order_by('collection'),
            lambda cp: cp.collection
        ):
            initial_data.append({
                'collection': collection,
                'permissions': [cp.permission for cp in collection_permissions]
            })

        super().__init__(
            data, files, initial=initial_data, prefix=prefix
        )
        for form in self.forms:
            form.fields['DELETE'].widget = forms.HiddenInput()

    @property
    def empty_form(self):
        empty_form = super().empty_form
        empty_form.fields['DELETE'].widget = forms.HiddenInput()
        return empty_form

    def clean(self):
        """Checks that no two total refer to the same collection object"""
        if any(self.errors):
            # Don't bother validating the formset unless each form is valid on its own
            return

        collections = [
            form.cleaned_data['collection']
            for form in self.forms
            # need to check for presence of 'collection' in cleaned_data,
            # because a completely blank form passes validation
            if form not in self.deleted_forms and 'collection' in form.cleaned_data
        ]
        if len(set(collections)) != len(collections):
            # collections list contains duplicates
            raise forms.ValidationError(
                _("You cannot have multiple permission records for the same collection.")
            )

    @transaction.atomic
    def save(self):
        if self.instance.pk is None:
            raise Exception(
                "Cannot save a GroupCollectionMemberPermissionFormSet "
                "for an unsaved group instance"
            )

        # get a set of (collection, permission) tuples for all ticked permissions
        forms_to_save = [
            form for form in self.forms
            if form not in self.deleted_forms and 'collection' in form.cleaned_data
        ]

        final_permission_records = set()
        for form in forms_to_save:
            for permission in form.cleaned_data['permissions']:
                final_permission_records.add((form.cleaned_data['collection'], permission))

        # fetch the group's existing collection permission records for this model,
        # and from that, build a list of records to be created / deleted
        permission_ids_to_delete = []
        permission_records_to_keep = set()

        for cp in self.instance.collection_permissions.filter(
            permission__in=self.permission_queryset,
        ):
            if (cp.collection, cp.permission) in final_permission_records:
                permission_records_to_keep.add((cp.collection, cp.permission))
            else:
                permission_ids_to_delete.append(cp.id)

        self.instance.collection_permissions.filter(id__in=permission_ids_to_delete).delete()

        permissions_to_add = final_permission_records - permission_records_to_keep
        GroupCollectionPermission.objects.bulk_create([
            GroupCollectionPermission(
                group=self.instance, collection=collection, permission=permission
            )
            for (collection, permission) in permissions_to_add
        ])

    def as_admin_panel(self):
        return render_to_string(
            self.template,
            {'formset': self},
        )


def collection_member_permission_formset_factory(
    model, permission_types, template, default_prefix=None
):

    permission_queryset = Permission.objects.filter(
        content_type__app_label=model._meta.app_label,
        codename__in=[codename for codename, short_label, long_label in permission_types]
    ).select_related('content_type')

    if default_prefix is None:
        default_prefix = '%s_permissions' % model._meta.model_name

    class CollectionMemberPermissionsForm(forms.Form):
        """
        For a given model with CollectionMember behaviour,
        defines the permissions that are assigned to an entity
        (i.e. group or user) for a specific collection
        """
        collection = forms.ModelChoiceField(
            queryset=Collection.objects.all().prefetch_related('group_permissions')
        )
        permissions = forms.ModelMultipleChoiceField(
            queryset=permission_queryset,
            required=False,
            widget=forms.CheckboxSelectMultiple
        )

    GroupCollectionMemberPermissionFormSet = type(
        str('GroupCollectionMemberPermissionFormSet'),
        (BaseGroupCollectionMemberPermissionFormSet, ),
        {
            'permission_types': permission_types,
            'permission_queryset': permission_queryset,
            'default_prefix': default_prefix,
            'template': template,
        }
    )

    return forms.formset_factory(
        CollectionMemberPermissionsForm,
        formset=GroupCollectionMemberPermissionFormSet,
        extra=0,
        can_delete=True
    )
