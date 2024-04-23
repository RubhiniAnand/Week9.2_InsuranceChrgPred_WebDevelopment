from django import forms
from .models import *


class insForm(forms.ModelForm):
    class Meta():
        model=insModel
        fields=['age','bmi','children','sex_male','smoker_yes']
