from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse, HttpRequest
from django.shortcuts import render, redirect
#from .forms import *
from django.contrib import messages
from django.shortcuts import render
from django.urls import reverse_lazy
from django.urls import reverse
from django.http import HttpResponse
from django.views.generic import (View,TemplateView,
ListView,DetailView,
CreateView,DeleteView,
UpdateView)
from . import models
from .forms import *
from django.core.files.storage import FileSystemStorage
#from topicApp.Topicfun import Topic
#from ckdApp.funckd import ckd
from sklearn.tree import export_graphviz #plot tree
from sklearn.metrics import roc_curve, auc #for model evaluation
#from sklearn.metrics import classification_report #for model evaluation
from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(df2.drop('classification_yes', 1), df2['classification_yes'], test_size = .2, random_state=10)

import time
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
#from sklearn.feature_selection import SelectKBest
#from sklearn.feature_selection import chi2
#from sklearn.model_selection import train_test_split
#from sklearn.decomposition import PCA
#from sklearn.feature_selection import RFE
#from sklearn.linear_model import LogisticRegression
import pickle
import matplotlib.pyplot as plt
#import eli5 #for purmutation importance
#from eli5.sklearn import PermutationImportance
#import shap #for SHAP values
#from pdpbox import pdp, info_plots #for partial plots
np.random.seed(123) #ensure reproduc
class dataUploadView(View):
    form_class = insForm
    success_url = reverse_lazy('success')
    template_name = 'create.html'
    failure_url= reverse_lazy('fail')
    filenot_url= reverse_lazy('filenot')
    def get(self, request, *args, **kwargs):
        form = self.form_class()
        return render(request, self.template_name, {'form': form})
    def post(self, request, *args, **kwargs):
        #print('inside post')
        form = self.form_class(request.POST, request.FILES)
        #print('inside form')
        if form.is_valid():
            form.save()
            data_ag=request.POST.get('age')
            data_bm=request.POST.get('bmi')
            data_ch=request.POST.get('children')
            data_sm=request.POST.get('sex_male')
            data_sy=request.POST.get('smoker_yes')
            #print (data)
            import pandas as pd
            dataset=pd.read_csv("insurance_pre.csv")
            #dataset
            dataset=pd.get_dummies(dataset,drop_first=True)
            #dataset
            independent=dataset[['age', 'bmi', 'children','sex_male', 'smoker_yes']]
            dependent=dataset[['charges']]
            #from sklearn.model_selection import train_test_split
            X_train,X_test,Y_train,Y_test=train_test_split(independent,dependent,test_size=0.30,random_state=0)
            from sklearn.ensemble import RandomForestRegressor
            regressor=RandomForestRegressor(criterion="poisson",n_estimators=100,random_state=0)
            regressor.fit(X_train,Y_train)
            y_pred=regressor.predict(X_test)
            from sklearn.metrics import r2_score
            r_score=r2_score(Y_test,y_pred)
            r_score
            import pickle
            filename = 'finalized_model_RF.sav'
            classifier = pickle.load(open(filename, 'rb'))

            data =np.array([data_ag,data_bm,data_ch,data_sm,data_sy])
            #sc = StandardScaler()
            #data = sc.fit_transform(data.reshape(-1,1))
            out=classifier.predict(data.reshape(1,-1))
# providing an indexfrom django.shortcuts import render
            return render(request, "succ_msg.html", {'data_ag':data_ag,'data_bm':data_bm,'data_ch':data_ch,'data_sm':data_sm,'data_sy':data_sy,
                                                        'out':out})


        else:
            return redirect(self.failure_url)

# Create your views here.
