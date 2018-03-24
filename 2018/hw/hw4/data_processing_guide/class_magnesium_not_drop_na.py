import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
get_ipython().magic('matplotlib inline')

import os
from copy import deepcopy

from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.feature_selection import SelectFromModel

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score
from sklearn.metrics import roc_curve, precision_recall_curve, average_precision_score, f1_score

try:
    from tqdm import tnrange, tqdm_notebook
    tqdm = True
except ImportError:
    tqdm = False
    pass

from matplotlib import colors as mcolors
from matplotlib import cm
import random

import re

import itertools
from sklearn.utils.multiclass import check_classification_targets

from imblearn.under_sampling import RandomUnderSampler

from scipy.stats.kde import gaussian_kde
from scipy.optimize import brentq
from numpy import linspace

import pickle

import sys
from IPython.core.display import clear_output
def change_output(x):
    clear_output()
    sys.stdout.write(x)
    sys.stdout.flush()

colour = ['#fc977c', '#929292']

class Magnesium(object):
    def __init__(self, file_, fold = "rna-ion-step2/", model = None, colours = ['#fc977c', '#929292'], name = ''):
        '''
            Класс Magnesium загружает данные из файла file_, находящегося в папке fold. Делает dropna (чтобы кас не ругался)
            model - модель, с которой вы хотите работатьб по умолчанию RFC
            Сохраняет переменные:
            data - загруженный DataFrame
            features - признаки
            groups - цепочки
            x - данные для бучения в виде Numpy-матрицы
            y - целевой признак
        '''
        self.filename = file_.split('.csv')[0]
        if model is not None:
            self.model = model            
        else:
            self.model = RandomForestClassifier(n_jobs=-1)
        self.colours = colours
        self.model_name = str(self.model).split('(')[0]
        self.trained_model = None
        change_output('Loading data...')        
        self.data = pd.read_table(fold+file_)
        change_output('Data processing...')  
        if ('DSSR' in self.data.columns):
            self.data.drop('DSSR', axis=1, inplace=True)            
        self.data = self.data.dropna()      
        self.y = deepcopy(np.array(np.matrix(self.data['mg']).flatten().tolist()[0])) 
        self.data_numpy = np.matrix(self.data)
        self.features = list(self.data.columns)
        self.features.remove('pdb_chain')
        self.features.remove('mg')
        self.groups = np.array(self.data['pdb_chain'])
        self.x = np.array(self.data[self.features])
        self.features.append('mg')
        self.xt = None  
        self.y_pred = []
        self.y_prob = []
        self.y_true = []
        self.indexes = []
        self.feature_inds = None
        self.name = name
        
        self.important_features = None
        self.train_score = []
        self.test_score= []
        self.test_roc_auc_score = []
        self.gridsearched_model = None
        self.tresholds = []
        self.prec_rec_data = {'precision':[], 'recall':[]}
        
        self.cnf_plot = []
        self.cnf_normed_plot = []
        self.prec_recall_plot = []
        self.roc_auc_plot = []
        self.probability_density_plot = []
        change_output('Everything is OK. Ready for your experiments!')

        
    def fit_predict(self, n_splits = 3, test_size = 0.3, with_groups = True, model = None,
                    plots = True, plot_splits = [-1], x = None, y = None):
        '''
          fit_predict осуществляет кроссвалидацию по следуюещему пайплайну:
                    если with_groups = True делит выборку с помощью GroupShuffleSplit,
                    если False - StratifiedShuffleSplit. 
                    C биологической точки зрения логичнее делать разбиения с учетом групп, 
                    так как группами являются цепочки => хорошо иметь данные всей цепочки в трейне.
          n_splits, test_size - параметры, окторые передаются GroupShuffleSplit или StratifiedShuffleSplit
          model - можете написать любую свою модель, по дефолту будет брать ту, которая была заложена при создании класса.
          plots - рисовать графики (ROC-AUC, precision-recall, confusion matrix, probability-densities) или нет
          plot_splits - для каких разбиений рисовать графики (разиения нумеруются с 0). По дефолту рисуется для последнего разюиения.
          x , y  - при желании можете подставить свои данные для кросс валидации. По дефолту берет загруженную выборку.

          Возвращает словарь с ключами:
          'test score', 'train score' - f1_score на всех сплитах
          'roc_auc': данные функции sklearn.metrics.roc_curve(y_test, y_prob)
          'prec_rec':[precision, recall, average_precision_score, prec_recall_plot]
          'confusion': [cnf_matrix, cnf_normed, cnf_plot, cnf_normed_plot] (normed - означает нормированная матрица)
          'plots':{'roc_auc', 'prec_recall', 'cnf_normed', 'cnf', 'prob_density'}
        '''
        fpr_tprs = []
        prec_recalls = []
        cnfs = []
        prob_dens_info = []
        
        if model is None:
            self.trained_model = deepcopy(self.model)
        else: 
            self.trained_model = deepcopy(model)
        
        if x is None:
            x = self.x
            y = self.y
        else:
            x = x
            y = y
        gss = GroupShuffleSplit(n_splits = n_splits, test_size = test_size, random_state = 0)
        sss = StratifiedShuffleSplit(n_splits = n_splits, test_size = test_size, random_state = 0)
   
        if with_groups:
            splitted = gss.split(x, y, groups = self.groups)
        else:            
            splitted = sss.split(x, y) 
    
        i = 0
        iterator = tqdm_notebook(splitted, desc = "Splits", leave = True) if tqdm else splitted            
        for train_index, test_index in iterator: 
            x_train = x[train_index]
            y_train = y[train_index]
            x_test = x[test_index]
            y_test = y[test_index]
            self.trained_model.fit(x_train, y_train)
            y_prob = self.trained_model.predict_proba(x_test)[:, 1]
            y_pred = self.trained_model.predict(x_test)
            self.train_score.append(f1_score(y_train, self.trained_model.predict(x_train)))
            self.test_score.append(f1_score(y_test, y_pred))
            
            self.y_prob.append(y_prob)            
            self.y_true.append(y_test)
            self.indexes.append(test_index)
#            treshold, _ = self.plot_probability_density(plots=False)
#            y_pred = [1 if i>=treshold else 0 for i in y_prob]
            self.y_pred.append(y_pred)
            
            i = i + 1
            fpr_tprs.append(roc_curve(y_test, y_prob))
            fpr, tpr, _ = fpr_tprs[-1]
            
            prec_recalls.append(self.prec_recall(y_test, y_prob, True))
            cnfs.append(self.plot_confusion_matrix(y_test, y_pred, True))
            prob_dens_info.append(self.plot_probability_density())
            
            self.roc_auc_plot.append(self.form_plot_string('plt.plot', fpr, tpr, color = self.colours[1], alpha=0.5))
        self.y_data = [y_prob, test_index]
#        print('Portion of sites in test: ', np.sum(y_test == 1)/y_test.shape[0])
#        print('Portion of sites in train: ', np.sum(y_train == 1)/y_train.shape[0])
 
        self.roc_auc_plot.append(self.form_plot_string('plt.legend', loc = 4, fontsize = 12))
        self.roc_auc_plot.append(self.form_plot_string('plt.title', self.model_name + ". ROC curves."))

        plot_splits = [plot_splits] if (type(plot_splits) == int) else plot_splits
        for i in plot_splits:
            fpr, tpr, _ = fpr_tprs[i]
            i = i-2 if i < 0 else i
            self.roc_auc_plot[i] = self.form_plot_string('plt.plot', fpr, tpr, 
                                   color = random.choice(list(mcolors.CSS4_COLORS.keys())), alpha=0.5, 
                                   label = 'Split %d'%i)       


        data = {'test score': self.test_score, 'train score':self.train_score, #'treshold':treshold, 
                'roc_auc':fpr_tprs, 'prec_rec':prec_recalls, 'confusion': cnfs,
               'plots':{'roc_auc': self.roc_auc_plot, 'prec_recall': [i[3] for i in prec_recalls],
                         'cnf_normed': [i[3] for i in cnfs], 'cnf': [i[2] for i in cnfs], 
                         'prob_density': [i[1] for i in prob_dens_info]}}
   
        
        if plots:
            self.show_plots({'roc_auc':self.roc_auc_plot})
            for i in plot_splits:
                data_to_plot = data['plots']
                data_to_plot = {key:value[i] if key != 'roc_auc' else value for key,value in data_to_plot.items()}
                del data_to_plot['roc_auc']                    
                self.show_plots(data_to_plot, suptitle = 'Split %d'%i)
        
#########   Saving the trained model in binary file ######################            
#        if (not os.path.isdir('trained_models')):
#            os.mkdir('trained_models')
            
#        model_name = '%s_depth=%d_leaves=%d_%s_validation'%(re.split("\.|\'", str(self.trained_model.__class__))[-2],
#                                              self.trained_model.__dict__['max_depth'], 
#                                              self.trained_model.__dict__['min_samples_leaf'],
#                                              self.filename)
#        with open("trained_models/"+model_name+".sav", 'wb') as file_to_save:
#            pickle.dump(self.trained_model, file_to_save)                  
        
        return data

    def predict(self, x = None, y = None, model = None, file_ = None, plots = True):      
        '''
           фунцкия класса Magnesium predict позволяет предказать с помощью натренированной модели 
           (по дефолту она берется из класса, но можно передать и извне в параметре model) на данных x, y;
           Оценить качество предсказания и построить графики. 
           Вовзращает словарь, аналогичный fit_predict.
           И по умолчанию рисует графики метрик качества, аналогичные fit_predict.
           
           Ecли y == None, все оценки качества и графики имеют значения None.
        '''
        if file_ is not None:
            data = pd.read_table(file_, sep=',').dropna()   
    
            data = data[~(data['chainlen']>1000)]

            if ('DSSR' in data.columns):
                data.drop('DSSR', axis=1, inplace=True)    

            features = list(deepcopy(data.columns))
            [features.remove(column) for column in ['Id','index', 'pdb_chain', 'mg'] if column in data.columns];
            x = np.array(data[features])
            try:
                y = np.array(data['mg'])
            except: 
                y = None
        trained_model = self.trained_model if model is None else model
        y_prob = trained_model.predict_proba(x)[:, 1]        
        y_pred = trained_model.predict(x)
        if y is not None:
            treshold, prob_dens = self.plot_probability_density(y_prob, y)
        #    y_pred = [1 if i>=treshold else 0 for i in y_prob]
            test_score = f1_score(y, y_pred)
            test_roc_auc_score = roc_auc_score(y, y_prob)
            fpr, tpr, _ = roc_curve(y, y_prob)        
            roc_auc_plot = [self.form_plot_string('plt.plot', fpr, tpr, color = self.colours[0], alpha=0.5, label = '')]
            roc_auc_plot.append(self.form_plot_string('plt.title', self.model_name + ". ROC curves."))
            pr = self.prec_recall(y, y_prob, plots) 
            cnf = self.plot_confusion_matrix(y, y_pred, plots)
        else:
            treshold, prob_dens, test_score, test_roc_auc_score, fpr, tpr, roc_auc_plot  = [None,]*7
            pr, cnf = [[None]*4]*2

        data = {'x': x, 'y': y, 'probability': y_prob, 'prediction': y_pred, 'treshold': treshold, 
                'test_score':test_score, 'roc_auc':[fpr, tpr], 'prec_rec':pr[:-1], 'confusion': cnf, 
                'plots':{'roc_auc': roc_auc_plot, 'prec_recall': pr[3],
                         'cnf_normed': cnf[3], 'cnf': cnf[2], 'prob_density': prob_dens}}
        if plots and y is not None:
            self.show_plots(data['plots'])    
        return data
  
        
    def prec_recall(self,y_test, y_prob, plots):
        precision, recall, treshold = precision_recall_curve(y_test,  y_prob)
        acc = average_precision_score(y_test, y_prob, average="micro")
        if plots:  
            prec_recall_plot = []
            prec_recall_plot.append(self.form_plot_string('plt.scatter', recall, precision, color = self.colours[0]))
            prec_recall_plot.append(self.form_plot_string('plt.plot', recall, precision, color = self.colours[0],
                                                          lw=1, label=self.model_name + ' (area = {0:0.2f})'''.format(acc)))
            prec_recall_plot.append(self.form_plot_string('plt.legend', fontsize = 12))
            prec_recall_plot.append(self.form_plot_string('plt.xlabel', 'Recall'))
            prec_recall_plot.append(self.form_plot_string('plt.ylabel', 'Precision'))
            prec_recall_plot.append(self.form_plot_string('plt.title', "Presicion-recall"))
        return [precision, recall, acc, prec_recall_plot]
    
    def plot_probability_density(self, y_prob = None, y = None, plots = True):
        if y_prob is None:
            y_prob, y = [self.y_prob[-1],self.y_true[-1]] 
        
        kde1 = gaussian_kde(y_prob[y == 1])
        kde2 = gaussian_kde(y_prob[y == 0])
        
        x1 = linspace(np.min(y_prob[y == 1]),np.max(y_prob[y == 1]),500)
        x2 = linspace(np.min(y_prob[y == 0]),np.max(y_prob[y == 0]),500)
        
        try:
            treshold = brentq(lambda x : kde1(x) - kde2(x), x2[np.argmax(kde1(x1))], x1[np.argmax(kde2(x2))])
        except ValueError:
            treshold = 0.5
            
        probability_density_plot = []
        if plots:
            probability_density_plot.append(self.form_plot_string('plt.fill_between',
                                                                       x1,kde1(x1),0, color='darkblue', alpha = 0.5, label = 'Sites'))
            probability_density_plot.append(self.form_plot_string('plt.fill_between',
                                                                      x2,kde2(x2), 0, color='darkgrey', alpha = 0.5, label = 'Non-sites'))
            probability_density_plot.append(self.form_plot_string('plt.axvline',
                                                                      x1[np.argmax(kde1(x1))], color='black', linestyle='--', alpha = 0.5))
            probability_density_plot.append(self.form_plot_string('plt.axvline',
                                                                      x2[np.argmax(kde2(x2))], color='black', linestyle='--', alpha = 0.5))
            probability_density_plot.append(self.form_plot_string('plt.axvline',
                                                                       treshold, color='black', linestyle='-.', alpha = 0.7, label = str(round(treshold,2)))) 
            probability_density_plot.append(self.form_plot_string('plt.xticks', [0, 0.2, 0.4, 0.6, 0.8, 1]))
            probability_density_plot.append(self.form_plot_string('plt.legend'))
            probability_density_plot.append(self.form_plot_string('plt.title', 'Probability Distributions'))
            probability_density_plot.append(self.form_plot_string('plt.xlabel', 'Probabilities'))
        return [treshold,probability_density_plot]

    
    def plot_confusion_matrix_(self, cm, normalize=False, title='Confusion matrix'):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        classes = ['Non-sites', 'Sites']
        general_plot_strings = []
        
        general_plot_strings.append(self.form_plot_string('plt.imshow', cm, interpolation='nearest', cmap="YlGnBu"))
        general_plot_strings.append(self.form_plot_string('plt.colorbar'))
        tick_marks = np.arange(len(classes))
        general_plot_strings.append(self.form_plot_string('plt.xticks', tick_marks, classes, rotation=45))
        general_plot_strings.append(self.form_plot_string('plt.yticks', tick_marks, classes))                               
        general_plot_strings.append(self.form_plot_string('plt.title', title))
        cm_not_normalized = deepcopy(cm)
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        thresh = 2* cm_not_normalized.max() / 3.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            colour="white" if (cm_not_normalized[i, j] > thresh) else "black"
            general_plot_strings.append(self.form_plot_string('plt.text', j, i, round(cm[i, j],2), horizontalalignment="center", color=colour))
        general_plot_strings.append(self.form_plot_string('plt.tight_layout'))
        general_plot_strings.append(self.form_plot_string('plt.ylabel', 'True label'))
        general_plot_strings.append(self.form_plot_string('plt.xlabel', 'Predicted label'))
        return general_plot_strings

                                                    
    def plot_confusion_matrix(self, y_test, y_pred, plots):
    # Compute confusion matrix
        cnf_matrix = confusion_matrix(y_test, y_pred)
        cnf_normed = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
        np.set_printoptions(precision=2)
        
        if plots:
            # Plot non-normalized confusion matrix
            cnf_plot = self.plot_confusion_matrix_(cnf_matrix, title='Confusion matrix, without normalization')

            # Plot normalized confusion matrix
            cnf_normed_plot = self.plot_confusion_matrix_(cnf_matrix, normalize=True,title='Normalized confusion matrix')
        
        return [cnf_matrix, cnf_normed, cnf_plot, cnf_normed_plot]
    
    def form_plot_string(self, type_of_plot, *args, **kwargs):
        arguments = ','.join([str(i.tolist()) if str(type(i)).split('.')[0] == "<class 'numpy" 
                              else ('"'+str(i)+'"' if type(i) == str else str(i)) for i in args])
        properties = ','.join(['='.join([str(name), str(value.tolist())]) if str(type(value)).split('.')[0] == "<class 'numpy"
                               else ('='.join([str(name), '"'+str(value)+'"']) if type(value) == str 
                                     else '='.join([str(name), str(value)]))  for name, value in kwargs.items()])
        return (type_of_plot+'('+arguments+','+properties+')') if (len(arguments) != 0 and len(properties) != 0) else (type_of_plot+'('+arguments+properties+')')
       

                                    
    def show_plots(self, plots, suptitle = None):
        possible_plots = ['roc_auc', 'prec_recall', 'cnf', 'cnf_normed', 'prob_density']
        all_plots = [i for i in possible_plots if i in plots.keys()]
        n_plots = len(all_plots)
        height = 3*n_plots if n_plots > 1 else 5        
        fz = (13, height)
        
        args = []
        ncols = 2 if 'cnf' in all_plots else 1
        if 'cnf' in all_plots:
            n_plots -= 1
        i = 0
        for plot_name in all_plots:
            if ('cnf' == plot_name):
                args.append([(n_plots,ncols), (i,0), 1,1])
            elif ('cnf_normed' == plot_name):
                i -= 1
                args.append([(n_plots,ncols), (i,1), 1, 1])
            else:
                args.append([(n_plots,ncols), (i,0), 1, ncols])
            i += 1
        fig = plt.figure(figsize=fz)
        plt.subplots_adjust(top=0.82)
        if suptitle is not None:
            plt.suptitle(suptitle, fontsize=18, y=1.02)
        for i,plot in enumerate(all_plots):
            ax = plt.subplot2grid(*args[i])       
            [eval(plot_string) for plot_string in plots[plot] if plots[plot] != []]
  
    
def plot_one_plot(plot_elements):
    [eval(plot_string) for plot_string in plot_elements]