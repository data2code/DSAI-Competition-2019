#!/usr/bin/env python
import sys
sys.path.insert(0,'./lib')
import pandas as pd
import numpy as np
import util
import os
import shutil
from shutil import copyfile
from pprint import pprint, pformat
import re
import copy

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from xgboost import XGBClassifier, XGBRFClassifier, XGBRegressor
from xgboost import plot_importance
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import log_loss, f1_score, accuracy_score, r2_score, make_scorer, confusion_matrix, precision_recall_fscore_support
from scipy.optimize import minimize_scalar, minimize
from sklearn.model_selection import PredefinedSplit, GridSearchCV, ParameterGrid, StratifiedKFold
import gzip
import pickle
import datetime as dt
import scipy.stats as ss

from configjson import CONFIG
import json
#import impute

import sklearn.feature_extraction.text as fet
from nltk import word_tokenize
from nltk import corpus
from nltk.stem import WordNetLemmatizer

import warnings
warnings.simplefilter("ignore")

import itertools

SEED=int(CONFIG.get('Seed',42))
np.random.seed(SEED)

if os.environ.get('NLTK_DATA')!='./nltk_data':
    util.error_msg('need to first run:  export NLTK_DATA="./nltk_data" ')

EVAL_ENV='EVAL'
if os.getenv("AICROWD_TEST_DATA_PATH") is None:
    EVAL_ENV='AICROWD'

import argparse as arg
opt=arg.ArgumentParser(description='Phase2-Approval Prediction')
opt.add_argument('-p','--hyperparameter', default=False, action='store_true', help='hyperparameter tuning')
opt.add_argument('-e','--estimator', default=False, action='store_true', help='estimator tuning')
opt.add_argument('-i','--impute_model', default=False, action='store_true', help='impute RF model')
opt.add_argument('-r','--recreate', default=False, action='store_true', help='recreate feature matrix')
opt.add_argument('-d','--debug', default=False, action='store_true', help='recreate feature matrix')
opt.add_argument('-s','--sort', default=False, action='store_true', help='sort features')
opt.add_argument('-k','--stacking', default=False, action='store_true', help='stacking')
args=opt.parse_args()

TAG=json.dumps(CONFIG)
TAG=re.sub('[{} "]', '', re.sub('false', '0', re.sub('true', '1', TAG)))
TAG=re.sub(',Feature.*','',TAG)

print(">>>>>>>>>TAG:", TAG, ">>", EVAL_ENV)

if EVAL_ENV=='EVAL':
    OUTPUT="."
else:
    OUTPUT=os.path.join('tags', re.sub('-', 'm', re.sub(',', '_', re.sub(r'[a-z:]', '', TAG))))
    os.makedirs(OUTPUT, exist_ok=True)
    copyfile('configjson.py', os.path.join(OUTPUT, 'configjson.py'))
print(">>>>>>>>>Folder:", OUTPUT)

if EVAL_ENV=='EVAL':
    # in docker eval environment
    TRAIN_FILE= os.path.join(os.getenv("AICROWD_TRAIN_DATA_PATH"),'training_data_2015_split_on_outcome.csv')
    TEST_FILE= os.getenv("AICROWD_TEST_DATA_PATH")
    SUBMISSION_FILE=os.getenv("AICROWD_PREDICTIONS_OUTPUT_PATH")
    RECREATE=True
    CORE=4
elif EVAL_ENV=='AICROWD':
    ###RECREATE=not os.path.exists(os.path.join(OUTPUT, 'feature_matrix.pkl.gz'))
    RECREATE=False
    TRAIN_FILE='/shared_data/data/training_data/training_data_2015_split_on_outcome.csv'
    TEST_FILE='/shared_data/data/test_data_sample/testing_phase2_release_small.csv'
    if not os.path.exists(TEST_FILE):
        TEST_FILE = 'testing_phase2_release.csv'
        #TEST_FILE = 'testing_phase2_release_small.csv'
    if not os.path.exists(TRAIN_FILE):
        TRAIN_FILE = 'training_data_2015_split_on_outcome.csv'
    SUBMISSION_FILE=os.path.join(OUTPUT, "submit.csv")
    EVAL_ENV=False
    CORE=36
#XXXXXXX

#CONFIG={'RemoveTrivialDrugs':True,'RemoveTerminated':False,'SetTerminatedToZero':False,'Model':'XGB','Validate':'2014Only'}
LOWER_BOUND=0.01
UPPER_BOUND=0.99
###TARGET_RATIO=0.15
TARGET_RATIO=float(CONFIG.get('TargetRatio',0.15))
FIX_20182019=CONFIG['FixYear']
DRUGKEY_AUGMENTATION=CONFIG['DrugindiMax']
WEIGHT_BY_DRUGKEY=True
WEIGHT_BY_OUTCOME=False
DRUGKEYS=['drugkey','indicationkey']

# Some drugs appear often in the training set
TRIVIAL_DRUGS=[1627, 1798, 1803, 4232, 5440, 7350, 11435, 14311, 15034, 15816, 18087, 19577, 23997, 26475, 28941, 34480, 111814]
TRIVIAL_OUTCOME={1627: 0.0, 1798: 0.0, 1803: 0.0, 4232: 1.0, 5440: 0.0, 7350: 1.0, 11435: 0.0, 13517: 0.9316239316239316, 14236: 0.07692307692307693, 14311: 0.0, 15034: 0.0, 15816: 1.0, 18087: 0.029850746268656716, 19577: 0.02857142857142857, 23997: 1.0, 26475: 0.0, 28941: 0.0, 34480: 0.0, 111814: 1.0}
#(40, 26, 26, 30, 21, 103, 26, 117, 26, 36, 25, 32, 67, 35, 36, 44, 25, 34, 21)

REMOVE_TRIVIAL_DRUGS=CONFIG['RemoveTrivialDrugs']
REMOVE_TERMINATED=CONFIG['RemoveTerminated']
FIX_TRIVIAL_DRUGS=False
SET_TERMINATED_TO_ZERO=CONFIG['SetTerminatedToZero']
MODEL_SELECTION=CONFIG['Model']
VALIDATION_MODE=CONFIG['Validate']
SAMPLE_WEIGHT=CONFIG['SampleWeight']
# one: 1, sqrt: 1/sqrt(n), inverse: 1/n
if SAMPLE_WEIGHT not in ('ONE','SQRT', 'INV', 'DIONE', 'DISQRT', 'DIINV'):
    util.error_msg('Unsupported SAMPLE_WEIGHT!')

# We currently don't know how to handle imputing other than MEAN
# RF approach can be applied to train, but tricky for test
IMPUTE=CONFIG['Impute']
HYPERPARAMETER=False #CONFIG['Hyper']
SCALE_POS_WEIGHT=CONFIG['PosScale']
if EVAL_ENV=='EVAL':
    HYPERPARAMETER=False

HYPERPARAMETER=args.hyperparameter
ONLY_TUNE_ESTIMATOR=args.estimator
IMPUTE_MODEL=args.impute_model
if EVAL_ENV=='EVAL':
    HYPERPARAMETER=ONLY_TUNE_ESTIMATOR=False
    IMPUTE_MODEL=False
elif ONLY_TUNE_ESTIMATOR:
    HYPERPARAMETER=ONLY_TUNE_ESTIMATOR=True
if IMPUTE_MODEL: RECREATE=True
if args.recreate or args.debug: RECREATE=True
if HYPERPARAMETER:
    if ONLY_TUNE_ESTIMATOR:
        if  MODEL_SELECTION in ('XGB'):
            if not os.path.exists('hyperparameter.json'):
                util.error_msg('Missing: hyperparameter.json')
    else:
        if not os.path.exists(os.path.join(OUTPUT, 'hyperparameter.range.json')):
            if MODEL_SELECTION in ('XGB'):
                copyfile("params/hyperparameter.range.json", os.path.join(OUTPUT, 'hyperparameter.range.json'))
            elif MODEL_SELECTION=='RF':
                copyfile("params/hyperparameter.range.rf.json", os.path.join(OUTPUT, 'hyperparameter.range.json'))

def myToken(s): return re.split('[|,]',s)
 

class Dump:
    """Utility for save/load object to/from a pickle file, the file is gzipped
    """

    @staticmethod
    def save(obj, pkl_filename):
        with gzip.open(pkl_filename, 'wb') as f:
            pickle.dump(obj, f)

    @staticmethod
    def load(pkl_filename):
        if not os.path.exists(pkl_filename):
            util.error_msg('File not exist: '+pkl_filename)
        with gzip.open(pkl_filename, 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def load_json(json_filename):
        if not os.path.exists(json_filename):
            util.error_msg('File not exist: '+json_filename)
        with open(json_filename) as f:
            data=json.load(f)
        return data

    @staticmethod
    def save_json(data, json_filename):
        with open(json_filename, "w") as f:
            data=json.dump(data, f)
        return data

class EDA:

    def __init__(self, impute=IMPUTE, recreate=True):
        self.EPS=1e-6
        self.QUANTILE =0.25 # use to normalize ratio
        self.IMPUTE=impute
        # impute can be MEAN (or median), ML (machine learning method)
        # missing value exists in duration, age, accrual columns

        # target or text columns no need for EDA plotting
        self.s_out=f'M{MODEL_SELECTION}_V{VALIDATION_MODE}_SW{SAMPLE_WEIGHT}_I{IMPUTE}_SD{SEED}'
        self.OUTCOME = 'outcome'
        self.YEAR='intphaseendyear'
        self.NEW_YEAR='YEAR'

        if not recreate:
            self.train, self.test, param=Dump.load(os.path.join(OUTPUT, 'feature_matrix.pkl.gz'))
            self.feature_cols=param['feature_cols']
            self.filter_datasets()
            self.ratio=param['ratio']
            self.ratio_in_terminate=param['ratio_in_terminate']
            print("Ratio:", self.ratio, "Ratio in Terminated:", self.ratio_in_terminate, "Features:", len(self.feature_cols))
            print("Number of Features:", len(self.feature_cols))
            return
        self.train= pd.read_csv(TRAIN_FILE)

        # add TRAIN column, as we may need to merge train+test and split them for feature engineering
        # (requiring all historical data)
        self.test=pd.read_csv(TEST_FILE)
        print('# in test set:',len(self.test))

        self.train.rename2({k:k.lower() for k in self.train.header()})
        self.test.rename2({k:k.lower() for k in self.test.header()})
        self.train['TRAIN']=True
        self.test['TRAIN']=False
        self.train['test_idx']=-1
        self.test['test_idx']=self.test.index.values

        # fix years first!
        self.fix_phase_year(self.YEAR, self.NEW_YEAR)

        if False: # identify problematic columns
            for s in self.train.header():
                if s!=self.OUTCOME and not s in self.test.header():
                    print("EDA> Delete feature not in test:", s)
                    self.drop_col(s)
                elif np.all(self.train[s].isnull()):
                    #
                    print("EDA> Delete feature of all NaN:", s)
                    self.drop_col(s)
                elif not np.any(self.train[s].isnull()) and self.train[s].nunique()==1:
                    # DrugCountryName, looks like a bug
                    if s=='TRAIN': continue
                    print("EDA> Delete feature of unique value:", s, self.train.loc[0, s])
                    self.drop_col(s)
                elif s=='unnamed: 0':
                    if 'unnamed: 0' in self.train.header():
                        self.train.drop('unnamed: 0', axis=1, inplace=True)
                    if 'unnamed: 0' in self.test.header():
                        self.test.drop('unnamed: 0', axis=1, inplace=True)

        if EVAL_ENV=='EVAL':
            self.test[self.OUTCOME]=0
        elif self.OUTCOME not in self.test.header():
            self.test[self.OUTCOME]=0

        #NEED TO MAKE SURE TRAIN AND TEST HAVE THE SAME COLUMNS IN THE SAME ORDER
        COLUMNS=sorted(['YEAR', 'drugkey', 'indicationkey', 'outcome', 'intduration', 'inttargetaccrual', 'intactualaccrual', 'intidentifiedsites', 'intsponsorid_approval_drugkey_indicationkey', 'intsponsorid_failure_drugkey_indicationkey', 'intpersonid_approval_drugkey_indicationkey', 'intpersonid_failure_drugkey_indicationkey', 'intsponsorid_p1_intclinicaltrialids', 'intsponsorid_p1_completed_intclinicaltrialids', 'intsponsorid_p1_terminated_intclinicaltrialids', 'intsponsorid_p1_positive_intclinicaltrialids', 'intsponsorid_p1_negative_intclinicaltrialids', 'intsponsorid_p2_intclinicaltrialids', 'intsponsorid_p2_completed_intclinicaltrialids', 'intsponsorid_p2_terminated_intclinicaltrialids', 'intsponsorid_p2_positive_intclinicaltrialids', 'intsponsorid_p2_negative_intclinicaltrialids', 'intsponsorid_p3_intclinicaltrialids', 'intsponsorid_p3_completed_intclinicaltrialids', 'intsponsorid_p3_terminated_intclinicaltrialids', 'intsponsorid_p3_positive_intclinicaltrialids', 'intsponsorid_p3_negative_intclinicaltrialids', 'intpersonid_p1_intclinicaltrialids', 'intpersonid_p1_completed_intclinicaltrialids', 'intpersonid_p1_terminated_intclinicaltrialids', 'intpersonid_p1_positive_intclinicaltrialids', 'intpersonid_p1_negative_intclinicaltrialids', 'intpersonid_p2_intclinicaltrialids', 'intpersonid_p2_completed_intclinicaltrialids', 'intpersonid_p2_terminated_intclinicaltrialids', 'intpersonid_p2_positive_intclinicaltrialids', 'intpersonid_p2_negative_intclinicaltrialids', 'intpersonid_p3_intclinicaltrialids', 'intpersonid_p3_completed_intclinicaltrialids', 'intpersonid_p3_terminated_intclinicaltrialids', 'intpersonid_p3_positive_intclinicaltrialids', 'intpersonid_p3_negative_intclinicaltrialids', 'intpriorapproval', 'decminage', 'decmaxage', 'strminageunit', 'strmaxageunit', 'strterminationreason', 'strsponsor','row_id','TRAIN','genericname','test_idx'])
        #'intclinicaltrialid'

        META_DATA=['row_id','TRAIN','outcome','drugkey','indicationkey',self.NEW_YEAR, 'indication_groupkey', 'test_idx']

        #'strterminationreason','strsponsor'])
        cols_multilabel=[#'originkey',
                         'mediumdescription',
                         'strregulatorystatus',
                         'strtherapeuticarea',
                         'strdesignkeyword',
                         'strdiseasetype',
                         'strmechanismofaction',
                         'drugdeliveryroutedescription',
                         'drugtarget',
                         'therapydescription',
                         'strpatientsegment',
                         'strsponsortype',
                         'strlocation',
                         'strterminationreason'
                         ]
        cols_freetext=[
                      #'strpatientpopulation',
                      #'strprimaryendpoint',
                      #'strexclusioncriteria',
                      #'strstudydesign'
                      ]

        #, 'strregulatorystatus', 'strtherapeuticarea', 'strdesignkeyword', 'strdiseasetype', 'strmechanismofaction', 'drugdeliveryroutedescription', 'drugtarget', 'strpatientpopulation', 'strprimaryendpoint', 'strexclusioncriteria', 'strstudydesign', 'genericname', 'therapydescription', 'strpatientsegment', 'strsponsortype', 'strlocation', 'mediumdescription', 'originkey']

        COLUMNS+=cols_multilabel
        COLUMNS+=cols_freetext
        #COLUMNS=[x for x in COLUMNS if self.train.col_type(x)!='s']
        COLUMNS=util.unique2(COLUMNS)
        self.train=self.train[COLUMNS].copy()
        self.test=self.test[COLUMNS].copy()
        self.COLUMNS=COLUMNS

        self.nan_cols=['intduration', 'inttargetaccrual', 'intactualaccrual'] #, 'decminage', 'decmaxage']

        if IMPUTE_MODEL:
            # create impute models, save into ./impute.pkl.gz
            self.impute_model_validate(self.nan_cols)
            self.impute_model(self.nan_cols)
            exit()

        if IMPUTE!="MEAN":
            self.impute_runtime()
            #for x in self.nan_cols:
            #    print(x, self.test[x].isnull().sum())
            #exit()

        # Trivial drugs
        FIND_TRIVIAL=False
        if FIND_TRIVIAL:
            cnt=self.train.groupby('drugkey').size()
            pos=self.train.groupby('drugkey')[self.OUTCOME].sum()
            out=[]
            outcome={}
            for k,v in cnt.items():
                if v>=20 and (pos[k]/v>=0.90 or pos[k]/v<=0.10):
                    out.append((k, pos[k]/v, v))
                    outcome[k]=pos[k]/v
            drugkey, prob, cnt=zip(*out)
            print(list(drugkey))
            print(outcome)
            exit()

        t=self.train[self.OUTCOME].value_counts()
        self.ratio=self.train.outcome.sum()/len(self.train)
        print("Overall approval ratio: ", self.ratio)
        ratio=self.train.groupby(self.NEW_YEAR)[self.OUTCOME].sum()/self.train.groupby(self.NEW_YEAR).size()
        print("Approval ratio by year: ")
        pprint(ratio)

        #print("======== numeric features ", self.train.shape, self.test.shape)
        self.fix_multilabel(cols_multilabel)
        print("======== added multilabel features ", len(cols_multilabel),self.train.shape, self.test.shape,len([x for x in self.train.columns for y in cols_multilabel if y+':' in x]))
        #self.fix_freetext(cols_freetext,coef_cut=0.02,n_cut=None)
        ##print("======== added free text features ", len(cols_freetext),self.train.shape, self.test.shape)

        self.fix_terminate_reason('strterminationreason')
        self.ratio_in_terminate=self.train[self.train['hopeful']<0][self.OUTCOME].mean()
        print("Approval ratio in terminated records:", self.ratio_in_terminate)

        self.filter_datasets()

        #print("\n".join(self.train.header()))
        self.fix_age_col('decminage','strminageunit', 'decmaxage','strmaxageunit')
        self.fix_accrual('inttargetaccrual', 'intactualaccrual')
        self.fix_duration('intduration')
        self.fix_identifiedsites('intidentifiedsites')
        self.fix_prior_approval('intpriorapproval')

        #self.fix_generic_name('genericname')
        self.fix_sponsor_cluster()

        #self.phase_performance(entity="sponsor", phase="p1", previous=5)
        #self.phase_performance(entity="sponsor", phase="p2", previous=5)
        #self.phase_performance(entity="sponsor", phase="p3", previous=5)

        #self.phase_performance(entity="sponsor", phase="p1", previous=3)
        #self.phase_performance(entity="sponsor", phase="p2", previous=3)
        #self.phase_performance(entity="sponsor", phase="p3", previous=3)

        # previous==0 must be at the last, as it modified the raw columns
        self.phase_performance(entity="sponsor", phase="p1")
        self.phase_performance(entity="sponsor", phase="p2")
        self.phase_performance(entity="sponsor", phase="p3")
        self.phase_performance(entity="person", phase="p1")
        self.phase_performance(entity="person", phase="p2")
        self.phase_performance(entity="person", phase="p3")

        #self.overall_performance(entity="sponsor", previous=5)
        #self.overall_performance(entity="sponsor", previous=3)
        self.overall_performance(entity="sponsor")
        self.overall_performance(entity="person")

        self.fix_indication_key("indicationkey")
        self.fix_indication_group("indicationkey")
        self.fix_drug_key('drugkey')

        self.downcast_dtypes(self.train)
        self.downcast_dtypes(self.test)

        self.feature_cols=util.read_list('feature_cols.txt')
        self.feature_cols=[x for x in self.feature_cols if x!='']

        REMOVE_FEATURES=False
        if REMOVE_FEATURES:
            t_rank=pd.read_csv(os.path.join(OUTPUT, 'FeatureRanking.csv'))
            # keep top 50 features
            n_TOP=30
            print("KEEPING >>>>>>>>>>>>>>")
            for x in t_rank[:n_TOP].Feature:
                print(x)
            print("REMOVING >>>>>>>>>>>>>>")
            for x in t_rank[n_TOP:].Feature:
                print(x)
            exit()

        REDUNDANT_FEATURES=False
        if REDUNDANT_FEATURES:
            t_rank=pd.read_csv(os.path.join(OUTPUT, 'FeatureRanking.csv'))
            #t_rank=pd.read_csv('notes/FeatureRanking.all265.csv')
            c_seen={}
            S_cols=[]
            for x in t_rank.Feature:
                s_root=re.sub(r'(_prev_.yers)?(_rank)?_norm(_by_year)?', '', x)
                if s_root in c_seen: continue
                #print(x)
                c_seen[s_root]=True
                S_cols.append(x)
            # remove highly correlated features
            m=self.train[S_cols].values
            r,c=m.shape
            R=np.corrcoef(m, rowvar=0)
            cols_del=[]
            cols_keep=[]
            for i in range(c):
                if S_cols[i] in cols_del: continue
                cols_keep.append(S_cols[i])
                print(S_cols[i])
                rm=[ (S_cols[j], R[i,j]) for j in range(i+1,c) if R[i, j]>=0.95 ]
                for x in rm:
                    if x[0] not in cols_del: cols_del.append(x[0])
                    #print(">>>", S_cols[i], x)
            print("Left:", len(cols_keep), "Removed:", len(t_rank)-len(cols_keep))
            exit()

        REDUNDANT_MultiFEATURES=False # remove identical multilabel features
        if REDUNDANT_MultiFEATURES:
            ##S_cols=util.read_list('features/feature_cols.multi607.txt')
            S_cols=util.read_list('features/feature_cols.n75_m399.txt')
            # remove highly correlated features
            m=self.train[S_cols].values
            r,c=m.shape
            cols_del=[]
            cols_keep=[]
            out={}
            for i in range(c):
                ci=S_cols[i]
                if ci in cols_del: continue
                cols_keep.append(ci)
                #print(S_cols[i])
                rm=[ S_cols[j] for j in range(i+1,c) if np.all(m[:,i] == m[:,j]) ]
                if len(rm) > 0:
                    out[ci]=[] 
                    for x in rm:
                        if x not in cols_del: cols_del.append(x)
                        #print(">>>", S_cols[i], x)
                        out[ci].append(x)
            ##Dump.save_json(out, 'features/Redundant_multicols.json')
            ##with open(f'features/feature_cols.multi_rmRd{len(cols_keep)}.txt', 'w') as f_col:
            ##    f_col.writelines("%s\n" % c for c in cols_keep)

            print("Left:", len(cols_keep), "Removed:", len(cols_del))
            exit()

        if EVAL_ENV!='EVAL':
            FOUND_BAD=False
            print("SEARCH for UNEXPECTED NEW FEATURES:")
            for x in self.train.header():
                if x in META_DATA and x!='YEAR': continue
                if self.train.col_type(x)!='s' and x not in self.feature_cols:
                    FOUND_BAD=True
                    print(x)
            if FOUND_BAD:
                print("There are new features!")
            else:
                print("No unexpected new feature, good!")

            s_file=os.path.join(OUTPUT, 'feature_matrix.pkl.gz')

            self.train=self.train[ util.unique2(META_DATA+self.feature_cols) ].copy()
            self.test=self.test[ util.unique2(META_DATA+self.feature_cols) ].copy()

            Dump.save([self.train, self.test, {'ratio':self.ratio, 'ratio_in_terminate':self.ratio_in_terminate, 'feature_cols':self.feature_cols}], s_file)
            print(os.path.join(OUTPUT, "feature_matrix.pkl.gz"))
            copyfile(os.path.join(OUTPUT,"feature_matrix.pkl.gz"), "feature_matrix.pkl.gz")
        CHECK_CORRELATION=False
        if CHECK_CORRELATION:
            self.check_correlation()
            exit()
        print("Number of Features:", len(self.feature_cols))

    def downcast_dtypes(self, df):
        """Save 50% of memory usage"""
        float_cols = [c for c in df if df[c].dtype == "float64"]
        int_cols = [c for c in df if df[c].dtype in ["int64", "int32"] and c not in ['drugkey','indicationkey','row_id']]
        df[float_cols] = df[float_cols].astype(np.float32)
        df[int_cols] = df[int_cols].astype(np.int16)
        return df

    def fix_generic_name(self, col):
        data=self.merge_datasets()
        data[col]=data[col].apply(lambda x: -1 if (pd.isnull(x) or x=='') else (1 if re.search(r'^[a-zA-Z ,]+$', str(x)) is not None else 0))
        self.split_datasets(data)

    def fix_sponsor_cluster_(self):
        ids=set(util.read_list('cluster_low.txt'))
        self.train['sponsor_class']=self.train.row_id.isin(ids).astype(int)
        self.test['sponsor_class']=self.train.row_id.isin(ids).astype(int)

    def fix_sponsor_cluster(self):
        cluster_ids=util.read_csv('sponsor_cluster.csv')
        cols=['intsponsorid_p2_positive_intclinicaltrialids','intsponsorid_p2_intclinicaltrialids']
        X=cluster_ids[cols].values
        max_X=np.max(X, axis=0)
        X=X/max_X
        y=cluster_ids['cluster_id'].values
        from sklearn.neighbors import KNeighborsClassifier
        knn=KNeighborsClassifier(n_neighbors=1)
        knn.fit(X, y)
        X_train=self.train[cols].values/max_X
        self.train['sponsor_class']=knn.predict(X_train)
        X_test=self.test[cols].values/max_X
        self.test['sponsor_class']=knn.predict(X_test)


    # year 1990 means nan, there are too few items for the year <=2001, so let's change the years
    def fix_phase_year(self, yr_col, newyr_col):
        self.train[newyr_col]=self.train[yr_col].apply(lambda x: 2001 if x<=2001 else x)
        self.train[newyr_col]=self.train[newyr_col]
        # we set
        #self.test[newyr_col]=2015-2000
        self.test[newyr_col]=self.test[yr_col]
        # maybe set 2019 as 2018 as there are not many records there?
        self.test[newyr_col]=self.test[newyr_col].clip(2001, 2025)

    def overall_performance(self, entity="sponsor", previous=0):
        # sponsor/person
        QUANTILE=0.1
        col_pos="int{}id_approval_drugkey_indicationkey".format(entity)
        col_neg="int{}id_failure_drugkey_indicationkey".format(entity)
        col_tot="int{}id_total_drugkey_indicationkey".format(entity)
        col_ratio="int{}id_postive.pct_drugkey_indicationkey".format(entity)
        if previous>0:
            if entity!='sponsor':
                print("ERROR> only have sponsor id for previous!=0")
                exit()
            col_entity='strsponsor'
            col_newtot=col_tot+"_prev_{}yrs".format(previous)
            col_newpos=col_pos+"_prev_{}yrs".format(previous)
            col_newneg=col_neg+"_prev_{}yrs".format(previous)
            col_newratio=col_ratio+"_prev_{}yrs".format(previous)
        data=self.merge_datasets()
        if previous==0:
            data[col_tot]=data[col_pos]+data[col_neg]
            # when calculate percentage, we need to add enough dummie counts to avoid high percentage due to few counts
            dummie=data.groupby(self.NEW_YEAR)[col_tot].quantile(QUANTILE).clip(1, np.inf)
            X=data[self.NEW_YEAR].map(dummie)
            data[col_ratio]=((data[col_pos])/(data[col_tot].values+X)).clip(0.0,1.0)
            #print(data[col_ratio])
            self.rank_normalize_dual(data, col_tot)
            self.rank_normalize_dual(data, col_pos)
            self.rank_normalize_dual(data, col_neg)
        else:
            for k,t_v in data.groupby(self.NEW_YEAR):
                df=data[data[self.NEW_YEAR]<k-previous+1].copy()
                prev_pos=df.groupby(col_entity)[col_pos].max().to_dict()
                prev_neg=df.groupby(col_entity)[col_neg].max().to_dict()
                pos=(t_v[col_pos]-t_v[col_entity].apply(lambda x: prev_pos.get(x, 0))).clip(0,999999)
                neg=(t_v[col_neg]-t_v[col_entity].apply(lambda x: prev_neg.get(x, 0))).clip(0,999999)
                tot=pos+neg
                dummie=tot.quantile(QUANTILE).clip(1,np.inf)
                data.loc[t_v.index, col_newtot]=tot
                data.loc[t_v.index, col_newpos]=pos
                data.loc[t_v.index, col_newneg]=neg
                data.loc[t_v.index, col_newratio]=(pos/(tot+dummie)).clip(0.0,1.0)
            self.rank_normalize_dual(data, col_newtot)
            self.rank_normalize_dual(data, col_newpos)
            self.rank_normalize_dual(data, col_newneg)
        self.split_datasets(data)

    # we can calculate an overall success rate
    # if we calculate POS over years, there are consistent high performers and low performers
    def phase_performance(self, entity="sponsor", phase="p1", previous=0):
        QUANTILE=0.25

        # sponsor/person,  p1/p2/p3
        col_trials="int{}id_{}_intclinicaltrialids".format(entity, phase)
        col_complete="int{}id_{}_completed_intclinicaltrialids".format(entity, phase)
        col_terminate="int{}id_{}_terminated_intclinicaltrialids".format(entity, phase)
        # b/c there are status unknown, complete+terminate!=trials
        col_positive="int{}id_{}_positive_intclinicaltrialids".format(entity, phase)
        col_negative="int{}id_{}_negative_intclinicaltrialids".format(entity, phase)
        # terminate is not successful, but terminate does not mean negative (not enough enrollment

        # there is some data leak in these columns, as it probably aggregate future data
        # e.g., by calculate diff of successful trials, we know how many trials are successful in the previous year.

        # however, since such data are

        # as these counts are accumulative, they grow over years, so we should correct the year-effect
        # we get differential data
        # we will basically use them to rank entity within each year to create a performance index (PI)

        # total number of trails run
        col_tot="int{}id_{}_total_intclinicaltrialids".format(entity, phase)
        col_inprog="int{}id_{}_progress.pct_intclinicaltrialids".format(entity, phase)
        col_comp="int{}id_{}_completed.pct_intclinicaltrialids".format(entity, phase)
        col_pos="int{}id_{}_positive.pct_intclinicaltrialids".format(entity, phase)

        if previous>0:
            if entity!='sponsor':
                print("ERROR> only have sponsor id for previous!=0")
                exit()
            col_entity='strsponsor'
            col_tot+="_prev_{}yrs".format(previous)
            col_inprog+="_prev_{}yrs".format(previous)
            col_comp+="_prev_{}yrs".format(previous)
            col_pos+="_prev_{}yrs".format(previous)

        data=self.merge_datasets()
        data[col_tot]=data[col_trials].copy()
        if previous==0:
            # when calculate percentage, we need to add enough dummie counts to avoid high percentage due to few counts
            dummie=data.groupby(self.NEW_YEAR)[col_tot].quantile(QUANTILE).clip(1, np.inf)
            X=data[self.NEW_YEAR].map(dummie)
            data[col_inprog]=((data[col_trials]-data[col_complete]-data[col_terminate])/(data[col_trials].values+X)).clip(0.0,1.0)
            data['_temp']=data[col_complete]+data[col_terminate]
            dummie=data.groupby(self.NEW_YEAR)['_temp'].quantile(QUANTILE).clip(1, np.inf)
            X=data[self.NEW_YEAR].map(dummie)
            data[col_comp]=((data[col_complete])/(data['_temp'].values+X)).clip(0.0,1.0)
            data['_temp']=data[col_positive]+data[col_negative]
            dummie=data.groupby(self.NEW_YEAR)['_temp'].quantile(QUANTILE).clip(1, np.inf)
            X=data[self.NEW_YEAR].map(dummie)
            data[col_pos]=((data[col_positive])/(data['_temp'].values+X)).clip(0.0,1.0)
            data.drop([col_tot,'_temp'], axis=1, inplace=True)
        else:
            for k,t_v in data.groupby(self.NEW_YEAR):
                #print(k, previous)
                df=data[data[self.NEW_YEAR]<k-previous+1].copy()
                prev_tot=df.groupby(col_entity)[col_trials].max().to_dict()
                tot=(t_v[col_trials]-t_v[col_entity].apply(lambda x: prev_tot.get(x, 0))).clip(0, 999999)
                prev_comp=data[data[self.NEW_YEAR]<k-previous+1].groupby(col_entity)[col_complete].max().to_dict()
                comp=(t_v[col_complete]-t_v[col_entity].apply(lambda x: prev_comp.get(x, 0))).clip(0, 999999)
                prev_term=df.groupby(col_entity)[col_terminate].max().to_dict()
                term=(t_v[col_terminate]-t_v[col_entity].apply(lambda x: prev_term.get(x, 0))).clip(0, 999999)
                inprog=(tot-comp-term).clip(0, 999999)
                prev_pos=df.groupby(col_entity)[col_positive].max().to_dict()
                pos=(t_v[col_positive]-t_v[col_entity].apply(lambda x: prev_pos.get(x, 0))).clip(0, 999999)
                prev_neg=df.groupby(col_entity)[col_negative].max().to_dict()
                neg=(t_v[col_negative]-t_v[col_entity].apply(lambda x: prev_neg.get(x, 0))).clip(0, 999999)
                data.loc[t_v.index, col_tot]=tot
                dummie=tot.quantile(QUANTILE).clip(1, np.inf)
                data.loc[t_v.index, col_inprog]=inprog/(tot+dummie)
                tmp=comp+term
                dummie=tmp.quantile(QUANTILE).clip(1, np.inf)
                data.loc[t_v.index, col_comp]=comp/(tmp+dummie)
                tmp=pos+neg
                dummie=tmp.quantile(QUANTILE).clip(1, np.inf)
                data.loc[t_v.index, col_pos]=pos/(tmp+dummie)

        if previous>0:
            self.rank_normalize_dual(data, col_tot)
        self.rank_normalize_dual(data, col_inprog)
        self.rank_normalize_dual(data, col_comp)
        self.rank_normalize_dual(data, col_pos)

        if previous==0:
            self.rank_normalize_dual(data, col_trials)
            self.rank_normalize_dual(data, col_complete)
            self.rank_normalize_dual(data, col_terminate)
            self.rank_normalize_dual(data, col_positive)
            self.rank_normalize_dual(data, col_negative)

        self.split_datasets(data)

    #drug id is useful, two drug accounts have over 100 trials each
    # fludarabine, oxaliplatin, the are concentrated on some years (what happened???)
    # they are 100% approved and can get hints from prior approvals
    # need to check drug overlap between train and test
    # test has a drug that appear a lot, but not much in training
    # so we may need to down weight, so that each drug is treated equally in the training?
    def fix_drug_key(self, drug_col):

        # besides outcome, calculate other p1,p2,p3....
        QUANTILE=0.15
        data=self.merge_datasets()
        col_cnt='drug_prior_trial_count'
        data[col_cnt]=0.0
        col_ratio='drug_prior_trial_positve.pct'
        data[col_ratio]=0.0
        for k,t_v in data.groupby(self.NEW_YEAR):
            k=min(2015, k) # since we need to access OUTCOME file and OUTCOME is not available for records >=2015
            df=data[data[self.NEW_YEAR]<k].copy() # <k has not count, don't do <=k, which include current year
            # then you peak into the outcome of the current year
            if len(df)==0: continue
            cnt=df.groupby(drug_col).size()
            data.loc[t_v.index, col_cnt]=t_v[drug_col].apply(lambda x: cnt.get(x, 0))

            #test dataset will not have outcome vaues
            #underestimate positive ratios, but nothing we can do
            pos=df[df[self.OUTCOME]>0.5].groupby(drug_col).size().to_dict()
            dummie=max(cnt.quantile(QUANTILE),1)
            cnt=cnt.to_dict()
            data.loc[t_v.index, col_ratio]=t_v[drug_col].apply(lambda x: pos.get(x, 0)/(cnt.get(x,0)+dummie))

        self.rank_normalize_dual(data, col_cnt)
        #print(k, sorted(data.header()))
        self.split_datasets(data)

    def fix_indication_key(self, drug_col):

        # besides outcome, calculate other p1,p2,p3....
        QUANTILE=0.15
        data=self.merge_datasets()
        col_cnt='indication_prior_trial_count'
        data[col_cnt]=0.0
        col_ratio='indication_prior_trial_positve.pct'
        data[col_ratio]=0.0
        for k,t_v in data.groupby(self.NEW_YEAR):
            k=min(2015, k) # since we need to access OUTCOME file and OUTCOME is not available for records >=2015
            df=data[data[self.NEW_YEAR]<k].copy() # <k has not count, don't do <=k, which include current year
            # then you peak into the outcome of the current year
            if len(df)==0: continue
            cnt=df.groupby(drug_col).size()
            data.loc[t_v.index, col_cnt]=t_v[drug_col].apply(lambda x: cnt.get(x, 0))

            #test dataset will not have outcome vaues
            #underestimate positive ratios, but nothing we can do
            pos=df[df[self.OUTCOME]>0.5].groupby(drug_col).size().to_dict()
            dummie=max(cnt.quantile(QUANTILE),1)
            cnt=cnt.to_dict()
            data.loc[t_v.index, col_ratio]=t_v[drug_col].apply(lambda x: pos.get(x, 0)/(cnt.get(x,0)+dummie))

        self.rank_normalize_dual(data, col_cnt)
        #print(k, sorted(data.header()))
        self.split_datasets(data)

    def fix_indication_group(self, ind_col):
        t=pd.read_csv('countvec/indicationkey2group.csv', delimiter="|")
        c_map={}
        for k,t_v in t.groupby('IndicationKey'):
            grpid=t_v.IndicationGroupKey.tolist()
            if len(grpid)==1:
                c_map[k]=grpid[0]
            else:
                grpid=[x for x in grpid if x!=24]
                c_map[k]=min(grpid)
        data=self.merge_datasets()
        indgrp_col='indication_groupkey'
        data[indgrp_col]=data[ind_col].map(c_map)
        QUANTILE=0.15
        col_cnt='indicationgrp_prior_trial_count'
        data[col_cnt]=0.0
        col_ratio='indicationgrp_prior_trial_positve.pct'
        data[col_ratio]=0.0
        for k,t_v in data.groupby(self.NEW_YEAR):
            k=min(2015, k) # since we need to access OUTCOME file and OUTCOME is not available for records >=2015
            df=data[data[self.NEW_YEAR]<k].copy() # <k has not count, don't do <=k, which include current year
            # then you peak into the outcome of the current year
            if len(df)==0: continue
            cnt=df.groupby(indgrp_col).size()
            data.loc[t_v.index, col_cnt]=t_v[indgrp_col].apply(lambda x: cnt.get(x, 0))
            pos=df[df[self.OUTCOME]>0.5].groupby(indgrp_col).size().to_dict()
            dummie=max(cnt.quantile(QUANTILE),1)
            cnt=cnt.to_dict()
            data.loc[t_v.index, col_ratio]=t_v[indgrp_col].apply(lambda x: pos.get(x, 0)/(cnt.get(x,0)+dummie))
        self.rank_normalize_dual(data, col_cnt)
        #print(k, sorted(data.header()))
        enc=OneHotEncoder()
        enc.fit(data[[indgrp_col]])
        t=pd.read_csv('countvec/indicationgroup.csv', delimiter="|")
        t['Name']=t.Name.apply(lambda x: re.sub(r'\W.*$', '', x))
        c_name={k:v for k,v in zip(t.IndicationGroupKey, t.Name)}
        S_col=["indicationgrp_key_{}_{}".format(int(x), c_name[int(x)]) for x in enc.categories_[0]]
        #print(S_col)
        m=enc.transform(data[[indgrp_col]]).toarray().astype(int)
        r,c=m.shape
        for i in range(c):
            data[S_col[i]]=m[:,i]
        #data.drop(indgrp_col, axis=1, inplace=True)
        self.split_datasets(data)

    # this is a very important column, [] stands for probably approved before, ['0'] for not approved before
    # [] has a much higher success rate
    def fix_prior_approval(self, app_col):
        c_map={'[]':1, "['0']":0}
        def fix(df):
            df[app_col]=df[app_col].map(c_map)
            df[app_col].fillna(0, inplace=True)
        fix(self.train)
        fix(self.test)

    def fix_terminate_reason(self, col):
        data=self.merge_datasets()
        def is_positive(s):
            s=s.lower().strip()
            if s=="": return 0
            if 'completed' in s:
                if 'positive' in s: return 1
                if 'negative' in s: return -1
                if 'indeterminate' in s: return 0
            if 'terminated' in s:
                if re.search(r'(negative|adverse|poor|lack|shift|repriorit|bussiness|other)', s) is not None:
                    return -1
                print(">>unseen terminateion reason>>", s)
                return -1
            return 0
        data[col].fillna('', inplace=True)
        data['hopeful']=data[col].apply(lambda x: is_positive(x))
        self.split_datasets(data)

    def fix_age_col(self, min_age_col, min_unit_col, max_age_col, max_unit_col):
        def fix_age(df, age_col, unit_col):
            norm={'months':12, 'weeks': 365/7, 'days': 365, 'years':1}
            df[age_col]=df.apply(lambda r: np.nan if pd.isnull(r[age_col]) else r[age_col]/norm.get(r[unit_col],1), axis=1)

        data=self.merge_datasets()

        def fix(df):
            fix_age(df, min_age_col, min_unit_col)
            fix_age(df, max_age_col, max_unit_col)
            df['age_range']=df[max_age_col]-df[min_age_col]
            df[min_age_col].fillna(df[min_age_col].mean(), inplace=True)
            df[max_age_col].fillna(df[max_age_col].mean(), inplace=True)
            df['age_range'].fillna(df['age_range'].mean(), inplace=True)
            df['for_baby']=df.apply(lambda r: 1 if (r[min_age_col]<=3 or r[max_age_col]<=3) else 0, axis=1)
            df['for_elder']=df.apply(lambda r: 1 if (r[min_age_col]>=65 or r[max_age_col]>=65) else 0, axis=1)
            df.drop([min_unit_col, max_unit_col], axis=1, inplace=True)

        fix(data)
        self.split_datasets(data)

    # (act-target)/target, there is a sweet window 0-1. To high is not good somehow, maybe indicating the trial was not well planned
    def fix_accrual(self, target_col, actual_col):
        QUANTILE=0.25
        data=self.merge_datasets()
        col='accrual_pct'

        #actual_mean=data[actual_col].mean()
        #target_mean=data[target_col].mean()
        #dummie=data[target_col].quantile(QUANTILE).clip(1, np.inf)
        #data[col]=(data[actual_col]-data[target_col])/(data[target_col]+dummie)
        #self.split_datasets(data)
        #exit()

        #data[col]=(data[actual_col]-data[target_col])/data[target_col].clip(100, np.inf)
        if True: #self.IMPUTE=='MEAN':
            for k,t_v in data.groupby(self.NEW_YEAR):
                t_v=t_v.copy()
                actual_mean=t_v[actual_col].mean()
                target_mean=t_v[target_col].mean()
                dummie=t_v[target_col].quantile(QUANTILE).clip(1, np.inf)
                t_v[actual_col].fillna(actual_mean, inplace=True)
                data.loc[t_v.index, actual_col]=t_v[actual_col]
                t_v[target_col].fillna(target_mean, inplace=True)
                data.loc[t_v.index, target_col]=t_v[target_col]
                data.loc[t_v.index, col]=(t_v[actual_col]-t_v[target_col])/(t_v[target_col]+dummie)
        self.rank_normalize_dual(data, 'accrual_pct')
        self.rank_normalize_dual(data, target_col)
        self.rank_normalize_dual(data, actual_col)
        self.split_datasets(data)

    # duration grows over the year, so maybe normalize within the year
    def fix_duration(self, col):
        data=self.merge_datasets()

        #dur_mean=t_v[col].mean()
        #data[col].fillna(dur_mean, inplace=True)
        #self.split_datasets(data)
        #exit()

        if True: #self.IMPUTE=='MEAN':
            for k,t_v in data.groupby(self.NEW_YEAR):
                t_v=t_v.copy()
                dur_mean=t_v[col].mean()
                t_v[col].fillna(dur_mean, inplace=True)
                data.loc[t_v.index, col]=t_v[col]
        #self.median_center_by_year(data, col)
        self.rank_normalize_dual(data, col)
        self.split_datasets(data)

    # not sure what is identified sites, but it also grows over the year
    def fix_identifiedsites(self, col):
        data=self.merge_datasets()
        #self.median_center_by_year(data, col)
        self.rank_normalize_dual(data, col)
        self.split_datasets(data)


    def rank_normalize(self, df, col, inplace=True):
        R=(df[col].values-df[col].mean())/(df[col].std()+self.EPS)
        # simply normalized without converting to rank
        df[col+"_norm"]=R
        R=pd.core.algorithms.rank(df[col].values)
        R/=np.sum(~ np.isnan(R))
        if inplace:
            df[col]=R
        else:
            df[col+"_rank_norm"]=R
        if inplace:
            df.rename2({col: col+"_rank_norm"})

    def rank_normalize_by_year(self, df, col, inplace=True):
        if not inplace:
            df[col+"_rank_norm_by_year"]=df[col].copy()
        for k,t_v in df.groupby(self.NEW_YEAR):
            # take non np.nan values in col and convert them into ranking
            R=(t_v[col].values-t_v[col].mean())/(t_v[col].std()+self.EPS)
            # simply normalized without converting to rank
            df.loc[t_v.index, col+"_norm_by_year"]=R
            R=pd.core.algorithms.rank(t_v[col].values)
            R/=np.sum(~ np.isnan(R))
            R=ss.norm.ppf(np.clip(R, 0.02, 0.98))
            if inplace:
                df.loc[t_v.index, col]=R
            else:
                df.loc[t_v.index, col+"_rank_norm_by_year"]=R
        if inplace:
            df.rename2({col: col+"_rank_norm_by_year"})

    def rank_normalize_dual(self, df, col, inplace=True):
        self.rank_normalize_by_year(df, col, inplace=False)
        #self.rank_normalize(df, col, inplace=inplace)

    def filter_datasets(self):
        if REMOVE_TRIVIAL_DRUGS:
            self.train=self.train[~self.train['drugkey'].isin(TRIVIAL_DRUGS)].copy()
        if REMOVE_TERMINATED:
            self.train=self.train[self.train.hopeful>=0].copy()

    def merge_datasets(self):
        self.test[self.OUTCOME].fillna(0, inplace=True)
        data=pd.concat([self.train, self.test], ignore_index=True, sort=True)
        return data

    def split_datasets(self, df):
        self.train=df[df.TRAIN].copy()
        self.train.index=range(len(self.train))
        self.test=df[~df.TRAIN].copy()
        self.test.index=range(len(self.test))
        #self.test.drop(self.OUTCOME, axis=1, inplace=True)

    def check_correlation(self):
        types=self.train.col_types()
        data=[]
        for i,s in enumerate(self.feature_cols):
            if types[i]!='s' and s not in ['TRAIN',self.OUTCOME,self.YEAR]:
                tmp=self.train[self.train[s].notnull()]
                r=np.corrcoef(tmp[[s, self.OUTCOME]].values, rowvar=0)
                r=r[0,1]
                data.append([s, r, abs(r)])
        t=pd.DataFrame(data, columns=['Column','Corr','AbsCorr'])
        t.sort_values('AbsCorr', ascending=False, inplace=True)
        t.display()

    def impute_model_validate(self, nan_cols):
        data=self.merge_datasets()
        col_feature=[x for x in self.COLUMNS if data.col_type(x)!='s' and x not in [self.OUTCOME,'TRAIN','phaseendyear','drugkey','indicationkey','intclinicaltrialid','row_id']]

        KFOLD=5
        seed=SEED
        kf=StratifiedKFold(n_splits=KFOLD, shuffle=True, random_state=SEED)
        out=[]
        res=[]
        for train_idx,validate_idx in kf.split(data, data[self.NEW_YEAR]):
            train=data.iloc[train_idx]
            validate=data.iloc[validate_idx]
            for col in nan_cols:
                clf = XGBRegressor(
                    objective='reg:linear',
                    random_state=seed,
                    seed=seed,
                    nthread=CORE,
                    max_depth=6,
                    learning_rate=0.05,
                    n_estimators=5000
                    )

                #rf=RandomForestRegressor(n_estimators=200, max_depth=5, n_jobs=CORE, random_state=SEED)
                X_cols=[x for x in col_feature if x!=col]
                mask=~ train[col].isnull()
                my_xtrain=train.loc[mask, X_cols].values
                my_ytrain=train.loc[mask, col].values
                mask=~ validate[col].isnull()
                my_xvalidate=validate.loc[mask, X_cols].values
                my_yvalidate=validate.loc[mask, col].values

                clf.fit(
                    my_xtrain,
                    my_ytrain,
                    eval_metric='rmse',
                    eval_set=[(my_xvalidate, my_yvalidate)],
                    early_stopping_rounds=50
                )

                Yp_validate=clf.predict(my_xvalidate)
                Yp_train=clf.predict(my_xtrain)
                n_estimators=len(clf.get_booster().get_dump())
                print(">>>Impute Train:", col, "R2:", r2_score(my_ytrain, Yp_train), 'r2(truth):', np.corrcoef(my_ytrain, Yp_train, rowvar=0)[0,1], "estimator:",n_estimators)
                res.append({'Col':col,'R2':r2_score(my_ytrain, Yp_train), 'Corr':np.corrcoef(my_ytrain, Yp_train, rowvar=0)[0,1], 'n_estimators':n_estimators})
                print(">>>Impute: Validate", col, "R2:", r2_score(my_yvalidate, Yp_validate), 'r2(truth):', np.corrcoef(my_yvalidate, Yp_validate, rowvar=0)[0,1], 'n_estimators',n_estimators)
                res.append({'Col':col,'R2':r2_score(my_yvalidate, Yp_validate), 'Corr':np.corrcoef(my_yvalidate, Yp_validate, rowvar=0)[0,1], 'n_estimators':n_estimators})

                t1=pd.DataFrame({'True':my_ytrain, 'Pred':Yp_train})
                t1['IsTrain']=1
                t1['Feature']=col
                t2=pd.DataFrame({'True':my_yvalidate, 'Pred':Yp_validate})
                t2['IsTrain']=0
                t2['Feature']=col
                k=int(len(out)/2)
                t1['Fold']=k
                t2['Fold']=k
                out.append(t1)
                out.append(t2)
        t=pd.concat(out, ignore_index=True)
        t.to_csv('impute.csv', index=False)
        t=pd.DataFrame(res)
        t.to_csv('impute.corr.csv', index=False)
        t.display()

    def impute_model(self, nan_cols):
        col_feature=[x for x in self.COLUMNS if self.train.col_type(x)!='s' and x not in [self.OUTCOME,'TRAIN','phaseendyear','drugkey','indicationkey','intclinicaltrialid','row_id']]
        #print(col_feature)
        data=self.merge_datasets()
        t_param=pd.read_csv('impute.corr.csv')
        c_estimator=t_param.groupby('Col')['n_estimators'].mean()
        out={'X_cols':col_feature, 'nan_cols':nan_cols}
        seed=SEED
        for col in nan_cols:
            clf = XGBRegressor(
                objective='reg:linear',
                random_state=seed,
                seed=seed,
                nthread=CORE,
                max_depth=6,
                learning_rate=0.05,
                n_estimators=5000
            )

            X_cols=[x for x in col_feature if x!=col]
            mask=~ data[col].isnull()
            my_xtrain=data.loc[mask, X_cols].values
            my_ytrain=data.loc[mask, col].values
            clf.set_params(n_estimators=int(c_estimator[col]))
            clf.fit(
                my_xtrain,
                my_ytrain,
            )
            Yp=clf.predict(my_xtrain)
            print(">>>Impute:", col, "R2:", r2_score(my_ytrain, Yp), 'r2(truth):', np.corrcoef(my_ytrain, Yp, rowvar=0)[0,1])
            out[col]=clf
            input("Press any key to continue ...")
        Dump.save(out, os.path.join(".",'impute.pkl.gz'))

    def impute_runtime(self):
        params=Dump.load(os.path.join(".",'impute.pkl.gz'))
        nan_cols=params['nan_cols']
        col_feature=params['X_cols']
        data=self.merge_datasets()
        for col in nan_cols:
            has_nan=data[col].isnull()
            print("impute:", col, "#nan:", sum(has_nan))
            X_cols=[x for x in col_feature if x!=col]
            X=data.loc[has_nan, X_cols].values
            data.loc[has_nan, col]=params[col].predict(X)
            print("impute:", col, "#nan:", sum(data[col].isnull()))
        #print(np.sum(data[nan_cols].isnull().values))
        self.split_datasets(data)

    def fix_multilabel(self,cols):
        data=self.merge_datasets() # compute tfidf with both train and test
        all_vec=[]
        for c in cols:
            T2V=Text2Vec(data,c,'multi')
            T2V.cleanSymbols()
            T2V.getCounts(max_df=0.99,min_df=0.01)
            #pickle.dump([self.train,T2V.df], open('t.pkl', 'wb'))
            #print("!!!!!!!!!!!!!!",T2V.df.header())
            if len(T2V.df.header())==0: continue # not useful term is found
            all_vec.append(T2V.df)
        if len(all_vec)==0: return
        data=pd.concat([data]+all_vec,1)
        self.split_datasets(data)

    def fix_freetext(self,cols,coef_cut=0.05,n_cut=100):
        data=self.merge_datasets() # compute tfidf with both train and test
        all_vec=[]
        for c in cols:
            data[c]=data[c].apply(lambda x: '' if pd.isnull(x) else re.sub(r'\s+', ' ', x))
            T2V=Text2Vec(data,c,'free')
            T2V.cleanSymbols()
            T2V.getCounts(ngram=(1,2),max_df=0.99,min_df=0.01)
            #T2V.filterTerms(coef_cut=coef_cut,n_cut=n_cut)
            #pickle.dump([self.train,T2V.df], open('t.pkl', 'wb'))
            all_vec.append(T2V.df)
        data=pd.concat([data]+all_vec,1)
        self.split_datasets(data)

class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]

class Text2Vec(object):
    def __init__(self,t,col,col_type):
        self.col=col
        self.txts=t[col].values
        self.txt_type=col_type #multi or free
        self.outcome=t.outcome.values if 'outcome' in t.columns else None
        self.is_train=t.TRAIN
        self.index=t.index

    def cleanSymbols(self,s_na='', s_strip='[]()',s_space='\n'):
        def run_cs(s,s_na,s_strip,s_space):
            if pd.isnull(s): return s_na
            s=re.sub(r'\(\d+\)','',s) # remove step numbers: (1) ... (2)...
            s_trans=s.maketrans(s_space,''.join([' ']*len(s_space)),s_strip)
            s=s.translate(s_trans)
            # fix special symbols
            s=re.sub(r'>\s*or\s*=','GreaterOrEqual', s)
            s=re.sub('<=','LessOrEqual',s)
            return s.strip()
        self.txts=np.array([run_cs(s,s_na,s_strip,s_space) for s in self.txts])

    def getCounts(self,ngram=(1,2),max_df=0.99,min_df=0.01, top=25):
        txt_train=self.txts[self.is_train]
        txt_test=self.txts[~self.is_train]
        os.makedirs('countvec', exist_ok=True)
        s_file=os.path.join('countvec', self.col+'.pkl.gz')
        if os.path.exists(s_file):
            vect=Dump.load(s_file)
        else:
            if self.txt_type == 'multi':
                vect = fet.CountVectorizer(tokenizer=myToken ,max_df=max_df,min_df=min_df,binary=True)
            if self.txt_type == 'free':
                tokenized_stop_words = word_tokenize(' '.join(corpus.stopwords.words('english'))) + ['doe', 'ha', 'wa']
                vect = fet.TfidfVectorizer(tokenizer=LemmaTokenizer(),stop_words=tokenized_stop_words,ngram_range=ngram,max_df=max_df,min_df=min_df)
            vect.fit(txt_train)
            Dump.save(vect, s_file)
        X_train=vect.transform(txt_train)
        X_test=vect.transform(txt_test)

        self.df=pd.DataFrame(index=range(len(self.txts)),columns=[self.col+':'+ feat for feat in vect.get_feature_names()])
        print(">>>>> total number of text features", self.df.shape)
        self.df[self.is_train]=X_train.toarray()
        self.df[~self.is_train]=X_test.toarray()

        #if self.txt_type == 'multi':
        #    X_train=np.clip(X_train, 0,1)
        #    X_test=np.clip(X_test, 0,1)

        if self.txt_type == 'free':
            self.df=pd.DataFrame(index=self.index,columns=[])
            R_outcome=self.outcome[self.is_train]
            mask_pos=R_outcome>0.5
            mask_neg=~mask_pos
            USEFUL=[]
            os.makedirs('useful_terms', exist_ok=True)
            s_term='useful_terms/{}.txt'.format(self.col)
            FIND_NEW=True
            if os.path.exists(s_term):
                USEFUL=set(util.read_list(s_term))
                FIND_NEW=False
            if EVAL_ENV=='EVAL': FIND_NEW=False
            if FIND_NEW:
                f_useful=open(s_term, 'a')
            if self.txt_type == 'multi':
                eps=1e-5
                log_prob_p=[]
                log_prob_n=[]
                for i,feat in enumerate(vect.get_feature_names()):
                    s_key="col:{},term:{}".format(self.col, feat)
                    if not FIND_NEW and s_key not in USEFUL: continue
                    if FIND_NEW and re.search(r'^\s*$', feat): continue
                    R_train=np.clip(X_train[:, i].toarray()[:,0], 0,1)
                    R_test=np.clip(X_test[:, i].toarray()[:,0], 0,1)
                    Xp=np.clip(np.mean(R_train[mask_pos]), eps, np.inf)
                    Xn=np.clip(np.mean(R_train[mask_neg]), eps, np.inf)
                    if FIND_NEW and abs(Xp-Xn)<0.1: continue # term not useful
                    log_prob_p.append(np.log(Xp))
                    log_prob_n.append(np.log(Xn))
                    self.df.loc[self.is_train, self.col+':'+ feat]=R_train
                    self.df.loc[~self.is_train, self.col+':'+ feat]=R_test
                    if FIND_NEW:
                        f_useful.write(s_key+"\n")
                        print("Found Useful Term:", s_key)
                if len(log_prob_p)>1:
                    prior_p=np.mean(R_outcome)
                    log_prob_p=np.array(log_prob_p)
                    log_prob_n=np.array(log_prob_n)
                    a=np.sum(self.df.values*log_prob_p, axis=1)+np.log(prior_p)
                    b=np.sum(self.df.values*log_prob_n, axis=1)+np.log(1-prior_p)
                    prob=1/(1+np.exp(b-a))
                    if np.min(prob)<np.max(prob):
                        #print("@@@@@@@@@@@@@@",self.col+'_naive_p')
                        self.df[self.col+'_naive_p']=prob
                    #else:
                        #print("$$$$$$$$$$", self.col+'_naive_p')
                        #self.df[self.col+'_naive_p']=prob
            else:
                eps=1e-5
                p_vals=[]
                log_prob_n=[]
                for i,feat in enumerate(vect.get_feature_names()):
                    s_key="col:{},term:{}".format(self.col, feat)
                    if not FIND_NEW and s_key not in USEFUL: continue
                    if FIND_NEW and re.search(r'^\s*$', feat): continue
                    if FIND_NEW and re.search(r'[\W]', re.sub(r'\s', '', feat)): continue # two gram may contain a , %, etc.
                    if FIND_NEW and re.search(r'^\d+$', re.sub(r'\s', '', feat)): continue
                    R_train=X_train[:, i].toarray()[:,0]
                    R_test=X_test[:, i].toarray()[:,0]
                    Xp=R_train[mask_pos]
                    Xn=R_train[mask_neg]
                    p_val=ss.mannwhitneyu(Xp,Xn)[1]*2
                    if FIND_NEW and p_val>0.05: continue # term not useful
                    p_vals.append((i, feat, p_val))
                    if not FIND_NEW and s_key in USEFUL:
                        self.df.loc[self.is_train, self.col+':'+ feat]=R_train
                        self.df.loc[~self.is_train, self.col+':'+ feat]=R_test
                if FIND_NEW:
                    p_vals=sorted(p_vals, key=lambda x: x[2])[:top]
                    print("Top 20 terms:")
                    for i,feat,p_val in p_vals:
                        print("Top:", i+1, feat)
                        s_key="col:{},term:{}".format(self.col, feat)
                        R_train=X_train[:, i].toarray()[:,0]
                        R_test=X_test[:, i].toarray()[:,0]
                        self.df.loc[self.is_train, self.col+':'+ feat]=R_train
                        self.df.loc[~self.is_train, self.col+':'+ feat]=R_test
                        print("Found new term", s_key)
                        f_useful.write(s_key+"\n")
                    f_useful.close()

        print(">>>>>>>>> Text2Vec: shape of feat matrix ", self.col, self.df.shape)

    def filterTerms(self,coef_cut=0.01,n_cut=100): # use correlation with outcome in training to filter terms (potential data leak?)
        if (self.outcome is not None) & (self.is_train is not None):
            df=self.df[self.is_train]
            outcome=self.outcome[self.is_train]
            out=[]
            for c in df.columns:
                coef=np.corrcoef(df[c].astype(float),outcome,rowvar=False)[0,1]
                out.append([c,coef])
            t_coef=pd.DataFrame(out,columns=['Term','Coef'])
            if coef_cut is not None:
                t_coef=t_coef[t_coef.Coef >= coef_cut]
            if n_cut is not None:
                t_coef=t_coef.sort_values('Coef',ascending=False)[:n_cut]
            self.df=self.df[t_coef.Term.tolist()]
            print(f">>>>>>>>> Text2Vec: top {n_cut} of feat matrix with pearson coef >={coef_cut}", self.col, self.df.shape)
        #else:
        #    util.error_msg('Missing outcome for text term filtering')

class Submission:

    @staticmethod
    def save(test, filename, lb=LOWER_BOUND, ub=UPPER_BOUND):
        df=test[['row_id','prob_approval']].copy()
        df['prob_approval']=df['prob_approval'].clip(lb, ub)
        df.to_csv(filename, index=False)

class MODEL:

    def __init__(self, name="NONAME", impute=IMPUTE, recreate=RECREATE):
        self.name=name
        self.tune_estimator= 'RF' not in name # for RF, no need to tune estimator
        self.impute=impute
        self.eda=EDA(impute=impute, recreate=recreate)
        self.NEW_YEAR=self.eda.NEW_YEAR
        self.OUTCOME=self.eda.OUTCOME
        self.feature_cols=self.eda.feature_cols
        self.hopeful_idx=util.index('hopeful', self.feature_cols)
        self.ratio_in_terminate=self.eda.ratio_in_terminate

        self.ds=self.prepare_datasets()

    def best_model(self):
        pass

    def pred_test(self):
        clf=Dump.load("model.pkl.gz")[-1]
        X_test=self.ds['test'][self.feature_cols].values
        row_id=self.ds['test']['row_id']
        Yp_test=self.augment_pred(self.ds['test'], clf.predict_proba(X_test)[:,1])
        if FIX_20182019:
            Yp_test[self.ds['test'][self.NEW_YEAR]>=2018]=LOWER_BOUND
        df=pd.DataFrame({'row_id':row_id, 'prob_approval':Yp_test, 'test_idx':self.ds['test']['test_idx']})
        df=df.sort_values('test_idx').drop('test_idx',1)
        Submission.save(df, SUBMISSION_FILE)

    def weight_by_drug(self, t):
        if 'ONE' in SAMPLE_WEIGHT:
            return np.ones(len(t))
        if SAMPLE_WEIGHT.startswith('DI'):
            key_list=self.make_key_list(t)
            c_drugkey_cnt=util.unique_count(key_list)
            weight=np.array([c_drugkey_cnt[x] for x in key_list])
        else:
            key_list=t.drugkey.astype(int)
            c_drugkey_cnt=util.unique_count(key_list)
            weight=np.array([c_drugkey_cnt[x] for x in key_list])
        if SAMPLE_WEIGHT.endswith('INV'):
            weight= 1/weight
        elif SAMPLE_WEIGHT.endswith('SQRT'):
            weight = 1/np.sqrt(weight)
        return weight

    def weight_by_outcome(self, t):
        p=t[self.OUTCOME].mean()
        w=TARGET_RATIO*(1-p)/p/(1-TARGET_RATIO)
        c={0:1, 1:w}
        weight=t[self.OUTCOME].map(c)
        return weight

    def make_key_list(self, t):
        return t.apply(lambda r: "{:d}:{:d}".format(int(r['drugkey']),int(r['indicationkey'])), axis=1)

    def make_key_set(self, t):
        return set(self.make_key_list(t))

    def in_key_set(self, t, key_set):
        return t[self.make_key_list(t).isin(key_set)].copy()

    def not_in_key_set(self, t, key_set):
        return t[~self.make_key_list(t).isin(key_set)].copy()

    def augment_pred(self, df_X, Yp):
        mask=(df_X[self.NEW_YEAR]<2018).values
        df_X=df_X[['drugkey','indicationkey', self.NEW_YEAR]].copy()
        df_X['PRED']=np.clip(Yp, LOWER_BOUND, UPPER_BOUND)
        if DRUGKEY_AUGMENTATION:
            t=pd.DataFrame({'KEY':self.make_key_list(df_X), 'PRED':df_X.PRED})
            if FIX_20182019:
                aggr_by_drug=t[mask].groupby('KEY')['PRED'].max()
            else:
                aggr_by_drug=t.groupby('KEY')['PRED'].max()
            t['PRED']=t.KEY.map(aggr_by_drug)
            df_X['PRED']=t.PRED.values
        if FIX_20182019:
            df_X.loc[~mask, 'PRED']=LOWER_BOUND
        return df_X.PRED.values

    def prepare_datasets(self):
        np.random.seed(SEED)
        ds={'train':[], 'validate':[], 'test':[], 'train_weight':[], 'validate_weight':[], 'test_weight':[], 'refit':None, 'refit_weight':None, 'test_name':[]}
        KFOLD=5
        seed=SEED

        if EVAL_ENV=='EVAL' or args.debug:
            ds['test']=self.eda.test
            ds['test_name']='full_test_set'
            return ds

        if VALIDATION_MODE=="CV":
            kf=StratifiedKFold(n_splits=KFOLD, shuffle=True, random_state=seed)
            if WEIGHT_BY_DRUGKEY:
                train_=self.eda.train.drop_duplicates(['drugkey','indicationkey',self.OUTCOME])
                for train_idx, validate_idx in kf.split(train_, train_[self.OUTCOME].values):
                    ds['train'].append(self.in_key_set(self.eda.train, self.make_key_set(train_.iloc[train_idx])))
                    ds['validate'].append(self.in_key_set(self.eda.train, self.make_key_set(train_.iloc[validate_idx])))
            else:
                for train_idx, validate_idx in kf.split(self.eda.train, self.eda.train[self.OUTCOME].values):
                    ds['train'].append(self.eda.train.iloc[train_idx].copy())
                    ds['validate'].append(self.eda.train.iloc[validate_idx].copy())
        elif VALIDATION_MODE=="1314":
            in_train=self.eda.train[self.NEW_YEAR]<2013
            if WEIGHT_BY_DRUGKEY:
                key_set=self.make_key_set(self.eda.train[~ in_train])
                ds['train'].append(self.not_in_key_set(self.eda.train, key_set))
                ds['validate'].append(self.in_key_set(self.eda.train, key_set))
            else:
                ds['train'].append(self.eda.train[in_train].copy())
                ds['validate'].append(self.eda.train[~in_train].copy())

        ds['train_weight']=[self.weight_by_drug(X) for X in ds['train']]
        ds['validate_weight']=[self.weight_by_outcome(X) for X in ds['validate']]
        ds['refit']=self.eda.train.copy()
        ds['refit_weight']=self.weight_by_drug(ds['refit'])

        n=int(len(self.eda.test)*0.45)
        ds['test'].extend([self.eda.test[self.eda.test[self.NEW_YEAR]<2018].copy(), self.eda.test.copy(), self.eda.test[:n].copy(), self.eda.test[n:].copy()])
        ds['test_name']=['lt8','all','1st','2nd']
        ds['test_weight']=[np.ones(len(X)) for X in ds['test']]
        return ds

    def fix_terminated_samples(self, X_train, Yp_train, X_validate, Yp_validate):
        ALPHA=max(0.01, LOWER_BOUND)
        if SET_TERMINATED_TO_ZERO:
            mask=X_train[:, self.hopeful_idx]<0
            Yp_train[mask]=np.clip(Yp_train[mask], ALPHA, self.ratio_in_terminate)
            mask=X_validate[:, self.hopeful_idx]<0
            Yp_validate[mask]=np.clip(Yp_validate[mask], ALPHA, self.ratio_in_terminate)
        return (Yp_train, Yp_validate)

    def fix_trivial_drugs(self, X_train, Yp_train, X_validate, Yp_validate):
        ALPHA=max(0.05, LOWER_BOUND)
        if FIX_TRIVIAL_DRUGS:
            c={k: min(max(v, ALPHA), 1-ALPHA) for k,v in TRIVIAL_OUTCOME}
            mask=X_train['drugkey'].isin(TRIVIAL_DRUGS)
            Yp_train[mask]=X_train[mask].drugkey.map(c)
            mask=X_validate['drugkey'].isin(TRIVIAL_DRUGS)
            Yp_validate[mask]=X_validate[mask].drugkey.map(c)
        return (Yp_train, Yp_validate)

    def rank_features(self, seed=SEED, step=1):
        """step: remove step features at a time to speed things up"""

        clf = Dump.load(os.path.join(OUTPUT,"model.pkl.gz"))[-1]

        method="GAIN"
        features=self.feature_cols[:]
        n=len(features)

        pg=util.Progress(int(n/step))
        cnt=[0]

        def score_features_total_gain(feautures):
            out=[]
            if 'RF' in self.name:
                trains=[self.ds['refit']]
                w_trains=[self.ds['refit_weight']]
                validates=self.ds['validate'][:1]
                w_validates=self.ds['validate_weight'][:1]
            else:
                trains=self.ds['train']
                w_trains=self.ds['train_weight']
                validates=self.ds['validate']
                w_validates=self.ds['validate_weight']

            for train, validate, w_train, w_validate in zip(trains, validates, w_trains, w_validates):
                X_train, Y_train=train[features].values, train[self.OUTCOME].values
                X_validate, Y_validate=validate[features].values, validate[self.OUTCOME].values
                eval_ds=[validate]
                eval_set_weight=[w_validate]

                fit_params={} if 'RF' in self.name else {'early_stopping_rounds':early_stopping}

                count2Cache = {}
                for df_X,w in zip(eval_ds,eval_set_weight):
                    mask = (df_X[self.NEW_YEAR] < 2018).values
                    df_X = df_X[['drugkey', 'indicationkey', self.NEW_YEAR]].copy()
                    t = pd.DataFrame({
                        'KEY': self.make_key_list(df_X),
                        'PRED': None
                    })
                    count2Cache[len(df_X)]=(mask,t,w)

                def my_scorer(y_pred, dtrain):
                    y_true = dtrain.get_label()
                    y_pred = np.clip(y_pred, LOWER_BOUND, UPPER_BOUND)
                    mask, t, w= count2Cache[len(y_true)]
                    t['PRED'] = y_pred
                    # mean_by_drug = t[mask].groupby('KEY')['PRED'].mean()
                    mean_by_drug = t[mask].groupby('KEY')['PRED'].max()
                    a = t.KEY.map(mean_by_drug)
                    a[~mask] = LOWER_BOUND
                    s = log_loss(y_true, a, sample_weight=w)
                    return 'my_',s

                clf.fit(
                    X_train,
                    Y_train,
                    sample_weight=w_train,
                    eval_metric=my_scorer, #'logloss',
                    eval_set=[(X_validate, Y_validate)],
                    sample_weight_eval_set=[ (w_validate) ],
                    early_stopping_rounds= 10 if 'RF' not in self.name else None,
                    verbose=False)

                Yp_validate=np.clip(clf.predict_proba(X_validate)[:,1], LOWER_BOUND, UPPER_BOUND)
                score=log_loss(Y_validate, Yp_validate, sample_weight=w_validate)
                clf.get_booster().feature_names=features
                importance = clf.get_booster().get_score(importance_type='total_gain')
                tuples = [(k, importance[k], score) for k in importance]
                tuples = sorted(tuples, key=lambda x: x[1])
                t=pd.DataFrame(tuples, columns=['Feature','TotalGain','LogLoss'])
                t['Rank']=range(len(t))
                out.append(t)
            t=pd.concat(out, ignore_index=True)
            t=t.groupby(['Feature']).mean().reset_index()
            t.sort_values('Rank', inplace=True)
            t.drop('Rank', axis=1, inplace=True)
            cnt[0]+=1
            print(t[:5])
            pg.check(cnt[0])
            return dict(t.iloc[0])

        out=[]
        del_features=[]
        for i in range(n, 0, -1):
            print(">>>loop", i, "of", n)
            one=score_features_total_gain(features)
            one['Rank']=i
            rm_f=one['Feature']
            a,b=self.ds['refit'][rm_f].values, self.ds['refit'][self.OUTCOME].values
            one['Correlation']=np.corrcoef(a,b, rowvar=0)[0,1]
            one['p-mann-whitney']=np.log10(ss.mannwhitneyu(a, b)[1]*2)
            out.append(one)
            print(">>>>>>Remove feature:", rm_f)
            del_features.append(rm_f)
            features=[x for x in self.feature_cols if x not in del_features]
            t=pd.DataFrame(data=out)[::-1]
            util.df2sdf(t, s_format="%.5f").to_csv(os.path.join(OUTPUT, 'FeatureRanking.csv'), index=False)
            S=t.Feature.apply(lambda x: '            "{}",'.format(x)).tolist()
            util.save_list(os.path.join(OUTPUT, 'FeatureRanking.txt'), S, s_end="\n")

    def get_XY(self, t):
        return (t[self.feature_cols].values, t[self.OUTCOME].values)

    def tune_hyperparameter(self, seed=SEED, jsonfile='hyperparameter.range.json', early_stopping=10):

        ESTIMATOR='estimator' in jsonfile
        params_grid=Dump.load_json(os.path.join(OUTPUT, jsonfile))
        grid=ParameterGrid(params_grid)

        out=[]
        best_score=999
        best_params=None
        best_model=None
        best_one=None

        for params in grid:
            clf=self.best_model(seed=seed)
            clf.set_params(**params)

            scores=[]
            confusion=[]
            confusion_hopeless=[]
            confusion_hopeful=[]
            test_names=['train','validate']
            for train, validate, w_train, w_validate in zip(self.ds['train'], self.ds['validate'], self.ds['train_weight'], self.ds['validate_weight']):
                X_train, Y_train=self.get_XY(train)
                eval_set_val=[self.get_XY(validate)]
                eval_set=[(X_train, Y_train)]+eval_set_val
                eval_set_weight=[w_train, w_validate]
                eval_ds=[train, validate]

                fit_params={} if 'RF' in self.name else {'early_stopping_rounds':early_stopping}

                count2Cache = {}
                for df_X,w in zip(eval_ds,eval_set_weight):
                    mask = (df_X[self.NEW_YEAR] < 2018).values
                    df_X = df_X[['drugkey', 'indicationkey', self.NEW_YEAR]].copy()
                    t = pd.DataFrame({
                        'KEY': self.make_key_list(df_X),
                        'PRED': None
                    })
                    count2Cache[len(df_X)]=(mask,t,w)

                def my_scorer(y_pred, dtrain):
                    y_true = dtrain.get_label()
                    y_pred = np.clip(y_pred, LOWER_BOUND, UPPER_BOUND)
                    mask, t, w= count2Cache[len(y_true)]
                    t['PRED'] = y_pred
                    mean_by_drug = t[mask].groupby('KEY')['PRED'].max()
                    a = t.KEY.map(mean_by_drug)
                    a[~mask] = LOWER_BOUND
                    s = log_loss(y_true, a, sample_weight=w)
                    return 'my_',s

                clf.fit(
                    X_train,
                    Y_train,
                    eval_metric=my_scorer, #'logloss',
                    sample_weight=w_train,
                    eval_set=eval_set,
                    sample_weight_eval_set=eval_set_weight,
                    **fit_params
                )
                Yps=[ clf.predict_proba(X)[:,1] for (X,Y) in eval_set ]
                X,Y=eval_set[-1]
                Yp=(self.augment_pred(eval_ds[-1], Yps[-1])>=0.5).astype(int)
                ###mask=eval_ds[-1].hopeful==-1

                ###for _i in range(3):
                ###    if _i==0:
                ###        _Y=Y
                ###        _Yp=Yp
                ###    elif _i==1:
                ###        _Y=Y[mask]
                ###        _Yp=Yp[mask]
                ###    else:
                ###        _Y=Y[~mask]
                ###        _Yp=Yp[~mask]
                ###    m=confusion_matrix(_Y, _Yp).reshape(4)
                ###    m=m/np.sum(m)
                ###    precision, recall, f1, _ =precision_recall_fscore_support(_Y, _Yp)
                ###    TP=sum((_Y==1) & (_Yp>=0.5))
                ###    P=sum(_Y==1)
                ###    TN=sum((_Y==0) & (_Yp<0.5))
                ###    N=sum(_Y==0)
                ###    tnr=TN/N
                ###    ppv=TP/sum(_Yp>=0.5)
                ###    npv=TN/sum(_Yp<0.5)
                ###    if _i==0:
                ###        confusion.append(np.append(m, [precision[1], recall[1], f1[1], tnr, ppv, npv]))
                ###    elif _i==1:
                ###        confusion_hopeless.append(np.append(m, [precision[1], recall[1], f1[1], tnr, ppv, npv]))
                ###    else:
                ###        confusion_hopeful.append(np.append(m, [precision[1], recall[1], f1[1], tnr, ppv, npv]))

                Yps=[ log_loss(Y, self.augment_pred(eval_ds[i], Yps[i]), sample_weight=eval_set_weight[i]) for i,(X,Y) in enumerate(eval_set) ]
                # the following matches clf score, that is not clipped
                #Yps=[ log_loss(Y, Yps[i], sample_weight=eval_set_weight[i]) for i,(X,Y) in enumerate(eval_set) ]
                scores.append(Yps)
            one=params.copy()
            scores=np.mean(np.array(scores), axis=0)
            ###confusion=np.mean(np.array(confusion), axis=0)
            ###confusion_hopeless=np.mean(np.array(confusion_hopeless), axis=0)
            ###confusion_hopeful=np.mean(np.array(confusion_hopeful), axis=0)
            one.update({'logloss_{}'.format(test_names[i]):x for i,x in enumerate(scores)})
            ###names=["confusion_00", "confusion_01", "confusion_10", "confusion_11", "precision", "recall (TPR)", "f1", "TNR", "PPV", "NPV"]
            ###one.update({names[i]:x for i,x in enumerate(confusion)})
            ###one.update({names[i]+"_hopeless":x for i,x in enumerate(confusion_hopeless)})
            ###one.update({names[i]+"_hopeful":x for i,x in enumerate(confusion_hopeful)})
            out.append(one)
            if scores[-1]<best_score:
                best_score=scores[-1]
                best_params=params.copy()
                best_model=copy.deepcopy(clf)
                best_one=one.copy()
                print("Find better solution:", best_score)
                print(one)

        if ESTIMATOR:
            best_params["n_estimators"]=len(best_model.get_booster().get_dump())
            Dump.save_json(best_params, os.path.join(OUTPUT, 'hyperparameter.best.json'))
            print("vi", os.path.join(OUTPUT, 'hyperparameter.estimator.json'))
            print("vi", os.path.join(OUTPUT, 'hyperparameter.best.json'))
        else:
            Dump.save_json(best_params, os.path.join(OUTPUT, 'hyperparameter.best.json'))
            print("Best params:", best_params)
            best_params={ k:[v] for k,v in best_params.items() }
            if "learning_rate" in best_params: # RF does not have this
                lr=best_params["learning_rate"][0]
                best_params["learning_rate"]=[lr, lr/2, lr/3, lr/5, lr/10]
            best_params["n_estimators"]=[5000]
            Dump.save_json(best_params, os.path.join(OUTPUT, 'hyperparameter.estimator.json'))
            print("vi", os.path.join(OUTPUT, 'hyperparameter.range.json'))
            print("vi", os.path.join(OUTPUT, 'hyperparameter.estimator.json'))

        print("Best parameter and scores:")
        print(best_one)
        best_one.update({'model':MODEL_SELECTION,'sample_weight':SAMPLE_WEIGHT,'validation':VALIDATION_MODE,'#features':len(self.feature_cols),'use_numeric':CONFIG['UseNumeric']})
        Dump.save_json(best_one,  os.path.join(OUTPUT, 'hyperparameter.bestscores.json'))

    def refit_model(self, seed=SEED):
        clf=self.best_model(seed=seed, s_file='hyperparameter.json')
        X,Y=self.get_XY(self.ds['refit'])

        clf.fit(
            X,
            Y,
            sample_weight=self.ds['refit_weight'],
            verbose=True)
     
        f_model=os.path.join(OUTPUT,f"model.pkl.gz")
        Dump.save([self.feature_cols, clf], f_model)
        print("Model with given parameters is saved as: ", os.path.join(OUTPUT,f"model.pkl.gz"))
        print("copy model to ./model.pkl.gz and submit")

class XGBOOST(MODEL):

    def __init__(self, impute=IMPUTE, recreate=RECREATE):
        super(XGBOOST, self).__init__(name='XGBOOST', impute=impute, recreate=recreate)

    def best_model(self, seed=SEED, s_file=os.path.join(OUTPUT, 'hyperparameter.best.json')):
        if os.path.exists(s_file):
            kw=Dump.load_json(s_file)
            kw['n_estimators']=int(kw['n_estimators'])
        else:
            kw={}
        #https://stats.stackexchange.com/questions/243207/what-is-the-proper-usage-of-scale-pos-weight-in-xgboost-for-imbalanced-datasets
        clf = XGBClassifier(
                **kw
                )
        return clf

class RF(MODEL):

    def __init__(self, impute=IMPUTE, recreate=RECREATE):
        super(RF, self).__init__(name='RF', impute=impute, recreate=recreate)

    def best_model(self, seed=SEED,s_file=os.path.join(OUTPUT, 'hyperparameter.best.json')):
        if os.path.exists(s_file):
            kw=Dump.load_json(s_file)
            kw['n_estimators']=int(kw['n_estimators'])
        else:
            kw={}

        clf = XGBRFClassifier(
                **kw
                )
        return clf

if __name__=="__main__":

    # by default, check if pickle dump exists, change to True, if you want to regenerate
    #RECREATE=True
    if args.recreate or args.impute_model:
        eda=EDA(impute=IMPUTE, recreate=True)
        exit()

    if 'XGB' in MODEL_SELECTION:
        pprint("XGBoost Model...")
        model=XGBOOST(impute=IMPUTE, recreate=RECREATE)
    elif 'RF' in MODEL_SELECTION:
        pprint("XGBRF Model...")
        model=RF(impute=IMPUTE, recreate=RECREATE)

    if EVAL_ENV=='EVAL' or args.debug:
        model.pred_test()
        exit()
    if args.sort:
        model.rank_features()
        exit()
    if HYPERPARAMETER:
        if ONLY_TUNE_ESTIMATOR:
            model.refit_model(seed=SEED)
        else:
            model.tune_hyperparameter(seed=SEED)
    else:
        print("Need to specify one of a flag, -r -p -e -d or -s. -e builds the final model already!")

