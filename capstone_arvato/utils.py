import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
 

# ======================================================================================
# DATA DIR ANALYSIS FUNCTIONS

# Function to insert row in the dataframe 
def Insert_row_(row_number, df, row_value): 
    # Slice the upper half of the dataframe 
    df1 = df[0:row_number] 
   
    # Store the result of lower half of the dataframe 
    df2 = df[row_number:] 
   
    # Inser the row in the upper half dataframe 
    df1.loc[row_number]=row_value 
   
    # Concat the two dataframes 
    df_result = pd.concat([df1, df2]) 
   
    # Reassign the index labels 
    df_result.index = [*range(df_result.shape[0])] 
   
    # Return the updated dataframe 
    return df_result 


def insert(df , row):
    
    first_index = row[0]
    row_series = row[1]
    
#     we need to generate 2 rows from this single row and then insert it into the df
    info_level = row_series['Information level']
    additional_notes = row_series['Additional notes']
    
    attribute_1,attribute_2 = list(filter(lambda x: x != '',row_series.Attribute.split('  '))) 
    title,time = row_series.Description.split('last')
    time_1 , time_2 = time.split('and')
    description_1 = title + 'last' + time_1 + 'months'
    description_2 = title + 'last' + time_2
 
    
    new_row_1 = [info_level,attribute_1.replace(" ",""),description_1 , additional_notes]
    
    df.loc[first_index] = pd.Series({key:value for key,value in zip(df.loc[first_index].index.tolist(),new_row_1) })
    
    new_row_2 = [info_level,attribute_2.replace(" ",""),description_2 , additional_notes]
    df = Insert_row_(first_index+1,df,new_row_2)
    
    return df

def clean_dias_attr(dias_attr):
    dias_attr_mapping = dict()
    var_type = {0:'Categorical',1:'Numerical'}
    for k,v in dias_attr.fillna(method='ffill').groupby(['Attribute','Description']):
    
        val_mean_dict = v.drop(['Attribute','Description'],axis =1).reset_index(drop=True).to_dict()
        mapping = { val_mean_dict['Value'][i]:val_mean_dict['Meaning'][i]  for i in range(len(v))}
        
        isNumeric = 1 if 'numeric' in list(mapping.values())[0] else 0
        
        dias_attr_mapping[k[0]] = {'description':k[1] ,'mapping':mapping ,'var_type':var_type[isNumeric]}
        
    return dias_attr_mapping

#======================================================================================================

            
            

#======================================================================
# DATAFRAME OPTIMIZERS
def optimize_dtypes(df,file_name = 'column_types_optimized',save = False):
    
    floats = df.select_dtypes(include=['float64','float32']).columns.tolist()
    df[floats] = df[floats].apply(pd.to_numeric, downcast='float')
    
    
    ints = df.select_dtypes(include=['int64','int32']).columns.tolist()
    df[ints] = df[ints].apply(pd.to_numeric, downcast='integer')

    if save:
        dtypes = df.dtypes
        colnames = dtypes.index
        types = [i.name for i in dtypes.values]
        column_types_optimized = dict(zip(colnames, types))

        outfile = open(file_name,'wb')
        pickle.dump(column_types_optimized,outfile)
        outfile.close()
    
    return df


def optimize(df,df_name,save=False):
    missing = df.notna().sum()
    non_na_cols = missing[missing == len(df)].index.tolist()
    f_32_cols = df[non_na_cols].dtypes[df[non_na_cols].dtypes == 'float32'].index.tolist()
    df[f_32_cols] = df[f_32_cols].astype(int)
    df = optimize_dtypes(df,f'prescale_opt_{df_name}',save)
    if save:
        df.to_pickle(f'{df_name}_pre_scale.pkl')
    return df


#================================================================================
# CLASSIFIERS
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
import shap
from sklearn.utils import class_weight

def get_weight_array(y_train):
    class_weights = list(class_weight.compute_class_weight('balanced',np.unique(y_train),y_train))
    weight_dict = dict(zip(np.unique(y_train),class_weights))
    w_array = [weight_dict[val] for val in y_train]
    
    return w_array

def get_shap(model,training_set,cols):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(training_set)

    shap.summary_plot(shap_values, cols, plot_type="bar")
    vals= np.abs(shap_values).mean(0)
    feature_importance = pd.DataFrame(list(zip(cols, vals)), columns=['col_name','feature_importance_vals'])
    feature_importance.sort_values(by=['feature_importance_vals'], ascending=False,inplace=True)
    
    return shap_values,feature_importance


def run_xgb_classifier(X,y,shap=False,eval_metric = 'auc',test_size = 0.2,params=dict(),CV =False,eval_set=None):
    cols = X.columns.tolist()
    if not CV:
        X_train, X_val, y_train, y_val  = train_test_split( X,y,stratify=y, test_size=test_size, random_state = 1)
    else:
        X_train = X
        y_train = y
        X_val = eval_set[0]
        y_val = eval_set[1]
    
    
    base_params = {
        "tree_method":"gpu_hist",
        "n_estimators":1000, 
        "gpu_id":0, 
        "random_state":1
        }
    
    base_params = {**base_params,**params}
    
    
    model = XGBClassifier(**base_params)
    model.fit(X_train,y_train,early_stopping_rounds = 100,eval_set=[(X_val,y_val)],eval_metric =eval_metric ,verbose=True, sample_weight = get_weight_array(y_train) )
    
    
    if shap:
        shap_values,feature_importance = get_shap(model,X,cols)

        return shap_values,feature_importance.loc[feature_importance.feature_importance_vals>0],model
    else:
        return model

    
    
def run_lgbm_classifier(X,y,shap=False,eval_metric = 'auc',test_size = 0.2,params=dict(),CV =False,eval_set=None):
    cols = X.columns.tolist()
    if not CV:
        X_train, X_val, y_train, y_val  = train_test_split( X,y,stratify=y, test_size=test_size, random_state = 1)
    else:
        X_train = X
        y_train = y
        X_val = eval_set[0]
        y_val = eval_set[1]
    
    base_params = {
        "device":"gpu",
        "n_estimators":1000,  
        "random_state":1
        }
    
    base_params = {**base_params,**params}
    
    
    model = LGBMClassifier(**base_params)
    model.fit(X_train,y_train,early_stopping_rounds = 1000,eval_set=(X_val,y_val),eval_metric=eval_metric, sample_weight = get_weight_array(y_train) )
    
    
    if shap:
        shap_values,feature_importance = get_shap(model,X,cols)

        return shap_values,feature_importance.loc[feature_importance.feature_importance_vals>0],model
    else:
        return model    

    

def run_cb_classifier(X,y,shap=False,eval_metric = 'AUC',test_size = 0.2,params=dict(),CV =False,eval_set=(None,None)):
    cols = X.columns.tolist()
    cat_cols = [X.columns.get_loc(col) for col in X.dtypes[X.dtypes == 'category'].index.tolist()]
    if not cat_cols:
        cat_cols = None
    if not CV:
        X_train, X_val, y_train, y_val  = train_test_split( X,y,stratify=y, test_size=test_size)
    else:
        X_train = X
        y_train = y
        X_val = eval_set[0]
        y_val = eval_set[1]
    
    base_params = {
    "task_type":"GPU",
    "iterations":1000, 
    "early_stopping_rounds":100, 
    "eval_metric":eval_metric,
    "random_state":1
        
    }
    
    base_params = {**base_params,**params}
    

    model = CatBoostClassifier(**base_params)
    model.fit(X_train,y_train,cat_features=cat_cols,eval_set=(X_val,y_val))
    
    if shap:
        shap_values,feature_importance = get_shap(model,X,cols)

        return shap_values,feature_importance.loc[feature_importance.feature_importance_vals>0],model
    else:
        return model


#============================================================================
# DATA PROCESSING

def inOneOrAnother(name_1,name_2):
    return (name_1 in name_2) or (name_2 in name_1)

def rename_features(df,mapping_df):
    missing_cols = set(mapping_df.Attribute) - set(df.columns)
    undoc_cols = set(df.columns) - set(mapping_df.Attribute)
    
    rename_dict = dict()
    for missing_col in missing_cols:
        for undoc_col in undoc_cols:
            if inOneOrAnother(missing_col,undoc_col):
                rename_dict[undoc_col] = missing_col
                
    df = df.rename(columns = rename_dict)
    
    rename_dict = {'CAMEO_INTL_2015':'CAMEO_DEUINTL_2015',
               'D19_BUCH_CD':'D19_BUCH_RZ',
               'SOHO_KZ':'SOHO_FLAG',
              'KBA13_CCM_1401_2500':'KBA13_CCM_1400_2500'}
    
    df = df.rename(columns = rename_dict) 
    
    return df


from sklearn.impute import KNNImputer,SimpleImputer
# UDF's
def get_missing_cols(df,show=False):
    '''
    get features with missing data and the percentage of values missing 
    
    inputs:
        Dataframe
    returns:
        Series of columns with missing values and the percentage of missing values if greater than 0
    '''
    
    missing_percentage = (df.isna().sum() / len(df))*100
    missing_percentage_sorted = missing_percentage.sort_values(ascending=False)
    if show:
        display(missing_percentage_sorted[missing_percentage_sorted > 0])
        print(len(missing_percentage_sorted[missing_percentage_sorted > 0]))
    
    return missing_percentage_sorted[missing_percentage_sorted > 0]

def fill_unknowns_to_nan(main_df,mapping_df):
    '''
    Replaces mappings of unknowns with nans
    '''
    search_df = mapping_df.set_index(['Attribute'])
#     cycle through the dfs columns
    for col in main_df.columns:
        try:
#             try to get the map from the mapping df
            mapp = search_df.loc[col].mapping  
            
        except KeyError:
#             if the map does not exist
            main_df[col] = main_df[col].replace(-1,np.nan)
            continue
#         map exists , invert the mapping so we can index using unknown 
        try:
            inv_map = {v: k for k, v in mapp.items()}
        except:
            continue
            
        if 'unknown' in inv_map.keys():
#            if unknown is found which is usually -1 or 0 or 9
            if inv_map['unknown'] == -1 or inv_map['unknown'] == 0 or inv_map['unknown'] == 9:
                
                main_df[col] = main_df[col].replace(inv_map['unknown'],np.nan)
#             if unknown contains 2 values eg '-1,9'
            else:
#                 convert to list of ints then check which unknown mapping appears in the col
                unknown_vals = list(map(int,inv_map['unknown'].split(',')))
                unique_vals = main_df[col].unique()
                detected_unknown_val = list(set(unique_vals).intersection(set(unknown_vals)))
#               
                if len(detected_unknown_val) > 1:
                    print('multiple unknown vals in',col)
                if detected_unknown_val: 
#                     if a value is found
                    main_df[col] = main_df[col].replace(detected_unknown_val[0],np.nan)
        else:
            main_df[col] = main_df[col].replace(-1,np.nan)
            
    return main_df

        
    
def fill_nan_to_unknowns(main_df,mapping_df):
    '''
    Replaces mappings of nans with unknowns
    '''
    search_df = mapping_df.set_index(['Attribute'])
#     cycle through the dfs columns
    for col in main_df.columns:
        try:
#             try to get the map from the mapping df
            mapp = search_df.loc[col].mapping
            
        except KeyError:
#             if the map does not exist
            continue
#         map exists , invert the mapping so we can index using unknown
        try:
            inv_map = {v: k for k, v in mapp.items()}
        except:
            continue
            
        if 'unknown' in inv_map.keys():
#            if unknown is found which is usually -1 or 0 
            if inv_map['unknown'] == -1 or inv_map['unknown'] == 0:
                
                main_df[col] = main_df[col].replace(np.nan,inv_map['unknown'])
#             if unknown contains 2 values eg '-1,9'
            else:
#                 convert to list of ints then check which unknown mapping appears in the col
                unknown_vals = list(map(int,inv_map['unknown'].split(',')))
                default_fill = list(unknown_vals)[0]
                unique_vals = main_df[col].unique()
                detected_unknown_val = list(set(unique_vals).intersection(set(unknown_vals)))
#                 
                if detected_unknown_val: 
#                     if a value is found
                    main_df[col] = main_df[col].replace(np.nan,detected_unknown_val[0])
                else:
                    main_df[col] = main_df[col].replace(np.nan,default_fill)
                
                    
        elif 'no transactions known' in inv_map.keys():
            main_df[col] = main_df[col].replace(np.nan,inv_map['no transactions known'])
        else:
            pass
            
            
    return main_df


def get_mismatched_mappings(main_df,mapping_dict):
    '''
    get columns where a value in the column is not present in the mapping
    
    '''
#     search_df = mapping_df.set_index(['Attribute'])
    miss_matched = list()
    for col in main_df.columns:
        try:
#             try to get the map from the mapping df
            mapp = mapping_dict[col]['mapping']
        
        except KeyError:
#             if the map does not exist
            continue
        try:
            inv_map = {v: k for k, v in mapp.items()}
        except:
            continue
            
        vals_in_col = main_df[col].value_counts().index.tolist()
        

        if 'unknown' not in inv_map.keys():
            
            mismatched_values = set(vals_in_col) - set(mapp.keys())
            if mismatched_values and mapping_dict[col]['var_type'] != 'Numerical':
                miss_matched += [col]
                
                
    return miss_matched
  
    
def fill_missing_method(df,method='simple',strategy = 'most_frequent'):
    '''
    compilation of imputation methods
    
    '''
    
    columns = df.columns.tolist()
    imputer = list()
    if method == 'simple':
        df.fillna(-1,inplace=True)  
    elif method == 'simple_impute':
        imputer = SimpleImputer(strategy=strategy)
        df[columns] = imputer.fit_transform(df.values)
    return df,imputer

def get_matched_columns(df,column_1_prefix,column_2_prefix):
    '''
    get columns with prefix 2 that match columns with prefix 1
    '''
    
    column_list_1 = list(filter(lambda x:column_1_prefix in x,df.columns))
    colum_list_2 = list(filter(lambda x:column_2_prefix in x,df.columns))
    
    matched_columns = list()
    for column_1 in column_list_1:
        column_1_stripped = column_1.replace(column_1_prefix,'')
        for column_2 in colum_list_2:
            column_2_stripped =column_2.replace(column_2_prefix,'')
            if (column_2_stripped in column_1_stripped):
                matched_columns += [column_2]
    return matched_columns



from scipy import stats

def get_dist_similarity(df,col,group_column):
    '''
    get statistical difference in distribution betwen columns grouped by some other binary feature
    '''
    merged = pd.merge(
        df.loc[getattr(df,group_column) == 0 , col].value_counts().sort_index().rename('population'),
        df.loc[getattr(df,group_column) == 1 , col].value_counts().sort_index().rename('customer'),
        left_index=True,
        right_index=True)

    return stats.ks_2samp(merged.customer, merged.population)[1]
    

def show_diff(df,col,group_column = 'customer_identifier' ):
    '''
    show the difference in the distribution of values for a feature based on group_column
    '''
    fig,ax =plt.subplots(ncols = 2, figsize = (15,5))
    
    gb = df.groupby(group_column)
    print(col)
    print(get_dist_similarity(df,col,group_column))
    getattr(gb.get_group(1),col).value_counts().sort_index().plot(kind='bar',ax= ax[0],title = f'{group_column} = {1}' , figsize = (20,5))
    getattr(gb.get_group(0),col).value_counts().sort_index().plot(kind='bar',ax = ax[1],title = f'{group_column} = {0}',figsize = (20,5))

    display(fig)
    plt.close()
    
    
    
def display_corr(df):
    '''
    shows correlation matrix with bg colored
    '''
    mask = np.tril(df)
    fig,ax = plt.subplots(figsize = (15,10))
    display( sns.heatmap(df,fmt ='.1g',annot =True, cmap ='coolwarm',mask=mask ,ax=ax) )



def preprocessing_cluster_1(df,cleaned_mapping_df,attr_by_info_lvl):
    
    df = fill_unknowns_to_nan(df,cleaned_mapping_df)
    
    df.ARBEIT = df.ARBEIT.replace(9,np.nan)
    df[['LP_FAMILIE_FEIN', 'LP_FAMILIE_GROB', 'LP_LEBENSPHASE_FEIN', 'LP_LEBENSPHASE_GROB']] = df[['LP_FAMILIE_FEIN', 'LP_FAMILIE_GROB', 'LP_LEBENSPHASE_FEIN', 'LP_LEBENSPHASE_GROB']].replace(0,np.nan)
    
    df.drop(['ALTER_KIND1','ALTER_KIND2','ALTER_KIND3','ALTER_KIND4' ],axis=1,inplace=True)
    
    # drop plz 8 features in favor of microcell features seems to be many features explaining the same thing
    plz8_cols = list(set(attr_by_info_lvl['PLZ8']).intersection(set(df.columns))) 
    df.drop( plz8_cols  ,axis=1,inplace =True)
    kba13_remaining = list(filter(lambda x:'KBA13' in x,df.columns))
    df.drop( kba13_remaining  ,axis=1,inplace =True)

    # will also drop kba05 features as there seems to me alot of redundent and noisy information as well as many unknown values
    kba05_cols = list(set(attr_by_info_lvl['Microcell (RR3_ID)']).intersection(set(df.columns))) 
    df.drop( kba05_cols  ,axis=1,inplace =True)
    
    # will drop all undocumented columns, we may loose predictability however we will not loose explainability, and trying to fill nans in undocumented columns may also introduce unwanted bias in 
    # the model, better to be more explainable and a little less predictable than very predictable and not very explainable after all the value for this part is from the insights we can extract and explain
    undoc_cols = set(df.columns) - set(cleaned_mapping_df.Attribute)
    df.drop( undoc_cols  ,axis=1,inplace =True)
    
    # SORT CAMEO COLUMNS OUT
    df.drop(['CAMEO_DEU_2015'],axis =1,inplace =True) 
    # redundant high cardinality column 

    df.CAMEO_DEUG_2015.replace('X',-1,inplace= True)
    df.CAMEO_DEUG_2015.fillna(-1,inplace= True)
    df.CAMEO_DEUG_2015 = df.CAMEO_DEUG_2015.apply(lambda x: int(float(x)))
    

    df.CAMEO_DEUINTL_2015.replace('XX',-1,inplace= True)
    df.CAMEO_DEUINTL_2015.fillna(-1,inplace= True)
    df.CAMEO_DEUINTL_2015 = df.CAMEO_DEUINTL_2015.apply(lambda x: int(float(x)))
    
    
    return df



def preprocessing_cluster_2(df,cleaned_mapping_df,attr_by_info_lvl):
    
    df = fill_nan_to_unknowns(df,cleaned_mapping_df)
    
    df[['LP_FAMILIE_GROB','LP_STATUS_GROB']]=df[['LP_FAMILIE_GROB','LP_STATUS_GROB']].replace(0,np.nan)

    missing_vals_azdias = get_missing_cols(df)
    
    df.D19_LETZTER_KAUF_BRANCHE = df.D19_LETZTER_KAUF_BRANCHE.fillna('D19_UNBEKANNT')

    D19_columns = list(filter(lambda x:'D19' in x ,missing_vals_azdias.index.tolist()))
    D19_columns.remove('D19_KONSUMTYP')
    df.D19_KONSUMTYP = df.D19_KONSUMTYP.fillna(9)
    df[D19_columns] = df[D19_columns].fillna(0)

    df.ANZ_KINDER = df.ANZ_KINDER.fillna(0)
    # ARBEIT has unknown value of 9 
    df.ARBEIT = df.ARBEIT.fillna(-1)
    #  where KONSUMNAEHE tells us that that record is a consumtion cell we can fill consumption cell flag as 1
    # from consumption cell 1: 'building is located in a 125 x 125m-grid cell (RA1), which is a consumption cell'
    df.loc[df.KONSUMZELLE.isna() & (df.KONSUMNAEHE == 1), 'KONSUMZELLE' ] = 1
    df.loc[df.KONSUMZELLE.isna() & (df.KONSUMNAEHE > 1), 'KONSUMZELLE' ] = 0

    df.TITEL_KZ = df.TITEL_KZ.map({-1:-1, 1:1,2:1,3:2,4:2,5:3})
    df.OST_WEST_KZ = df.OST_WEST_KZ.map({-1:-1,'W':1,'O':0})
    
  
    remap = cleaned_mapping_df.loc[cleaned_mapping_df.Attribute == 'D19_LETZTER_KAUF_BRANCHE','mapping'].iloc[0]
    
    remap = {cat:cat_code for cat_code,cat in remap.items()}
    df.D19_LETZTER_KAUF_BRANCHE = df.D19_LETZTER_KAUF_BRANCHE.map(remap)
    

    # droping these columns either due to high cardinality , other columns containing same or similar info that are more complete 
    df.drop(['EINGEFUEGT_AM','ALTERSKATEGORIE_FEIN','MIN_GEBAEUDEJAHR','EINGEZOGENAM_HH_JAHR'],axis=1,inplace=True)
    # will drop this in favor of ANZ_STATISTISCHE_HAUSHALTE as it seems more calculated and correct
    df.drop(['ANZ_HAUSHALTE_AKTIV'],axis=1,inplace=True)
   
    anz_stat_func = lambda x: 1 if x==1 else 2 if (x >= 2 and x<=3) else 3 if (x >=4 and x<=8) else 4 if x>8 else np.nan 
    df.ANZ_STATISTISCHE_HAUSHALTE = df.ANZ_STATISTISCHE_HAUSHALTE.apply(anz_stat_func)
   
    anz_person_func = lambda x: 1 if x==1 else 2 if x==2 else 3 if x > 2 else np.nan 
    df.ANZ_PERSONEN = df.ANZ_PERSONEN.apply(anz_person_func)
    
    anz_hh_titel_func = lambda x: 0 if x==0 else 1 if x==1 else 2 if x==2 else 3 if x>2 else np.nan 
    df.ANZ_HH_TITEL = df.ANZ_HH_TITEL.apply(anz_hh_titel_func)

    # will drop all these houshold transaction features as I cannot interpret them and they seem to be highly correlated
    household_transactions = list(filter(lambda x: 'D19_' in x , attr_by_info_lvl['Household']))
    # except for these 
    household_transactions.remove('D19_KK_KUNDENTYP')
    household_transactions.remove('D19_KONSUMTYP')
    household_transactions.remove('D19_LETZTER_KAUF_BRANCHE')
    df.drop(household_transactions,axis=1,inplace=True)
    
    fein_cols = list(filter(lambda x:'_FEIN' in x ,df.columns))
    df.drop(fein_cols+['LP_LEBENSPHASE_GROB'],axis=1,inplace=True)
    
    
    missing_vals_azdias = get_missing_cols(df)
    display(df.info())
    print('optimizing...')
    non_na_cols = list(set(df.columns) -  set(missing_vals_azdias.index.tolist()))
    df[non_na_cols] = optimize(df[non_na_cols],'',False)
    display(df.info())
    
    
    return df


# custom imputer
def train_xgb(train,target,eval_metric = 'auc',objective = 'binary:logistic',test_size = 0.2):
     
#     X, X_val, y, y_val = train_test_split( train,target,stratify=target, test_size=test_size, random_state = 1)
    
    model = XGBClassifier(tree_method='gpu_hist',n_estimators=200, gpu_id=0, eval_metric=eval_metric,objective=objective,random_state = 1 )
    model.fit(train,target,early_stopping_rounds = 10,eval_set =[(train,target)],sample_weight = get_weight_array(target))
    
    return model

def xgb_imputer(df,verbose = 0 ):
    '''
    predict nans in a feature using all other features with xgboost model
    '''    
    
    cols_with_missing = get_missing_cols(df,verbose)
    if cols_with_missing.empty:
        print("No features with missing values")
        return 0
    
    for col in cols_with_missing.index.tolist():
        
        print(f'imputing {col}...')
        
        n_classes = getattr(df,col).nunique()
        objective = 'binary:logistic' if n_classes == 2 else 'multi:softmax'
        eval_metric = ['auc','error'] if n_classes == 2 else ['mlogloss','merror']
        
        print(f'{"Binary" if n_classes == 2 else "Multiclass"} classification')
            
        test_set = df.loc[getattr(df,col).isna()]
        test_X = test_set.drop([col],axis=1)
        test_idx = test_set.index.tolist()
        
        train_set = df.loc[getattr(df,col).notna()]
        train_y = getattr(train_set,col)
        train_X = train_set.drop([col],axis=1)
        
        if verbose:
            display(train_y.value_counts(normalize=True)*100)

            
        model = train_xgb(train_X,train_y,eval_metric=eval_metric,objective=objective)
        
        predictions = model.predict(test_X,ntree_limit = model.get_booster().best_ntree_limit)
        
        if verbose:
            print('predictions:')
            display(pd.Series(predictions).value_counts())
            
            
        df.loc[test_idx,col] = predictions
        print(f'done {col}')
        
    return df
    
    
def feature_engineering(df):
    
    df['age'] = df.GEBURTSJAHR.apply(lambda x:x if x==0 else (2020-x))
    df.drop(['GEBURTSJAHR'],axis=1,inplace=True)
    
    generation_map = {-1:-1 , 1:1, 2:1, 3:2, 4:2, 5:3, 6:3, 7:3, 8:4, 9:4, 10:5, 11:5, 12:5, 13:5, 14:6, 15:6}
    df['generation'] = df.PRAEGENDE_JUGENDJAHRE.map(generation_map)
    df.drop(['PRAEGENDE_JUGENDJAHRE'],axis = 1 , inplace = True)
    
    df.loc[(df.generation == -1) & (df.age != 0  ),'generation'] = 5
    df.drop(['ALTERSKATEGORIE_GROB'],axis=1,inplace=True)
    
    df.LP_STATUS_GROB =df.LP_STATUS_GROB.astype('category')
    df.LP_FAMILIE_GROB =df.LP_FAMILIE_GROB.astype('category')
    
    finanz_cols = df.filter(like='FINANZ_').columns.tolist()
    df.drop(finanz_cols,axis=1,inplace=True)
    df.FINANZTYP =df.FINANZTYP.astype('category') 
    
    df['HH_wealth']= df.CAMEO_DEUINTL_2015.apply(lambda x: x//10) # extract the units
    df['HH_family_type'] = df.CAMEO_DEUINTL_2015.apply(lambda x: x - (x//10)*10 if x != -1 else -1 ) #
    df.drop(['CAMEO_DEUINTL_2015'],axis = 1 , inplace = True)


    df['area_type'] = df.WOHNLAGE.map( {-1:-1,0:0,1:0,2:0,3:0,4:0,5:0,7:1,8:1})
    df['neighbourhood_class'] = df.WOHNLAGE.map({-1:-1,0:0,1:1,2:2,3:3,4:4,5:5,7:0,8:0})
    df.drop(['WOHNLAGE'],axis = 1 , inplace = True)

    df.CJT_GESAMTTYP = df.CJT_GESAMTTYP.astype('category')
    
    df.D19_KK_KUNDENTYP = df.D19_KK_KUNDENTYP.astype('category')
    
    df.drop(['age'],axis=1,inplace=True)
    
    cat_cols = df.dtypes[df.dtypes == 'category'].index.tolist()
    df[cat_cols] = df[cat_cols].astype(int)
    
    df.drop(['HH_wealth'],axis=1,inplace=True)
    
    df.info()
    print('optimizing')
    df = optimize(df,'')
    df.info()
    
    return df

def get_corr(df,corr_percent = 0.8):
    
    pos_dict  =dict()
    neg_dict = dict()
    
    corr_matrix = df.corr()
    so = corr_matrix.unstack().sort_values(kind="quicksort")
    
    pos_corr_cols = so.loc[(so>corr_percent) & (so != 1)].drop_duplicates().index.tolist()
    pos_corr = so.loc[(so>corr_percent) & (so != 1)].drop_duplicates().values.tolist()
    pos_dict['columns'] = pos_corr_cols
    pos_dict['values'] = pos_corr
    
    neg_corr_cols = so.loc[(so <= corr_percent) & (so != 1)].drop_duplicates().index.tolist()
    neg_corr = so.loc[(so <= corr_percent) & (so != 1)].drop_duplicates().values.tolist()
    neg_dict['columns'] = neg_corr_cols
    neg_dict['values'] = neg_corr
    
    return pos_dict,neg_dict

import joblib
import ppscore as pps

def gen_clusters():
    mailout_trans = pd.read_pickle('mailout_custom.pkl')
    mca = joblib.load('mca.pkl')
    km = joblib.load('km.pkl')
    mailout_trans= mca.transform(mailout_trans) 
    return km.predict(mailout_trans)


def minimal(df,cleaned_mapping_df):
    # SORT CAMEO COLUMNS OUT
    df.drop(['CAMEO_DEU_2015'],axis =1,inplace =True) 
    # redundant high cardinality column 

    df.CAMEO_DEUG_2015.replace('X',-1,inplace= True)
    df.CAMEO_DEUG_2015.fillna(-1,inplace= True)
    df.CAMEO_DEUG_2015 = df.CAMEO_DEUG_2015.apply(lambda x: int(float(x)))
    

    df.CAMEO_DEUINTL_2015.replace('XX',-1,inplace= True)
    df.CAMEO_DEUINTL_2015.fillna(-1,inplace= True)
    df.CAMEO_DEUINTL_2015 = df.CAMEO_DEUINTL_2015.apply(lambda x: int(float(x)))
    

    df.D19_LETZTER_KAUF_BRANCHE = df.D19_LETZTER_KAUF_BRANCHE.fillna('D19_UNBEKANNT')
    df.OST_WEST_KZ = df.OST_WEST_KZ.map({-1:-1,'W':1,'O':0})
    remap = cleaned_mapping_df.loc[cleaned_mapping_df.Attribute == 'D19_LETZTER_KAUF_BRANCHE','mapping'].iloc[0]

    remap = {cat:cat_code for cat_code,cat in remap.items()}
    df.D19_LETZTER_KAUF_BRANCHE = df.D19_LETZTER_KAUF_BRANCHE.map(remap)
    df.drop(['EINGEFUEGT_AM','ALTERSKATEGORIE_FEIN','ALTERSKATEGORIE_GROB','MIN_GEBAEUDEJAHR','EINGEZOGENAM_HH_JAHR','ANZ_HAUSHALTE_AKTIV','KBA13_ANZAHL_PKW'],axis=1,inplace=True)

    generation_map = {-1:-1 , 1:1, 2:1, 3:2, 4:2, 5:3, 6:3, 7:3, 8:4, 9:4, 10:5, 11:5, 12:5, 13:5, 14:6, 15:6}
    df['generation'] = df.PRAEGENDE_JUGENDJAHRE.map(generation_map)
 
    df['HH_family_type'] = df.CAMEO_DEUINTL_2015.apply(lambda x: x - (x//10)*10 if x != -1 else -1 )
    df.drop(['CAMEO_DEUINTL_2015'],axis = 1 , inplace = True)
    df['area_type'] = df.WOHNLAGE.map({0:0,1:0,2:0,3:0,4:0,5:0,7:1,8:1})
    df['neighbourhood_class'] = df.WOHNLAGE.map({0:0,1:1,2:2,3:3,4:4,5:5,7:0,8:0})
    
    anz_stat_func = lambda x: 1 if x==1 else 2 if (x >= 2 and x<=3) else 3 if (x >=4 and x<=8) else 4 if x>8 else np.nan 
    df.ANZ_STATISTISCHE_HAUSHALTE = df.ANZ_STATISTISCHE_HAUSHALTE.apply(anz_stat_func)
   
    anz_person_func = lambda x: 1 if x==1 else 2 if x==2 else 3 if x > 2 else np.nan 
    df.ANZ_PERSONEN = df.ANZ_PERSONEN.apply(anz_person_func)
    
    anz_hh_titel_func = lambda x: 0 if x==0 else 1 if x==1 else 2 if x==2 else 3 if x>2 else np.nan 
    df.ANZ_HH_TITEL = df.ANZ_HH_TITEL.apply(anz_hh_titel_func)

    
    missing_vals_azdias = get_missing_cols(df)
    display(df.info())
    print('optimizing...')
    non_na_cols = list(set(df.columns) -  set(missing_vals_azdias.index.tolist()))
    df[non_na_cols] = optimize(df[non_na_cols],'',False)
    display(df.info())
    
    return df

def drop_corr(df,target_column,correlation_percent = 0.8,return_cols=False):
    '''
    use predictive power score to choose which of the correlated columns to drop
    '''
    temp_df = df.copy()
    temp_df['target'] =  target_column
    pos,_ = get_corr(df,0.8)
    correlated_columns = pos['columns']
    cols_to_drop = []
    for pair in correlated_columns:
        col_1,col_2 = pair
        score_1 = pps.score(temp_df,col_1,'target')['ppscore']
        score_2 = pps.score(temp_df,col_2,'target')['ppscore']
        if score_1 > score_2:
            cols_to_drop.append(col_2)
        else:
            cols_to_drop.append(col_1)
            
    if return_cols:
        return df.drop(cols_to_drop,axis=1),cols_to_drop 
    return df.drop(cols_to_drop,axis=1)

def simple_fill(df,cleaned_mapping_df):
    num_cols = cleaned_mapping_df.loc[cleaned_mapping_df.var_type == 'Numerical'].Attribute.tolist()
    num_cols = set(df.columns).intersection(set(num_cols))
    cat_cols = cleaned_mapping_df.loc[cleaned_mapping_df.var_type != 'Numerical'].Attribute.tolist()
    cat_cols = set(df.columns).intersection(set(cat_cols))
    
    df[num_cols].fillna(0,inplace=True)
    df[cat_cols].fillna(-1,inplace=True)
    df.fillna(-1,inplace=True)
    return df






