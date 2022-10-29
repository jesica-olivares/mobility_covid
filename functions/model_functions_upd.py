import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import lightgbm as lgb
import xgboost

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns



def shift_df(df,days):
    """Will return a dataframe wiht a shift of t days, for the specified columns.

    Args:
        df: dataframe that will be shifted
        days: number of days to be shifted
        columns_shift: specific columns inside de df that will be shifted

    returns:
        the dataframe with yhe columns shifted in the specified number of days

    
    """
    df_i=df.set_index("date").copy()
    #df_i=df_i.loc[:,columns_shift]
    df_i=df_i.groupby('country_region').shift(periods=-days, freq='D')
    df_i=df_i.reset_index(level=0, drop=True)
    df_i=df_i.reset_index(drop=False)
    df_return=df.drop(columns="new_cases_smoothed_per_million").merge(df_i[["date","new_cases_smoothed_per_million","country_region"]], on=["date","country_region"])


    return df_return


def model_shift_df(df,days, model, areas):
    """Will model the shifted df, and return the r2 and mse metrics
    
    Args:
        df: dataframe to be modeled
        days: number of days to be shifted
        model: type of model used
        areas: columns to be shifted

    return: r2 and mse metrics of the model

    """
    df_shift=shift_df(df,days)
    df_shift=df_shift.dropna()
    #df_shift=df_shift.drop(columns=areas)
    X_train, X_test, y_train, y_test=train_test_split(df_shift.drop(columns=['new_cases_smoothed_per_million','date','country_region']), df_shift['new_cases_smoothed_per_million'], test_size=.33, shuffle=True)
    model = model
    model.fit(X_train, y_train)
    y_hat=model.predict(X_test)
    r2=r2_score(y_test,y_hat)
    mse=mean_squared_error(y_test,y_hat)
    return r2, mse

def run_models(model, df, iterations, areas):
    """run the model given model for the number of iterations, to obtain the average r2 and mse metrics for a shift between 0 - 35 fays
    
    Args:
        model: type of model used
        df: dataframe to be modeled
        iterations: number of times the model will be run to obtain the average
        model: type of model used
        areas: columns to be shifted

    return: a grapgh showing the average r2 and mse for the model 
    
    """
    r2_list2=[]
    mean_squared_error_list2=[]
    for j in list(range(1,iterations)):
        r2_list=[]
        mean_squared_error_list=[]
        shift_days=list(range(1,50))
        for i in shift_days:
            r2, mse= model_shift_df(df,i,model, areas )
            r2_list.append(r2)
            mean_squared_error_list.append(mse)
        r2_list2.append(r2_list)
        mean_squared_error_list2.append(mean_squared_error_list)
        df_r2=pd.DataFrame(r2_list2)
        df_mse=pd.DataFrame(mean_squared_error_list2)
    return shift_days, df_r2.mean(), df_mse.mean()
    


def graf_curvas_1_var(model,X_train,a,sim=10):
    j=0
    for i in X_train:
        j+=1
        globals()['x_%s' % j] = np.full( (1,sim), X_train.mean()[i]).ravel()
    globals()['X_%s' % str(a+1)] = np.linspace(X_train.min()[a], X_train.max()[a], sim)
    j=0
    lis_vstack=[]
    for i in X_train:
        j+=1
        if (j==a+1):
            aux=globals()['X_%s' % str(j)].ravel()
        else:
            aux=globals()['x_%s' % str(j)]
        lis_vstack.append(aux)
    predict_matrix = np.vstack(lis_vstack)
    prediction = model.predict(predict_matrix.T)
    prediction_plot = prediction.reshape(globals()['X_%s' % str(a+1)].shape)
    fig = plt.figure(figsize=(6,3))
    sns.set_style("whitegrid")
    sns.set_palette("Paired_r")
    palette=sns.color_palette()

    sns.lineplot(x=globals()['X_%s' % str(a+1)],y=prediction, linewidth = 1.5, color=palette[5])
    plt.xlabel(X_train.columns[a], fontsize=10)
    plt.grid(False)

    plt.show()

def graf_curvas_nivel(model,X_train,a,b,):
    j=0
    for i in X_train:
        j+=1
        globals()['x_%s' % j] = np.full( (10,10), X_train.mean()[i]).ravel()
    globals()['x_%s' % str(a+1)] = np.linspace(max(X_train.min()[a],X_train.mean()[a]- 3*X_train.std()[a]), min(X_train.max()[a],X_train.mean()[a]+ 3*X_train.std()[a]), 10)
    globals()['x_%s' %str(b+1)] = np.linspace(max(X_train.min()[b],X_train.mean()[b]- 3*X_train.std()[b]), min(X_train.max()[b],X_train.mean()[b]+ 3*X_train.std()[b]), 10)
    globals()['X_%s' % str(a+1)], globals()['X_%s' % str(b+1)] = np.meshgrid(globals()['x_%s' % str(a+1)], globals()['x_%s' % str(b+1)])
    j=0
    lis_vstack=[]
    for i in X_train:
        j+=1
        if (j==a+1) or (j==b+1):
            aux=globals()['X_%s' % str(j)].ravel()
        else:
            aux=globals()['x_%s' % str(j)]
        lis_vstack.append(aux)
    predict_matrix = np.vstack(lis_vstack)
    prediction = model.predict(predict_matrix.T)
    prediction_plot = prediction.reshape(globals()['X_%s' % str(a+1)].shape)
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111)
    mycmap1 = plt.get_cmap('YlGnBu') #viridis
    cp = plt.contourf(globals()['X_%s' % str(a+1)], globals()['X_%s' % str(b+1)], prediction_plot, 10, alpha=.8,cmap=mycmap1)
    plt.colorbar(cp)
    plt.title('{} in function {} of COVID cases'.format(X_train.columns[b],X_train.columns[a]))
    ax.set_xlabel(X_train.columns[a], fontsize=10)
    ax.set_ylabel(X_train.columns[b], fontsize=10)
    ax.patch.set_edgecolor('black')  
    #ax.patch.set_linewidth('2')  
    plt.show()
    
