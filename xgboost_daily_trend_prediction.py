import pandas as pd
from sklearn import preprocessing
import numpy as np
from os import listdir
import time
from matplotlib import pyplot as plt
from scipy.stats import linregress
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve


def binary_class_classifier(current, future):

    if float(future) > float(current):
        return 1
    else:
        return 0
def preprocess_df(df):
    for col in df.columns:                                #
        if col != 'target' and col != 'Overnight_Return'and col!= 'ROC' and col!= 'ForceIndex' and col!= 'Momentum' and col!= 'Volatility' : # Overnight_return is already in pct_change format
            df[col] = df[col].pct_change()
    df = df.replace([np.inf, -np.inf], np.nan) # to replace the infinite numbers by NAN
    df.dropna(inplace = True) # to drop NAN
    partial_df = df.iloc[:,:-1] # data without target column
    partial_np_scaled = preprocessing.scale(partial_df) #scaled data
    scaled_df = pd.DataFrame(partial_np_scaled, columns = df.columns[:-1], index =partial_df.index)
    scaled_df['target'] = df['target'].values
    return scaled_df

def Tech_Indicators(df):

    #Volatility #10
    df['Volatility']= df['Close'].pct_change().rolling(10).std()
    
    #Create 10 days Moving Average
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    #Create Bollinger Bands()
    df['sd_using_Close'] = df['Close'].rolling(10).std()
    df['Upper_BB'] = df['SMA_10'] + (df['sd_using_Close']*2)
    df['Lower_BB'] = df['SMA_10'] - (df['sd_using_Close']*2)
    df = df.drop("sd_using_Close", axis =1)
    
    #Momentum  #10 #8  # 14 is even better  
    def momentum(Close):
        returns = np.log(Close)
        x = np.arange(len(returns))
        slope, _, rvalue, _, _ = linregress(x, returns)
        return ((1 + slope) ** 252) * (rvalue ** 2)
    df['Momentum'] = df['Close'].rolling(14).apply(momentum, raw=False)

    #Overnight returns
    df["Overnight_Return"] = df['Open']/df['Close'].shift(1)-1
    # Force Index #1
    df["ForceIndex"] = df['Close'].diff(1) * df['Volume']
    
    # Commodity Channel Index (CCI) #days =20 or 10(better) 5(even better)
    TP = (df['High'] + df['Low'] + df['Close']) / 3 
    df['CCI'] = (TP - TP.rolling(5).mean()) / (0.015 * TP.rolling(5).std())

    # Ease Of Movement (EVM) #days =14
    dm = ((df['High'] + df['Low'])/2) - ((df['High'].shift(1) + df['Low'].shift(1))/2)
    br = (df['Volume'] / 100000000) / ((df['High'] - df['Low']))
    EVM = dm / br 
    df["EVM"] = EVM.rolling(14).mean()

    # Rate of Change (ROC) #5
    N = df['Close'].diff(7)
    D = df['Close'].shift(7)
    df['ROC'] = N/D

    df.dropna(inplace =True)

    return df


#********************* Preparing the data **********************************************
df = pd.read_csv(f"AAPL.csv")
df.set_index('Date', inplace = True)
df.dropna(inplace = True)
df = df[['Open','Close', 'Volume','High', 'Low']]
df = Tech_Indicators(df)
df['future'] = df['Open'].shift(-1)
df['target'] = list(map(binary_class_classifier, df["Open"], df["future"]))
df.drop('future', axis = 1, inplace =True)
df.dropna(inplace = True)
df = preprocess_df(df)
splitting = int(0.80 * len(df))  #splitting ratio
X_y_train = df[:splitting] #80% training set 
X_y_test = df[splitting:]  #20% testing set
X_y_train = X_y_train.sample(frac=1, random_state =123) # we shuffle the training set ONLY 

X_train,y_train = X_y_train.iloc[:,:-1],X_y_train.iloc[:,-1]
X_test,y_test = X_y_test.iloc[:,:-1], X_y_test.iloc[:,-1]

#************************* Building the model **************************************************

model = xgb.XGBClassifier() 
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_pred_training = model.predict(X_train)
acc_score = accuracy_score(y_test,y_pred)
acc_score_training = accuracy_score(y_train,y_pred_training)

print('Training Accuracy',acc_score_training)
print('Testing Accuracy', acc_score)

#********************** Features Importance *************************************************************

xgb.plot_importance(model)
plt.rcParams['figure.figsize'] = [10, 5]
plt.show()

#************************ AUC ROC CURVE******************************************************************
model_proba = model.predict_proba(X_test)
model_proba = model_proba[:,1] #take only the probabilities that '1' is True
model_auc = roc_auc_score(y_test,model_proba)
print('Model AUC: ',model_auc)
model_fpr, model_tpr,_ =  roc_curve(y_test,model_proba)

random_proba =[0 for _ in range(len(y_test))] # we suppose that the random guesses are all 0
random_auc = roc_auc_score(y_test,random_proba)
random_fpr, random_tpr, _ = roc_curve(y_test, random_proba)

plt.plot(random_fpr,random_tpr,linestyle ='--' ,label ='Random Prediction (AUROC =%0.2f)'% random_auc)
plt.plot(model_fpr,model_tpr, label ='XGBoost (AUROC = %0.2f)'% model_auc)

plt.title('ROC Plot')
plt.xlabel('False Positive Rate (1-Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.legend()
plt.show()


