import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import seaborn as sns
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


def getCompanyData(paynet_dataFolder, kf_id, use_cols = None, numFiles = None):
    
    fileList = os.listdir(paynet_dataFolder)
    if '.DS_Store' in fileList:
        fileList.remove('.DS_Store')
        
    fileList = sorted(fileList, key = lambda x : int(x.split("_")[-1].split(".")[0]) )
    
    if numFiles:
        fileList = fileList[:numFiles]
        
        
    df = pd.DataFrame()
    
    for i in tqdm(range(len(fileList))):
        
        if use_cols:
            tempDf = pd.read_csv(os.path.join(paynet_dataFolder, fileList[i]), usecols=use_cols )
        else:
            tempDf = pd.read_csv(os.path.join(paynet_dataFolder, fileList[i]))

        tempDf = tempDf[tempDf.KF_ID==kf_id]

        df = pd.concat([df, tempDf])
       
        
        
    return df




def addPowers(X,num = 2):
    X_ = X.copy()
    
    for i in range(1,num):
        x = X_**(i+1)
        X = np.concatenate([X,x], axis =1)
        
    return X


def perform_skill_salary_analysis(df, salaryVariable, skillVariable, maxPower = 1, removeOutlier = True, plot = True, lowerLimit = 0.001, upperLimit = 0.999):
    
    
    if removeOutlier:
        

        subDf = df.copy()

        salaryLowLimit, salaryUpLimit = df[salaryVariable ].quantile(lowerLimit), df[salaryVariable ].quantile(upperLimit)
        skillLowLimit, skillUpLimit = df[skillVariable].quantile(lowerLimit), df[skillVariable].quantile(upperLimit)

        subDf = subDf[subDf[salaryVariable ].between(salaryLowLimit, salaryUpLimit)]
        subDf = subDf[subDf[skillVariable].between(skillLowLimit, skillUpLimit)]
        
        df = subDf.copy()
    
    # Removing ReferenceLevelNum values of 99
    if skillVariable=='ReferenceLevelNum':
        if max(df.ReferenceLevelNum) == 99:
            df = df[df.ReferenceLevelNum<99]
        
        
    lrModel = LinearRegression()

    X = df[skillVariable].values.reshape(-1,1)
    minX,maxX = int(min(X)), int(max(X))
    
    if maxPower>1:
        X = addPowers(X, maxPower)
        
    y = df[salaryVariable]

    lrModel.fit(X,y)

    print(f'Slope : {np.round(lrModel.coef_,2)} Intercept : {np.round(lrModel.intercept_,2)}')

    if plot:
        
        x = list(range(minX, maxX+1))
        
        if maxPower==1:    
            y = [i*lrModel.coef_ + lrModel.intercept_ for i in x]
        else:
            y = [i*lrModel.coef_[0]  + (i**2)*lrModel.coef_[1] + lrModel.intercept_ for i in x]
        

        sns.scatterplot(df[skillVariable], df[salaryVariable])
        plt.plot(x,y,'r')
    
    return lrModel

    

def removeOutliersYearwise(df, varList, upperLimit  = 0.95, lowerLimit = 0.05):
    
    subDfList = []
    
    outputDf = []
    
    for i in range(2008, 2020):
        subDf = df[df.CalendarYear == i]
        
        if len(subDf)==0:
            continue
        
        tempDf = subDf.copy()
                
        for var in varList:
            subDf = subDf[subDf[var].between(tempDf[var].quantile(lowerLimit) , tempDf[var].quantile(upperLimit) )]
            
        if len(subDf)==0:
            continue
            
        outputDf.append(subDf)
        
    outputDf = pd.concat(outputDf)
    
    return outputDf
        
        
        

def skillSalaryPlot(df, salaryVariable, skillVariable, salaryLimits = None, skillLimits = None):
    """
    y1 : Salary Variable
    y2 : Skill Variable
    """
    
    fig, ax1 = plt.subplots(figsize = (15,5))
    
    
    
    if salaryLimits is not None:
        if salaryLimits[0] is None:
            salaryLimits[0] = min(df[salaryVariable])
        if salaryLimits[1] is None:
            salaryLimits[1] = max(df[salaryVariable])
            
        df = df[df[salaryVariable].between(salaryLimits[0], salaryLimits[1])]
            
    if skillLimits is not None:
        if skillLimits[0] is None:
            skillLimits[0] = min(df[skillVariable])
        if skillLimits[1] is None:
            skillLimits[1] = max(df[skillVariable])
            
        df = df[df[skillVariable].between(skillLimits[0], skillLimits[1])]
            
    
    x = list(range(len(df)))
            

    df = df.sort_values(skillVariable)

    color = 'tab:red'
    ax1.set_xlabel('Samples')
    ax1.set_ylabel(salaryVariable, color=color)
    ax1.scatter(x, df[salaryVariable], color=color)
    ax1.tick_params(axis='y', labelcolor=color)


    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel(skillVariable, color=color)  # we already handled the x-label with ax1
    ax2.plot(x, df[skillVariable], color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()
    
    


def SalaryByYearPlot(df, salaryVariable, salaryLimits = None, countLimits = None):
    """
    y1 : Salary Variable
    y2 : Count Variable
    """
    
    fig, ax1 = plt.subplots(figsize = (15,5))
    
    
    out = df.groupby("CalendarYear").agg({salaryVariable:'mean','KF_ID':'count'})
    out = out.reset_index()
    out.columns = ['CalendarYear',salaryVariable,'numRows']
    
    df = out.copy()
    
    if salaryLimits is not None:
        if salaryLimits[0] is None:
            salaryLimits[0] = min(df[salaryVariable])
        if salaryLimits[1] is None:
            salaryLimits[1] = max(df[salaryVariable])
            
        df = df[df[salaryVariable].between(salaryLimits[0], salaryLimits[1])]
            
    if countLimits is not None:
        if countLimits[0] is None:
            countLimits[0] = min(df['numRows'])
        if countLimits[1] is None:
            countLimits[1] = max(df['numRows'])
            
        df = df[df['numRows'].between(countLimits[0], countLimits[1])]

            

    df = df.sort_values('CalendarYear')
    x = df.CalendarYear

    color = 'tab:red'
    ax1.set_xlabel('CalendarYear')
    ax1.set_ylabel(salaryVariable, color=color)
    ax1.plot(x, df[salaryVariable], color=color)
    ax1.tick_params(axis='y', labelcolor=color)


    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('# of Samples', color=color)  # we already handled the x-label with ax1
    ax2.plot(x, df['numRows'], color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()
    
    