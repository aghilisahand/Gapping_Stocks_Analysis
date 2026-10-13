import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression




class MultiStats:


   def __init__(self, mainDataset, columnNames, dependentFeat):
       self.dependentFeat = dependentFeat
       self.mainDataset = mainDataset
       self.columnNames = columnNames


       # deleting "__" from row names
       self.rowList = []
       for row in range(len(columnNames)):
           columnHeaders = columnNames['1DayRangePct'].str.split("__").str[0]
           self.rowList.append(columnHeaders[row])


       #creating lists for function use
       self.rSquaredList = []
       self.corCoefList = []
       self.xValList = []
       self.sigRSquaredList = []
       self.sigCorCoefList = []
       for xVal in self.rowList:
           if xVal != self.dependentFeat:
               self.xValList.append(xVal)




   def linRegression(cls, SigValue):
       for xVal in cls.rowList:
           if xVal != cls.dependentFeat:
               tmpDf = cls.mainDataset[[xVal, cls.dependentFeat]].copy(deep=True).dropna()
               corCoef = tmpDf[xVal].corr(tmpDf[cls.dependentFeat])
               cls.corCoefList.append(corCoef)
       # save correlation coefficients to folder
       corDirectory = "correlation_coefficients"
       os.makedirs(corDirectory, exist_ok=True)
       corCoefDf = pd.DataFrame({'variable name': cls.xValList, 'Correlation Coefficients': cls.corCoefList})
       corCoefDf.to_csv(os.path.join(corDirectory, cls.dependentFeat + '_corrCoef.txt'))
       #save significant values to separate txt file
       sigCorCoefDf = corCoefDf.loc[corCoefDf['Correlation Coefficients'] >= SigValue]
       sigCorCoefDf.to_csv(os.path.join(corDirectory, cls.dependentFeat + '_significantCorrCoef.txt'))


   def rSquared(cls, SigValue):
       for xVal in cls.rowList:
           if xVal != cls.dependentFeat:
               tmpDf =cls.mainDataset[[xVal, cls.dependentFeat]].copy(deep=True).dropna()
               model = LinearRegression()
               x, y = tmpDf[[xVal]], tmpDf[cls.dependentFeat]
               model.fit(x, y)
               r_squared = model.score(x, y)
               cls.rSquaredList.append(r_squared)
       #save r squared values in a folder
       rSquareDirectory = 'r-Squared_Values'
       os.makedirs(rSquareDirectory, exist_ok=True)
       rSquaredDf = pd.DataFrame({'variable name': cls.xValList, 'r-Squared value': cls.rSquaredList})
       rSquaredDf.to_csv(os.path.join(rSquareDirectory, cls.dependentFeat + '_r-Squared.txt'))
       #save significant r squared values to a new txt file
       sigRSquareDf = rSquaredDf.loc[rSquaredDf['r-Squared value'] >= SigValue]
       sigRSquareDf.to_csv(os.path.join(rSquareDirectory, cls.dependentFeat + '_significantR-Squared.txt'))




   def ScatterGraph(cls):


       # Graphing out the values
       for xVal in cls.rowList:
           if xVal != cls.dependentFeat:
               tmpDf = cls.mainDataset[[xVal, cls.dependentFeat]].copy(deep=True).dropna()


               # Changing data for better graph
               if cls.dependentFeat == "DayRet":
                   tmpDf = tmpDf[tmpDf[cls.dependentFeat] != -100]
                   tmpDf = tmpDf[tmpDf[cls.dependentFeat] < 100]
               else:
                   tmpDf = tmpDf[tmpDf[cls.dependentFeat] != -99999]
               tmpDf = tmpDf[tmpDf[xVal] != -99999]


               plt.scatter(tmpDf[xVal], tmpDf[cls.dependentFeat], label='Data', s=5)
               plt.title(cls.dependentFeat + " vs variables")
               plt.xlabel(xVal)
               plt.ylabel(cls.dependentFeat)
               plt.legend()


               # Add line of best fit
               plt.plot(np.unique(tmpDf[xVal]),
                        np.poly1d(np.polyfit(tmpDf[xVal], tmpDf[cls.dependentFeat], 1))(np.unique(tmpDf[xVal])),
                        color='red')
               plt.show()


               # Save the graphs
               directory = cls.dependentFeat + " Scatters"
               os.makedirs(directory, exist_ok=True)  # create the directory if it doesn't exist
               xValReplaced = xVal.replace(':', '_')   # some characters aren't allowed
               plt.savefig(os.path.join(directory, 'scatter_{}.png'.format(xValReplaced)))
               plt.close()
