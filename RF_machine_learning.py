__author__ = 'mustafa_dogan'
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import ensemble, datasets, linear_model
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error
import pandas as pd
from math import sqrt, fabs, exp
import matplotlib.pyplot as plt
from pandas.tools.plotting import parallel_coordinates
from pylab import *
# import seaborn as sns
# sns.set_style("white")

# Read organized data. Organize data in a way that last column is labels and rest is attributes
data = pd.read_csv('data.csv', header=0)

# data summary (you need to print to see results - below)
summary = data.describe()

# Number of rows and columns in data
nDataRow = len(data.index)
nDataCol = len(data.columns)

# Normalize data
dataNormalized = data.iloc[:,0:nDataCol]
for i in range(nDataCol):
    mean = summary.iloc[1, i]
    sd = summary.iloc[2, i]
dataNormalized.iloc[:,i:(i+1)] = (dataNormalized.iloc[:,i:(i+1)]-mean)/sd

# True if you want to normalize data (preferred), False if not.
Normalization = True

if Normalization == True:
    data = dataNormalized

# Separate attributes and labels
xList = []
labels = []
names = data.keys()
for i in range(nDataRow):
    dataRow = list(data.iloc[i,0:nDataCol])
    xList.append(dataRow[0:nDataCol-1])
    labels.append(float(dataRow[nDataCol-1]))

# number of rows and columns in attributes
nrows = len(xList)
ncols = len(xList[0])

# store attributes (Xlist) and labels (y) in numpy arrays
X = np.array(xList)
y = np.array(labels)
Names = np.array(names)

#take fixed holdout set 30% of data rows
xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.30, random_state=531)

# ******* Random Forest Model *******
#train random forest at a range of ensemble sizes in order to see how the mse changes
mse = [] # mean squared error
nTreeList = range(100, 400, 20)
for iTrees in nTreeList:
    depth = None
    maxFeat  = 2 #try tweaking
    RFModel = ensemble.RandomForestRegressor(n_estimators=iTrees, max_depth=depth, max_features=maxFeat,
                                                 oob_score=False, random_state=531)
    RFModel.fit(xTrain,yTrain)
    #Accumulate mse on test set
    prediction = RFModel.predict(xTest)
    mse.append(mean_squared_error(yTest, prediction))


# ******* Penalized Linear Regression Model *******
# # least absolute shrinkage and selection operator regression
# LassoModel = LassoCV(cv=10,random_state=531).fit(X, y)
# prediction = LassoModel.predict(xTest)
# plt.figure()
# plt.plot(LassoModel.alphas_, LassoModel.mse_path_, ':')
# plt.plot(LassoModel.alphas_, LassoModel.mse_path_.mean(axis=-1),
#                 label='Average MSE Across Folds', linewidth=2)
# plt.axvline(LassoModel.alpha_, linestyle='--',
#                 label='CV Estimate of Best alpha')
# plt.semilogx()
# plt.legend()
# ax = plt.gca()
# ax.invert_xaxis()
# plt.xlabel('alpha')
# plt.ylabel('Mean Squared Error')
# plt.axis('tight')
# plt.savefig('lassoCVerror.pdf', transparent=True)
# plt.show()

# # Plot coefficients
# alphas, coefs, _  = linear_model.lasso_path(X, y,  return_models=False)
# plt.plot(alphas,coefs.T)
# plt.xlabel('alpha')
# plt.ylabel('Coefficients')
# plt.axis('tight')
# plt.semilogx()
# plt.legend(Names[0:len(Names)-1])
# ax = plt.gca()
# ax.invert_xaxis()
# plt.savefig('lassoCVcoeffs.pdf', transparent=True)
# plt.show()

# ******* Data Properties *******

# # describe data
# print(data.head())
# print(data.tail())
# print(summary)

# # parallel axis plot of input data
# parallel_coordinates(data, str(data.keys()[-1]), alpha=0.5)
# plt.gca().legend_.remove()
# plt.savefig('parallel.pdf', transparent=True)
# plt.show()

# # another way of creating parallel axis plot
# for i in range(nDataRow):
#     #plot rows of data as if they were series data
#     dataRow = data.iloc[i,0:nDataCol-1]
#     normTarget = data.iloc[i,nDataCol-1]
#     labelColor = 1.0/(1.0 + exp(-normTarget)) # logit function value
#     plt.plot(dataRow.values,color=plt.cm.RdYlBu(labelColor), alpha=0.5)
# plt.xlabel("Attribute")
# plt.ylabel("Attribute Values")
# plt.xticks(np.arange(len(Names)-1),Names[0:len(Names)-1])
# plt.savefig('parallel.pdf', transparent=True)
# plt.show()

# # calculate correlations between real-valued attributes
# corMat = pd.DataFrame(data.corr())
# # visualize correlations using heatmap
# plt.pcolor(corMat,vmin=-1,vmax=1)
# plt.yticks(np.arange(len(Names)),Names)
# plt.xticks(np.arange(len(Names)),Names)
# plt.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.1)
# cb = plt.colorbar()
# cb.set_label('Correlation')
# cb.set_ticks([1, 0, -1])  # force there to be only 3 ticks
# # cb.set_ticklabels(['High', 'No', 'Negative'])  # put text labels on them
# plt.savefig('corrmap.pdf', transparent=True)
# plt.show()

# # create a boxplot of data
# boxplot(data.iloc[:,0:nDataCol].values)
# plt.ylabel("Quartile Ranges")
# plt.xticks(np.arange(1,len(Names)+1),Names)
# plt.savefig('boxplot.pdf', transparent=True)
# show()

# # or with seaborn you can create a boxplot
# sns.boxplot(data=data)
# plt.ylabel("Quartile Ranges")
# plt.savefig('boxplot.pdf', transparent=True)
# plt.show()

# # scatter plot two variables to see how they look like
# # pick two variables, indexing starts from zero
# i = 0
# j = 1
# # print(data.keys()) # look at column names to choose what to plot
# plt.scatter(data[data.keys()[i]],data[data.keys()[j]])
# plt.xlabel(data.keys()[i])
# plt.ylabel(data.keys()[j])
# plt.savefig(str(data.keys()[i])+'_'+str(data.keys()[j])+'.pdf', transparent=True)
# plt.show()

# # ******* Performance indicators *******

# # plot training and test errors vs number of trees in ensemble
# plt.plot(nTreeList, mse, marker='o')
# plt.xlabel('Number of Trees in Ensemble')
# plt.ylabel('Mean Squared Error')
# plt.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.1)
# plt.savefig('error.pdf', transparent=True)
# plt.show()

# # plot histogram of errors
# errorVector = yTest-RFModel.predict(xTest)
# plt.hist(errorVector)
# plt.xlabel("Bin Boundaries")
# plt.ylabel("Counts")
# plt.savefig('histogram.pdf', transparent=True)
# plt.show()

# # Plot feature importance
# featureImportance = RFModel.feature_importances_
# # normalize by max importance
# featureImportance = featureImportance / featureImportance.max()
# sorted_idx = np.argsort(featureImportance)
# barPos = np.arange(sorted_idx.shape[0]) 
# plt.barh(barPos, featureImportance[sorted_idx], align='center')
# plt.yticks(barPos, Names[sorted_idx])
# plt.xlabel('Variable Importance')
# plt.subplots_adjust(left=0.3, right=0.9, top=0.9, bottom=0.1)
# plt.savefig('FeatureImportance.pdf', transparent=True)
# plt.show()

# # scatter plot predicted values vs test labels. it plots last prediction in nTreesList range
# diag = np.arange(-0,12) # adjust diagonal initial and ending values
# plt.plot(diag, diag, linestyle=':', color='red')
# plt.scatter(yTest,prediction)
# plt.xlim = [min(yTest),max(yTest)]
# plt.ylim = [min(prediction),max(prediction)]
# plt.xlabel('Test')
# plt.ylabel('Prediction')
# plt.savefig('TestvsPredict.pdf', transparent=True)
# plt.show()

