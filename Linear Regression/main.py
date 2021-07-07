from sklearn import datasets

from sklearn import linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
boston=datasets.load_boston()

#features/labels

X=boston.data
y=boston.target




#algorithm
l_reg=linear_model.LinearRegression()
plt.scatter(X.T[12],y)
plt.show()

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
model=l_reg.fit(X_train,y_train)

predictions=model.predict(X_test)

print("predictions:  " , predictions)
print("coed :", l_reg.coef_)
print("intercept:  ",l_reg.intercept_)




