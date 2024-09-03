# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

4.Assign the points for representing in the graph.

5.Predict the regression for marks by using the representation of the graph.

6.Compare the graphs and hence we obtained the linear regression for the given datas. 
 
## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: SARANYA S
RegisterNumber:  212223110044
*/
```
```
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt
```
```
dataset = pd.read_csv('/content/student_scores.csv')
print(dataset.head())
print(dataset.tail())
```

![image](https://github.com/user-attachments/assets/2bac1755-5c64-47a3-85c5-57877520c444)

dataset.info()

![image](https://github.com/user-attachments/assets/32953087-1d24-48c6-90e3-d332e690f54a)
```
x=dataset.iloc[:,:-1].values
print(x)
y=dataset.iloc[:,-1].values
print(y)
```

![image](https://github.com/user-attachments/assets/98e6005b-bda5-46f6-8ced-c7f22533e2fb)
```
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
x_train.shape
```

![image](https://github.com/user-attachments/assets/075bcdb1-0ea7-4010-80b0-46dbaa1ea6c5)

```
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
x_train.shape
x_test.shape
```

![image](https://github.com/user-attachments/assets/79877366-6de6-4337-8544-19a3f5b7e410)

```
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
x_train.shape
x_test.shape
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)
```
![image](https://github.com/user-attachments/assets/009e3b37-5832-4716-9192-c5e24cb9d25d)
```
y_pred=reg.predict(x_test)
print(y_pred)
print(y_test)
```

![image](https://github.com/user-attachments/assets/a6a809ef-63a1-4612-8ba9-f07a24269cba)
```
plt.scatter(x_train,y_train,color="green")
plt.plot(x_train,reg.predict(x_train),color="red")
plt.title('Traning set(H vs S)')
plt.xlabel("Hours")
plt.ylabel("scores")
plt.show()
plt.scatter(x_test,y_test,color="blue")
plt.plot(x_test,reg.predict(x_test),color="silver")
plt.title('Test set(H vs S)')
plt.xlabel("Hours")
plt.ylabel("scores")
plt.show()
```
```
mse = mean_squared_error(y_test, y_pred)
print('MSE = ', mse)
```
![image](https://github.com/user-attachments/assets/1b28b6f1-0b8b-498b-b0df-daf6abb19c13)
```
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
```
![image](https://github.com/user-attachments/assets/cc7c303d-a9ef-44fb-bb05-5db045dc3bf7)
```
rmse=np.sqrt(mse)
print("RMSE = ",rmse)
```
![image](https://github.com/user-attachments/assets/a5f7f009-d62b-4714-a3d2-c49ea464c3eb)


## Output:
![image](https://github.com/user-attachments/assets/b381025b-fa60-4f8f-9868-bcd084a65908)
![image](https://github.com/user-attachments/assets/ab742061-4ede-470d-aef7-13e51890f045)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
