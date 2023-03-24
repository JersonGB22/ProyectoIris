from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.svm import SVC
import pickle

#Carga de datos
iris=datasets.load_iris()
X=iris.data
y=iris.target

#Separación de variables
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42)

lir=LinearRegression()
lor=LogisticRegression()

svc=SVC()

#Entrenamiento del modelo
lin_reg=lir.fit(X_train,y_train)
log_reg=lor.fit(X_train,y_train)
svc_train=svc.fit(X_train,y_train)

with open("lin_reg.pkl","wb") as li:
    pickle.dump(lin_reg,li)

with open("log_reg.pkl","wb") as lo:
    pickle.dump(log_reg,lo)

with open("svc.pkl","wb") as s:
    pickle.dump(svc_train,s)