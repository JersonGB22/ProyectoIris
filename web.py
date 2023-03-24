import pandas as pd
import pickle
import streamlit as st

#Extracción de los archivos pickle
with open("lin_reg.pkl","rb") as li:
    lin_reg=pickle.load(li)

with open("log_reg.pkl","rb") as lo:
    log_reg=pickle.load(lo)

with open("svc.pkl","rb") as s:
    svc_train=pickle.load(s)

#Función para clasificar las plantas
def classify(num):
    if num==0:
        c="setosa"
    elif num==1:
        c="versicolor"
    else:
        c="virginica"
    return c

def main():
    #Título
    st.title("Modelamiento del Dataset Iris")
    #Título de la barra lateral
    st.sidebar.header("User input parameters")

    def parameters():
        sepal_length=st.sidebar.slider("Sepal length",4.3,7.9,6.1,0.1)#título_barra_deslizante, val_min, val_max, val_pred_web, aumento
        sepal_width=st.sidebar.slider("Sepal width",2.0,4.4,3.2,0.1)#todos deben ser float o bien int
        petal_length=st.sidebar.slider("Petal length",1.0,6.9,3.95,0.1)
        petal_width=st.sidebar.slider("Petal width",0.1,2.5,1.3,0.1)
        data={"Sepal length":sepal_length,"Sepal width":sepal_width,"Petal length":petal_length,"Petal width":petal_width}
        features=pd.DataFrame(data,index=[0])
        return features
    df=parameters()
    option=["Linear Regression","Logistic Regression","SVM"]
    model=st.sidebar.selectbox("Que modelo desea utilizar?",option)
    st.subheader("User Input Parameters")
    st.subheader(model)
    st.write(df)
    if st.button("run"):
        if model==option[0]:
            st.success(classify(lin_reg.predict(df)))
        elif model==option[1]:
            st.success(classify(log_reg.predict(df)))
        else:
            st.success(classify(svc_train.predict(df)))

if __name__=="__main__":
    main()
#Los archivos Procfile y setup.sh son los predeterminados para Heroku
#requirements.txt: pip list (pickle es una biblioteca estándar de Python, por lo que no se añade)
#Crearlo de manera automática: pip freeze > requirements.txt (pero nos creará con muchas otras dependencias)