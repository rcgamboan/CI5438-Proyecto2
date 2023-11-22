import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from network import Network
import numpy as np

def main():

    graficar = True

    with open("../docs/spambase.names", "r") as f:
        column_names = f.readlines()
    
    nombres = []
    nombres = column_names[33:]
    for i, name in enumerate(nombres):
        nombres[i] = name.split(":")[0]

    nombres.append("spam")

    X = pd.read_csv("../docs/spambase.data", names=nombres)
    y = X["spam"]
    X.drop("spam", axis=1, inplace=True)
    #print(X.head())
    #print(y.head())

    X_train, X_test, y_train, y_test = train_test_split(X, 
                                                        y, 
                                                        test_size=0.2, 
                                                        random_state=42, 
                                                        shuffle=True,
                                                        stratify=y)

    red_spam = Network(57,[5],1)

    iteraciones = 10000
    tasa_aprendizaje = 0.0000005
    tolerancia = 1e-8

    errores_spam = red_spam.entrenar_red(X_train.values, 
                                         y_train.values, 
                                         iteraciones, 
                                         tasa_aprendizaje, 
                                         tolerancia)

    if graficar:
        red_spam.graficar_mse(len(errores_spam),
                                errores_spam,
                                guardar=True, 
                                nombre="spam",
                                titulo="Error de la red para el conjunto de datos spam")
        
    # Evaluar el modelo en el conjunto de prueba
    y_pred = red_spam.feedforward(X_test)
    y_pred_class = [round(i[0]) for i in y_pred]
    red_spam.calcular_precision(y_pred_class, y_test)
    red_spam.calcular_fp_y_fn(y_pred_class, y_test)
    red_spam.mostrar_info()

if __name__ == "__main__":
    main()