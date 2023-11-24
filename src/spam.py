import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from network import Network
import numpy as np

def main():

    tasas_aprendizaje = [0.00005, 0.000005, 0.0000005]
    capas_ocultas = [[], [40], [30], [20], [40, 30, 20], [30, 20, 10], [20, 10, 5]]

    graficar = True

    with open("./docs/spambase/spambase.names", "r") as f:
        column_names = f.readlines()
    
    nombres = []
    nombres = column_names[33:]
    for i, name in enumerate(nombres):
        nombres[i] = name.split(":")[0]

    nombres.append("spam")

    X = pd.read_csv("./docs/spambase/spambase.data", names=nombres)
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

    # Configuracion de los parametros a usar para el entrenamiento de la red
    iteraciones = 10000
    tasa_aprendizaje = 0.0000005
    tolerancia = 1e-8

    for tasa  in tasas_aprendizaje:
        print(f"Tasa de aprendizaje: {tasa}")
        for capa in capas_ocultas:
            print(f"Capa oculta: {capa}")
            # Creacion de la red neuronal
            red_spam = Network(57,capa,1)

            # Entrenamiento de la red
            errores_spam = red_spam.entrenar_red(X_train.values, 
                                                y_train.values, 
                                                iteraciones, 
                                                tasa, 
                                                tolerancia)

            # Graficar el error de la red
            if graficar:
                red_spam.graficar_mse(len(errores_spam),
                                        errores_spam,
                                        guardar=True, 
                                        nombre=f"./img/spam/{tasa}_{capa}_mse.png",
                                        titulo="Error de la red para el conjunto de datos spam")
                
            # Evaluar el modelo en el conjunto de prueba
            y_pred = red_spam.feedforward(X_test)
            y_pred_class = [round(i[0]) for i in y_pred]
            red_spam.calcular_precision(y_pred_class, y_test)
            red_spam.calcular_fp_y_fn(y_pred_class, y_test)
            red_spam.mostrar_info(tasa, capa, "spam")

if __name__ == "__main__":
    main()