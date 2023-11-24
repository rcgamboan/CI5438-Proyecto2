from network import Network
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import sys

def main():

    if len(sys.argv) != 2:
        print("Uso: python clasificacion.py <b|m>")
        exit()

    graficar = True
    # Leer archivo
    data_iris = pd.read_csv("../docs/iris.csv", sep=',')

    # Separar data de entrenamiento y de prueba
    # Se mantiene la proporcion de cada especie en cada conjunto
    X = data_iris[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
    y = data_iris["species"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Clasificadores binarios 
    if sys.argv[1] == "b":

        # Declaracion de los parametros a usar en el entrenamiento de la red neuronal
        iteraciones = 10000
        tasa_aprendizaje = 0.05
        tolerancia = 1e-8

        # Creacion de las redes neuronales para cada especie,
        # cada red neuronal tiene 4 neuronas en la capa de entrada ya que se tienen 4 caracteristicas,
        # y una neurona en la capa de salida ya que se buscan clasificadores binarios.
        # las capas ocultas se modificaran a conveniencia del usuario

        setosa_red = Network(4,[],1)
        versicolor_red = Network(4,[],1)
        virginica_red = Network(4,[],1)

        # Iris Setosa
        # Se convierten los datos de la especie a un valor binario
        # Comparando los datos con el nombre de la especie
        setosa_train_class = y_train == "Iris-setosa"
        setosa_test_class = y_test == "Iris-setosa"
        print("\nIris Setosa")
        
        errores_setosa = setosa_red.entrenar_red(X_train.values, 
                                                setosa_train_class.values, 
                                                iteraciones, 
                                                tasa_aprendizaje,
                                                tolerancia)
        
        # Evaluar el modelo con los datos del conjunto de prueba
        setosa_pred = setosa_red.feedforward(X_test.values)

        # Se redondean los valores de la prediccion
        # De esta manera solo se tendran valores binarios y se podrán comparar con los datos de prueba
        setosa_pred_class = [round(i[0]) for i in setosa_pred]
        setosa_red.calcular_precision(setosa_pred_class, setosa_test_class)
        setosa_red.calcular_fp_y_fn(setosa_pred_class, setosa_test_class)
        setosa_red.mostrar_info()

        # Iris Versicolor
        versicolor_train_class = y_train == "Iris-versicolor"
        versicolor_test_class = y_test == "Iris-versicolor"
        print("\nIris Versicolor")

        errores_veriscolor = versicolor_red.entrenar_red(X_train.values, 
                                                        versicolor_train_class.values, 
                                                        iteraciones, 
                                                        tasa_aprendizaje,
                                                        tolerancia)
        
        versicolor_pred = versicolor_red.feedforward(X_test.values)
        versicolor_pred_class = [round(i[0]) for i in versicolor_pred]
        versicolor_red.calcular_precision(versicolor_pred_class, versicolor_test_class)
        versicolor_red.calcular_fp_y_fn(versicolor_pred_class, versicolor_test_class)
        versicolor_red.mostrar_info()

        # Iris Virginica
        virginica_train_class = y_train == "Iris-virginica"
        virginica_test_class = y_test == "Iris-virginica"
        print("\nIris Virginica")

        errores_virginica = virginica_red.entrenar_red(X_train.values, 
                                                    virginica_train_class.values, 
                                                    iteraciones, 
                                                    tasa_aprendizaje,
                                                    tolerancia)
        
        virginica_pred = virginica_red.feedforward(X_test.values)
        virginica_pred_class = [round(i[0]) for i in virginica_pred]
        virginica_red.calcular_precision(virginica_pred_class, virginica_test_class)
        virginica_red.calcular_fp_y_fn(virginica_pred_class, virginica_test_class)
        virginica_red.mostrar_info()
        
        # Crea las 3 graficas en una sola ventana
        if graficar:
            fig, axs = plt.subplots(2, 2)
            fig.suptitle('MSE vs # Iteracion')
            axs[0, 0].plot(np.arange(len(errores_setosa)), errores_setosa, 'r')
            axs[0, 0].set_title('Clasificador Iris Setosa')
            axs[0, 1].plot(np.arange(len(errores_veriscolor)), errores_veriscolor, 'r')
            axs[0, 1].set_title('Clasificador Iris Versicolor')
            axs[1, 0].plot(np.arange(len(errores_virginica)), errores_virginica, 'r')
            axs[1, 0].set_title('Clasificador Iris Virginica')
            plt.show()
            # plt.savefig(f'escribir_directorio_aqui.png')

    # Clasificador multiclase
    if sys.argv[1] == "m":
        # Convertir los datos de la especie a un enteros para poder usarlos
        # con la red neuronal
        print("\nClasificador multiclase")
        species_to_int = {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}
        y_train_mc = [species_to_int[x] for x in y_train]
        y_test_mc = [species_to_int[x] for x in y_test]

        # Declaracion de los parametros a usar en el entrenamiento de la red neuronal
        iter_mc = 10000
        lr_mc = 0.005
        tol_mc = 1e-8

        # Creacion de la red
        red_mc = Network(4,capa,3)
        
        errores_mc = red_mc.entrenar_red(X_train.values, 
                                        y_train_mc, 
                                        iter_mc, 
                                        lr_mc,
                                        tol_mc)
        if graficar:
            red_mc.graficar_mse(len(errores_mc), 
                                errores_mc, 
                                guardar=False, 
                                nombre=f'error_clasificador_multiclase.png', 
                                titulo='Clasificador Multiclase')
        
        mc_setosa_fp = 0
        mc_setosa_fn = 0
        mc_versicolor_fp = 0
        mc_versicolor_fn = 0
        mc_virginica_fp = 0
        mc_virginica_fn = 0

        # Evaluar el modelo en el conjunto de prueba
        y_pred_mc = red_mc.feedforward(X_test)
        
        # Se selecciona la clase con el valor mas alto
        y_pred_mc_class = np.argmax(y_pred_mc, axis=1)

        red_mc.calcular_precision(y_pred_mc_class, y_test_mc)
        
        #"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2
        for i in range(len(y_pred_mc_class)):
            if y_pred_mc_class[i] != y_test_mc[i]:
                
                if y_test_mc[i] == 0:
                    # Clase real es setosa
                    # Falso negativo en setosa
                    mc_setosa_fn += 1
                    
                    if y_pred_mc_class[i] == 1:
                        # Se clasifica como versicolor
                        # Falso positivo en versicolor
                        mc_versicolor_fp += 1
                    
                    elif y_pred_mc_class[i] == 2:
                        # Se clasifica como virginica
                        # Falso positivo en virginica
                        mc_virginica_fp += 1
                
                elif y_test_mc[i] == 1:
                    # Class real es versicolor
                    # Falso negativo en versicolor
                    mc_versicolor_fn += 1
                    
                    if y_pred_mc_class[i] == 0:
                        # Se clasifica como setosa
                        # Falso positivo en setosa
                        mc_setosa_fp += 1
                    
                    elif y_pred_mc_class[i] == 2:
                        # Se clasifica como virginica
                        # Falso positivo en virginica
                        mc_virginica_fp += 1
                
                elif y_test_mc[i] == 2:
                    # Class real es virginica
                    # Falso negativo en virginica
                    mc_virginica_fn += 1
                    
                    if y_pred_mc_class[i] == 0:
                        # Se clasifica como setosa
                        # Falso positivo en setosa
                        mc_setosa_fp += 1
                    
                    elif y_pred_mc_class[i] == 1:
                        # Se clasifica como versicolor
                        # Falso positivo en versicolor
                        mc_versicolor_fp += 1
                                    
        red_mc.fp = mc_setosa_fp + mc_versicolor_fp + mc_virginica_fp
        red_mc.fn = mc_setosa_fn + mc_versicolor_fn + mc_virginica_fn

        red_mc.mostrar_info()

    if sys.argv[1] != "b" and sys.argv[1] != "m":
        print("Uso: python clasificacion.py <b|m>")
        exit()


if __name__ == "__main__":
    main()