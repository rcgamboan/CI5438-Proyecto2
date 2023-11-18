from network import Network
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def main():
    graficar = True
    # Leer archivo
    data_iris = pd.read_csv("../docs/iris.csv", sep=',')

    # Separar data de entrenamiento y de prueba
    X = data_iris[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
    y = data_iris["species"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    # Creacion de las redes neuronales para cada especie,
    # cada red neuronal tiene 4 neuronas en la capa de entrada ya que se tienen 4 caracteristicas, 
    # 1 capa oculta con 5 neuronas
    # y una neurona en la capa de salida ya que se buscan clasificadores binarios
    setosa_red = Network(4,[5],1)
    versicolor_red = Network(4,[5],1)
    virginica_red = Network(4,[5],1)

    iteraciones = 500
    tasa_aprendizaje = 0.1

    setosa_train_class = y_train == "Iris-setosa"
    setosa_test_class = y_test == "Iris-setosa"
    print("\nIris Setosa")
    #print(setosa_train_class)
    
    errores_setosa = setosa_red.entrenar_red(X_train.values, setosa_train_class.values, iteraciones, tasa_aprendizaje)

    versicolor_train_class = y_train == "Iris-versicolor"
    versicolor_test_class = y_test == "Iris-versicolor"
    print("\nIris Versicolor")
    #print(versicolor_train_class)

    errores_veriscolor = versicolor_red.entrenar_red(X_train.values, versicolor_train_class.values, iteraciones, tasa_aprendizaje)

    virginica_train_class = y_train == "Iris-virginica"
    virginica_test_class = y_test == "Iris-virginica"
    print("\nIris Virginica")
    #print(virginica_train_class)

    errores_virginica = virginica_red.entrenar_red(X_train.values, virginica_train_class.values, iteraciones, tasa_aprendizaje)

    # Crea las 3 graficas separadas
    """fig1 = plt.figure()
    fig2 = plt.figure()
    fig3 = plt.figure()

    ax1 = fig1.add_subplot(111)
    ax2 = fig2.add_subplot(111)
    ax3 = fig3.add_subplot(111)

    ax1.plot(np.arange(len(errores_setosa)), errores_setosa, 'r')
    ax2.plot(np.arange(len(errores_veriscolor)), errores_veriscolor, 'r')
    ax3.plot(np.arange(len(errores_virginica)), errores_virginica, 'r')

    ax1.set_xlabel('Iteracion #')
    ax1.set_ylabel('MSE')
    ax1.set_title('Clasificador Iris Setosa')

    ax2.set_xlabel('Iteracion #')
    ax2.set_ylabel('MSE')
    ax2.set_title('Clasificador Iris Versicolor')

    ax3.set_xlabel('Iteracion #')
    ax3.set_ylabel('MSE')
    ax3.set_title('Clasificador Iris Virginica')
    
    plt.style.use('fivethirtyeight')
    plt.show()"""

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

    # Evaluar el modelo en el conjunto de prueba y reportar resultados
    setosa_pred = setosa_red.feedforward(X_test.values)
    setosa_pred_class = [round(i[0]) for i in setosa_pred]
    setosa_fp = 0
    setosa_fn = 0
    for i in range(len(setosa_pred_class)):
        if setosa_pred_class[i] == 1 and setosa_test_class.iloc[i] == 0:
            setosa_fp += 1
        elif setosa_pred_class[i] == 0 and setosa_test_class.iloc[i] == 1:
            setosa_fn += 1
    setosa_accuracy = np.mean(setosa_pred_class == setosa_test_class)
    
    versicolor_pred = versicolor_red.feedforward(X_test.values)
    versicolor_pred_class = [round(i[0]) for i in versicolor_pred]
    versicolor_fp = 0
    versicolor_fn = 0
    for i in range(len(versicolor_pred_class)):
        if versicolor_pred_class[i] == 1 and versicolor_test_class.iloc[i] == 0:
            versicolor_fp += 1
        elif versicolor_pred_class[i] == 0 and versicolor_test_class.iloc[i] == 1:
            versicolor_fn += 1
    versicolor_accuracy = np.mean(versicolor_pred_class == versicolor_test_class)
    

    virginica_pred = virginica_red.feedforward(X_test.values)
    virginica_pred_class = [round(i[0]) for i in virginica_pred]
    virginica_fp = 0
    virginica_fn = 0
    for i in range(len(virginica_pred_class)):
        if virginica_pred_class[i] == 1 and virginica_test_class.iloc[i] == 0:
            virginica_fp += 1
        elif virginica_pred_class[i] == 0 and virginica_test_class.iloc[i] == 1:
            virginica_fn += 1
    virginica_accuracy = np.mean(virginica_pred_class == virginica_test_class)
    
    print(f"\nPrecision en clase Iris-setosa: {round(setosa_accuracy*100,2)}%")
    print(f"Cantidad de datos de prueba: {len(setosa_test_class)}")
    print(f"Falsos positivos en clase Iris-setosa: {setosa_fp}")
    print(f"Falsos negativos en clase Iris-setosa: {setosa_fn}")

    print(f"\nPrecision en clase Iris-versicolor: {round(versicolor_accuracy*100,2)}%")
    print(f"Cantidad de datos de prueba: {len(versicolor_test_class)}")
    print(f"Falsos positivos en clase Iris-versicolor: {versicolor_fp}")
    print(f"Falsos negativos en clase Iris-versicolor: {versicolor_fn}")

    print(f"\nPrecision en clase Iris-virginica: {round(virginica_accuracy*100,2)}%")
    print(f"Cantidad de datos de prueba: {len(virginica_test_class)}")
    print(f"Falsos positivos en clase Iris-virginica: {virginica_fp}")
    print(f"Falsos negativos en clase Iris-virginica: {virginica_fn}")

    # Clasificador multiclase

    # Convertir los datos de la especie a un enteros para poder usarlos
    # con la red neuronal
    print("\nClasificador multiclase")
    species_to_int = {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}
    y_train_mc = [species_to_int[x] for x in y_train]
    y_test_mc = [species_to_int[x] for x in y_test]

    # Creacion de la red
    red_mc = Network(4,[5],3)
    
    errores_mc = red_mc.entrenar_red(X_train.values, y_train_mc, iteraciones, tasa_aprendizaje)
    if graficar:
        red_mc.graficar_mse(len(errores_mc), errores_mc, guardar=False, nombre=f'grafica_multiclase.png', titulo='Clasificador Multiclase')
    mc_setosa_fp = 0
    mc_setosa_fn = 0
    mc_versicolor_fp = 0
    mc_versicolor_fn = 0
    mc_virginica_fp = 0
    mc_virginica_fn = 0

    # Evaluar el modelo en el conjunto de prueba
    y_pred_mc = red_mc.feedforward(X_test)
    y_pred_mc_class = np.argmax(y_pred_mc, axis=1)
    accuracy_mc = np.mean(y_pred_mc_class == y_test_mc)
    
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
                                
    fp = mc_setosa_fp + mc_versicolor_fp + mc_virginica_fp
    fn = mc_setosa_fn + mc_versicolor_fn + mc_virginica_fn

    print(f"\nPrecision con el clasificador multiclase: {round(accuracy_mc*100,2)}%")
    print(f"Cantidad de datos de prueba: {len(y_test_mc)}")
    print(f"Falsos positivos en clasificador multiclase: {fp}")
    print(f"Falsos negativos en clasificador multiclase: {fn}")


if __name__ == "__main__":
    main()