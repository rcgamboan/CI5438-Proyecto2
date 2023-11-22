import numpy as np
import matplotlib.pyplot as plt

class Network:

    def __init__(self, n_inputs=4, n_hidden=[5, 4], n_outputs=3):
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_outputs = n_outputs
        self.accuracy = 0
        self.fp = 0
        self.fn = 0
        self.errores = []

        if (n_hidden == []):
            layers = [self.n_inputs] + [self.n_outputs]
        else:
            layers = [self.n_inputs] + self.n_hidden + [self.n_outputs]

        self.layers = layers
        self.weights = [np.random.rand(self.layers[i], self.layers[i+1] ) for i in range(len(layers)-1)]
        self.val_activacion = [np.zeros(self.layers[i]) for i in range(len(layers))]
        self.derivadas = [np.zeros(shape = (self.layers[i], self.layers[i+1])) for i in range(len(layers)-1)]


    def error_cuadratico(self,Ypred, Yreal):
        # Calculo del error cuadratico medio
        error = np.mean((Yreal - Ypred) ** 2.0) / 2.0

        return error

    def feedforward(self, inputs):
        
        activacion = inputs
        self.val_activacion[0] = inputs

        for i, w in enumerate(self.weights):
            net_inputs = np.dot(activacion, w)
            activacion = self.sigmoid(net_inputs)
            self.val_activacion[i+1] = activacion

        return activacion

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))
    
    def sigmoid_der(self, x):
        return x * (1.0 - x)
    
    def backpropagation(self, error, tasa_aprendizaje):

        for i in reversed(range(len(self.derivadas))):
            activation = self.val_activacion[i+1]
            
            delta = error * self.sigmoid_der(activation)
            delta_reshaped = delta.reshape(delta.shape[0], -1).T
            
            activacion_actual = self.val_activacion[i]
            activacion_actual_reshaped = activacion_actual.reshape(activacion_actual.shape[0] , -1)

            self.derivadas[i] = np.dot(activacion_actual_reshaped, delta_reshaped)

            error = np.dot(delta, self.weights[i].T)
        
        
        # Descenso de gradiente
        for i in range(len(self.weights)):
            self.weights[i] += tasa_aprendizaje * self.derivadas[i]
        #print(i,"backpropagation")
        return error
    
    def entrenar_red(self, X, y, iteraciones, tasa_aprendizaje, tolerancia=1e-6):
        errores = []
        for i in range(iteraciones):
            for (inputs, targets) in zip(X, y):
                
                # Forward propagation
                outputs = self.feedforward(inputs)

                # Calcular error
                error = targets - outputs

                # Backpropagation
                self.backpropagation(error,tasa_aprendizaje)

            errores.append(self.error_cuadratico(targets,outputs))
            
            if i != 0:
                if (abs(errores[i] - errores[i-1]) <= tolerancia):
                    print(f"Convergencia en la iteracion {i}")
                    break
        self.errores = errores
        return errores

    def graficar_mse(self,iterations, cost_num, guardar=False, nombre='mse.png', titulo='MSE vs # Iteracion'):
        fig, ax = plt.subplots()
        ax.plot(np.arange(iterations), cost_num, 'r')
        ax.set_xlabel('Iteracion #')
        ax.set_ylabel('MSE')
        ax.set_title(titulo)
        plt.style.use('fivethirtyeight')
        if guardar:
            plt.savefig(nombre)
        plt.show()

    def calcular_precision(self, y_pred_class, y_test):
        self.accuracy = np.mean(y_pred_class == y_test)
        return self.accuracy
    
    def calcular_fp_y_fn(self, y_pred_class, y_test):
        self.fp, self.fn = 0, 0
        for i in range(len(y_pred_class)):
            if y_pred_class[i] == 1 and y_test.iloc[i] == 0:
                self.fp += 1
            elif y_pred_class[i] == 0 and y_test.iloc[i] == 1:
                self.fn += 1
        return self.fp, self.fn

    def mostrar_info(self):
        print(f"\nPrecision de la red: {round(self.accuracy*100,2)}%")
        print(f"Falsos positivos en la red: {self.fp}")
        print(f"Falsos negativos en la red: {self.fn}")
        #print(f"Errores: {self.errores}")
        if self.errores != []:    
            print(f"Error promedio de la red: {np.mean(self.errores)}")
            print(f"Error maximo de la red: {np.max(self.errores)}")
            print(f"Error minimo de la red: {np.min(self.errores)}")


            
