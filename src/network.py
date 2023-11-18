import numpy as np
import matplotlib.pyplot as plt

class Network:

    def __init__(self, n_inputs=4, n_hidden=[5, 4], n_outputs=3):
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_outputs = n_outputs

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
        error = np.mean((Yreal - Ypred) ** 2) / 2

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
        
        return errores

    def graficar_mse(self,iterations, cost_num, guardar=False, nombre='mse.png'):
        fig, ax = plt.subplots()
        ax.plot(np.arange(iterations), cost_num, 'r')
        ax.set_xlabel('Iteracion #')
        ax.set_ylabel('MSE')
        ax.set_title('MSE vs # Iteracion')
        plt.style.use('fivethirtyeight')
        if guardar:
            plt.savefig(nombre)
        plt.show()    
            

if __name__ == '__main__':

    # Crear red neuronal
    red = Network(2,[5],1)
    
    # Crear data
    X = np.random.rand(1000,2)
    y = np.array([[i[0] + i[1]] for i in X])

    tasa_aprendizaje = 0.1
    iteraciones = 500
    # Entrenar red con data
    errores = red.entrenar_red(X, y, iteraciones, tasa_aprendizaje)

    # Imprimir error
    red.graficar_mse(len(errores), errores, guardar=False, nombre=f'grafica{tasa_aprendizaje}.png')

    # Predecir datos
    X = np.array([0.3, 0.1])
    y = np.array([0.4])

    prediccion = red.feedforward(X)

    for i in range(len(prediccion)):
        print(f"Yreal: {y[i]} Ypred: {round(prediccion[i],2)}")

    # Ejemplo de https://www.aprendemachinelearning.com/crear-una-red-neuronal-en-python-desde-cero/
    """
    red = Network(2,[3],2)
    X = np.array([[0, 0],   # sin obstaculos
              [0, 1],   # sin obstaculos
              [0, -1],  # sin obstaculos
              [0.5, 1], # obstaculo detectado a derecha
              [0.5,-1], # obstaculo a izq
              [1,1],    # demasiado cerca a derecha
              [1,-1]])  # demasiado cerca a izq
    
    y = np.array([[0,1],    # avanzar
                [0,1],    # avanzar
                [0,1],    # avanzar
                [-1,1],   # giro izquierda
                [1,1],    # giro derecha
                [0,-1],   # retroceder
                [0,-1]])  # retroceder

    red.entrenar_red(X, y, 15000, 0.03)

    index=0
    for e in X:
        print(f"Yreal: {y[index]} Ypred: {red.feedforward(e)}")
        index=index+1
    """