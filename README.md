# CI5438-Proyecto2

El objetivo de este proyecto será la implementación del algoritmo de backpropagation para redes neuronales feedforward, tal cuál como fue visto en clases, y su uso para crear redes neuronales para resolver problemas de clasificación. Primero se trabajará con un conjunto de datos de plantas Iris, probando distintas topologías, incluyendo modelos de neuronas únicas y de una capa. Luego, trabajaremos con un problema de filtros de spam sobre correos electrónicos, de manera más libre.

La elección de lenguaje de programación es libre. Más allá de la implementación del algoritmo, la cuál es obligatoria, puede usar cualquier libreria o herramienta que considere necesaria para facilitar su trabajo.

## Parte 1: Implementación

Se requiere que elabore, de la manera que usted considere conveniente, una implementación de redes neuronales feedforward para clasificación, con las siguientes características:
* Debe permitir la configuración de tantas capas ocultas como se desee, permitiendo 0 capas ocultas para tener clasificadores lineales.
* Debe permitir configurar la cantidad de neuronas en cada capa.
* Debe usar la función logística como función de activación para las neuronas.
* Debe implementar backpropagation con descenso del gradiente como algoritmo de entrenamiento, usando la función de pérdida cuadrática, de la manera vista en clases.

Adiciones como momentum, regularización y tasas adaptativas son opcionales.

## Parte 2: Iris Dataset

Se incluye en el archivo `iris.csv`[1], que contiene información para 3 clases, cada una siendo un tipo de planta del género iris. Este conjunto de datos está completo, perfectamente balanceado entre las tres clases y no requiere ningún tipo de preprocesamiento. Por lo tanto, se requiere que usted haga experimentos para hallar clasificadores binarios y multiclase para determinar la clase de una planta iris, usando TODOS los atributos y TODOS los ejemplos del conjunto de datos.

Debe aplicar un proceso de validación cruzada. Para ello, debe separar su conjunto de datos en conjuntos de entrenamiento y prueba, para lo cuál se sugiere una separación de 80% y 20% respectivamente. Puede elegir seleccionar también un conjunto de validación, pero no es obligatorio. Trate de mantener el balance en las clases al hacer la separación. Considere que para los clasificadores binarios podría querer considerar una separación distinta.

Para el proceso de entrenamiento, asegúrese de probar con distintos valores para la tasa de entrenamiento. Grafique las curvas de aprendizaje para los experimentos realizados.

### Parte 2.1: Clasificadores binarios

La idea de un clasificador binario será simplemente clasificar si un ejemplo pertenece a una de las clases o no. Es decir, tendrá una sola neurona en la capa de salida, y deberá entrenar un clasificador por clase para cada experimento realizados.

Para cada experimento realizado, debe incluir tablas con:
* Error promedio, mínimo y máximo en el conjunto de entrenamiento y prueba.
* Cantidad de falsos positivos tanto en el conjunto de entrenamiento como de prueba. Un falso positivo en un clasificador binario es un ejemplo para el que la hipótesis incluye al ejemplo en la categoría clasificada, cuando no pertenece a ella en realidad.
* Cantidad de falsos negativos tanto en el conjunto de entrenamiento como de prueba. Un falso negativo en un clasificador binario es un ejemplo para el que la hipótesis excluye de la categoría clasificada, cuando en realidad sí pertenece a ella. 


Se requiere que realice experimentos para las siguientes topologías:

* Una única neurona, variando la tasa de aprendizaje para hallar la mejor hipótesis.
* Utilizando una capa oculta, variando la tasa de aprendizaje y probando con cuantas neuronas quiera en capa oculta.
* Utilizando dos capas ocultas, siguiendo la misma idea.

Se sabe que de las clases en el conjunto es linealmente separable. ¿Pueden sus resultados corroborar esto? ¿Por qué?

### Parte 2.2: Clasificadores multiclase

Para este tipo de clasificador, tendremos tres neuronas en la capa de salida, cada una representando una de las clases.

Para cada experimento realizado, debe incluir tablas con:
* Error promedio, mínimo y máximo en el conjunto de entrenamiento y prueba.

Se requiere que realice experimentos para las siguientes topologías:

* Utilizando una capa oculta, variando la tasa de aprendizaje y probando con cuantas neuronas quiera en capa oculta.
* Utilizando dos capas ocultas, siguiendo la misma idea.

Compare los resultados con los obtenidos usando clasificadores binarios.

## Parte 3: Clasificación de spam

Se incluye el comprimido `spambase.zip`[2] con información para clasificar correos electrónicos como spam, dada la frecuencia de ciertas palabras. Se requiere que usted construya un clasificador binario para esta tarea. Experimente con varias hipótesis, variando las topologías de red y tasas de aprendizaje y documente su proceso y conclusiones. El conjunto de datos consiste únicamente de atributos continuos y no tiene valores faltantes, pero puede que obtenga mejores resultados aplicando alguna técnica de normalización de los datos.


## Entrega

Para la entrega del proyecto, haga fork de este repositorio. Su repositorio deberá contener todo el código usado para el proyecto, las gráficas que fueron generadas durante el entrenamiento, y un informe discutiendo los detalles de su implementación, su proceso de preprocesamiento y su proceso de entrenamiento, así como las preguntas planteadas en el enunciado.


[1] Fisher,R. A.. (1988). Iris. UCI Machine Learning Repository. https://doi.org/10.24432/C56C76.

[2] Hopkins,Mark, Reeber,Erik, Forman,George, and Suermondt,Jaap. (1999). Spambase. UCI Machine Learning Repository. https://doi.org/10.24432/C53G6X.
