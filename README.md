# CNN desde cero para clasificación de MNIST

Red neuronal convolucional implementada **solo con NumPy** (sin TensorFlow ni PyTorch) para clasificar dígitos escritos a mano del dataset MNIST. Incluye el forward, el backpropagation completo (convolución, ReLU, max pooling y capa densa), el entrenamiento con mini-batches y las visualizaciones de lo que aprende la red.

La capa densa final reutiliza el MLP implementado desde cero en un proyecto anterior, con una pequeña modificación (ver más abajo).

## Resultados

| Métrica | Valor |
|---|---|
| Accuracy en entrenamiento (primeras 15.000 imágenes) | 97,62 % |
| Accuracy en test (primeras 15.000 imágenes del test) | 95,95 % |
| Pérdida (cross entropy) época 0 → época 9 | 0,498 → 0,099 |
| Tiempo de entrenamiento | ~15.460 s (~4,3 h) |

Configuración del entrenamiento: 15.000 imágenes, 10 épocas, tasa de aprendizaje `0.05`, mini-batches de 32.

> La red se entrenó solo con 15.000 de las 52.500 imágenes de entrenamiento porque la implementación con bucles de Python tarda unos 0,1 s por imagen. Con más datos y más épocas el resultado debería mejorar (la pérdida seguía bajando al terminar).

## Arquitectura

```
Imagen (1, 28, 28)
   │
   ▼
Convolución: 8 filtros 3x3, sin padding, stride 1  →  (8, 26, 26)
   │
   ▼
ReLU                                                →  (8, 26, 26)
   │
   ▼
Max pooling 2x2                                     →  (8, 13, 13)
   │
   ▼
Flatten                                             →  (1, 1352)
   │
   ▼
Capa densa (1352 → 10) + Softmax                    →  (1, 10) probabilidades
```

- **Parámetros entrenables:** 72 pesos y 8 sesgos en la convolución, más 13.520 pesos y 10 sesgos en la capa densa.
- **Inicialización:** He para los filtros (por usar ReLU) y Xavier/Glorot para la capa densa.
- **Pérdida:** categorical cross entropy.
- **Optimizador:** descenso de gradiente con mini-batches.

### Qué hay en el notebook

1. Carga y preprocesamiento de MNIST (normalización a [0, 1], división 75 % / 25 % estratificada, one-hot de las etiquetas).
2. Funciones auxiliares: `softmax`, `cross_entropy`.
3. Clase `MLP` (capa densa de la CNN).
4. Funciones de la CNN con sus derivadas: `ReLU` / `d_relu`, `max_pooling` / `d_pooling`.
5. Clase `CNN` con los métodos:
   - `conv2d` y `d_conv2d`: convolución y su gradiente.
   - `feed_forward`: recorrido completo hacia adelante, devuelve las probabilidades y un `cache`.
   - `backpropagation_cnn`: recorrido hacia atrás encadenando los gradientes.
   - `gradiente_descendente`: actualización de todos los parámetros.
   - `fit`: entrenamiento con mini-batches.
   - `predict`: devuelve el dígito predicho de cada imagen.
6. Entrenamiento, evaluación y matriz de confusión.
7. Guardado y carga de los pesos.
8. Gráfico de la pérdida y visualización de filtros, feature maps, activation maps, pooling y flatten.

## Cómo funciona

### Forward

Para cada imagen, cada filtro calcula en cada posición de la ventana:

```
z = suma(region * filtro) + sesgo
```

Después se aplica ReLU, se reduce con max pooling (guardando una máscara con la posición de cada máximo), se aplana y se pasa a la capa densa con softmax.

### Backward

Se recorre la red en orden inverso, multiplicando en cada capa el gradiente que llega por su derivada local (regla de la cadena):

```
(probs - y) → capa densa → deshacer flatten → d_pooling (usa la máscara)
            → d_relu (usa z > 0) → d_conv2d → gradientes de filtros y sesgos
```

El `cache` que devuelve `feed_forward` guarda lo que cada capa necesita en el backward: la imagen `x`, el feature map `z`, la máscara del pooling, la forma antes del flatten y las netas y activaciones del MLP.

### Cambio al MLP original

El método `back_propagation` del MLP ahora devuelve **tres** valores en lugar de dos: `dW`, `db` y `dx`, el gradiente respecto a la entrada de la capa densa. Ese `dx` es el que se le pasa hacia atrás a la parte convolucional. 

### Verificación

Los gradientes se verificaron con **gradient check** (comparación con el gradiente numérico por diferencias centrales). Coinciden hasta unos 10 decimales en los filtros, los sesgos y la capa densa. Al verificar los sesgos hay que usar valores distintos de cero: con sesgos en 0 y el fondo negro de MNIST, muchas posiciones caen exactamente en el punto no derivable de la ReLU y el gradiente numérico no es comparable.

## Uso

### Requisitos

```
numpy
scikit-learn
matplotlib
```

### Entrenar

```python
cnn = CNN(num_filters=8, filter_size=3, input_channels=1,
          input_size=28, pool_size=2, output_size=10)

hist = cnn.fit(x_train[:15000], y_train_oh[:15000], epocas=10, ln=0.05, batch_size=32)
```

`y` debe estar en one-hot, con forma `(n, 10)`.

### Guardar y cargar el modelo

```python
# guardar
np.savez("cnn_pesos.npz", cnn_w=cnn.weights, cnn_b=cnn.bias,
         mlp_w=cnn.mlp.weights[0], mlp_b=cnn.mlp.biases[0], loss=np.array(hist))

# cargar (con los mismos hiperparámetros con los que se entrenó)
cnn = CNN(num_filters=8, filter_size=3, input_channels=1,
          input_size=28, pool_size=2, output_size=10)

datos = np.load("cnn_pesos.npz")
cnn.weights = datos["cnn_w"]
cnn.bias = datos["cnn_b"]
cnn.mlp.weights[0] = datos["mlp_w"]    # mlp.weights es una lista de matrices: se asigna por índice
cnn.mlp.biases[0] = datos["mlp_b"]
hist = datos["loss"].tolist()
```

### Predecir

```python
y_pred = cnn.predict(x_test[:100])                 # una predicción por imagen
print(np.mean(y_pred == y_test[:100]))

cnn.predict(x_test[5:6])                           # una sola imagen: usar x[i:i+1], no x[i]
```

## Visualizaciones

El notebook genera:

- Matriz de confusión (conteos y normalizada por fila).
- Evolución de la pérdida por época.
- Una figura resumen con la imagen, las probabilidades, la pérdida, los filtros aprendidos, el feature map, el activation map, el max pooling y el vector flatten.

## Limitaciones

- **Lenta.** La convolución y el pooling usan bucles de Python y cada imagen se procesa por separado (~0,1 s por imagen en entrenamiento y ~0,03 s en predicción). Evaluar el test completo tarda varios minutos.
- **`padding` y `stride` no están implementados.** La convolución asume `padding=0` y `stride=1`. Si el tamaño de la salida de la convolución no es divisible entre `pool_size`, `d_pooling` rellena con ceros las filas y columnas descartadas.
- **Una sola capa convolucional.** `d_conv2d` ya calcula `dx` (el gradiente respecto a la entrada) para poder apilar más capas, pero hoy no se usa.
- **Sin regularización ni validación por época.** La pérdida que se imprime se calcula mientras los pesos cambian durante la época, por lo que es una aproximación.

## Posibles mejoras

1. Vectorizar la convolución con `sliding_window_view` y `einsum`, y el pooling con `reshape`.
2. Procesar el lote completo de una vez, eliminando el bucle por imágenes.
3. Entrenar con las 52.500 imágenes y evaluar sobre el test completo.
4. Implementar `padding` y `stride`.
5. Generalizar a varias capas convolucionales con una clase `ConvLayer` y una lista de capas.
6. Calcular la pérdida y el accuracy de validación en cada época para detectar sobreajuste.
