
****************************************************************************************************************
****************************************************************************************************************
****************************************************************************************************************
****************************************************************************************************************
PROYECTO MLOps: Clasificación de canciones

Autor: Javier Díaz-Cano Rincón.
****************************************************************************************************************
****************************************************************************************************************
****************************************************************************************************************
****************************************************************************************************************


---



**********************************
PROYECTO EN GITHUB
**********************************

El enlace para acceder al repositorio del proyecto es el siguiente: "https://github.com/Javuchenn/Trabajo_MLOPS"

...

**********************************
FUNCIONALIDADES
**********************************

Este proyecto implementa un sistema completo de Machine Learning para la clasificación de géneros musicales a partir de características numéricas extraídas de audio. Incluye todo el flujo end-to-end, desde la preparación de datos hasta el despliegue de una API de inferencia.

El sistema permite cargar y preprocesar un dataset de audio, realizando tareas de limpieza, codificación de variables categóricas, normalización de características y división en conjuntos de entrenamiento, validación y test. Este proceso está automatizado para garantizar consistencia entre ejecuciones.

Incluye un modelo de red neuronal construido con TensorFlow/Keras, diseñado específicamente para problemas de clasificación multiclase. El modelo se entrena utilizando técnicas de regularización y early stopping para mejorar la generalización y evitar sobreajuste.

El entrenamiento del modelo se monitoriza mediante métricas como accuracy y loss tanto en entrenamiento como en validación. Además, se guarda automáticamente el mejor modelo obtenido durante el proceso de entrenamiento.

El proyecto incorpora un sistema de logging para registrar eventos importantes durante la ejecución, facilitando la depuración y el seguimiento del flujo del sistema.

También se integra opcionalmente Weights & Biases (wandb) para el seguimiento de experimentos, aunque puede desactivarse fácilmente sin afectar la ejecución del código.

Se incluye un `Dockerfile` que permite construir una imagen del sistema de forma reproducible y aislada. Gracias a este archivo, es posible ejecutar la API de inferencia en cualquier entorno que tenga Docker instalado, sin necesidad de configurar manualmente dependencias o versiones de librerías. Esto facilita el despliegue, mejora la portabilidad del proyecto y garantiza que el comportamiento del modelo sea consistente entre entornos de desarrollo y producción.

También hay disponibles tests automatizados implementados con `pytest` para garantizar la correcta funcionalidad de las principales partes del sistema. En concreto, se han definido pruebas en los archivos `test_model.py`, `test_train.py` y `test_utils.py`, que cubren respectivamente el comportamiento del modelo, el proceso de entrenamiento y las funciones auxiliares del sistema. Estas pruebas permiten verificar que cada componente funciona correctamente de forma aislada, asegurando la robustez del código, facilitando el mantenimiento y detectando posibles errores de forma temprana durante el desarrollo.

Finalmente, se expone una API de inferencia desarrollada con FastAPI que permite realizar predicciones en tiempo real. Esta API incluye un endpoint `/predict` que recibe un vector de características y devuelve la clase predicha, el género musical asociado y las probabilidades de cada clase. La API puede ejecutarse tanto en entorno local como mediante Docker, manteniendo el mismo comportamiento funcional en ambos casos.

---

******************************************************************************************************
LANZAMIENTO DEL PROYECTO PARA UN ENTORNO LOCAL DE DESARROLLO
******************************************************************************************************

Para ejecutar el proyecto en local, primero se debe levantar la API FastAPI. Esto se hace desde la raíz del proyecto utilizando el comando `uvicorn src.api_inference:app --reload`. Este comando inicia el servidor en modo desarrollo y permite la recarga automática cada vez que se modifica el código.

Una vez ejecutado correctamente, la API queda disponible en la dirección local que el servidor indique en consola. Desde ahí se puede acceder al servicio y comprobar que está funcionando correctamente.

Para probar los endpoints de forma sencilla, FastAPI proporciona una interfaz interactiva de documentación automática accesible mediante la ruta `/docs`. Esta interfaz permite ejecutar peticiones HTTP directamente desde el navegador sin necesidad de herramientas externas como Postman o curl.

Dentro de esta interfaz, se selecciona el endpoint `POST /predict` y se pulsa el botón “Try it out”. A continuación, se introduce un JSON con los datos de entrada del modelo. El formato correcto del input es el siguiente:

```json
{
  "features": [
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
  ]
}
```

Las features que indicamos en ese "json" son las siguientes variables:
  - chroma_stft_mean / chroma_stft_var
  - rms_mean / rms_var
  - spectral_centroid_mean / spectral_centroid_var
  - spectral_bandwidth_mean / spectral_bandwidth_var
  - rolloff_mean / rolloff_var
  - zero_crossing_rate_mean / zero_crossing_rate_var
  - harmony_mean / harmony_var
  - perceptr_mean / perceptr_var
  - tempo
  - mfcc1_mean ... mfcc20_mean
  - mfcc1_var ... mfcc20_var
  - length

Una vez enviado el request pulsando “Execute”, la API procesa la petición y devuelve la respuesta en la sección “Response body”. En esta respuesta se incluye la clase predicha, el nombre del género musical correspondiente, las probabilidades de cada clase y el nivel de confianza del modelo.

Es importante tener en cuenta que el vector de entrada debe contener exactamente 58 features, en el mismo orden utilizado durante el entrenamiento del modelo. Si el número de features no coincide, la API devolverá un error de validación.

---

******************************************************************************************************
LANZAMIENTO DEL PROYECTO PARA UN ENTORNO LOCAL DE DESARROLLO **UTILIZANDO DOCKER**
******************************************************************************************************

Para ejecutar el proyecto utilizando Docker, primero es necesario construir la imagen a partir del `Dockerfile` incluido en el repositorio. Esto se realiza ejecutando el comando "docker build -t mlops-api ." desde la raíz del proyecto.

Una vez construida la imagen, se puede lanzar el contenedor con el comando "docker run -p 8000:8000 mlops-api". Esto iniciará la API de inferencia dentro de un entorno aislado y reproducible.

Cuando el contenedor esté en ejecución, la API estará disponible en `http://localhost:8000`, y la documentación interactiva de FastAPI podrá consultarse en `http://localhost:8000/docs`. Desde esta interfaz se puede acceder al endpoint `/predict` y realizar pruebas enviando un JSON con el vector de características del modelo, siguiendo exactamente el mismo procedimiento que se ha explicado anteriormente para el entorno local sin Docker. El funcionamiento es el mismo, con la única diferencia de que en este caso la aplicación se ejecuta dentro de un contenedor aislado y reproducible.

NOTA: Es necesario tener Docker Desktop abierto y en ejecución antes de lanzar los comandos de Docker, ya que el servicio Docker Engine debe estar activo para poder construir y ejecutar contenedores.

---

**********************************
PROYECTO EN WEIGHT AND BIASES
**********************************

Se ha añadido una invitación al profesor de la asignatura (antonio.gpardo@urjc.es) al TEAM de Weights & Biases "diazcanorinconjavier3a-universidad-polit-cnica-de-madrid", donde se encuentra centralizado el proyecto. El acceso permite visualizar los experimentos, métricas, configuraciones de entrenamiento y resultados generados durante el desarrollo del modelo dentro del workspace compartido. Por lo tanto, el profesor sólo debe acceder desde su correo al enlace "https://wandb.ai/diazcanorinconjavier3a-universidad-polit-cnica-de-madrid/MLOPS_Project?nw=nwuserdiazcanorinconjavier3a"

---

********************************************************************
ENDPOINT ACCESIBLE CON EL SERVICIO EN PRODUCCIÓN
********************************************************************

En este proyecto no se ha realizado un despliegue en un entorno cloud (como AWS, GCP o Azure). La razón principal es que el objetivo de la asignatura ha sido centrarse en las buenas prácticas de MLOps a nivel de desarrollo, experimentación y estructuración del proyecto, incluyendo el entrenamiento del modelo, la creación de una API de inferencia y la contenedorización básica.

El despliegue en producción no se ha abordado en profundidad durante el curso, por lo que se ha priorizado garantizar un entorno reproducible y funcional en local. En su lugar, el proyecto se puede ejecutar completamente en entorno local mediante FastAPI, permitiendo realizar inferencias sobre el modelo de forma sencilla y controlada.

De este modo, cualquier usuario puede levantar la API en su propio entorno utilizando las instrucciones del README, sin necesidad de servicios cloud. Esta decisión permite mantener el enfoque en los aspectos fundamentales de MLOps trabajados en la asignatura, asegurando al mismo tiempo la reproducibilidad y funcionalidad del sistema.














