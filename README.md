# ClasificacionLimonesCNN
Se plantea la clasificacion de limones Tahiti, teniendo tres caracteristicas, limones buenos(con coloracion uniformemente verde y sin ningun tipo de daños en su cascara), limones con manchas amarillas y limones con acaros(cualquier dano que sea visible en el limon)

Para esto se creo una base de datos de limones con mas de 600 imagenes de cada limon, se entreno una red neuronal **MobilNet v1**  utilizando la tecnica transfer learning, y luego se implmenta en un sistema embebido, Raspberry pi 3 b+ y la MAix BiT, con los formatos .tflite y kmodel respectivamente.

## Contenido 
Los documentos adjuntados realizan las siguiente opereciones

Codigo_proyecto_final : Toma una carpeta que tenga clasificadas las imagenes de lo limones en 3 categorias, las separa en dataset para Train, Validacion y Prueba. Escoge el modelo de red neuronal, lo modifica segun la necesidad, entrena y guarda en formato deseado. (Aclarar que para el proyecto  se hizo uso del  [framework **aXeleRate**](https://github.com/AIWintermuteAI/aXeleRate) diseñado por Dmitry Maslov  debido a que nos proporciona los modelos en formato necesarios para los sistemas embebidos utilizados)
 
 Video : este .py nos permite correr el modelo cargando un Video que se tenga en la Raspberry e ir clasificando cada parte del video
 Imagene : este .py nos permite correr el modelo cargando Imagenes desde alguna carpeta que se encuentre en la Raspberry e ir clasificando cada Imagen
 (en estos dos .py se debe modificar el path donde se encuentra el video(con su nombre) o la carpeta de imagenes )
 
 
 PiCamera: este .py nos permite correr el modelo cargando imagenes directas desde la camara Picamera.
 
 Los anteriores .py se corren desde la terminal con el siguiente codigo, teniendo en cuenta los path de donde se encuetra y el nombre del modelo entrenado con sus etiquetas.
 ```bash
python3   tensorflowDemo . py−−model   /home/ p i /Documents/ ultimo .  t f l i t e−−l a b e l s   /home/p i /Documents/ l a b e l s . t x 
```

![](/4.PNG)

