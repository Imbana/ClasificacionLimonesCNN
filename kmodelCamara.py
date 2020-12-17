# MicroPython v0.5.0-29-g97fad3a on 2020-03-13; Sipeed_M1 with kendryte-k210

# Importe de librerias
import sensor, image, lcd, time, utime
import KPU as kpu

# Configuración inicial de la pantalla LCD y la camara OV2640
lcd.init() # Inicializa la pantalla
sensor.reset() # Inicializa la camara
sensor.set_pixformat(sensor.RGB565) # Define el formato de color de la imagen
sensor.set_framesize(sensor.QVGA) # Establece la captura de imagen como QVGA (320x240)
sensor.set_windowing((224,224)) # Establece el tamaño de imagen con el que se entreno la red
sensor.set_vflip(1) # Rotación vertical de la imagen
sensor.set_saturation(-3) # Saturacion
sensor.set_brightness(-3) # brightness
sensor.set_contrast(-3) # contrast
lcd.clear() # Limpia la pantalla y la deja en negro

# Descripción y carga del modelo
labels = ['Acaro','Bueno','Manchado'] # Etiquetas de la ultima capa de la red
task = kpu.load('/sd/3clases.kmodel') # Acá va al ubicación del archivo .kmodel   (CARGA)
kpu.set_outputs(task, 0, 1, 1, 3) # Aqúi van las dimensiones de la ultima capa de la red

while(True):

    tick1 = utime.ticks_ms()

    # Ejecucion del modelo en tiempo real
    kpu.memtest() # Verifica la memoria disponible
    img = sensor.snapshot() # Captura de la imagen
    fmap = kpu.forward(task, img) # Ejecuta la red neuronal con la imagen capturada
    plist = fmap[:] # Extrae las probabilidades dentro de una lista
    pmax = max(plist) # Escoge de las probabilidades la mayor
    max_index = plist.index(pmax) # Extrae el indice de la clase con la mayor probabilidad

    tick2 = utime.ticks_ms()

    etime = utime.ticks_diff(tick2,tick1)

    # Impresion del resultado en la pantalla LCD (Etiqueta de la clase y probabilidad)
    a = img.draw_string(0,0, str(labels[max_index].strip()), color = (73, 235, 52), scale=2)
    a = img.draw_string(0,20, str(pmax), color = (73, 235, 52), scale=2)
    a = img.draw_string(0,40, str('Time:'), color = (73, 235, 52), scale=2)
    a = img.draw_string(0,60, str(etime),color = (73, 235, 52), scale=2)
    print((pmax, labels[max_index].strip()))
    a = lcd.display(img)

a = kpu.deinit(task)
