import numpy as np
import cv2

import matplotlib.pylab as plt
import argparse
import io
import os
import sys
import time

import picamera
from picamera.array import PiRGBArray
from picamera import PiCamera

from PIL import Image
from tflite_runtime.interpreter import Interpreter

# devuelve un diccionario con todas las clases
def load_labels(path):
  with open(path, 'r') as f:
    return {i: line.strip() for i, line in enumerate(f.readlines())}
def clasificar_imagen(interpreter,imagen,top_k=1):
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    input_shape = input_details[0]['shape']
    input_data = np.array(imagen, dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    
    interpreter.invoke()


    output = np.squeeze(interpreter.get_tensor(output_details[0]['index']))
    
    ordered = np.argpartition(-output, top_k)
    return [(i, output[i]) for i in ordered[:top_k]]
    

def main():

    # de aqui para asignar la ruta del modelo desde la terminal
    parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
      '--model', help='File path of .tflite file.', required=True)
    parser.add_argument(
      '--labels', help='File path of labels file.', required=True)
    
    args = parser.parse_args()

    
    #pathLabels=('/home/pi/Documents/labels.txt')    
    #pathTflite=('/home/pi/Documents/ultimo.tflite')
    
    labels = load_labels(args.labels)
    
        
    interpreter = Interpreter(args.model)
    interpreter.allocate_tensors()
    
    camera = PiCamera()
    camera.resolution = (640, 480)
    camera.framerate = 30
    rawCapture = PiRGBArray(camera, size=(640, 480))   

        
    stream = io.BytesIO()
   
    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        
        
        image = frame.array 
    
        frame=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        
    
        Imagen_normed = frame/ 255
    
        
        Imagen_show = cv2.resize(image, (600, 600))
    
        Imagen_resized = cv2.resize(Imagen_normed, (224, 224))
        Imagen_espnaded=np.expand_dims(Imagen_resized, axis=0)
    
          
        start_time = time.time()
    
        results = clasificar_imagen(interpreter, Imagen_espnaded)

        elapsed_ms = round((time.time() - start_time) * 1000,2)
        

        label_id, prob = results[0]
    
        print(labels[label_id]+"   "+str(elapsed_ms))       
        font = cv2.FONT_HERSHEY_SIMPLEX
        color=(14,129,60)
        images = cv2.putText(Imagen_show,str(elapsed_ms)+"ms  "+labels[label_id]+"   "+str(prob) , (00,100), font, 1,color,2, cv2.LINE_AA) 
               

        cv2.imshow("Frame", images)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        rawCapture.truncate(0)

        
        
    cv2.destroyAllWindows()

    
if __name__ == '__main__':
  main()