from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2
import argparse
import io
import time
import picamera
from picamera.array import PiRGBArray
from picamera import PiCamera



from PIL import Image
from tflite_runtime.interpreter import Interpreter


def load_labels(path):
  with open(path, 'r') as f:
    return {i: line.strip() for i, line in enumerate(f.readlines())}


def set_input_tensor(interpreter, image):
  tensor_index = interpreter.get_input_details()[0]['index']
  input_tensor = interpreter.tensor(tensor_index)()[0]
  input_tensor[:, :] = image


def classify_image(interpreter, image, top_k=1):
  """Returns a sorted array of classification results."""
  #obtiene los tensores en la imagen de entrada con la funcion set_input_tensor
  set_input_tensor(interpreter, image)
  interpreter.invoke()
  
  #obtener los tensores de salida  
  output_details = interpreter.get_output_details()[0]
  output = np.squeeze(interpreter.get_tensor(output_details['index']))

  # If the model is quantized (uint8 data), then dequantize the results
  if output_details['dtype'] == np.uint8:
    scale, zero_point = output_details['quantization']
    output = scale * (output - zero_point)

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



#cargar y asignar los ternsores del modelo y etiquetas
  labels = load_labels(args.labels)

  interpreter = Interpreter(args.model)
  interpreter.allocate_tensors()
  
#obtener los tensores de entrada  
  _, height, width, _ = interpreter.get_input_details()[0]['shape']
  
#Caputa de la imagen con PiCamera , primero se especifican las caracteristicas de la captura 
  camera = PiCamera()
  camera.resolution = (640, 480)
  camera.framerate = 30
  rawCapture = PiRGBArray(camera, size=(640, 480))
 
 
  time.sleep(0.1)
  # se hace una captura continua
  for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        

    #preparar la imagen coonvirtiendola en un array Numpy
    image = frame.array
  
    frame_resized = cv2.resize(image, (224, 224))
    #input_data = np.expand_dims(frame_resized, axis=0)
    
    
    #clasificar la imagen entrada y tomar el tiempo que se demora en este proceso
      
    start_time = time.time()
    
    results = classify_image(interpreter, frame_resized)

    elapsed_ms = (time.time() - start_time) * 1000
      
    # Mostrar resultados en la pantalla
    label_id, prob = results[0]
    
    print(labels[label_id]+"   "+str(elapsed_ms))       
    font = cv2.FONT_HERSHEY_SIMPLEX
    color=(0,0,255)
    images = cv2.putText(image,labels[label_id]+"   "+str(prob) , (00,185), font, 1,color,2, cv2.LINE_AA) 
        
        # Show the frame
    cv2.imshow("Frame", images)
    
     
    # salir si se preciona q 
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
    
    # clear the stream in preparation for the next frame
    rawCapture.truncate(0)
    
  cv2.destroyAllWindows()


#funcion principal
if __name__ == '__main__':
  main()
