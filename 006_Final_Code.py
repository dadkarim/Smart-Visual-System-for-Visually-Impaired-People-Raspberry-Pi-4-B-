import sys
import os
import cv2
import numpy as np
from picamera.array import PiRGBArray
from picamera import PiCamera
import tensorflow as tf
import argparse
import RPi.GPIO as GPIO
from gpiozero import Button
import pygame
import time
from time import sleep
import pyttsx3
import pytesseract

#GPIO Number
Button1 = Button(9)
Button2 = Button(11)
Button3 = Button(10)

TRIG_L = 6
TRIG_F = 16
TRIG_R = 20
TRIG_G = 5

ECHO_L = 19
ECHO_F = 26
ECHO_R = 21
ECHO_G = 12
    
GPIO.setmode(GPIO.BCM) 

GPIO.setwarnings(False)
    
#Setup Button
GPIO.setup(10,GPIO.IN,pull_up_down=GPIO.PUD_UP)
GPIO.setup(9,GPIO.IN,pull_up_down=GPIO.PUD_UP)
GPIO.setup(11,GPIO.IN,pull_up_down=GPIO.PUD_UP)

pygame.mixer.init()

sound_W = pygame.mixer.Sound("/home/pi/Documents/Sound_Effects/Welcome.wav")

# Find a free channel to play the sound on.
channel_W = pygame.mixer.find_channel()
channel_W.set_volume(1, 1)
channel_W.play(sound_W)
sleep(6)

# initialisation
engine = pyttsx3.init()

engine.setProperty('rate', 150) # Decrease the Speed Rate default is 200

welcome = "Welcome"
Text_1= "Press button 1 to activate object recognation mode"
Text_2= "press button 2 to activate Nevigation mode"
Text_3= "press button 3 to activate readig mode"
#print("", Text)
Text_4= "Press any Button: 1,2 or 3 until you hear a beep sound"
print("", Text_4)

# testing
engine.say(welcome)
engine.say(Text_1)
engine.say(Text_2)
engine.say(Text_3)
engine.say(Text_4)
engine.runAndWait()

def B1():
    pygame.mixer.init()

    sound_B = pygame.mixer.Sound("/home/pi/Documents/Sound_Effects/beep.wav")

    # Find a free channel to play the sound on.
    channel_B = pygame.mixer.find_channel()
    channel_B.set_volume(1, 1)
    channel_B.play(sound_B)
    sleep(1)
    Text_5 = "object recognation mode is activated"
    print("Button 1 is pressed")
    engine.say(Text_5)
    engine.say("To Exit Press Button 2 or 3")
    engine.runAndWait()
    #while True:
        
    # Set up camera constants
    #camera.resolution = (2592, 1944)
    IM_WIDTH = 320#640#1280
    IM_HEIGHT = 320#480#720
    #IM_WIDTH = 640    Use smaller resolution for
    #IM_HEIGHT = 480   slightly faster framerate

    # Select camera type (if user enters --usbcam when calling this script,
    # a USB webcam will be used)
    camera_type = 'picamera'
    parser = argparse.ArgumentParser()
    parser.add_argument('--usbcam', help='Use a USB webcam instead of picamera',
                        action='store_true')
    args = parser.parse_args()
    if args.usbcam:
        camera_type = 'usb'

    # This is needed since the working directory is the object_detection folder.
    sys.path.append('..')

    # Import utilites
    from utils import label_map_util
    from utils import visualization_utils as vis_util

    # Name of the directory containing the object detection module we're using
    MODEL_NAME = 'ssdlite_mobilenet_v2_coco_2018_05_09'

    # Grab path to current working directory
    CWD_PATH = os.getcwd()

    # Path to frozen detection graph .pb file, which contains the model that is used
    # for object detection.
    PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

    # Path to label map file
    PATH_TO_LABELS = os.path.join(CWD_PATH,'data','mscoco_label_map.pbtxt')

    # Number of classes the object detector can identify
    NUM_CLASSES = 90

    ## Load the label map.
    # Label maps map indices to category names, so that when the convolution
    # network predicts `5`, we know that this corresponds to `airplane`.
    # Here we use internal utility functions, but anything that returns a
    # dictionary mapping integers to appropriate string labels would be fine
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    # Load the Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        sess = tf.compat.v1.Session(graph=detection_graph)


    # Define input and output tensors (i.e. data) for the object detection classifier

    # Input tensor is the image
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # Output tensors are the detection boxes, scores, and classes
    # Each box represents a part of the image where a particular object was detected
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    # Each score represents level of confidence for each of the objects.
    # The score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

    # Number of objects detected
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    # Initialize frame rate calculation
    frame_rate_calc = 1
    freq = cv2.getTickFrequency()
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Initialize camera and perform object detection.
    # The camera has to be set up and used differently depending on if it's a
    # Picamera or USB webcam.

    # I know this is ugly, but I basically copy+pasted the code for the object
    # detection loop twice, and made one work for Picamera and the other work
    # for USB.

    ### Picamera ###
    if camera_type == 'picamera':
        # Initialize Picamera and grab reference to the raw capture
        camera = PiCamera()
        camera.rotation = 270
        camera.resolution = (IM_WIDTH,IM_HEIGHT)
        camera.framerate = 5
        rawCapture = PiRGBArray(camera, size=(IM_WIDTH,IM_HEIGHT))
        rawCapture.truncate(0)

        for frame1 in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):

            t1 = cv2.getTickCount()
            
            # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
            # i.e. a single-column array, where each item in the column has the pixel RGB value
            frame = np.copy(frame1.array)
            frame.setflags(write=1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_expanded = np.expand_dims(frame_rgb, axis=0)

            # Perform the actual detection by running the model with the image as input
            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: frame_expanded})

            # Draw the results of the detection (aka 'visulaize the results')
            vis_util.visualize_boxes_and_labels_on_image_array(
                frame,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8,
                min_score_thresh=0.10)

            cv2.putText(frame,"FPS: {0:.2f}".format(frame_rate_calc),(30,50),font,1,(255,255,0),2,cv2.LINE_AA)

            # All the results have been drawn on the frame, so it's time to display it.
            cv2.imshow('Object detector', frame)

            t2 = cv2.getTickCount()
            time1 = (t2-t1)/freq
            frame_rate_calc = 1/time1

            # Press 'q' to quit
            if cv2.waitKey(1) == ord('q'):
                break
            if Button2.is_pressed or Button3.is_pressed:
                break
            #sleep(3)
            rawCapture.truncate(0)

        camera.close()

    ### USB webcam ###
    elif camera_type == 'usb':
        # Initialize USB webcam feed
        camera = cv2.VideoCapture(0)
        ret = camera.set(3,IM_WIDTH)
        ret = camera.set(4,IM_HEIGHT)

        while(True):

            t1 = cv2.getTickCount()

            # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
            # i.e. a single-column array, where each item in the column has the pixel RGB value
            ret, frame = camera.read()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_expanded = np.expand_dims(frame_rgb, axis=0)

            # Perform the actual detection by running the model with the image as input
            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: frame_expanded})

            # Draw the results of the detection (aka 'visulaize the results')
            vis_util.visualize_boxes_and_labels_on_image_array(
                frame,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8,
                min_score_thresh=0.85)

            cv2.putText(frame,"FPS: {0:.2f}".format(frame_rate_calc),(30,50),font,1,(255,255,0),2,cv2.LINE_AA)
            
            # All the results have been drawn on the frame, so it's time to display it.
            cv2.imshow('Object detector', frame)

            t2 = cv2.getTickCount()
            time1 = (t2-t1)/freq
            frame_rate_calc = 1/time1

            # Press 'q' to quit
            if cv2.waitKey(1) == ord('q'):
                break
            if Button2.is_pressed or Button3.is_pressed:
                break
            
        camera.release()

    #if Button2.is_pressed or Button3.is_pressed:
     #   break
    cv2.destroyAllWindows()
    #sleep(2)
def B2():
    pygame.mixer.init()
    sound_B = pygame.mixer.Sound("/home/pi/Documents/Sound_Effects/beep.wav")

    # Find a free channel to play the sound on.
    channel_B = pygame.mixer.find_channel()
    channel_B.set_volume(1, 1)
    channel_B.play(sound_B)
    sleep(0.10)
    Text_6 = "navigation mode is activated"
    print("Button:2 is pressed")
    engine.say(Text_6)
    engine.say("To Exit Press Button 1 or 3")
    engine.runAndWait()
    sleep(0.1)
    print ("Warming up...")
    print ("Starting Measuring...")
    
    while True:    
        GPIO.setmode(GPIO.BCM) 
        GPIO.setwarnings(False)
        
        #Setup Button
        GPIO.setup(10,GPIO.IN,pull_up_down=GPIO.PUD_UP)
        GPIO.setup(9,GPIO.IN,pull_up_down=GPIO.PUD_UP)
        GPIO.setup(11,GPIO.IN,pull_up_down=GPIO.PUD_UP)
        
        GPIO.setup(TRIG_L,GPIO.OUT)
        GPIO.setup(TRIG_F,GPIO.OUT)
        GPIO.setup(TRIG_R,GPIO.OUT)
        GPIO.setup(TRIG_G,GPIO.OUT)

        GPIO.setup(ECHO_L,GPIO.IN)
        GPIO.setup(ECHO_F,GPIO.IN)
        GPIO.setup(ECHO_R,GPIO.IN)
        GPIO.setup(ECHO_G,GPIO.IN)

        GPIO.output(TRIG_L, False)
        sleep(1)
        GPIO.output(TRIG_L, True)
        sleep(0.00001)
        GPIO.output(TRIG_L, False)
        while GPIO.input(ECHO_L)==0:
            start_time_L = time.time()
        while GPIO.input(ECHO_L)==1: 
            Bounce_end_L = time.time()
        sleep(0.01)
        pulse_duration_L = Bounce_end_L - start_time_L
        
        GPIO.output(TRIG_F, False)
        sleep(1)
        GPIO.output(TRIG_F, True)
        sleep(0.00001)
        GPIO.output(TRIG_F, False)
        while GPIO.input(ECHO_F)==0:
            start_time_F = time.time()
        while GPIO.input(ECHO_F)==1: 
            Bounce_end_F = time.time()
        pulse_duration_F = Bounce_end_F - start_time_F
        
        GPIO.output(TRIG_R, False)
        sleep(1)
        GPIO.output(TRIG_R, True)
        sleep(0.00001)
        GPIO.output(TRIG_R, False)    
        while GPIO.input(ECHO_R)==0:
            start_time_R = time.time()    
        while GPIO.input(ECHO_R)==1: 
            Bounce_end_R = time.time()
        pulse_duration_R = Bounce_end_R - start_time_R
        
        GPIO.output(TRIG_G, False)
        sleep(1)
        GPIO.output(TRIG_G, True)
        sleep(0.00001)
        GPIO.output(TRIG_G, False)    
        while GPIO.input(ECHO_G)==0:
            start_time_G = time.time()    
        while GPIO.input(ECHO_G)==1: 
            Bounce_end_G = time.time()
        pulse_duration_G = Bounce_end_G - start_time_G

        pygame.init()

        sound_L_R = pygame.mixer.Sound("/home/pi/Documents/Sound_Effects/beep_L.wav")

        # Find a free channel to play the sound on.
        channel_L_R = pygame.mixer.find_channel()

        distance_L = round(pulse_duration_L * 17150, 2)
        print ("distance_LEFT:",distance_L,"cm")

        if (distance_L >=0 and distance_L <= 50):
            x = 1.0
            print("x:",x)
            
        if (distance_L >50 and distance_L <= 100):
            x = 0.5
            print("x:",x)
            
        if (distance_L >100 and distance_L <= 200):
            x = 0.1
            print("x:",x)

        if (distance_L >200):
            x = 0.0
            print("x:",x)

        distance_R = round(pulse_duration_R * 17150, 2)
        print ("distance_RIGHT:",distance_R,"cm")

        if (distance_R >=0 and distance_R <= 50):
            y = 1.0
            print("y:",y)
            
        if (distance_R >50 and distance_R <= 100):
            y = 0.5
            print("y:",y)
            
        if (distance_R >100 and distance_R <= 200):
            y = 0.1
            print("y:",y)

        if (distance_R >200):
            y = 0.0
            print("y:",y)
            
        # pan volume loudness on the left, and right.
        channel_L_R.set_volume(x, y)
        channel_L_R.play(sound_L_R)

        sleep(0.8)
        
        pygame.mixer.init()

        sound_F = pygame.mixer.Sound("/home/pi/Documents/Sound_Effects/beep.wav")

        # Find a free channel to play the sound on.
        channel_F = pygame.mixer.find_channel()
        
        distance_F = round(pulse_duration_F * 17150, 2)
        print ("distance_FORWARD:",distance_F,"cm")

        if (distance_F >=0 and distance_F <= 50):
            z = 1.0
            print("z:",z)
            
        if (distance_F >50 and distance_F <= 100):
            z = 0.5
            print("z:",z)
            
        if (distance_F >100 and distance_F <= 200):
            z = 0.1
            print("z:",z)

        if (distance_F >200):
            z = 0.0
            print("z:",z)
            
        # pan volume loudness.
        channel_F.set_volume(z, z)
        channel_F.play(sound_F)
        
        sleep(0.8)
        
        pygame.mixer.init()

        sound_G = pygame.mixer.Sound("/home/pi/Documents/Sound_Effects/beep_G.wav")

        # Find a free channel to play the sound on.
        channel_G = pygame.mixer.find_channel()
        
        distance_G = round(pulse_duration_G * 17150, 2)
        print ("distance_GROUND:",distance_G,"cm")

        if (distance_G >=0 and distance_G <= 200):
            G = 0.0
            print("G:",G)
            
        if (distance_G >200 and distance_G <= 250):
            G = 0.5
            print("G:",G)

        if (distance_G >250):
            G = 1.0
            print("G:",G)
            
        # pan volume loudness.
        channel_G.set_volume(G, G)
        channel_G.play(sound_G)
        if Button1.is_pressed or Button2.is_pressed or Button3.is_pressed:
            break              
def B3():
    # initialisation
    #pygame.mixer.init()
    sound_B = pygame.mixer.Sound("/home/pi/Documents/Sound_Effects/beep.wav")

    # Find a free channel to play the sound on.
    channel_B = pygame.mixer.find_channel()
    channel_B.set_volume(1, 1)
    channel_B.play(sound_B)
    sleep(1)
    Text_7 = "Reading mode is activated"
    print("Button:3 is pressed")
    engine.say(Text_7)
    engine.say("To Exit Press Button 1 or 2")
    engine.runAndWait()
    
    IM_WIDTH = 320
    IM_HEIGHT = 320
    camera_type = 'picamera'
    while True:
        if camera_type == 'picamera':
            # Initialize Picamera and grab reference to the raw capture
            camera = PiCamera()
            camera.rotation = 270
            camera.resolution = (IM_WIDTH,IM_HEIGHT)
            camera.framerate = 60
            camera.brightness = 60
            #camera.contrast = 5
            #camera.image_effect = 'colorbalance'
            #camera.exposure_mode = 'nightpreview'
            #camera.awb_mode = 'flash'
            rawCapture = PiRGBArray(camera, size=(IM_WIDTH,IM_HEIGHT))
            rawCapture.truncate(0)
            # allow the camera to warmup
            sleep(0.20)
            
            for frame in camera.capture_continuous\
                (rawCapture, format="bgr",use_video_port=True):
                # grab the raw NumPy array representing the image, then initialize the timestamp
                # and occupied/unoccupied text
                image = frame.array
                
                imgH,imgW,_ = image.shape
                x1,y1,w1,h1 = 0,0,imgH,imgW
                
                gray_1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                threshold_1 = cv2.adaptiveThreshold\
                              (gray_1, 255,\
                               cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                               cv2.THRESH_BINARY, 51, 10)
                image = cv2.fastNlMeansDenoising(threshold_1, 100, 7, 21)
                
                img2str = pytesseract.image_to_string(image)
                imgboxes = pytesseract.image_to_boxes(image)
                for boxes in imgboxes.splitlines():
                    boxes = boxes.split(' ')
                    x,y,w,h = int(boxes[1]),int(boxes[2]),int(boxes[3]),int(boxes[4])
                    cv2.rectangle(image, (x, imgH-y),(w,imgH-h),(0,0,255),3)
                cv2.putText(image, img2str, (x1 + int(w1/50),y1 + int(h1/50)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.imshow("Video", image)
                print("", img2str)
                text = img2str
                if not img2str:
                    text = "No text is detected"
                engine.say(text)
                engine.runAndWait()
                    
                # clear the stream in preparation for the next frame
                rawCapture.truncate(0)
                # Press 'q' to quit
                if cv2.waitKey(1) == ord('q'):
                    break
                if Button1.is_pressed or Button2.is_pressed:
                    break

            camera.close()
        cv2.destroyAllWindows()
        if Button1.is_pressed or Button2.is_pressed:
            break
        sleep(0.20)

while True:
    while (Button1.is_pressed):
        while (Button1.is_pressed or
           not Button2.is_pressed or
           not Button3.is_pressed ):
        
            
            sleep(0.10)
            Button1.when_pressed = B1
            sleep(2)
            if Button2.is_pressed or Button3.is_pressed:
                print("Stop button is pressed for B1")
                engine.say("Exiting Object Detection Mode")
                #engine.runAndWait()
                sleep(0.10)
                break
        sleep(0.10)
    while (Button2.is_pressed):
        while (not Button1.is_pressed or
               Button2.is_pressed or
               not Button3.is_pressed):
            
            sleep(0.10)    
            Button2.when_pressed = B2
            if Button1.is_pressed or Button3.is_pressed:
                print("Stop button is pressed for B2")
                engine.say("Exiting Navigation Mode")
                engine.runAndWait()
                break
        sleep(0.20)
    while (Button3.is_pressed):
        while (not Button1.is_pressed or
           not Button2.is_pressed or
           Button3.is_pressed):
        
            
                
            sleep(1)
            Button3.when_pressed = B3
            if Button1.is_pressed or Button2.is_pressed:
                print("Stop button is pressed for B3")
                engine.say("Exiting Reading Mode")
                sleep(1)
                break
        sleep(1)    
    sleep(2)
    if cv2.waitKey(1) == ord('e'):
        break
cv2.destroyAllWindows()   
