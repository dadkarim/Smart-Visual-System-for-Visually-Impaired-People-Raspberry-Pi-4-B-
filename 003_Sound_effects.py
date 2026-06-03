import pygame
import numpy as np
import time

pygame.init()

sound = pygame.mixer.Sound("/home/pi/Documents/Sound_Effects/beep.wav")

# Find a free channel to play the sound on.
channel = pygame.mixer.find_channel()

#for float values use this for loop
#for i in np.arange(Rangelow, Rangehigh, Delta):
distance_left = 100
distance_right = 1000
distance_forward = 20

if (distance_left >=0 and distance_left <= 50):
    x = 1
    print("x:",x)
    
if (distance_left >50 and distance_left <= 100):
    x = 0.5
    print("x:",x)
    
if (distance_left >100 and distance_left <= 200):
    x = 0.1
    print("x:",x)

if (distance_left >200):
    x = 0.0
    print("x:",x)

if (distance_right >=0 and distance_right <= 50):
    y = 1
    print("y:",y)
    
if (distance_right >50 and distance_right <= 100):
    y = 0.5
    print("y:",y)
    
if (distance_right >100 and distance_right <= 200):
    y = 0.1
    print("y:",y)

if (distance_right >200):
    y = 0.0
    print("y:",y)
    
if (distance_forward >=0 and distance_forward <= 50):
    z = 1
    print("z:",z)
    
if (distance_forward >50 and distance_forward <= 100):
    z = 0.5
    print("z:",z)
    
if (distance_forward >100 and distance_forward <= 200):
    z = 0.1
    print("z:",z)

if (distance_forward >200):
    z = 0.0
    print("z:",z)

# pan volume full loudness on the left, and silent on right.
channel.set_volume(x, y)
channel.play(sound)

time.sleep(0.2) 

channel.set_volume(z, z)
channel.play(sound)