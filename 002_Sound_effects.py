from pygame import mixer
mixer.init() 
sound=mixer.Sound("/home/pi/Documents/Sound_Effects/beep.wav")


sound.play()

sound.set_volume(0.1)