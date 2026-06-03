# import required modules
from pydub import AudioSegment
from pydub.playback import play
  
# for playing wav file
song = AudioSegment.from_wav("/home/pi/Documents/Sound_Effects/beep.wav")
print('playing sound using  pydub')
play(song)

# for playing mp3 file
#song = AudioSegment.from_mp3("/home/pi/Documents/Sound_Effects/emergency_alarm.mp3")
print('playing sound using  pydub')
play(song)