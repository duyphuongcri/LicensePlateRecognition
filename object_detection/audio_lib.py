from pygame import mixer

file = 'sound_3h_lien_tuc.mp3'
mixer.init()
mixer.music.load(file)
mixer.music.play()