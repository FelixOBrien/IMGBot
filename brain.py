# -*- coding: utf-8 -*-
from keras.models import load_model
from keras.preprocessing import image
from network import convertLabels
import numpy as np
from PIL import ImageTk, Image
import tkinter as tk
#Load the model
model = load_model('model.h5')
#Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
test_image= image.load_img('rose.jpg', color_mode = "rgb", target_size = (64, 64,3)) 
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = model.predict(test_image)   
print(convertLabels(np.argmax(result)))
win = tk.Tk()
win.title("Mnist Analysis")
img =ImageTk.PhotoImage(Image.open('rose.jpg'))
string = "The model thinks this is a {}".format( convertLabels(np.argmax(result)))
tk.Label(win, image = img).pack()
tk.Label(win, text=string).pack()
win.mainloop()