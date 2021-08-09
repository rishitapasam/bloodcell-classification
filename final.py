from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
from keras.models import load_model
import cv2
import numpy as np
from tensorflow import keras

main = tkinter.Tk()
main.title("Blood Cell Classification")
main.geometry("1300x1200")

def upload():
    global filename
    text.delete('1.0', END)
    filename = askopenfilename(initialdir = ".")
    pathlabel.config(text=filename)
    text.insert(END,"Dataset loaded\n\n")

def imagePreprocess():
    global filename
    global test
    test_data=[]
    image=cv2.imread(filename)
    img3 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img3 = cv2.resize(img3,(60,60))
    test_data.append(img3)
    test=np.array(test_data).reshape(-1,60,60,3)
    test=test/255.0
    test.shape
    text.insert(END,"Image preprossed\n\n")

def loadmodel():
    global model
    model=model = keras.models.load_model('bc_model.h5')
    text.insert(END,"trained Model loaded")
    

def predict():
    global test
    global model
    classes=['EOSINOPHIL', 'LYMPHOCYTE', 'MONOCYTE', 'NEUTROPHIL']
    pred = model.predict_classes(test)
    text.insert(END," Predicted class for Image: "+str(classes[pred[0]])+"\n\n")

font = ('times', 16, 'bold')
title = Label(main, text='Blood Cell Classifiction using CNN')
title.config(bg='dark salmon', fg='black')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 14, 'bold')
upload = Button(main, text="Upload Image File", command=upload)
upload.place(x=700,y=100)
upload.config(font=font1)

pathlabel = Label(main)
pathlabel.config(bg='dark orchid', fg='white')  
pathlabel.config(font=font1)
pathlabel.place(x=700,y=150)

imp= Button(main, text="Image Preprocess", command=imagePreprocess)
imp.place(x=700,y=200)
imp.config(font=font1)

ml = Button(main, text="Model Load", command=loadmodel)
ml.place(x=700,y=250)
ml.config(font=font1)

pt = Button(main, text="Predict For Image", command=predict)
pt.place(x=700,y=300)
pt.config(font=font1)

font1 = ('times', 12, 'bold')
text=Text(main,height=30,width=80)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=100)
text.config(font=font1)

main.config(bg='dodger blue')
main.mainloop()




