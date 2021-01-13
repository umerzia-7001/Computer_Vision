import os
import tkinter as tk
from tkinter import *
from tkinter import messagebox

import cv2  # for image processing
import easygui  # to open the filebox
import matplotlib.pyplot as plt

print('done')

top = tk.Tk()

top.geometry('400x400')
top.title('Cartoonify Your Image !')
top.configure(background='dark gray')
label = Label(top, background='snow2', font=('Helvetica', 30, 'bold'))


def upload():
    ImagePath = easygui.fileopenbox()
    cartoonify(ImagePath)



def cartoonify(ImagePath):
    # read the image
    originalmage = cv2.imread(ImagePath)
    originalmage = cv2.resize(originalmage, (960, 540))
    originalmage = cv2.cvtColor(originalmage, cv2.COLOR_BGR2RGB)
    # print(image)  # image is stored in array

    # confirm that image is chosen
    if originalmage is None:
        print("Can not find any image. Choose appropriate file")
        sys.exit()

    ReSized1 = cv2.resize(originalmage, (960, 540))


    # converting an image to grayscale
    grayScaleImage = cv2.cvtColor(originalmage, cv2.COLOR_BGR2GRAY)
    ReSized2 = cv2.resize(grayScaleImage, (960, 540))
    plt.imshow(ReSized2, cmap='gray')

    # applying median blur to smoothen an image
    # subtracts center pixel value from neighbouring pixel means's
    smoothGrayScale = cv2.medianBlur(grayScaleImage, 5)
    ReSized3 = cv2.resize(smoothGrayScale, (960, 540))


    ''' retrieving the edges for cartoon effect , gives one value to pixel having threshold and 
     gives other values  to rest pixels
     by using thresholding technique thresh_binary'''
    getEdge = cv2.adaptiveThreshold(smoothGrayScale, 255,
                                    cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY, 9, 9)

    ReSized4 = cv2.resize(getEdge, (960, 540))


    # applying bilateral filter to remove noise
    # and keep edge sharp as required
    colorImage = cv2.bilateralFilter(originalmage, 9, 300, 300)
    ReSized5 = cv2.resize(colorImage, (960, 540))


    # masking edged image with our "BEAUTIFY" image
    cartoonImage = cv2.bitwise_and(colorImage, colorImage, mask=getEdge)

    ReSized6 = cv2.resize(cartoonImage, (960, 540))


    # Plotting the whole transition
    images = [ReSized1, ReSized2, ReSized3, ReSized4, ReSized5, ReSized6]

    fig, axes = plt.subplots(3, 2, figsize=(8, 8), subplot_kw={'xticks': [], 'yticks': []},
                             gridspec_kw=dict(hspace=0.1, wspace=0.1))
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i], cmap='gray')

    save1 = Button(top, text="Save cartoon image", command=lambda: save(ReSized6, ImagePath), padx=30, pady=5,anchor='w',bd=10)
    save1.configure(background="snow2", foreground='gray6', font=('Helvetica', 20, 'bold'))
    save1.pack(side=TOP, pady=50)

    plt.show()

def save(ReSized6, ImagePath):
    # saving an image using imwrite()
    newName = "cartoonified_Image"
    path1 = os.path.dirname(ImagePath)
    extension = os.path.splitext(ImagePath)[1]
    path = os.path.join(path1, newName + extension)
    cv2.imwrite(path, cv2.cvtColor(ReSized6, cv2.COLOR_RGB2BGR))
    I = "Image saved by name " + newName + " at " + path
    tk.messagebox.showinfo(title=None, message=I)


upload = Button(top, text="Select Image To Cartoonify", command=upload, padx=10, pady=5,bd=10,bg="snow2", foreground='gray6', font=('Helvetica', 20, 'bold'))
upload.configure(background="steel blue", font=('Helvetica', 20, 'bold'))
upload.pack(side=TOP, pady=50)

top.mainloop()

