from django.shortcuts import render
from django.http import HttpResponse

from .models import Img 
from .forms import ImgForm

import numpy as np
import cv2


def main_page(request):
    return render(request, 'main\MainPage.html')

def model_page(request):

    def ReLU(v):
        return np.maximum(0, v)

    def softmax(v):
        out = np.exp(v)
        return out / np.sum(out)

    def predict(x):
        t1 = x @ w1 + b1
        h1 = ReLU(t1)
        t2 = h1 @ w2 + b2
        z = softmax(t2)
        return z

    answer = ""

    if request.method == 'POST':
        try:
            aw = np.load("weights.npz")
            b1, w1, b2, w2 = aw["arr_0"], aw["arr_1"], aw["arr_2"], aw["arr_3"]
            class_names = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
            
            file = request.FILES
            f = file['img'].file
            f = f.read()

            nparr = np.fromstring(f, np.uint8)
            img_np = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

            up_points = (8, 8)
            res_img = cv2.resize(img_np, up_points, interpolation = cv2.INTER_AREA)
            #bin_res_img = cv2.threshold(res_img, 110, 255, cv2.THRESH_BINARY)[1]

            #cv2.imwrite('imgs\\r.jpg', res_img)
            #cv2.imwrite('imgs\\b.jpg', bin_res_img)

            gray_nparr = []
            for i in range(8):
                for j in range(8):
                    gray_nparr.append((1 - (res_img[i][j])/255)*16)

            # bin_nparr = []
            # for i in range(8):
            #     for j in range(8):
            #         bin_nparr.append((1 - (bin_res_img[i][j])/255)*16)


            answer_g = class_names[np.argmax(predict(gray_nparr))]
            #answer_b = class_names[np.argmax(predict(bin_nparr))]

            answer = str(answer_g)
        except:
            answer = "Unknown error - function 'model_page' have information processing problem"

    form = ImgForm()
    return render(request, 'main\ModelPage.html', {'form': form, 'answer': answer})

def contacts_page(request):
    return render(request, 'main\ContactsPage.html')