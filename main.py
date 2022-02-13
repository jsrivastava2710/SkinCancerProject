# check Pillow version number
import PIL
from matplotlib import pyplot as plt
import cv2
from cv2 import imshow
print(cv2.__version__)
import pandas as pd
from tqdm import tqdm
import cv2
import csv
import os
import np

# load and show an image with Pillow
from PIL import Image, ImageOps, ImageFilter

# load the image

# summarize some details about the image
#print(image.format)
#print(image.mode)
#print(image.size)
# show the image


IMG_WIDTH = 100
IMG_HEIGHT = 75

X_images = []
X_gray = []
y_labels = []



rows = []

metadata = '/Users/janvisrivastava/PycharmProjects/SkinCancer/metadata (1).csv'
with open(metadata, 'r') as file:
    csvreader = csv.reader(file)
    header = next(csvreader)
    for row in csvreader:
        rows.append(row)

X_images = [i[1] for i in rows]
y_labels = [i[2] for i in rows]

print(X_images[5])


#TO COUNT NUMBER OF IMAGES OF EACH TYPE
objects = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
y_pos = np.arange(len(objects))
occurences = []

for obj in objects:
    i = 0
    for item in y_labels:
        if (item == obj):
            i = i + 1
    occurences.append(i)

plt.bar(y_pos, occurences, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Samples')
plt.title('Distribution of Classes Within Dataset')

plt.show()

#for item in X_images:
 #   dir_path = os.path.dirname(os.path.realpath(__file__))
  #  image_to_gray = Image.open(dir_path+'/images/HAM10000_images_all/' + item + '.jpg')
    #image_to_gray.show()
   # image_to_gray = ImageOps.grayscale(image_to_gray)
    #image_to_gray.show()
    #X_gray.append(image_to_gray)

X_images_new = []
y_labels_new = []

objects = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

akiec_X = []
akiec_y = []

bcc_X = []
bcc_y = []

bkl_X = []
bkl_y = []

df_X = []
df_y = []

mel_X = []
mel_y = []

nv_X = []
nv_y = []

vasc_X = []
vasc_y = []

for i in range(len(y_labels)):
    y_label_current = y_labels[i]
    if (y_label_current == 'akiec'):
        akiec_X.append(X_images[i])
        akiec_y.append(y_labels[i])
    elif (y_label_current == 'bcc'):
        bcc_X.append(X_images[i])
        bcc_y.append(y_labels[i])
    elif (y_label_current == 'bkl'):
        bkl_X.append(X_images[i])
        bkl_y.append(y_labels[i])
    elif (y_label_current == 'df'):
        df_X.append(X_images[i])
        df_y.append(y_labels[i])
    elif (y_label_current == 'mel'):
        mel_X.append(X_images[i])
        mel_y.append(y_labels[i])
    elif (y_label_current == 'nv'):
        nv_X.append(X_images[i])
        nv_y.append(y_labels[i])
    elif (y_label_current == 'vasc'):
        vasc_X.append(X_images[i])
        vasc_y.append(y_labels[i])

del akiec_X[115:]
del akiec_y[115:]

del bcc_X[115:]
del bcc_y[115:]

del bkl_X[115:]
del bkl_y[115:]

del df_X[115:]
del df_y[115:]

del mel_X[115:]
del mel_y[115:]

del nv_X[115:]
del nv_y[115:]

del vasc_X[115:]
del vasc_y[115:]

X_images_new = akiec_X + bcc_X + bkl_X + df_X + mel_X + nv_X + vasc_X
y_labels_new = akiec_y + bcc_y + bkl_y + df_y + mel_y + nv_y + vasc_y

occurences_new = []
for obj in objects:
    i = 0
    for item in y_labels_new:
        if (item == obj):
            i = i + 1
    occurences_new.append(i)
print(occurences_new)

plt.bar(y_pos, occurences_new, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Samples')
plt.title('Distribution of Classes Within Dataset')

plt.show()

for i in X_images_new:
    img = Image.open('/Users/janvisrivastava/PycharmProjects/SkinCancerKNN/images/' + i + '.jpg')
    imgGray = img.convert('L')
    imgGray.save('/Users/janvisrivastava/PycharmProjects/SkinCancerKNN/images/' + i + '.jpg')

#print(X_images_new)

for x in range(len(X_images_new)):
    newimg = Image.open('/Users/janvisrivastava/PycharmProjects/SkinCancerKNN/images/' + X_images_new[x] + '.jpg')
    blurImage = newimg.filter(ImageFilter.BoxBlur(5))
    blurImage.save('images/' + X_images_new[x] + 'blur.jpg')
    X_images_new.append(X_images_new[x] + 'blur')
    y_labels_new.append(y_labels_new[x])



