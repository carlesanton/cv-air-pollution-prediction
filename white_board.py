from PIL import Image
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

'''
path='images/'
quantes=10

a=os.listdir(path)
immm = mpimg.imread(path + a[2])
print(immm.shape)
array=np.zeros((immm.shape[0],immm.shape[1],immm.shape[2],quantes))

for i in range(quantes):
	array[:,:,0,i]=immm[:,:,0]/3+immm[:,:,1]/3+immm[:,:,2]/3
	array[:,:,1,i]=immm[:,:,0]/3+immm[:,:,1]/3+immm[:,:,2]/3
	array[:,:,2,i]=immm[:,:,0]/3+immm[:,:,1]/3+immm[:,:,2]/3
	
	if (i+1)%3 == 1:
		array[:,:,i]=immm[:,:,0]
	if (i+1)%3 == 2:
		array[:,:,i]=immm[:,:,1]
	if (i+1)%3 == 0:
		array[:,:,i]=immm[:,:,2]
	

for x in range(quantes):
	plt.subplot(5,2,x+1)
	plt.imshow(np.array(array[:,:,:,x]))


	
plt.show()


'''

for img in glob.glob("Data_HD/Users/Carels/TFG/swimcat_dataset/E-veil_train/images"):
    print(img)
    im=mpimg.imread("swimcat_dataset/E-veil_train/images/" + img)
    plt.imshow(im)
    plt.show()