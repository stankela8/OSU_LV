
import numpy as np
import matplotlib.pyplot as plt
'''
x=np.array([1,2,3,3,1],float)
y=np.array([1,2,2,1,1],float)

plt.plot(x,y,'b',linewidth=1,marker=".",markersize=5)
plt.axis([0,4,0,4])
plt . xlabel ("x")
plt . ylabel ("vrijednosti funkcije")
plt . title ("slika")
plt.show()
'''

'''
data = np.loadtxt("data.csv", delimiter=',', skiprows=1)
 
rows, cols = np.shape(data)
print(str(rows),"people")
 
height = data[:,1]
weight = data[:,2]
 
plt.scatter(height, weight, s=1)
 
plt.xlabel("Height [cm]")
plt.ylabel("Weight [kg]")
plt.title("Height to weight ratio")
plt.show()
 
height50 = height[::50]
weight50 = weight[::50]
 
plt.scatter(height50, weight50,s=5)
plt.xlabel("Height [cm]")
plt.ylabel("Weight [kg]")
plt.title("every 50th person")
plt.show()
 
print("Min height: ", str(np.min(height)))
print("Max height: ", str(np.max(height)))
print("Average height: ", str(np.mean(height)))
 
men=data[np.where(data[:,0]==1)]
women=data[np.where(data[:,0]==0)]
 
print("Min male height: ", str(np.min(men[:,1])))
print("Max male height: ", str(np.max(men[:,1])))
print("Average male height: ", str(np.mean(men[:,1])))
 
print("Min female height: ", str(np.min(women[:,1])))
print("Max female height: ", str(np.max(women[:,1])))
print("Average female height: ", str(np.mean(women[:,1])))
'''
#Treci

img = plt.imread("road.jpg")
factor=100
brImg=np.clip(img.astype(np.float32)+factor,0,255).astype(np.uint8)
plt.figure()
plt.imshow(brImg,cmap="gray")
plt.show()

img = plt.imread("road.jpg")
height,width,_=img.shape
start_width= width // 2
end_witdh=width
cetvrtina=img[:,start_width:end_witdh]
plt.figure()
plt.imshow(cetvrtina,cmap="gray")
plt.show()

img = plt.imread("road.jpg")
rotirana=np.rot90(img,k=3)
plt.figure()
plt.imshow(rotirana,cmap="gray")
plt.show()


img = plt.imread("road.jpg")
zrcaljena=np.fliplr(img)
plt.figure()
plt.imshow(zrcaljena,cmap="gray")
plt.show()

#Cetvrti
black=np.zeros((50,50),dtype=np.uint8)
white=np.ones((50,50),dtype=np.uint8)

image=np.vstack((np.hstack((black,white)),np.hstack((white,black))))
plt.figure()
plt.imshow(image)
plt.show()
