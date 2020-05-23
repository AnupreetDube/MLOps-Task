from keras.datasets import mnist
(xtrain,ytrain) , (xtest,ytest) = mnist.load_data()

xtrain=xtrain.reshape(60000,28,28,1)
xtest=xtest.reshape(10000,28,28,1)


from keras.utils import to_categorical as tc
ytrain=tc(ytrain)
ytest=tc(ytest)

from keras.models import Sequential as seq
from keras.layers import Dense, Conv2D, Flatten
model = seq()
model.add(Conv2D(2, kernel_size=3, activation="relu", input_shape=(28,28,1)))

i=1
n=4
for i in range(i):
	model.add(Conv2D(filters=n, kernel_size=3, activation="relu"))
	n=n*2

model.add(Flatten())
model.add(Dense(10, activation="softmax"))



model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(xtrain, ytrain,epochs=1)

pred1= model.evaluate