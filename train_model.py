import tensorflow as tf
from tensorflow.keras import layers

#coverting to one size
img_size=(160,160)
model="cat_dog_classify.keras"
dataset_train="dataset/train"
dataset_test="dataset/validation"
train_ds=tf.keras.utils.image_dataset_from_directory(dataset_train,image_size=img_size,label_mode='binary')
#if many category->categorical
testing_ds=tf.keras.utils.image_dataset_from_directory(dataset_test,image_size=img_size,label_mode='binary')

print("training dataset length:",len(train_ds))
print("testing dataset length:",len(testing_ds))

print("classes:",train_ds.class_names)

base=tf.keras.applications.MobileNetV2(input_shape=(160,160,3),include_top=False,weights='imagenet')
base.trainable=False
model=tf.keras.Sequential([
    layers.Rescaling(1./127.5,input_shape=(160,160,3)),
    base,
    layers.GlobalAveragePooling2D(),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(train_ds,validation_data=testing_ds,epochs=5)
model.save("cat_dog_classify.keras")
print("model saved successfully")