from tensorflow import keras

train_ds = keras.utils.image_dataset_from_directory(
    directory='/home/vj/Dropbox/temp/train',
    labels='inferred',
    label_mode='categorical',
    batch_size=8,
    image_size=(256, 256))
validation_ds = keras.utils.image_dataset_from_directory(
    directory='/home/vj/Dropbox/temp/val',
    labels='inferred',
    label_mode='categorical',
    batch_size=8,
    image_size=(256, 256))


model = keras.applications.Xception(
    weights=None, input_shape=(256, 256, 3), classes=2)
model.compile(optimizer='rmsprop', loss='binary_crossentropy')
model.fit(train_ds, epochs=10, validation_data=validation_ds)
