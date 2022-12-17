import tensorflow as tf

validation_split = 0.2
batch_size = 1
img_size = (224, 224) # Default resolution for mobilenet

# Set dataset directory
dataset_dir = 'yumi_dataset'

# Setup Dataset
train_dataset = tf.keras.utils.image_dataset_from_directory(
                                                            dataset_dir,
                                                            shuffle=True,
                                                            validation_split=validation_split,
                                                            subset="training",
                                                            seed=42,
                                                            image_size=img_size,
                                                            batch_size=batch_size
                                                            )

validation_dataset = tf.keras.utils.image_dataset_from_directory(
                                                                dataset_dir,
                                                                shuffle=True,
                                                                validation_split=validation_split,
                                                                subset="validation",
                                                                seed=42,
                                                                image_size=img_size,
                                                                batch_size=batch_size
                                                                )

class_names = train_dataset.class_names # Save class name for inference

AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)

# Set Augmentation
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal_and_vertical'),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomBrightness(0.2),
    tf.keras.layers.RandomContrast(0.2),
    tf.keras.layers.RandomZoom(0.2),
    tf.keras.layers.RandomHeight(0.2),
    tf.keras.layers.RandomWidth(0.2)
])

# Create Model
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

# Create the base model from the pre-trained model MobileNet V2
img_shape = img_size + (3,)
base_model = tf.keras.applications.MobileNetV2(input_shape=img_shape,
                                                include_top=False,
                                                weights='imagenet')

image_batch, label_batch = next(iter(train_dataset))
feature_batch = base_model(image_batch)

base_model.trainable = False

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(feature_batch)

prediction_layer = tf.keras.layers.Dense(1) # Binary classification
prediction_batch = prediction_layer(feature_batch_average)

inputs = tf.keras.Input(shape=(img_size[0], img_size[1], 3))
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)
x = global_average_layer(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = prediction_layer(x)
outputs = tf.keras.layers.Activation('sigmoid')(x)
model = tf.keras.Model(inputs, outputs)

# Set training parameters
initial_epochs = 50
base_learning_rate = 0.0001

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])

# Train model
history = model.fit(train_dataset,
                    epochs=initial_epochs,
                    validation_data=validation_dataset)

# Fine-tuning
base_model.trainable = True

# Fine-tune from this layer onwards
fine_tune_at = 100

# Freeze all the layers before the `fine_tune_at` layer
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer = tf.keras.optimizers.RMSprop(learning_rate=base_learning_rate/10),
              metrics=['accuracy'])

fine_tune_epochs = 50
total_epochs =  initial_epochs + fine_tune_epochs

# Fine-tune model
history_fine = model.fit(train_dataset,
                         epochs=total_epochs,
                         initial_epoch=history.epoch[-1],
                         validation_data=validation_dataset)

# Save Model in SavedModel format
model.save('saved_model')

# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model('saved_model')
tflite_model = converter.convert()

# Save the model.
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
