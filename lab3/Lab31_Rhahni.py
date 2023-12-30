# Importation des bibliothèques nécessaires
import tensorflow as tf
import matplotlib.pyplot as plt
import os




# Définition des paramètres
img_height, img_width = 32, 32
batch_size = 20

# Étape 1: Chargement du jeu de données

# Définition des chemins vers les répertoires d'entraînement, de validation et de test
train_dir = "datasets/sport/train"
val_dir = "datasets/sport/validation"
test_dir = "datasets/sport/test"

# Chargement des jeux de données d'entraînement, de validation et de test
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=(img_height, img_width),
    batch_size=batch_size
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    image_size=(img_height, img_width),
    batch_size=batch_size
)


test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# Affichage de quelques images du jeu de données d'entraînement
class_names = train_ds.class_names
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
plt.show()
# Affiche les noms des fichiers dans le répertoire d'entraînement
print(os.listdir(train_dir))

# Étape 2: Construction du modèle

model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255),
    tf.keras.layers.Conv2D(32, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(3)  # 3 classes correspondant à vos logos de ligues
])

# Compilation du modèle
model.compile(
    optimizer="adam",
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# Étape 3: Entraînement du modèle

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=70
)

# Étape 4: Évaluation sur le jeu de données de test

model.evaluate(test_ds)
# Sauvegarde du modèle TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("model.tflite", 'wb') as f:
    f.write(tflite_model)



