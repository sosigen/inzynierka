1. Znajdź którego datasetu użyć do treningu. Kandydaci:
  - FER+ - pobrane
  - JAFFE - tylko japońskie twarze? chyba sobie podaruje
  - CK+ - pobrane
  - Aff-Wild2 - https://ibug.doc.ic.ac.uk/resources/aff-wild2/, trzeba wypełnić dokument ręcznie. Weź walnij maila jutro i wydrukuj to
  - RAF-DB - http://www.whdeng.cn/raf/model1.html#dataset, wysłałeś maila z umka, pobrana wersja z 12k zdjec
  - MMI - https://mmifacedb.eu/accounts/register/, musisz zrobić konto i ponownie wydrukować i podpisać EULA https://mmifacedb.eu/accounts/register/
  - AffectNet - http://mohammadmahoor.com/affectnet/, nie udostępniane studentom. Tylko profesorom -> STUDENTS: Please ask your academic advisor/supervisor to request access to AffectNet.

  - https://www.kaggle.com/datasets/mejiaescobarchris/fer-stable-diffusion-dataset -> mozesz se to sprawdzic, to jest wszystko AI generated
2. Zewaluuj czy są potrzebne jakieś dodatkowe augmentacje na nim

3. CNN -> 
 - EfficientNet: są wersje B0 - B7 (najmniejsza do najwiekszej)
 - InceptionNet: V1 - V3, V3 rekomendowane
 - ResNet: jest 34, 50 i 101. To jest ilosc warstw. Im wiecej tym lepiej
 - VGGNet: jest 16 i 19, ta sama bajka. Wiekszy numer wiekszy model
 - SqueezeNet-FER: jeden model
 Mozna pobrac gotowy model i dotrenowac. Głównie chodzi o to że nie trzeba tam jakoś mega dużo zmieniać, wystarczy dopasować do datasetu i dotrenować na swoich danych, mozna zaimportować gotowe moduły 

No transfer, raw CNN -> PAYWALL https://link.springer.com/article/10.1007/s40031-021-00681-8

Specific examples -> https://github.com/liwei109/Auto-FERNet (uses attention), https://github.com/omarsayed7/Deep-Emotion, https://github.com/co60ca/EmotionNet2

jakbys chcial dataset nowy tu masz za darmola -> https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset/data


https://medium.com/@aniketthomas27/efficientnet-implementation-from-scratch-in-pytorch-a-step-by-step-guide-a7bb96f2bdaa - wytlumaczone coz to jet efficient net, taki artykulik do czytania

# Główne myśli

Coding from scratch w przypadku CNN już się raczej nie robi, raczej się używa pre-trained modeli. Daj jakiś jeden dla przykładu może ale no raczej nie.

## EfficientNet - transfer learning

#### GitHub #1 Chorko
https://github.com/Chorko/Emotion-recognition-using-efficientnet: 
```python
# Load EfficientNetV2B2 and apply to grayscale input
efficient_net_model = EfficientNetV2B2(weights='imagenet', include_top=False, input_shape=input_shape)
x = efficient_net_model(x)

# Global Average Pooling
x = GlobalAveragePooling2D()(x)

# Add Dense layers for classification
x = Dense(512, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.25)(x)

x = Dense(256, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.25)(x)

x = Dense(128, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.25)(x)

x = Dense(64, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.25)(x)
```
https://github.com/Chorko/Emotion-recognition-using-efficientnet/blob/master/emotion_classification_efficientnet.py

### GitHub #2 An-Sunghyun
https://github.dev/An-Sunghyun/Emotion_Recognition_DNN/blob/main/Final%20Model/EfficientNetB0_pytorch_Final.ipynb

Surowy efficient net, Typ ogólnie dużo modeli zrobił
![](img/20241021211036.png)

# GitHub #3 
https://github.com/av-savchenko/face-emotion-recognition

# GitHub #4
https://github.dev/amrkld/Facial_Expression_Recognizer_Fine-Tuning/tree/master/EfficientNetB2%20Model%20Code
```python
x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense(128, activation = 'relu')(x)
x = tf.keras.layers.Dense(7, activation='softmax')(x)
```

### Paper #1 
https://www.researchgate.net/publication/368191544_Efficient_Net-XGBoost_An_Implementation_for_Facial_Emotion_Recognition_Using_Transfer_Learning - opis efektywnego classifiera, nie ma tu modelu per se, jest ten classifier

Jedyna implementacja jest w czytaniu dokumentow -> https://github.com/IndoML22/EfficientNet-XGBoost-Model.ipynb


## Moje 

Najpierw zrobiłem prowizoryczny model ktory miał ~50% trafnosci, wytrenował się w 15 minut. Usprawniłem nastepujace rzeczy:
- przerobilem obrazki zeby byly 224 x 224 i RGB, efficientNet sie uczyl na takich i to podobno pomaga
- dodałem early stopping, bo widziałem ze w pewnym momencie accuracy spada podczas treningu
- unfreezowałem 20 warstw z efficient neta zeby bardziej dopasowac model
- przez powiekszenie obrazkow musialem dodać data_generator, cały zbior nie miescil sie w pamieci

po dodaniu tych rzeczy czas treningu skoczył z 15 minut do kilku godzin

- nastepne w planach jest dodanie class_weights - FER2013 jest niezbalansowany, np klasa disgust ma bardzo malo danych
- chcialbym tez dodac warstwy BatchNormalization pomiedzy moje warstwy z 1szego prototypu zeby obnizyc overfitting
- moze druga warstaw dense? 256 zamiast 128
- pisza ze kernel_regularizer=l2(0.001) 'discourages large weights, improving generalization'
- moze lepszy klasyfikator niz softmax? W paper #1 jest jakis lepszy

training started
Epoch 1/50
897/897 ━━━━━━━━━━━━━━━━━━━━ 672s 738ms/step - accuracy: 0.2248 - loss: 1.8579 - precision: 0.2519 - recall: 0.0011
Epoch 2/50
897/897 ━━━━━━━━━━━━━━━━━━━━ 781s 871ms/step - accuracy: 0.2494 - loss: 1.8206 - precision: 0.0000e+00 - recall: 0.0000e+00
Epoch 35/50
897/897 ━━━━━━━━━━━━━━━━━━━━ 718s 801ms/step - accuracy: 0.2461 - loss: 1.8132 - precision: 0.0000e+00 - recall: 0.0000e+00
Epoch 36/50
897/897 ━━━━━━━━━━━━━━━━━━━━ 759s 846ms/step - accuracy: 0.2472 - loss: 1.8123 - precision: 0.0000e+00 - recall: 0.0000e+00
Epoch 37/50
306/897 ━━━━━━━━━━━━━━━━━━━━ 8:03 819ms/step - accuracy: 0.2452 - loss: 1.8115 - precision: 0.0000e+00 - recall: 0.0000e+00

niestety slabe wyniki, moze wywale resize?


to byla ta linijka w ImageDataGenerator rescale=1./255. Dodam normalizacje w dobrym miejscu

Po naprawieniu tego model daje ~51%, zatrzymał sie po 36 epochach
Epoch 36/50
897/897 ━━━━━━━━━━━━━━━━━━━━ 136s 152ms/step - accuracy: 0.5166 - loss: 1.2744 - precision: 0.7145 - recall: 0.3049 - val_accuracy: 0.4945 - val_loss: 1.3462 - val_precision: 0.6579 - val_recall: 0.3254

Zmniejszam learning rate do Adam(learning_rate=1e-5)
, intensywnosc augmentacji 0.1 -> 0.2, 10 -> 20,
dodaje wagi do klas, -> jednak nie, nie mozna uzywac wag z generatorem, trzeba to zrobic naokolo przez custom loss fn
ReduceLROnPlateau
zamiast B0 uzyje B3
i odfreezowałem 20 warstw zamiast 10


próba 3
Epoch 26/50
897/897 ━━━━━━━━━━━━━━━━━━━━ 0s 170ms/step - accuracy: 0.3153 - loss: 2.4848 - precision: 0.4491 - recall: 0.1269

próba 4

odfreezuje 40 warstw, zmniejsze z powrotem augmentation i wywale BatchNormalization

Epoch 20/50
897/897 ━━━━━━━━━━━━━━━━━━━━ 0s 176ms/step - accuracy: 0.4078 - loss: 2.0302 - precision: 0.6775 - recall: 0.1560


## EfficientNet v2 - mało co o tym jest a niby dużo lepsze!!!!!!!!!!!

^^^^^^^^^^^


# InceptionNet
Trzeba znaleźć czemu V3 jest rekomendowane i znaleźć coś z V3 rzeczywiście




# TODO

I need 3 blocks of code to switch betwen
- block for reading and preprocessing data
- block for training
- block for validation & charts

okazuje sie, ze przy transfer learningu fine-tuning sie robi na kilku egzekucjach model.fit(). Najpiew puszczam wszystkie zamrozone i z kazda egzekucja odmrazam ich wiecej i zwalniam LR

B5 i 50 odmrozonych warstw okazal sie najgorszy (jak  narazie)


podejscie z gradually unfreezing totalnie NIE DZIAŁA!!! Moge osiagnac 50% bez tego, to mi daje jakieś 24%

```for iteration in range(total_unfreeze_iterations):
    # Calculate the range of layers to unfreeze for this iteration
    start_layer = max(len(base_model.layers) - (iteration + 1) * unfreeze_steps, 0)
    end_layer = len(base_model.layers)

    # Unfreeze the specified range of layers
    for layer in base_model.layers[start_layer:end_layer]:
        layer.trainable = True

    # Compile with slightly reduced learning rate per stage
    model.compile(optimizer=Adam(learning_rate=initial_learning_rate / (iteration + 1)),
                  loss='categorical_crossentropy', metrics=['accuracy', 'Precision', 'Recall'])

    # Training for the current stage
    print(f"\n--- Training stage {iteration + 1}/{total_unfreeze_iterations} ---")
    print(f"Unfreezing layers {start_layer} to {end_layer}")

    history = model.fit(
        train_generator_augmented,
        steps_per_epoch=len(train_data_raw) // BATCH_SIZE,
        epochs=max_epochs_per_stage,
        callbacks=[early_stopping, reduce_lr],
        validation_data=augmented_data_generator(data_generator(test_data_raw), datagen),
        validation_steps=len(test_data_raw) // BATCH_SIZE,
    )

    # Check early stopping to break out of the loop if no improvement
    if early_stopping.stopped_epoch > 0:
        print("Early stopping triggered.")
        break
```


67% accuracy!!!!!!!
```
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator


max_epochs_per_stage = 30
BATCH_SIZE = 64

# Load the EfficientNetB5 model pre-trained on ImageNet, excluding the top layers
base_model = EfficientNetB3(weights='imagenet', include_top=False, input_shape=(48, 48, 3))

# Add custom top layers with higher dropout rates
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu', kernel_regularizer=l2(0.001))(x)
x = Dropout(0.5)(x)  # Increased dropout
x = Dense(256, activation='relu', kernel_regularizer=l2(0.001))(x)
x = Dropout(0.6)(x)  # Increased dropout
x = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(x)
x = Dropout(0.6)(x)
x = Dense(64, activation='relu', kernel_regularizer=l2(0.001))(x)
x = Dropout(0.3)(x) 
predictions = Dense(len(emotion_map), activation='softmax')(x)

# Create the final model
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze all layers initially
for layer in base_model.layers[-60:]:
  layer.trainable = True

# Compile with initial settings
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', 'Precision', 'Recall'])
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)

# Create an ImageDataGenerator for data augmentation suited for facial expression data
datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.15,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Function to augment data
def augmented_data_generator(generator, datagen, batch_size=32):
    while True:
        batch_features, batch_labels = next(generator)
        augmented_data = datagen.flow(batch_features, batch_labels, batch_size=batch_size, shuffle=False)
        yield next(augmented_data)

train_generator_augmented = augmented_data_generator(data_generator(train_data_raw), datagen, BATCH_SIZE)


model.compile(optimizer='adam',
              loss='categorical_crossentropy', metrics=['accuracy', 'Precision', 'Recall'])

history = model.fit(
    train_generator_augmented,
    steps_per_epoch=len(train_data_raw) // BATCH_SIZE,
    epochs=max_epochs_per_stage,
    callbacks=[early_stopping, reduce_lr],
    validation_data=augmented_data_generator(data_generator(test_data_raw), datagen),
    validation_steps=len(test_data_raw) // BATCH_SIZE,
)


# Save the final model after modular unfreezing
model.save('efficientnetb5_finetuned_model.h5')
print("Model fine-tuning complete and saved as 'efficientnetb5_finetuned_model.h5'.")

```







