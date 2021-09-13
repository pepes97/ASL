
from tensorflow.keras import layers, models


def ASL_ModelCNN(params):
    model = models.Sequential(name=params.model_name)
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=params.shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dropout(params.drop_rate))
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dropout(params.drop_rate))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(params.num_classes, activation="softmax"))
    model.compile(loss=params.loss_fn, optimizer=params.optimizer, metrics=params.metrics) # configure model for training

    return model