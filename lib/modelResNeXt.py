from tensorflow.keras.layers import Conv2D, BatchNormalization, Add, Activation, MaxPooling2D, Dropout, Flatten, Dense, AveragePooling2D
import tensorflow as tf
from tensorflow.keras import  Input


def res_next_block(input_data, filters, conv_size, n_paths):

    result_paths = []
    for i in range(n_paths):
      x = Conv2D(filters, conv_size, activation='relu', padding='same')(input_data)
      x = BatchNormalization()(x)
      x = Conv2D(filters, conv_size, activation=None, padding='same')(x)
      result_paths.append(BatchNormalization()(x))
    
    x = Add()(result_paths)
    x = Add()([x, input_data])
    x = Activation('relu')(x)   
    return x

def ASL_ModelResNeXt(params):

    inputs = Input(params.shape)
    x = Conv2D(32, (7,7), activation='relu')(inputs)
    x = MaxPooling2D((2, 2))(x)

    for l in range(params.n_res_net_blocks):
      x = res_next_block(x, 32, (3, 3), params.n_paths)

    x = Flatten()(x)
    x = Dropout(params.drop_rate)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(params.drop_rate)(x)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(params.num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs, name = params.model_name)
    model.compile(loss=params.loss_fn, optimizer=params.optimizer, metrics=params.metrics) # configure model for training

    return model