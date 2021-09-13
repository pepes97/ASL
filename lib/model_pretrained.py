from tensorflow import keras


class ASL_Pretrained():
  def __init__(self,name, trainable, input_shape, params):

    """
      Args:
        - name: name of pretrained model
        - trainable: boolean used to indicate if layers of pretrained model are freezed or trained
    """
    self.name = name
    self.params = params
    self.trainable = trainable
    self.input_shape = input_shape
    self.model = self.build_model()
  
  def build_model(self):
    if self.name == "Xception":
      base_model = keras.applications.Xception(
          weights='imagenet',
          input_shape=self.input_shape,
          include_top=False
      )
    elif self.name == "ResNet":
      base_model = keras.applications.ResNet152V2(
          weights='imagenet',
          input_shape=self.input_shape,
          include_top=False
      )
    elif self.name == "VGG16":
      base_model = keras.applications.VGG16(
          weights='imagenet',
          input_shape=self.input_shape,
          include_top=False
      )
    else:
      base_model = keras.applications.VGG19(
          weights='imagenet',
          input_shape=self.input_shape,
          include_top=False
      )

    base_model.trainable = self.trainable

    inputs = keras.Input(shape=self.input_shape)

    # We freeze pretrained model weigths
    x = base_model(inputs, training=False)

    # Convert features of shape `base_model.output_shape[1:]` to vectors
    x = keras.layers.GlobalAveragePooling2D()(x)

    x = keras.layers.Dense(1024, activation='relu')(x)
    x = keras.layers.Dropout(self.params.drop_rate)(x)
    x = keras.layers.Dense(128, activation='relu')(x)
    outputs = keras.layers.Dense(self.params.num_classes, activation="softmax")(x)
    asl_model = keras.Model(inputs, outputs, name=self.params.model_name)

    asl_model.compile(loss=self.params.loss_fn, optimizer=self.params.optimizer, metrics=self.params.metrics)

    return asl_model