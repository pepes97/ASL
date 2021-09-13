from tensorflow.keras import optimizers


class HParams():
  def __init__(self, input_shape, num_classes, lr, model_name):
    self.shape = input_shape
    self.num_classes = num_classes
    self.lr = lr
    self.optimizer = optimizers.Adam(self.lr)
    self.loss_fn = 'categorical_crossentropy'
    self.metrics = ['accuracy']
    self.drop_rate = 0.5
    self.n_res_net_blocks = 3
    self.n_paths = 4
    self.model_name = model_name