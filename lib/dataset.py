import os
from tqdm import tqdm
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator


class SMSDataset():

  def __init__(self,train, test, batch, target_size, data_aug):

    """
      Args:
        - train: path to the train file
        - test: path to the test file
        - batch: batch size
        - target_size: dimension of images
        - data_aug: boolean that we use for data augmentation
    """

    self.train_file = train
    self.test_file = test
    self.batch_size = batch
    self.target_size = target_size
    self.data_aug = data_aug

    self.crop()
    self.train_generator, self.valid_generator, self.test_generator = self.create_dataset()

  def crop(self):
      """
        Cropping images to remove blue border
      """
      dirs = os.listdir(self.train_file)
      for directory in tqdm(dirs):
          fullpath = os.path.join(self.train_file,directory)
          for f in os.listdir(fullpath):
            image_file = os.path.join(fullpath, f )
            if os.path.isfile(image_file):
                im = Image.open(image_file)
                f, e = os.path.splitext(image_file)
                imCrop = im.crop((5, 5, 195, 195))
                imCrop.save(image_file, "jpeg", quality=100)

  def create_dataset(self):
      if self.data_aug:
          train_data = ImageDataGenerator(
              rescale = 1. / 255, # convert from uint8 to float32 in range 0,1
              validation_split=0.2,
              rotation_range= 5,
              brightness_range=[0.9,1.1],
              width_shift_range=0.1,
              height_shift_range=0.1,
          )
      else:
          train_data = ImageDataGenerator(
              rescale = 1. / 255, # convert from uint8 to float32 in range 0,1
              validation_split=0.2 
          )

      train_generator = train_data.flow_from_directory(
          directory=self.train_file,
          target_size=self.target_size,
          color_mode="rgb",
          batch_size=self.batch_size,
          class_mode="categorical",
          shuffle=True,
          subset='training'
      )
      valid_data = ImageDataGenerator( rescale = 1. / 255,  validation_split=0.2)

      valid_generator = valid_data.flow_from_directory(
          directory=self.train_file, # same directory as training data
          target_size=self.target_size,
          batch_size=self.batch_size,
          shuffle=False,
          class_mode='categorical',
          subset='validation') # set as validation data

      test_datagen = ImageDataGenerator(
          rescale = 1. / 255)

      test_generator = test_datagen.flow_from_directory(
          directory=self.test_file,
          target_size=self.target_size,
          color_mode="rgb",
          batch_size=self.batch_size,
          class_mode="categorical",
          shuffle=False
      )

      return train_generator, valid_generator, test_generator