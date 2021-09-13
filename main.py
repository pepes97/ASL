import os
import tensorflow as tf
from tensorflow import keras
import argparse

from lib.dataset import SMSDataset
from lib.modelCNN import ASL_ModelCNN
from lib.params import HParams
from lib.modelResNeXt import ASL_ModelResNeXt
from lib.model_pretrained import ASL_Pretrained
from datetime import datetime
from lib.utils import show_batch, show_predicted_batch


def main(data_aug, target_size, batch_size, name_model, learning_rate, fine_tune, epochs, only_test):

    device_name = tf.test.gpu_device_name()
    if device_name != '/device:GPU:0':
        raise SystemError('GPU device not found')
    print('Found GPU at: {}'.format(device_name))

    train_dir = "./dataset/asl_alphabet_train_kaggle"
    test_dir = "./dataset/asl_alphabet_test_real_world_+_kaggle"

    print(f"\033[1mTrain dir \033[0m: {train_dir} \033[0m")
    print(f"\033[1mTest dir \033[0m: {test_dir} \033[0m")

    dataset = SMSDataset(train_dir, test_dir, batch_size, target_size, data_aug)
    train_generator, valid_generator, test_generator = dataset.train_generator, dataset.valid_generator, dataset.test_generator

    input_shape = train_generator.image_shape
    num_classes = train_generator.num_classes
    classnames = [k for k,v in train_generator.class_indices.items()]

    print("\nClasses:\n%r" %classnames)
    print(f"\033[1mShow Batch images \033[0m")
    image_batch, label_batch = next(train_generator)
    show_batch(image_batch, label_batch, classnames)

    params = HParams(input_shape, num_classes, learning_rate, name_model)
    print(f"\033[1mModel {name_model}\033[0m")
    if name_model == "CNN":
        asl_model = ASL_ModelCNN(params)
    elif name_model == "ResNeXt":
        asl_model = ASL_ModelResNeXt(params)
    else:
        asl_model = ASL_Pretrained(name_model, False, input_shape, params).model

    nome_prova = f"{params.model_name}_input{input_shape}_lr{params.lr}_drop{params.drop_rate}_resbloks{params.n_res_net_blocks}_paths{params.n_paths}_{data_aug}(all)"  
    model_path = "/models/checkpoints/"+nome_prova+"/"
    if not only_test:
        
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        logs_path = "models/tensorboard_logs/"+nome_prova
        if not os.path.exists(logs_path):
            os.makedirs(logs_path)

        my_callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint(filepath=model_path+'model.h5', save_best_only=True),
            tf.keras.callbacks.TensorBoard(log_dir=logs_path),
        ]

        print(f"\033[1mTraining...\033[0m")
        history = asl_model.fit(train_generator, epochs=epochs, verbose=1, validation_data=valid_generator, callbacks=my_callbacks)

        if fine_tune:
            print(f"\033[1mFine Tuning...\033[0m")
            params = HParams(input_shape, num_classes, learning_rate*10^-1, name_model)
            asl_model = ASL_Pretrained(name_model, True, input_shape, params).model
            history = asl_model.fit(train_generator, epochs=epochs, verbose=1, validation_data=valid_generator, callbacks=my_callbacks)


    asl_model = keras.models.load_model(model_path+'model.h5')
    asl_model.compile(loss=params.loss_fn, optimizer=params.optimizer, metrics=params.metrics)
        
    print(f"\033[1mTesting...\033[0m")
    loss, acc = asl_model.evaluate(test_generator, verbose=1)
    print(f"\033[1mLOSS: {loss}\033[0m")
    print(f"\033[1mACCYRACY {acc}\033[0m\n")

    print(f"\033[1mShow predicted batch\033[0m")
    image_batch, label_batch = next(test_generator)
    show_predicted_batch(image_batch, label_batch, asl_model, classnames)
    print(f"\033[Done\033[0m")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--data-aug', type=bool, default=True, help='flag for data augmentation')
    parser.add_argument('--target-size', type=tuple, default=(100,100), help='target size for images')
    parser.add_argument('--batch-size', type=int, default=64, help='batch size')   
    parser.add_argument('--name-model', type=str, default="VGG19", help='name of the model')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--fine-tune', type=bool, default=False, help='Fine Tuning')
    parser.add_argument('--epochs', type=int, default=50, help='Epochs')
    parser.add_argument('--only-test', type=bool, default=False, help='Only Test')


    args = parser.parse_args()
    data_aug = args.data_aug
    target_size = args.target_size
    batch_size = args.batch_size
    name_model = args.name_model
    learning_rate = args.lr
    fine_tune = args.fine_tune
    epochs = args.epochs
    only_test = args.only_test

    main(data_aug, target_size, batch_size, name_model, learning_rate, fine_tune, epochs, only_test)