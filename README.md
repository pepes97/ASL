# Classification of American Sign Language

## Requirements

```
conda create --name asl python==3.7.11
conda activate asl
pip install -r requirements.txt
```

## Run

```
python main.py --data-aug  DATA-AUG --target-size TARGET-SIZE --epochs EPOCHS --batch-size BATCH-SIZE --lr LR --name-model NAME-MODEL --fine-tune FINE-TUNE --only-test ONLY-TEST
```

where:
- `DATA-AUG`: data aumentation, default is True
- `TARGET-SIZE`: target size of the image, default is (100, 100)
- `EPOCHS`: number of epochs, default is 50
- `BATCH-SIZE`: batch size, default is 64
- `LR`: learning rate, default is 0.0001
- `NAME-MODEL`: name of model, defaul is VGG19
- `FINE-TUNE`: fine tuning, default is False
- `ONLY-TEST`: only test, default is False

for example, for training

```
python main.py
```

for testing:

```
python main.py --only-test True
```

## Demo


## Credits

* [Marco Pennese](https://github.com/MarcoPenne)

* [Sveva Pepe](https://github.com/pepes97)

* [Simone Tedeschi](https://github.com/sted97)