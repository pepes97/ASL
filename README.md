# Classification of American Sign Language

Our goal was to design a classifier that, given an image, is able to understand and recognize the American sign language (ASL) and output the right character represented by the sign.

## Table of contents

* [Development Setup](#Development-Setup)
* [Requirements](#Requirements)
  * [Download dataset](#Download-Dataset)
  * [Run](#Run)
* [Best Model](#Best-Model)
* [Demo](#Demo)
* [Slides](#Slides)
* [Credits](#Credits)

### Development Setup

<p align="center">
  <img width="600" height="350" src="https://user-images.githubusercontent.com/56698309/149672309-aaf338b5-d527-4ad1-aaad-058ec08e199d.png">
</p>

### Requirements

```
conda create --name asl python==3.7.11
conda activate asl
pip install -r requirements.txt
```
### Download Dataset

```
$ cd dataset
$ python dataset_download.py
```

### Run

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

### Best Model

![portfolio-1](https://user-images.githubusercontent.com/56698309/135748858-91e971e1-3152-4f08-bbe5-6f8577f8c661.png)


### Demo

https://user-images.githubusercontent.com/56698309/133144003-bb8d5f9e-00d3-4f08-a194-4a7f541f16c5.mp4

### Slides

<a href="https://docs.google.com/presentation/d/e/2PACX-1vRG6HQlMQ6BwrWWcnURHSyP_m0RtJJI3Ur5MYK46NToF9yDpjFdTnVL2KtfM-0x4jsCKBdxACpZiIcu/pub?start=false&loop=false&delayms=60000" target="_blank"> 
  <img src="https://user-images.githubusercontent.com/56698309/133143736-0ae49c74-02b9-459a-a359-69f3b0e09e71.png">
</a>

### Credits

* [Marco Pennese](https://github.com/MarcoPenne)

* [Sveva Pepe](https://github.com/pepes97)

* [Simone Tedeschi](https://github.com/sted97)
