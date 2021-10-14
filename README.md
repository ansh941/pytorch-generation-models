# Pytorch Generation Models
Generation models using Pytorch.<br>
I'm going to add new generation models every weekday.<br>
Starting with the basic models!

2021/10/14
After InfoGAN, it will be updated in form of notebook(.ipynb).
<br>

# Environments

Pytorch : 1.6.0<br>
Torchvision : 0.7.0<br>
torchsummary : 1.5.1<br>
numpy : 1.19.1<br>

# Training and Test
```
python train.py --logdir='temp' --gpu=0 --seed=0 --epochs=30
python test.py --logdir='temp' --gpu=0 --seed=0
```
## Next tasks

1. Add StyleGAN series<br>

2. Need New generation models.<br>
3. Korean to English<br>

4. New dataset

# Reference
* Papers for each model
* [Hwalsuklee/tensorflow-generative-model-collections](https://github.com/hwalsuklee/tensorflow-generative-model-collections)
