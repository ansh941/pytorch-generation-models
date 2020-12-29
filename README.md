# Pytorch Generation Models
Generation models using Pytorch.<br>
I'm going to add new generation models every weekday.<br>
Starting with the basic models!
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

1. Should add convolution layers in CVAE.<br>
Current CVAE based on DNN.<br>
So, I should change CVAE to convolutional CVAE.<br>

2. Need New generation models.<br>
  - AAE<br>
  - WGAN<br>

3. Korean to English<br>

4. New dataset
