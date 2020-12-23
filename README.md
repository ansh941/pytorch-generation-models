# Pytorch Generation Models
Generation models using Pytorch.<br>
I'm going to add new generation models every day.<br>
Starting with the basic models!
<br>

## VAE(Variational AutoEncoder)

### English
VAE is derived model of AutoEncoder and consists of Encoder and Decoder.<br>
AutoEncoder does Encoding and Decoding of inputs literally,<br>
VAE give diversity to reconstruction function in Decoder by adding little noise to the value made by Encoder. <br>

VAE is explicit generation model, it suppose the distribution of the dataset and then train for fitting that.<br>
In VAE, it suppose that normal distributions.<br>

At that point is difference from AutoEncoder. Encoder in VAE fitted Multinomial distribution that sythesized many Normal distributions.<br>
Normal distribution has 2 parameters, mean and standard deviation(std). So, Encoder shows that fit input data at normal distribution using means and stds.<br>
Also, Encoder give diversity to trained distribution using noise. This process is called "Reparametrize".<br>
(Numerical Experssion)<br>
<br>

The advantage of adding noise is that Decoder can be decoding the data clearly though little different from original input data.<br>
Decoder is reconstructing the encoding values like original data.<br>

Then, the loss function use kl-divergence for fitting input data to multinomial distribution composed of normal distributions,<br>
and use cross entropy for reducing difference between reconstruction data of Decoder and original data.<br>
(Loss function Expression)<br>
(Figure)

### Korean
VAE는 AutoEncoder의 파생 모델로 AutoEncoder와 동일하게 Encoder와 Decoder로 구성된다.<br>
AutoEncoder는 말 그대로 입력의 Encoding과 Decoding을 하는 모델이라면,<br>
VAE는 Encoder로부터 만들어진 값에 약간의 noise를 섞어 Decoder의 재구성 기능에 다양성을 주는 것이라고 할 수 있다.<br>

VAE는 명시적 생성 모델로 입력 데이터의 분포를 가정하고 그에 맞춰 학습하게 된다.<br>
여기서는 Normal Distribution으로 가정하고 fitting 한다.<br>

이 점 때문에 일반적인 AutoEncoder와의 차이점이 생기는데, VAE의 Encoder는 여러 개의 Normal distribution을 사용하여 합성한 Multinomial distribution에 fitting 시킨다.<br>
Normal distribution의 parameter로는 mean 과 standard deviation(std) 이 있다. 따라서 Encoder에서는 여러 개의 mean 값과 std 값을 만들어 Normal distribution에 입력 데이터를 fitting 시키는 역할을 한다고 볼 수 있다.<br>
또한, 이렇게 만들어진 각 mean 값과 std 값에 noise를 이용해서 기존에 학습된 distribution에 다양성을 준다. 이 과정을 reparameterize라고 한다.<br>
(수식 추가 예정)<br>
<br>
Noise를 섞어 다양성을 줌으로써 얻는 이점으로는 Decoder에 학습된 데이터에서 어느 정도 벗어나는 입력이 들어오더라도 대응하여 의도대로 Decoding 할 수 있도록 한 것이다.<br>
Decoder는 AutoEncoder와 동일하게 Encoder로부터 encoding 값을 다시 원래 모습으로 재구성하는 것이다.<br>
<br>
따라서 손실 함수는 Encoder를 학습시키기 위해 입력 데이터를 Normal distribution으로 이루어진 Multinomial distribution에 fitting 하기 위해 kl-divergence를 사용하고,<br>
Decoder의 재구성 데이터와 원본 데이터를 최대한 비슷하게 만들기 위해 cross entropy를 사용한다.<br>
(Loss function 수식 추가 예정)<br>
(모델 그림 추가 예정)<br>
<br>

## CVAE(Conditional Variational AutoEncoder)

### English
CVAE is the simliar model to VAE, but give input data included label to model. So, CVAE train the class of dataset.<br>
There is no other difference.<br>
Generally, CVAE's result is known more clear than VAE's.<br>
(Model structure)<br>

### Korean
CVAE는 기존의 VAE에 데이터만이 아니라 Label 값까지 같이 입력으로 줘 VAE 모델로 하여금 Class를 학습하게 하는 모델이다.<br>
그 외에는 차이점이 없다고 봐도 무방하다.<br>
보통 결과가 좀 더 깔끔하게 나온다고 알려져 있다.<br>
(모델 그림 추가 예정)<br>
<br>

## GAN(Generative Adversarial Network)

### English
GAN is a Generative model that composed of Generator and Discriminator.<br>
In GAN, Generator and Discriminator compete each other.<br>
Generator deceive Discriminator that fake data be decided real data at Discriminator.<br>
And Discriminator separate betwenn real data and fake data.<br>
(Model structure)<br>

### Korean
GAN은 Generator와 Discriminator로 구성된 생성 모델이다.<br>
GAN에서 Generator와 Discriminator는 서로 경쟁하는 모델인데,<br>
Generator는 가짜 데이터를 만들어 Discriminator가 원본 데이터라 판단하도록 속이려하고,<br>
Discriminator는 원본 데이터와 가짜 데이터를 구분하려고 한다.<br>

## Next tasks

1. Should add convolution layers in CVAE.<br>
Current CVAE based on DNN.<br>
So, I should change CVAE to convolutional CVAE.<br>

2. Need New generation models.<br>
First, add GAN and DCGAN<br>

3. Korean to English<br>
