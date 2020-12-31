## GAN(Generative Adversarial Network)

### English
GAN is a Generative model that composed of Generator and Discriminator.<br>
In GAN, Generator and Discriminator compete each other.<br>
Generator deceive Discriminator that fake data be decided real data at Discriminator.<br>
And Discriminator separate betwenn real data and fake data.<br>
![GAN_loss.png](README_images/GAN_loss.png)<br>
![GAN_model_structure.png](README_images/GAN_model_structure.png)<br>

### Korean
GAN은 Generator와 Discriminator로 구성된 생성 모델이다.<br>
GAN에서 Generator와 Discriminator는 서로 경쟁하는 모델인데,<br>
Generator는 가짜 데이터를 만들어 Discriminator가 원본 데이터라 판단하도록 속이려하고,<br>
Discriminator는 원본 데이터와 가짜 데이터를 구분하려고 한다.<br>
수식은 다음과 같다. 아래 수식에서 G는 Generator, D는 Discriminator, x는 원본 데이터, z는 noise이다.
여기서 G를 최소화 하고, D를 최대화하는데, 이는 D(x)를 최대화해 x를 잘 구분하게 하고, G(z)를 최소화해 D로 하여금 G가 만들어낸 값을 제대로 구분하지 못하게 하기 위한 것이다.
![GAN_loss.png](README_images/GAN_loss.png)<br>
