## CGAN(Conditional Generative Adversarial Network)

### English
CGAN has the same structure with Vanilla GAN and operation method is the same also.<br>
CVAE gets the effect like supervised training by giving additional information such as class label in VAE.<br>
CGAN works the same but use GAN.<br>
We can generate the class data what we want by giving the label to model.<br>
(Model structure)<br>

### Korean
CGAN은 기본적인 GAN과 같은 구조를 가지고 있으며, 동작 방식 역시 동일하게 Generator와 Discriminator의 경쟁으로 학습되는 모델이다.<br>
CVAE가 VAE에 class label 정보를 추가로 주고 지도 학습과 같은 효과를 나타나게 했는데,<br>
마찬가지로 CGAN은 GAN에 class label 정보를 추가로 주고 지도 학습 효과를 내는 모델이다.<br>
모델에 Label을 줌으로써 원하는 클래스의 데이터를 생성할 수 있게 되었다.<br>
