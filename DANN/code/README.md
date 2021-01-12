## DCGAN(Deep Convolutional Generative Adversarial Network)

### English
DCGAN is the development model of Vanilla GAN, add Convolutional layers in traditional GAN architecture.<br>
DCGAN can be learned stably by adding a convolutional layer. Except for this, the almost parts are the same as GAN.<br>
In Paper, they showed enter created continuous noise using interpolation to model,<br>
the output change continuously.<br>
This phenomenon is called "Walking in the latent space".
<br>

### Korean

DCGAN은 기본적인 GAN의 발전 모델로, 기존의 GAN 구조에 Convolution layer를 사용한 것이다.<br>
Convolutional layer를 넣어 보다 안정적으로 학습할 수 있도록 하였고 사용 과정 등 모든 부분이 GAN과 동일하다.
논문에서는 보간법(interpolation)을 이용해서 noise를 연속적으로 생성해서 입력으로 넣으면,
출력도 연속적으로 변하는 모습을 보인다는 것을 보였다.
논문에서는 이런 현상을 Walking in the latent space라 부른다.
