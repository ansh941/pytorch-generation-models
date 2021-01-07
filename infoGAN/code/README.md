## infoGAN(information Generative Adversarial Network)

### English
infoGAN is the model that introduce information theory in traditional GAN.<br>
The Traditional GAN generate samples using only noise z, but infoGAN uses latent variable c too.<br>
The infoGAN's idea is that maximize the mutual information between the generated sample G(z,c) and c.<br>
As a result, the generated sample is changed according to c.<br>
And also, there is no contraint at generting noise in normal GAN.<br>
So, the model has the probability that model is trained representation entangled.<br>
But, infoGAN intend to train direction for learning representation unentangled according to latent variable by mutual information maximization.<br>
Ultimately, infoGAN can be said that use a mutual information for the model trained clear and unentangled shape at learning representation.

### Korean
infoGAN은 기존의 GAN에 정보 이론을 도입한 모델이다. 원래의 GAN에서는 noise z만 이용해서 생성하지만 infoGAN에서는 Latent variable c가 추가로 주어진다. Generator를 G라고 할 때, 생성된 샘플 G(z,c)와 c의 상호 정보량을 최대로 하는 것이 infoGAN의 아이디어이다. 결과적으로 보면 c의 변화에 따라 생성되는 샘플의 결과가 변하도록 하는 것이다. 또한, 일반적인 GAN에서는 noise를 생성할 때 어떠한 제약도 없기 때문에 모델이 representation을 학습할 때 꼬인(entangled) 형태로 학습할 가능성이 있다. 하지만 c와 G(z,c)의 상호 정보량을 최대화하는 것으로 latent variable에 따라 꼬이지 않은(unentangled) 하게 학습하도록 유도한다. 결국 infoGAN은 representation을 학습하는데 있어 모델이 좀 더 명확하고 꼬이지 않은 형태로 학습될 수 있도록 하기 위해 상호 정보량을 이용한 것이라고 볼 수 있다.
