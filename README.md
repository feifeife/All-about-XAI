# All-about-XAI
This repository is all about papers and tools of Explainable AI

## Contents

- [Surveys](#surveys)
- [Background](#background)
- [XAL Methods:](#XAL-Method)
- [1. Transparent Models](#Transparent-Model)

- [2. Post-Hoc Explainability](#Post-Hoc-Explainability)
	- [Model-Agnostic](#Model-Agnostic)
		- [visualization](#visualization)
		- [reinforcement learning](#reinforcement-learning)
		- [recommend](#recommend)
	- [Model-Specific](#Model-Specific)
		- [CNN](#cnn)
		- [RNN](#rnn)

## Surveys
>*[Visual Interpretability for Deep Learning: a Survey](https://arxiv.org/abs/1802.00614) Quanshi Zhang, Song-Chun Zhu (2018) CVPR*
	
interpretable/disentangled middle-layer representations
>*[Towards a rigorous science of interpretable machine learning.](https://arxiv.org/abs/1702.08608) F. Doshi-Velez and B. Kim. (2018).*

>*[Trends and trajectories for explainable, accountable and intelligible systems: An HCI research agenda.]() A. Abdul, J. Vermeulen, D. Wang, B. Y. Lim, and M. Kankanhalli,in Proc. SIGCHI Conf. Hum. FactorsComput. Syst. (CHI), 2018, p. 582*
	
most focus on HCI research
	
>*[A survey of methods for explaining black box models.](https://arxiv.org/abs/1802.01933) R. Guidotti, A. Monreale, F. Turini, D. Pedreschi, and F. Giannotti.(2018).*

presented a detailed taxonomy of explainability methods according to the type of problem faced. 
>*[Peeking Inside the Black-Box: A Survey on Explainable Artificial Intelligence (XAI)](https://ieeexplore.ieee.org/document/8466590) A. Adadi and M. Berrada,in IEEE Access, vol. 6, pp. 52138-52160, 2018.*
>*[Explainable Artificial Intelligence (XAI): Concepts, Taxonomies, Opportunities and Challenges toward Responsible AI](https://arxiv.org/abs/1910.10045)Alejandro Barredo Arrieta, Natalia Díaz-Rodríguez.arxiv.(2019)*
## XAI Method
### Transparent Model
As long as the model is accurate for the task, and uses a reasonably restricted number of internal components, intrinsic interpretable models are suffcient. Otherwise, use post-hoc methods。
#### Decision Trees
#### General Additive Models
#### Bayesian Models

### Post-Hoc Explainability
Including natural language explanations [71], visualizations of learned models [72], and explanations by example [73].
#### Model Agnostic
##### Visualization
- Saliency Map
- Surrogate Models
- Partial Dependence Plot (PDP)
- ndividual Conditional Expectation (ICE)
##### Knowledge Extraction
##### Feature Relevance Method


##### Example-based explanation
##### Global interpretability

Understanding of the whole logic of a model and follows the entire reasoning leading to all the different possible outcomes.
##### Local interpretability

Explaining the reasons for a specific decision or single pre-diction
- LIME
>*[Why should i trust you?: Explaining the predictions of any classifier]() M. T. Ribeiro, S. Singh, and C. Guestrin,in Proc. 22nd
ACM SIGKDD Int. Conf. Knowl. Discovery Data Mining, 2016,*

Extracts image regions that are highly sensitive to the network output. 
- anchor
>*[Anchors: High-precision model-agnostic explanations]() M. T. Ribeiro, S. Singh, and C. Guestrin, in Proc. AAAI Conf. Artif. Intell., 2018.*
- LOCO
>*[Distribution-free predictive inference for regression](http://www.stat.cmu.edu/~ryantibs/papers/conformal.pdf) J. Lei, M. G’Sell, A. Rinaldo, R. J. Tibshirani, and L.Wasserman.*
- LRP
>*On pixel-wise explanations for non-linear classifier decisions by layer-wise relevance propagation. S. Bach, A. Binder, G. Montavon, F. Klauschen, K.-R. Müller, and W. Samek, PLoS ONE, 2015.*
#### Model Specific
##### CNN
- 1. Visualization of filters(CNN representations)

1) Compute gradients of the score of a given CNN unit.
- First propose Deconv
>*M. D. Zeiler, D. Krishnan, G. W. Taylor, R. Fergus, [Deconvolutional networks.](https://ieeexplore.ieee.org/document/5539957)in: CVPR, Vol. 10,2010, p. 7.*
- Use Deconv to visualize CNN
>*Matthew D. Zeiler and Rob Fer-gus. [Visualizing and understanding convolutional net-works.](https://cs.nyu.edu/~fergus/papers/zeilerECCV2014.pdf) In ECCV, 2014.*
>*Aravindh Mahendran and Andrea Vedaldi. [Understanding deep image representa-tions by inverting them.](https://arxiv.org/abs/1412.0035) In CVPR, 2015.*

- CAM - Class Activation Map
The CAM highlights the class-specific discriminative regions.
>*Zhou, B., Khosla, A., Lapedriza, À., Oliva, A., & Torralba, A. (2015). [Learning Deep Features for Discriminative Localization.](https://arxiv.org/abs/1512.04150) 2016 IEEE (CVPR), 2921-2929.*

Note:https://medium.com/@ahmdtaha/learning-deep-features-for-discriminative-localization-aa73e32e39b2

>*R. R. Selvaraju, A. Das, R. Vedantam, M. Cogswell, D. Parikh, D. Batra, Grad-CAM: Why did you say that? (2016).*

- Saliency Maps
>*Karen Simonyan, Andrea Vedaldi, and Andrew Zisserman. Deep inside convolutional networks: visualising image classification models and saliency maps. In arXiv:1312.6034, 2013.*
- Viusalization System
>*M. Liu, J. Shi, Z. Li, C. Li, J. Zhu, S. Liu, [Towards better analysis of deep convolutional neural
networks](https://arxiv.org/abs/1604.07043), IEEE transactions on visualization and computer graphics 23 (1) (2016) 91–100.*

<p align="center"><img width="50%" height="50%" src="images/visualization-system.png?raw=true" /></p>
	
>*ost Tobias Springenberg, Alexey Dosovitskiy, Thomas Brox, and Martin Ried-miller. Striving for simplicity: the all convolutional net. ICLR workshop, 2015. *

Objext Detection:replace maxpooling layer with all conv-layers

2) Invert CNN feature maps to image

>*Alexey Dosovitskiy and Thomas Brox. Inverting visual representations with convolutional networks. In CVPR, 2016.*
>*Anh Nguyen, Jeff Clune, Yoshua Ben-gio, Alexey Dosovitskiy, and Jason Yosinski. Plug & play generative networks: Conditional iterative generation of images in latent space. CVPR, 2017.*
>*Bolei Zhou, Aditya Khosla, Agata Lapedriza, Aude Oliva, and Antonio Torralba. Object de-tectors emerge in deep scene cnns. In ICRL, 2015.*

compute actual receptive field of filters.

##### RNN
