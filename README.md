# All-about-XAI
This repository is all about papers and tools of Explainable AI

## Contents

- [Surveys](#surveys)
- [Background](#background)
- [XAL Methods:](#XAL-Method)
- [1. Transparent Models](#Transparent-Model)

- [2. Post-Hoc Explainability](#Post-Hoc-Explainability)
	- [Model-Agnostic](#Model-Agnostic)
		- [Visualization](#visualization)
		- [Feature Relevance/Importance Method](#Feature-RelevanceImportance-Method)
		- [Local interpretability](#Local-interpretability)
		- [Reinforcement learning](#reinforcement-learning)
		- [Recommend](#recommend)
	- [Model-Specific](#Model-Specific)
		- [CNN](#cnn)
			- [Visualization](#1-visualization)
			- [Transparent Model](#2-using-explainable-model)
			- [Model Modification](#3-archtecture-modification)
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
## Transparent Model
As long as the model is accurate for the task, and uses a reasonably restricted number of internal components, intrinsic interpretable models are suffcient. Otherwise, use post-hoc methods。
### Decision Trees
### General Additive Models
### Bayesian Models

## Post-Hoc Explainability
Including natural language explanations [71], visualizations of learned models [72], and explanations by example [73].
### Model Agnostic
#### Visualization
##### <span id="saliency">1. Saliency</span>
	>*[Interpretable explanations of black boxes by meaningful perturbation](), R. C. Fong, A. Vedaldi, in IEEE International Conference on Computer Vision, 2017, pp. 3429–3437.*
	
	>*[Real time image saliency for black box classifiers], P. Dabkowski, Y. Gal, in: Advances in Neural Information Processing Systems, 2017, pp. 6967–6976.*
	
##### <span id="sensitivity">2. Sensitivity</span>
Sensitivity refers to how an ANN output is influenced by its input and/or weight perturbations

>*[Opening black box data mining models using sensitivity analysis](https://core.ac.uk/download/pdf/55616214.pdf), P.Cortez and M.J.Embrechts, in Proc. IEEE Symp.Comput.Intell.Data Mining (CIDM), (2011)*
	
>*[Using sensitivity analysis and visualization techniques to open black box data mining models](https://www.sciencedirect.com/science/article/pii/S0020025512007098), P. Cortez and M. J. Embrechts,Inf. Sci. (2013).*	

##### 3. Shapely explanations & Game theory

>*[A unified approach to interpreting model predictions], S.M. Lundberg and S.I. Lee, in Proc. Adv. Neural Inf. Process. Syst., 2017.*
	
##### 4. Partial Dependence Plot (PDP)
	
>*[Auditing black-box models for indirect influence](), P. Adler, C. Falk, S. A. Friedler, T. Nix, G. Rybeck, C. Scheidegger, B. Smith, S. Venkatasubramanian, Knowledge and Information Systems (2018)*

##### 5. ICE: Individual Conditional Expectation
	
>*[Peeking inside the black box: Visualizing statistical learning with plots of individual conditional expectation], A. Goldstein, A. Kapelner, J. Bleich, E. Pitkin, Journal of Computational and Graphical Statistics 24 (1) (2015) 44–65.*

ICE plots extend PDP, reveal interactions and individual differences by disaggregating the PDP output.

##### 6. Dependence
	
>*[Visualizing the feature importance for black box models], G. Casalicchio, C. Molnar, B. Bischl, Joint European Conference on Machine Learning and Knowledge Discovery in Databases,Springer, 2018, pp. 655–670*
	
##### 7. Surrogate Models

>*[LIME](#lime)*
	
>*[Interpretability via model extraction.](https://arxiv.org/abs/1706.09773) O. Bastani, C. Kim, and H. Bastani. (2017).*
	
>*[TreeView: Peeking into deep neural networks via feature-space partitioning.](https://arxiv.org/abs/1611.07429) J. J. Thiagarajan, B. Kailkhura, P. Sattigeri, and K. N. Ramamurthy.(2016)*
	
##### 8. Loss Function Vis
>*[Visualizing the Loss Landscape of Neural Nets.](https://arxiv.org/abs/1712.09913) NeurIPS.Li, H., Xu, Z., Taylor, G., & Goldstein, T. (2017).*
	
<p align="center"><img width="50%" height="50%" src="images/loss-landscape.png?raw=true" /></p>
	
#### Feature Relevance/Importance Method
- [Saliency](#saliency)

- [Sensitivity](#sensitivity)

- Influence functions
	>*[Understanding black-box predictions via influence functions], P. W. Koh, P. Liang, in: Proceedings of the 34th International Conference on Machine Learning. (2017)*

- Game theory
	>*[An efficient explanation of individual classifications using game theory.]I. Kononenko et al.Journal of Machine Learning Research, 11(Jan):1–18, 2010.*

- Interacticon based

	>*GoldenEye: [A peek into the black box: exploring classifiers by randomization], A. Henelius, K. Puolamaki, H. Bostrom, L. Asker, P. Papapetrou, Data mining and knowledge discovery (2014)*
	>*[Interpreting classifiers through attribute interactions in datasets] A. Henelius, K. Puolamaki, A. Ukkonen, (2017).arXiv:1707.07576.*

- Others
	>*[Iterative orthogonal feature projection for diagnosing bias in black-box models] J. Adebayo, L. Kagal, (2016). arXiv:1611.04967.*
	
#### Local interpretability
Global is Understanding of the whole logic of a model and follows the entire reasoning leading to all the different possible outcomes.

While local Explaining the reasons for a specific decision or single pre-diction

- <span id="lime">LIME</span>
	>*[Why should i trust you?: Explaining the predictions of any classifier]() M. T. Ribeiro, S. Singh, and C. Guestrin,in Proc. 22nd ACM SIGKDD Int. Conf. Knowl. Discovery Data Mining, 2016,*
	
	Approximates a DNN’s predictions using sparse linear models where we can easily identify important features.
	Extracts image regions that are highly sensitive to the network output. 
- anchor

	>*[Anchors: High-precision model-agnostic explanations]() M. T. Ribeiro, S. Singh, and C. Guestrin, in Proc. AAAI Conf. Artif. Intell., 2018.*
- LOCO

	>*[Distribution-free predictive inference for regression](http://www.stat.cmu.edu/~ryantibs/papers/conformal.pdf) J. Lei, M. G’Sell, A. Rinaldo, R. J. Tibshirani, and L.Wasserman.*
- LRP

	>*On pixel-wise explanations for non-linear classifier decisions by layer-wise relevance propagation. S. Bach, A. Binder, G. Montavon, F. Klauschen, K.-R. Müller, and W. Samek, PLoS ONE, 2015.*
### Model Specific
#### CNN
#### 1. Visualization

**(1) Max Activation**

Synthesize input pattern that can cause maximal activation of a neuron

- Saliency Maps(2013)

	>*[Deep inside convolutional networks: visualising image classification models and saliency maps.](https://arxiv.org/abs/1312.6034) Karen Simonyan, Andrea Vedaldi, and Andrew Zisserman. In arXiv:1312.6034, 2013.*

	Saliency maps are usually rendered as a heatmap, where hotness corresponds to regions that have a big impact on the model’s final decision

	<p align="center"><img width="50%" height="50%" src="images/saliency-map.png?raw=true" /></p>
	
- CAM: Class Activation Map(2016)

	The CAM highlights the class-specific discriminative regions.
	>*[Learning Deep Features for Discriminative Localization.](https://arxiv.org/abs/1512.04150) Zhou, B., Khosla, A., Lapedriza, À., Oliva, A., & Torralba, A.2016 IEEE (CVPR), 2921-2929.*

	Note:https://medium.com/@ahmdtaha/learning-deep-features-for-discriminative-localization-aa73e32e39b2

	<p align="center"><img width="50%" height="50%" src="images/CAM.jpeg?raw=true" /></p>

	>*[Grad-CAM: Why did you say that? ] R. R. Selvaraju, A. Das, R. Vedantam, M. Cogswell, D. Parikh, D. Batra,(2016).*


- Filter Activation
	>*[Convergent learning: Do different neural networks learn the same representations?](http://arxiv.org/abs/1511.07543), Y. Li, J. Yosinski, J. Clune, H. Lipson, J. E. Hopcroft, in: ICLR, 2016.*

	computing the correlation between activations of different filters. 	
**(2) Deconvolution(2010)**

Finds the selective patterns from the input image that activate a specific neuron in the convolutional layers by projecting the lowdimension neurons'feature maps back to the image dimension

1. First propose Deconv

	>*M. D. Zeiler, D. Krishnan, G. W. Taylor, R. Fergus, [Deconvolutional networks.](https://ieeexplore.ieee.org/document/5539957)in: CVPR, Vol. 10,2010, p. 7.*

2. Use Deconv to visualize CNN

	>*[Visualizing and understanding convolutional net-works.](https://cs.nyu.edu/~fergus/papers/zeilerECCV2014.pdf) Matthew D. Zeiler and Rob Fer-gus. In ECCV, 2014.*

**(3) Inversion**

Different from the above, which visualize the CNN from a single neuron’s activation, this methods is from Layer-level.

Reconstructs an input image based from a specific layer's feature maps, which reveals what image information is preserved in that layer

- Regularizer-based

	>*[Understanding deep image representations by inverting them.](https://arxiv.org/abs/1412.0035) Aravindh Mahendran and Andrea Vedaldi. In CVPR, 2015.*
	
	<p align="center"><img width="50%" height="50%" src="images/invertcnn.jpg?raw=true" /></p>
	
	>*[Visualizing deep convolutional neural networks using natural pre-images](http://dx.doi.org/10.1007/s11263-016-0911-8), A. Mahendran and A. Vedaldi, International Journal of Computer Vision, 120 (2016), 233–255.*
	
- Up-conv net

	>*[Inverting visual representations with convolutional networks.]( https://arxiv.org/abs/1506.02753) Alexey Dosovitskiy and Thomas Brox. In CVPR, 2016.*
	
	>*[Plug & play generative networks: Conditional iterative generation of images in latent space.] Anh Nguyen, Jeff Clune, Yoshua Ben-gio, Alexey Dosovitskiy, and Jason Yosinski. CVPR, 2017.*
	
	>*[Object detectors emerge in deep scene cnns. ] Bolei Zhou, Aditya Khosla, Agata Lapedriza, Aude Oliva, and Antonio Torralba. In ICRL, 2015.*

**(4) Viusalization System: Understanding, Diagnosis, Refinement**

- Visual system
	>*[Towards better analysis of deep convolutional neural networks](https://arxiv.org/abs/1604.07043), M. Liu, J. Shi, Z. Li, C. Li, J. Zhu, S. Liu, IEEE transactions on visualization and computer graphics 23 (1) (2016) 91–100.*

	<p align="center"><img width="50%" height="50%" src="images/visualization-system.png?raw=true" /></p>
		
	>*[An Interactive Node-Link Visualization of Convolutional Neural Networks.](https://link.springer.com/chapter/10.1007/978-3-319-27857-5_77) Harley A.W. (2015) Advances in Visual Computing. ISVC*
	
	showing not only what it has learned, but how it behaves given new user-provided input.
	
- Toolbox for visualization CNN

	>*[Understanding Neural Networks Through Deep Visualization.](https://arxiv.org/abs/1506.06579) Yosinski, J., Clune, J., Nguyen, A.M., Fuchs, T.J., & Lipson, H. (2015). ArXiv, abs/1506.06579.*
	
	<p align="center"><img width="50%" height="50%" src="images/toolbox.jpg?raw=true" /></p>
	
	>*[Picasso: A Modular Framework for Visualizing the Learning Process of Neural Network Image Classifiers.](https://medium.com/merantix/picasso-a-free-open-source-visualizer-for-cnns-d8ed3a35cfc5) Henderson, R. & Rothe, R., (2017). Journal of Open Research Software. 5(1), p.22.*
	

	compute actual receptive field of filters.
#### 2. Using explainable Model
- Decision Tree

	>*[Interpreting CNNs via decision trees](https://arxiv.org/abs/1802.00121) , Q. Zhang, Y. Yang, H. Ma, Y. N. Wu, in: IEEE Conference on Computer Vision and Pattern Recognition, 2019, pp. 6261–6270.*

#### 3. Archtecture Modification

- Layer Modification
	>*[Striving for simplicity: the all convolutional net.] ost Tobias Springenberg, Alexey Dosovitskiy, Thomas Brox, and Martin Ried-miller.  ICLR workshop, 2015.*

	Objext Detection:replace maxpooling layer with all conv-layers
	
#### RNN
- Visualization

	>*[Visualizing and understanding recurrent networks]A. Karpathy, J. Johnson, L. Fei-Fei, (2015).arXiv:1506.02078.*
- Feature Relevence

	>*[Explaining recurrent neural network predictions in sentiment analysis ](https://arxiv.org/abs/1706.07206)L. Arras, G. Montavon, K.-R. Muller, W. Samek, (2017). arXiv:1706.07206.*

