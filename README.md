# All-about-XAI
This repository is all about papers and tools of Explainable AI

## Contents

- [Surveys](#surveys)
- [Visualization Systems/Tools](#Visualization-SystemsTools)
- [Background](#background)
- [XAL Methods:](#XAL-Method)
- [1. Transparent Models](#Transparent-Model)

- [2. Post-Hoc Explainability](#Post-Hoc-Explainability)
	- [Model-Agnostic](#Model-Agnostic)
		- [Visualization](#visualization)
		- [Feature Relevance/Importance Method](#Feature-RelevanceImportance-Method)
		- [Local interpretability](#Local-interpretability)
	- [Model-Specific](#Model-Specific)
		- [Tree-based](#tree-based-model)
		- [CNN](#cnn)
			- [Visualization](#1-visualization)
			- [Transparent Model](#2-using-explainable-model)
			- [Model Modification](#3-archtecture-modification)
		- [RNN](#rnn)
		- [Reinforcement learning](#reinforcement-learning)

## Surveys
>*[Towards better analysis of machine learning models: A visual analytics perspective.](https://arxiv.org/abs/1810.08174)Liu, S., Wang, X., Liu, M., & Zhu, J. (2017). Visual Informatics, 1, 48-56.*

>*[Visual Interpretability for Deep Learning: a Survey](https://arxiv.org/abs/1802.00614) Quanshi Zhang, Song-Chun Zhu (2018) CVPR*
	
interpretable/disentangled middle-layer representations
>*[Towards a rigorous science of interpretable machine learning.](https://arxiv.org/abs/1702.08608) F. Doshi-Velez and B. Kim. (2018).*

>*[Trends and trajectories for explainable, accountable and intelligible systems: An HCI research agenda.]() A. Abdul, J. Vermeulen, D. Wang, B. Y. Lim, and M. Kankanhalli,in Proc. SIGCHI Conf. Hum. FactorsComput. Syst. (CHI), 2018, p. 582*
	
most focus on HCI research
	
>*[A survey of methods for explaining black box models.](https://arxiv.org/abs/1802.01933) R. Guidotti, A. Monreale, F. Turini, D. Pedreschi, and F. Giannotti.(2018).*

presented a detailed taxonomy of explainability methods according to the type of problem faced. 

>*[Peeking Inside the Black-Box: A Survey on Explainable Artificial Intelligence (XAI)](https://ieeexplore.ieee.org/document/8466590) A. Adadi and M. Berrada,in IEEE Access, vol. 6, pp. 52138-52160, 2018.*
	
>*[Explainable Artificial Intelligence (XAI): Concepts, Taxonomies, Opportunities and Challenges toward Responsible AI](https://arxiv.org/abs/1910.10045) Alejandro Barredo Arrieta, Natalia Díaz-Rodríguez.arxiv.(2019)*

>*[How convolutional neural network see the world - A survey of convolutional neural network visualization methods.](https://arxiv.org/abs/1804.11191) Qin, Z., Yu, F., Liu, C., & Chen, X. (2018).  ArXiv, abs/1804.11191.*

## Visualization Systems/Tools
- explAIner
	>*[explAIner: A Visual Analytics Framework for Interactive and Explainable Machine Learning](https://arxiv.org/abs/1908.00087), Spinner, T., Schlegel, U., Schäfer, H., & El-Assady, M. (2019).  IEEE VAST, Transactions on Visualization and Computer Graphics, 26, 1064-1074.*
	
- CSI: collaborative semantic inference
	>*[Visual Interaction with Deep Learning Models through Collaborative Semantic Inference.](https://arxiv.org/abs/1907.10739) Gehrmann, S., Strobelt, H., Krüger, R., Pfister, H., & Rush, A.M. (2019). IEEE VAST. Transactions on Visualization and Computer Graphics, 26, 884-894.*
	
	User can both understand and control parts of the model reasoning process. eg. in text summarization system, user can  collaborative writing a summary with machines suggestion.
- Manifold
	>*[Manifold: A Model-Agnostic Framework for Interpretation and Diagnosis of Machine Learning Models.](https://arxiv.org/abs/1808.00196)Zhang, J., Wang, Y., Molino, P., Li, L., & Ebert, D.S. (2019). IEEE VAST, Transactions on Visualization and Computer Graphics, 25, 364-373.*
	
	inspection (hypothesis), explanation (reasoning), and refinement (verification)
	
	<p align="center"><img width="50%" height="50%" src="images/manifold.png?raw=true" /></p>
- DeepVID
	>*[DeepVID: Deep Visual Interpretation and Diagnosis for Image Classifiers via Knowledge Distillation. ](https://junpengw.bitbucket.io/image/research/pvis19.pdf) Wang, J., Gou, L., Zhang, W., Yang, H.T., & Shen, H. (2019). IEEE Transactions on Visualization and Computer Graphics, 25, 2168-2180.*
	
- ActiVis
	>*[Activis: Visual exploration of industry-scale deep neural network models.](https://arxiv.org/abs/1704.01942) M. Kahng, P. Y. Andrews, A. Kalro, and D. H. P. Chau. IEEE transactions on visualization and computer graphics, 24(1):88–97, 2018* Facebook
	
	unify instance- and subset-level inspections of neuron activations
	
	<p align="center"><img width="50%" height="50%" src="images/activis.png?raw=true" /></p>
	
## XAI Method
## Transparent Model
As long as the model is accurate for the task, and uses a reasonably restricted number of internal components, intrinsic interpretable models are suffcient. Otherwise, use post-hoc methods.
### Decision Trees
### General Additive Models
### Bayesian Models

## Post-Hoc Explainability
Including natural language explanations, visualizations of learned models , and explanations by example.
### Model Agnostic
#### Visualization
##### <span id="saliency">1. Saliency</span>

>*[Interpretable explanations of black boxes by meaningful perturbation](https://arxiv.org/abs/1704.03296), R. C. Fong, A. Vedaldi, in IEEE International Conference on Computer Vision, 2017, pp. 3429–3437.*
	
>*[Real time image saliency for black box classifiers](https://arxiv.org/abs/1705.07857), P. Dabkowski, Y. Gal, in: Advances in Neural Information Processing Systems, 2017, pp. 6967–6976.*
	
##### <span id="sensitivity">2. Sensitivity</span>

Sensitivity refers to how an ANN output is influenced by its input and/or weight perturbations

>*[Opening black box data mining models using sensitivity analysis](https://core.ac.uk/download/pdf/55616214.pdf), P.Cortez and M.J.Embrechts, in Proc. IEEE Symp.Comput.Intell.Data Mining (CIDM), (2011)*
	
>*[Using sensitivity analysis and visualization techniques to open black box data mining models](https://www.sciencedirect.com/science/article/pii/S0020025512007098), P. Cortez and M. J. Embrechts,Inf. Sci. (2013).*	

##### 3. SHAP: SHapley Additive exPlanations

Assign importance values for each feature, for a given prediction based on the game theoretic concept of Shapley values

>*[A unified approach to interpreting model predictions](https://arxiv.org/abs/1705.07874), S.M. Lundberg and S.I. Lee, in Proc. Adv. Neural Inf. Process. Syst., 2017.*
	
##### 4. Partial Dependence Plot (PDP)
- PDP
>*[Auditing black-box models for indirect influence](https://arxiv.org/abs/1602.07043), P. Adler, C. Falk, S. A. Friedler, T. Nix, G. Rybeck, C. Scheidegger, B. Smith, S. Venkatasubramanian, Knowledge and Information Systems (2018)*

- ICE: Individual Conditional Expectation(extends PDP)
	
>*[Peeking inside the black box: Visualizing statistical learning with plots of individual conditional expectation](https://arxiv.org/abs/1309.6392), A. Goldstein, A. Kapelner, J. Bleich, E. Pitkin, Journal of Computational and Graphical Statistics 24 (1) (2015) 44–65.*

ICE plots extend PDP, reveal interactions and individual differences by disaggregating the PDP output.

- PI & ICI

>*[Visualizing the feature importance for black box models], G. Casalicchio, C. Molnar, B. Bischl, Joint European Conference on Machine Learning and Knowledge Discovery in Databases,Springer, 2018, pp. 655–670*
	
##### 5. Surrogate Models

>*[LIME](#lime)*
	
>*[Interpretability via model extraction.](https://arxiv.org/abs/1706.09773) O. Bastani, C. Kim, and H. Bastani. (2017).*
	
>*[TreeView: Peeking into deep neural networks via feature-space partitioning.](https://arxiv.org/abs/1611.07429) J. J. Thiagarajan, B. Kailkhura, P. Sattigeri, and K. N. Ramamurthy.(2016)*
	
##### 6. Loss Function Vis
>*[Visualizing the Loss Landscape of Neural Nets.](https://arxiv.org/abs/1712.09913) NeurIPS.Li, H., Xu, Z., Taylor, G., & Goldstein, T. (2017).*
	
<p align="center"><img width="50%" height="50%" src="images/loss-landscape.png?raw=true" /></p>
	
#### 7. Feature Relevance/Importance Method
- [Saliency](#saliency)

- [Sensitivity](#sensitivity)

- [Partial Dependence Plot](#Partial-Dependence-Plot)

- Influence functions

	>*[Understanding black-box predictions via influence functions], P. W. Koh, P. Liang, in: Proceedings of the 34th International Conference on Machine Learning. (2017)*

- Interacticon based

	>*GoldenEye: [A peek into the black box: exploring classifiers by randomization], A. Henelius, K. Puolamaki, H. Bostrom, L. Asker, P. Papapetrou, Data mining and knowledge discovery (2014)*
	
	>*[Interpreting classifiers through attribute interactions in datasets] A. Henelius, K. Puolamaki, A. Ukkonen, (2017).arXiv:1707.07576.*

- Others
	>*[Iterative orthogonal feature projection for diagnosing bias in black-box models] J. Adebayo, L. Kagal, (2016). arXiv:1611.04967.*

#### 8. Model Distillation

	
#### Local interpretability
Global is Understanding of the whole logic of a model and follows the entire reasoning leading to all the different possible outcomes.

While local Explaining the reasons for a specific decision or single pre-diction

- <span id="lime">LIME</span>
	>*[Why should i trust you?: Explaining the predictions of any classifier]() M. T. Ribeiro, S. Singh, and C. Guestrin,in Proc. 22nd ACM SIGKDD Int. Conf. Knowl. Discovery Data Mining, 2016,*
	
	Approximates a DNN’s predictions using sparse linear models where we can easily identify important features.
	Extracts image regions that are highly sensitive to the network output. 

- LOCO

	>*[Distribution-free predictive inference for regression](http://www.stat.cmu.edu/~ryantibs/papers/conformal.pdf) J. Lei, M. G’Sell, A. Rinaldo, R. J. Tibshirani, and L.Wasserman.*
- LRP
	>*On pixel-wise explanations for non-linear classifier decisions by layer-wise relevance propagation. S. Bach, A. Binder, G. Montavon, F. Klauschen, K.-R. Müller, and W. Samek, PLoS ONE, 2015.*
- anchor
	>*[Anchors: High-precision model-agnostic explanations]() M. T. Ribeiro, S. Singh, and C. Guestrin, in Proc. AAAI Conf. Artif. Intell., 2018.*
	
### Model Specific
#### Tree-based Model
- random forest
	>*[iForest: Interpreting Random Forests via Visual Analytics](https://ieeexplore.ieee.org/document/8454906) Xun Zhao, Yanhong Wu, Dik Lun Lee, and Weiwei Cui. IEEE VIS 2018*

	Summarize the decision paths in random forests.
	
	<p align="center"><img width="50%" height="50%" src="images/iForest-system.png?raw=true" /></p>

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

- CNNVis
	>*[Towards better analysis of deep convolutional neural networks](https://arxiv.org/abs/1604.07043), M. Liu, J. Shi, Z. Li, C. Li, J. Zhu, S. Liu, IEEE transactions on visualization and computer graphics 23 (1) (2016) 91–100.*

	<p align="center"><img width="50%" height="50%" src="images/visualization-system.png?raw=true" /></p>
		
	>*[An Interactive Node-Link Visualization of Convolutional Neural Networks.](https://link.springer.com/chapter/10.1007/978-3-319-27857-5_77) Harley A.W. (2015) Advances in Visual Computing. ISVC*
	
	showing not only what it has learned, but how it behaves given new user-provided input.
- Block
	>*[Do convolutional neural networks learn class hierarchy? ](https://arxiv.org/abs/1710.06501) A. Bilal, A. Jourabloo, M. Ye, X. Liu, and L. Ren. IEEE transactions on visualization and computer graphics, 24(1):152–162, 2018.*

	Including a class hierarchy and confusion matrix showing misclassified samples only, bands indicate the selected classes in both dimensions and a sample viewer
	
	<p align="center"><img width="50%" height="50%" src="images/block.png?raw=true" /></p>
	
- DeepEyes
	>*[A. DeepEyes: Progressive Visual Analytics for Designing Deep Neural Networks. ](https://graphics.tudelft.nl/Publications-new/2018/PHVLEV18/paper216.pdf) Pezzotti, N., Höllt, T., van Gemert, J., Lelieveldt, B. P., Eisemann, E., & Vilanova, VAST 2017.*
	
	Identification of stable layers, degenerated filters,patterns undetected, oversized layers, unnecessary layers or the need of additional layers
	
	<p align="center"><img width="50%" height="50%" src="images/deepeyes.png?raw=true" /></p>
	
- Toolbox for visualization CNN
	
	>*[Understanding Neural Networks Through Deep Visualization.](https://arxiv.org/abs/1506.06579) Yosinski, J., Clune, J., Nguyen, A.M., Fuchs, T.J., & Lipson, H. (2015). ArXiv, abs/1506.06579.*
	
	<p align="center"><img width="50%" height="50%" src="images/toolbox.jpg?raw=true" /></p>
	
	>*[Visualizing the Hidden Activity of Artificial Neural Networks.](https://ieeexplore.ieee.org/document/7539329)Rauber, P.E., Fadel, S.G., Falcão, A.X., & Telea, A. (2017). IEEE Transactions on Visualization and Computer Graphics, 23, 101-110.*
	
	Using dimensionality reduction for: 1.visualizing the relationships between learned representations of observations, and 2. visualizing the relationships between artificial neurons.
	
	<p align="center"><img width="50%" height="50%" src="images/hiddenacitivity.png?raw=true" /></p>
	
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
	
	>*[Visualizing and understanding recurrent networks](https://arxiv.org/abs/1506.02078) A. Karpathy, J. Johnson, L. Fei-Fei, (2015).arXiv:1506.02078.*
- Feature Relevence

	>*[Explaining recurrent neural network predictions in sentiment analysis ](https://arxiv.org/abs/1706.07206)L. Arras, G. Montavon, K.-R. Muller, W. Samek, (2017). arXiv:1706.07206.*
- RNNVis
	>*[Understanding hidden memories of recurrent neural networks. ](https://arxiv.org/abs/1710.10777) Y. Ming, S. Cao, R. Zhang, Z. Li, Y. Chen, Y. Song, and H. Qu. In Visual Analytics Science and Technology (VAST), 2017 IEEE Conference on.IEEE, 2017.*
- LISA
	>*[LISA: Explaining Recurrent Neural Network Judgments via Layer-wIse Semantic Accumulation and Example to Pattern Transformation.](https://arxiv.org/abs/1808.01591) Gupta, P., & Schütze, H. (2018). BlackboxNLP@EMNLP.*

-LSTMVis
	>*[Lstmvis: A tool for visual analysis of hidden state dynamics in recurrent neural networks.](https://arxiv.org/abs/1606.07461) H. Strobelt, S. Gehrmann, H. Pfister, and A. M. Rush. IEEE transactions on visualization and computer graphics, 24(1):667–676,2018.*
	
- Seq2Seq
	>*[Seq2seq-Vis: A Visual Debugging Tool for Sequence-to-Sequence Models.](https://arxiv.org/abs/1804.09299) Strobelt, H., Gehrmann, S., Behrisch, M., Perer, A., Pfister, H., & Rush, A.M. (2018). IEEE Transactions on Visualization and Computer Graphics, 25, 353-363.*

- Attention
	>*[Visualizing Attention in Transformer-Based Language models.](https://arxiv.org/abs/1904.02679)Vig, J. (2019).*
	
	>*[Deep Features Analysis with Attention Networks.](https://arxiv.org/abs/1901.10042) Xie, S., Chen, D., Zhang, R., & Xue, H. (2019).  ArXiv, abs/1901.10042.*
- Self-Attention
	>*[SANVis: Visual Analytics for Understanding Self-Attention Networks. ](https://arxiv.org/abs/1909.09595) Park, C., Na, I., Jo, Y., Shin, S., Yoo, J., Kwon, B.C., Zhao, J., Noh, H., Lee, Y., & Choo, J. (2019). ArXiv, abs/1909.09595.*
- Bert
	>*[Visualizing and Measuring the Geometry of BERT.](https://arxiv.org/pdf/1906.02715.pdf) Coenen, A., Reif, E., Yuan, A., Kim, B., Pearce, A., Viégas, F.B., & Wattenberg, M. (2019). NeurlIPS, abs/1906.02715.*

#### Generative Model
- GANViz
	>*[GANViz: A Visual Analytics Approach to Understand the Adversarial Game. ](https://junpengw.bitbucket.io/image/research/pvis18.pdf) Wang, J., Gou, L., Yang, H.T., & Shen, H. (2018). IEEE Transactions on Visualization and Computer Graphics, 24, 1905-1917.*
- DGTracker
	>*[Analyzing the training processes of deep generative models.](http://cgcad.thss.tsinghua.edu.cn/mengchen/publications/dgmtracker/paper.pdf) M. Liu, J. Shi, K. Cao, J. Zhu, and S. Liu. IEEE transactions on visualization and computer graphics, 24(1):77–87, 2018.*
	
	<p align="center"><img width="50%" height="50%" src="images/dgtracker.png?raw=true" /></p>
	
- GAN Lab
	>*[GAN Lab: Understanding Complex Deep Generative Models using Interactive Visual Experimentation. ](https://arxiv.org/abs/1809.01587)Kahng, M., Thorat, N., Chau, D.H., Viégas, F.B., & Wattenberg, M. (2018). IEEE Transactions on Visualization and Computer Graphics, 25, 310-320.*
#### Reinforcement Learning
- t-SNE
	>*[Graying the black box: Understanding DQNs.](https://arxiv.org/pdf/1602.02658.pdf) Zahavy, T., Baram, N., and Mannor, S. International Conference on Machine Learning, pp. 1899–1908, 2016.*
	
	using SAMDPs to analyze high-level policy behavior
- Saliency maps
	>*[Visualizing and Understanding Atari Agents.](https://arxiv.org/abs/1711.00138) Greydanus, S., Koul, A., Dodge, J., & Fern, A. (2017).  ICML, abs/1711.00138.*

	how inputs influence individual decisions using saliency maps
	
	<p align="center"><img width="50%" height="50%" src="images/atari.png?raw=true" /></p>
	
- Entropy
	>*[Establishing appropriate trust via critical states.](https://arxiv.org/abs/1810.08174) S. H. Huang, K. Bhatia, P. Abbeel, and A. D. Dragan.International Conference on Intelligent Robots (IROS), 2018*

	finds critical states of an agent based on the entropy of the output of a policy. 
- AM
	>*[Finding and Visualizing Weaknesses of Deep Reinforcement Learning Agents. ](https://arxiv.org/abs/1904.01318v1) Rupprecht, C., Ibrahim, C., & Pal, C.J. (2019). ArXiv, abs/1904.01318.*

 	using activation maximization methods for visualization.
- DQNViz
	>*[DQNViz: A Visual Analytics Approach to Understand Deep Q-Networks.](https://ieeexplore.ieee.org/document/8454905) Wang, J., Gou, L., Shen, H., & Yang, H.T. (2018). IEEE VAST, Transactions on Visualization and Computer Graphics(honorable mention), 25, 288-298.*
	
	Extract useful action/reward patterns that help to interpret the model and control the training
	
	<p align="center"><img width="50%" height="50%" src="images/dqn.png?raw=true" /></p>
	
	
