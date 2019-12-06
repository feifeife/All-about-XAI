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
- anchor
>*[Anchors: High-precision model-agnostic explanations]() M. T. Ribeiro, S. Singh, and C. Guestrin, in Proc. AAAI Conf. Artif. Intell., 2018.*
- LOCO
>*[Distribution-free predictive inference for regression](http://www.stat.cmu.edu/~ryantibs/papers/conformal.pdf) J. Lei, M. G’Sell, A. Rinaldo, R. J. Tibshirani, and L.Wasserman.*
#### Model Specific
##### CNN
##### RNN
