# Exploring Hyperbole Research in NLP

**Author:** Ali Sartaz Khan  
**Supervisor:** Ellen Riloff  
**Affiliation:** Department of Computer Science, University of Arizona  

## 1. Hyperbole-related Tasks

### 1.1 Detection and Analysis

#### 1.1.1 Identifying Exaggerated Language
We utilize a variety of NLP models, including BERT, LSTM, and CNNs, to identify hyperbolic statements across different datasets. These efforts are crucial for enhancing language understanding models so that they can correctly interpret exaggerations without taking them literally.

#### 1.1.2 Harnessing Privileged Information
Improving model performance by incorporating contextual insights, such as literal paraphrases, to better differentiate between exaggerated and literal statements.

### 1.2 Multitask Learning

#### 1.2.1 A Multi-task Framework
Training models like RoBERTa to simultaneously detect hyperbole and metaphor shows how leveraging multitask learning frameworks can enhance the detection capabilities of NLP systems.

### 1.3 Hyperbole Generation

#### 1.3.1 Innovative Text Generation
Exploring creative applications of hyperbole generation using techniques like over-generation and ranking contributes to more sophisticated NLP tools capable of producing or transforming hyperbolic content.

### 1.4 Miscellaneous Studies

#### 1.4.1 Detecting Sarcasm and Hyperbole in Social Media
The first few studies focus on detecting sarcasm and hyperbole in social media platforms like Twitter. The study "Signaling sarcasm: From hyperbole to hashtag" investigates the use of hashtags to mark sarcasm in Dutch tweets (Liebrecht, Kunneman, and van den Bosch, 2013), while "Features and Categories of Hyperbole in Cyberbullying Discourse on Social Media" analyzes the linguistic features of hyperbole in cyberbullying utterances on social media (Akoury et al., 2022).

These studies highlight the importance of understanding sarcasm and hyperbole in online communication, as they can convey implicit meanings and nuances that might be challenging to detect using traditional NLP techniques.

#### 1.4.2 Exaggeration in Science Communication
Several studies address the issue of exaggeration in science communication, particularly in how scientific findings are represented in the media. The "NLP Analysis of Exaggerated Claims in Science News" and "Semi-Supervised Exaggeration Detection of Health Science Press Releases" papers focus on detecting and mitigating the distortion of scientific findings in media reports (Sumner et al., 2014; Al-Hity and Islam, 2021).

Additionally, "A Simple Three-Step Approach for the Automatic Detection of Exaggerated Statements in Health Science News" proposes a three-step method for automatically detecting exaggerated statements in media coverage of scientific research (Braverman, John, and Roberts, 2021).

#### 1.4.3 Probing Hyperbole in Pre-Trained Language Models
The study "Probing for Hyperbole in Pre-Trained Language Models" takes a different approach by investigating how hyperbolic information is encoded in pre-trained language models (PLMs). The researchers use edge and minimal description length (MDL) probing experiments to explore the representations of hyperbole within these models (Akoury et al., 2023).

This research area contributes to our understanding of how language models handle figurative language and could potentially lead to improvements in tasks such as sarcasm detection, sentiment analysis, and natural language generation.

## 2. Overview of Datasets for Hyperbole Detection

### 2.1 HYPO-cn
This dataset was created specifically for studying hyperbole in the Chinese language. It consists of 4,762 sentences, split into 2,680 hyperbolic and 2,082 non-hyperbolic. The dataset was annotated by having experts create hyperbolic versions of non-hyperbolic sentences and vice versa (Kong et al., 2020).

### 2.2 HYPO
This dataset includes 709 hyperbolic utterances, 709 literal paraphrases, and 709 non-hyperbolic sentences, totaling 2,127 sentences. The HYPO dataset provides a rich set of examples sourced from various contexts, such as headlines and cartoons, labeled as either hyperbolic or non-hyperbolic (Troiano et al., 2018).

### 2.3 HYPO-L
Part of the broader HYPO initiative, this dataset is used for advanced text generation experiments, including hyperbole generation. It is a more recent addition to the HYPO datasets, aiding in the development of models that can generate exaggerated text (Zhang and Wan, 2022).

### 2.4 TroFi
This dataset focuses on figurative language and includes a list of words categorized into literal and nonliteral clusters. It is used to study the recognition of nonliteral language through unsupervised learning methods (Birke and Sarkar, 2006).

### 2.5 LCC
This dataset includes metaphor annotations and is used to study the broader context of figurative language, which encompasses hyperbole as well (Mohler et al., 2016).

## 3. In-depth Analysis of the HYPO Dataset

The HYPO dataset stands out for its comprehensive design, which facilitates both the detection and understanding of hyperboles in natural language.

### 3.1 Dataset Composition
The HYPO dataset consists of 709 hyperbolic utterances sourced from diverse mediums like headlines and cartoons. It also includes equal numbers of literal paraphrases and non-hyperbolic sentences using the same words, totaling 2,127 sentences. This design helps contrast hyperbolic statements against their non-hyperbolic counterparts to better train models on distinguishing exaggerated language.

## 4. Experiment and Results

For the task of classifying hyperbolic and non-hyperbolic statements, a BERT-based model was developed and refined using the HYPO dataset (Troiano et al., 2018). The model architecture consisted of a pre-trained BERT model followed by a linear classification layer on top.

The dataset was split into training, development, and test sets using a specific approach. 20% of the data was randomly pulled out as the test set, 10% was selected as the development set, and the remaining 70% constituted the training set. Importantly, all columns (hyperbolic sentence, literal paraphrase, and non-hyperbolic sentence) were kept together in the same split to prevent information leakage.

### 4.1 Performance Metrics

| Metric            | Hyperbole (%) | Non-Hyperbole (%) |
|-------------------|---------------|-------------------|
| **Precision**     | 82.86         | 76.57             |
| **Recall**        | 41.43         | 95.71             |
| **F-1 Score**     | 55.24         | 85.08             |

These metrics provide a comprehensive view of the model's performance. The precision metric indicates that the model is more reliable in identifying true hyperbolic statements. However, the recall for hyperbolic statements stands at 41.43%, which suggests challenges in detecting subtler forms of exaggeration.

## 5. Evaluation

### 5.1 Strengths
The model is effective in detecting non-hyperbolic statements with high reliability.

### 5.2 Weaknesses
Detecting hyperbolic statements presents challenges, particularly in recognizing less obvious exaggerations, as reflected by the lower recall rate.

## 6. Future Directions

### 6.1 Data Augmentation
Incorporating more diverse examples of hyperbole, including subtler forms, might train the model to recognize a broader range of exaggerations.

### 6.2 Model Tuning
Adjusting model parameters or exploring more complex architectures could enhance sensitivity to nuanced language features typical of hyperbole.

### 6.3 Multitask Learning
Multitasking learning could aid the model by simultaneously learning related tasks, such as sarcasm detection, which shares linguistic characteristics with hyperbole.

## 7. References

- Troiano, E., Strapparava, C., Özbal, G., & Tekiroğlu, S. S. A computational exploration of exaggeration. 2018. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing (pp. 3296-3304).
- Liebrecht, C., Kunneman, F., and van den Bosch, A. 2013. "The Perfect Solution for Detecting Sarcasm in Texts: Machine Learning with Word Embeddings and Linguistic Knowledge." arXiv preprint arXiv:1312.5056.
- Sumner, P., et al. 2014. "The Association between Exaggeration in Health Related Science News and Academic Press Releases: Retrospective Observational Study." BMJ, vol. 349.
- Al-Hity, Z., and Islam, M. R. 2021. "Semi-Supervised Exaggeration Detection of Health Science Press Releases." arXiv preprint arXiv:2108.13493.
- Braverman, M., John, V., and Roberts, K. 2021. "A Simple Three-Step Approach for the Automatic Detection of Exaggerated Statements in Health Science News." Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pp. 10518-10529.
- Akoury, N., Weston, J., Yokoi, S., and Zhang, M. 2023. "Probing for Hyperbole in Pre-Trained Language Models." arXiv preprint arXiv:2305.08126.
- Akoury, N., Al-Dohki, M., Zhang, Y., and Tekiroglu, S. S. 2022. "Features and Categories of Hyperbole in Cyberbullying Discourse on Social Media." Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 5789-5803.
- Yunxiang Zhang and Xiaojun Wan. 2022. MOVER: Mask, over-generate and rank for hyperbole generation. In Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 6018–6030, Seattle, United States. Association for Computational Linguistics.
- Julia Birke and Anoop Sarkar. 2006. A clustering approach for nearly unsupervised recognition of nonliteral language. In 11th Conference of the European Chapter of the Association for Computational Linguistics, pages 329–336, Trento, Italy. Association for Computational Linguistics.
- Michael Mohler, Mary Brunson, Bryan Rink, and Marc Tomlinson. 2016. Introducing the LCC metaphor datasets. In Proceedings of the Tenth International Conference on Language Resources and Evaluation (LREC’16), pages 4221–4227, Portorož, Slovenia. European Language Resources Association (ELRA).
- Kong, Li, et al. "Identifying Exaggerated Language." Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), 2020, pp. 7024-7034.
