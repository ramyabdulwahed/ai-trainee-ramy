Day 3 Case Study: TF-IDF Baseline vs. DistilBERT Transformer
Summary

For Day 3, I fine-tuned distilbert-base-uncased on the IMDB sentiment classification task and compared it against my Day 2 classical NLP baseline using TF-IDF features with Logistic Regression. The goal was to see whether a transformer-based model could improve over the simpler baseline while also understanding the tradeoffs in training time, model size, inference speed, and deployment complexity.

Metric	TF-IDF + Logistic Regression (Day 2)	DistilBERT (Day 3)
Accuracy	TODO	0.8985
F1 (macro)	TODO	0.8985
Training time	TODO	132.87 seconds
Peak GPU memory	—	TODO
Disk footprint (model)	TODO	TODO
Inference latency (100 samples)	TODO	TODO


The DistilBERT run completed successfully for 2 epochs with a final validation accuracy of 0.8985 and macro F1 of 0.8985. The training runtime was about 132.87 seconds. The training loss decreased from 0.2734 after epoch 1 to 0.2253 after epoch 2, showing that the model continued learning on the training data. However, validation loss increased from 0.2633 to 0.2974, which suggests the model may have started to overfit slightly by the second epoch.

Compared to the TF-IDF baseline, DistilBERT is more powerful because it can use word context and sentence structure rather than mainly relying on weighted word and phrase counts. For example, a transformer should be better at understanding phrases where meaning depends on context, such as negation or mixed opinions. However, in this run, the DistilBERT score was only around 89.85%, which is close to the Day 2 baseline. This shows an important practical lesson: a transformer is not automatically better in every setup. Dataset size, max sequence length, training settings, runtime limits, and evaluation design all matter.

I would still choose TF-IDF with Logistic Regression in production when latency, simplicity, cost, or interpretability matter more than maximum accuracy. TF-IDF is much smaller, faster on CPU, easier to deploy, and easier to explain. It can be a better choice for simple classification tasks, small datasets, strict latency requirements, or systems where GPU access is unavailable. I would choose DistilBERT when the task needs stronger language understanding, when context matters, and when the extra model size and compute cost are justified by better performance.