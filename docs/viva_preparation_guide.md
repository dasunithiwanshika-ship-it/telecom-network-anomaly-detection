# Viva Voce & Technical Defense Preparation Guide
## Frequently Asked Questions & Defensible Answers

**Project:** AI-Based Network Anomaly & Fault-Risk Detection for Telecom Networks  
**Dataset:** Hugging Face `AliMaatouk/TelecomTS`  

---

### Q1: Why did you summarize 128 timesteps into 4 statistical moments rather than feeding raw time series into an LSTM or CNN?
**Answer:**  
In cellular operations, computing 4 summary statistics ($\mu, \sigma, \min, \max$) across short observation windows (~12.7 seconds) provides an ultra-low-latency tabular representation ($<0.2\text{ ms}$ inference) suitable for real-time edge or base-station telemetry. It creates scale-invariant features for decision trees and establishes a clean, interpretable baseline before introducing more complex, computationally expensive sequential deep learning architectures.

---

### Q2: Why is accuracy an unsuitable metric for this research?
**Answer:**  
The TelecomTS dataset is heavily imbalanced: 96.14% of the traces are normal, and only 3.86% are anomalous. A trivial dummy model predicting "Normal" for every trace achieves 96.14% accuracy while failing to detect a single network fault. Therefore, we evaluate using Precision-Recall Area Under the Curve (PR-AUC), F1-score, and precision/recall, which directly penalize false alarms and missed anomalies on the minority class.

---

### Q3: Why does Random Forest achieve 100% precision with zero false positives on the test set while Logistic Regression produced 9 false alarms?
**Answer:**  
Logistic Regression is a linear generalized linear model that forms an additive linear decision boundary $\mathbf{w}^T\mathbf{x} + b$. In traces with heavy but legitimate user traffic (e.g., high PRB utilization and large byte volume), positive additive contributions push the linear dot product past the threshold, triggering false alarms. Random Forest partitions feature space orthogonally using hierarchical decision thresholds, enabling it to distinguish high-traffic healthy connections from genuine faults where high PRB utilization is accompanied by degraded SNR and elevated BLER.

---

### Q4: Why did unsupervised models (Isolation Forest and Local Outlier Factor) perform so poorly (F1 < 30%) compared to Random Forest (F1 = 96.44%)?
**Answer:**  
Unsupervised anomaly detectors operate on geometric isolation or density in feature space without label guidance. In cellular networks, legitimate user behaviors (e.g. video streaming, massive file downloads) create sparse, high-dimensional observations that look like "outliers" to an unsupervised model, generating 174–193 false alarms. Furthermore, many genuine network faults (such as handover oscillation or gradual buffer buildup) occur within the normal operating range of individual KPIs and are only distinguishable through specific multi-variable combinations that require supervised labels to learn.

---

### Q5: How did you ensure a fair comparison between supervised and unsupervised models?
**Answer:**  
First, unsupervised models were fitted strictly on unlabelled training data ($X_{\text{train}}$) without any access to ground-truth labels ($y$). Second, all four models were evaluated on the exact same reserved 80/20 test split ($N=6,400$). Third, unsupervised decision scores were inverted so that higher values consistently represent greater anomaly likelihood, enabling identical ROC and Precision-Recall curve derivations across all four paradigms.

---

### Q6: Can you explain Sample 1047? Why did Logistic Regression detect it while Random Forest missed it?
**Answer:**  
Sample 1047 is an uplink buffer overflow anomaly during a Twitch streaming session. The trace had an elevated buffer (`Estimated_UL_Buffer_mean = 708.01 KB`), placing it in the 86th percentile of buffer sizes. Because Logistic Regression assigned a large standardized positive coefficient ($\beta \approx +3.65$) to buffer size, this single elevated feature pushed the linear probability to $P=1.0000$. In Random Forest, exactly 40 out of 100 trees voted anomaly, yielding an ensemble probability of $\hat{P}=0.4000$. Because $0.4000 < 0.50$ (the default decision threshold), the ensemble classified it as Normal. Lowering the threshold to $\tau \le 0.40$ allows Random Forest to capture Sample 1047 as well.

---

### Q7: What are the 16 mutually missed anomalies ($\mathcal{FN}_{\text{both}} = 16$)?
**Answer:**  
These 16 traces correspond primarily to rapid temporal anomalies, such as `Faulty Handover Algorithm (Too Frequent)`. Because our feature engineering compresses 128 timesteps into static summary moments ($\mu, \sigma, \min, \max$), the sequential ordering of rapid ping-pong handover transitions is lost. The summary statistics of these traces overlap almost completely with normal operating ranges, making them undetectable from windowed summary moments alone.

---

### Q8: What are the limitations regarding SLTMobitel?
**Answer:**  
We must be completely clear: this prototype was developed and evaluated strictly using the public `AliMaatouk/TelecomTS` cellular benchmark. It does NOT use proprietary SLTMobitel operational telemetry, does not represent Sri Lankan telecommunications infrastructure, and cannot predict physical DP/FDP cable cuts or dispatch physical repair crews. It is an algorithmic research study on cellular KPI anomaly detection.

---

### Q9: How was the contamination parameter in Isolation Forest chosen, and is there risk of test-set data leakage?
**Answer:**  
In scikit-learn, Isolation Forest tree fitting is strictly unsupervised: trees are grown by random orthogonal splits on $X_{\text{train}}$ without labels. Contamination only sets the percentile cutoff for the discrete `predict()` output.
We evaluated two settings:
1. **Pure Unsupervised Baseline (`contamination='auto'`)**: Assumes zero prior knowledge of ground truth, yielding Precision: 18.07%, Recall: 34.82%, F1: 23.79%.
2. **Informed Training Prior (`contamination=0.0386`)**: Set strictly from the training set prevalence ($y_{\text{train}} = 988 / 25,600 = 3.86\%$) without touching test labels, yielding Precision: 26.05%, Recall: 27.53%, F1: 26.77%.
Crucially, ranking metrics (ROC-AUC = 76.12%, PR-AUC = 21.21%) evaluate the inverted decision score directly and are completely invariant to the contamination parameter.

---

### Q10: How was the Random Forest decision threshold of 0.30 chosen, and why does canonical 0.50 remain the primary benchmark?
**Answer:**  
Reporting a threshold selected by searching on the test set is a classic form of data snooping (optimism bias). To ensure strict methodological validity:
1. Our primary unbiased benchmark is the canonical threshold $\tau = 0.50$ (Precision: 100.00%, Recall: 93.12%, F1: 96.44%).
2. To explore threshold optimization, we ran 5-Fold Stratified Cross-Validation strictly on the training set ($X_{\text{train}}$). Out-of-fold validation F1 peaked at $\tau = 0.30$ (Validation F1 = 97.08%).
3. When evaluated once on the unseen test set, this CV-selected threshold ($\tau = 0.30$) achieved Precision: 99.17%, Recall: 97.17%, and F1: 98.16% (recovering 240 out of 247 faults with only 2 false positives).
This demonstrates operational flexibility without test-set leakage.

---

### Q11: Why does Permutation Importance rank `UL_NPRB_min` as #1 while Gini MDI ranks `RSRP_mean` as #1?
**Answer:**  
1. **Gini MDI (Mean Decrease in Impurity)** is computed during training across tree splits. Continuous features with high granularity (like `RSRP_mean` and `UL_SNR_mean`) offer numerous candidate thresholds and naturally accumulate high split counts near the root, creating a well-known cardinality bias.
2. **Permutation Importance** was evaluated on held-out test data without refitting the model. It measures the direct degradation in classification accuracy when a feature column is randomly shuffled. Shuffling `UL_NPRB_min` severely breaks the orthogonal decision boundaries used by the forest to distinguish uplink starvation and jamming anomalies, causing the largest accuracy drop ($\approx 1.88\%$). Both metrics agree that uplink channel and resource allocation features dominate detection.
