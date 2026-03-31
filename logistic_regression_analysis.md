# Logistic Regression — Phishing Website Detection

> **Implementation:** From scratch using only Python built-in modules (`csv`, `math`, `random`)
> **Dataset:** [dataset_processed.csv](file:///Users/apoorvpal/Desktop/ML-COURSE%20PROJECT-1/dataset_processed.csv) | **Code:** [logistic_regression.py](file:///Users/apoorvpal/Desktop/ML-COURSE%20PROJECT-1/logistic_regression.py) | **Full Report:** [logistic_regression_report.txt](file:///Users/apoorvpal/Desktop/ML-COURSE%20PROJECT-1/logistic_regression_report.txt)

---

## 1. Dataset Overview

| Property | Value |
|---|---|
| Total Samples | 11,430 |
| Features | 48 (URL-based + web page-based) |
| Target Variable | `status` — binary (legitimate / phishing) |
| Legitimate (Class 0) | 5,715 (50.0%) |
| Phishing (Class 1) | 5,715 (50.0%) |
| Training Set | 9,144 samples (80%) |
| Test Set | 2,286 samples (20%) |
| Missing Values | 0 |

> [!NOTE]
> The dataset is perfectly balanced (50/50 split), so no class imbalance handling was needed.

---

## 2. Model Configuration

| Parameter | Value |
|---|---|
| Algorithm | Logistic Regression (Binary Classification) |
| Optimization | Batch Gradient Descent |
| Learning Rate | 0.1 |
| Iterations | 1000 |
| Feature Scaling | Z-Score Standardization |
| Train/Test Split | 80/20 (random seed = 42) |
| Bias (Intercept) Term | 0.414063 |
| External Libraries | **None** — implemented entirely from scratch |

---

## 3. Confusion Matrix

```
                    Predicted
                Legitimate    Phishing
              ┌────────────┬────────────┐
Actual Legit  │    1089     │     66     │
              ├────────────┼────────────┤
Actual Phish  │     67      │    1064    │
              └────────────┴────────────┘
```

| Component | Count | Meaning |
|---|---|---|
| **TP** (True Positives) | 1,064 | Phishing sites correctly identified |
| **TN** (True Negatives) | 1,089 | Legitimate sites correctly identified |
| **FP** (False Positives) | 66 | Legitimate sites wrongly flagged as phishing |
| **FN** (False Negatives) | 67 | Phishing sites missed (went undetected) |

---

## 4. Primary Performance Metrics

| Metric | Value | Percentage | Formula |
|---|---|---|---|
| **Accuracy** | 0.9418 | **94.18%** | (TP + TN) / Total |
| **Error Rate** | 0.0582 | 5.82% | (FP + FN) / Total |
| **Precision (PPV)** | 0.9416 | 94.16% | TP / (TP + FP) |
| **Recall (Sensitivity / TPR)** | 0.9408 | 94.08% | TP / (TP + FN) |
| **F1-Score** | 0.9412 | 94.12% | 2 × (Prec × Rec) / (Prec + Rec) |
| **Specificity (TNR)** | 0.9429 | 94.29% | TN / (TN + FP) |
| **NPV** | 0.9420 | 94.20% | TN / (TN + FN) |
| **ROC-AUC** | 0.9850 | **98.50%** | Area under ROC Curve |

> [!IMPORTANT]
> The model achieves **94.18% accuracy** with a **98.50% ROC-AUC**, indicating excellent discriminative ability between phishing and legitimate websites.

---

## 5. Advanced Performance Metrics

| Metric | Value | Interpretation |
|---|---|---|
| **Balanced Accuracy** | 0.9418 | Average of sensitivity and specificity — robust for imbalanced data |
| **Matthews Correlation Coefficient (MCC)** | 0.8836 | Best single metric for binary classification quality (range: -1 to +1) |
| **Cohen's Kappa** | 0.8836 | Agreement beyond chance — "Almost Perfect" (>0.81) |
| **Youden's J Statistic** | 0.8836 | Informedness = Sensitivity + Specificity − 1 |
| **Gini Coefficient** | 0.9700 | 2×AUC − 1; measures discriminatory power |
| **Log-Loss (Cross-Entropy)** | 0.1545 | Lower is better; measures prediction confidence |
| **Diagnostic Odds Ratio (DOR)** | 262.03 | Ratio of correct-to-incorrect odds; higher = better |
| **Positive Likelihood Ratio (LR+)** | 16.4633 | How much more likely a positive test is from a phishing site |
| **Negative Likelihood Ratio (LR−)** | 0.0628 | How much less likely a negative test is from a phishing site |

### Metric Interpretation Guide

| MCC / Kappa Range | Quality |
|---|---|
| 0.81 – 1.00 | **Almost Perfect** ✅ |
| 0.61 – 0.80 | Substantial |
| 0.41 – 0.60 | Moderate |
| 0.21 – 0.40 | Fair |
| < 0.20 | Poor |

> Our MCC = **0.8836** falls in the "Almost Perfect" range.

---

## 6. Error Analysis

| Error Type | Rate | Percentage | Meaning |
|---|---|---|---|
| **False Positive Rate (FPR)** — Type I Error | 0.0571 | 5.71% | Legitimate sites wrongly flagged |
| **False Negative Rate (FNR)** — Type II Error | 0.0592 | 5.92% | Phishing sites that slipped through |
| **Prevalence** | 0.4948 | 49.48% | Proportion of phishing in test set |
| **Detection Rate** | 0.4654 | 46.54% | Proportion of total correctly detected as phishing |
| **Detection Prevalence** | 0.4943 | 49.43% | Proportion predicted as phishing |

> [!TIP]
> Both error rates (FPR & FNR) are below 6%, meaning the model is well-calibrated and not biased toward either class.

---

## 7. Per-Class Classification Report

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| **legitimate** | 0.9420 | 0.9429 | 0.9424 | 1,155 |
| **phishing** | 0.9416 | 0.9408 | 0.9412 | 1,131 |
| **macro avg** | 0.9418 | 0.9418 | 0.9418 | 2,286 |
| **weighted avg** | 0.9418 | 0.9418 | 0.9418 | 2,286 |

> Macro and weighted averages are virtually identical because the dataset is balanced.

---

## 8. Overfitting Check — Train vs Test

| Metric | Train | Test | Difference |
|---|---|---|---|
| **Accuracy** | 0.9453 | 0.9418 | 0.0035 |
| **ROC-AUC** | 0.9858 | 0.9850 | 0.0008 |

> [!NOTE]
> The train-test gap is minimal (~0.35% accuracy, ~0.08% AUC), confirming **no significant overfitting**. The model generalizes well to unseen data.

---

## 9. Training Convergence (Loss Curve)

| Iteration | Loss (Cross-Entropy) |
|---|---|
| 1 | 0.693147 |
| 101 | 0.194960 |
| 201 | 0.173054 |
| 301 | 0.164493 |
| 401 | 0.159847 |
| 501 | 0.156910 |
| 601 | 0.154881 |
| 701 | 0.153396 |
| 801 | 0.152262 |
| 901 | 0.151372 |
| 1000 | 0.150662 |

```
Loss Curve (text visualization):
Iter    1 │ ████████████████████████████████████████ │ 0.693147
Iter  101 │ ███████████                              │ 0.194960
Iter  201 │ █████████                                │ 0.173054
Iter  301 │ █████████                                │ 0.164493
Iter  401 │ █████████                                │ 0.159847
Iter  501 │ █████████                                │ 0.156910
Iter  601 │ ████████                                 │ 0.154881
Iter  701 │ ████████                                 │ 0.153396
Iter  801 │ ████████                                 │ 0.152262
Iter  901 │ ████████                                 │ 0.151372
```

- **Initial Loss:** 0.693147 (random guessing — log(2))
- **Final Loss:** 0.150662
- **Total Reduction:** **78.26%**

---

## 10. Feature Importance Analysis (Top 15)

| Rank | Feature | Weight | Direction |
|---|---|---|---|
| 1 | `google_index` | +1.3218 | → phishing |
| 2 | `page_rank` | −1.0842 | → legitimate |
| 3 | `phish_hints` | +1.0709 | → phishing |
| 4 | `nb_www` | −0.9444 | → legitimate |
| 5 | `nb_hyperlinks` | −0.7838 | → legitimate |
| 6 | `nb_hyphens` | −0.7668 | → legitimate |
| 7 | `domain_age` | −0.6872 | → legitimate |
| 8 | `nb_qm` | +0.5685 | → phishing |
| 9 | `ratio_digits_host` | +0.5530 | → phishing |
| 10 | `empty_title` | −0.4999 | → legitimate |
| 11 | `web_traffic` | −0.4830 | → legitimate |
| 12 | `longest_words_raw` | +0.4332 | → phishing |
| 13 | `length_hostname` | +0.3988 | → phishing |
| 14 | `shortening_service` | +0.3773 | → phishing |
| 15 | `ip` | +0.3679 | → phishing |

### Key Findings:
- **`google_index`** is the #1 predictor — not being indexed by Google strongly indicates phishing
- **`page_rank`** is the strongest negative predictor — higher PageRank means legitimate
- **`phish_hints`** in URLs (words like "login", "verify", "secure") strongly predict phishing
- **`domain_age`** — newer domains tend to be phishing sites
- **`shortening_service`** usage and **IP addresses** in URLs are phishing indicators

---

## 11. Complete Feature Weights (All 48)

| Rank | Feature | Weight |
|---|---|---|
| 1 | google_index | +1.321816 |
| 2 | page_rank | −1.084232 |
| 3 | phish_hints | +1.070787 |
| 4 | nb_www | −0.944355 |
| 5 | nb_hyperlinks | −0.783803 |
| 6 | nb_hyphens | −0.766791 |
| 7 | domain_age | −0.687195 |
| 8 | nb_qm | +0.568493 |
| 9 | ratio_digits_host | +0.552975 |
| 10 | empty_title | −0.499899 |
| 11 | web_traffic | −0.483044 |
| 12 | longest_words_raw | +0.433170 |
| 13 | length_hostname | +0.398776 |
| 14 | shortening_service | +0.377301 |
| 15 | ip | +0.367862 |
| 16 | domain_in_title | +0.355118 |
| 17 | https_token | −0.318367 |
| 18 | dns_record | +0.314419 |
| 19 | nb_semicolumn | +0.290555 |
| 20 | nb_at | +0.287107 |
| 21 | nb_dots | +0.267314 |
| 22 | avg_word_path | +0.264649 |
| 23 | safe_anchor | −0.250878 |
| 24 | nb_slash | +0.231896 |
| 25 | domain_with_copyright | −0.220940 |
| 26 | longest_word_host | −0.201217 |
| 27 | tld_in_subdomain | +0.199568 |
| 28 | ratio_extMedia | −0.179339 |
| 29 | ratio_extRedirection | −0.169292 |
| 30 | ratio_intHyperlinks | −0.151633 |
| 31 | http_in_path | +0.145449 |
| 32 | nb_subdomains | −0.141380 |
| 33 | shortest_word_path | +0.133344 |
| 34 | external_favicon | +0.127940 |
| 35 | shortest_word_host | −0.126024 |
| 36 | statistical_report | +0.102810 |
| 37 | tld_in_path | −0.076805 |
| 38 | prefix_suffix | +0.057776 |
| 39 | whois_registered_domain | −0.057452 |
| 40 | ratio_intMedia | −0.045823 |
| 41 | nb_extCSS | −0.042206 |
| 42 | domain_in_brand | −0.041299 |
| 43 | abnormal_subdomain | +0.041198 |
| 44 | length_url | +0.038313 |
| 45 | domain_registration_length | +0.033619 |
| 46 | nb_com | +0.025147 |
| 47 | nb_colon | −0.019604 |
| 48 | nb_and | +0.007604 |

---

## 12. Summary of All Performance Measures

| # | Metric | Value |
|---|---|---|
| 1 | Accuracy | 0.9418 (94.18%) |
| 2 | Error Rate (Misclassification) | 0.0582 (5.82%) |
| 3 | Precision (Phishing) | 0.9416 |
| 4 | Recall / Sensitivity / TPR | 0.9408 |
| 5 | F1-Score (Phishing) | 0.9412 |
| 6 | Specificity / TNR | 0.9429 |
| 7 | Negative Predictive Value (NPV) | 0.9420 |
| 8 | ROC-AUC | 0.9850 |
| 9 | Balanced Accuracy | 0.9418 |
| 10 | Matthews Correlation Coefficient | 0.8836 |
| 11 | Cohen's Kappa | 0.8836 |
| 12 | Youden's J Statistic | 0.8836 |
| 13 | Gini Coefficient | 0.9700 |
| 14 | Log-Loss (Cross-Entropy) | 0.1545 |
| 15 | Diagnostic Odds Ratio (DOR) | 262.03 |
| 16 | Positive Likelihood Ratio (LR+) | 16.4633 |
| 17 | Negative Likelihood Ratio (LR−) | 0.0628 |
| 18 | False Positive Rate (Type I) | 0.0571 |
| 19 | False Negative Rate (Type II) | 0.0592 |
| 20 | Prevalence | 0.4948 |
| 21 | Detection Rate | 0.4654 |
| 22 | Detection Prevalence | 0.4943 |
| 23 | Macro Avg Precision | 0.9418 |
| 24 | Macro Avg Recall | 0.9418 |
| 25 | Macro Avg F1 | 0.9418 |
| 26 | Weighted Avg Precision | 0.9418 |
| 27 | Weighted Avg Recall | 0.9418 |
| 28 | Weighted Avg F1 | 0.9418 |

---

## 13. Conclusion

The from-scratch Logistic Regression model demonstrates **strong performance** across all metrics for phishing website detection:

- **94.18% accuracy** with balanced precision/recall across both classes
- **98.50% ROC-AUC** indicates near-excellent separability
- **MCC of 0.8836** confirms "Almost Perfect" classification quality
- **No overfitting** detected (train-test gap < 0.4%)
- The model correctly identifies both phishing and legitimate sites with ~94% success rate
- Only **133 misclassifications** out of 2,286 test samples (66 false alarms + 67 missed phishing)
