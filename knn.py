"""
K-Nearest Neighbors (KNN) for Phishing Website Detection
===================================================
Implemented FROM SCRATCH — no external libraries (no numpy, pandas, sklearn, etc.)
Only uses Python built-in modules: csv, math, random
"""

import csv
import math
import random

# ══════════════════════════════════════════════════════════════
# HELPER FUNCTIONS (replacing numpy/sklearn)
# ══════════════════════════════════════════════════════════════

def mean(lst):
    """Mean of a list."""
    return sum(lst) / len(lst)


def std_dev(lst):
    """Standard deviation of a list."""
    m = mean(lst)
    variance = sum((x - m) ** 2 for x in lst) / len(lst)
    return math.sqrt(variance) if variance > 0 else 1.0


def standardize(X_train, X_test):
    """Standardize features (z-score normalization) using train stats."""
    n_features = len(X_train[0])
    means = []
    stds = []
    for j in range(n_features):
        col = [row[j] for row in X_train]
        m = mean(col)
        s = std_dev(col)
        means.append(m)
        stds.append(s if s > 0 else 1.0)

    X_train_scaled = []
    for row in X_train:
        X_train_scaled.append([(row[j] - means[j]) / stds[j] for j in range(n_features)])

    X_test_scaled = []
    for row in X_test:
        X_test_scaled.append([(row[j] - means[j]) / stds[j] for j in range(n_features)])

    return X_train_scaled, X_test_scaled, means, stds


def train_test_split(X, y, test_size=0.2, random_seed=42):
    """Stratified-ish train/test split."""
    random.seed(random_seed)
    indices = list(range(len(X)))
    random.shuffle(indices)

    split_idx = int(len(X) * (1 - test_size))

    X_train = [X[i] for i in indices[:split_idx]]
    X_test = [X[i] for i in indices[split_idx:]]
    y_train = [y[i] for i in indices[:split_idx]]
    y_test = [y[i] for i in indices[split_idx:]]

    return X_train, X_test, y_train, y_test


# ══════════════════════════════════════════════════════════════
# KNN CLASS (from scratch)
# ══════════════════════════════════════════════════════════════

class KNNScratch:
    def __init__(self, k=5):
        self.k = k
        self.X_train = []
        self.y_train = []

    def fit(self, X, y):
        """Fit simply stores the training data in KNN."""
        self.X_train = X
        self.y_train = y

    def _euclidean_distance_sq(self, row1, row2):
        """Calculates the squared Euclidean distance."""
        return sum((x1 - x2) ** 2 for x1, x2 in zip(row1, row2))

    def predict_proba(self, X):
        """Return probability of class 1."""
        probs = []
        n_test = len(X)
        for i, test_row in enumerate(X):
            distances = []
            for j, train_row in enumerate(self.X_train):
                dist_sq = self._euclidean_distance_sq(test_row, train_row)
                distances.append((dist_sq, self.y_train[j]))
            
            # Sort by squared distance
            distances.sort(key=lambda x: x[0])
            
            # Get top k
            top_k = distances[:self.k]
            pos_count = sum(1 for _, label in top_k if label == 1)
            probs.append(pos_count / self.k)
            
            # Print progress
            if (i + 1) % 100 == 0 or i == 0 or (i + 1) == n_test:
                print(f"   Predicted {i + 1:>4d}/{n_test} samples...", end="\r")
        print() # Newline after progress
        return probs

    def predict(self, X, threshold=0.5):
        """Return class predictions (0 or 1)."""
        probs = self.predict_proba(X)
        return [1 if p >= threshold else 0 for p in probs]


# ══════════════════════════════════════════════════════════════
# EVALUATION METRICS (from scratch)
# ══════════════════════════════════════════════════════════════

def confusion_matrix(y_true, y_pred):
    """Returns TP, TN, FP, FN."""
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)
    return tp, tn, fp, fn


def accuracy(y_true, y_pred):
    correct = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
    return correct / len(y_true)


def precision(tp, fp):
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def recall(tp, fn):
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def f1_score(prec, rec):
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0


def roc_auc(y_true, y_prob):
    """Compute ROC-AUC using the trapezoidal rule."""
    # Create (prob, label) pairs and sort by prob descending
    pairs = list(zip(y_prob, y_true))
    pairs.sort(key=lambda x: -x[0])

    tp = 0
    fp = 0
    total_pos = sum(y_true)
    total_neg = len(y_true) - total_pos

    if total_pos == 0 or total_neg == 0:
        return 0.5

    tpr_list = [0.0]
    fpr_list = [0.0]

    for prob, label in pairs:
        if label == 1:
            tp += 1
        else:
            fp += 1
        tpr_list.append(tp / total_pos)
        fpr_list.append(fp / total_neg)

    # Trapezoidal rule
    auc = 0.0
    for i in range(1, len(fpr_list)):
        auc += (fpr_list[i] - fpr_list[i - 1]) * (tpr_list[i] + tpr_list[i - 1]) / 2
    return auc


def print_confusion_matrix_visual(tp, tn, fp, fn, labels=("legitimate", "phishing")):
    """Print a visual confusion matrix."""
    print(f"\n   {'':>15s}  {'Predicted':^25s}")
    print(f"   {'':>15s}  {labels[0]:>12s}  {labels[1]:>12s}")
    print(f"   {'':>15s}  {'─'*12}  {'─'*12}")
    print(f"   Actual {labels[0]:>6s} │ {tn:>10d}  │ {fp:>10d}  │")
    print(f"   {'':>15s}  {'─'*12}  {'─'*12}")
    print(f"   Actual {labels[1]:>6s} │ {fn:>10d}  │ {tp:>10d}  │")
    print(f"   {'':>15s}  {'─'*12}  {'─'*12}")


# ══════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  K-NEAREST NEIGHBORS (KNN) — PHISHING WEBSITE DETECTION")
    print("  (Implemented from scratch — NO external libraries)")
    print("=" * 60)

    # ── 1. Load CSV ──────────────────────────────────────────
    print("\n📂 Loading dataset...")
    data = []
    with open("dataset_processed.csv", "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            data.append(row)

    print(f"   Total samples: {len(data)}")
    print(f"   Total columns: {len(header)}")

    # ── 2. Parse features and target ─────────────────────────
    # Find column indices
    url_idx = header.index("url")
    status_idx = header.index("status")
    feature_indices = [i for i in range(len(header)) if i != url_idx and i != status_idx]
    feature_names = [header[i] for i in feature_indices]

    X = []
    y = []
    skipped = 0
    for row in data:
        try:
            features = [float(row[i]) for i in feature_indices]
            label = 1 if row[status_idx].strip().lower() == "phishing" else 0
            X.append(features)
            y.append(label)
        except (ValueError, IndexError):
            skipped += 1

    if skipped > 0:
        print(f"   ⚠️  Skipped {skipped} rows with invalid data")

    print(f"   Valid samples: {len(X)}")
    print(f"   Features:      {len(feature_names)}")

    # Class distribution
    n_phishing = sum(y)
    n_legitimate = len(y) - n_phishing
    print(f"\n   📊 Class Distribution:")
    print(f"      Legitimate (0): {n_legitimate} ({n_legitimate/len(y)*100:.1f}%)")
    print(f"      Phishing   (1): {n_phishing} ({n_phishing/len(y)*100:.1f}%)")

    # ── 3. Train/Test Split ──────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_seed=42)
    print(f"\n📂 Train set: {len(X_train)} samples")
    print(f"📂 Test  set: {len(X_test)} samples")

    # ── 4. Standardize Features ──────────────────────────────
    print("\n🔄 Standardizing features...")
    X_train_s, X_test_s, _, _ = standardize(X_train, X_test)

    # ── 5. Train Model ───────────────────────────────────────
    K = 5
    print(f"\n🏋️  Training KNN (K={K})...")
    
    model = KNNScratch(k=K)
    model.fit(X_train_s, y_train)

    # ── 6. Predict on TEST set ──────────────────────────────
    print("\n🔮 Predicting on TEST set...")
    y_prob = model.predict_proba(X_test_s)
    y_pred = [1 if p >= 0.5 else 0 for p in y_prob]

    # ── 6b. Predict on TRAIN set (sampled for comparison) ────
    print("\n🔮 Predicting on TRAIN set (first 500 samples for basic check)...")
    # For KNN, evaluating on the full train set takes a long time and returns near perfect due to k=5. 
    # Just sample 500 for a quick metric check.
    sample_size = min(500, len(X_train_s))
    X_train_sample = X_train_s[:sample_size]
    y_train_sample = y_train[:sample_size]
    y_train_prob = model.predict_proba(X_train_sample)
    y_train_pred = [1 if p >= 0.5 else 0 for p in y_train_prob]

    # ── 7. COMPREHENSIVE EVALUATION ──────────────────────────
    tp, tn, fp, fn = confusion_matrix(y_test, y_pred)
    n_test = len(y_test)

    # ── Primary Metrics ──
    acc = accuracy(y_test, y_pred)
    prec_ph = precision(tp, fp)                              # Precision (phishing)
    rec_ph = recall(tp, fn)                                  # Recall / Sensitivity / TPR
    f1_ph = f1_score(prec_ph, rec_ph)                        # F1-Score (phishing)
    auc = roc_auc(y_test, y_prob)

    # ── Legitimate (negative class) metrics ──
    prec_leg = tn / (tn + fn) if (tn + fn) > 0 else 0       # Precision (legitimate) = NPV
    rec_leg = tn / (tn + fp) if (tn + fp) > 0 else 0        # Recall (legitimate) = Specificity = TNR
    f1_leg = f1_score(prec_leg, rec_leg)

    # ── Additional Metrics ──
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0    # TNR
    sensitivity = rec_ph                                      # TPR = Recall
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0            # Negative Predictive Value
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0            # False Positive Rate (Type I Error Rate)
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0            # False Negative Rate (Type II Error Rate)
    error_rate = 1 - acc                                      # Misclassification Rate
    balanced_acc = (sensitivity + specificity) / 2            # Balanced Accuracy
    prevalence = (tp + fn) / n_test                           # Prevalence of positive class
    detection_rate = tp / n_test                               # Detection Rate
    detection_prevalence = (tp + fp) / n_test                  # Detection Prevalence

    # Matthews Correlation Coefficient (MCC)
    mcc_num = (tp * tn) - (fp * fn)
    mcc_den = math.sqrt((tp+fp) * (tp+fn) * (tn+fp) * (tn+fn)) if (tp+fp)*(tp+fn)*(tn+fp)*(tn+fn) > 0 else 1
    mcc = mcc_num / mcc_den

    # Cohen's Kappa
    p_observed = acc
    p_expected = ((tp+fp)/n_test * (tp+fn)/n_test) + ((tn+fn)/n_test * (tn+fp)/n_test)
    kappa = (p_observed - p_expected) / (1 - p_expected) if (1 - p_expected) > 0 else 0

    # Youden's J Statistic (Informedness)
    youden_j = sensitivity + specificity - 1

    # Diagnostic Odds Ratio (DOR)
    if fn > 0 and fp > 0:
        dor = (tp * tn) / (fp * fn)
    else:
        dor = float('inf')

    # Gini Coefficient
    gini = 2 * auc - 1

    # Log-Loss (Binary Cross-Entropy on test set)
    log_loss_val = 0.0
    for i in range(n_test):
        p = max(min(y_prob[i], 1 - 1e-15), 1e-15)
        log_loss_val += -(y_test[i] * math.log(p) + (1 - y_test[i]) * math.log(1 - p))
    log_loss_val /= n_test

    # Positive / Negative Likelihood Ratios
    lr_pos = sensitivity / fpr if fpr > 0 else float('inf')     # LR+
    lr_neg = fnr / specificity if specificity > 0 else float('inf')  # LR-

    # Macro & Weighted averages
    macro_precision = (prec_ph + prec_leg) / 2
    macro_recall = (rec_ph + rec_leg) / 2
    macro_f1 = (f1_ph + f1_leg) / 2
    w_leg = (tn + fp) / n_test
    w_ph = (tp + fn) / n_test
    weighted_precision = prec_leg * w_leg + prec_ph * w_ph
    weighted_recall = rec_leg * w_leg + rec_ph * w_ph
    weighted_f1 = f1_leg * w_leg + f1_ph * w_ph

    # Train set metrics
    acc_train = accuracy(y_train_sample, y_train_pred)
    auc_train = roc_auc(y_train_sample, y_train_prob)

    # ══════════════════════════════════════════════════════════
    # PRINT ALL RESULTS
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 65)
    print("  COMPREHENSIVE MODEL EVALUATION RESULTS")
    print("=" * 65)

    print("\n┌─────────────────────────────────────────────────────────────┐")
    print("│  A. CONFUSION MATRIX                                       │")
    print("└─────────────────────────────────────────────────────────────┘")
    print_confusion_matrix_visual(tp, tn, fp, fn)
    print(f"\n   TP (True Positives)  = {tp:>5d}   (Phishing correctly detected)")
    print(f"   TN (True Negatives)  = {tn:>5d}   (Legitimate correctly identified)")
    print(f"   FP (False Positives) = {fp:>5d}   (Legitimate misclassified as Phishing)")
    print(f"   FN (False Negatives) = {fn:>5d}   (Phishing missed, classified as Legitimate)")

    print("\n┌─────────────────────────────────────────────────────────────┐")
    print("│  B. PRIMARY PERFORMANCE METRICS                            │")
    print("└─────────────────────────────────────────────────────────────┘")
    print(f"   {'Metric':<35s} {'Value':>10s}  {'Percentage':>12s}")
    print(f"   {'─'*60}")
    print(f"   {'Accuracy':<35s} {acc:>10.4f}  {acc*100:>11.2f}%")
    print(f"   {'Error Rate (Misclassification)':<35s} {error_rate:>10.4f}  {error_rate*100:>11.2f}%")
    print(f"   {'Precision (Phishing / PPV)':<35s} {prec_ph:>10.4f}  {prec_ph*100:>11.2f}%")
    print(f"   {'Recall / Sensitivity / TPR':<35s} {rec_ph:>10.4f}  {rec_ph*100:>11.2f}%")
    print(f"   {'F1-Score (Phishing)':<35s} {f1_ph:>10.4f}  {f1_ph*100:>11.2f}%")
    print(f"   {'Specificity / TNR':<35s} {specificity:>10.4f}  {specificity*100:>11.2f}%")
    print(f"   {'Negative Predictive Value (NPV)':<35s} {npv:>10.4f}  {npv*100:>11.2f}%")
    print(f"   {'ROC-AUC Score':<35s} {auc:>10.4f}  {auc*100:>11.2f}%")

    print("\n┌─────────────────────────────────────────────────────────────┐")
    print("│  C. ADVANCED PERFORMANCE METRICS                           │")
    print("└─────────────────────────────────────────────────────────────┘")
    print(f"   {'Metric':<35s} {'Value':>10s}")
    print(f"   {'─'*47}")
    print(f"   {'Balanced Accuracy':<35s} {balanced_acc:>10.4f}")
    print(f"   {'Matthews Correlation Coeff (MCC)':<35s} {mcc:>10.4f}")
    ck_label = "Cohen's Kappa"
    yj_label = "Youden's J Statistic"
    print(f"   {ck_label:<35s} {kappa:>10.4f}")
    print(f"   {yj_label:<35s} {youden_j:>10.4f}")
    print(f"   {'Gini Coefficient':<35s} {gini:>10.4f}")
    print(f"   {'Log-Loss (Cross-Entropy)':<35s} {log_loss_val:>10.4f}")
    dor_str = f"{dor:.4f}" if dor != float('inf') else "∞"
    print(f"   {'Diagnostic Odds Ratio (DOR)':<35s} {dor_str:>10s}")
    lr_pos_str = f"{lr_pos:.4f}" if lr_pos != float('inf') else "∞"
    lr_neg_str = f"{lr_neg:.4f}" if lr_neg != float('inf') else "∞"
    print(f"   {'Positive Likelihood Ratio (LR+)':<35s} {lr_pos_str:>10s}")
    print(f"   {'Negative Likelihood Ratio (LR-)':<35s} {lr_neg_str:>10s}")

    print("\n┌─────────────────────────────────────────────────────────────┐")
    print("│  D. ERROR RATES                                            │")
    print("└─────────────────────────────────────────────────────────────┘")
    print(f"   {'False Positive Rate (FPR / Type I)':<35s} {fpr:>10.4f}  {fpr*100:>11.2f}%")
    print(f"   {'False Negative Rate (FNR / Type II)':<35s} {fnr:>10.4f}  {fnr*100:>11.2f}%")
    print(f"   {'Prevalence':<35s} {prevalence:>10.4f}  {prevalence*100:>11.2f}%")
    print(f"   {'Detection Rate':<35s} {detection_rate:>10.4f}  {detection_rate*100:>11.2f}%")
    print(f"   {'Detection Prevalence':<35s} {detection_prevalence:>10.4f}  {detection_prevalence*100:>11.2f}%")

    print("\n┌─────────────────────────────────────────────────────────────┐")
    print("│  E. PER-CLASS CLASSIFICATION REPORT                        │")
    print("└─────────────────────────────────────────────────────────────┘")
    print(f"   {'Class':<15s}  {'Precision':>10s}  {'Recall':>10s}  {'F1-Score':>10s}  {'Support':>8s}")
    print(f"   {'─'*58}")
    print(f"   {'legitimate':<15s}  {prec_leg:>10.4f}  {rec_leg:>10.4f}  {f1_leg:>10.4f}  {tn+fp:>8d}")
    print(f"   {'phishing':<15s}  {prec_ph:>10.4f}  {rec_ph:>10.4f}  {f1_ph:>10.4f}  {tp+fn:>8d}")
    print(f"   {'─'*58}")
    print(f"   {'macro avg':<15s}  {macro_precision:>10.4f}  {macro_recall:>10.4f}  {macro_f1:>10.4f}  {n_test:>8d}")
    print(f"   {'weighted avg':<15s}  {weighted_precision:>10.4f}  {weighted_recall:>10.4f}  {weighted_f1:>10.4f}  {n_test:>8d}")

    print("\n┌─────────────────────────────────────────────────────────────┐")
    print("│  F. TRAIN vs TEST COMPARISON (Overfitting Check)           │")
    print("└─────────────────────────────────────────────────────────────┘")
    print(f"   {'Metric':<25s} {'Train':>10s}  {'Test':>10s}  {'Difference':>12s}")
    print(f"   {'─'*60}")
    print(f"   {'Accuracy':<25s} {acc_train:>10.4f}  {acc:>10.4f}  {abs(acc_train-acc):>+12.4f}")
    print(f"   {'ROC-AUC':<25s} {auc_train:>10.4f}  {auc:>10.4f}  {abs(auc_train-auc):>+12.4f}")
    overfit_flag = "⚠️  Potential overfitting!" if (acc_train - acc) > 0.05 else "✅ No significant overfitting"
    print(f"\n   Assessment: {overfit_flag}")

    # ── Model Parameters Summary ─────────────────────────────
    print("\n┌─────────────────────────────────────────────────────────────┐")
    print("│  G. MODEL PARAMETERS                                       │")
    print("└─────────────────────────────────────────────────────────────┘")
    print(f"   Algorithm:        K-Nearest Neighbors (KNN)")
    print(f"   K Neighbors:      {K}")
    print(f"   Distance Metric:  Euclidean")
    print(f"   Features Used:    {len(feature_names)}")
    print(f"   Scaling:          Z-Score Standardization")
    print(f"   Train/Test Split: 80/20")
    print(f"   Random Seed:      42")

    print("\n" + "=" * 65)
    print("  ✅ KNN CLASSIFICATION COMPLETE (No external libraries!)")
    print("=" * 65)

    # ══════════════════════════════════════════════════════════
    # WRITE REPORT TO FILE
    # ══════════════════════════════════════════════════════════
    report_file = "knn_report.txt"
    with open(report_file, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("  K-NEAREST NEIGHBORS (KNN) PERFORMANCE REPORT\n")
        f.write("  Phishing Website Detection — From Scratch Implementation\n")
        f.write("=" * 70 + "\n\n")

        f.write("A. DATASET SUMMARY\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Total Samples:     {len(X)}\n")
        f.write(f"  Features:          {len(feature_names)}\n")
        f.write(f"  Legitimate (0):    {n_legitimate} ({n_legitimate/len(y)*100:.1f}%)\n")
        f.write(f"  Phishing   (1):    {n_phishing} ({n_phishing/len(y)*100:.1f}%)\n")
        f.write(f"  Training Samples:  {len(X_train)}\n")
        f.write(f"  Testing Samples:   {len(X_test)}\n\n")

        f.write("B. CONFUSION MATRIX\n")
        f.write("-" * 40 + "\n")
        f.write(f"                   Pred Legit   Pred Phish\n")
        f.write(f"  Actual Legit:    {tn:>10d}   {fp:>10d}\n")
        f.write(f"  Actual Phish:    {fn:>10d}   {tp:>10d}\n\n")
        f.write(f"  TP={tp}  TN={tn}  FP={fp}  FN={fn}\n\n")

        f.write("C. PRIMARY PERFORMANCE METRICS\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Accuracy:                     {acc:.4f} ({acc*100:.2f}%)\n")
        f.write(f"  Error Rate:                   {error_rate:.4f} ({error_rate*100:.2f}%)\n")
        f.write(f"  Precision (Phishing/PPV):      {prec_ph:.4f}\n")
        f.write(f"  Recall (Sensitivity/TPR):      {rec_ph:.4f}\n")
        f.write(f"  F1-Score (Phishing):           {f1_ph:.4f}\n")
        f.write(f"  Specificity (TNR):             {specificity:.4f}\n")
        f.write(f"  Negative Predictive Val (NPV): {npv:.4f}\n")
        f.write(f"  ROC-AUC Score:                 {auc:.4f}\n\n")

        f.write("D. ADVANCED PERFORMANCE METRICS\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Balanced Accuracy:             {balanced_acc:.4f}\n")
        f.write(f"  Matthews Corr Coeff (MCC):     {mcc:.4f}\n")
        f.write(f"  Cohen's Kappa:                 {kappa:.4f}\n")
        f.write(f"  Youden's J Statistic:          {youden_j:.4f}\n")
        f.write(f"  Gini Coefficient:              {gini:.4f}\n")
        f.write(f"  Log-Loss (Cross-Entropy):      {log_loss_val:.4f}\n")
        f.write(f"  Diagnostic Odds Ratio:         {dor_str}\n")
        f.write(f"  Positive Likelihood Ratio:     {lr_pos_str}\n")
        f.write(f"  Negative Likelihood Ratio:     {lr_neg_str}\n\n")

        f.write("E. ERROR RATES\n")
        f.write("-" * 40 + "\n")
        f.write(f"  False Positive Rate (Type I):  {fpr:.4f} ({fpr*100:.2f}%)\n")
        f.write(f"  False Negative Rate (Type II): {fnr:.4f} ({fnr*100:.2f}%)\n")
        f.write(f"  Prevalence:                    {prevalence:.4f}\n")
        f.write(f"  Detection Rate:                {detection_rate:.4f}\n")
        f.write(f"  Detection Prevalence:          {detection_prevalence:.4f}\n\n")

        f.write("F. PER-CLASS CLASSIFICATION REPORT\n")
        f.write("-" * 60 + "\n")
        f.write(f"  {'Class':<15s}  {'Precision':>10s}  {'Recall':>10s}  {'F1-Score':>10s}  {'Support':>8s}\n")
        f.write(f"  {'-'*56}\n")
        f.write(f"  {'legitimate':<15s}  {prec_leg:>10.4f}  {rec_leg:>10.4f}  {f1_leg:>10.4f}  {tn+fp:>8d}\n")
        f.write(f"  {'phishing':<15s}  {prec_ph:>10.4f}  {rec_ph:>10.4f}  {f1_ph:>10.4f}  {tp+fn:>8d}\n")
        f.write(f"  {'-'*56}\n")
        f.write(f"  {'macro avg':<15s}  {macro_precision:>10.4f}  {macro_recall:>10.4f}  {macro_f1:>10.4f}  {n_test:>8d}\n")
        f.write(f"  {'weighted avg':<15s}  {weighted_precision:>10.4f}  {weighted_recall:>10.4f}  {weighted_f1:>10.4f}  {n_test:>8d}\n\n")

        f.write("G. TRAIN vs TEST COMPARISON (Sampled Train Set)\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Train Accuracy: {acc_train:.4f}  |  Test Accuracy: {acc:.4f}  |  Diff: {abs(acc_train-acc):.4f}\n")
        f.write(f"  Train ROC-AUC:  {auc_train:.4f}  |  Test ROC-AUC:  {auc:.4f}  |  Diff: {abs(auc_train-auc):.4f}\n\n")

        f.write(f"H. MODEL PARAMETERS\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Algorithm:         K-Nearest Neighbors (KNN)\n")
        f.write(f"  K Neighbors:       {K}\n")
        f.write(f"  Distance Metric:   Euclidean\n")
        f.write(f"  Features Used:     {len(feature_names)}\n")
        f.write(f"  Scaling:           Z-Score Standardization\n")
        f.write(f"  Train/Test Split:  80/20\n")
        f.write(f"  Random Seed:       42\n")
        f.write(f"  Implementation:    From scratch (no external libraries)\n\n")

        f.write("=" * 70 + "\n")
        f.write("  END OF REPORT\n")
        f.write("=" * 70 + "\n")

    print(f"\n📄 Full report saved → {report_file}")
