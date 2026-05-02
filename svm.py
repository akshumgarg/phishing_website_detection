

import csv
import math
import random






def dot_product(a, b):

    return sum(ai * bi for ai, bi in zip(a, b))


def mean(lst):

    return sum(lst) / len(lst)


def std_dev(lst):

    m = mean(lst)
    variance = sum((x - m) ** 2 for x in lst) / len(lst)
    return math.sqrt(variance) if variance > 0 else 1.0


def standardize(X_train, X_test):

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


def parse_status_label(raw_status):

    status = raw_status.strip().lower()
    if status == "phishing":
        return 1
    if status == "legitimate":
        return 0
    raise ValueError(f"Unknown status label: {raw_status!r}")


def train_test_split(X, y, test_size=0.2, random_seed=42):

    random.seed(random_seed)
    indices = list(range(len(X)))
    random.shuffle(indices)

    split_idx = int(len(X) * (1 - test_size))

    X_train = [X[i] for i in indices[:split_idx]]
    X_test = [X[i] for i in indices[split_idx:]]
    y_train = [y[i] for i in indices[:split_idx]]
    y_test = [y[i] for i in indices[split_idx:]]

    return X_train, X_test, y_train, y_test







class SVMScratch:


    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iterations=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.loss_history = []

    def fit(self, X, y):

        n_samples = len(X)
        n_features = len(X[0])


        self.weights = [0.0] * n_features
        self.bias = 0.0
        self.loss_history = []

        for iteration in range(self.n_iterations):

            indices = list(range(n_samples))
            random.shuffle(indices)

            epoch_loss = 0.0

            for idx in indices:
                xi = X[idx]
                yi = y[idx]


                decision = dot_product(xi, self.weights) + self.bias
                condition = yi * decision

                if condition >= 1:


                    for j in range(n_features):
                        self.weights[j] -= self.lr * (2 * self.lambda_param * self.weights[j])

                else:

                    for j in range(n_features):
                        self.weights[j] -= self.lr * (
                            2 * self.lambda_param * self.weights[j] - yi * xi[j]
                        )
                    self.bias -= self.lr * (-yi)

                    epoch_loss += 1 - condition


            reg_term = self.lambda_param * sum(w * w for w in self.weights)
            avg_loss = epoch_loss / n_samples + reg_term
            self.loss_history.append(avg_loss)


            if (iteration + 1) % 100 == 0 or iteration == 0:
                print(f"   Epoch {iteration + 1:>4d}/{self.n_iterations}  |  Loss: {avg_loss:.6f}")

    def decision_function(self, X):

        scores = []
        for row in X:
            score = dot_product(row, self.weights) + self.bias
            scores.append(score)
        return scores

    def predict(self, X):

        scores = self.decision_function(X)
        return [1 if s >= 0 else -1 for s in scores]

    def predict_binary(self, X):

        preds = self.predict(X)
        return [1 if p == 1 else 0 for p in preds]







def confusion_matrix_vals(y_true, y_pred):

    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)
    return tp, tn, fp, fn


def accuracy_score(y_true, y_pred):
    correct = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
    return correct / len(y_true)


def precision_score(tp, fp):
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def recall_score(tp, fn):
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def f1_score_calc(prec, rec):
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0


def roc_auc_score(y_true_01, decision_scores):

    pairs = list(zip(decision_scores, y_true_01))
    pairs.sort(key=lambda x: -x[0])

    tp = 0
    fp = 0
    total_pos = sum(y_true_01)
    total_neg = len(y_true_01) - total_pos

    if total_pos == 0 or total_neg == 0:
        return 0.5

    tpr_list = [0.0]
    fpr_list = [0.0]

    idx = 0
    while idx < len(pairs):
        score = pairs[idx][0]
        pos_in_group = 0
        neg_in_group = 0
        while idx < len(pairs) and pairs[idx][0] == score:
            _, label = pairs[idx]
            if label == 1:
                pos_in_group += 1
            else:
                neg_in_group += 1
            idx += 1
        tp += pos_in_group
        fp += neg_in_group
        tpr_list.append(tp / total_pos)
        fpr_list.append(fp / total_neg)

    auc = 0.0
    for i in range(1, len(fpr_list)):
        auc += (fpr_list[i] - fpr_list[i - 1]) * (tpr_list[i] + tpr_list[i - 1]) / 2
    return auc


def print_confusion_matrix_visual(tp, tn, fp, fn, labels=("legitimate", "phishing")):

    print(f"\n   {'':>15s}  {'Predicted':^25s}")
    print(f"   {'':>15s}  {labels[0]:>12s}  {labels[1]:>12s}")
    print(f"   {'':>15s}  {'─' * 12}  {'─' * 12}")
    print(f"   Actual {labels[0]:>6s} │ {tn:>10d}  │ {fp:>10d}  │")
    print(f"   {'':>15s}  {'─' * 12}  {'─' * 12}")
    print(f"   Actual {labels[1]:>6s} │ {fn:>10d}  │ {tp:>10d}  │")
    print(f"   {'':>15s}  {'─' * 12}  {'─' * 12}")






if __name__ == "__main__":
    print("=" * 65)
    print("  SUPPORT VECTOR MACHINE — PHISHING WEBSITE DETECTION")
    print("  (Implemented from scratch — NO external libraries)")
    print("=" * 65)


    print("\n📂 Loading dataset...")
    data = []
    with open("dataset_processed.csv") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            data.append(row)

    print(f"   Total samples: {len(data)}")
    print(f"   Total columns: {len(header)}")


    url_idx = header.index("url")
    status_idx = header.index("status")
    feature_indices = [i for i in range(len(header)) if i != url_idx and i != status_idx]
    feature_names = [header[i] for i in feature_indices]

    X = []
    y_binary = []
    skipped = 0
    for row in data:
        try:
            features = [float(row[i]) for i in feature_indices]
            label = parse_status_label(row[status_idx])
            X.append(features)
            y_binary.append(label)
        except (ValueError, IndexError):
            skipped += 1

    if skipped > 0:
        print(f"   ⚠️  Skipped {skipped} rows with invalid data")

    print(f"   Valid samples: {len(X)}")
    print(f"   Features:      {len(feature_names)}")

    n_phishing = sum(y_binary)
    n_legitimate = len(y_binary) - n_phishing
    print("\n   📊 Class Distribution:")
    print(f"      Legitimate (0 / -1): {n_legitimate} ({n_legitimate / len(y_binary) * 100:.1f}%)")
    print(f"      Phishing   (1 / +1): {n_phishing} ({n_phishing / len(y_binary) * 100:.1f}%)")



    X_train, X_test, y_train_bin, y_test_bin = train_test_split(
        X, y_binary, test_size=0.2, random_seed=42
    )


    y_train_svm = [1 if y == 1 else -1 for y in y_train_bin]
    y_test_svm = [1 if y == 1 else -1 for y in y_test_bin]

    print(f"\n📂 Train set: {len(X_train)} samples")
    print(f"📂 Test  set: {len(X_test)} samples")


    print("\n🔄 Standardizing features...")
    X_train_s, X_test_s, _, _ = standardize(X_train, X_test)


    print("\n🏋️  Training SVM (Hinge Loss + L2 Regularization + SGD)...")
    print("   Learning rate: 0.001  |  Lambda (C⁻¹): 0.01  |  Epochs: 1000\n")

    random.seed(42)
    model = SVMScratch(learning_rate=0.001, lambda_param=0.01, n_iterations=1000)
    model.fit(X_train_s, y_train_svm)


    y_pred_bin = model.predict_binary(X_test_s)
    decision_scores = model.decision_function(X_test_s)


    y_train_pred_bin = model.predict_binary(X_train_s)
    decision_scores_train = model.decision_function(X_train_s)


    tp, tn, fp, fn = confusion_matrix_vals(y_test_bin, y_pred_bin)
    n_test = len(y_test_bin)


    acc = accuracy_score(y_test_bin, y_pred_bin)
    prec_ph = precision_score(tp, fp)
    rec_ph = recall_score(tp, fn)
    f1_ph = f1_score_calc(prec_ph, rec_ph)
    auc = roc_auc_score(y_test_bin, decision_scores)


    prec_leg = tn / (tn + fn) if (tn + fn) > 0 else 0
    rec_leg = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1_leg = f1_score_calc(prec_leg, rec_leg)


    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = rec_ph
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    error_rate = 1 - acc
    balanced_acc = (sensitivity + specificity) / 2
    prevalence = (tp + fn) / n_test
    detection_rate = tp / n_test
    detection_prevalence = (tp + fp) / n_test


    mcc_num = (tp * tn) - (fp * fn)
    mcc_den = (
        math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        if (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) > 0
        else 1
    )
    mcc = mcc_num / mcc_den


    p_observed = acc
    p_expected = ((tp + fp) / n_test * (tp + fn) / n_test) + (
        (tn + fn) / n_test * (tn + fp) / n_test
    )
    kappa = (p_observed - p_expected) / (1 - p_expected) if (1 - p_expected) > 0 else 0


    youden_j = sensitivity + specificity - 1


    if fn > 0 and fp > 0:
        dor = (tp * tn) / (fp * fn)
    else:
        dor = float("inf")


    gini = 2 * auc - 1


    hinge_loss_val = 0.0
    for i in range(n_test):
        margin = y_test_svm[i] * decision_scores[i]
        hinge_loss_val += max(0, 1 - margin)
    hinge_loss_val /= n_test


    lr_pos = sensitivity / fpr if fpr > 0 else float("inf")
    lr_neg = fnr / specificity if specificity > 0 else float("inf")


    macro_precision = (prec_ph + prec_leg) / 2
    macro_recall = (rec_ph + rec_leg) / 2
    macro_f1 = (f1_ph + f1_leg) / 2
    w_leg = (tn + fp) / n_test
    w_ph = (tp + fn) / n_test
    weighted_precision = prec_leg * w_leg + prec_ph * w_ph
    weighted_recall = rec_leg * w_leg + rec_ph * w_ph
    weighted_f1 = f1_leg * w_leg + f1_ph * w_ph


    acc_train = accuracy_score(y_train_bin, y_train_pred_bin)
    auc_train = roc_auc_score(y_train_bin, decision_scores_train)


    train_margins = [y_train_svm[i] * decision_scores_train[i] for i in range(len(y_train_svm))]
    margin_mean = mean(train_margins)
    margin_min = min(train_margins)
    margin_max = max(train_margins)
    support_vectors_approx = sum(1 for m in train_margins if m <= 1.0)




    print("\n" + "=" * 65)
    print("  COMPREHENSIVE SVM EVALUATION RESULTS")
    print("=" * 65)

    print("\n┌─────────────────────────────────────────────────────────────┐")
    print("│  A. CONFUSION MATRIX                                       │")
    print("└─────────────────────────────────────────────────────────────┘")
    print_confusion_matrix_visual(tp, tn, fp, fn)
    print(f"\n   TP (True Positives)  = {tp:>5d}   (Phishing correctly detected)")
    print(f"   TN (True Negatives)  = {tn:>5d}   (Legitimate correctly identified)")
    print(f"   FP (False Positives) = {fp:>5d}   (Legitimate misclassified as Phishing)")
    print(f"   FN (False Negatives) = {fn:>5d}   (Phishing missed)")

    print("\n┌─────────────────────────────────────────────────────────────┐")
    print("│  B. PRIMARY PERFORMANCE METRICS                            │")
    print("└─────────────────────────────────────────────────────────────┘")
    print(f"   {'Metric':<35s} {'Value':>10s}  {'Percentage':>12s}")
    print(f"   {'─' * 60}")
    print(f"   {'Accuracy':<35s} {acc:>10.4f}  {acc * 100:>11.2f}%")
    print(
        f"   {'Error Rate (Misclassification)':<35s} {error_rate:>10.4f}  {error_rate * 100:>11.2f}%"
    )
    print(f"   {'Precision (Phishing / PPV)':<35s} {prec_ph:>10.4f}  {prec_ph * 100:>11.2f}%")
    print(f"   {'Recall / Sensitivity / TPR':<35s} {rec_ph:>10.4f}  {rec_ph * 100:>11.2f}%")
    print(f"   {'F1-Score (Phishing)':<35s} {f1_ph:>10.4f}  {f1_ph * 100:>11.2f}%")
    print(f"   {'Specificity / TNR':<35s} {specificity:>10.4f}  {specificity * 100:>11.2f}%")
    print(f"   {'Negative Predictive Value (NPV)':<35s} {npv:>10.4f}  {npv * 100:>11.2f}%")
    print(f"   {'ROC-AUC Score':<35s} {auc:>10.4f}  {auc * 100:>11.2f}%")

    print("\n┌─────────────────────────────────────────────────────────────┐")
    print("│  C. ADVANCED PERFORMANCE METRICS                           │")
    print("└─────────────────────────────────────────────────────────────┘")
    print(f"   {'Metric':<35s} {'Value':>10s}")
    print(f"   {'─' * 47}")
    print(f"   {'Balanced Accuracy':<35s} {balanced_acc:>10.4f}")
    print(f"   {'Matthews Correlation Coeff (MCC)':<35s} {mcc:>10.4f}")
    ck_label = "Cohen's Kappa"
    yj_label = "Youden's J Statistic"
    print(f"   {ck_label:<35s} {kappa:>10.4f}")
    print(f"   {yj_label:<35s} {youden_j:>10.4f}")
    print(f"   {'Gini Coefficient':<35s} {gini:>10.4f}")
    print(f"   {'Hinge Loss (Test Set)':<35s} {hinge_loss_val:>10.4f}")
    dor_str = f"{dor:.4f}" if dor != float("inf") else "inf"
    print(f"   {'Diagnostic Odds Ratio (DOR)':<35s} {dor_str:>10s}")
    lr_pos_str = f"{lr_pos:.4f}" if lr_pos != float("inf") else "inf"
    lr_neg_str = f"{lr_neg:.4f}" if lr_neg != float("inf") else "inf"
    print(f"   {'Positive Likelihood Ratio (LR+)':<35s} {lr_pos_str:>10s}")
    print(f"   {'Negative Likelihood Ratio (LR-)':<35s} {lr_neg_str:>10s}")

    print("\n┌─────────────────────────────────────────────────────────────┐")
    print("│  D. ERROR RATES                                            │")
    print("└─────────────────────────────────────────────────────────────┘")
    print(f"   {'False Positive Rate (FPR / Type I)':<35s} {fpr:>10.4f}  {fpr * 100:>11.2f}%")
    print(f"   {'False Negative Rate (FNR / Type II)':<35s} {fnr:>10.4f}  {fnr * 100:>11.2f}%")
    print(f"   {'Prevalence':<35s} {prevalence:>10.4f}  {prevalence * 100:>11.2f}%")
    print(f"   {'Detection Rate':<35s} {detection_rate:>10.4f}  {detection_rate * 100:>11.2f}%")
    print(
        f"   {'Detection Prevalence':<35s} {detection_prevalence:>10.4f}  {detection_prevalence * 100:>11.2f}%"
    )

    print("\n┌─────────────────────────────────────────────────────────────┐")
    print("│  E. PER-CLASS CLASSIFICATION REPORT                        │")
    print("└─────────────────────────────────────────────────────────────┘")
    print(
        f"   {'Class':<15s}  {'Precision':>10s}  {'Recall':>10s}  {'F1-Score':>10s}  {'Support':>8s}"
    )
    print(f"   {'─' * 58}")
    print(
        f"   {'legitimate':<15s}  {prec_leg:>10.4f}  {rec_leg:>10.4f}  {f1_leg:>10.4f}  {tn + fp:>8d}"
    )
    print(f"   {'phishing':<15s}  {prec_ph:>10.4f}  {rec_ph:>10.4f}  {f1_ph:>10.4f}  {tp + fn:>8d}")
    print(f"   {'─' * 58}")
    print(
        f"   {'macro avg':<15s}  {macro_precision:>10.4f}  {macro_recall:>10.4f}  {macro_f1:>10.4f}  {n_test:>8d}"
    )
    print(
        f"   {'weighted avg':<15s}  {weighted_precision:>10.4f}  {weighted_recall:>10.4f}  {weighted_f1:>10.4f}  {n_test:>8d}"
    )

    print("\n┌─────────────────────────────────────────────────────────────┐")
    print("│  F. TRAIN vs TEST COMPARISON (Overfitting Check)           │")
    print("└─────────────────────────────────────────────────────────────┘")
    print(f"   {'Metric':<25s} {'Train':>10s}  {'Test':>10s}  {'Difference':>12s}")
    print(f"   {'─' * 60}")
    print(f"   {'Accuracy':<25s} {acc_train:>10.4f}  {acc:>10.4f}  {abs(acc_train - acc):>+12.4f}")
    print(f"   {'ROC-AUC':<25s} {auc_train:>10.4f}  {auc:>10.4f}  {abs(auc_train - auc):>+12.4f}")
    overfit_flag = (
        "Potential overfitting!" if (acc_train - acc) > 0.05 else "No significant overfitting"
    )
    print(f"\n   Assessment: {overfit_flag}")

    print("\n┌─────────────────────────────────────────────────────────────┐")
    print("│  G. SVM-SPECIFIC: TRAINING MARGIN ANALYSIS                 │")
    print("└─────────────────────────────────────────────────────────────┘")
    print(f"   {'Mean Margin (y * f(x))':<35s} {margin_mean:>10.4f}")
    print(f"   {'Min Margin':<35s} {margin_min:>10.4f}")
    print(f"   {'Max Margin':<35s} {margin_max:>10.4f}")
    print(f"   {'Training samples with margin <= 1':<35s} {support_vectors_approx:>10d}")
    print(f"   {'% of training set within margin':<35s} {support_vectors_approx / len(train_margins) * 100:>9.2f}%")


    print("\n┌─────────────────────────────────────────────────────────────┐")
    print("│  H. TOP 15 FEATURE IMPORTANCES                             │")
    print("└─────────────────────────────────────────────────────────────┘")
    feature_weight_pairs = list(zip(feature_names, model.weights))
    feature_weight_pairs.sort(key=lambda x: abs(x[1]), reverse=True)

    print(f"   {'Rank':<6s} {'Feature':<30s} {'Weight':>12s}  {'Direction'}")
    print(f"   {'─' * 65}")
    for rank, (fname, w) in enumerate(feature_weight_pairs[:15], 1):
        direction = "-> phishing" if w > 0 else "-> legitimate"
        print(f"   {rank:<6d} {fname:<30s} {w:>+12.6f}  ({direction})")


    print(f"\n   Complete Feature Weights ({len(feature_names)} features):")
    print(f"   {'─' * 65}")
    for rank, (fname, w) in enumerate(feature_weight_pairs, 1):
        print(f"   {rank:<4d} {fname:<30s} {w:>+12.6f}")


    print("\n┌─────────────────────────────────────────────────────────────┐")
    print("│  I. TRAINING LOSS CURVE                                    │")
    print("└─────────────────────────────────────────────────────────────┘")
    max_bar = 40
    sampled = [(i, model.loss_history[i]) for i in range(0, len(model.loss_history), 100)]
    max_loss = max(loss for _, loss in sampled) if sampled else 1
    for idx, loss_val in sampled:
        bar_len = int((loss_val / max_loss) * max_bar)
        bar = "█" * bar_len
        print(f"   Epoch {idx + 1:>4d} │ {bar:<{max_bar}s} │ {loss_val:.6f}")
    print(f"\n   Initial Loss: {model.loss_history[0]:.6f}")
    print(f"   Final Loss:   {model.loss_history[-1]:.6f}")
    if model.loss_history[0] > 0:
        print(
            f"   Loss Reduction: {(1 - model.loss_history[-1] / model.loss_history[0]) * 100:.2f}%"
        )


    print("\n┌─────────────────────────────────────────────────────────────┐")
    print("│  J. MODEL PARAMETERS                                       │")
    print("└─────────────────────────────────────────────────────────────┘")
    print("   Algorithm:        Linear SVM (Support Vector Machine)")
    print("   Kernel:           Linear")
    print("   Loss Function:    Hinge Loss + L2 Regularization")
    print("   Optimization:     Stochastic Gradient Descent (SGD)")
    print(f"   Learning Rate:    {model.lr}")
    print(f"   Lambda (reg):     {model.lambda_param}")
    print(f"   Epochs:           {model.n_iterations}")
    print(f"   Features Used:    {len(feature_names)}")
    print(f"   Bias Term:        {model.bias:.6f}")
    print("   Scaling:          Z-Score Standardization")
    print("   Train/Test Split: 80/20")
    print("   Random Seed:      42")

    print("\n" + "=" * 65)
    print("  ✅ SVM CLASSIFICATION COMPLETE (No external libraries!)")
    print("=" * 65)




    report_file = "svm_report.txt"
    with open(report_file, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("  SVM (SUPPORT VECTOR MACHINE) PERFORMANCE REPORT\n")
        f.write("  Phishing Website Detection — From Scratch Implementation\n")
        f.write("=" * 70 + "\n\n")

        f.write("A. DATASET SUMMARY\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Total Samples:     {len(X)}\n")
        f.write(f"  Features:          {len(feature_names)}\n")
        f.write(
            f"  Legitimate:        {n_legitimate} ({n_legitimate / len(y_binary) * 100:.1f}%)\n"
        )
        f.write(f"  Phishing:          {n_phishing} ({n_phishing / len(y_binary) * 100:.1f}%)\n")
        f.write(f"  Training Samples:  {len(X_train)}\n")
        f.write(f"  Testing Samples:   {len(X_test)}\n\n")

        f.write("B. CONFUSION MATRIX\n")
        f.write("-" * 40 + "\n")
        f.write("                   Pred Legit   Pred Phish\n")
        f.write(f"  Actual Legit:    {tn:>10d}   {fp:>10d}\n")
        f.write(f"  Actual Phish:    {fn:>10d}   {tp:>10d}\n\n")
        f.write(f"  TP={tp}  TN={tn}  FP={fp}  FN={fn}\n\n")

        f.write("C. PRIMARY PERFORMANCE METRICS\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Accuracy:                     {acc:.4f} ({acc * 100:.2f}%)\n")
        f.write(f"  Error Rate:                   {error_rate:.4f} ({error_rate * 100:.2f}%)\n")
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
        f.write(f"  Hinge Loss (Test Set):         {hinge_loss_val:.4f}\n")
        f.write(f"  Diagnostic Odds Ratio:         {dor_str}\n")
        f.write(f"  Positive Likelihood Ratio:     {lr_pos_str}\n")
        f.write(f"  Negative Likelihood Ratio:     {lr_neg_str}\n\n")

        f.write("E. ERROR RATES\n")
        f.write("-" * 40 + "\n")
        f.write(f"  False Positive Rate (Type I):  {fpr:.4f} ({fpr * 100:.2f}%)\n")
        f.write(f"  False Negative Rate (Type II): {fnr:.4f} ({fnr * 100:.2f}%)\n")
        f.write(f"  Prevalence:                    {prevalence:.4f}\n")
        f.write(f"  Detection Rate:                {detection_rate:.4f}\n")
        f.write(f"  Detection Prevalence:          {detection_prevalence:.4f}\n\n")

        f.write("F. PER-CLASS CLASSIFICATION REPORT\n")
        f.write("-" * 60 + "\n")
        f.write(
            f"  {'Class':<15s}  {'Precision':>10s}  {'Recall':>10s}  {'F1-Score':>10s}  {'Support':>8s}\n"
        )
        f.write(f"  {'-' * 56}\n")
        f.write(
            f"  {'legitimate':<15s}  {prec_leg:>10.4f}  {rec_leg:>10.4f}  {f1_leg:>10.4f}  {tn + fp:>8d}\n"
        )
        f.write(
            f"  {'phishing':<15s}  {prec_ph:>10.4f}  {rec_ph:>10.4f}  {f1_ph:>10.4f}  {tp + fn:>8d}\n"
        )
        f.write(f"  {'-' * 56}\n")
        f.write(
            f"  {'macro avg':<15s}  {macro_precision:>10.4f}  {macro_recall:>10.4f}  {macro_f1:>10.4f}  {n_test:>8d}\n"
        )
        f.write(
            f"  {'weighted avg':<15s}  {weighted_precision:>10.4f}  {weighted_recall:>10.4f}  {weighted_f1:>10.4f}  {n_test:>8d}\n\n"
        )

        f.write("G. TRAIN vs TEST COMPARISON\n")
        f.write("-" * 40 + "\n")
        f.write(
            f"  Train Accuracy: {acc_train:.4f}  |  Test Accuracy: {acc:.4f}  |  Diff: {abs(acc_train - acc):.4f}\n"
        )
        f.write(
            f"  Train ROC-AUC:  {auc_train:.4f}  |  Test ROC-AUC:  {auc:.4f}  |  Diff: {abs(auc_train - auc):.4f}\n\n"
        )

        f.write("H. SVM TRAINING MARGIN ANALYSIS\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Mean Margin:              {margin_mean:.4f}\n")
        f.write(f"  Min Margin:               {margin_min:.4f}\n")
        f.write(f"  Max Margin:               {margin_max:.4f}\n")
        f.write(f"  Training samples with margin <= 1: {support_vectors_approx}\n")
        f.write(
            f"  % of training set within margin:  {support_vectors_approx / len(train_margins) * 100:.2f}%\n\n"
        )

        f.write("I. TOP 15 FEATURE IMPORTANCES\n")
        f.write("-" * 60 + "\n")
        f.write(f"  {'Rank':<6s} {'Feature':<30s} {'Weight':>12s}\n")
        f.write(f"  {'-' * 50}\n")
        for rank, (fname, w) in enumerate(feature_weight_pairs[:15], 1):
            direction = "phishing" if w > 0 else "legitimate"
            f.write(f"  {rank:<6d} {fname:<30s} {w:>+12.6f}  ({direction})\n")

        f.write(f"\nJ. ALL FEATURE WEIGHTS ({len(feature_names)} features)\n")
        f.write("-" * 60 + "\n")
        for rank, (fname, w) in enumerate(feature_weight_pairs, 1):
            f.write(f"  {rank:<4d} {fname:<30s} {w:>+12.6f}\n")

        f.write("\nK. TRAINING LOSS HISTORY\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Initial Loss: {model.loss_history[0]:.6f}\n")
        f.write(f"  Final Loss:   {model.loss_history[-1]:.6f}\n\n")
        for i in range(0, len(model.loss_history), 100):
            f.write(f"  Epoch {i + 1:>4d}: Loss = {model.loss_history[i]:.6f}\n")

        f.write("\nL. MODEL PARAMETERS\n")
        f.write("-" * 40 + "\n")
        f.write("  Algorithm:         Linear SVM\n")
        f.write("  Kernel:            Linear\n")
        f.write("  Loss Function:     Hinge Loss + L2 Regularization\n")
        f.write("  Optimization:      SGD\n")
        f.write(f"  Learning Rate:     {model.lr}\n")
        f.write(f"  Lambda (reg):      {model.lambda_param}\n")
        f.write(f"  Epochs:            {model.n_iterations}\n")
        f.write(f"  Features Used:     {len(feature_names)}\n")
        f.write(f"  Bias Term:         {model.bias:.6f}\n")
        f.write("  Scaling:           Z-Score Standardization\n")
        f.write("  Train/Test Split:  80/20\n")
        f.write("  Random Seed:       42\n")
        f.write("  Implementation:    From scratch (no external libraries)\n\n")

        f.write("=" * 70 + "\n")
        f.write("  END OF REPORT\n")
        f.write("=" * 70 + "\n")

    print(f"\n📄 Full report saved → {report_file}")
