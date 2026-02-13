import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Utilities: metrics + ROC
# ------------------------------------------------------------
def confusion_from_scores(y_true, y_score, thr=0.5):
    y_pred = (y_score >= thr).astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return tp, tn, fp, fn

def classification_metrics(y_true, y_score, thr=0.5):
    tp, tn, fp, fn = confusion_from_scores(y_true, y_score, thr)
    acc = (tp + tn) / max(1, (tp + tn + fp + fn))
    prec = tp / max(1, (tp + fp))
    rec = tp / max(1, (tp + fn))
    return acc, prec, rec, (tp, tn, fp, fn)

def roc_curve_points(y_true, y_score):
    # thresholds from high->low (include sentinels)
    thresholds = np.unique(y_score)[::-1]
    thresholds = np.r_[thresholds, -np.inf]  # ensures (1,1) endpoint
    P = max(1, int(np.sum(y_true == 1)))
    N = max(1, int(np.sum(y_true == 0)))

    tpr_list, fpr_list = [], []
    for thr in thresholds:
        tp, tn, fp, fn = confusion_from_scores(y_true, y_score, thr)
        tpr = tp / P
        fpr = fp / N
        tpr_list.append(tpr)
        fpr_list.append(fpr)

    fpr = np.array(fpr_list)
    tpr = np.array(tpr_list)
    # AUC via trapezoid rule over FPR-sorted
    order = np.argsort(fpr)
    auc = np.trapz(tpr[order], fpr[order])
    return fpr, tpr, auc

def plot_roc(y_true, y_score, title):
    fpr, tpr, auc = roc_curve_points(y_true, y_score)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{title} (AUC={auc:.4f})")
    plt.grid(True, alpha=0.3)
    plt.show()

# ------------------------------------------------------------
# Data generation / loading
# ------------------------------------------------------------
def generate_data_via_provided_script(seed=0):
    """
    Uses the same distribution as Generate_data.py (your provided file).
    We re-implement the generation here so this script is self-contained,
    but it matches the provided generator (N1=300, N2=80, means/covs, labels 0/1).
    """
    rng = np.random.default_rng(seed)
    N1 = 300
    N2 = 80
    sigma = 2

    mean1 = np.array([10, 14])
    cov1 = np.array([[sigma, 0], [0, sigma]])
    X1 = rng.multivariate_normal(mean1, cov1, N1)

    mean2 = np.array([14, 18])
    cov2 = np.array([[sigma, 0], [0, sigma]])
    X2 = rng.multivariate_normal(mean2, cov2, N2)

    X = np.vstack([X1, X2])
    T = np.array([0] * N1 + [1] * N2, dtype=int)

    return X, T, X1, X2

def show_scatter(X1, X2, title="Generated data scatter"):
    plt.figure()
    plt.scatter(X1[:, 0], X1[:, 1], marker='o', label="Class 1 (t=0)")
    plt.scatter(X2[:, 0], X2[:, 1], marker='o', label="Class 2 (t=1)")
    plt.title(title)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# ------------------------------------------------------------
# Part 1: MLP 2-3-1 from scratch
# ------------------------------------------------------------
def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return (x > 0).astype(float)

def sigmoid(x):
    x = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-x))

class MLP231:
    """
    2 inputs -> 3 hidden -> 1 output
    Total weight connections: 2*3 + 3*1 = 9 (not counting biases)
    Biases are included for model flexibility.
    """
    def __init__(self, seed=0, w_std=1):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, w_std, size=(2, 3))   # 6 weights
        self.b1 = rng.normal(0, w_std/2, size=(3,))   # bias (small init)
        self.W2 = rng.normal(0, w_std, size=(3, 1))   # 3 weights
        self.b2 = rng.normal(0, w_std/2, size=(1,))   # bias (small init)

    def forward(self, X):
        z1 = X @ self.W1 + self.b1
        a1 = relu(z1)
        z2 = a1 @ self.W2 + self.b2
        y = sigmoid(z2).reshape(-1)  # probability-like score
        cache = (X, z1, a1, z2, y)
        return y, cache

    def predict_score(self, X):
        y, _ = self.forward(X)
        return y

    def batch_backprop_update(self, X, T, lr=0.001, l2=0.0):
        """
        Binary cross-entropy loss with sigmoid output:
        For BCE + sigmoid, delta_out = (y - t)
        """
        y, cache = self.forward(X)
        Xc, z1, a1, z2, yv = cache

        t = T.astype(float)
        # output delta: (N,)
        delta2 = (yv - t)  # (N,)

        # gradients W2, b2
        # a1: (N,3), delta2: (N,)
        dW2 = (a1.T @ delta2.reshape(-1, 1)) / X.shape[0]
        db2 = np.mean(delta2)

        # hidden delta
        # (N,3) = (N,1) @ (1,3) elementwise * relu'(z1)
        delta1 = (delta2.reshape(-1, 1) @ self.W2.T) * relu_deriv(z1)  # (N,3)

        dW1 = (X.T @ delta1) / X.shape[0]
        db1 = np.mean(delta1, axis=0)

        # L2 regularization (optional)
        dW2 += l2 * self.W2
        dW1 += l2 * self.W1

        # gradient step
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1

def plot_decision_boundary_score(model_score_fn, X, T, title, thr=0.5, grid_steps=250):
    x_min, x_max = X[:, 0].min() - 1.0, X[:, 0].max() + 1.0
    y_min, y_max = X[:, 1].min() - 1.0, X[:, 1].max() + 1.0
    xs = np.linspace(x_min, x_max, grid_steps)
    ys = np.linspace(y_min, y_max, grid_steps)
    XX, YY = np.meshgrid(xs, ys)
    grid = np.c_[XX.ravel(), YY.ravel()]

    score = model_score_fn(grid).reshape(XX.shape)

    plt.figure()
    plt.contour(XX, YY, score, levels=[thr])
    plt.scatter(X[T == 0, 0], X[T == 0, 1], marker='o', label="Class 1 (t=0)")
    plt.scatter(X[T == 1, 0], X[T == 1, 1], marker='o', label="Class 2 (t=1)")
    plt.title(title)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# ------------------------------------------------------------
# Part 2: RBF networks (Gaussian width sigma=1), random centers
# ------------------------------------------------------------
def rbf_design_matrix(X, centers, sigma=1.0, add_bias=True):
    # X: (N,2), centers: (M,2)
    # Phi_ij = exp(-||x_i - c_j||^2/(2*sigma^2))
    X2 = np.sum(X**2, axis=1, keepdims=True)          # (N,1)
    C2 = np.sum(centers**2, axis=1, keepdims=True).T  # (1,M)
    dist2 = X2 - 2*(X @ centers.T) + C2               # (N,M)
    Phi = np.exp(-dist2 / (2 * sigma**2))             # (N,M)
    if add_bias:
        Phi = np.c_[np.ones((X.shape[0], 1)), Phi]    # (N,M+1)
    return Phi

def solve_ls_normal_eq(Phi, t, ridge=1e-8):
    """
    Least-squares from scratch using normal equation:
    w = (Phi^T Phi + ridge*I)^(-1) Phi^T t
    (no scipy.linalg.lstsq)
    """
    t = t.astype(float).reshape(-1, 1)
    A = Phi.T @ Phi
    I = np.eye(A.shape[0])
    w = np.linalg.inv(A + ridge * I) @ (Phi.T @ t)
    return w.reshape(-1)

class RBFNet:
    def __init__(self, centers, sigma=1.0):
        self.centers = np.array(centers, dtype=float)
        self.sigma = float(sigma)
        self.w = None  # includes bias as w[0]

    def fit_ls(self, X, T):
        Phi = rbf_design_matrix(X, self.centers, sigma=self.sigma, add_bias=True)
        self.w = solve_ls_normal_eq(Phi, T)

    def predict_score(self, X):
        Phi = rbf_design_matrix(X, self.centers, sigma=self.sigma, add_bias=True)
        # linear output then squashed to (0,1) for ROC
        y_lin = Phi @ self.w
        # Clip to reasonable range before sigmoid for numerical stability
        y_lin = np.clip(y_lin, -10, 10)
        return sigmoid(y_lin)

def random_centers_within_data_range(X, M, seed=0):
    rng = np.random.default_rng(seed)
    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    centers = rng.uniform(mins, maxs, size=(M, X.shape[1]))
    return centers

def plot_rbf_centers(X, T, centers, title):
    plt.figure()
    plt.scatter(X[T == 0, 0], X[T == 0, 1], marker='o', label="Class 1 (t=0)")
    plt.scatter(X[T == 1, 0], X[T == 1, 1], marker='o', label="Class 2 (t=1)")
    plt.scatter(centers[:, 0], centers[:, 1], marker='X', s=120, label="RBF centers")
    plt.title(title)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# ------------------------------------------------------------
# Main: executes all required tasks and plots
# ------------------------------------------------------------
def main():
    # =============================
    # Part 1a: Generate + scatter
    # =============================
    # This matches the provided Generate_data.py distribution.
    X, T, X1, X2 = generate_data_via_provided_script(seed=1)
    
    # Normalize data to improve training stability
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0) + 1e-8
    X = (X - X_mean) / X_std
    
    show_scatter(X1, X2, title="Part 1a: Generated data scatter")

    # =============================
    # Part 1b: Initial MLP
    # =============================
    mlp = MLP231(seed=2, w_std=0.6)

    y_init = mlp.predict_score(X)
    acc, prec, rec, cm = classification_metrics(T, y_init, thr=0.5)
    print("Part 1b (Initial MLP) metrics @thr=0.5")
    print(f"  Accuracy={acc:.4f}, Precision={prec:.4f}, Recall={rec:.4f}, (TP,TN,FP,FN)={cm}")

    plot_decision_boundary_score(
        mlp.predict_score, X, T,
        title="Part 1b: Initial MLP decision boundary (contour at 0.5)",
        thr=0.5
    )
    plot_roc(T, y_init, title="Part 1b: Initial MLP ROC")

    # =============================
    # Part 1c: Batch backprop update
    # =============================
    epochs = 1000
    lr = 0.02
    l2 = 0.001

    for _ in range(epochs):
        mlp.batch_backprop_update(X, T, lr=lr, l2=l2)

    y_trained = mlp.predict_score(X)
    acc2, prec2, rec2, cm2 = classification_metrics(T, y_trained, thr=0.5)
    print("\nPart 1c (Updated MLP) metrics @thr=0.5")
    print(f"  Accuracy={acc2:.4f}, Precision={prec2:.4f}, Recall={rec2:.4f}, (TP,TN,FP,FN)={cm2}")

    plot_decision_boundary_score(
        mlp.predict_score, X, T,
        title="Part 1c: Updated MLP decision boundary (contour at 0.5)",
        thr=0.5
    )
    plot_roc(T, y_trained, title="Part 1c: Updated MLP ROC")

    # =============================
    # Part 2a: RBF, 3 random centers, sigma=1, LS weights
    # =============================
    centers3 = random_centers_within_data_range(X, M=3, seed=42)
    rbf3 = RBFNet(centers=centers3, sigma=1)
    rbf3.fit_ls(X, T)

    plot_rbf_centers(X, T, centers3, title="Part 2a: RBF centers (M=3)")

    y_rbf3 = rbf3.predict_score(X)
    acc3, prec3, rec3, cm3 = classification_metrics(T, y_rbf3, thr=0.5)
    print("\nPart 2a (RBF M=3) metrics @thr=0.5")
    print(f"  Accuracy={acc3:.4f}, Precision={prec3:.4f}, Recall={rec3:.4f}, (TP,TN,FP,FN)={cm3}")

    plot_decision_boundary_score(
        rbf3.predict_score, X, T,
        title="Part 2a: RBF (M=3) decision boundary (contour at 0.5)",
        thr=0.5
    )
    plot_roc(T, y_rbf3, title="Part 2a: RBF (M=3) ROC")

    # =============================
    # Part 2b: RBF, 6 random centers, sigma=1, LS weights
    # =============================
    centers6 = random_centers_within_data_range(X, M=6, seed=4)
    rbf6 = RBFNet(centers=centers6, sigma=3.0)
    rbf6.fit_ls(X, T)

    plot_rbf_centers(X, T, centers6, title="Part 2b: RBF centers (M=6)")

    y_rbf6 = rbf6.predict_score(X)
    acc4, prec4, rec4, cm4 = classification_metrics(T, y_rbf6, thr=0.5)
    print("\nPart 2b (RBF M=6) metrics @thr=0.5")
    print(f"  Accuracy={acc4:.4f}, Precision={prec4:.4f}, Recall={rec4:.4f}, (TP,TN,FP,FN)={cm4}")

    plot_decision_boundary_score(
        rbf6.predict_score, X, T,
        title="Part 2b: RBF (M=6) decision boundary (contour at 0.5)",
        thr=0.5
    )
    plot_roc(T, y_rbf6, title="Part 2b: RBF (M=6) ROC")

    # Quick comparisons (textual)
    print("\n=== Quick comparison summary ===")
    print(f"Initial MLP:  acc={acc:.4f},  prec={prec:.4f},  rec={rec:.4f}")
    print(f"Updated MLP:  acc={acc2:.4f}, prec={prec2:.4f}, rec={rec2:.4f}")
    print(f"RBF (M=3):    acc={acc3:.4f},  prec={prec3:.4f},  rec={rec3:.4f}")
    print(f"RBF (M=6):    acc={acc4:.4f},  prec={prec4:.4f},  rec={rec4:.4f}")

if __name__ == "__main__":
    main()