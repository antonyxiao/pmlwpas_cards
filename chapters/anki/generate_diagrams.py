#!/usr/bin/env python3
"""
Generate ML diagrams for Anki flashcards.

Usage:
    python generate_diagrams.py          # Generate all diagrams
    python generate_diagrams.py --list   # List available diagrams
    python generate_diagrams.py ch05     # Generate specific chapter diagrams

Requirements:
    pip install matplotlib numpy networkx
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
from matplotlib.lines import Line2D
import warnings
warnings.filterwarnings('ignore')

# Output directory
IMAGES_DIR = os.path.join(os.path.dirname(__file__), 'images')
os.makedirs(IMAGES_DIR, exist_ok=True)

# Common style settings
plt.style.use('default')
COLORS = {
    'primary': '#3498db',
    'secondary': '#2ecc71',
    'accent': '#e74c3c',
    'neutral': '#95a5a6',
    'dark': '#2c3e50',
    'light': '#ecf0f1',
    'purple': '#9b59b6',
    'orange': '#e67e22',
}


def save_figure(name, fig=None, dpi=150):
    """Save figure to images directory."""
    if fig is None:
        fig = plt.gcf()
    path = os.path.join(IMAGES_DIR, f'{name}.png')
    fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"  Generated: {name}.png")


# =============================================================================
# Chapter 3: Classifiers
# =============================================================================

def ch03_decision_boundary():
    """Decision boundary visualization."""
    fig, ax = plt.subplots(figsize=(6, 5))

    np.random.seed(42)
    # Class 0
    x0 = np.random.randn(30, 2) * 0.8 + np.array([-1.5, -1])
    # Class 1
    x1 = np.random.randn(30, 2) * 0.8 + np.array([1.5, 1])

    ax.scatter(x0[:, 0], x0[:, 1], c=COLORS['primary'], s=60, label='Class 0', edgecolors='white', linewidth=0.5)
    ax.scatter(x1[:, 0], x1[:, 1], c=COLORS['accent'], s=60, label='Class 1', edgecolors='white', linewidth=0.5)

    # Decision boundary
    x_line = np.linspace(-4, 4, 100)
    y_line = -x_line  # Simple linear boundary
    ax.plot(x_line, y_line, 'k--', linewidth=2, label='Decision Boundary')
    ax.fill_between(x_line, y_line, 4, alpha=0.1, color=COLORS['accent'])
    ax.fill_between(x_line, y_line, -4, alpha=0.1, color=COLORS['primary'])

    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_xlabel('Feature 1', fontsize=11)
    ax.set_ylabel('Feature 2', fontsize=11)
    ax.set_title('Linear Decision Boundary', fontsize=12, fontweight='bold')
    ax.legend(loc='upper left')
    ax.set_aspect('equal')

    save_figure('ch03_decision_boundary', fig)


def ch03_svm_margin():
    """SVM margin visualization."""
    fig, ax = plt.subplots(figsize=(6, 5))

    np.random.seed(42)
    # Support vectors (on margin)
    sv0 = np.array([[-1, 0.5], [-0.5, -0.5]])
    sv1 = np.array([[1, -0.5], [0.5, 0.5]])

    # Other points
    x0 = np.random.randn(15, 2) * 0.5 + np.array([-2, 0])
    x1 = np.random.randn(15, 2) * 0.5 + np.array([2, 0])

    ax.scatter(x0[:, 0], x0[:, 1], c=COLORS['primary'], s=50, edgecolors='white', linewidth=0.5)
    ax.scatter(x1[:, 0], x1[:, 1], c=COLORS['accent'], s=50, edgecolors='white', linewidth=0.5)
    ax.scatter(sv0[:, 0], sv0[:, 1], c=COLORS['primary'], s=150, edgecolors='black', linewidth=2, marker='o')
    ax.scatter(sv1[:, 0], sv1[:, 1], c=COLORS['accent'], s=150, edgecolors='black', linewidth=2, marker='o')

    # Decision boundary and margins
    x_line = np.linspace(-4, 4, 100)
    ax.plot(x_line, -x_line * 0.2, 'k-', linewidth=2, label='Decision Boundary')
    ax.plot(x_line, -x_line * 0.2 + 1, 'k--', linewidth=1, alpha=0.7)
    ax.plot(x_line, -x_line * 0.2 - 1, 'k--', linewidth=1, alpha=0.7)

    # Margin annotation
    ax.annotate('', xy=(0, 1), xytext=(0, -1),
                arrowprops=dict(arrowstyle='<->', color=COLORS['secondary'], lw=2))
    ax.text(0.2, 0, 'Margin', fontsize=10, color=COLORS['secondary'], fontweight='bold')

    ax.set_xlim(-4, 4)
    ax.set_ylim(-3, 3)
    ax.set_xlabel('Feature 1', fontsize=11)
    ax.set_ylabel('Feature 2', fontsize=11)
    ax.set_title('SVM: Maximizing the Margin', fontsize=12, fontweight='bold')
    ax.text(-3.5, 2.5, 'Support Vectors\n(circled points)', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    save_figure('ch03_svm_margin', fig)


# =============================================================================
# Chapter 5: Dimensionality Reduction
# =============================================================================

def ch05_pca_projection():
    """PCA projection visualization."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    np.random.seed(42)
    # Correlated 2D data
    cov = [[1, 0.8], [0.8, 1]]
    data = np.random.multivariate_normal([0, 0], cov, 100)

    # PCA direction
    eigenvalues, eigenvectors = np.linalg.eigh(np.cov(data.T))
    idx = np.argsort(eigenvalues)[::-1]
    pc1 = eigenvectors[:, idx[0]]

    # Original data
    ax = axes[0]
    ax.scatter(data[:, 0], data[:, 1], c=COLORS['primary'], s=30, alpha=0.7)
    ax.arrow(0, 0, pc1[0]*2, pc1[1]*2, head_width=0.15, head_length=0.1,
             fc=COLORS['accent'], ec=COLORS['accent'], linewidth=2)
    ax.set_xlabel('Feature 1', fontsize=11)
    ax.set_ylabel('Feature 2', fontsize=11)
    ax.set_title('Original 2D Data + PC1', fontsize=12, fontweight='bold')
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal')
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
    ax.axvline(x=0, color='gray', linestyle='-', linewidth=0.5)
    ax.text(pc1[0]*2.2, pc1[1]*2.2, 'PC1', fontsize=11, color=COLORS['accent'], fontweight='bold')

    # Projected data
    ax = axes[1]
    projected = data @ pc1
    ax.scatter(projected, np.zeros_like(projected), c=COLORS['secondary'], s=30, alpha=0.7)
    ax.set_xlabel('PC1', fontsize=11)
    ax.set_title('Projected to 1D', fontsize=12, fontweight='bold')
    ax.set_ylim(-1, 1)
    ax.set_yticks([])
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=1)

    plt.tight_layout()
    save_figure('ch05_pca_projection', fig)


def ch05_variance_explained():
    """Cumulative variance explained plot."""
    fig, ax = plt.subplots(figsize=(6, 4))

    # Example explained variance ratios
    var_ratio = np.array([0.45, 0.25, 0.15, 0.08, 0.04, 0.02, 0.01])
    cumsum = np.cumsum(var_ratio)
    x = np.arange(1, len(var_ratio) + 1)

    ax.bar(x, var_ratio, color=COLORS['primary'], alpha=0.7, label='Individual')
    ax.step(x, cumsum, where='mid', color=COLORS['accent'], linewidth=2, label='Cumulative')
    ax.axhline(y=0.95, color=COLORS['secondary'], linestyle='--', linewidth=1.5, label='95% threshold')

    ax.set_xlabel('Principal Component', fontsize=11)
    ax.set_ylabel('Variance Explained', fontsize=11)
    ax.set_title('Scree Plot: Variance Explained by PCs', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_ylim(0, 1.05)
    ax.legend(loc='center right')

    save_figure('ch05_variance_explained', fig)


# =============================================================================
# Chapter 6: Model Evaluation
# =============================================================================

def ch06_confusion_matrix():
    """Confusion matrix visualization."""
    fig, ax = plt.subplots(figsize=(5, 4))

    cm = np.array([[85, 15], [10, 90]])
    im = ax.imshow(cm, cmap='Blues')

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Predicted\nNegative', 'Predicted\nPositive'])
    ax.set_yticklabels(['Actual\nNegative', 'Actual\nPositive'])

    # Add text annotations
    labels = [['TN', 'FP'], ['FN', 'TP']]
    for i in range(2):
        for j in range(2):
            color = 'white' if cm[i, j] > 50 else 'black'
            ax.text(j, i, f'{labels[i][j]}\n{cm[i, j]}', ha='center', va='center',
                   fontsize=14, color=color, fontweight='bold')

    ax.set_title('Confusion Matrix', fontsize=12, fontweight='bold')
    plt.tight_layout()
    save_figure('ch06_confusion_matrix', fig)


def ch06_roc_curve():
    """ROC curve visualization."""
    fig, ax = plt.subplots(figsize=(5, 5))

    # Simulated ROC curves
    fpr = np.linspace(0, 1, 100)
    tpr_good = 1 - (1 - fpr) ** 3  # Good classifier
    tpr_bad = fpr ** 0.5  # Bad classifier

    ax.plot(fpr, tpr_good, color=COLORS['primary'], linewidth=2, label='Good Model (AUC=0.92)')
    ax.plot(fpr, tpr_bad, color=COLORS['orange'], linewidth=2, label='Poor Model (AUC=0.67)')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC=0.50)')

    ax.fill_between(fpr, tpr_good, alpha=0.2, color=COLORS['primary'])

    ax.set_xlabel('False Positive Rate (FPR)', fontsize=11)
    ax.set_ylabel('True Positive Rate (TPR)', fontsize=11)
    ax.set_title('ROC Curve', fontsize=12, fontweight='bold')
    ax.legend(loc='lower right')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')

    save_figure('ch06_roc_curve', fig)


def ch06_bias_variance():
    """Bias-variance tradeoff visualization."""
    fig, ax = plt.subplots(figsize=(6, 4))

    complexity = np.linspace(0.1, 3, 100)
    bias2 = 1 / complexity
    variance = 0.1 * complexity ** 2
    total_error = bias2 + variance + 0.1  # irreducible error

    ax.plot(complexity, bias2, color=COLORS['primary'], linewidth=2, label='Bias²')
    ax.plot(complexity, variance, color=COLORS['accent'], linewidth=2, label='Variance')
    ax.plot(complexity, total_error, color=COLORS['dark'], linewidth=2, linestyle='--', label='Total Error')
    ax.axhline(y=0.1, color=COLORS['neutral'], linewidth=1, linestyle=':', label='Irreducible Error')

    # Optimal point
    opt_idx = np.argmin(total_error)
    ax.axvline(x=complexity[opt_idx], color=COLORS['secondary'], linewidth=1.5, linestyle='--', alpha=0.7)
    ax.scatter([complexity[opt_idx]], [total_error[opt_idx]], color=COLORS['secondary'], s=100, zorder=5)

    ax.set_xlabel('Model Complexity', fontsize=11)
    ax.set_ylabel('Error', fontsize=11)
    ax.set_title('Bias-Variance Tradeoff', fontsize=12, fontweight='bold')
    ax.legend(loc='upper center')
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 2)

    # Labels
    ax.text(0.5, 1.5, 'Underfitting', fontsize=10, ha='center')
    ax.text(2.5, 1.5, 'Overfitting', fontsize=10, ha='center')

    save_figure('ch06_bias_variance', fig)


# =============================================================================
# Chapter 10: Clustering
# =============================================================================

def ch10_kmeans_steps():
    """K-means algorithm steps."""
    np.random.seed(42)

    # Generate clustered data
    c1 = np.random.randn(20, 2) * 0.5 + np.array([-2, 2])
    c2 = np.random.randn(20, 2) * 0.5 + np.array([2, 2])
    c3 = np.random.randn(20, 2) * 0.5 + np.array([0, -1])
    data = np.vstack([c1, c2, c3])

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    colors = [COLORS['primary'], COLORS['accent'], COLORS['secondary']]

    # Step 1: Random centroids
    ax = axes[0]
    ax.scatter(data[:, 0], data[:, 1], c=COLORS['neutral'], s=40)
    centroids = np.array([[-1, 0], [1, 1], [0, 2]])
    ax.scatter(centroids[:, 0], centroids[:, 1], c=colors, s=200, marker='X', edgecolors='black', linewidth=2)
    ax.set_title('Step 1: Initialize Centroids', fontsize=11, fontweight='bold')
    ax.set_xlim(-4, 4)
    ax.set_ylim(-3, 4)

    # Step 2: Assign clusters
    ax = axes[1]
    # Simple assignment based on distance
    assignments = []
    for p in data:
        dists = [np.linalg.norm(p - c) for c in centroids]
        assignments.append(np.argmin(dists))
    assignments = np.array(assignments)

    for i, color in enumerate(colors):
        mask = assignments == i
        ax.scatter(data[mask, 0], data[mask, 1], c=color, s=40)
    ax.scatter(centroids[:, 0], centroids[:, 1], c=colors, s=200, marker='X', edgecolors='black', linewidth=2)
    ax.set_title('Step 2: Assign Points', fontsize=11, fontweight='bold')
    ax.set_xlim(-4, 4)
    ax.set_ylim(-3, 4)

    # Step 3: Update centroids
    ax = axes[2]
    new_centroids = np.array([data[assignments == i].mean(axis=0) for i in range(3)])
    for i, color in enumerate(colors):
        mask = assignments == i
        ax.scatter(data[mask, 0], data[mask, 1], c=color, s=40)
    ax.scatter(new_centroids[:, 0], new_centroids[:, 1], c=colors, s=200, marker='X', edgecolors='black', linewidth=2)
    # Show movement
    for old, new, color in zip(centroids, new_centroids, colors):
        ax.annotate('', xy=new, xytext=old, arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    ax.set_title('Step 3: Update Centroids', fontsize=11, fontweight='bold')
    ax.set_xlim(-4, 4)
    ax.set_ylim(-3, 4)

    plt.tight_layout()
    save_figure('ch10_kmeans_steps', fig)


def ch10_elbow_method():
    """Elbow method visualization."""
    fig, ax = plt.subplots(figsize=(6, 4))

    k_values = np.arange(1, 11)
    inertia = [1000, 400, 200, 120, 100, 90, 85, 82, 80, 79]

    ax.plot(k_values, inertia, 'o-', color=COLORS['primary'], linewidth=2, markersize=8)
    ax.axvline(x=4, color=COLORS['accent'], linestyle='--', linewidth=1.5, label='Elbow point (k=4)')

    ax.set_xlabel('Number of Clusters (k)', fontsize=11)
    ax.set_ylabel('Inertia (Within-cluster SSE)', fontsize=11)
    ax.set_title('Elbow Method for Optimal k', fontsize=12, fontweight='bold')
    ax.set_xticks(k_values)
    ax.legend()

    save_figure('ch10_elbow_method', fig)


# =============================================================================
# Chapter 11: Neural Networks
# =============================================================================

def ch11_mlp_architecture():
    """MLP architecture visualization."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(-0.5, 4.5)

    layer_sizes = [3, 4, 4, 2]
    layer_names = ['Input\nLayer', 'Hidden\nLayer 1', 'Hidden\nLayer 2', 'Output\nLayer']
    layer_colors = [COLORS['primary'], COLORS['secondary'], COLORS['secondary'], COLORS['accent']]

    positions = []
    for l, size in enumerate(layer_sizes):
        y_positions = np.linspace(4 - size * 0.4, size * 0.4, size) + 2 - size * 0.2
        positions.append([(l * 1.3 + 0.5, y) for y in y_positions])

    # Draw connections
    for l in range(len(layer_sizes) - 1):
        for pos1 in positions[l]:
            for pos2 in positions[l + 1]:
                ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]],
                       color=COLORS['neutral'], linewidth=0.5, alpha=0.5)

    # Draw neurons
    for l, (layer_pos, color) in enumerate(zip(positions, layer_colors)):
        for pos in layer_pos:
            circle = Circle(pos, 0.15, facecolor=color, edgecolor='black', linewidth=1.5, zorder=10)
            ax.add_patch(circle)
        ax.text(l * 1.3 + 0.5, -0.3, layer_names[l], ha='center', fontsize=9)

    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Multilayer Perceptron (MLP) Architecture', fontsize=12, fontweight='bold', pad=20)

    save_figure('ch11_mlp_architecture', fig)


def ch11_activation_functions():
    """Common activation functions."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))

    x = np.linspace(-5, 5, 200)

    # Sigmoid
    ax = axes[0]
    sigmoid = 1 / (1 + np.exp(-x))
    ax.plot(x, sigmoid, color=COLORS['primary'], linewidth=2)
    ax.axhline(y=0.5, color=COLORS['neutral'], linestyle='--', linewidth=1)
    ax.axvline(x=0, color=COLORS['neutral'], linestyle='--', linewidth=1)
    ax.set_title('Sigmoid: σ(z) = 1/(1+e⁻ᶻ)', fontsize=11, fontweight='bold')
    ax.set_xlabel('z')
    ax.set_ylabel('σ(z)')
    ax.set_ylim(-0.1, 1.1)

    # Tanh
    ax = axes[1]
    ax.plot(x, np.tanh(x), color=COLORS['secondary'], linewidth=2)
    ax.axhline(y=0, color=COLORS['neutral'], linestyle='--', linewidth=1)
    ax.axvline(x=0, color=COLORS['neutral'], linestyle='--', linewidth=1)
    ax.set_title('Tanh: tanh(z)', fontsize=11, fontweight='bold')
    ax.set_xlabel('z')
    ax.set_ylabel('tanh(z)')
    ax.set_ylim(-1.2, 1.2)

    # ReLU
    ax = axes[2]
    relu = np.maximum(0, x)
    ax.plot(x, relu, color=COLORS['accent'], linewidth=2)
    ax.axhline(y=0, color=COLORS['neutral'], linestyle='--', linewidth=1)
    ax.axvline(x=0, color=COLORS['neutral'], linestyle='--', linewidth=1)
    ax.set_title('ReLU: max(0, z)', fontsize=11, fontweight='bold')
    ax.set_xlabel('z')
    ax.set_ylabel('ReLU(z)')
    ax.set_ylim(-1, 5)

    plt.tight_layout()
    save_figure('ch11_activation_functions', fig)


# =============================================================================
# Chapter 14: CNNs
# =============================================================================

def ch14_convolution():
    """Convolution operation visualization."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_xlim(-0.5, 12)
    ax.set_ylim(-0.5, 5)

    # Input matrix (5x5)
    input_size = 5
    for i in range(input_size):
        for j in range(input_size):
            rect = Rectangle((j * 0.6, (input_size - 1 - i) * 0.6), 0.55, 0.55,
                            facecolor=COLORS['light'], edgecolor=COLORS['dark'], linewidth=1)
            ax.add_patch(rect)
            ax.text(j * 0.6 + 0.27, (input_size - 1 - i) * 0.6 + 0.27,
                   str(np.random.randint(0, 9)), ha='center', va='center', fontsize=8)

    # Highlight kernel position
    for i in range(3):
        for j in range(3):
            rect = Rectangle((j * 0.6, (4 - i) * 0.6), 0.55, 0.55,
                            facecolor=COLORS['primary'], edgecolor=COLORS['dark'],
                            linewidth=1, alpha=0.5)
            ax.add_patch(rect)

    ax.text(1.5 * 0.6, -0.3, 'Input (5×5)', ha='center', fontsize=10)

    # Kernel (3x3)
    kernel_x = 4.5
    kernel = [[1, 0, -1], [2, 0, -2], [1, 0, -1]]
    for i in range(3):
        for j in range(3):
            rect = Rectangle((kernel_x + j * 0.6, (2 - i) * 0.6 + 0.9), 0.55, 0.55,
                            facecolor=COLORS['secondary'], edgecolor=COLORS['dark'], linewidth=1)
            ax.add_patch(rect)
            ax.text(kernel_x + j * 0.6 + 0.27, (2 - i) * 0.6 + 0.27 + 0.9,
                   str(kernel[i][j]), ha='center', va='center', fontsize=8)

    ax.text(kernel_x + 0.9, 0.5, 'Kernel (3×3)', ha='center', fontsize=10)

    # Arrow
    ax.annotate('', xy=(8.5, 1.8), xytext=(7.2, 1.8),
                arrowprops=dict(arrowstyle='->', color=COLORS['dark'], lw=2))
    ax.text(7.85, 2.1, '*', fontsize=20, ha='center')

    # Output (3x3)
    output_x = 9
    for i in range(3):
        for j in range(3):
            rect = Rectangle((output_x + j * 0.6, (2 - i) * 0.6 + 0.9), 0.55, 0.55,
                            facecolor=COLORS['accent'], edgecolor=COLORS['dark'],
                            linewidth=1, alpha=0.7)
            ax.add_patch(rect)
            ax.text(output_x + j * 0.6 + 0.27, (2 - i) * 0.6 + 0.27 + 0.9,
                   '?', ha='center', va='center', fontsize=8, color='white')

    ax.text(output_x + 0.9, 0.5, 'Output (3×3)', ha='center', fontsize=10)

    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Convolution Operation', fontsize=12, fontweight='bold')

    save_figure('ch14_convolution', fig)


def ch14_pooling():
    """Max pooling and average pooling."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    input_vals = [[1, 3, 2, 4], [5, 6, 1, 2], [3, 2, 8, 1], [4, 5, 3, 7]]

    for ax_idx, (ax, pool_type) in enumerate(zip(axes, ['Max Pooling', 'Avg Pooling'])):
        ax.set_xlim(-0.5, 6)
        ax.set_ylim(-0.5, 3)

        # Input 4x4
        colors_input = [[COLORS['primary'], COLORS['secondary'], COLORS['primary'], COLORS['secondary']],
                        [COLORS['primary'], COLORS['secondary'], COLORS['primary'], COLORS['secondary']],
                        [COLORS['accent'], COLORS['orange'], COLORS['accent'], COLORS['orange']],
                        [COLORS['accent'], COLORS['orange'], COLORS['accent'], COLORS['orange']]]

        for i in range(4):
            for j in range(4):
                rect = Rectangle((j * 0.5, (3 - i) * 0.5), 0.45, 0.45,
                                facecolor=colors_input[i][j], edgecolor=COLORS['dark'],
                                linewidth=1, alpha=0.5)
                ax.add_patch(rect)
                ax.text(j * 0.5 + 0.22, (3 - i) * 0.5 + 0.22, str(input_vals[i][j]),
                       ha='center', va='center', fontsize=9)

        ax.text(1, -0.3, 'Input (4×4)', ha='center', fontsize=10)

        # Arrow
        ax.annotate('', xy=(3.3, 1), xytext=(2.5, 1),
                    arrowprops=dict(arrowstyle='->', color=COLORS['dark'], lw=2))
        ax.text(2.9, 1.3, '2×2\npool', ha='center', fontsize=8)

        # Output 2x2
        output_colors = [COLORS['primary'], COLORS['secondary'], COLORS['accent'], COLORS['orange']]
        if pool_type == 'Max Pooling':
            output_vals = [[6, 4], [5, 8]]
        else:
            output_vals = [[3.75, 2.25], [3.5, 4.75]]

        for i in range(2):
            for j in range(2):
                rect = Rectangle((3.8 + j * 0.7, (1 - i) * 0.7 + 0.5), 0.65, 0.65,
                                facecolor=output_colors[i * 2 + j], edgecolor=COLORS['dark'],
                                linewidth=1, alpha=0.7)
                ax.add_patch(rect)
                val = output_vals[i][j]
                ax.text(3.8 + j * 0.7 + 0.32, (1 - i) * 0.7 + 0.5 + 0.32,
                       f'{val:.1f}' if isinstance(val, float) else str(val),
                       ha='center', va='center', fontsize=10, fontweight='bold')

        ax.text(4.45, 0.2, 'Output (2×2)', ha='center', fontsize=10)
        ax.set_title(pool_type, fontsize=12, fontweight='bold')
        ax.set_aspect('equal')
        ax.axis('off')

    plt.tight_layout()
    save_figure('ch14_pooling', fig)


# =============================================================================
# Chapter 15: RNNs
# =============================================================================

def ch15_rnn_unrolled():
    """RNN unrolled through time."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_xlim(-0.5, 10)
    ax.set_ylim(-0.5, 4)

    # Draw 4 time steps
    for t, x_pos in enumerate([1, 3.5, 6, 8.5]):
        # Hidden state
        circle = Circle((x_pos, 2), 0.4, facecolor=COLORS['secondary'],
                        edgecolor='black', linewidth=1.5, zorder=10)
        ax.add_patch(circle)
        ax.text(x_pos, 2, f'h{t}', ha='center', va='center', fontsize=10, fontweight='bold')

        # Input
        rect = Rectangle((x_pos - 0.3, 0.3), 0.6, 0.5, facecolor=COLORS['primary'],
                         edgecolor='black', linewidth=1.5, zorder=10)
        ax.add_patch(rect)
        ax.text(x_pos, 0.55, f'x{t}', ha='center', va='center', fontsize=10, fontweight='bold')

        # Output
        rect = Rectangle((x_pos - 0.3, 3.2), 0.6, 0.5, facecolor=COLORS['accent'],
                         edgecolor='black', linewidth=1.5, zorder=10)
        ax.add_patch(rect)
        ax.text(x_pos, 3.45, f'y{t}', ha='center', va='center', fontsize=10, fontweight='bold')

        # Arrows: input to hidden
        ax.annotate('', xy=(x_pos, 1.55), xytext=(x_pos, 0.85),
                    arrowprops=dict(arrowstyle='->', color=COLORS['dark'], lw=1.5))

        # Arrows: hidden to output
        ax.annotate('', xy=(x_pos, 3.15), xytext=(x_pos, 2.45),
                    arrowprops=dict(arrowstyle='->', color=COLORS['dark'], lw=1.5))

        # Arrows: hidden to next hidden
        if t < 3:
            next_x = [3.5, 6, 8.5][t]
            ax.annotate('', xy=(next_x - 0.45, 2), xytext=(x_pos + 0.45, 2),
                        arrowprops=dict(arrowstyle='->', color=COLORS['purple'], lw=2))

    ax.text(5, -0.2, 'Time →', ha='center', fontsize=11, style='italic')
    ax.text(-0.2, 2, 'Hidden\nState', ha='center', va='center', fontsize=9)
    ax.text(-0.2, 0.55, 'Input', ha='center', va='center', fontsize=9)
    ax.text(-0.2, 3.45, 'Output', ha='center', va='center', fontsize=9)

    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('RNN Unrolled Through Time', fontsize=12, fontweight='bold')

    save_figure('ch15_rnn_unrolled', fig)


def ch15_lstm_cell():
    """LSTM cell architecture (simplified)."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(-0.5, 8)
    ax.set_ylim(-0.5, 6)

    # Cell state line (top)
    ax.annotate('', xy=(7.5, 5), xytext=(0.5, 5),
                arrowprops=dict(arrowstyle='->', color=COLORS['primary'], lw=3))
    ax.text(4, 5.3, 'Cell State (Cₜ)', ha='center', fontsize=10, fontweight='bold', color=COLORS['primary'])

    # Gates as boxes
    gate_y = 3
    gate_info = [
        (1.5, 'Forget\nGate', COLORS['accent']),
        (4, 'Input\nGate', COLORS['secondary']),
        (6.5, 'Output\nGate', COLORS['purple']),
    ]

    for x, label, color in gate_info:
        rect = FancyBboxPatch((x - 0.5, gate_y - 0.4), 1, 0.8,
                               boxstyle="round,pad=0.05", facecolor=color,
                               edgecolor='black', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x, gate_y, label, ha='center', va='center', fontsize=8, fontweight='bold')

    # Hidden state output
    ax.annotate('', xy=(7.5, 1.5), xytext=(6.5, 2.5),
                arrowprops=dict(arrowstyle='->', color=COLORS['dark'], lw=2))
    ax.text(7.5, 1.2, 'hₜ', fontsize=11, fontweight='bold')

    # Inputs
    ax.annotate('', xy=(4, 2.5), xytext=(4, 0.5),
                arrowprops=dict(arrowstyle='->', color=COLORS['dark'], lw=2))
    ax.text(4, 0.2, 'xₜ, hₜ₋₁', ha='center', fontsize=10)

    # Previous cell state
    ax.text(0.2, 5, 'Cₜ₋₁', fontsize=10, fontweight='bold', color=COLORS['primary'])

    # Previous hidden state
    ax.text(0.2, 1.5, 'hₜ₋₁', fontsize=10, fontweight='bold')
    ax.annotate('', xy=(1, 2.5), xytext=(0.5, 1.5),
                arrowprops=dict(arrowstyle='->', color=COLORS['dark'], lw=1.5))

    # Connections from gates to cell state
    ax.plot([1.5, 1.5], [gate_y + 0.4, 4.8], color=COLORS['accent'], lw=1.5, linestyle='--')
    ax.plot([4, 4], [gate_y + 0.4, 4.8], color=COLORS['secondary'], lw=1.5, linestyle='--')
    ax.plot([6.5, 6.5], [gate_y + 0.4, 4.5], color=COLORS['purple'], lw=1.5, linestyle='--')

    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('LSTM Cell (Simplified)', fontsize=12, fontweight='bold')

    save_figure('ch15_lstm_cell', fig)


# =============================================================================
# Chapter 17: GANs
# =============================================================================

def ch17_gan_architecture():
    """GAN architecture and training loop."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(-0.5, 10)
    ax.set_ylim(-0.5, 5)

    # Noise input
    rect = FancyBboxPatch((0.2, 2), 1, 1, boxstyle="round,pad=0.1",
                          facecolor=COLORS['neutral'], edgecolor='black', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(0.7, 2.5, 'Noise\nz', ha='center', va='center', fontsize=9, fontweight='bold')

    # Generator
    rect = FancyBboxPatch((2, 1.8), 1.5, 1.4, boxstyle="round,pad=0.1",
                          facecolor=COLORS['secondary'], edgecolor='black', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(2.75, 2.5, 'Generator\nG', ha='center', va='center', fontsize=10, fontweight='bold')

    # Arrow: noise to generator
    ax.annotate('', xy=(1.95, 2.5), xytext=(1.25, 2.5),
                arrowprops=dict(arrowstyle='->', color=COLORS['dark'], lw=2))

    # Fake image
    rect = FancyBboxPatch((4, 2), 1, 1, boxstyle="round,pad=0.1",
                          facecolor=COLORS['orange'], edgecolor='black', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(4.5, 2.5, 'Fake\nImage', ha='center', va='center', fontsize=9, fontweight='bold')

    # Arrow: generator to fake
    ax.annotate('', xy=(3.95, 2.5), xytext=(3.55, 2.5),
                arrowprops=dict(arrowstyle='->', color=COLORS['dark'], lw=2))

    # Real image
    rect = FancyBboxPatch((4, 3.8), 1, 1, boxstyle="round,pad=0.1",
                          facecolor=COLORS['primary'], edgecolor='black', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(4.5, 4.3, 'Real\nImage', ha='center', va='center', fontsize=9, fontweight='bold')

    # Discriminator
    rect = FancyBboxPatch((6, 2.5), 1.5, 1.4, boxstyle="round,pad=0.1",
                          facecolor=COLORS['accent'], edgecolor='black', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(6.75, 3.2, 'Discriminator\nD', ha='center', va='center', fontsize=10, fontweight='bold')

    # Arrows to discriminator
    ax.annotate('', xy=(5.95, 3), xytext=(5.05, 2.5),
                arrowprops=dict(arrowstyle='->', color=COLORS['dark'], lw=2))
    ax.annotate('', xy=(5.95, 3.4), xytext=(5.05, 4.3),
                arrowprops=dict(arrowstyle='->', color=COLORS['dark'], lw=2))

    # Output
    rect = FancyBboxPatch((8, 2.7), 1.2, 1, boxstyle="round,pad=0.1",
                          facecolor=COLORS['light'], edgecolor='black', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(8.6, 3.2, 'Real or\nFake?', ha='center', va='center', fontsize=9, fontweight='bold')

    # Arrow: discriminator to output
    ax.annotate('', xy=(7.95, 3.2), xytext=(7.55, 3.2),
                arrowprops=dict(arrowstyle='->', color=COLORS['dark'], lw=2))

    # Feedback arrows (training signal)
    ax.annotate('', xy=(6.75, 2.45), xytext=(6.75, 0.8),
                arrowprops=dict(arrowstyle='->', color=COLORS['purple'], lw=2, linestyle='--'))
    ax.annotate('', xy=(2.75, 1.75), xytext=(2.75, 0.8),
                arrowprops=dict(arrowstyle='->', color=COLORS['purple'], lw=2, linestyle='--'))
    ax.plot([2.75, 6.75], [0.8, 0.8], color=COLORS['purple'], lw=2, linestyle='--')
    ax.text(4.75, 0.5, 'Training Signal (Backprop)', ha='center', fontsize=9,
            color=COLORS['purple'], style='italic')

    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('GAN Architecture', fontsize=12, fontweight='bold')

    save_figure('ch17_gan_architecture', fig)


# =============================================================================
# Chapter 19: Reinforcement Learning
# =============================================================================

def ch19_rl_loop():
    """RL agent-environment interaction loop."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_xlim(-0.5, 8)
    ax.set_ylim(-0.5, 5)

    # Agent
    rect = FancyBboxPatch((0.5, 1.5), 2, 1.5, boxstyle="round,pad=0.1",
                          facecolor=COLORS['primary'], edgecolor='black', linewidth=2)
    ax.add_patch(rect)
    ax.text(1.5, 2.25, 'Agent', ha='center', va='center', fontsize=14, fontweight='bold', color='white')

    # Environment
    rect = FancyBboxPatch((5.5, 1.5), 2, 1.5, boxstyle="round,pad=0.1",
                          facecolor=COLORS['secondary'], edgecolor='black', linewidth=2)
    ax.add_patch(rect)
    ax.text(6.5, 2.25, 'Environment', ha='center', va='center', fontsize=14, fontweight='bold', color='white')

    # Action arrow (agent to environment)
    ax.annotate('', xy=(5.4, 2.7), xytext=(2.6, 2.7),
                arrowprops=dict(arrowstyle='->', color=COLORS['accent'], lw=3))
    ax.text(4, 3.1, 'Action (aₜ)', ha='center', fontsize=11, fontweight='bold', color=COLORS['accent'])

    # State arrow (environment to agent, top)
    ax.annotate('', xy=(2.6, 4), xytext=(5.4, 4),
                arrowprops=dict(arrowstyle='->', color=COLORS['purple'], lw=3))
    ax.text(4, 4.4, 'State (sₜ₊₁)', ha='center', fontsize=11, fontweight='bold', color=COLORS['purple'])

    # Reward arrow (environment to agent, bottom)
    ax.annotate('', xy=(2.6, 1.8), xytext=(5.4, 1.8),
                arrowprops=dict(arrowstyle='->', color=COLORS['orange'], lw=3))
    ax.text(4, 1.3, 'Reward (rₜ)', ha='center', fontsize=11, fontweight='bold', color=COLORS['orange'])

    # Curved arrows to show loop
    from matplotlib.patches import FancyArrowPatch, Arc
    arc1 = Arc((4, 2.25), 6, 4, angle=0, theta1=160, theta2=200, color=COLORS['neutral'], lw=1.5, linestyle='--')
    ax.add_patch(arc1)
    arc2 = Arc((4, 2.25), 6, 4, angle=0, theta1=-20, theta2=20, color=COLORS['neutral'], lw=1.5, linestyle='--')
    ax.add_patch(arc2)

    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Reinforcement Learning: Agent-Environment Loop', fontsize=12, fontweight='bold')

    save_figure('ch19_rl_loop', fig)


def ch19_q_learning():
    """Q-learning update visualization."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_xlim(-0.5, 10)
    ax.set_ylim(-0.5, 4)

    # States
    for i, (x, label) in enumerate([(1, 'S'), (4, "S'")]):
        circle = Circle((x, 2), 0.5, facecolor=COLORS['primary'] if i == 0 else COLORS['secondary'],
                        edgecolor='black', linewidth=2, zorder=10)
        ax.add_patch(circle)
        ax.text(x, 2, label, ha='center', va='center', fontsize=14, fontweight='bold', color='white')

    # Action arrow
    ax.annotate('', xy=(3.4, 2), xytext=(1.6, 2),
                arrowprops=dict(arrowstyle='->', color=COLORS['accent'], lw=3))
    ax.text(2.5, 2.5, 'action a', ha='center', fontsize=10, fontweight='bold', color=COLORS['accent'])

    # Reward
    ax.text(2.5, 1.4, 'reward r', ha='center', fontsize=10, fontweight='bold', color=COLORS['orange'])

    # Q-value update equation
    eq_text = r"$Q(S,a) \leftarrow Q(S,a) + \alpha \cdot (r + \gamma \cdot \max_{a'} Q(S',a') - Q(S,a))$"
    ax.text(5, 0.5, "Q-learning update:", fontsize=10, fontweight='bold')
    ax.text(5, 0, eq_text, fontsize=11)

    # Legend
    ax.text(6, 3.5, 'α = learning rate', fontsize=9)
    ax.text(6, 3.1, 'γ = discount factor', fontsize=9)

    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Q-Learning: Update Rule', fontsize=12, fontweight='bold')

    save_figure('ch19_q_learning', fig)


# =============================================================================
# Main execution
# =============================================================================

# Registry of all diagram functions
DIAGRAMS = {
    'ch03_decision_boundary': ch03_decision_boundary,
    'ch03_svm_margin': ch03_svm_margin,
    'ch05_pca_projection': ch05_pca_projection,
    'ch05_variance_explained': ch05_variance_explained,
    'ch06_confusion_matrix': ch06_confusion_matrix,
    'ch06_roc_curve': ch06_roc_curve,
    'ch06_bias_variance': ch06_bias_variance,
    'ch10_kmeans_steps': ch10_kmeans_steps,
    'ch10_elbow_method': ch10_elbow_method,
    'ch11_mlp_architecture': ch11_mlp_architecture,
    'ch11_activation_functions': ch11_activation_functions,
    'ch14_convolution': ch14_convolution,
    'ch14_pooling': ch14_pooling,
    'ch15_rnn_unrolled': ch15_rnn_unrolled,
    'ch15_lstm_cell': ch15_lstm_cell,
    'ch17_gan_architecture': ch17_gan_architecture,
    'ch19_rl_loop': ch19_rl_loop,
    'ch19_q_learning': ch19_q_learning,
}


def generate_all():
    """Generate all diagrams."""
    print(f"Generating {len(DIAGRAMS)} diagrams to {IMAGES_DIR}/\n")
    for name, func in DIAGRAMS.items():
        try:
            func()
        except Exception as e:
            print(f"  ERROR generating {name}: {e}")
    print(f"\nDone! Generated {len(DIAGRAMS)} diagrams.")


def generate_chapter(chapter: str):
    """Generate diagrams for a specific chapter."""
    prefix = chapter.lower()
    matching = {k: v for k, v in DIAGRAMS.items() if k.startswith(prefix)}
    if not matching:
        print(f"No diagrams found for '{chapter}'")
        print(f"Available prefixes: ch03, ch05, ch06, ch10, ch11, ch14, ch15, ch17, ch19")
        return
    print(f"Generating {len(matching)} diagrams for {chapter}:\n")
    for name, func in matching.items():
        func()


def list_diagrams():
    """List all available diagrams."""
    print("Available diagrams:\n")
    current_chapter = None
    for name in sorted(DIAGRAMS.keys()):
        chapter = name.split('_')[0]
        if chapter != current_chapter:
            current_chapter = chapter
            print(f"\n{chapter.upper()}:")
        print(f"  - {name}")
    print(f"\nTotal: {len(DIAGRAMS)} diagrams")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate ML diagrams for Anki flashcards')
    parser.add_argument('chapter', nargs='?', default=None, help='Chapter prefix (e.g., ch05)')
    parser.add_argument('--list', action='store_true', help='List available diagrams')

    args = parser.parse_args()

    if args.list:
        list_diagrams()
    elif args.chapter:
        generate_chapter(args.chapter)
    else:
        generate_all()
