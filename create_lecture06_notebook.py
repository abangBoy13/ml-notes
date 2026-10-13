#!/usr/bin/env python3
"""
Script to create a properly formatted Jupyter notebook for Lecture 06:
Clustering
"""

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell
import json

def create_lecture06_notebook():
    # Create a new notebook
    notebook = nbformat.v4.new_notebook()

    # List to hold all cells
    cells = []

    # First code cell: pip install
    cells.append(new_code_cell(
        source="!pip install numpy matplotlib scikit-learn seaborn"
    ))

    # Markdown cell: Title and introduction
    cells.append(new_markdown_cell(
        source="# Lecture 06: Clustering\n\n"
               "## Unsupervised Learning: Finding Hidden Patterns\n\n"
               "## Big Idea\n"
               "Clustering groups similar data points together without predefined labels, "
               "revealing inherent structure in unlabeled data through similarity measures.\n\n"
               "## Everyday Metaphor\n\n"
           "> 💡 **Clustering = organizing a messy desk**: Imagine your desk is covered with "
           "various objects. Clustering is like grouping similar items together - pens with pens, "
           "papers with papers, etc. - without being told what categories to use.\n"
           "> \n"
           "> **Similarity Measure = how we decide what's 'like' what**: The rule we use to "
           "determine if two items belong together (e.g., same color, same function, same size).\n"
           "> \n"
           "> **Cluster Centroid = the representative of a group**: The average or most typical "
           "item in a cluster that represents the group's characteristics.\n\n"
           "## Why It Matters\n"
           "- Discovers hidden patterns in unlabeled data\n"
           "- Useful for customer segmentation, image compression, anomaly detection\n"
           "- Prepares data for other machine learning algorithms\n"
           "- Helps understand data structure before supervised learning\n"
           "- Essential for exploratory data analysis\n\n"
           "## Formulas\n"
    ))

    # Markdown cell: Distance Metrics
    cells.append(new_markdown_cell(
        source="### Distance and Similarity Metrics\n\n"
               "Clustering relies on measuring similarity between data points. Common metrics:\n\n"
               "$$\n"
               "d(\\mathbf{x}, \\mathbf{y}) = \\sqrt{\\sum_{i=1}^n (x_i - y_i)^2}\n"
               "$$\n\n"
               "*Euclidean distance*\n\n"
               "$$\n"
               "d(\\mathbf{x}, \\mathbf{y}) = \\sum_{i=1}^n |x_i - y_i|\n"
               "$$\n\n"
               "*Manhattan distance*\n\n"
               "$$\n"
               "d(\\mathbf{x}, \\mathbf{y}) = 1 - \\frac{\\mathbf{x} \\cdot \\mathbf{y}}{\\|\\mathbf{x}\\| \\|\\mathbf{y}\\|}\n"
               "$$\n\n"
               "*Cosine distance*\n\n"
               "*What each symbol means:*\n"
               "- **\\mathbf{x}, \\mathbf{y}**: Data points in n-dimensional space\n"
               "- **x_i, y_i**: Individual features/coordinates\n"
               "- **\\cdot**: Dot product\n"
               "- **\\|\\cdot\\|**: Vector magnitude (Euclidean norm)\n"
    ))

    # Markdown cell: K-Means Clustering
    cells.append(new_markdown_cell(
        source="### K-Means Clustering Algorithm\n\n"
               "K-means partitions data into K clusters by minimizing within-cluster variance.\n\n"
               "$$\n"
               "\\arg\\min_{\\mathbf{S}} \\sum_{i=1}^k \\sum_{\\mathbf{x} \\in S_i} \\|\\mathbf{x} - \\boldsymbol{\\mu}_i\\|^2\n"
               "$$\n\n"
               "*Objective function: Minimize sum of squared distances to cluster centroids*\n\n"
               "*What each symbol means:*\n"
               "- **\\mathbf{S} = {S_1, S_2, ..., S_k}**: Partition of data into k clusters\n"
               "- **\\boldsymbol{\\mu}_i**: Centroid (mean) of cluster S_i\n"
               "- **\\|\\cdot\\|**: Euclidean distance\n"
               "- **k**: Number of clusters (chosen in advance)\n\n"
               "### K-Means Algorithm Steps:\n"
               "1. Initialize k cluster centroids randomly\n"
               "2. Assign each point to nearest centroid\n"
               "3. Update centroids as mean of assigned points\n"
               "4. Repeat steps 2-3 until convergence\n"
    ))

    # Markdown cell: Hierarchical Clustering
    cells.append(new_markdown_cell(
        source="### Hierarchical Clustering\n\n"
               "Builds a tree of clusters (dendrogram) by iteratively merging or splitting clusters.\n\n"
               "### Agglomerative (Bottom-Up) Approach:\n"
               "1. Start with each point as its own cluster\n"
               "2. Iteratively merge closest pair of clusters\n"
               "3. Update distances between clusters\n"
               "4. Continue until one cluster remains\n\n"
               "### Linkage Criteria (How to measure distance between clusters):\n"
               "- **Single linkage**: Minimum distance between points in different clusters\n"
               "- **Complete linkage**: Maximum distance between points in different clusters\n"
               "- **Average linkage**: Average distance between all pairs of points\n"
               "- **Ward's method**: Minimize increase in total within-cluster variance\n"
    ))

    # Markdown cell: DBSCAN Clustering
    cells.append(new_markdown_cell(
        source="### DBSCAN: Density-Based Spatial Clustering\n\n"
               "Groups together points that are closely packed, marking as outliers points "
               "that lie alone in low-density regions.\n\n"
               "$$\n"
               "\\text{Core point}: \\quad |\\{ \\mathbf{y} : d(\\mathbf{x}, \\mathbf{y}) \\leq \\epsilon \\}| \\geq \\text{minPts}\n"
               "$$\n\n"
               "*What each symbol means:*\n"
               "- **\\mathbf{x}**: A data point\n"
               "- **d(\\mathbf{x}, \\mathbf{y})**: Distance between points \\mathbf{x} and \\mathbf{y}\n"
               "- **\\epsilon**: Neighborhood radius\n"
               "- **minPts**: Minimum number of points to form a dense region\n"
               "- **Border point**: Reachable from core point but has fewer than minPts neighbors\n"
               "- **Outlier**: Not a core point and not reachable from any core point\n\n"
               "### DBSCAN Algorithm:\n"
               "1. For each point, retrieve its \\epsilon-neighborhood\n"
               "2. If neighborhood has ≥ minPts points, mark as core point and start new cluster\n"
               "3. Add all points in neighborhood to cluster\n"
               "4. For each newly added point, if it's also a core point, repeat from step 1\n"
               "5. Points not assigned to any cluster are outliers\n"
    ))

    # Markdown cell: Gaussian Mixture Models
    cells.append(new_markdown_cell(
        source="### Gaussian Mixture Models (GMM)\n\n"
               "Assumes data is generated from a mixture of several Gaussian distributions.\n\n"
               "$$\n"
               "p(\\mathbf{x}) = \\sum_{k=1}^K \\phi_k \\mathcal{N}(\\mathbf{x} | \\boldsymbol{\\mu}_k, \\boldsymbol{\\Sigma}_k)\n"
               "$$\n\n"
               "*What each symbol means:*\n"
               "- **p(\\mathbf{x})**: Probability density of observing point \\mathbf{x}\n"
               "- **K**: Number of mixture components (clusters)\n"
               "- **\\phi_k**: Mixing coefficient for component k (\\phi_k ≥ 0, \\sum \\phi_k = 1)\n"
               "- **\\mathcal{N}(\\mathbf{x} | \\boldsymbol{\\mu}_k, \\boldsymbol{\\Sigma}_k)**: Gaussian distribution\n"
               "- **\\boldsymbol{\\mu}_k**: Mean vector of component k\n"
               "- **\\boldsymbol{\\Sigma}_k**: Covariance matrix of component k\n\n"
               "### Expectation-Maximization (EM) Algorithm:\n"
               "1. **E-step**: Compute responsibility \\gamma_{zi} = P(z_i = k | \\mathbf{x}_i)\n"
               "2. **M-step**: Update parameters using weighted averages\n"
               "3. Repeat until convergence\n"
    ))

    # Markdown cell: Worked Example by Hand
    cells.append(new_markdown_cell(
        source="## Worked Example by Hand\n\n"
               "> 🧮 Simple clustering: Group these 2D points into 2 clusters: "
               "[1,1], [1,2], [2,1], [8,8], [8,9], [9,8]\n\n"
               "### Step 1: Visualize the data\n\n"
               "Points form two obvious groups: "
               "Group 1: (1,1), (1,2), (2,1) clustered around (1,1)\n"
               "Group 2: (8,8), (8,9), (9,8) clustered around (8,8)\n\n"
               "### Step 2: Apply K-Means with K=2\n\n"
               "**Initialization**: Randomly choose 2 centroids\n"
               "Let's say we pick C1 = [1,1] and C2 = [8,8]\n\n"
               "**Assignment Step**:\n"
               "- Distance from [1,1] to C1: 0, to C2: √98 ≈ 9.9 → Assign to C1\n"
               "- Distance from [1,2] to C1: 1, to C2: √85 ≈ 9.2 → Assign to C1\n"
               "- Distance from [2,1] to C1: 1, to C2: √85 ≈ 9.2 → Assign to C1\n"
               "- Distance from [8,8] to C1: √98 ≈ 9.9, to C2: 0 → Assign to C2\n"
               "- Distance from [8,9] to C1: √85 ≈ 9.2, to C2: 1 → Assign to C2\n"
               "- Distance from [9,8] to C1: √85 ≈ 9.2, to C2: 1 → Assign to C2\n\n"
               "**Update Step**:\n"
               "- New C1 = mean([1,1], [1,2], [2,1]) = [4/3, 4/3] ≈ [1.33, 1.33]\n"
               "- New C2 = mean([8,8], [8,9], [9,8]) = [25/3, 25/3] ≈ [8.33, 8.33]\n\n"
               "**Repeat Assignment**: All points still assign to same clusters\n\n"
               "**Convergence**: Centroids no longer change significantly\n\n"
               "**Final Clusters**:\n"
               "- Cluster 1: [1,1], [1,2], [2,1] (Centroid ≈ [1.33, 1.33])\n"
               "- Cluster 2: [8,8], [8,9], [9,8] (Centroid ≈ [8.33, 8.33])\n\n"
               "<em>Answer: The algorithm correctly identifies the two natural groups in the data.</em>"
    ))

    # Code cell: Import libraries
    cells.append(new_code_cell(
        source="import numpy as np\n"
               "import matplotlib.pyplot as plt\n"
               "import seaborn as sns\n"
               "from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering\n"
               "from sklearn.mixture import GaussianMixture\n"
               "from sklearn.datasets import make_blobs, make_circles, make_moons\n"
               "from sklearn.preprocessing import StandardScaler\n"
    ))

    # Code cell: Generate sample data
    cells.append(new_code_cell(
        source="# Generate sample datasets for clustering demonstration\n"
               "# 1. Blobs dataset (well-separated clusters)\n"
               "X_blobs, y_blobs = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)\n"
               "\n"
               "# 2. Circles dataset (non-convex clusters)\n"
               "X_circles, y_circles = make_circles(n_samples=300, factor=0.5, noise=0.05)\n"
               "\n"
               "# 3. Moons dataset (non-convex clusters)\n"
               "X_moons, y_moons = make_moons(n_samples=300, noise=0.05)\n"
               "\n"
               "# Standardize features\n"
               "scaler = StandardScaler()\n"
               "X_blobs_scaled = scaler.fit_transform(X_blobs)\n"
               "X_circles_scaled = scaler.fit_transform(X_circles)\n"
               "X_moons_scaled = scaler.fit_transform(X_moons)\n"
               "\n"
               "print(\"Dataset shapes:\")\n"
               "print(f\"Blobs: {X_blobs.shape}\")\n"
               "print(f\"Circles: {X_circles.shape}\")\n"
               "print(f\"Moons: {X_moons.shape}\")\n"
    ))

    # Code cell: K-Means clustering on blobs
    cells.append(new_code_cell(
        source="# K-Means Clustering on blobs dataset\n"
               "kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)\n"
               "y_pred_kmeans = kmeans.fit_predict(X_blobs_scaled)\n"
               "\n"
               "# Plot results\n"
               "plt.figure(figsize=(12, 4))\n"
               "\n"
               "plt.subplot(1, 3, 1)\n"
               "plt.scatter(X_blobs_scaled[:, 0], X_blobs_scaled[:, 1], c=y_blobs, cmap='viridis')\n"
               "plt.title('True Labels')\n"
               "plt.xlabel('Feature 1')\n"
               "plt.ylabel('Feature 2')\n"
               "\n"
               "plt.subplot(1, 3, 2)\n"
               "plt.scatter(X_blobs_scaled[:, 0], X_blobs_scaled[:, 1], c=y_pred_kmeans, cmap='viridis')\n"
               "plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], \n"
               "            c='red', marker='x', s=200, linewidths=3, label='Centroids')\n"
               "plt.title('K-Means Clustering')\n"
               "plt.xlabel('Feature 1')\n"
               "plt.ylabel('Feature 2')\n"
               "plt.legend()\n"
               "\n"
               "plt.subplot(1, 3, 3)\n"
               "plt.scatter(X_blobs_scaled[:, 0], X_blobs_scaled[:, 1], c=y_pred_kmeans == y_blobs, \n"
               "            cmap='coolwarm')\n"
               "plt.title('Correct vs Incorrect Assignments')\n"
               "plt.xlabel('Feature 1')\n"
               "plt.ylabel('Feature 2')\n"
               "\n"
               "plt.tight_layout()\n"
               "plt.show()\n"
               "\n"
               "print(f\"K-Means inertia: {kmeans.inertia_:.2f}\")\n"
    ))

    # Code cell: DBSCAN clustering on moons
    cells.append(new_code_cell(
        source="# DBSCAN Clustering on moons dataset\n"
               "dbscan = DBSCAN(eps=0.3, min_samples=5)\n"
               "y_pred_dbscan = dbscan.fit_predict(X_moons_scaled)\n"
               "\n"
               "# Plot results\n"
               "plt.figure(figsize=(12, 4))\n"
               "\n"
               "plt.subplot(1, 3, 1)\n"
               "plt.scatter(X_moons_scaled[:, 0], X_moons_scaled[:, 1], c=y_moons, cmap='viridis')\n"
               "plt.title('True Labels')\n"
               "plt.xlabel('Feature 1')\n"
               "plt.ylabel('Feature 2')\n"
               "\n"
               "plt.subplot(1, 3, 2)\n"
               "plt.scatter(X_moons_scaled[:, 0], X_moons_scaled[:, 1], c=y_pred_dbscan, cmap='viridis')\n"
               "plt.title('DBSCAN Clustering')\n"
               "plt.xlabel('Feature 1')\n"
               "plt.ylabel('Feature 2')\n"
               "\n"
               "plt.subplot(1, 3, 3)\n"
               "# Show core samples\n"
               "core_samples_mask = np.zeros_like(y_pred_dbscan, dtype=bool)\n"
               "core_samples_mask[dbscan.core_sample_indices_] = True\n"
               "plt.scatter(X_moons_scaled[:, 0], X_moons_scaled[:, 1], c=y_pred_dbscan, cmap='viridis')\n"
               "plt.scatter(X_moons_scaled[core_samples_mask, 0], X_moons_scaled[core_samples_mask, 1], \n"
               "            c='yellow', marker='o', s=50, edgecolors='k', label='Core Points')\n"
               "plt.title('DBSCAN with Core Points')\n"
               "plt.xlabel('Feature 1')\n"
               "plt.ylabel('Feature 2')\n"
               "plt.legend()\n"
               "\n"
               "plt.tight_layout()\n"
               "plt.show()\n"
               "\n"
               "n_clusters_ = len(set(y_pred_dbscan)) - (1 if -1 in y_pred_dbscan else 0)\n"
               "n_noise_ = list(y_pred_dbscan).count(-1)\n"
               "print(f\"Estimated number of clusters: {n_clusters_}\")\n"
               "print(f\"Estimated number of noise points: {n_noise_}\")\n"
    ))

    # Code cell: Gaussian Mixture Model
    cells.append(new_code_cell(
        source="# Gaussian Mixture Model on blobs dataset\n"
               "gmm = GaussianMixture(n_components=4, random_state=42)\n"
               "gmm.fit(X_blobs_scaled)\n"
               "y_pred_gmm = gmm.predict(X_blobs_scaled)\n"
               "\n"
               "# Plot results\n"
               "plt.figure(figsize=(12, 4))\n"
               "\n"
               "plt.subplot(1, 3, 1)\n"
               "plt.scatter(X_blobs_scaled[:, 0], X_blobs_scaled[:, 1], c=y_blobs, cmap='viridis')\n"
               "plt.title('True Labels')\n"
               "plt.xlabel('Feature 1')\n"
               "plt.ylabel('Feature 2')\n"
               "\n"
               "plt.subplot(1, 3, 2)\n"
               "plt.scatter(X_blobs_scaled[:, 0], X_blobs_scaled[:, 1], c=y_pred_gmm, cmap='viridis')\n"
               "plt.scatter(gmm.means_[:, 0], gmm.means_[:, 1], \n"
               "            c='red', marker='x', s=200, linewidths=3, label='Means')\n"
               "plt.title('Gaussian Mixture Model')\n"
               "plt.xlabel('Feature 1')\n"
               "plt.ylabel('Feature 2')\n"
               "plt.legend()\n"
               "\n"
               "plt.subplot(1, 3, 3)\n"
               "# Plot uncertainty\n"
               "probs = gmm.predict_proba(X_blobs_scaled)\n"
               "uncertainty = 1 - np.max(probs, axis=1)\n"
               "plt.scatter(X_blobs_scaled[:, 0], X_blobs_scaled[:, 1], c=uncertainty, cmap='hot')\n"
               "plt.colorbar(label='Uncertainty')\n"
               "plt.title('Prediction Uncertainty')\n"
               "plt.xlabel('Feature 1')\n"
               "plt.ylabel('Feature 2')\n"
               "\n"
               "plt.tight_layout()\n"
               "plt.show()\n"
               "\n"
               "print(f\"GMM convergence: {gmm.converged_}\")\n"
               "print(f\"Number of iterations: {gmm.n_iter_}\")\n"
               "print(f\"Lower bound: {gmm.lower_bound_:.2f}\")\n"
    ))

    # Markdown cell: Watch Out
    cells.append(new_markdown_cell(
        source="## Watch Out: Common Mistakes & Exam Traps\n\n"
           "> ⚠️ **Common mistakes to avoid:**\n"
           "> 1. **Assuming K-means finds globally optimal solution**: K-means converges to local optimum; "
           "results depend on initialization.\n"
           "> 2. **Ignoring feature scaling**: Distance-based clustering is sensitive to feature scales; "
           "always standardize features first.\n"
           "> 3. **Choosing wrong distance metric**: Euclidean distance assumes spherical clusters; "
           "use Manhattan for high-dimensional sparse data, cosine for text.\n"
           "> 4. **Misinterpreting hierarchical clustering dendrogram**: Vertical height represents "
           "distance/dissimilarity, not time or iteration count.\n"
           "> 5. **Overlooking DBSCAN parameter sensitivity**: Results heavily depend on eps and minPts; "
           "use k-distance graph to choose eps.\n"
           "> 6. **Assuming all points must belong to a cluster**: DBSCAN and other methods can identify outliers.\n"
           "> 7. **Using accuracy to evaluate clustering**: Accuracy requires true labels; use internal "
           "metrics like silhouette score when true labels unknown.\n"
           "> 8. **Ignoring the curse of dimensionality**: Distance metrics become less meaningful in "
           "very high dimensions; consider dimensionality reduction first.\n"
    ))

    # Markdown cell: Experiment Link
    cells.append(new_markdown_cell(
        source="## Experiment Link\n\n"
           "> 🔬 Run **Section 2: Clustering Comparison Experiment** in this notebook to compare "
           "different clustering algorithms on various datasets and understand their strengths "
           "and weaknesses.\n\n"
           "This section will include:\n"
           "- Comparison of K-Means, Hierarchical, DBSCAN, and GMM on different dataset types\n"
           "- Evaluation using internal validity metrics (silhouette score, Calinski-Harabasz)\n"
           "- Visualization of clustering results and decision boundaries\n"
           "- Analysis of algorithm sensitivity to parameters\n"
           "- Discussion of when to use each algorithm\n"
    ))

    # Markdown cell: Videos to Watch
    cells.append(new_markdown_cell(
        source="## Videos to Watch\n\n"
           "> 📺 **Recommended video searches:**\n\n"
           "1. **Clustering Algorithms Explained**\n"
           "   - Search: https://www.youtube.com/results?search_query=clustering+algorithms+explained\n"
           "   - Watch for: Clear explanation of K-means, hierarchical clustering, DBSCAN\n\n"
           "2. **StatQuest: K-means Clustering**\n"
           "   - Search: https://www.youtube.com/results?search_query=StatQuest+K-means+Clustering\n"
           "   - Watch for: Intuitive breakdown of how K-means works and when to use it\n\n"
           "3. **Machine Learning Zoomcamp: Clustering**\n"
           "   - Search: https://www.youtube.com/results?search_query=clustering+machine+learning+zoomcamp\n"
           "   - Watch for: Practical implementation and real-world examples\n"
    ))

    # Markdown cell: Practice Questions
    cells.append(new_markdown_cell(
        source="## Practice Questions\n\n"
           "> ❓ Try to solve these yourself first, then check the answers below!\n\n"
           "### Question 1 (Concept): What is the main difference between partitioning and hierarchical clustering?\n\n"
           "<details>\n"
           "<summary>Click to reveal answer</summary>\n\n"
           "Partitioning clustering (like K-means) divides data into non-overlapping clusters without "
           "any hierarchical structure. Hierarchical clustering creates a tree of clusters (dendrogram) "
           "showing relationships between clusters at different levels of granularity.\n\n"
           "</details>\n\n"
           "\n"
           "### Question 2 (Concept): Why is feature scaling important before clustering?\n\n"
           "<details>\n"
           "<summary>Click to reveal answer</summary>\n\n"
           "Clustering algorithms based on distance metrics are sensitive to the scale of features. "
           "Features with larger scales will dominate the distance calculation, leading to biased "
           "clusters. Standardization ensures all features contribute equally to distance computations.\n\n"
           "</details>\n\n"
           "\n"
           "### Question 3 (Calculation): Given two points A(2,3) and B(5,7), what is the Euclidean distance between them?\n\n"
           "<details>\n"
           "<summary>Click to reveal answer</summary>\n\n"
           "d(A,B) = √[(5-2)² + (7-3)²] = √[3² + 4²] = √[9 + 16] = √25 = 5\n\n"
           "</details>\n\n"
           "\n"
           "### Question 4 (Concept): When would you prefer DBSCAN over K-means?\n\n"
           "<details>\n"
           "<summary>Click to reveal answer</summary>\n\n"
           "Prefer DBSCAN when: you want to find arbitrarily shaped clusters, you expect noise/outliers "
           "in your data, you don't want to specify the number of clusters in advance, or your "
           "clusters have varying densities.\n\n"
           "</details>\n\n"
           "\n"
           "### Question 5 (Explain it simply): What does the 'silhouette score' measure in clustering evaluation?\n\n"
           "<details>\n"
           "<summary>Click to reveal answer</summary>\n\n"
           "The silhouette score measures how similar a point is to its own cluster compared to other "
           "clusters. It ranges from -1 to 1, where high values indicate good clustering (points "
           "well-matched to their own cluster and poorly matched to neighboring clusters).\n\n"
           "</details>\n"
    ))

    # Markdown cell: Cheat Sheet Draft
    cells.append(new_markdown_cell(
        source="## Cheat Sheet Draft\n\n"
           "> 📝 This section is optimized for printing - it should fit approximately 1/7 of an A4 page when printed.\n\n"
           "- Optimized for printing - fits approximately 1/7 of an A4 page\n"
           "- Contains key formulas, definitions, and common mistakes\n"
           "- Designed for quick review and reference\n\n"
           "### Key Formulas\n\n"
           "$$\n"
           "d(\\mathbf{x}, \\mathbf{y}) = \\sqrt{\\sum_{i=1}^n (x_i - y_i)^2}\n"
           "$$\n\n"
           "*Euclidean distance*\n\n"
           "$$\n"
           "\\arg\\min_{\\mathbf{S}} \\sum_{i=1}^k \\sum_{\\mathbf{x} \\in S_i} \\|\\mathbf{x} - \\boldsymbol{\\mu}_i\\|^2\n"
           "$$\n\n"
           "*K-means objective function*\n\n"
           "$$\n"
           "p(\\mathbf{x}) = \\sum_{k=1}^K \\phi_k \\mathcal{N}(\\mathbf{x} | \\boldsymbol{\\mu}_k, \\boldsymbol{\\Sigma}_k)\n"
           "$$\n\n"
           "*Gaussian Mixture Model*\n\n"
           "### Key Definitions\n\n"
           "- **Centroid:** The mean position of all points in a cluster\n"
           "- **Inertia:** Sum of squared distances of samples to their closest cluster center\n"
           "- **Linkage Criteria:** Method for calculating distance between clusters in hierarchical clustering\n"
           "- **Core Point (DBSCAN):** A point with at least minPts neighbors within distance eps\n"
           "- **Noise Point (DBSCAN):** A point that is neither a core point nor reachable from a core point\n"
           "- **Silhouette Score:** Measure of how similar an object is to its own cluster vs other clusters\n\n"
           "### Common Mistakes to Avoid\n\n"
           "- Assuming K-means finds globally optimal solution (finds local optimum)\n"
           "- Ignoring feature scaling before distance-based clustering\n"
           "- Choosing wrong number of clusters K without validation\n"
           "- Misinterpreting hierarchical clustering dendrogram heights\n"
           "- Overlooking DBSCAN's sensitivity to eps and minPts parameters\n"
           "- Assuming all clustering points must belong to a cluster (algorithms can identify outliers)\n"
           "- Using accuracy to evaluate clustering without ground truth labels\n"
           "- Ignoring the curse of dimensionality in high-dimensional feature spaces\n"
    ))

    # Add all cells to the notebook
    notebook.cells = cells

    # Save the notebook
    with open('notebooks/06_clustering.ipynb', 'w', encoding='utf-8') as f:
        nbformat.write(notebook, f)

    # Validate the notebook
    nbformat.validate(notebook)

    # Load the file and count cells
    with open('notebooks/06_clustering.ipynb', 'r', encoding='utf-8') as f:
        loaded_notebook = nbformat.read(f, as_version=4)

    code_cells = sum(1 for cell in loaded_notebook.cells if cell.cell_type == 'code')
    markdown_cells = sum(1 for cell in loaded_notebook.cells if cell.cell_type == 'markdown')

    print(f"Code cells: {code_cells}")
    print(f"Markdown cells: {markdown_cells}")

    return notebook

if __name__ == "__main__":
    create_lecture06_notebook()