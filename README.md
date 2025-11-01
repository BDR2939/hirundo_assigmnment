# TracInCP Pipeline for Influence Analysis

A flexible pipeline for computing **TracInCP** (TracIn Checkpoint) self-influence and cross-sample influence scores. Identifies potentially mislabeled training examples, influential samples, and relationships between training and test data.

## Quick Summary

**What it does:** Computes how much each training example influences model predictions by analyzing gradients across training checkpoints.

**Outputs:**
- **Self-influence mode**: Influence scores per training sample + visualizations (histograms, recovery curves, precision-recall) + metrics (if mislabeled info available)
- **Cross-influence mode**: Influence matrix showing how each training example affects each test example

**Key Features:**
- Supports HuggingFace datasets and PyTorch datasets (torchvision)
- Custom preprocessing and data loading via preprocessor/collate patterns
- Automatic visualization with consistent color scheme
- Multiple clean subset selection methods (percentile, knee point, GMM)
- Efficient batch-wise gradient computation using `torch.func`

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Self-Influence Analysis

```python
from tracin_pipeline import TracInPipeline
from ResNet import ResNet
from datasets import load_dataset

# Initialize model and checkpoints
model = ResNet(num_classes=100, n=9).to(device)
checkpoints = [f'checkpoints/resnet_epoch_{i}.pth' for i in range(30, 301, 30)]

# Create pipeline (default mode="self")
pipeline = TracInPipeline(
    model=model,
    checkpoints=checkpoints,
    batch_size=128,
    device="cuda"
)

# Run analysis
dataset = load_dataset("your-dataset", split='train')
total_influence, results, metrics = pipeline.run(dataset, plot_results=True)
```

### Cross-Sample Influence

```python
# Initialize pipeline in cross-influence mode
pipeline = TracInPipeline(
    model=model,
    checkpoints=checkpoints,
    mode="cross",
    batch_size=128,
    device="cuda"
)

# Compute influence matrix
train_dataset = load_dataset("your-dataset", split='train')
test_dataset = load_dataset("your-dataset", split='test')
influence_matrix = pipeline.run(train_dataset, test_dataset)
# Shape: [num_train, num_test]
```

## How to Run

### Basic Usage

```python
pipeline = TracInPipeline(
    model=model,                    # PyTorch model matching checkpoint architecture
    checkpoints=[...],              # List of checkpoint file paths
    batch_size=128,                 # Evaluation batch size
    mode="self",                    # "self" or "cross" (default: "self")
    preprocessor=None,              # Optional: dataset preprocessor
    collate_fn=None,                # Optional: custom collate function
    device="cuda"                   # "cuda", "cpu", or "mps"
)

# Self-influence mode
total_influence, results, metrics = pipeline.run(
    train_dataset=dataset,
    plot_results=True,              # Show visualizations
    save_path=None                  # Optional: directory to save results
)

# Cross-influence mode
influence_matrix = pipeline.run(
    train_dataset=train_dataset,
    test_dataset=test_dataset,      # Required for cross mode
    save_path="results/"
)
```

### Custom Preprocessing

Preprocessors implement a `.process(dataset)` method that returns a processed dataset:

```python
class CustomPreprocessor:
    def process(self, dataset):
        def add_meta(example):
            example['label'] = get_label(example)
            example['mislabeled'] = check_mislabeled(example)
            return example
        return dataset.map(add_meta)

pipeline = TracInPipeline(..., preprocessor=CustomPreprocessor())
```

See `example.ipynb` for a complete CIFAR-100 preprocessing example.

### Custom Collate Function

For datasets requiring special batching (e.g., image transforms):

```python
def make_collate_fn(transform):
    def collate_fn(batch):
        images = torch.stack([transform(x["image"]) for x in batch])
        labels = torch.tensor([x["label"] for x in batch])
        return images, labels
    return collate_fn

pipeline = TracInPipeline(..., collate_fn=make_collate_fn(transform))
```

**Requirements:** Returns tuple `(inputs, labels)` where `inputs` is a tensor suitable for model forward pass.

## Input/Output Format

### Input

**Dataset Format:**
- HuggingFace `datasets.Dataset` or PyTorch `torch.utils.data.Dataset`
- Optional metadata fields (auto-detected):
  - `mislabeled`: Boolean flag (enables metrics computation)
  - `label`: Class label (enables per-class analysis)
  - `true_label`: Ground truth (if different from `label`)
  - `__key__`: Unique identifier

**Checkpoint Format:**
- PyTorch checkpoint files with `'model_state_dict'` key
- Model architecture must match provided model instance

### Output

#### Self-Influence Mode

Returns tuple `(total_influence, results, metrics)`:

1. **`total_influence`** (torch.Tensor)
   - Shape: `[N]` where N = dataset size
   - Aggregated self-influence scores across checkpoints

2. **`results`** (pd.DataFrame)
   - Columns: `influence`, `rank`, `norm_rank`, plus optional metadata columns
   - `rank`: 1 = highest influence
   - `norm_rank`: Normalized rank (0-1)

3. **`metrics`** (dict)
   - **Only computed if `mislabeled` column exists in dataset**
   - Keys: `fractions`, `recovery` (recall), `precision`, `total_mislabeled`, `auc`
   - Recovery curve: fraction of mislabeled found vs. fraction examined

**Saved Files** (if `save_path` provided):
- `influence_results.csv`: Full results DataFrame
- `influence_histogram.html`: Score distribution
- `recovery_curve.html`: Mislabeled recovery curve
- `precision_vs_recall.html`: Precision-recall curve
- `influence_by_label.html`: Box plot by class (if labels available)

#### Cross-Influence Mode

Returns `influence_matrix` (np.ndarray):
- Shape: `[num_train, num_test]`
- Entry `[i, j]` = influence of training example i on test example j
- Saved as `.npy` file if `save_path` provided

## Design Choices

### Design Highlights

| Aspect | Choice | Rationale |
|--------|--------|-----------|
| **Gradient scope** | Last layer only (default) | Reduces memory; captures decision boundary signal |
| **Checkpoint aggregation** | Equal weights (eta_i = 1.0) | Simple default; can be customized via `eta_list` |
| **Computation method** | Batch-wise with `torch.func.vmap` | Efficient per-sample gradients without loops |
| **Dataset interface** | Preprocessor + collate patterns | Supports HuggingFace and PyTorch datasets |
| **Output exposure** | Three levels (raw scores, DataFrame, metrics) | Supports both detailed analysis and summary stats |
| **Visualization** | Consistent color scheme | Professional, readable plots |
| **Selection methods** | Percentile, knee point, GMM | Multiple strategies for different data distributions |

### Detailed Design Decisions

**1. Modular Architecture**
- `Influence`: Core computation engine
- `TracInPipeline`: High-level orchestration
- Enables direct use of `Influence` for custom workflows

**2. Dual Mode Support**
- **Self-influence**: Identifies mislabeled/influential training examples
- **Cross-influence**: Analyzes training→test relationships
- Unified API with mode parameter

**3. Flexible Dataset Handling**
- Automatic detection of HuggingFace vs PyTorch datasets
- Optional preprocessor for metadata enrichment
- Custom collate function for domain-specific batching

**4. Efficient Gradient Computation**
- Uses `torch.func.functional_call`, `vmap`, and `grad` for vectorized per-sample gradients
- Last layer only reduces memory while maintaining signal quality

**5. Comprehensive Outputs**
- Raw influence scores for programmatic use
- Structured DataFrame for analysis and filtering
- Aggregated metrics only when mislabeled annotations available
- Interactive visualizations with consistent styling

**6. Clean Subset Selection**
- Three methods in `selection.py`:
  - **Percentile**: Simple, interpretable threshold
  - **Knee point**: Adaptive elbow detection
  - **GMM**: Statistical two-component model

## Evaluation Metrics

**Important:** Precision, recall, and recovery metrics are **only computed if the dataset contains a `mislabeled` column**. Without this, only influence scores and rankings are available.

When `mislabeled` exists:
- **Recovery (Recall)**: Fraction of all mislabeled examples found in top-k% of influence rankings
- **Precision**: Fraction of top-k% examples that are actually mislabeled
- **AUC**: Area under recovery curve

## Pipeline Versatility

The pipeline adapts to different use cases:

- **Mode Selection (`mode`)**:
  - The pipeline supports both **self-influence** (`mode="self"`)—quantifying how much each training example influences itself (ideal for mislabeling detection)—and **cross-influence** (`mode="cross"`)—measuring how training examples influence validation/test points (useful for debugging and model interpretation). Choose the mode that suits your use case: `self` to surface anomalies within your training data, `cross` to audit your model’s behavior on specific evaluation samples.

- **Preprocessing (`preprocessor`)**:
  - You may pass a custom preprocessor to automatically enrich samples with metadata or additional features. This enables transformations such as normalization, label remapping, or adding extra columns (e.g., tagging suspected mislabels) before influence computation, and works seamlessly for both HuggingFace and PyTorch datasets.

- **Custom Collation (`collate_fn`)**:
  - The pipeline accepts an optional `collate_fn` argument, facilitating domain-specific batching strategies (e.g., custom image augmentation, padding strategies, or multi-modal input assembly). This is crucial for handling non-standard datasets or when samples require special preparation before entering the model.

These options allow TracInPipeline to flexibly interoperate with a wide range of datasets and research goals, from clean mislabeling detection to class-specific analysis and beyond. For highly specialized use, you may access raw influence scores directly, or further extend the pipeline.

See `example.ipynb` for complete examples including:
- Custom preprocessing with metadata
- Image transforms via collate function
- Clean subset selection evaluation

## Architecture

**ResNet Implementation:**
- Optimized for CIFAR-10/100 (32×32 images)
- Configurable depth via `n` parameter:
  - ResNet-20: `n=3`
  - ResNet-56: `n=9`
  - ResNet-110: `n=18`

## Requirements

Key dependencies:
- PyTorch (with `torch.func` support)
- HuggingFace datasets
- Plotly
- scikit-learn
- pandas

See `requirements.txt` for complete list.
