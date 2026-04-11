# Lab 6 — Knowledge Distillation

**Author:** Ackshay Nagamallu Rajasekar
**Course:** MLOps

This lab implements **Knowledge Distillation** — a model compression technique where a small "student" model learns from a larger, pre-trained "teacher" model. The goal is to show that a compact student model can generalise better when guided by the teacher's soft predictions rather than training on hard labels alone.

---

## What I Changed from the Original Lab

| Item | Original | My Version |
|------|----------|------------|
| Dataset | Cats vs Dogs (2 classes, 224×224, ~787 MB download) | **CIFAR-10** (10 classes, 32×32, built into Keras) |
| Teacher architecture | 4× Conv + 2× Dropout + Dense(512) | 3× Conv + **BatchNormalization** + 3× Dropout + Dense(256) |
| Student architecture | 1× Conv + Dense(2) | 2× Conv + Dense(64) + Dense(10) |
| Output classes | 2 | **10** |
| Distillation temperature | 5 | **10** (softer targets across 10 classes) |
| Alpha | 0.05 | **0.1** (10% ground truth, 90% distillation) |
| `tensorflow_datasets` | Required | **Removed** — CIFAR-10 loads directly from Keras |
| `compiled_metrics` | Used (removed in Keras 3) | Fixed — uses `_custom_metrics` list |
| `v.get_shape()` | Used (removed in TF 2.16+) | Fixed — uses `v.shape` |
| Model save format | `model.save('name')` | `model.save('name.keras')` (Keras 3 format) |

---

## Dataset — CIFAR-10

- **60,000** colour images at **32×32 pixels**
- **10 classes:** airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- **Split used:** 45,000 train / 5,000 validation / 10,000 test
- Built into Keras — no download needed (`keras.datasets.cifar10`)

---

## Model Architectures

### Teacher (`create_big_model`)

```
Conv2D(64) → BatchNorm → ReLU → MaxPool          # Block 1 — 16×16×64
Conv2D(128) → BatchNorm → ReLU → MaxPool → Dropout(0.3)   # Block 2 — 8×8×128
Conv2D(256) → BatchNorm → ReLU → MaxPool → Dropout(0.5)   # Block 3 — 4×4×256
Flatten → Dense(256, ReLU) → Dropout(0.4) → Dense(10)     # logits
```

### Student (`create_small_model`)

```
Conv2D(32, ReLU, same) → MaxPool     # 16×16×32
Conv2D(64, ReLU, same) → MaxPool     # 8×8×64
Flatten → Dense(64, ReLU) → Dense(10)   # logits
```

> Both models output raw **logits** (no softmax) — required for temperature-scaled distillation loss.

---

## Distillation Parameters

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `alpha` | 0.1 | 10% weight on student cross-entropy loss, 90% on distillation loss |
| `temperature` | 10 | Softens teacher's output distribution — reveals inter-class similarities |
| `distillation_loss_fn` | KL Divergence | Measures difference between teacher and student soft targets |
| `student_loss_fn` | SparseCategoricalCrossentropy | Standard classification loss against ground truth |

**Combined loss:** `loss = 0.1 × student_loss + 0.9 × distillation_loss`

Higher temperature (T=10) is especially useful with 10 classes — the teacher may assign small probability to "automobile" when seeing a "truck", teaching the student that those classes share visual features.

---

## Training

| Model | Epochs |
|-------|--------|
| Teacher | 10 |
| Student (scratch) | 7 |
| Student (distilled) | 7 |

---

## Results

| Model | Final Validation Accuracy |
|-------|--------------------------|
| Teacher | higher (larger model, 10 epochs) |
| Student — distilled | **~57.5%** (stable, no overfitting) |
| Student — scratch | **~46.5%** (crashed — clear overfitting after epoch 3) |



**Key findings from the plot:**
- The **scratch student** peaks at ~62.5% validation accuracy at epoch 3, then crashes to ~46.5% — a classic overfitting pattern
- The **distilled student** stays stable at ~57.5% through the final epochs
- The distilled model learned the teacher's **BatchNormalization + Dropout regularization** implicitly through the soft targets — without having any explicit regularization in its own architecture

---

## Project Structure

```
LAB6-Model_Development/
├── Knowledge_Distillation.ipynb   # Main notebook
├── teacher_model.keras            # Saved teacher model (after running)
├── student_scratch.keras          # Saved scratch student (after running)
└── README.md
```

---

## How to Run

### Prerequisites

Python 3.11 recommended (TF 2.16+ requires Python 3.9–3.12).

```bash
# Create and activate virtual environment
python3.11 -m venv tf_env
source tf_env/bin/activate

# Install dependencies
pip install tensorflow        # or tensorflow-macos + tensorflow-metal on Apple Silicon
pip install jupyter numpy pandas seaborn matplotlib
```

### Run the notebook

```bash
jupyter notebook Knowledge_Distillation.ipynb
```

Run all cells top to bottom. The full pipeline:
1. Loads and preprocesses CIFAR-10
2. Defines the `Distiller` custom Keras model
3. Trains the teacher (10 epochs)
4. Trains the student from scratch (7 epochs)
5. Trains the student with distillation (7 epochs)
6. Evaluates and plots all three models

> **Tip:** Use [Google Colab](https://colab.research.google.com) with a free T4 GPU to train significantly faster — just upload the notebook and run all cells.
