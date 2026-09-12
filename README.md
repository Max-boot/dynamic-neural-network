# Dynamic Neural Network (DNN) - Biologically Plausible Self-Organizing Architecture

A **biologically accurate** neural network that mirrors human brain development, combining **STDP**, **Homeostatic Plasticity**, **Developmental Stages**, and **Structural Plasticity** for human detection. Optimized for **ESP32-CAM AI Thinker** deployment.

## 🆕 Version 5 - Hybrid Biological CUDA (Latest)

**The best of both worlds: v3 Performance + v4 Biological Accuracy**

```
┌────────────────────────────────────────────────────────────────────────────────────┐
│              HYBRID BIOLOGICAL NEURAL NETWORK v5                                   │
│                    (RGB 160×120 Color + CUDA Optimized)                            │
├────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                    │
│   INPUT LAYER              HIDDEN LAYER                   OUTPUT LAYER             │
│   (57,600 neurons)         (Dynamic: 50-8192)             (2 neurons)              │
│   160×120×3 RGB            Biological F-I Curve           Sigmoid                  │
│                                                                                    │
│   ┌─────┐  Color           ┌─────┐      GPU-STDP          ┌─────┐                  │
│   │ R₀  │─Opponent─────────│ h₀  │────────●───────────────│ y₀  │ Person           │
│   ├─────┤  Processing      ├─────┤        ↓               ├─────┤                  │
│   │ G₀  │───────●──────────│ h₁  │───Homeostatic──────────│ y₁  │ No Person        │
│   ├─────┤       ↓          ├─────┤    Scaling             └─────┘                  │
│   │ B₀  │───BDNF───────────│ ... │        ↓                                        │
│   ├─────┤   Growth         ├─────┤    Adaptive                                     │
│   │ ... │    Factor        │ hₙ  │───Thresholds──────────────────                  │
│   └─────┘                  └─────┘                                                 │
│                                                                                    │
│   ═══════════════════════════════════════════════════════════════════              │
│                     DEVELOPMENTAL STAGES (GPU-Accelerated)                         │
│   ═══════════════════════════════════════════════════════════════════              │
│                                                                                    │
│   🥒 EMBRYONIC → 👶 INFANT → 🧒 CHILDHOOD → 🧑 ADOLESCENT → 🧑‍🦳 ADULT              │
│                                                                                    │
└────────────────────────────────────────────────────────────────────────────────────┘
```

### v5 Key Features

| Feature | Description |
|---------|-------------|
| **RGB Color Input** | 160×120×3 = 57,600 input neurons (vs. 9,216 in v4) |
| **GPU-Accelerated STDP** | Vectorized correlation-based learning on CUDA |
| **Higher Resolution** | 160×120 vs 96×96 in v4 |
| **Adaptive Thresholds** | Spike-Frequency Adaptation per neuron |
| **BDNF Growth Control** | Activity-dependent neurogenesis |
| **ESP32 Export** | Int8 quantized, ~200KB model size |

## 🧠 Neuroscience Foundation

This project implements cutting-edge computational neuroscience principles:

| Biological Mechanism | Reference | Implementation |
|---------------------|-----------|----------------|
| **STDP** (Spike-Timing Dependent Plasticity) | Bi & Poo, 1998 | Correlation-based weight updates |
| **Homeostatic Plasticity** | Turrigiano, 2008 | Synaptic scaling & threshold adaptation |
| **Critical Periods** | Hensch, 2005 | Stage-dependent plasticity modulation |
| **Synaptic Pruning** | Huttenlocher, 1979 | Experience-dependent elimination |
| **BDNF Signaling** | Lu, 2003 | Activity-dependent growth factors |
| **Neurogenesis** | Eriksson et al., 1998 | New neuron creation during development |

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│              BIOLOGICALLY PLAUSIBLE NEURAL NETWORK v4                            │
│                    (ESP32-CAM Optimized)                                         │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   INPUT LAYER              HIDDEN LAYER                   OUTPUT LAYER           │
│   (9,216 neurons)          (Dynamic: 50-2000)             (2 neurons)            │
│   96×96 Grayscale          Biological Activation          Sigmoid                │
│                                                                                  │
│   ┌─────┐    BDNF          ┌─────┐     STDP              ┌─────┐                │
│   │ p₀  │─────●────────────│ h₀  │──────●────────────────│ y₀  │ Person         │
│   ├─────┤     ↓            ├─────┤      ↓                ├─────┤                │
│   │ p₁  │───Axon───────────│ h₁  │───Hebbian─────────────│ y₁  │ No Person      │
│   ├─────┤   Guidance       ├─────┤   Learning            └─────┘                │
│   │ ... │      ↓           │ ... │      ↓                                        │
│   ├─────┤  Correlation     ├─────┤   Homeostatic                                │
│   │p₉₂₁₅│───Based──────────│ hₙ  │───Scaling────────────────────                │
│   └─────┘   Growth         └─────┘                                              │
│                                                                                  │
│   ═══════════════════════════════════════════════════════════════════           │
│                     DEVELOPMENTAL STAGES                                        │
│   ═══════════════════════════════════════════════════════════════════           │
│                                                                                  │
│   EMBRYONIC → INFANT → CHILDHOOD → ADOLESCENT → ADULT                           │
│      ↓          ↓          ↓            ↓          ↓                            │
│   High       Synapse    Critical     Aggressive  Stable                         │
│   Neurogenesis Explosion  Period      Pruning    Network                        │
│                                                                                  │
└──────────────────────────────────────────────────────────────────────────────────┘
```

## Core Concepts

### 1. Biological Activation Function (F-I Curve)

Unlike standard ReLU/Sigmoid, we approximate the **Firing Rate vs Input Current** relationship of real neurons:

```python
# Biological F-I Curve Approximation (Gerstner & Kistler, 2002)
f(x) = saturation × sigmoid(steepness × (x - threshold))

# Properties:
# - Below threshold: no firing (sparse coding)
# - Above threshold: monotonic increase with saturation
# - Smooth (differentiable for training)
# - ESP32-friendly (no exp overflow)
```

```
         Firing Rate
              │
         1.0  │                    ●●●●●●●●●●●●●
              │                ●●●●
              │             ●●●
              │           ●●
              │         ●●
         0.0  │●●●●●●●●●●─────────────────────────
              └──────────┼───────────────────────→ Input Current
                      threshold
```

### 2. STDP - Spike-Timing Dependent Plasticity

The **gold standard** of biological learning (Bi & Poo, 1998):

```
         Δw (Weight Change)
              │
         LTP  │    ●
              │   ●●
              │  ●●●
         ─────┼●●●●●●────────────
              │       ●●●
         LTD  │        ●●
              │         ●
              └───────────────────→ Δt = t_post - t_pre
                 post    pre
                before  before
                 pre    post
```

**Rate-based approximation:**
```python
# LTP (Long-Term Potentiation): "Fire together, wire together"
ltp = A₊ × pre_activity × post_activity × (w_max - w)

# LTD (Long-Term Depression): Decorrelation
ltd = A₋ × (1 - pre_activity) × post_activity × (w - w_min)

# Net change with neuromodulator (dopamine/reward)
Δw = learning_rate × neuromodulator × (ltp - ltd)
```

### 3. Homeostatic Plasticity

Neurons maintain stable firing rates through **synaptic scaling** (Turrigiano, 2008):

```python
# Synaptic Scaling
w_i ← w_i × (r_target / r_actual)

# Intrinsic Plasticity (Threshold Adaptation)
θ ← θ + η × (r_actual - r_target)
```

**Purpose:**
- Prevents runaway excitation (seizures)
- Prevents network silence
- Maintains functional dynamic range

### 4. Developmental Stages

The network progresses through **biologically accurate developmental stages**:

| Stage | Age (steps) | Neurogenesis | Synaptogenesis | Pruning | Plasticity |
|-------|-------------|--------------|----------------|---------|------------|
| **EMBRYONIC** | 0-500 | ⬆⬆⬆ High | ⬆ Moderate | ❌ None | 50% |
| **INFANT** | 500-2000 | ⬆ Moderate | ⬆⬆⬆ Explosion | ⬇ Minimal | 100% |
| **CHILDHOOD** | 2000-5000 | ⬇ Low | ⬆⬆ Moderate | ⬆ Start | 90% |
| **ADOLESCENT** | 5000-10000 | ❌ None | ⬇ Low | ⬆⬆⬆ Aggressive | 60% |
| **ADULT** | 10000+ | ❌ None | ⬇⬇ Rare | ⬆⬆ Maintenance | 30% |

### 5. BDNF-like Growth Factors

**Brain-Derived Neurotrophic Factor** promotes growth:

```python
BDNF_level = 0.3 × activity + 0.2 × |error_signal|

# Effects:
# - High BDNF → promotes neurogenesis
# - High BDNF → promotes synaptogenesis
# - Low BDNF → synapse elimination
```

### 6. Axon Guidance

New neurons connect based on biological guidance cues:

```python
connection_probability = (
    0.30 × distance_factor +      # Local connectivity bias
    0.35 × activity_gradient +    # Connect to active regions
    0.35 × bdnf_gradient          # Growth factor tropism
)
```

## Hebbian Learning

Based on Donald Hebb's postulate (1949): *"Neurons that fire together, wire together."*

The network uses correlation-based learning inspired by biological synaptic plasticity:

```python
# Oja's Rule (Normalized Hebbian Learning)
Δw = η · y · (x - y · w)

# Where:
# η = learning rate
# y = post-synaptic activation
# x = pre-synaptic activation
# w = current weight
```

This rule ensures:
- Weights grow when pre/post neurons are correlated
- Automatic normalization prevents unbounded growth
- Competitive learning emerges naturally

### 2. Structural Plasticity (Growth Rules)

Implemented based on [Soltoggio et al. 2008](https://ieeexplore.ieee.org/document/4634274), the network dynamically modifies its topology using four rules:

#### Rule A: Correlation-based Connection Growth
```
IF corr(neuron_i, neuron_j) > threshold AND NOT connected(i, j):
    CREATE synapse(i → j)
```
Creates new connections between neurons that fire together.

#### Rule B: Bridging Rule
```
IF corr(A, B) > strong_threshold AND loss > loss_threshold:
    new_neuron = CREATE hidden neuron
    REMOVE synapse(A → B)
    CREATE synapse(A → new_neuron)
    CREATE synapse(new_neuron → B)
```
Increases network depth by inserting neurons between strongly correlated pairs.

```
Before:  A ────────────────→ B

After:   A ─────→ [new] ─────→ B
```

#### Rule C: Branching Rule
```
IF neuron.activity_mean > high_threshold:
    child = CLONE neuron (partial connections)
    CONNECT child to same outputs
```
Increases network width by duplicating overactive neurons.

```
Before:  [parent] ─────────→ [output]

After:   [parent] ─────────→ [output]
              │
         [child] ──────────→ [output]
```

#### Rule D: Separation Rule
```
IF corr(neuron_i, neuron_j) < anti_correlation_threshold:
    CREATE two new hidden neurons
    CONNECT i → new_i
    CONNECT j → new_j
```
Separates anti-correlated processing pathways.

### 3. Magnitude-based Pruning

Global unstructured pruning removes weak synapses to maintain network efficiency:

```python
threshold = percentile(|all_weights|, pruning_rate × 100)

FOR each synapse:
    IF |weight| < threshold:
        REMOVE synapse

FOR each neuron:
    IF no_incoming_connections AND no_outgoing_connections:
        IF neuron.type == HIDDEN:
            REMOVE neuron
```

## Version Comparison

| Feature | v1 | v2 | v3 | v4 | **v5 (Hybrid)** |
|---------|----|----|-----|-----|-----------------|
| Backend | NumPy | NumPy | PyTorch/CUDA | PyTorch/CUDA | **PyTorch/CUDA** |
| Activation | Sigmoid | Sigmoid | ReLU + Sigmoid | F-I Curve | **F-I Curve + Adaptive** |
| Learning Rule | Delta | Hebbian | STDP-lite | STDP + Homeostatic | **GPU-STDP + Homeostatic** |
| Hidden Layer Forward | ❌ Bug | ❌ Bug | ✅ Topo Sort | ✅ Topo Sort | **✅ Topo Sort** |
| Growth Control | None | Fixed Rules | Fixed Rules | Developmental | **Developmental + BDNF** |
| Pruning | None | Magnitude | Magnitude | Activity+Age+Mag | **Activity+Age+Mag** |
| ESP32 Support | ❌ | ❌ | ❌ | ✅ Int8 | **✅ Int8 (RGB)** |
| Input Size | 128×128 | 128×128 | 128×128 | 96×96 Gray | **160×120 RGB** |
| Input Neurons | 16,384 | 16,384 | 16,384 | 9,216 | **57,600** |
| Color Support | ❌ | ❌ | ❌ | ❌ | **✅ RGB** |
| Max Model Size | Unlimited | Unlimited | Unlimited | ~150 KB | **~200 KB** |
| Biological Accuracy | ⭐ | ⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | **⭐⭐⭐⭐⭐** |
| Training Speed | ⭐ | ⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | **⭐⭐⭐⭐** |

## Mathematical Foundation

### Forward Pass

For each neuron in topological order:

```
z_j = Σᵢ (wᵢⱼ · aᵢ)           # Weighted sum of inputs

a_j = ReLU(z_j)               # If hidden neuron
a_j = σ(z_j)                  # If output neuron

Where:
  ReLU(x) = max(0, x)
  σ(x) = 1 / (1 + e⁻ˣ)
```

### Weight Update (Delta Rule + Hebbian)

```
# Output neurons (Delta Rule / Gradient Descent approximation)
Δwᵢⱼ = η · (target_j - output_j) · aᵢ

# Hidden→Output (Oja's Rule)
Δwᵢⱼ = η · aⱼ · (aᵢ - aⱼ · wᵢⱼ)
```

### Loss Function

Mean Squared Error (MSE):
```
L = (1/n) · Σⱼ (yⱼ - ŷⱼ)²
```

### Correlation Matrix

Pearson correlation computed on GPU:

```python
# Center the data
X_centered = X - mean(X, axis=0)

# Normalize
X_norm = X_centered / std(X_centered, axis=0)

# Correlation matrix
C = (X_norm.T @ X_norm) / n
```

## Hyperparameters

### Growth Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `corr_thresh` | 0.6 | Correlation threshold for Rule A |
| `strong_corr` | 0.8 | Strong correlation for Rule B |
| `anti_corr` | -0.4 | Anti-correlation for Rule D |
| `loss_thresh` | 0.15 | Loss threshold for Rule B |
| `high_activity` | 0.7 | Activity threshold for Rule C |
| `max_neurons` | 25,000 | Maximum network size |
| `max_new_per_step` | 30 | New neurons per growth step |

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `learning_rate` | 0.01 | Weight update rate |
| `epochs` | 50 | Training iterations |
| `growth_interval` | 100 | Steps between growth |
| `prune_interval` | 500 | Steps between pruning |
| `pruning_rate` | 0.05 | Fraction of weights to prune |
| `history_length` | 32 | Activity buffer size |
| `ema_alpha` | 0.1 | Activity mean smoothing |

## File Structure

```
Neural Network/
├── Code/
│   ├── notebook version 1 (standard learning).ipynb    # Basic implementation
│   ├── notebook version 2, hebbian learning.ipynb      # Hebbian + Pruning
│   ├── notebook version 3, cuda accelerated.ipynb      # CUDA + ReLU + Fixes
│   ├── notebook version 4, biologically accurate.ipynb # Full biological model
│   ├── notebook version 5, hybrid biological cuda.ipynb # 🆕 Hybrid (Best of v3+v4)
│   ├── saves_v3/
│   │   ├── network_epoch_*.h                           # C-Header exports
│   │   └── training_history.png                        # Loss/Accuracy plots
│   ├── saves_v4_bio/
│   │   ├── network_final.h                             # ESP32 C header
│   │   ├── network_final.c                             # ESP32 C source
│   │   ├── network_final.bin                           # Quantized binary weights
│   │   └── training_history.png                        # Development visualization
│   ├── saves_v5/                                       # 🆕 v5 outputs
│   │   ├── final/
│   │   │   ├── network_v5.h                            # ESP32 RGB header
│   │   │   ├── network_v5.c                            # ESP32 RGB source
│   │   │   ├── network_v5.bin                          # Quantized RGB weights
│   │   │   └── metadata.json                           # Model metadata
│   │   └── training_history.png                        # Colored by dev stage
│   └── README.md
└── dataset/
    └── human detection dataset/
        ├── 1/  (positive samples - humans)
        └── 0/  (negative samples - no humans)
```

## ESP32-CAM Deployment

### Hardware Requirements

- **ESP32-CAM AI Thinker** (or compatible)
- **SRAM**: 520 KB (uses ~150-200 KB)
- **PSRAM**: Optional (for larger models)
- **Flash**: 4 MB

### Memory Budget (v5 RGB Model)

| Component | Size | Notes |
|-----------|------|-------|
| Weights (int8) | ~65 KB | Quantized from float32 |
| Synapse Indices | ~130 KB | uint16 pairs (65k synapses) |
| Thresholds | ~4 KB | Quantized int8 |
| Activations | ~16 KB | Runtime float32 |
| **Total** | **~200 KB** | Fits in SRAM |

### Generated Files (v5)

```c
// network_v5.h
#define IMG_WIDTH     160
#define IMG_HEIGHT    120
#define IMG_CHANNELS  3
#define NUM_INPUTS    57600  // 160×120×3 RGB
#define NUM_HIDDEN    500    // Grown during training
#define NUM_OUTPUTS   2
#define NUM_NEURONS   58102
#define NUM_SYNAPSES  65000  // Pruned for efficiency
#define WEIGHT_SCALE  63.5f  // Int8 quantization scale

// Biological activation (ESP32-optimized)
float biological_activation(float x, float threshold) {
    float steepness = 5.0f;
    return 1.0f / (1.0f + expf(-steepness * (x - threshold)));
}
```

### Integration Example (v5 RGB)

```c
#include "network_v5.h"
#include "esp_camera.h"

void detect_human_rgb() {
    camera_fb_t *fb = esp_camera_fb_get();

    // Preprocess: resize to 160x120 RGB, normalize
    uint8_t rgb_input[NUM_INPUTS];
    preprocess_rgb_image(fb->buf, rgb_input, 160, 120);

    // Initialize network state
    NetworkState state;
    network_init(&state);

    // Inference
    float confidence;
    int prediction = network_forward(&state, rgb_input, &confidence);

    // Result
    const char* classes[] = {"No Person", "Person"};
    printf("Detection: %s (%.1f%%)\n",
           classes[prediction],
           confidence * 100);

    esp_camera_fb_return(fb);
}
```
```

## Usage

### Requirements

```bash
pip install torch numpy matplotlib opencv-python scikit-learn
```

### Quick Start

```python
from notebook_v3 import DynamicNeuralNetwork, train_network

# Initialize
network = DynamicNeuralNetwork(
    num_inputs=16384,  # 128×128 pixels
    num_outputs=2,     # Binary classification
    device=torch.device('cuda')
)

# Add initial hidden neurons
for _ in range(20):
    new_id = network.add_neuron(neuron_type=1)
    # Connect to inputs and outputs...

# Train
history = train_network(
    network=network,
    train_X=images,
    train_y=labels,
    epochs=50
)

# Export for embedded systems
network.export_to_header("model.h")
```

### Inference

```python
# Load image
image = cv2.imread("test.jpg", cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, (128, 128))
x = image.flatten() / 255.0

# Predict
output = network.forward(torch.tensor(x, device=DEVICE))
prediction = output.argmax().item()
confidence = output[prediction].item()

print(f"Prediction: {'Person' if prediction == 1 else 'No Person'}")
print(f"Confidence: {confidence:.2%}")
```

## Embedded Deployment

The network exports to a C-header file for microcontrollers:

```c
// network_final.h
#define NUM_INPUTS   16384
#define NUM_HIDDEN   127
#define NUM_OUTPUTS  2
#define NUM_NEURONS  16513
#define NUM_SYNAPSES 45892

struct Synapse {
  int from_id;
  int to_id;
  float weight;
};

const Synapse network_synapses[45892] = {
  {0, 16386, 0.023456f},
  {1, 16387, -0.012345f},
  // ...
};
```

## Performance Benchmarks

| Metric | v2 (CPU) | v3 (CUDA) | Speedup |
|--------|----------|-----------|---------|
| Forward Pass (1 sample) | 15ms | 0.8ms | **19×** |
| Correlation Matrix (256 neurons) | 45ms | 2ms | **23×** |
| Training Epoch (500 samples) | 12s | 1.5s | **8×** |
| Memory (32k synapses) | 6.5 MB | 384 KB | **17×** |

## Biological Inspiration

This architecture draws from neuroscience concepts:

1. **STDP** - Spike-Timing Dependent Plasticity (Bi & Poo, 1998)
2. **Homeostatic Plasticity** - Synaptic scaling (Turrigiano, 2008)
3. **Critical Periods** - Time-limited plasticity windows (Hensch, 2005)
4. **Synaptic Pruning** - Experience-dependent elimination (Huttenlocher, 1979)
5. **Neurogenesis** - New neuron creation (Eriksson et al., 1998)
6. **BDNF Signaling** - Activity-dependent growth (Lu, 2003)
7. **Axon Guidance** - Growth cone navigation (Tessier-Lavigne & Goodman, 1996)
8. **Sparse Coding** - Efficient representation (Olshausen & Field, 1996)

## Brain Development Analogy

```
Human Brain Development          This Network
═════════════════════════        ═════════════════════════

Prenatal (Embryonic)             EMBRYONIC Stage
├─ Massive neurogenesis          ├─ High neuron creation rate
├─ Neural tube formation         ├─ Basic structure established
└─ Random synapse formation      └─ Sparse random connections

Infant (0-2 years)               INFANT Stage
├─ Synaptogenesis explosion      ├─ synaptogenesis_prob = 0.5
├─ 2× adult synapse density      ├─ Network doubles in size
└─ Critical period begins        └─ plasticity = 1.0 (maximum)

Childhood (2-11 years)           CHILDHOOD Stage
├─ Experience-dependent          ├─ corr_based synaptogenesis
├─ Critical periods peak         ├─ High STDP learning rate
└─ Language/visual learning      └─ Activity-dependent growth

Adolescence (12-18 years)        ADOLESCENT Stage
├─ Aggressive pruning            ├─ pruning_threshold = 0.1
├─ 50% synapse elimination       ├─ "Use it or lose it"
└─ Prefrontal maturation         └─ Network optimization

Adult (18+ years)                ADULT Stage
├─ Stable network                ├─ Minimal structural changes
├─ Reduced plasticity            ├─ plasticity = 0.3
└─ Lifelong learning             └─ Continued STDP (slower)
```

## References

### Primary Sources

- **Bi, G. Q., & Poo, M. M.** (1998). *Synaptic modifications in cultured hippocampal neurons: dependence on spike timing, synaptic strength, and postsynaptic cell type.* Journal of Neuroscience, 18(24), 10464-10472.

- **Turrigiano, G. G.** (2008). *The self-tuning neuron: synaptic scaling of excitatory synapses.* Cell, 135(3), 422-435.

- **Hensch, T. K.** (2005). *Critical period plasticity in local cortical circuits.* Nature Reviews Neuroscience, 6(11), 877-888.

- **Huttenlocher, P. R.** (1979). *Synaptic density in human frontal cortex-developmental changes and effects of aging.* Brain Research, 163(2), 195-205.

- **Eriksson, P. S., et al.** (1998). *Neurogenesis in the adult human hippocampus.* Nature Medicine, 4(11), 1313-1317.

### Additional References

- Hebb, D.O. (1949). *The Organization of Behavior*
- Soltoggio, A., et al. (2008). *Evolving Neuromodulatory Topologies*
- Oja, E. (1982). *Simplified neuron model as a principal component analyzer*
- Gerstner, W., & Kistler, W. M. (2002). *Spiking Neuron Models*
- Olshausen, B. A., & Field, D. J. (1996). *Sparse coding in V1*
- Han, S., et al. (2015). *Learning both Weights and Connections for Efficient Neural Networks*

## License

MIT License - See LICENSE file for details.

## Contributing

Contributions welcome! Please open an issue or PR for:
- Bug fixes
- Performance optimizations
- New growth/pruning rules
- Documentation improvements
