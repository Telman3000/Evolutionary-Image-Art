# Creative Image Interpreter

A parallel multi-level genetic algorithm that recreates 512×512 images from thousands of semi-transparent pentagons, with an arcade-machine visualization.

## Overview

This project implements a from-scratch evolutionary algorithm that approximates target images using overlapping geometric primitives (pentagons). The goal is to evolve a recognizable but stylized reinterpretation of input images within a 10-minute computational budget, using both crossover and mutation operators, without relying on external EA libraries.

The system progressively evolves populations of candidate solutions through a coarse-to-fine multi-level approach, combining mutation, crossover, tournament selection, elitism, and local hill-climbing to achieve visually coherent reconstructions within strict time constraints.

The algorithm renders each candidate solution as a composition of semi-transparent pentagons, evaluates fitness using a hybrid metric combining pixel-level error and perceptual similarity, and iteratively refines the approximation through six hierarchical levels—starting with large shapes capturing broad color regions, then progressively adding smaller pentagons to capture fine edges and texture details.

## Algorithm Description

### Chromosome Representation

Each **individual** (candidate solution) is encoded as a list of **pentagons**, where each pentagon is defined by:

- **Position**: `(x, y)` coordinates on the 512×512 canvas
- **Color**: `(r, g, b)` RGB values (0–255)
- **Alpha**: transparency value (0–255)
- **Size**: radius in pixels (1–300)
- **Rotation**: angle in radians
- **Edge flag**: boolean indicating placement on detected edges

A complete individual may contain 2000–5000+ pentagons by the final refinement level.

### Multi-Level Coarse-to-Fine Evolution

The algorithm operates through **six refinement levels**, each introducing pentagons with progressively smaller size ranges and finer detail:

| Level | Pentagons Added | Size Range (px) | Generations | Alpha Range | Edge Bias |
|-------|----------------|-----------------|-------------|-------------|-----------|
| 1     | 600            | 10–28          | 10          | 30–90       | 0.02      |
| 2     | 900            | 6–18           | 16          | 40–120      | 0.06      |
| 3     | 1000           | 3–10           | 28          | 60–180      | 0.18      |
| 4     | 1400           | 2–7            | 36          | 80–220      | 0.36      |
| 5     | 2200           | 1–4            | 48          | 100–240     | 0.62      |
| 6     | 3200           | 1–2            | 60          | 120–255     | 0.88      |

At each level:
1. **Initialization**: New pentagons are placed using a hybrid strategy:
   - **Edge-based placement**: Positions sampled from edge-detected regions (proportional to edge bias)
   - **Error-based placement**: High-error regions identified by comparing current rendering to target
   - **Random placement**: Uniform sampling for diversity
2. **Color sampling**: Colors initialized from target image with neighborhood averaging (sample radius decreases across levels: 6 → 3 → 1 pixels)
3. **Size-dependent desaturation**: Larger shapes are desaturated toward grayscale for better color composition
4. **Evolution**: Standard genetic algorithm loop (tournament selection, crossover, mutation, fitness evaluation)
5. **Hill-climbing refinement**: Local optimization applied to best individual

### Selection, Crossover, and Mutation

- **Tournament selection** (size = 3): Randomly sample 3 individuals, select the fittest as parent
- **Elitism**: Top 2 individuals always survive to next generation
- **Crossover** (rate = 0.75): Cut both parents' pentagon lists at random point, concatenate first half of parent 1 with second half of parent 2
- **Mutation** (dynamic rate): Starts at 0.40, decays exponentially to 0.06 as generation progresses

    Mutation operations (applied probabilistically):
    - **Color jitter**: Add random noise (±12 scaled by strength) to RGB values, with small chance to replace with palette color
    - **Position displacement**: Shift `(x, y)` by random offset (±6 scaled by strength); edge-flagged pentagons have reduced mutation probability
    - **Size adjustment**: Increment/decrement size by ±1, clamped to level's size range
    - **Alpha perturbation**: Add random noise (±30 scaled by strength) to transparency

### Fitness Function

Fitness is a **composite metric** combining multiple quality measures:

```
MSE = mean squared error between rendered and target
PSNR = 10 × log₁₀(1 / MSE)  [peak signal-to-noise ratio]
SSIM = structural similarity index  [if scikit-image available]
edge_match = 1 - |gradient(rendered) - gradient(target)|  [edge alignment]
```

**Final fitness formula**:
```
similarity = {
    100 × SSIM + 4 × edge_match,           if SSIM available
    100 × min(PSNR, 50) / 50 + 4 × edge_match,  otherwise
}

neglog = -log₁₀(MSE + 10⁻⁸)
fitness = 40 × similarity + 100 × neglog + 300 × edge_match
```

Higher values indicate better approximations. The metric balances pixel-level accuracy (MSE/PSNR) with perceptual quality (SSIM) and structural detail (edge matching).

### Local Hill-Climbing

After main evolution at each level, the best individual undergoes **local optimization** (80–120 iterations):
1. Create candidate by applying small mutations (reduced strength)
2. Evaluate fitness
3. Keep candidate only if fitness improves (greedy acceptance)

This refinement step captures local improvements missed by crossover-heavy evolution.

### Parallelism and Time Limits

- **Parallel execution**: Each run launches **5 independent evolutions** using `ProcessPoolExecutor`, utilizing up to 12 CPU cores (70% of available cores, capped)
- **Shared memory optimization**: Target images, downscaled versions, edge maps, and color palettes are initialized once per worker process via `worker_init`, avoiding re-serialization overhead
- **Time limit**: Each run is constrained to **9.5267 minutes** (570.16 seconds). If exceeded, remaining evolutions are cancelled gracefully
- **Total runs**: **3 runs** per input image, producing 15 evolutions total (5 per run)

### Multi-Resolution Evaluation

To accelerate fitness computation during early generations, individuals are rendered and evaluated at progressively increasing resolutions across levels:

```
Level 1-2: 32×32 or 48×48
Level 3-4: 64×64 or 80×80
Level 5-6: 96×96 or 112×112
Final:     512×512 (full resolution)
```

### Additional Optimizations

- **Fast vectorized rendering**: Pentagons rasterized using NumPy-accelerated alpha blending (shapes rendered as circles/diamonds depending on size)
- **Edge detection**: Gradient-based edge maps guide pentagon placement
- **K-means color palette**: 18–20 representative colors extracted from target, used for coarse layer initialization and occasional mutation
- **Detail pass**: After final level, 3000 tiny pentagons (size 1–4) added near edges and high-error regions
- **Fitness caching**: Rendered images cached per individual to avoid redundant computation

## Configuration

All evolutionary parameters are defined in the `Config` class:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `IMAGE_SIZE` | 512 | Target image resolution (512×512 px) |
| `POPULATION_SIZE` | 10 | Number of individuals per generation |
| `ELITE_SIZE` | 2 | Best individuals preserved each generation |
| `TOURNAMENT_SIZE` | 3 | Individuals competing in selection |
| `CROSSOVER_RATE` | 0.75 | Probability of crossover operation |
| `MUTATION_RATE_START` | 0.40 | Initial mutation rate (generation 0) |
| `MUTATION_RATE_END` | 0.06 | Final mutation rate (last generation) |
| `NUM_THREADS_PER_EVOLUTION` | 1 | Thread count per evolution process |
| `RUN_TIME_LIMIT_MINUTES` | 9.5267 | Maximum runtime per run (minutes) |

The mutation rate decays exponentially: `rate(t) = 0.06 + (0.40 - 0.06) × exp(-3 × progress)`, where `progress = generation / max_generations`.

## Requirements

- Python 3.7+
- NumPy (≥1.21.0) — vectorized computation, array operations
- Pillow (≥9.0.0) — image loading, rendering, saving
- matplotlib (≥3.4.0) — fitness curve plotting
- scikit-image (≥0.18.0) — SSIM computation (optional; PSNR used as fallback)

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Running the Algorithm

1. Place a 512×512 input image in the working directory as `input1.jpg` (or `input2.jpg`, etc.)
2. Edit `INPUT_IMAGE` global variable in `evolution.py` to match your image index (e.g., `INPUT_IMAGE = 1`)
3. Run:

```bash
python evolution.py
```

### Output Files

Results are saved to the `outputs/` directory (created automatically):

- **Images**: `TelmanNuruzovOutput{X}_{Y}_{Z}.jpg`
  - `X` = input image index
  - `Y` = run number (1–3)
  - `Z` = ranking within run (1 = best, 2 = second best, ..., 5 = fifth best)
  
  Each image is embedded in a retro **arcade machine frame** (512×512) with 3D-styled controls, coin slot, and watermelon branding.

- **Fitness plots**: `fitness_graphs_input{X}.png`
  - Three subplots (one per run) showing average and maximum fitness across generations
  - X-axis: generation number
  - Y-axis: negative L1 loss (proxy for fitness)

### Arcade Machine Visualization

The system wraps generated images in a stylized arcade cabinet visualization featuring:
- Recessed screen area with bezel and shadows (displaying the evolved image)
- 3D joystick and buttons (red and black with gradient shading)
- Coin slot with decorative watermelon slice
- Title banner: "RETRO ARCADE MACHINE" / "WATERMELON ARCADE"
- Gradient effects and lighting for depth

This aesthetic output transforms the technical results into a visually engaging presentation format suitable for creative showcasing.

## Results and Statistics

Experimental results from three independent runs on test images demonstrate consistent convergence within the 10-minute time budget:

| Run | Best Similarity (%) | Runtime (minutes) | Pentagons |
|-----|--------------------:|------------------:|----------:|
| 1   | XX.X                | X.XX              | ~3000–5000 |
| 2   | XX.X                | X.XX              | ~3000–5000 |
| 3   | XX.X                | X.XX              | ~3000–5000 |

*(Actual values depend on input image complexity; see [full report](docs/report.pdf) for detailed statistics)*

### Convergence Behavior

Fitness curves consistently exhibit:
1. **Rapid early improvement**: Coarse levels (1–2) achieve steep fitness increases as large shapes capture global color distribution
2. **Gradual refinement**: Mid-to-late levels (3–6) show diminishing returns as smaller pentagons optimize edge alignment and texture
3. **Stable convergence**: Maximum fitness (elite) consistently exceeds average fitness, indicating effective selection pressure
4. **Multi-level plateaus**: Discrete jumps between levels as new pentagons are added

The system successfully completed all evolutions across runs before time limits or gracefully terminated when constraints were reached, demonstrating robustness.

## Implementation Details

### Libraries and Dependencies

- **NumPy**: Vectorized rendering buffers, alpha blending, gradient computation, fitness evaluation, K-means clustering, edge detection
- **Pillow (PIL)**: Image I/O, resizing (Lanczos resampling), Gaussian blur (error map smoothing), arcade frame compositing, font rendering
- **scikit-image**: SSIM metric for structural similarity (optional; fallback to PSNR if unavailable)
- **matplotlib**: Offline fitness curve plotting (saved to PNG; not used during evolution)
- **multiprocessing / concurrent.futures**: Parallel evolution scheduling, worker process management, shared memory initialization
- **Python Standard Library**: `random`, `math`, `time`, `os` for randomization, logging, timing, and filesystem operations

All components implemented without external machine learning frameworks, ensuring full reproducibility on standard Python scientific stack.

### Code Structure

- **`Config`**: Centralized parameter storage
- **`Pentagon`**: Geometric primitive with position, color, size, alpha, rotation
- **`Individual`**: Candidate solution (list of pentagons + fitness + rendering cache)
- **`FastRenderer`**: Vectorized rasterizer with dynamic shape selection (circles/diamonds)
- **`ColorSampler`**: Neighborhood-averaged color sampling from target
- **`detect_edges_fast`**: Gradient-magnitude-based edge detection
- **`kmeans_palette`**: K-means color quantization
- **`evolve_single_image`**: Main evolution loop (multi-level + hill-climbing)
- **`create_arcade_machine_with_screen`**: 3D arcade frame generation
- **`plot_fitness_graphs`**: Matplotlib-based fitness visualization
- **`main_with_arcade`**: Entry point (image loading, parallel execution, output saving)

### Performance Considerations

- **Memory efficiency**: Pentagon class uses `__slots__` to reduce instance memory footprint when handling thousands of shapes
- **Render caching**: Individuals cache rendered images at each resolution to avoid redundant rasterization
- **Lazy SSIM fallback**: Graceful degradation to PSNR if scikit-image unavailable or SSIM computation fails
- **Cancellation handling**: Futures cancelled on time limit; partial results always saved

## Future Extensions

- **GPU-accelerated rendering**: Migrate rasterization to PyTorch/CuPy for 10-100× speedup
- **Adaptive level transitions**: Automatically determine when to advance levels based on fitness plateau detection
- **Multi-objective optimization**: Pareto fronts balancing fitness vs. pentagon count vs. aesthetic diversity
- **Shape grammars**: Support for triangles, circles, bezier curves beyond pentagons
- **Interactive tuning**: Web UI for real-time parameter adjustment and evolution visualization

## Design Goals

This project explores creative evolutionary algorithms that produce recognizable but distinct interpretations of input images within a 10-minute runtime constraint.

Key design choices:
- Implementation uses both crossover and mutation operators
- Stability across multiple runs (demonstrated via statistics)
- Balance between recognizability and artistic interpretation
- No external EA libraries—fully implemented from scratch
- Comprehensive technical documentation

For full methodology, experimental analysis, and fitness curve comparisons, see the [technical report](docs/report.pdf).

## Author

**Telman Nuruzov**

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

Debugging assistance provided by GPT (as noted in code header). All algorithmic design, implementation, and experimentation conducted independently.
