# Telman Nuruzov DSAI-03 
# Used GPT for debaging code

"""
Genetic Algorithm-based Image Evolution System with Arcade Machine Visualization

This module implements a parallel evolutionary algorithm that evolves images
by generating them from geometric shapes (pentagons). The system uses multiple
evolutionary levels with increasing detail, and can optionally render results
as arcade machine graphics.

Key Features:
- Multi-level evolutionary approach with progressive detail refinement
- Parallel processing using ProcessPoolExecutor for CPU-intensive operations
- Advanced fitness metrics combining MSE, PSNR, SSIM, and edge matching
- Dynamic shape rendering optimized for different size ranges
- Arcade machine visualization with 3D effects

"""

import os
import time
import math
import random
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple, Dict
from PIL import Image, ImageFilter, ImageDraw, ImageFont
import numpy as np


# GRAPH PLOTTING SECTION 

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    _PLOTTING_AVAILABLE = True
except ImportError:
    _PLOTTING_AVAILABLE = False


try:
    from skimage.metrics import structural_similarity as ssim_fn
    _SSIM_AVAILABLE = True
except Exception:
    _SSIM_AVAILABLE = False

INPUT_IMAGE = 1
OUTPUT_DIR = "outputs"
RUN_TIME_LIMIT_MINUTES = 9.5267  # Time limit per run in minutes
os.makedirs(OUTPUT_DIR, exist_ok=True)


class Config:
    """
    Configuration class containing all evolutionary algorithm parameters.
    
    Attributes:
        IMAGE_SIZE (int): Target image resolution (512x512)
        LEVELS (List[Tuple]): Evolution levels configuration.
            Each tuple: (num_pentagons, size_range, max_generations, alpha_range, edge_bias)
        POPULATION_SIZE (int): Number of individuals in each generation
        ELITE_SIZE (int): Number of best individuals preserved between generations
        TOURNAMENT_SIZE (int): Number of individuals competing in tournament selection
        CROSSOVER_RATE (float): Probability of crossover operation (0.0-1.0)
        MUTATION_RATE_START (float): Initial mutation rate at generation start
        MUTATION_RATE_END (float): Final mutation rate at generation end
        NUM_THREADS_PER_EVOLUTION (int): Thread count per evolution process
    """
    IMAGE_SIZE = 512
    
    LEVELS = [
        (600,  (10, 28), 10, (30, 90), 0.02),
        (900,  (6, 18),  16, (40, 120), 0.06),
        (1000, (3, 10),  28, (60, 180), 0.18),
        (1400, (2, 7),   36, (80, 220), 0.36),
        (2200, (1, 4),   48, (100, 240), 0.62),
        (3200, (1, 2),   60, (120, 255), 0.88),
    ]
    POPULATION_SIZE = 10
    ELITE_SIZE = 2
    TOURNAMENT_SIZE = 3
    CROSSOVER_RATE = 0.75
    MUTATION_RATE_START = 0.40
    MUTATION_RATE_END = 0.06
    NUM_THREADS_PER_EVOLUTION = 1


EVAL_RESOLUTIONS = [32, 48, 64, 80, 96, 112]
SIZE_DESAT_FACTOR = 0.82
MAX_PARALLEL_LIMIT = 12

GLOBAL_TARGET_FULL: np.ndarray = None
GLOBAL_TARGETS: Dict[int, np.ndarray] = {}
GLOBAL_EDGE_POINTS: List[Tuple[int, int]] = []
GLOBAL_PALETTE_COARSE: List[Tuple[int, int, int]] = []


def worker_init(target_full, targets_dict, edge_points, palette_coarse):
    """
    Initialize global variables for worker processes in multiprocessing pool.
    
    This function is called once per worker process to set up shared data
    structures, avoiding the need to serialize large arrays for each task.
    
    Args:
        target_full (np.ndarray): Full resolution target image (512x512)
        targets_dict (Dict[int, np.ndarray]): Dictionary of scaled target images
            Key: resolution (int), Value: target image array
        edge_points (List[Tuple[int, int]]): List of edge point coordinates
        palette_coarse (List[Tuple[int, int, int]]): K-means color palette
    """
    global GLOBAL_TARGET_FULL, GLOBAL_TARGETS, GLOBAL_EDGE_POINTS, GLOBAL_PALETTE_COARSE
    GLOBAL_TARGET_FULL = target_full
    GLOBAL_TARGETS = targets_dict
    GLOBAL_EDGE_POINTS = edge_points
    GLOBAL_PALETTE_COARSE = palette_coarse


def compute_psnr(img_a: np.ndarray, img_b: np.ndarray) -> float:
    """
    Compute Peak Signal-to-Noise Ratio (PSNR) between two images.
    
    PSNR measures image quality by comparing pixel differences.
    Higher values indicate better similarity.
    
    Args:
        img_a (np.ndarray): First image array (normalized 0.0-1.0)
        img_b (np.ndarray): Second image array (normalized 0.0-1.0)
    
    Returns:
        float: PSNR value in decibels. Returns 100.0 for identical images.
    """
    mse = float(np.mean((img_a - img_b) ** 2))
    if mse <= 1e-12:
        return 100.0
    PIXEL_MAX = 1.0
    return 10.0 * math.log10((PIXEL_MAX * PIXEL_MAX) / mse)


def calculate_fitness_metrics(rendered: np.ndarray, target: np.ndarray, level: int = 0) -> Tuple[float, float, Dict]:
    """
    Calculate comprehensive fitness metrics combining multiple quality measures.
    
    Uses hybrid approach: MSE for base similarity, SSIM for structural similarity
    (if available), PSNR as fallback, and edge matching for detail preservation.
    
    Args:
        rendered (np.ndarray): Rendered image array (uint8, 0-255)
        target (np.ndarray): Target image array (uint8, 0-255)
        level (int): Current evolution level (0-5). Higher levels use edge matching.
    
    Returns:
        Tuple[float, float, Dict]:
            - fitness (float): Combined fitness score (higher is better)
            - similarity_pct (float): Similarity percentage (0-100)
            - debug (Dict): Dictionary with individual metric values:
                {'mse': float, 'psnr': float, 'ssim': float|None, 'edge': float}
    """
    R = rendered.astype(np.float32) / 255.0
    T = target.astype(np.float32) / 255.0
    mse = float(np.mean((R - T) ** 2))
    ssim_val = None
    if _SSIM_AVAILABLE:
        try:
            gray_r = R.mean(axis=2)
            gray_t = T.mean(axis=2)
            ssim_val = float(ssim_fn(gray_t, gray_r, data_range=1.0))
            ssim_val = max(0.0, min(1.0, ssim_val))
        except Exception:
            ssim_val = None
    psnr = compute_psnr(R, T)
    edge_match = 0.0
    if level >= 2:
        gray_r = R.mean(axis=2)
        gray_t = T.mean(axis=2)
        gx_r = np.abs(np.diff(gray_r, axis=1))
        gx_t = np.abs(np.diff(gray_t, axis=1))
        if gx_t.size > 0 and gx_r.shape == gx_t.shape:
            edge_diff = float(np.mean(np.abs(gx_r - gx_t)))
            edge_match = 1.0 - np.clip(edge_diff / 1.0, 0.0, 1.0)
    if ssim_val is not None:
        similarity_pct = ssim_val * 100.0 + edge_match * 4.0
    else:
        similarity_pct = (min(psnr, 50.0) / 50.0) * 100.0 + edge_match * 4.0
    similarity_pct = float(np.clip(similarity_pct, 0.0, 100.0))
    neglog = -math.log10(mse + 1e-8)
    fitness = similarity_pct * 40.0 + neglog * 100.0 + edge_match * 300.0
    return float(fitness), float(similarity_pct), {'mse': mse, 'psnr': psnr, 'ssim': ssim_val, 'edge': edge_match}


class FastRenderer:
    """
    High-performance renderer for geometric shapes with dynamic shape selection.
    
    Optimizes rendering by choosing different shape algorithms based on size:
    - Large shapes (>=14): Circular rendering
    - Medium shapes (6-13): Diamond-shaped rendering for better coverage
    - Small shapes (<6): Circular rendering
    
    Uses alpha blending for proper color composition and vectorized operations
    for performance.
    """
    
    @staticmethod
    def render_fast(pents: List['Pentagon'], size: int = 512) -> np.ndarray:
        """
        Render list of pentagons into image array.
        
        Args:
            pents (List[Pentagon]): List of pentagon objects to render
            size (int): Output image size (default: 512)
        
        Returns:
            np.ndarray: Rendered image as uint8 array (size x size x 3)
        """
        canvas = np.full((size, size, 3), 255.0, dtype=np.float32)
        alpha_buf = np.zeros((size, size), dtype=np.float32)
        scale = size / 512.0
        pents_sorted = sorted(pents, key=lambda p: p.size, reverse=True)
        for pent in pents_sorted:
            x = int(pent.x * scale)
            y = int(pent.y * scale)
            radius = max(1, int(pent.size * scale))
            y0 = max(0, y - radius)
            y1 = min(size, y + radius + 1)
            x0 = max(0, x - radius)
            x1 = min(size, x + radius + 1)
            if x0 >= x1 or y0 >= y1:
                continue
            ys = np.arange(y0, y1)[:, None]
            xs = np.arange(x0, x1)[None, :]
            dy = ys - y
            dx = xs - x
            if pent.size >= 14:
                mask = (dx*dx + dy*dy) <= (radius * radius)
            elif pent.size >= 6:
                mask = (np.abs(dx) + np.abs(dy)) <= radius
            else:
                mask = (dx*dx + dy*dy) <= (radius * radius)
            if not mask.any():
                continue
            c = np.array([pent.r, pent.g, pent.b], dtype=np.float32)
            alpha = float(pent.alpha) / 255.0
            sub_canvas = canvas[y0:y1, x0:x1]
            sub_alpha = alpha_buf[y0:y1, x0:x1]
            mask_f = mask.astype(np.float32)[..., None]
            old_alpha = sub_alpha[..., None]
            blend_alpha = alpha + old_alpha * (1.0 - alpha)
            denom = np.where(blend_alpha == 0.0, 1.0, blend_alpha)
            sub_canvas[:] = np.where(mask_f,
                                     (c * alpha + sub_canvas * old_alpha * (1.0 - alpha)) / denom,
                                     sub_canvas)
            sub_alpha[:] = np.where(mask, blend_alpha[..., 0], sub_alpha)
        canvas = np.clip(canvas, 0, 255).astype(np.uint8)
        return canvas


class ColorSampler:
    """
    Color sampling utility that always returns averaged colors over neighborhoods.
    
    Prevents single-pixel color reads by averaging over a radius, providing
    more stable and representative color values for initialization.
    
    Attributes:
        image (np.ndarray): Source image for color sampling
        h (int): Image height
        w (int): Image width
    """
    
    def __init__(self, image: np.ndarray):
        """
        Initialize color sampler with source image.
        
        Args:
            image (np.ndarray): Source image array (float32 or uint8)
        """
        self.image = image
        self.h, self.w = image.shape[:2]

    def sample(self, x: int, y: int, radius: int = 1) -> Tuple[int, int, int]:
        """
        Sample color at position, averaged over neighborhood.
        
        Always returns average color over specified radius, never single-pixel reads.
        This provides more stable color initialization.
        
        Args:
            x (int): X coordinate
            y (int): Y coordinate
            radius (int): Sampling radius (minimum 1, ensures averaging)
        
        Returns:
            Tuple[int, int, int]: RGB color tuple (0-255)
        """
        xx = max(0, min(self.w-1, int(round(x))))
        yy = max(0, min(self.h-1, int(round(y))))
        radius = max(1, int(radius))
        x1 = max(0, xx-radius)
        x2 = min(self.w, xx+radius+1)
        y1 = max(0, yy-radius)
        y2 = min(self.h, yy+radius+1)
        patch = self.image[y1:y2, x1:x2]
        if patch.size == 0:
            v = self.image[yy, xx].astype(int)
            return int(v[0]), int(v[1]), int(v[2])
        avg = patch.mean(axis=(0, 1))
        return int(avg[0]), int(avg[1]), int(avg[2])


def detect_edges_fast(image: np.ndarray) -> np.ndarray:
    """
    Fast edge detection using gradient magnitude calculation.
    
    Computes edge strength map by calculating horizontal and vertical gradients,
    then computing magnitude. Result is normalized to 0.0-1.0 range.
    
    Args:
        image (np.ndarray): Input image array (H x W x 3)
    
    Returns:
        np.ndarray: Edge magnitude map (H x W), normalized to [0.0, 1.0]
    """
    gray = image.mean(axis=2).astype(np.float32)
    gx = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
    gy = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
    mag = np.sqrt(gx*gx + gy*gy)
    if mag.max() > 0:
        mag = mag / mag.max()
    return mag


def sample_edge_points(edge_map: np.ndarray, max_points: int = None) -> List[Tuple[int, int]]:
    """
    Sample edge points from edge map using probability-weighted selection.
    
    Points with higher edge strength have higher probability of being selected,
    ensuring better coverage of important image features.
    
    Args:
        edge_map (np.ndarray): Edge magnitude map from detect_edges_fast()
        max_points (int, optional): Maximum number of points to sample.
            Defaults to min(20000, 35% of image pixels)
    
    Returns:
        List[Tuple[int, int]]: List of (x, y) coordinate tuples
    """
    h, w = edge_map.shape
    if max_points is None:
        max_points = min(20000, int(h*w*0.35))
    flat = edge_map.flatten()
    idxs = np.arange(len(flat))
    mask = flat > 0.01
    non_zero = idxs[mask]
    probs = flat[mask]
    if len(non_zero) == 0:
        choice = np.random.choice(len(idxs), size=min(max_points, len(idxs)), replace=False)
    else:
        if probs.sum() > 0:
            probs = probs / probs.sum()
        sample = min(max_points, len(non_zero))
        if len(non_zero) <= sample:
            choice = non_zero.copy()
        else:
            choice = np.random.choice(non_zero, size=sample, replace=False, p=probs)
    xs = (choice % w).tolist()
    ys = (choice // w).tolist()
    return list(zip(xs, ys))


class Pentagon:
    """
    Represents a single geometric shape (pentagon) in the evolved image.
    
    Each pentagon has position, color, transparency, size, and rotation properties.
    Uses __slots__ for memory efficiency when handling thousands of instances.
    
    Attributes:
        x (int): X coordinate (0-511)
        y (int): Y coordinate (0-511)
        r (int): Red color component (0-255)
        g (int): Green color component (0-255)
        b (int): Blue color component (0-255)
        alpha (int): Transparency value (0-255)
        size (int): Size/radius of pentagon (1-300)
        rotation (float): Rotation angle in radians
        is_edge (bool): Flag indicating if placed on edge
    """
    __slots__ = ('x', 'y', 'r', 'g', 'b', 'alpha', 'size', 'rotation', 'is_edge')
    
    def __init__(self, x, y, color, alpha, size, rotation=None, is_edge=False):
        """
        Initialize pentagon with specified properties.
        
        Args:
            x (int): X coordinate
            y (int): Y coordinate
            color (Tuple[int, int, int]): RGB color tuple
            alpha (int): Transparency (0-255)
            size (int): Size/radius
            rotation (float, optional): Rotation angle. Random if None
            is_edge (bool): Whether placed on image edge
        """
        self.x = int(max(0, min(511, x)))
        self.y = int(max(0, min(511, y)))
        self.r, self.g, self.b = color
        self.alpha = int(max(0, min(255, alpha)))
        self.size = int(max(1, min(300, size)))
        self.rotation = rotation if rotation is not None else random.random()
        self.is_edge = bool(is_edge)
    
    def copy(self) -> 'Pentagon':
        """
        Create deep copy of pentagon.
        
        Returns:
            Pentagon: New pentagon instance with copied properties
        """
        return Pentagon(self.x, self.y, (self.r, self.g, self.b), self.alpha, self.size, self.rotation, self.is_edge)


class Individual:
    """
    Represents a single candidate solution in the evolutionary algorithm.
    
    An individual consists of a list of pentagons that, when rendered,
    form an image. Fitness is calculated by comparing rendered image
    to target image.
    
    Attributes:
        pentagons (List[Pentagon]): List of pentagons forming the image
        fitness (float): Calculated fitness score
        similarity_pct (float): Similarity percentage (0-100)
        valid (bool): Whether fitness has been calculated
        _cache (Dict[int, np.ndarray]): Cache of rendered images by size
    """
    
    def __init__(self, pents: List[Pentagon]):
        """
        Initialize individual with list of pentagons.
        
        Args:
            pents (List[Pentagon]): List of pentagon objects
        """
        self.pentagons = pents
        self.fitness = 0.0
        self.similarity_pct = 0.0
        self.valid = False
        self._cache = {}
    
    def render(self, size: int = 512) -> np.ndarray:
        """
        Render individual's pentagons into image array.
        
        Uses caching to avoid re-rendering at same resolution.
        
        Args:
            size (int): Output image size
        
        Returns:
            np.ndarray: Rendered image array (uint8)
        """
        if size in self._cache:
            return self._cache[size]
        arr = FastRenderer.render_fast(self.pentagons, size)
        self._cache[size] = arr
        return arr
    
    def calc_fitness(self, target, res, level: int = 0) -> float:
        """
        Calculate fitness score by comparing rendered image to target.
        
        Args:
            target (np.ndarray): Target image array
            res (int): Resolution for rendering
            level (int): Evolution level for metric selection
        
        Returns:
            float: Fitness score (higher is better)
        """
        if self.valid:
            return self.fitness
        rendered = self.render(res).astype(np.float32)
        self.fitness, self.similarity_pct, debug = calculate_fitness_metrics(rendered, target, level)
        self.valid = True
        self._debug = debug
        return self.fitness
    
    def copy(self) -> 'Individual':
        """
        Create deep copy of individual.
        
        Returns:
            Individual: New individual with copied pentagons
        """
        return Individual([p.copy() for p in self.pentagons])
    
    def invalidate(self):
        """
        Invalidate fitness cache, forcing recalculation on next fitness call.
        
        Used after mutations to ensure fitness reflects current state.
        """
        self.valid = False
        self._cache.clear()
        self._debug = None


def mutate_colors(pentagons: List[Pentagon], strength: float, num_mutations: int):
    """
    Mutate color values of selected pentagons using stochastic jitter.
    
    Uses unbiased random jitter around current color values. No direct
    pull toward target colors. Small probability of replacing with
    palette color for diversity.
    
    Args:
        pentagons (List[Pentagon]): List of pentagons to potentially mutate
        strength (float): Mutation strength (0.0-1.0), controls jitter amount
        num_mutations (int): Number of pentagons to mutate
    """
    global GLOBAL_PALETTE_COARSE
    if num_mutations <= 0 or not pentagons:
        return
    selected = random.sample(pentagons, min(num_mutations, len(pentagons)))
    for pent in selected:
        d = max(1, int(12 * strength))
        pent.r = int(np.clip(pent.r + random.randint(-d, d), 0, 255))
        pent.g = int(np.clip(pent.g + random.randint(-d, d), 0, 255))
        pent.b = int(np.clip(pent.b + random.randint(-d, d), 0, 255))
        if GLOBAL_PALETTE_COARSE and random.random() < 0.015 * (1.0 + strength):
            pal = GLOBAL_PALETTE_COARSE[random.randint(0, len(GLOBAL_PALETTE_COARSE)-1)]
            pent.r, pent.g, pent.b = pal


def mutate_positions(pentagons: List[Pentagon], strength: float, num_mutations: int):
    """
    Mutate position coordinates of selected pentagons.
    
    Edge-marked pentagons have lower mutation probability to preserve
    edge placement. Movement distance scales with strength.
    
    Args:
        pentagons (List[Pentagon]): List of pentagons to potentially mutate
        strength (float): Mutation strength (0.0-1.0), controls movement distance
        num_mutations (int): Number of pentagons to mutate
    """
    if num_mutations <= 0 or not pentagons:
        return
    selected = random.sample(pentagons, min(num_mutations, len(pentagons)))
    for pent in selected:
        if pent.is_edge and random.random() < 0.35:
            continue
        d = max(1, int(6*strength))
        pent.x = int(np.clip(pent.x + random.randint(-d, d), 0, 511))
        pent.y = int(np.clip(pent.y + random.randint(-d, d), 0, 511))


def mutate_sizes(pentagons: List[Pentagon], size_range: Tuple[int, int], num_mutations: int):
    """
    Mutate size values of selected pentagons.
    
    Edge-marked pentagons have lower mutation probability. Size changes
    by ±1 pixel, clamped to valid range.
    
    Args:
        pentagons (List[Pentagon]): List of pentagons to potentially mutate
        size_range (Tuple[int, int]): Valid size range (min, max)
        num_mutations (int): Number of pentagons to mutate
    """
    if num_mutations <= 0 or not pentagons:
        return
    selected = random.sample(pentagons, min(num_mutations, len(pentagons)))
    for pent in selected:
        if pent.is_edge and random.random() < 0.4:
            continue
        pent.size = int(np.clip(pent.size + random.randint(-1, 1), size_range[0], size_range[1]))


def mutate_alphas(pentagons: List[Pentagon], alpha_range: Tuple[int, int], strength: float, num_mutations: int):
    """
    Mutate transparency (alpha) values of selected pentagons.
    
    Args:
        pentagons (List[Pentagon]): List of pentagons to potentially mutate
        alpha_range (Tuple[int, int]): Valid alpha range (min, max)
        strength (float): Mutation strength (0.0-1.0), controls change amount
        num_mutations (int): Number of pentagons to mutate
    """
    if num_mutations <= 0 or not pentagons:
        return
    selected = random.sample(pentagons, min(num_mutations, len(pentagons)))
    for pent in selected:
        d = max(1, int(30*strength))
        pent.alpha = int(np.clip(pent.alpha + random.randint(-d, d), alpha_range[0], alpha_range[1]))


def make_coarse_layer(palette: List[Tuple[int, int, int]], target_arr: np.ndarray, 
                      grid: int = 10, alpha: int = 30) -> List[Pentagon]:
    """
    Create initial coarse layer of large pentagons using grid-based placement.
    
    Distributes pentagons across image in grid pattern with slight randomization.
    Colors sampled from target image with averaging, optionally from palette.
    
    Args:
        palette (List[Tuple[int, int, int]]): K-means color palette
        target_arr (np.ndarray): Target image for color sampling
        grid (int): Grid size (NxN grid)
        alpha (int): Base alpha value for pentagons
    
    Returns:
        List[Pentagon]: List of initialized pentagons for coarse layer
    """
    pts = []
    h, w = (target_arr.shape[:2] if target_arr is not None else (512, 512))
    step_x = max(8, w//grid)
    step_y = max(8, h//grid)
    sampler = ColorSampler(target_arr)
    for gy in range(grid):
        for gx in range(grid):
            cx = gx*step_x + step_x//2 + random.randint(-step_x//6, step_x//6)
            cy = gy*step_y + step_y//2 + random.randint(-step_y//6, step_y//6)
            cx = max(0, min(w-1, cx))
            cy = max(0, min(h-1, cy))
            c = sampler.sample(cx, cy, radius=max(1, min(step_x, step_y)//8))
            color = (palette[random.randint(0, len(palette)-1)] if palette and random.random() < 0.75 else c)
            size = int(max(12, min(step_x, step_y)//2 + random.randint(-step_x//8, step_x//8)))
            alpha_final = int(max(6, alpha + random.randint(-8, 8)))
            pts.append(Pentagon(cx, cy, color, alpha_final, size, None, False))
    return pts


def desaturate_color(color: Tuple[int, int, int], size: int, size_range: Tuple[int, int]) -> Tuple[int, int, int, float]:
    """
    Desaturate color based on pentagon size, creating size-dependent color variation.
    
    Larger pentagons become more desaturated (grayish), smaller ones retain
    original color. Also returns alpha factor for size-dependent transparency.
    
    Args:
        color (Tuple[int, int, int]): RGB color tuple
        size (int): Current pentagon size
        size_range (Tuple[int, int]): Valid size range (min, max)
    
    Returns:
        Tuple[int, int, int, float]:
            - Desaturated RGB color tuple
            - Alpha factor (0.0-1.0) for transparency adjustment
    """
    r, g, b = color
    lum = 0.299*r + 0.587*g + 0.114*b
    min_s, max_s = size_range
    rel = float(size - min_s) / float(max(1, max_s - min_s))
    rel = np.clip(rel, 0.0, 1.0)
    desat = SIZE_DESAT_FACTOR * rel
    r2 = int(np.clip((1.0-desat)*r + desat*lum, 0, 255))
    g2 = int(np.clip((1.0-desat)*g + desat*lum, 0, 255))
    b2 = int(np.clip((1.0-desat)*b + desat*lum, 0, 255))
    alpha_factor = 1.0 - 0.55 * rel
    return r2, g2, b2, alpha_factor


def final_detail_pass(pents: List[Pentagon], target_full_arr: np.ndarray, 
                      edge_points: List[Tuple[int, int]], num_small: int = 3500) -> List[Pentagon]:
    """
    Add final layer of small detail pentagons near edges and high-error regions.
    
    Creates many tiny pentagons (size 1-4) to capture fine details. Preferentially
    places them near detected edges. Uses ColorSampler for averaged color sampling.
    
    Args:
        pents (List[Pentagon]): Existing pentagons list
        target_full_arr (np.ndarray): Full resolution target image
        edge_points (List[Tuple[int, int]]): List of edge coordinates
        num_small (int): Number of small pentagons to add
    
    Returns:
        List[Pentagon]: Extended list with new detail pentagons appended
    """
    sampler = ColorSampler(target_full_arr)
    added = []
    ep = edge_points if edge_points else []
    for i in range(num_small):
        if ep and random.random() < 0.65:
            x, y = random.choice(ep)
            x = int(np.clip(x + random.randint(-4, 4), 0, 511))
            y = int(np.clip(y + random.randint(-4, 4), 0, 511))
        else:
            x = random.randint(0, 511)
            y = random.randint(0, 511)
        base = sampler.sample(x, y, radius=1)
        size = random.randint(1, 2) if random.random() < 0.85 else random.randint(2, 4)
        r, g, b, af = desaturate_color(base, size, (1, 4))
        alpha = int(np.clip(random.randint(100, 230) * af, 8, 255))
        p = Pentagon(x, y, (r, g, b), alpha, size, None, (x, y) in ep)
        added.append(p)
    pents.extend(added)
    return pents


def local_hillclimb(ind: Individual, target: np.ndarray, res: int, iters: int = 80) -> Individual:
    """
    Perform local hill-climbing optimization on individual.
    
    Iteratively applies small mutations and keeps improvements. Helps refine
    details after main evolution phases.
    
    Args:
        ind (Individual): Individual to optimize
        target (np.ndarray): Target image array
        res (int): Resolution for fitness evaluation
        iters (int): Number of hill-climbing iterations
    
    Returns:
        Individual: Best individual found during hill-climbing
    """
    best = ind
    best_f = best.fitness
    for _ in range(iters):
        cand = best.copy()
        mutate_colors(cand.pentagons, strength=0.35, num_mutations=max(1, len(cand.pentagons)//40))
        mutate_positions(cand.pentagons, strength=0.35, num_mutations=max(1, len(cand.pentagons)//60))
        mutate_alphas(cand.pentagons, (0, 255), 0.25, num_mutations=max(1, len(cand.pentagons)//80))
        cand.invalidate()
        f = cand.calc_fitness(target, res)
        if f > best_f:
            best = cand
            best_f = f
    return best


def get_top_error_boxes(rendered: np.ndarray, target: np.ndarray, k: int = 6, box: int = 64) -> List[Tuple[int, int]]:
    """
    Identify regions with highest reconstruction error.
    
    Samples random boxes, computes error sum, returns coordinates of
    top-k highest error regions. Used for targeted detail placement.
    
    Args:
        rendered (np.ndarray): Rendered image array
        target (np.ndarray): Target image array
        k (int): Number of top error regions to return
        box (int): Box size for error sampling
    
    Returns:
        List[Tuple[int, int]]: List of (center_x, center_y) coordinates
    """
    err = np.mean((rendered.astype(np.float32)-target.astype(np.float32))**2, axis=2)
    h, w = err.shape
    samples = []
    for _ in range(700):
        cx = random.randint(0, w-1)
        cy = random.randint(0, h-1)
        x0 = max(0, cx-box//2)
        x1 = min(w, cx+box//2)
        y0 = max(0, cy-box//2)
        y1 = min(h, cy+box//2)
        sc = err[y0:y1, x0:x1].sum()
        samples.append((sc, (cx, cy)))
    samples.sort(reverse=True)
    return [b for _, b in samples[:k]]


def kmeans_palette(img_arr: np.ndarray, k: int = 18, iters: int = 8) -> List[Tuple[int, int, int]]:
    """
    Extract color palette from image using K-means clustering.
    
    Samples pixels, clusters them into k colors using K-means algorithm.
    Returns representative color palette for coarse layer initialization.
    
    Args:
        img_arr (np.ndarray): Input image array (H x W x 3)
        k (int): Number of palette colors
        iters (int): K-means iteration count
    
    Returns:
        List[Tuple[int, int, int]]: List of RGB color tuples
    """
    pts = img_arr.reshape(-1, 3).astype(np.float32)
    n = pts.shape[0]
    sample_n = min(25000, n)
    idx = np.random.choice(n, sample_n, replace=False)
    samp = pts[idx]
    centers = samp[np.random.choice(sample_n, k, replace=False)]
    for _ in range(iters):
        dists = np.sum((samp[:, None, :] - centers[None, :, :])**2, axis=2)
        nearest = np.argmin(dists, axis=1)
        for j in range(k):
            sel = samp[nearest == j]
            if len(sel) > 0:
                centers[j] = sel.mean(axis=0)
    centers = np.clip(centers, 0, 255).astype(np.uint8)
    return [tuple(map(int, c)) for c in centers]


def evolve_single_image(args: Tuple[int, int, int]) -> Dict:
    """
    Execute single independent evolution run for one image.
    
    Performs multi-level evolution: starts with coarse layer, then progressively
    adds detail through 6 levels. Uses error-based placement to focus on
    high-error regions. Applies local hill-climbing for refinement.
    
    Args:
        args (Tuple[int, int, int]): (evolution_id, run_num, seed)
            - evolution_id: Unique identifier for this evolution
            - run_num: Run number (1-3)
            - seed: Random seed for reproducibility
    
    Returns:
        Dict: Result dictionary containing:
            - 'evolution_id': int
            - 'fitness': float
            - 'similarity': float (0-100)
            - 'image_array': np.ndarray (512x512x3)
            - 'pentagons': int (count)
            - 'debug': Dict with metric details
    """
    evolution_id, run_num, seed = args
    random.seed(seed)
    np.random.seed(seed ^ 0x5a5a5a5a)
    target_full = GLOBAL_TARGET_FULL
    targets = GLOBAL_TARGETS
    edge_points = GLOBAL_EDGE_POINTS
    palette_coarse = GLOBAL_PALETTE_COARSE

    all_pents = []
    try:
        coarse = make_coarse_layer(palette_coarse, target_full, grid=9, alpha=28)
        all_pents = [p.copy() for p in coarse]
    except Exception:
        all_pents = []

    # Track fitness data for plotting (global generation counter across all levels)
    generation_data = []  # List of (global_gen, avg_fitness, max_fitness)
    global_generation = 0

    for level_idx, (num, size_range, max_gen, alpha_range, edge_bias) in enumerate(Config.LEVELS):
        n_edge = int(num * edge_bias)
        n_random = num - n_edge
        if n_edge > 0 and edge_points:
            edge_positions = random.sample(edge_points, min(n_edge, len(edge_points)))
        else:
            edge_positions = []

        error_positions = []
        if len(all_pents) > 0:
            try:
                temp_ind = Individual(all_pents)
                rend = temp_ind.render(128).astype(np.float32)
                tgt128 = GLOBAL_TARGETS.get(128, np.array(Image.fromarray(GLOBAL_TARGET_FULL.astype(np.uint8)).resize((128, 128))))
                err = np.mean((rend - tgt128) ** 2, axis=2)
                err_img = Image.fromarray(np.uint8(np.clip(err/(err.max() if err.max()>0 else 1.0)*255.0, 0, 255)))
                err_img = err_img.filter(ImageFilter.GaussianBlur(radius=5))
                arr_err = np.array(err_img).astype(np.float32)
                if arr_err.sum() > 0:
                    probs = arr_err.flatten()
                    probs = probs / probs.sum()
                    flat_idx = np.arange(probs.size)
                    sample_count = min(500, len(flat_idx))
                    pick = np.random.choice(flat_idx, size=sample_count, replace=False, p=probs)
                    h128, w128 = arr_err.shape
                    xs = (pick % w128) * (512//w128) + (512//w128)//2
                    ys = (pick // w128) * (512//h128) + (512//h128)//2
                    error_positions = list(zip(xs.tolist(), ys.tolist()))
            except Exception:
                error_positions = []

        positions = []
        n_from_edge = int(n_random * 0.50)
        n_from_error = int(n_random * 0.34)
        n_from_rand = n_random - n_from_edge - n_from_error
        if edge_positions:
            positions += random.sample(edge_positions, min(n_from_edge, len(edge_positions)))
        if error_positions:
            positions += random.sample(error_positions, min(n_from_error, len(error_positions)))
        positions += [(random.randint(0, 511), random.randint(0, 511)) for _ in range(n_from_rand)]
        while len(positions) < n_random:
            positions.append((random.randint(0, 511), random.randint(0, 511)))
        positions = (edge_positions + positions)[:num]

        if level_idx <= 1:
            color_sampler = ColorSampler(GLOBAL_TARGET_FULL)
            sample_radius = 6
        elif level_idx <= 3:
            color_sampler = ColorSampler(GLOBAL_TARGET_FULL)
            sample_radius = 3
        else:
            color_sampler = ColorSampler(GLOBAL_TARGET_FULL)
            sample_radius = 1

        new_pents = []
        for (x, y) in positions:
            base_color = color_sampler.sample(x, y, radius=sample_radius)
            size = random.randint(size_range[0], size_range[1])
            r2, g2, b2, af = desaturate_color(base_color, size, size_range)
            alpha_base = random.randint(alpha_range[0], alpha_range[1])
            alpha_final = int(np.clip(alpha_base * af, 6, 255))
            is_edge = (x, y) in edge_positions
            new_pents.append(Pentagon(x, y, (r2, g2, b2), alpha_final, size, None, is_edge))

        population = []
        for _ in range(Config.POPULATION_SIZE):
            all_copy = [p.copy() for p in all_pents] + [p.copy() for p in new_pents]
            population.append(Individual(all_copy))

        eval_res = EVAL_RESOLUTIONS[min(level_idx, len(EVAL_RESOLUTIONS)-1)]
        target = targets[eval_res]

        for ind in population:
            ind.calc_fitness(target, eval_res, level_idx)

        best = max(population, key=lambda x: x.fitness).copy()

        for gen in range(max_gen):
            population.sort(key=lambda x: x.fitness, reverse=True)
            if population[0].fitness > best.fitness:
                best = population[0].copy()
            
            # Collect fitness data for plotting
            avg_fitness = sum(ind.fitness for ind in population) / len(population)
            max_fitness = population[0].fitness
            generation_data.append((global_generation, avg_fitness, max_fitness))
            global_generation += 1
            
            progress = gen / max(1, max_gen)
            mut_rate = Config.MUTATION_RATE_END + (Config.MUTATION_RATE_START - Config.MUTATION_RATE_END) * math.exp(-3.0 * progress)
            strength = 0.95 * math.exp(-3.0 * progress) + 0.05
            new_pop = [population[i].copy() for i in range(Config.ELITE_SIZE)]
            while len(new_pop) < Config.POPULATION_SIZE:
                t1 = random.sample(population, Config.TOURNAMENT_SIZE)
                p1 = max(t1, key=lambda x: x.fitness)
                t2 = random.sample(population, Config.TOURNAMENT_SIZE)
                p2 = max(t2, key=lambda x: x.fitness)
                if random.random() < Config.CROSSOVER_RATE:
                    pt = random.randint(0, len(p1.pentagons))
                    child = Individual([p.copy() for p in p1.pentagons[:pt]] + [p.copy() for p in p2.pentagons[pt:]])
                else:
                    child = p1.copy()
                if random.random() < mut_rate:
                    total = len(child.pentagons)
                    mutate_colors(child.pentagons, strength, max(1, total//6))
                    mutate_positions(child.pentagons, strength, max(1, total//6))
                    mutate_sizes(child.pentagons, size_range, max(1, total//8))
                    mutate_alphas(child.pentagons, alpha_range, strength, max(1, total//6))
                child.invalidate()
                new_pop.append(child)
            population = new_pop[:Config.POPULATION_SIZE]
            for ind in population:
                if not ind.valid:
                    ind.calc_fitness(target, eval_res, level_idx)

        try:
            best = local_hillclimb(best, target, eval_res, iters=40)
        except Exception:
            pass

        all_pents = [p.copy() for p in best.pentagons]

    try:
        all_pents = final_detail_pass(all_pents, GLOBAL_TARGET_FULL, GLOBAL_EDGE_POINTS, num_small=3000)
        tmp_ind = Individual(all_pents)
        tmp_ind.invalidate()
        tmp_ind.calc_fitness(GLOBAL_TARGET_FULL, 512, len(Config.LEVELS)-1)
        boxes = get_top_error_boxes(tmp_ind.render(256).astype(np.float32),
                                    np.array(Image.fromarray(GLOBAL_TARGET_FULL.astype(np.uint8)).resize((256, 256))),
                                    k=8, box=56)
        boxes512 = [(int(cx*512/256), int(cy*512/256)) for (cx, cy) in boxes]
        final_ind = local_hillclimb(tmp_ind, GLOBAL_TARGET_FULL, 256, iters=120)
    except Exception:
        final_ind = Individual(all_pents)
        final_ind.invalidate()

    final_ind.invalidate()
    final_ind.calc_fitness(GLOBAL_TARGET_FULL, 512, len(Config.LEVELS)-1)
    final_arr = final_ind.render(512)

    return {'evolution_id': evolution_id, 'fitness': final_ind.fitness, 'similarity': final_ind.similarity_pct,
            'image_array': final_arr, 'pentagons': len(final_ind.pentagons), 'debug': getattr(final_ind, '_debug', None),
            'generation_data': generation_data}


def create_arcade_machine_with_screen(generated_image: np.ndarray) -> Image.Image:
    """
    Create 3D-styled arcade machine graphic with embedded generated image.
    
    Generates a retro arcade cabinet visualization with 3D effects, including:
    - Recessed screen area with bezel and shadows
    - Control panel with 3D joystick and buttons
    - Coin slot with watermelon slice decoration
    - Gradient effects and lighting for depth
    
    Args:
        generated_image (np.ndarray): Generated image array (512x512x3, uint8)
            to be displayed in the arcade machine screen
    
    Returns:
        Image.Image: Complete arcade machine image (512x512, RGB)
    """
    img = Image.new('RGBA', (512, 512), color=(15, 15, 20, 255))
    draw = ImageDraw.Draw(img)
    
    BLACK = (10, 10, 15)
    DARK_BLACK = (5, 5, 8)
    RED = (220, 20, 30)
    DARK_RED = (150, 10, 20)
    BRIGHT_RED = (255, 40, 50)
    WHITE = (255, 255, 255)
    GRAY = (40, 40, 45)
    LIGHT_GRAY = (80, 80, 85)
    DARK_GRAY = (25, 25, 30)
    
    WATERMELON_GREEN = (60, 150, 60)
    WATERMELON_RED = (220, 50, 50)
    WATERMELON_DARK_GREEN = (30, 100, 30)
    
    def draw_gradient_rect(x1, y1, x2, y2, color1, color2, vertical=True):
        """Draw gradient rectangle with smooth color transition."""
        for i in range(y1, y2 + 1) if vertical else range(x1, x2 + 1):
            if vertical:
                t = (i - y1) / max(1, y2 - y1)
                r = int(color1[0] * (1-t) + color2[0] * t)
                g = int(color1[1] * (1-t) + color2[1] * t)
                b = int(color1[2] * (1-t) + color2[2] * t)
                draw.rectangle([x1, i, x2, i], fill=(r, g, b))
            else:
                t = (i - x1) / max(1, x2 - x1)
                r = int(color1[0] * (1-t) + color2[0] * t)
                g = int(color1[1] * (1-t) + color2[1] * t)
                b = int(color1[2] * (1-t) + color2[2] * t)
                draw.rectangle([i, y1, i, y2], fill=(r, g, b))
    
    def draw_rounded_rect(x, y, w, h, radius, fill_color, outline=None, width=1):
        """Draw rounded rectangle with fallback to regular rectangle if not supported."""
        try:
            draw.rounded_rectangle([x, y, x + w, y + h], radius=radius, fill=fill_color, outline=outline, width=width)
        except:
            draw.rectangle([x, y, x + w, y + h], fill=fill_color, outline=outline, width=width)
    
    top_strip_h = 15
    draw_gradient_rect(0, 0, 512, top_strip_h, DARK_RED, RED, vertical=True)
    draw.rectangle([0, 0, 512, 3], fill=BRIGHT_RED)
    
    title_y = top_strip_h + 2
    title_h = 38
    
    draw.rectangle([2, title_y + 2, 510, title_y + title_h + 2], fill=DARK_BLACK)
    draw.rectangle([0, title_y, 512, title_y + title_h], fill=BLACK)
    draw_gradient_rect(0, title_y, 512, title_y + title_h, DARK_BLACK, BLACK, vertical=True)
    
    try:
        font = ImageFont.truetype("arial.ttf", 20) if os.path.exists("arial.ttf") else ImageFont.load_default()
        small_font = ImageFont.truetype("arial.ttf", 14) if os.path.exists("arial.ttf") else ImageFont.load_default()
    except:
        font = ImageFont.load_default()
        small_font = ImageFont.load_default()
    
    draw.text((256, title_y + title_h // 2 - 6), "RETRO ARCADE MACHINE", fill=WHITE, anchor="mm", font=font)
    draw.text((256, title_y + title_h // 2 + 8), "WATERMELON ARCADE", fill=WHITE, anchor="mm", font=small_font)
    
    screen_x, screen_y = 50, 70
    screen_w, screen_h = 412, 300
    
    bezel_thickness = 12
    bezel_outer = bezel_thickness + 4
    
    for i in range(3):
        offset = i
        shadow_alpha = max(0, 40 - i * 10)
        shadow_color = (max(0, DARK_BLACK[0] - shadow_alpha),
                       max(0, DARK_BLACK[1] - shadow_alpha),
                       max(0, DARK_BLACK[2] - shadow_alpha))
        draw_rounded_rect(screen_x - bezel_outer + offset, screen_y - bezel_outer + offset,
                         screen_w + 2 * (bezel_outer - offset),
                         screen_h + 2 * (bezel_outer - offset),
                         radius=8, fill_color=shadow_color, outline=BLACK, width=2)
    
    draw_rounded_rect(screen_x - bezel_thickness, screen_y - bezel_thickness,
                     screen_w + 2 * bezel_thickness, screen_h + 2 * bezel_thickness,
                     radius=6, fill_color=DARK_BLACK, outline=BLACK, width=2)
    
    inner_bezel = 6
    draw_rounded_rect(screen_x - inner_bezel, screen_y - inner_bezel,
                     screen_w + 2 * inner_bezel, screen_h + 2 * inner_bezel,
                     radius=4, fill_color=BLACK, outline=DARK_GRAY, width=1)
    
    screen_frame = 3
    draw_gradient_rect(screen_x - screen_frame, screen_y - screen_frame,
                      screen_x + screen_w + screen_frame, screen_y + screen_h + screen_frame,
                      DARK_GRAY, GRAY, vertical=False)
    
    draw_rounded_rect(screen_x, screen_y, screen_w, screen_h,
                     radius=2, fill_color=(5, 5, 8))
    
    gen_img = Image.fromarray(generated_image.astype(np.uint8), 'RGB')
    gen_img = gen_img.convert('RGBA')
    screen_img = gen_img.resize((screen_w, screen_h), Image.Resampling.LANCZOS)
    
    screen_array = np.array(screen_img, dtype=np.float32)
    screen_array[:, :, :3] = screen_array[:, :, :3] * 0.85
    screen_array = np.clip(screen_array, 0, 255).astype(np.uint8)
    screen_img = Image.fromarray(screen_array, 'RGBA')
    
    img.paste(screen_img, (screen_x, screen_y), screen_img)
    
    glare_overlay = Image.new('RGBA', (screen_w, screen_h // 3), (255, 255, 255, 0))
    glare_draw = ImageDraw.Draw(glare_overlay)
    for i in range(screen_h // 3):
        alpha = int(15 * (1 - i / (screen_h // 3)))
        glare_draw.rectangle([0, i, screen_w, i + 1], fill=(255, 255, 255, alpha))
    img.paste(glare_overlay, (screen_x, screen_y), glare_overlay)
    
    panel_y = screen_y + screen_h + 20
    panel_h = 90
    
    draw.rectangle([3, panel_y + 3, 509, panel_y + panel_h + 3], fill=DARK_BLACK)
    draw_gradient_rect(0, panel_y, 512, panel_y + panel_h, BLACK, DARK_BLACK, vertical=True)
    draw.rectangle([0, panel_y, 512, panel_y + 2], fill=LIGHT_GRAY)
    
    joy_x, joy_y = 80, panel_y + panel_h // 2
    stick_height = 20
    ball_radius = 14
    
    base_width = 24
    base_height = 8
    base_y = joy_y + stick_height // 2
    
    shadow_color = (max(0, BLACK[0] - 60), max(0, BLACK[1] - 60), max(0, BLACK[2] - 60))
    draw.ellipse([joy_x - base_width // 2 + 1, base_y + 1,
                 joy_x + base_width // 2 + 1, base_y + base_height + 1],
                fill=shadow_color)
    
    draw.ellipse([joy_x - base_width // 2, base_y,
                 joy_x + base_width // 2, base_y + base_height],
                fill=DARK_RED, outline=BLACK, width=2)
    
    draw.ellipse([joy_x - base_width // 2 + 2, base_y + 1,
                 joy_x + base_width // 2 - 2, base_y + 3],
                fill=RED)
    
    stick_width = 7
    stick_top_y = joy_y - stick_height // 2
    stick_bottom_y = base_y
    
    draw.rectangle([joy_x - stick_width // 2 + 1, stick_top_y + 1,
                   joy_x + stick_width // 2 + 1, stick_bottom_y + 1],
                  fill=shadow_color)
    
    for i in range(stick_width):
        intensity = int(50 * (1 - abs(i - stick_width/2) / (stick_width/2)))
        stick_color = (max(0, DARK_RED[0] - intensity),
                      max(0, DARK_RED[1] - intensity),
                      max(0, DARK_RED[2] - intensity))
        draw.rectangle([joy_x - stick_width // 2 + i, stick_top_y,
                       joy_x - stick_width // 2 + i + 1, stick_bottom_y],
                      fill=stick_color)
    
    draw.rectangle([joy_x - stick_width // 2, stick_top_y,
                   joy_x - stick_width // 2 + 2, stick_bottom_y],
                  fill=RED)
    
    ball_y = stick_top_y - ball_radius
    
    draw.ellipse([joy_x - ball_radius + 2, ball_y + 2,
                 joy_x + ball_radius + 2, ball_y + ball_radius * 2 + 2],
                fill=shadow_color)
    
    draw.ellipse([joy_x - ball_radius, ball_y,
                 joy_x + ball_radius, ball_y + ball_radius * 2],
                fill=DARK_RED, outline=BLACK, width=1)
    
    draw.ellipse([joy_x - ball_radius + 3, ball_y + 3,
                 joy_x + ball_radius - 3, ball_y + ball_radius * 2 - 3],
                fill=RED)
    
    highlight_size = 7
    draw.ellipse([joy_x - ball_radius // 2, ball_y + 3,
                 joy_x - ball_radius // 2 + highlight_size, ball_y + 3 + highlight_size],
                fill=BRIGHT_RED)
    
    button_y = panel_y + 22
    
    for i in range(3):
        btn_x = 175 + i * 38
        shadow_color = (max(0, BLACK[0] - 30), max(0, BLACK[1] - 30), max(0, BLACK[2] - 30))
        draw.ellipse([btn_x - 8 + 1, button_y - 8 + 1, btn_x + 8 + 1, button_y + 8 + 1],
                    fill=shadow_color)
        draw.ellipse([btn_x - 8, button_y - 8, btn_x + 8, button_y + 8],
                    fill=BLACK, outline=GRAY, width=1)
        draw.ellipse([btn_x - 5, button_y - 5, btn_x - 2, button_y - 2],
                    fill=LIGHT_GRAY)
    
    for i in range(2):
        btn_x = 295 + i * 55
        shadow_color = (max(0, BLACK[0] - 60), max(0, BLACK[1] - 60), max(0, BLACK[2] - 60))
        draw.ellipse([btn_x - 13 + 2, button_y - 13 + 2, btn_x + 13 + 2, button_y + 13 + 2],
                    fill=shadow_color)
        draw.ellipse([btn_x - 13, button_y - 13, btn_x + 13, button_y + 13],
                    fill=DARK_RED, outline=BLACK, width=2)
        draw.ellipse([btn_x - 10, button_y - 10, btn_x + 10, button_y + 10],
                    fill=RED)
        draw.ellipse([btn_x - 6, button_y - 6, btn_x - 2, button_y - 2],
                    fill=BRIGHT_RED)
    
    button_y2 = panel_y + 52
    for i in range(2):
        btn_x = 295 + i * 55
        shadow_color = (max(0, BLACK[0] - 60), max(0, BLACK[1] - 60), max(0, BLACK[2] - 60))
        draw.ellipse([btn_x - 13 + 2, button_y2 - 13 + 2, btn_x + 13 + 2, button_y2 + 13 + 2],
                    fill=shadow_color)
        draw.ellipse([btn_x - 13, button_y2 - 13, btn_x + 13, button_y2 + 13],
                    fill=DARK_RED, outline=BLACK, width=2)
        draw.ellipse([btn_x - 10, button_y2 - 10, btn_x + 10, button_y2 + 10],
                    fill=RED)
        draw.ellipse([btn_x - 6, button_y2 - 6, btn_x - 2, button_y2 - 2],
                    fill=BRIGHT_RED)
    
    coin_slot_x = 430
    coin_text_y = panel_y + 8
    try:
        coin_font = ImageFont.truetype("arial.ttf", 13) if os.path.exists("arial.ttf") else ImageFont.load_default()
    except:
        coin_font = ImageFont.load_default()
    
    text_bg_x = coin_slot_x - 45
    text_bg_y = coin_text_y - 7
    text_bg_w = 90
    text_bg_h = 14
    draw_rounded_rect(text_bg_x, text_bg_y, text_bg_w, text_bg_h,
                     radius=3, fill_color=(30, 30, 35), outline=DARK_GRAY, width=1)
    
    text_shadow_coin = (0, 0, 0)
    for offset_x in [-1, 0, 1]:
        for offset_y in [-1, 0, 1]:
            if offset_x != 0 or offset_y != 0:
                draw.text((coin_slot_x + offset_x, coin_text_y + offset_y),
                         "1 PLAY = 1 ARBUZ", fill=text_shadow_coin, anchor="mm", font=coin_font)
    
    draw.text((coin_slot_x, coin_text_y),
             "1 PLAY = 1 ARBUZ", fill=(255, 255, 255), anchor="mm", font=coin_font)
    
    coin_slot_y = coin_text_y + 18
    coin_slot_w = 8
    coin_slot_h = 25
    
    draw_rounded_rect(coin_slot_x + 1, coin_slot_y + 1, coin_slot_w, coin_slot_h,
                     radius=4, fill_color=DARK_BLACK)
    
    draw_rounded_rect(coin_slot_x, coin_slot_y, coin_slot_w, coin_slot_h,
                     radius=4, fill_color=(0, 0, 0), outline=DARK_GRAY, width=1)
    
    draw_rounded_rect(coin_slot_x, coin_slot_y, coin_slot_w, 2,
                     radius=2, fill_color=LIGHT_GRAY)
    
    slice_y = coin_slot_y + coin_slot_h + 10
    slice_width = 42
    slice_height = 35
    slice_x = coin_slot_x + coin_slot_w // 2 - slice_width // 2
    
    rind_center_x = slice_x + slice_width // 2
    rind_thickness = 6
    
    flesh_bottom_y = slice_y + slice_height - rind_thickness
    flesh_points = [
        (rind_center_x, slice_y),
        (slice_x + 6, flesh_bottom_y - 1),
        (slice_x + slice_width - 6, flesh_bottom_y - 1),
    ]
    draw.polygon(flesh_points, fill=(240, 60, 60), outline=(240, 60, 60))
    
    highlight_points = [
        (rind_center_x - 4, slice_y + 4),
        (rind_center_x + 4, slice_y + 4),
        (rind_center_x, slice_y + 10),
    ]
    draw.polygon(highlight_points, fill=(255, 140, 140))
    
    seed_positions_slice = [
        (rind_center_x - 7, slice_y + 7),
        (rind_center_x + 7, slice_y + 8),
        (rind_center_x - 3, slice_y + 13),
        (rind_center_x + 5, slice_y + 15),
        (rind_center_x - 5, slice_y + 20),
        (rind_center_x + 2, slice_y + 22),
    ]
    for sx, sy in seed_positions_slice:
        draw.ellipse([sx - 2 + 1, sy - 1 + 1, sx + 2 + 1, sy + 1 + 1], fill=(20, 20, 20))
        draw.ellipse([sx - 2, sy - 1, sx + 2, sy + 1], fill=(0, 0, 0))
    
    inner_rind_y = flesh_bottom_y
    inner_rind_points = [
        (rind_center_x, inner_rind_y),
        (slice_x + 5, slice_y + slice_height - 3),
        (slice_x + slice_width - 5, slice_y + slice_height - 3),
    ]
    draw.polygon(inner_rind_points, fill=(255, 255, 240), outline=(250, 250, 230))
    
    for i in range(rind_thickness):
        y_pos = slice_y + slice_height - rind_thickness + i
        progress = i / rind_thickness
        width_at_y = int(slice_width * (0.7 + 0.3 * progress))
        x_start = rind_center_x - width_at_y // 2
        x_end = rind_center_x + width_at_y // 2
        arc_height = max(2, int(width_at_y * 0.15))
        draw.ellipse([x_start, y_pos - arc_height // 2, x_end, y_pos + arc_height // 2],
                    fill=WATERMELON_GREEN, outline=WATERMELON_DARK_GREEN, width=1)
    
    bottom_y = panel_y + panel_h + 15
    bottom_h = 512 - bottom_y - 15
    
    draw.rectangle([2, bottom_y + 2, 510, 512 - 15 + 2], fill=DARK_BLACK)
    draw_gradient_rect(0, bottom_y, 512, 512 - 15, BLACK, DARK_BLACK, vertical=True)
    
    bottom_strip_y = 512 - 15
    draw_gradient_rect(0, bottom_strip_y, 512, 512, DARK_RED, RED, vertical=True)
    draw.rectangle([0, bottom_strip_y, 512, bottom_strip_y + 2], fill=BRIGHT_RED)
    
    for x in range(512):
        if x < 20:
            shadow_intensity = int(30 * (1 - x / 20))
            shadow_color = (max(0, BLACK[0] - shadow_intensity),
                          max(0, BLACK[1] - shadow_intensity),
                          max(0, BLACK[2] - shadow_intensity))
            draw.rectangle([x, 0, x, 512], fill=shadow_color)
        elif x > 492:
            shadow_intensity = int(30 * ((x - 492) / 20))
            shadow_color = (max(0, BLACK[0] - shadow_intensity),
                          max(0, BLACK[1] - shadow_intensity),
                          max(0, BLACK[2] - shadow_intensity))
            draw.rectangle([x, 0, x, 512], fill=shadow_color)
    
    img = img.convert('RGB')
    return img



# GRAPH PLOTTING FUNCTION 

def plot_fitness_graphs(runs_data: Dict[int, List[Tuple[int, float, float]]], output_dir: str, input_image: int):
    """
    Plot fitness graphs for all runs.
    
    Args:
        runs_data: Dictionary mapping run number (1-3) to list of (generation, avg_fitness, max_fitness) tuples
        output_dir: Directory to save graphs
        input_image: Input image index
    """
    if not _PLOTTING_AVAILABLE:
        print("Warning: matplotlib not available, skipping graph generation")
        return
    
    if not runs_data:
        return
    
    # Create figure with 3 subplots (one for each run)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'Fitness Evolution - Input {input_image}', fontsize=16, fontweight='bold')
    
    for run_num in [1, 2, 3]:
        if run_num not in runs_data or not runs_data[run_num]:
            continue
        
        ax = axes[run_num - 1]
        data = runs_data[run_num]
        
        # Extract data
        generations = [d[0] for d in data]
        avg_fitness = [d[1] for d in data]
        max_fitness = [d[2] for d in data]
        
        # Convert fitness to negative L1 loss (for compatibility with example graph style)
        # Fitness is positive and higher is better, so we invert and scale to -50 to -25 range
        if avg_fitness and max_fitness:
            # Find min and max across all fitness values
            all_fitness = avg_fitness + max_fitness
            min_fit = min(all_fitness)
            max_fit = max(all_fitness)
            
            if max_fit > min_fit:
                # Map fitness range to -50 to -25 (inverted: higher fitness -> less negative loss)
                # Higher fitness should map to -25, lower fitness to -50
                range_size = max_fit - min_fit
                neg_avg_fitness = [-50.0 + 25.0 * (1.0 - (f - min_fit) / range_size) for f in avg_fitness]
                neg_max_fitness = [-50.0 + 25.0 * (1.0 - (f - min_fit) / range_size) for f in max_fitness]
            else:
                # All fitness values are the same
                neg_avg_fitness = [-37.5] * len(avg_fitness)
                neg_max_fitness = [-37.5] * len(max_fitness)
        else:
            neg_avg_fitness = []
            neg_max_fitness = []
        
        # Plot lines
        ax.plot(generations, neg_avg_fitness, '--', color='lightblue', linewidth=1.5, label='Avg Fitness', alpha=0.8)
        ax.plot(generations, neg_max_fitness, '-', color='red', linewidth=2, label='Max Fitness (Elite)', alpha=0.9)
        
        # Styling
        ax.set_title(f'Run {run_num}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Global Generations', fontsize=11)
        ax.set_ylabel('Negative L1 Loss', fontsize=11)
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_facecolor('#f8f8f8')
        
        # Set axis limits similar to example
        if generations:
            ax.set_xlim(0, max(generations) if max(generations) > 0 else 6000)
        else:
            ax.set_xlim(0, 6000)
        ax.set_ylim(-50, -25)
        
        # Add legend only for first plot
        if run_num == 1:
            ax.legend(loc='lower center', fontsize=9, framealpha=0.9)
    
    plt.tight_layout()
    
    # Save graph
    graph_filename = os.path.join(output_dir, f'fitness_graphs_input{input_image}.png')
    plt.savefig(graph_filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 Fitness graphs saved: {graph_filename}")



def main_with_arcade():
    """
    Main execution function with arcade machine visualization.
    
    Orchestrates the complete evolution process:
    1. Loads and preprocesses target image
    2. Extracts edges and generates color palette
    3. Runs multiple parallel evolutions (5 per run, 3 runs total)
    4. Creates arcade machine visualization for best result
    5. Saves all outputs to OUTPUT_DIR
    
    Outputs:
        - Evolution results: TelmanNuruzovOutput{X}_{Y}_{Z}.jpg
        where X is the index of the image, Y is the number of run for image from 1 to 3, Z is the value from
        1 to 5, indicating the index among the best results per run. 
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    input_file = f"input{INPUT_IMAGE}.jpg"
    if not os.path.exists(input_file):
        print("Input not found:", input_file)
        return
    img = Image.open(input_file).convert('RGB')
    img = img.resize((512, 512), Image.Resampling.LANCZOS)
    target_full = np.array(img, dtype=np.float32)
    targets = {}
    for r in EVAL_RESOLUTIONS:
        targets[int(r)] = np.array(Image.fromarray(target_full.astype(np.uint8)).resize((r, r), Image.Resampling.LANCZOS), dtype=np.float32)
    edge_map = detect_edges_fast(target_full)
    edge_points = sample_edge_points(edge_map, max_points=min(22000, int(target_full.shape[0]*target_full.shape[1]*0.25)))
    PALETTE_COARSE = kmeans_palette(target_full, k=20, iters=9)

    num_cores = mp.cpu_count()
    max_parallel = max(1, int(num_cores * 0.7))
    MAX_PARALLEL_EVOLUTIONS = min(max_parallel, MAX_PARALLEL_LIMIT)

    print("="*70)
    print("Using parallel evolutions:", MAX_PARALLEL_EVOLUTIONS)
    print("SSIM available:", _SSIM_AVAILABLE)
    print("="*70)
    total_start = time.time()
    results_summary = []
    
    
    # GRAPH DATA COLLECTION 
    
    runs_fitness_data = {}  # Dictionary: run_num -> list of (generation, avg_fitness, max_fitness)
    

    with ProcessPoolExecutor(max_workers=MAX_PARALLEL_EVOLUTIONS, initializer=worker_init,
                             initargs=(target_full, targets, edge_points, PALETTE_COARSE)) as executor:
        for run in range(1, 4):
            print(f"\n=== RUN {run}/3 ===")
            run_start = time.time()
            run_time_limit = RUN_TIME_LIMIT_MINUTES * 60  # Convert to seconds
            tasks = []
            base_seed = run*999999 + INPUT_IMAGE*100000 + int(time.time()*1000)%100000
            for evo_id in range(1, 6):
                tasks.append((evo_id, run, base_seed + evo_id*123457))
            futures = {executor.submit(evolve_single_image, t): t[0] for t in tasks}
            evo_results = []
            time_limit_reached = False
            
            for future in as_completed(futures):
                # Check if time limit exceeded
                elapsed_time = time.time() - run_start
                if elapsed_time >= run_time_limit:
                    time_limit_reached = True
                    print(f"\n⏰ TIME LIMIT REACHED for Run {run}: {RUN_TIME_LIMIT_MINUTES:.4f} minutes")
                    print(f"   Stopping remaining evolutions...")
                    # Cancel remaining futures
                    for f in futures:
                        if not f.done():
                            f.cancel()
                    break
                
                evo_id = futures[future]
                try:
                    r = future.result(timeout=0.1)  # Small timeout to check cancellation
                    evo_results.append(r)
                    elapsed_evo = time.time() - run_start
                    print(f" ✓ Evo {r['evolution_id']}: fitness={r['fitness']:.0f}, sim={r['similarity']:.1f}%, pents={r['pentagons']} (elapsed {elapsed_evo:.1f}s)")
                except Exception as e:
                    if "cancelled" in str(e).lower() or "cancelled" in str(type(e)).lower():
                        print(f" ✗ Evo {evo_id} cancelled due to time limit")
                    else:
                        print(" ✗ Evo", evo_id, "failed:", e)
            
            # If time limit reached, wait a bit for any pending results
            if time_limit_reached:
                print(f"   Waiting up to 2 seconds for pending results...")
                wait_start = time.time()
                processed_evo_ids = {r['evolution_id'] for r in evo_results}
                for future in futures:
                    if time.time() - wait_start > 2.0:
                        break
                    if future.done():
                        try:
                            r = future.result(timeout=0.1)
                            if r['evolution_id'] not in processed_evo_ids:
                                evo_results.append(r)
                                processed_evo_ids.add(r['evolution_id'])
                                print(f" ✓ Got late result: Evo {r['evolution_id']}")
                        except:
                            pass
            
            if not evo_results:
                print("No successful evolutions this run.")
                results_summary.append({'run': run, 'fitness': 0, 'similarity': 0, 'time': time.time()-run_start})
                continue
            # Sort by fitness (best first) - Z=1 is best, Z=2 is second best, etc.
            evo_results.sort(key=lambda x: x['fitness'], reverse=True)
            run_time = time.time() - run_start
            
            if time_limit_reached:
                print(f"Run {run} stopped by time limit: {run_time/60:.4f}m (limit: {RUN_TIME_LIMIT_MINUTES:.4f}m)")
                print(f"   Completed evolutions: {len(evo_results)}/5")
            else:
                print(f"Run {run} complete: {run_time/60:.1f}m")
            
            
            # COLLECT FITNESS DATA FOR GRAPHS 
            
            # Aggregate generation data from all evolutions in this run (use best evolution's data)
            if evo_results and 'generation_data' in evo_results[0]:
                best_evo_data = evo_results[0].get('generation_data', [])
                runs_fitness_data[run] = best_evo_data
            
            
            for i, res in enumerate(evo_results, 1):
                # Create arcade machine with generated image
                # Z = i (1-5), where 1 is best, 2 is second best, etc.
                arcade_img = create_arcade_machine_with_screen(res['image_array'])
                fname = f"TelmanNuruzovOutput{INPUT_IMAGE}_{run}_{i}.jpg"
                arcade_img.save(os.path.join(OUTPUT_DIR, fname), 'JPEG', quality=95)
                print(f"  Saved: {fname} (Z={i}, fitness={res['fitness']:.0f}, sim={res['similarity']:.1f}%)")
            best = evo_results[0]
            results_summary.append({'run': run, 'fitness': best['fitness'], 'similarity': best['similarity'], 'time': run_time})

    total_time = time.time() - total_start
    print("\nALL RUNS DONE. Total time: %.1fm" % (total_time/60.0))
    for r in results_summary:
        print(" Run", r['run'], "fit=%.0f sim=%.1f time=%.1fm" % (r['fitness'], r['similarity'], r['time']/60.0))
    
    
    # GENERATE FITNESS GRAPHS 
    
    if runs_fitness_data:
        plot_fitness_graphs(runs_fitness_data, OUTPUT_DIR, INPUT_IMAGE)
    


if __name__ == "__main__":
    mp.freeze_support()
    main_with_arcade()
