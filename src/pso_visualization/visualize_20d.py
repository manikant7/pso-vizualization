import numpy as np
import cv2

from .optimizer import ParticleSwarmOptimizer
from .functions import rastrigin
from .interpolate import interpolate_history

PANEL_SIZE = 200
ROWS = 2
COLS = 5
PADDING = 2
N_DIMS = 20
N_SUB_FRAMES = 5


def build_contour_panel(bounds, size=PANEL_SIZE):
    low, high = bounds
    x = np.linspace(low, high, size)
    y = np.linspace(low, high, size)
    X, Y = np.meshgrid(x, y)
    Z = rastrigin(np.stack([X, Y], axis=-1))

    Z_norm = np.log1p(Z)
    z_range = Z_norm.max() - Z_norm.min()
    Z_norm = ((Z_norm - Z_norm.min()) / (z_range if z_range > 0 else 1) * 255).astype(np.uint8)
    Z_norm = np.flipud(Z_norm)
    return cv2.applyColorMap(Z_norm, cv2.COLORMAP_VIRIDIS)


def world_to_pixel(coords_2d, bounds, size=PANEL_SIZE):
    low, high = bounds
    normalized = (coords_2d - low) / (high - low if high != low else 1)
    px = np.clip((normalized[:, 0] * (size - 1)).astype(int), 0, size - 1)
    py = np.clip(((1 - normalized[:, 1]) * (size - 1)).astype(int), 0, size - 1)
    return np.column_stack([px, py])


def main():
    bounds = (-5.12, 5.12)
    n_iterations = 150
    pso = ParticleSwarmOptimizer(rastrigin, bounds, n_particles=60, n_dims=N_DIMS, w=0.6, c1=1.8, c2=1.8)
    pso.run(n_iterations)

    frames = interpolate_history(pso.history, N_SUB_FRAMES)

    contour = build_contour_panel(bounds)
    dim_pairs = [(2 * i, 2 * i + 1) for i in range(10)]

    canvas_h = ROWS * PANEL_SIZE + (ROWS + 1) * PADDING + 50
    canvas_w = COLS * PANEL_SIZE + (COLS + 1) * PADDING

    window_name = "20D PSO - Rastrigin (press q to quit)"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    while True:
        for frame_idx, positions in enumerate(frames):
            canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

            scores = np.array([rastrigin(p) for p in positions])
            best_idx = np.argmin(scores)

            for panel_idx, (d1, d2) in enumerate(dim_pairs):
                row = panel_idx // COLS
                col = panel_idx % COLS
                x0 = col * PANEL_SIZE + (col + 1) * PADDING
                y0 = row * PANEL_SIZE + (row + 1) * PADDING

                panel = contour.copy()

                pair_positions = positions[:, [d1, d2]]
                pixels = world_to_pixel(pair_positions, bounds)

                for px, py in pixels:
                    cv2.circle(panel, (px, py), 3, (0, 0, 255), -1)
                    cv2.circle(panel, (px, py), 4, (255, 255, 255), 1)

                best_pair = positions[best_idx, [d1, d2]].reshape(1, 2)
                best_px = world_to_pixel(best_pair, bounds)[0]
                s = 8
                cx, cy = int(best_px[0]), int(best_px[1])
                cv2.line(panel, (cx - s, cy), (cx + s, cy), (0, 255, 255), 2)
                cv2.line(panel, (cx, cy - s), (cx, cy + s), (0, 255, 255), 2)
                cv2.line(panel, (cx - s // 2, cy - s // 2), (cx + s // 2, cy + s // 2), (0, 255, 255), 1)
                cv2.line(panel, (cx - s // 2, cy + s // 2), (cx + s // 2, cy - s // 2), (0, 255, 255), 1)

                cv2.putText(panel, f"d{d1}-d{d2}",
                            (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

                canvas[y0:y0 + PANEL_SIZE, x0:x0 + PANEL_SIZE] = panel

            iteration = frame_idx // N_SUB_FRAMES
            info_y = ROWS * PANEL_SIZE + (ROWS + 1) * PADDING + 25
            cv2.putText(canvas, f"Iteration: {iteration}/{n_iterations}   Best: {scores[best_idx]:.4f}",
                        (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            cv2.imshow(window_name, canvas)
            key = cv2.waitKey(20) & 0xFF
            if key == ord("q") or key == 27:
                cv2.destroyAllWindows()
                return

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
