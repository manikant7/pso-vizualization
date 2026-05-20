import numpy as np
import cv2

from .optimizer import ParticleSwarmOptimizer
from .functions import rastrigin

IMG_SIZE = 600


def build_contour_image(bounds, size=IMG_SIZE):
    low, high = bounds
    x = np.linspace(low, high, size)
    y = np.linspace(low, high, size)
    X, Y = np.meshgrid(x, y)
    Z = rastrigin(X, Y)

    Z_norm = np.log1p(Z)
    Z_norm = ((Z_norm - Z_norm.min()) / (Z_norm.max() - Z_norm.min()) * 255).astype(np.uint8)
    Z_norm = np.flipud(Z_norm)
    return cv2.applyColorMap(Z_norm, cv2.COLORMAP_VIRIDIS)


def world_to_pixel(positions, bounds, size=IMG_SIZE):
    low, high = bounds
    normalized = (positions - low) / (high - low)
    px = (normalized[:, 0] * (size - 1)).astype(int)
    py = ((1 - normalized[:, 1]) * (size - 1)).astype(int)
    return np.column_stack([px, py])


def draw_star(img, center, size, color, thickness=2):
    cx, cy = int(center[0]), int(center[1])
    s = size
    cv2.line(img, (cx - s, cy), (cx + s, cy), color, thickness)
    cv2.line(img, (cx, cy - s), (cx, cy + s), color, thickness)
    cv2.line(img, (cx - s // 2, cy - s // 2), (cx + s // 2, cy + s // 2), color, thickness)
    cv2.line(img, (cx - s // 2, cy + s // 2), (cx + s // 2, cy - s // 2), color, thickness)


def main():
    bounds = (-5.12, 5.12)
    n_iterations = 100
    pso = ParticleSwarmOptimizer(rastrigin, bounds, n_particles=40, w=0.6, c1=1.8, c2=1.8)
    pso.run(n_iterations)

    contour = build_contour_image(bounds)
    window_name = "PSO - Rastrigin Function (press q to quit)"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    while True:
        for frame_idx, positions in enumerate(pso.history):
            frame = contour.copy()
            pixels = world_to_pixel(positions, bounds)

            for px, py in pixels:
                cv2.circle(frame, (px, py), 4, (0, 0, 255), -1)
                cv2.circle(frame, (px, py), 5, (255, 255, 255), 1)

            scores = np.array([rastrigin(x, y) for x, y in positions])
            best_idx = np.argmin(scores)
            best_px = pixels[best_idx]
            draw_star(frame, best_px, 12, (0, 255, 255), 2)

            cv2.putText(frame, f"Iteration: {frame_idx}/{n_iterations}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Best: {scores[best_idx]:.4f}",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(80) & 0xFF
            if key == ord("q") or key == 27:
                cv2.destroyAllWindows()
                return

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
