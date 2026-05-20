import numpy as np


def smooth_step(t):
    return t * t * (3 - 2 * t)


def interpolate_history(history, n_sub_frames=5):
    frames = []
    for i in range(len(history) - 1):
        start = history[i]
        end = history[i + 1]
        for s in range(n_sub_frames):
            t = smooth_step(s / n_sub_frames)
            frames.append(start + t * (end - start))
    frames.append(history[-1])
    return frames
