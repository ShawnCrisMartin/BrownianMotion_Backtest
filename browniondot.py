import numpy as np

def generate_brownian_motion(S0, steps, num_paths):
    dt = 1/252
    paths = np.zeros((num_paths, steps))
    paths[:, 0] = S0
    for i in range(num_paths):
        for j in range(1, steps):
            Z = np.random.normal()
            paths[i, j] = paths[i, j-1] * np.exp(-0.5 * dt + np.sqrt(dt) * Z)
    return paths

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    paths = generate_brownian_motion(100, 504, 1)
    plt.plot(paths[0])
    plt.show()
