import functools

import jax
import jax.numpy as jnp
import numpy as np


@functools.partial(jax.jit, static_argnames=("size", "exponent", "fmin"))
def colored_noise(
    key: jax.Array,
    exponent: float,
    size: tuple[int, ...],  # (..., N) — last axis is time
    fmin: float = 0.0,
) -> jnp.ndarray:
    """
    Sample colored noise with power-law PSD ~ (1/f)^exponent.

    Based on https://github.com/felixpatzelt/colorednoise

    Args:
        key:      JAX random key.
        exponent: Power-law exponent β.
                    0 -> white noise
                    1 -> pink / flicker noise
                    2 -> brown / red noise
        size:     Output shape. Last axis is the time dimension.
        fmin:     Low-frequency cutoff in [0, 0.5] (relative to unit sample rate).

    Returns:
        Array of shape `size`, unit-variance, temporally correlated Gaussian noise.
    """
    samples = size[-1]

    # use numpy to keep f concrete under JIT — samples must be a static int.
    f = np.fft.rfftfreq(samples)  # shape (n_freqs,)
    n_freqs = len(f)

    if not (0 <= fmin <= 0.5):
        raise ValueError("fmin must be in [0, 0.5].")
    fmin = max(fmin, 1.0 / samples)

    ix = int(np.sum(f < fmin))
    if ix and ix < n_freqs:
        f[:ix] = f[ix]

    s_scale = f ** (-exponent / 2.0)  # shape (n_freqs,)
    w = s_scale[1:].copy()
    w[-1] *= (1 + samples % 2) / 2.0
    sigma = 2.0 * np.sqrt(np.sum(w**2)) / samples

    s_scale = jnp.asarray(s_scale)

    freq_size = size[:-1] + (n_freqs,)
    key_r, key_i = jax.random.split(key)
    sr = jax.random.normal(key_r, shape=freq_size) * s_scale
    si = jax.random.normal(key_i, shape=freq_size) * s_scale

    si = si.at[..., 0].set(0.0)
    sr = sr.at[..., 0].mul(jnp.sqrt(2.0))

    if samples % 2 == 0:
        si = si.at[..., -1].set(0.0)
        sr = sr.at[..., -1].mul(jnp.sqrt(2.0))

    s = sr + 1j * si
    y = jnp.fft.irfft(s, n=samples, axis=-1) / sigma  # shape == size

    return y


if __name__ == "__main__":
    """
    Example usage of colored noise generation.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    key = jax.random.PRNGKey(0)
    exponents = [0, 1, 2, 3, 4, 10]
    labels = ["White (β=0)", "Pink (β=1)", "Brown (β=2)", "β=3", "β=4", "β=10"]
    size = (8, 1024)

    n = len(exponents)
    fig, axes = plt.subplots(n, 2, figsize=(12, 2 * n), sharex=True)
    fig.subplots_adjust(
        hspace=0.15, wspace=0.08, bottom=0.08, top=0.92, left=0.08, right=0.97
    )
    for row, (exp, label) in enumerate(zip(exponents, labels)):
        y = np.asarray(colored_noise(key, exponent=exp, size=size))
        integral = np.cumsum(y, axis=-1)

        ax_n, ax_i = axes[row, 0], axes[row, 1]
        ax_n.plot(y.T, linewidth=0.8, alpha=0.7)
        ax_n.set_ylabel(label, fontsize=9)
        ax_n.set_yticks([])

        ax_i.plot(integral.T, linewidth=0.8, alpha=0.7)
        ax_i.set_yticks([])

    axes[0, 0].set_title("Noise", fontsize=10)
    axes[0, 1].set_title("Integral", fontsize=10)
    axes[-1, 0].set_xlabel("Time (samples)")
    axes[-1, 1].set_xlabel("Time (samples)")
    fig.suptitle("Colored Noise Examples", fontsize=13)
    plt.tight_layout()
    plt.show()
