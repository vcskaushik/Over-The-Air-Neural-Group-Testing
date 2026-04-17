"""Shared SNR-to-noise-std conversion matching main.snr_update_function."""
import math


def snr_to_noise_std(snr_db, gt_alg=1, coded_pwr=1.0):
    """Convert SNR (dB) to additive Gaussian noise standard deviation.

    Mirrors ``main.snr_update_function``:

        noise_std = sqrt(coded_pwr / (code_rate * 10^(SNR/10)))

    where ``code_rate`` is 1.0 for ITIT (gt_alg=1) and
    ``(3*224*224) / (512*28*28)`` for GTGT-FM (gt_alg=2).

    Args:
        snr_db:     SNR in dB.  Pass ``None`` to get ``None`` back (no channel noise).
        gt_alg:     Group-testing algorithm id — 1 for ITIT, 2 for GTGT-FM.
        coded_pwr:  Average power of the coded signal (from Stage A checkpoint key
                    ``"coded_pwr"``).  Defaults to 1.0 for backward compatibility
                    so that unit tests work without a real checkpoint.

    Returns:
        float noise standard deviation, or ``None`` when ``snr_db`` is ``None``.
    """
    if snr_db is None:
        return None
    if gt_alg == 2:
        code_rate = (3 * 224 * 224) / (512 * 28 * 28)
    else:
        code_rate = 1.0
    return float(math.sqrt(coded_pwr / (code_rate * (10 ** (snr_db / 10)))))
