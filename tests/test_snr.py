"""Tests for snr_to_noise_std: arch-derived GTGT-FM code rate (feat_channels)."""
import math

import pytest

from privacy._snr import snr_to_noise_std


def test_none_snr_returns_none():
    assert snr_to_noise_std(None, gt_alg=2, coded_pwr=1.0, feat_channels=128) is None


def test_itit_ignores_feat_channels():
    # gt_alg=1 uses code_rate=1.0 regardless of feat_channels.
    a = snr_to_noise_std(10.0, gt_alg=1, coded_pwr=1.0, feat_channels=128)
    b = snr_to_noise_std(10.0, gt_alg=1, coded_pwr=1.0, feat_channels=512)
    assert a == b
    assert math.isclose(a, math.sqrt(1.0 / (10 ** 1.0)), rel_tol=1e-9)


def test_gtgtfm_default_matches_original_512():
    # Default feat_channels preserves the original hardcoded ResNeXt-101 (512) behavior.
    default = snr_to_noise_std(0.0, gt_alg=2, coded_pwr=1.0)
    explicit_512 = snr_to_noise_std(0.0, gt_alg=2, coded_pwr=1.0, feat_channels=512)
    assert default == explicit_512
    expected = math.sqrt(1.0 / ((3 * 224 * 224) / (512 * 28 * 28)))
    assert math.isclose(default, expected, rel_tol=1e-9)


def test_gtgtfm_resnet18_128ch_halves_noise_std():
    # ResNet-18 layer2 = 128 channels -> code_rate 4x larger -> noise_std halved vs 512.
    std_512 = snr_to_noise_std(0.0, gt_alg=2, coded_pwr=1.0, feat_channels=512)
    std_128 = snr_to_noise_std(0.0, gt_alg=2, coded_pwr=1.0, feat_channels=128)
    assert math.isclose(std_128, std_512 / 2.0, rel_tol=1e-9)
