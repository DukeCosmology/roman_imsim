from roman_imsim.utils import roman_utils


def test_getPSF_Image():
    config_path = "../SimHackJune2025/hack.yaml"
    pointing, sca = 12909, 4
    config = roman_utils(config_path, pointing, sca)
    size, x, y = 64, 2048, 2048
    psf = config.getPSF_Image(size, x, y)
    assert psf is not None
