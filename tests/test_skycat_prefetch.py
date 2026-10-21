from unittest.mock import Mock

import pytest

from roman_imsim import skycat
from roman_imsim.skycat import SkyCatalogInterface


class _FakeObject:
    def __init__(self, object_type, calls):
        self.object_type = object_type
        self.calls = calls

    def clear_prefetched_seds(self):
        self.calls.append(("clear",))

    def prefetch_seds(self, objects):
        self.calls.append(("prefetch", tuple(objects)))


def _interface(objects):
    interface = SkyCatalogInterface.__new__(SkyCatalogInterface)
    interface._objects = objects
    interface._prefetched_sed_batch = None
    interface.logger = Mock()
    return interface


def test_prefetch_seds_ignores_legacy_diffsky_objects():
    calls = []
    diffsky = [_FakeObject("diffsky_galaxy", calls) for _ in range(3)]
    star = _FakeObject("star", calls)
    interface = _interface([diffsky[0], star, diffsky[1], diffsky[2]])

    interface.prefetch_seds(start=0, stop=3)
    assert calls == []


def test_prefetch_seds_selects_diffsky26_objects():
    calls = []
    diffsky26 = [_FakeObject("diffsky26_galaxy", calls) for _ in range(2)]
    interface = _interface(diffsky26)

    interface.prefetch_seds(start=0, stop=2)

    assert calls == [("clear",), ("prefetch", tuple(diffsky26))]

    # Per-object GalSim calls for the same range must not recompute the batch.
    interface.prefetch_seds(start=0, stop=2)
    assert len(calls) == 2


def test_skycat_obj_prefetches_repeated_explicit_indices(monkeypatch):
    calls = []
    diffsky26 = _FakeObject("diffsky26_galaxy", calls)
    interface = _interface([diffsky26])
    diffsky26.id = 1
    center = Mock()
    center.distanceTo.return_value = skycat.galsim.Angle(0, skycat.galsim.arcsec)
    interface.get_sca_center = Mock(return_value=center)
    interface.getNObjects = Mock(return_value=1)
    interface.getObj = Mock(return_value=object())
    monkeypatch.setattr(skycat.galsim.config, "GetInputObj", lambda *args: interface)
    monkeypatch.setattr(skycat.galsim.config, "SetDefaultIndex", lambda *args: None)
    monkeypatch.setattr(skycat.galsim.config, "GetAllParams", lambda *args, **kwargs: ({"index": 0}, False))
    monkeypatch.setattr(skycat.galsim.config, "GetRNG", lambda *args: None)

    config = {"index": 0}
    base = {"_roman_imsim_sed_batch": (0, 2), "world_center": center}
    skycat.SkyCatObj(config, base, ignore=None, gsparams=None, logger=Mock())

    assert calls == [("clear",), ("prefetch", (diffsky26,))]


def test_prefetch_seds_validates_batch_bounds():
    interface = _interface([])
    with pytest.raises(IndexError, match="Invalid SkyCatalog SED prefetch"):
        interface.prefetch_seds(start=0, stop=1)
