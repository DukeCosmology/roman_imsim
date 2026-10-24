"""CMOS read timing and Romanisim-compatible L1 detector readout.

Source electrons are supplied by the CMOS builders, not apportioned from a
completed exposure.  Physical charge persists between reads; electronics are
applied to copies, and digitization follows resultant averaging.
"""
from dataclasses import dataclass

import galsim
import numpy as np
from astropy.time import Time
from romanisim import l1
from romanisim import models


def validate_read_pattern(pattern):
    """Return independent integer read groups, rejecting ambiguous timing."""
    if not isinstance(pattern, (list, tuple)) or not pattern:
        raise ValueError("A read pattern must contain nonempty resultant groups")
    result = []
    previous = 0
    for group in pattern:
        if not isinstance(group, (list, tuple)) or not group:
            raise ValueError("Each resultant must contain at least one read")
        normalized = []
        for read in group:
            if isinstance(read, (bool, np.bool_)) or not isinstance(read, (int, np.integer)):
                raise ValueError("Read indices must be positive integers")
            if read <= previous:
                raise ValueError("Read indices must be positive and strictly increasing across resultants")
            normalized.append(int(read))
            previous = read
        result.append(normalized)
    return result


@dataclass
class CMOSRamp:
    """One exposure's absolute DN resultants and associated simulation data."""
    resultants: np.ndarray
    dq: np.ndarray
    read_pattern: list
    frame_time: float
    gain: object
    reference_read: object = None

    @property
    def exposure_time(self):
        return self.frame_time * self.read_pattern[-1][-1]


class CMOSReadout:
    """Apply configured Roman detector models without modifying stored charge."""

    def __init__(self, config, base, image, rng, logger):
        noise = config.get("noise", {})
        if not isinstance(noise, dict):
            raise galsim.GalSimConfigError('image.noise must be a dictionary')
        noise_type = noise.get("type", "NoNoise")
        if noise_type not in ("NoNoise", "RomanNoise"):
            raise galsim.GalSimConfigError("CMOS L1 simulation supports RomanNoise or NoNoise")
        self.enabled = noise_type == "RomanNoise"
        opt = {"mjd": float, "stray_light": bool, "thermal_background": bool,
               "reciprocity_failure": bool, "dark_current": dict,
               "nonlinearity": dict, "ipc": dict, "read_noise": dict,
               "sky_subtract": bool, "gain": dict}
        params = galsim.config.GetAllParams(noise, base, opt=opt)[0] if self.enabled else {}
        if params.get("sky_subtract", False):
            raise galsim.GalSimConfigError("L1 raw resultants cannot be sky-subtracted")
        if params.get("reciprocity_failure", False):
            raise galsim.GalSimConfigError("Reciprocity failure is not yet implemented for CMOS intervals")
        self.rng = rng
        self.background_rate = image.copy()
        self.background_rate.setZero()
        self.dark = self.nonlinearity = self.ipc = None
        self.read_noise = 0.
        self.gain = float(models.gain)
        filter_name = galsim.config.ParseValue(config, 'filter', base, str)[0]
        metadata = {"instrument": {"detector": f"WFI{base['SCA']:02d}",
                                   "optical_element": 'F' + filter_name[1:]},
                    "exposure": {"start_time": Time(galsim.config.ParseValue(config, 'mjd', base, float)[0], format='mjd')}}
        if self.enabled:
            sky = models.backgrounds.getSkyLevel(
                base['bandpass'], world_pos=base['wcs'].toWorld(image.true_center),
                exptime=1., date=metadata['exposure']['start_time'].to_datetime())
            if params.get('stray_light', False):
                sky *= 1 + models.parameters.stray_light_fraction
            base['wcs'].makeSkyImage(self.background_rate, sky)
            if params.get('thermal_background', False):
                self.background_rate += models.backgrounds.thermal_backgrounds[base['bandpass'].name]
            for key, cls in [('dark_current', models.DarkCurrent),
                             ('nonlinearity', models.Nonlinearity), ('ipc', models.IPC),
                             ('read_noise', models.ReadNoise), ('gain', models.Gain)]:
                settings = params.get(key, {})
                if not settings.get('turn_on', False):
                    if key == 'gain' and key in params:
                        self.gain = 1.
                    continue
                kwargs = {'usecrds': settings.get('use_crds', False), 'metadata': metadata}
                if key in ('dark_current', 'read_noise'):
                    kwargs['rng'] = rng
                if key == 'nonlinearity':
                    if kwargs['usecrds']:
                        kwargs['getdq'] = True
                    else:
                        # The default model constructor omits the zero-order
                        # coefficient; supply the documented quadratic NLfunc.
                        kwargs['coeffs'] = np.array([0., 1., models.nonlinearity.nonlinearity_beta])
                model = cls(**kwargs)
                if key == 'dark_current':
                    self.dark = model
                elif key == 'nonlinearity':
                    self.nonlinearity = model
                elif key == 'ipc':
                    self.ipc = model
                elif key == 'gain':
                    self.gain = model.gain
                else:
                    # Use the same read-noise values and DN convention as
                    # romanisim.image.gather_reference_data and l1.make_l1.
                    self.read_noise = model.read_noise
        def value(name, default):
            return galsim.config.ParseValue(config, name, base, float)[0] if name in config else default

        self.pedestal = value('pedestal', models.parameters.pedestal)
        self.pedestal_image = image * 0 + self.pedestal
        pedestal_sigma = value('pedestal_extra_noise',
                               models.parameters.pedestal_extra_noise if self.enabled else 0.)
        if pedestal_sigma < 0:
            raise ValueError('Pedestal noise must be nonnegative')
        if pedestal_sigma:
            self.pedestal_image.addNoise(galsim.GaussianNoise(rng, sigma=pedestal_sigma))
        self.saturation = value('saturation', models.parameters.reference_data['saturation'])
        if (not np.all(np.isfinite(self.gain)) or np.any(np.asarray(self.gain) <= 0)
                or not np.isfinite(self.saturation) or self.saturation <= 0):
            raise ValueError('Gain and saturation must be positive')

    def accumulate_background(self, charge, duration):
        increment = self.background_rate * duration
        increment.addNoise(galsim.PoissonNoise(self.rng))
        charge += increment
        if self.dark is not None:
            self.dark.apply(img=charge, exptime=duration)

    def read(self, charge):
        image = charge.copy()
        image += self.pedestal_image
        if self.nonlinearity is not None:
            self.nonlinearity.apply(img=image, electrons=True)
        return image.array.copy()

    def digitize(self, resultants, pattern, frame_time):
        """Match Romanisim's IPC, gain, read-noise, rounding, and saturation order."""
        cube = np.asarray(resultants, dtype='f4')
        if self.ipc is not None:
            for plane in cube:
                self.ipc.apply(img=plane)
        cube /= self.gain
        times = [np.asarray(group) * frame_time for group in pattern]
        l1.add_read_noise_to_resultants(cube, times, read_noise=self.read_noise, rng=self.rng)
        cube = np.round(cube)
        dq = np.zeros(cube.shape, dtype='uint32')
        if self.nonlinearity is not None and getattr(self.nonlinearity, 'dq', None) is not None:
            dq |= self.nonlinearity.dq
        dq[cube >= self.saturation] |= models.parameters.dqbits['saturated']
        cube = np.clip(cube, 0, self.saturation * 1.1)
        return cube, dq
