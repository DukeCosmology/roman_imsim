[![Build](https://github.com/DukeCosmology/roman_imsim/actions/workflows/build.yml/badge.svg?branch=main)](https://github.com/DukeCosmology/roman_imsim/actions/workflows/build.yml)
[![End-to-end Test](https://github.com/DukeCosmology/roman_imsim/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/DukeCosmology/roman_imsim/actions/workflows/ci.yml)

# roman_imsim


Nonstandard dependencies are represented in setup.py.

See wiki for information on how to run and documentation.

## installation

Installing this software does _not_ install certain data that the user may expect, such as stellar SED libraries.
Information about downloading those data can be found [here](https://lsstdesc.org/skyCatalogs/installation.html#install-needed-data-files).
Additionally, the relevant environment variables (i.e., `SIMS_SED_LIBRARY_DIR`) need be set for skyCatalogs to work properly.

## roman_imsim_testdata

Testing data is provided as a [git submodule](https://git-scm.com/book/en/v2/Git-Tools-Submodules) pointing to <https://github.com/DukeCosmology/roman_imsim_testdata>.

To access these data, run
```bash
git submodule init
git submodule update
```

### CMOS up-the-ramp L1 output

Use `roman_imsim/config/cmos_asdf.yaml` as a template with your normal observation
sequence and sky-catalog inputs. It selects `roman_sca_cmos`, `Roman_stamp_CMOS`,
and `RomanASDF` output at level 1. The existing non-CMOS ASDF configuration keeps
writing a two-dimensional `ImageModel`.

The template sets `image.exptime` from the selected pattern at Romanisim's
physical frame cadence. Override it explicitly for another cadence.

The CMOS builder shoots each source separately during each frame and keeps
cumulative electrons between reads. Resultants are the mean of their constituent
nondestructive reads. A read pattern can be selected with
`input.resultant_data.ma_table_number`, supplied directly as `read_pattern`
(e.g. `[[1], [2, 3], [5]]`), or loaded using `file_name` and `strategy_name`.
Indices are positive, globally increasing frame numbers; skipped frames still
accumulate charge. `image.exptime` is the reset-to-final-read duration, so the
frame cadence is `exptime / read_pattern[-1][-1]`. To use Romanisim's physical
cadence, set that duration to `romanisim.models.parameters.read_time` times the
last read index.

The ASDF file contains a `roman` `ScienceRawModel` with uint16 DN resultants,
reference-pixel borders, amp33, and exposure/read-pattern metadata. Simulation DQ
and gain are stored under `romanisim`, following Romanisim's L1 packaging. This
uses the actual CMOS resultants, not a redistribution of a completed exposure.
Separate `resultant_N.fits` files are no longer written. Keep `image.nproc: 1`
(the default) so each source's RNG stream persists across frames; independent
files may still use output-level multiprocessing.

Use `image.noise.type: RomanNoise` to enable interval sky backgrounds and the
configured detector models; `NoNoise` suppresses backgrounds and random detector
noise. L1 readout always adds a pedestal (default: Romanisim's electron pedestal),
averages reads, converts to DN (default: Romanisim's gain), rounds, and flags and
clips saturation. Override `image.pedestal` or `image.saturation` for controlled
experiments. `RomanNoise` uses the existing per-effect `turn_on`/`use_crds`
dictionaries. An explicitly disabled gain means unity gain. Nonlinearity is
applied to read copies before averaging; IPC, gain, read noise, rounding, and
saturation follow Romanisim's resultant-level order. Sky subtraction and
reciprocity failure are rejected for this raw L1 path. Charge-dependent effects,
persistence, and cosmic rays are not yet simulated by these builders.

Optional `image.reference_read: true` simulates the reset/reference read and
stores it alongside resultants encoded with Romanisim's reference subtraction.
`output.data_encoding_offset` optionally overrides the encoding offset. Decode
these files with `romanisim.l1.decode_reference_read`. No reference subtraction
is performed by default. Frame-level readout is supported; intra-frame rolling
read timing is not modeled.
