import galsim as galsim
import galsim.roman as roman
import numpy as np
from roman_imsim.obseq import ObSeqDataLoader

sca_number_to_file = {
    1: "SCA_22066_211227_v001.fits",
    2: "SCA_21815_211221_v001.fits",
    3: "SCA_21946_211225_v001.fits",
    4: "SCA_22073_211229_v001.fits",
    5: "SCA_21816_211222_v001.fits",
    6: "SCA_20663_211102_v001.fits",
    7: "SCA_22069_211228_v001.fits",
    8: "SCA_21641_211216_v001.fits",
    9: "SCA_21813_211219_v001.fits",
    10: "SCA_22078_211230_v001.fits",
    11: "SCA_21947_211226_v001.fits",
    12: "SCA_22077_211230_v001.fits",
    13: "SCA_22067_211227_v001.fits",
    14: "SCA_21814_211220_v001.fits",
    15: "SCA_21645_211228_v001.fits",
    16: "SCA_21643_211218_v001.fits",
    17: "SCA_21319_211211_v001.fits",
    18: "SCA_20833_211116_v001.fits",
}


class get_pointing(object):
    """
    Class to store stuff about the telescope
    """

    def __init__(self, params, visit, SCA):

        self.params = params
        file_name = params["input"]["obseq_data"]["file_name"]
        obseq_data = ObSeqDataLoader(file_name, visit, SCA, logger=None)
        self.filter = obseq_data.ob["filter"]
        self.sca = obseq_data.ob["sca"]
        self.visit = obseq_data.ob["visit"]
        self.date = obseq_data.ob["date"]
        self.exptime = obseq_data.ob["exptime"]
        self.bpass = roman.getBandpasses()[self.filter]
        self.WCS = roman.getWCS(
            world_pos=galsim.CelestialCoord(ra=obseq_data.ob["ra"], dec=obseq_data.ob["dec"]),
            PA=obseq_data.ob["pa"],
            date=self.date,
            SCAs=self.sca,
            PA_is_FPA=True,
        )[self.sca]
        self.radec = self.WCS.toWorld(galsim.PositionI(int(roman.n_pix / 2), int(roman.n_pix / 2)))


def translate_cvz(orig_radec, field_ra=9.5, field_dec=-44, cvz_ra=61.24, cvz_dec=-48.42):
    ra = orig_radec.ra / galsim.degrees - field_ra
    dec = orig_radec.dec / galsim.degrees - field_dec
    ra += cvz_ra / np.cos(cvz_dec * np.pi / 180)
    dec += cvz_dec
    return galsim.CelestialCoord(ra * galsim.degrees, dec * galsim.degrees)
