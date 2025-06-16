# imports, etc.
import datetime

import galsim
import galsim.wfirst as wf
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os

from radec_to_chip import radec_to_chip

matplotlib.use("agg")

ra_cen = 26.25  # degrees
dec_cen = -26.25  # degrees
ra_cen_rad = 0.45814892864851153  # radians
dec_cen_rad = -0.45814892864851153  # radians
pa_rad = 0.0696662245219  # radians
date = datetime.datetime(2025, 1, 12)
seed = 314159

# create coords.txt
ralims = [23, 30]
declims = [-30, -23]
ud = galsim.UniformDeviate(seed)
ra_vals = []
dec_vals = []
for i in range(200000):
    ra_vals.append(ra_cen + (ud() - 0.5) / np.cos((dec_cen * galsim.degrees) / galsim.radians))
    dec_vals.append(dec_cen + ud() - 0.5)
ra_vals = np.array(ra_vals)
dec_vals = np.array(dec_vals)
mask = (ra_vals > ralims[0]) & (ra_vals < ralims[1]) & (dec_vals > declims[0]) & (dec_vals < declims[1])
ra_vals = ra_vals[mask] * np.pi / 180.0
dec_vals = dec_vals[mask] * np.pi / 180.0
np.savetxt("coords.txt", np.vstack((ra_vals, dec_vals)).T)

fpa_center = galsim.CelestialCoord(ra=ra_cen * galsim.degrees, dec=dec_cen * galsim.degrees)

wcs = wf.getWCS(fpa_center, PA=pa_rad * galsim.radians, date=date, PA_is_FPA=True)

# Find the SCAs from Chris's code (Python version) for the same points
sca_ch = radec_to_chip(ra_cen_rad, dec_cen_rad, pa_rad, ra_vals, dec_vals)
print(np.min(sca_ch), np.max(sca_ch))
sca_ch[np.where(sca_ch is None)[0]] = 0
np.savetxt("python.txt", sca_ch)

# Find the SCAs
sca = []
for i in range(len(ra_vals)):
    sca.append(
        wf.findSCA(
            wcs, galsim.CelestialCoord(ra=ra_vals[i] * galsim.radians, dec=dec_vals[i] * galsim.radians)
        )
    )
sca = np.array(sca)
for i in range(len(ra_vals)):
    if sca[i] is None:
        sca[i] = 0
print(np.min(sca_ch), np.max(sca_ch))
np.savetxt("galsim.txt", sca.astype(int))

np.savetxt("obsra.txt", np.array([ra_cen_rad]), fmt="%1.9f")
np.savetxt("obsdec.txt", np.array([dec_cen_rad]), fmt="%1.9f")
np.savetxt("obspa.txt", np.array([pa_rad]), fmt="%1.9f")
np.savetxt("len.txt", np.array([len(ra_vals)]).astype(int), fmt="%06d")
os.system("./a.out > c.txt")
sca_c = np.loadtxt("c.txt")
print(np.min(sca_c), np.max(sca_c))

# ----------------

coords = np.loadtxt("coords.txt")
ra_vals = coords[:, 0] * 180.0 / np.pi
dec_vals = coords[:, 1] * 180.0 / np.pi
sca_ch = np.loadtxt("python.txt")
sca = np.loadtxt("galsim.txt")
sca_c = np.loadtxt("c.txt")

# make a plot showing the points colored by their WCS, also with original pointing position shown

fig = plt.figure(figsize=(18, 5))
ax = fig.add_subplot(131)
mask = sca != 0
sc = ax.scatter(ra_vals[mask], dec_vals[mask], c=sca[mask], s=1, lw=0, cmap=plt.cm.viridis)
# The previous line is a change to make defaults like the newer matplotlib
# since the Ohio Supercomputer Center comp seems to have an older mpl by default
ax.scatter([ra_cen], [dec_cen], c="w", marker="o", s=40)
plt.xlabel("RA")
plt.ylabel("dec")
plt.colorbar(sc)
plt.title("GalSim #675")
xlim = ax.get_xlim()
ylim = ax.get_ylim()

ax2 = fig.add_subplot(132)
mask = sca_ch != 0
sc2 = ax2.scatter(ra_vals[mask], dec_vals[mask], c=sca_ch[mask], s=1, lw=0, cmap=plt.cm.viridis)
ax2.scatter([ra_cen], [dec_cen], c="w", marker="o", s=40)
plt.xlabel("RA")
plt.ylabel("dec")
plt.colorbar(sc2)
plt.title("Python vers of CH code")
ax2.set_xlim(xlim)
ax2.set_ylim(ylim)

ax3 = fig.add_subplot(133)
print(len(ra_vals), len(dec_vals), len(sca_c))
mask = sca_c != 0
sc3 = ax3.scatter(ra_vals[mask], dec_vals[mask], c=sca_c[mask], s=1, lw=0, cmap=plt.cm.viridis)
ax3.scatter([ra_cen], [dec_cen], c="w", marker="o", s=40)
plt.xlabel("RA")
plt.ylabel("dec")
plt.colorbar(sc3)
plt.title("Original CH code")
ax3.set_xlim(xlim)
ax3.set_ylim(ylim)

plt.savefig("panel.png")
plt.close()

for i in range(18):
    mask = sca_c == i + 1
    sc1 = plt.scatter(ra_vals[mask], dec_vals[mask], c="r", marker=".", s=1, lw=0)
    sc2 = plt.scatter(ra_vals[mask], dec_vals[mask], c="b", marker=".", s=1, lw=0)
    sc3 = plt.scatter(ra_vals[mask], dec_vals[mask], c="g", marker=".", s=1, lw=0)
    plt.xlabel("RA")
    plt.ylabel("dec")
    plt.title("Chip comparison " + str(i + 1))
    plt.savefig("chip_" + str(i + 1) + ".png")
    plt.close()


fig = plt.figure(figsize=(5, 5))
ax2 = fig.add_subplot(111)
mask = np.where(sca != sca_ch)[0]
sc2 = ax2.scatter(ra_vals[mask], dec_vals[mask], c=sca_ch[mask], s=1, lw=0, cmap=plt.cm.viridis)
sc2 = ax2.scatter(ra_vals[mask] + 0.002, dec_vals[mask], c=sca[mask], s=1, lw=0, cmap=plt.cm.viridis)
ax2.scatter([ra_cen], [dec_cen], c="w", marker="o", s=40)
plt.xlabel("RA")
plt.ylabel("dec")
plt.colorbar(sc2)
plt.title("Differences of galsim vs chris-sourced")
ax2.set_xlim(xlim)
ax2.set_ylim(ylim)
plt.savefig("panel_diff.png")
plt.close()

fig = plt.figure(figsize=(5, 5))
ax2 = fig.add_subplot(111)
mask = np.where(sca != sca_ch)[0]
sc2 = ax2.scatter(ra_vals[mask], dec_vals[mask], c=sca_ch[mask], s=1, lw=0, cmap=plt.cm.viridis)
ax2.scatter([ra_cen], [dec_cen], c="w", marker="o", s=40)
plt.xlabel("RA")
plt.ylabel("dec")
plt.colorbar(sc2)
plt.title("Differences of chris-sourced")
ax2.set_xlim(xlim)
ax2.set_ylim(ylim)
plt.savefig("panel_diff_chris.png")
plt.close()

fig = plt.figure(figsize=(5, 5))
ax2 = fig.add_subplot(111)
mask = np.where(sca != sca_ch)[0]
sc2 = ax2.scatter(ra_vals[mask] + 0.002, dec_vals[mask], c=sca[mask], s=1, lw=0, cmap=plt.cm.viridis)
ax2.scatter([ra_cen], [dec_cen], c="w", marker="o", s=40)
plt.xlabel("RA")
plt.ylabel("dec")
plt.colorbar(sc2)
plt.title("Differences of galsim")
ax2.set_xlim(xlim)
ax2.set_ylim(ylim)
plt.savefig("panel_diff_galsim.png")
plt.close()
