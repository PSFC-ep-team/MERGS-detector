""" calculate and plot expected stuff from some real experiments """


import os

import matplotlib
import matplotlib.pyplot as plt
from numpy import pi, partition, histogram, linspace, arange

from simulation import simulate, Beam, Solid


matplotlib.rcParams['axes.prop_cycle'] = matplotlib.cycler(color=["#e6b648", "#2abd41", "#04d6e7", "#000000"])


os.makedirs("figures", exist_ok=True)

energy_bins = linspace(0.03, 2.58, 52)

# combined EJ276 to stop whole beam
tracks = simulate(
	"EJ-276",
	[
		Solid("box", x=10, y=100, z=8, z_position=4.0),
		Solid("box", x=8, y=100, z=5, z_position=-2.5),
	],
	Beam("electron", 2.5, width=0.1),
)
response = histogram(tracks["EventID"], weights=tracks["E_depositedMeV"], bins=arange(-1/2, tracks["EventID"].max() + 1))[0]
plt.figure()
counts, _, _ = plt.hist(response, bins=energy_bins)
plt.xlim(0, energy_bins[-1])
plt.ylim(0, min(counts.max()*1.05, partition(counts, -2)[-2]*1.5))
plt.xlabel("Energy deposited (MeV)")
plt.title("Triple block of EJ-276")
plt.tight_layout()
plt.savefig("figures/experiment_triple_block.pdf")
print(f"{counts.max()/(tracks["EventID"].max() + 1):.1%} of the electrons are fully stopped")

# adjacent EJ276 to look at cross-talk
tracks = simulate(
	"EJ-276",
	[
		Solid("box", x=5, y=100, z=8, x_position=-5),
		Solid("box", x=5, y=100, z=8, x_position=0),
		Solid("box", x=5, y=100, z=8, x_position=5),
	],
	Beam("electron", 2.5, width=0.1),
)
plt.figure()
for detector in range(3):
	here = tracks["detector"] == detector
	response = histogram(tracks[here]["EventID"], weights=tracks[here]["E_depositedMeV"], bins=arange(-1/2, tracks["EventID"].max() + 1))[0]
	plt.hist(response, bins=energy_bins)
plt.xlim(0, energy_bins[-1])
plt.xlabel("Energy deposited (MeV)")
plt.title("Adjacent EJ-276s")
plt.tight_layout()
plt.savefig("figures/experiment_block_crosstalk.pdf")

# stacked EJ276 to look at stopping
tracks = simulate(
	"EJ-276",
	[
		Solid("box", x=8, y=100, z=5, z_position=-5),
		Solid("box", x=8, y=100, z=5, z_position=0),
		Solid("box", x=8, y=100, z=5, z_position=5),
	],
	Beam("electron", 2.5, width=0.1),
)
plt.figure()
for detector in range(3):
	here = tracks["detector"] == detector
	response = histogram(tracks[here]["EventID"], weights=tracks[here]["E_depositedMeV"], bins=arange(-1/2, tracks["EventID"].max() + 1))[0]
	plt.hist(response, bins=energy_bins)
plt.xlim(0, energy_bins[-1])
plt.xlabel("Energy deposited (MeV)")
plt.title("Stacked EJ-276s")
plt.tight_layout()
plt.savefig("figures/experiment_block_stopping.pdf")

# fibers to stop whole beam and look at cross-talk
solids = []
for i in range(15):
	for j in range(11):
		solids.append(Solid("box", x=0.94, y=100, z=0.94, x_position=i - 7, z_position=j - 5, material="EJ-100"))  # fiber
for i in range(15):
	for j in range(12):
		solids.append(Solid("box", x=0.94, y=100, z=0.03 if j == 0 or j == 11 else 0.06, x_position=(i - 7), z_position=j - 5.5, material="PMMA"))  # top cladding
	solids.append(Solid("box", x=0.03 if i == 0 else 0.06, y=100, z=11, x_position=(i - 7.5), z_position=0, material="PMMA"))  # side cladding
tracks = simulate(
	"EJ-100",
	solids,
	Beam("electron", 2.5, width=0.1),
)
plt.figure()
for detector_group in [(0, 50), (50, 100), (100, 150)]:
	here = (tracks["detector"] >= detector_group[0]) & (tracks["detector"] < detector_group[1])
	response = histogram(tracks[here]["EventID"], weights=tracks[here]["E_depositedMeV"], bins=arange(-1/2, tracks["EventID"].max() + 1))[0]
	plt.hist(response, bins=energy_bins)
plt.xlim(0, energy_bins[-1])
plt.xlabel("Energy deposited (MeV)")
plt.title("Adjacent fiber blocks")
plt.tight_layout()
plt.savefig("figures/experiment_fiber_crosstalk.pdf")

# fibers to stop whole beam and look at stopping
solids = []
for i in range(15):
	for j in range(11):
		solids.append(Solid("box", x=0.94, y=100, z=0.94, z_position=(i - 7), x_position=(j - 5), material="EJ-100"))  # fiber
for i in range(15):
	for j in range(12):
		solids.append(Solid("box", x=0.03 if j == 0 or j == 11 else 0.06, y=100, z=0.94, x_position=j - 5.5, z_position=i - 7, material="PMMA"))  # side cladding
	solids.append(Solid("box", x=11, y=100, z=0.03 if i == 0 else 0.06, x_position=0, z_position=i - 7.5, material="PMMA"))  # top cladding
tracks = simulate(
	"EJ-100",
	solids,
	Beam("electron", 2.5, width=0.1),
)
plt.figure()
for detector_group in [(0, 50), (50, 100), (100, 150)]:
	here = (tracks["detector"] >= detector_group[0]) & (tracks["detector"] < detector_group[1])
	response = histogram(tracks[here]["EventID"], weights=tracks[here]["E_depositedMeV"], bins=arange(-1/2, tracks["EventID"].max() + 1))[0]
	plt.hist(response, bins=energy_bins)
plt.xlim(0, energy_bins[-1])
plt.xlabel("Energy deposited (MeV)")
plt.title("Stacked fiber blocks")
plt.tight_layout()
plt.savefig("figures/experiment_fiber_stopping.pdf")

# LaBr₃
tracks = simulate(
	"LaBr3",
	[
		Solid("tube", z=25.4, x_rotation=90., deltaphi=2*pi, rmax=12.7, material="LaBr3"),
		Solid("tube", z=25.4, x_rotation=90., deltaphi=2*pi, rmin=12.7, rmax=14.2875, material="aluminum"),
	],
	Beam("electron", 2.5, width=0.1),
)
response = histogram(tracks["EventID"], weights=tracks["E_depositedMeV"], bins=arange(-1/2, tracks["EventID"].max() + 1))[0]
plt.figure()
counts, _, _ = plt.hist(response, bins=energy_bins)
plt.xlim(0, energy_bins[-1])
plt.ylim(0, min(counts.max()*1.05, partition(counts, -2)[-2]*1.5))
plt.xlabel("Energy deposited (MeV)")
plt.title("Single encased LaBr₃ cylinder")
plt.tight_layout()
plt.savefig("figures/experiment_LaBr.pdf")
print(f"{counts.sum()/(tracks["EventID"].max() + 1):.1%} of the electrons reach the crystal")

# silicon strip detector
tracks = simulate(
	"silicon",
	[Solid("box", x=20, y=20, z=0.36)],
	Beam("electron", 2.5, width=0.1),
)
response = histogram(tracks["EventID"], weights=tracks["E_depositedMeV"], bins=arange(-1/2, tracks["EventID"].max() + 1))[0]
plt.figure()
counts, _, _ = plt.hist(response, bins=energy_bins)
plt.xlim(0, energy_bins[-1])
plt.ylim(0, min(counts.max()*1.05, partition(counts, -2)[-2]*1.5))
plt.xlabel("Energy deposited (MeV)")
plt.title("Single silicon strip")
plt.tight_layout()
plt.savefig("figures/experiment_silicon_strip.pdf")

plt.show()
