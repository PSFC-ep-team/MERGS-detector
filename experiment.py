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
		Solid("box", x=1.0, y=10.0, z=0.8, z_position=0.4),
		Solid("box", x=0.8, y=10.0, z=0.5, z_position=-0.25),
	],
	Beam("electron", 2.5, diameter=0.01),
	num_particles=10000, debug_mode=True,
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
		Solid("box", x=0.5, y=10.0, z=0.8, x_position=-0.5),
		Solid("box", x=0.5, y=10.0, z=0.8, x_position=0.0),
		Solid("box", x=0.5, y=10.0, z=0.8, x_position=0.5),
	],
	Beam("electron", 2.5, diameter=0.01),
	num_particles=10000, debug_mode=True,
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
		Solid("box", x=0.8, y=10.0, z=0.5, z_position=-0.5),
		Solid("box", x=0.8, y=10.0, z=0.5, z_position=0.0),
		Solid("box", x=0.8, y=10.0, z=0.5, z_position=0.5),
	],
	Beam("electron", 2.5, diameter=0.01),
	num_particles=10000, debug_mode=True,
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
		solids.append(Solid("box", x=0.094, y=10.0, z=0.094, x_position=0.1*i - 0.7, z_position=0.1*j - 0.5, material="EJ-100"))  # fiber
for i in range(15):
	for j in range(12):
		solids.append(Solid("box", x=0.094, y=10.0, z=0.003 if j == 0 or j == 11 else 0.006, x_position=0.1*i - 0.7, z_position=0.1*j - 0.55, material="PMMA"))  # top cladding
	solids.append(Solid("box", x=0.003 if i == 0 else 0.006, y=10.0, z=1.1, x_position=0.1*i - 0.75, z_position=0, material="PMMA"))  # side cladding
tracks = simulate(
	"EJ-100",
	solids,
	Beam("electron", 2.5, diameter=0.01),
	num_particles=10000, debug_mode=True,
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
		solids.append(Solid("box", x=0.094, y=10.0, z=0.094, z_position=0.1*i - 0.7, x_position=0.1*j - 0.5, material="EJ-100"))  # fiber
for i in range(15):
	for j in range(12):
		solids.append(Solid("box", x=0.003 if j == 0 or j == 11 else 0.006, y=10.0, z=0.094, x_position=0.1*j - 0.55, z_position=0.1*i - 0.7, material="PMMA"))  # side cladding
	solids.append(Solid("box", x=1.1, y=10.0, z=0.003 if i == 0 else 0.006, x_position=0, z_position=0.1*i - 0.75, material="PMMA"))  # top cladding
tracks = simulate(
	"EJ-100",
	solids,
	Beam("electron", 2.5, diameter=0.01),
	num_particles=10000, debug_mode=True,
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
		Solid("tube", z=2.54, x_rotation=90., deltaphi=2*pi, rmax=1.27, material="LaBr3"),
		Solid("tube", z=2.54, x_rotation=90., deltaphi=2*pi, rmin=1.27, rmax=1.42875, material="aluminum"),
	],
	Beam("electron", 2.5, diameter=0.01),
	num_particles=10000, debug_mode=True,
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
	[Solid("box", x=2.0, y=2.0, z=0.036)],
	Beam("electron", 2.5, diameter=0.01),
	num_particles=10000, debug_mode=True,
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
