""" calculate and plot expected stuff from some real experiments """


import os

import matplotlib
import matplotlib.pyplot as plt
from numpy import pi, partition, histogram, linspace, arange

from simulation import simulate, Beam, Solid

matplotlib.rcParams['axes.prop_cycle'] = matplotlib.cycler(color=["#e6b648", "#2abd41", "#04d6e7", "#000000"])


os.makedirs("figures", exist_ok=True)

energy_bins = linspace(0.03, 2.58, 52)

CHANNEL_DEPTH = 1.0
CHANNEL_WIDTH = 1.0

# combined EJ276 to stop whole beam
entries, num_particles = simulate(
	"EJ-276",
	[
		Solid("box", x=CHANNEL_DEPTH, y=10.0, z=2*CHANNEL_WIDTH),
	],
	Beam("electron", 2.5, diameter=0.01),
	num_particles=10000, debug_mode=True,
)
plt.figure()
counts, _, _ = plt.hist(entries["E_depositedMeV"], bins=energy_bins)
plt.xlim(0, energy_bins[-1])
plt.ylim(0, min(counts.max()*1.05, partition(counts, -2)[-2]*1.5))
plt.xlabel("Energy deposited (MeV)")
plt.title("Triple block of EJ-276")
plt.tight_layout()
plt.savefig("figures/experiment_triple_block.pdf")
print(f"{counts.max()/num_particles:.1%} of the electrons are fully stopped")

# adjacent EJ276 to look at cross-talk
entries, _ = simulate(
	"EJ-276",
	[
		Solid("box", x=CHANNEL_WIDTH, y=10.0, z=CHANNEL_DEPTH, x_position=0.0),
		Solid("box", x=CHANNEL_WIDTH, y=10.0, z=CHANNEL_DEPTH, x_position=CHANNEL_WIDTH),
	],
	Beam("electron", 2.5, diameter=0.01),
	num_particles=10000, debug_mode=True,
)
plt.figure()
for detector in range(3):
	here = entries["detector"] == detector
	plt.hist(entries[here]["E_depositedMeV"], bins=energy_bins)
plt.xlim(0, energy_bins[-1])
plt.xlabel("Energy deposited (MeV)")
plt.title("Adjacent EJ-276s")
plt.tight_layout()
plt.savefig("figures/experiment_block_crosstalk.pdf")

# stacked EJ276 to look at stopping
entries, _ = simulate(
	"EJ-276",
	[
		Solid("box", x=CHANNEL_DEPTH, y=10.0, z=CHANNEL_WIDTH, z_position=0.0),
		Solid("box", x=CHANNEL_DEPTH, y=10.0, z=CHANNEL_WIDTH, z_position=CHANNEL_WIDTH),
	],
	Beam("electron", 2.5, diameter=0.01),
	num_particles=10000, debug_mode=True,
)
plt.figure()
for detector in range(3):
	here = entries["detector"] == detector
	plt.hist(entries[here]["E_depositedMeV"], bins=energy_bins)
plt.xlim(0, energy_bins[-1])
plt.xlabel("Energy deposited (MeV)")
plt.title("Stacked EJ-276s")
plt.tight_layout()
plt.savefig("figures/experiment_block_stopping.pdf")

# fibers to stop whole beam
n_width = round(10*CHANNEL_WIDTH)
n_depth = round(10*CHANNEL_DEPTH)
Δx = -0.005
solids = []
for i in range(2*n_width):
	for j in range(n_depth):
		solids.append(Solid("box", x=0.094, y=10.0, z=0.094, z_position=0.1*(i + 1/2 - n_width/2), x_position=0.1*(j + 1/2 - n_depth/2) + Δx, material="EJ-100"))  # fiber
for i in range(2*n_width):
	for j in range(n_depth + 1):
		solids.append(Solid("box", x=0.003 if j == 0 or j == n_depth else 0.006, y=10.0, z=0.094, x_position=0.1*(j - n_depth/2) + Δx, z_position=0.1*(i + 1/2 - n_width/2), material="PMMA"))  # side cladding
for i in range(2*n_width + 1):
	solids.append(Solid("box", x=0.1*n_depth, y=10.0, z=0.003 if i == 0 or i == 2*n_width else 0.006, x_position=Δx, z_position=0.1*(i - n_width/2), material="PMMA"))  # top cladding
entries, num_particles = simulate(
	"EJ-100",
	solids,
	Beam("electron", 2.5, diameter=0.01),
	num_particles=10000, debug_mode=True,
)
plt.figure()
response = histogram(entries["EventID"], weights=entries["E_depositedMeV"], bins=arange(-1/2, num_particles))[0]
counts, _, _ = plt.hist(response, bins=energy_bins)
plt.xlim(0, energy_bins[-1])
plt.ylim(0, min(counts.max()*1.05, partition(counts, -2)[-2]*1.5))
plt.xlabel("Energy deposited (MeV)")
plt.title("Large fiber block")
plt.tight_layout()
plt.savefig("figures/experiment_triple_fiber.pdf")
print(f"{counts.max()/num_particles:.1%} of the electrons are fully stopped")

# fibers to look at cross-talk
solids = []
for i in range(2*n_width):
	for j in range(n_depth):
		solids.append(Solid("box", x=0.094, y=10.0, z=0.094, x_position=0.1*(i + 1/2 - n_width/2) + Δx, z_position=0.1*(j + 1/2 - n_depth/2), material="EJ-100"))  # fiber
for i in range(2*n_width):
	for j in range(n_depth + 1):
		solids.append(Solid("box", x=0.094, y=10.0, z=0.003 if j == 0 or j == n_depth else 0.006, x_position=0.1*(i + 1/2 - n_width/2) + Δx, z_position=0.1*(j - n_depth/2), material="PMMA"))  # top cladding
for i in range(2*n_width + 1):
	solids.append(Solid("box", x=0.003 if i == 0 or i == 2*n_width else 0.006, y=10.0, z=0.1*n_depth, x_position=0.1*(i - n_width/2) + Δx, z_position=0, material="PMMA"))  # side cladding
entries, _ = simulate(
	"EJ-100",
	solids,
	Beam("electron", 2.5, diameter=0.01),
	num_particles=10000, debug_mode=True,
)
plt.figure()
for detector_group in [(0, n_width*n_depth), (n_width*n_depth, 2*n_width*n_depth)]:
	here = (entries["detector"] >= detector_group[0]) & (entries["detector"] < detector_group[1])
	response = histogram(entries[here]["EventID"], weights=entries[here]["E_depositedMeV"], bins=arange(-1/2, num_particles))[0]
	plt.hist(response, bins=energy_bins)
plt.xlim(0, energy_bins[-1])
plt.xlabel("Energy deposited (MeV)")
plt.title("Adjacent fiber blocks")
plt.tight_layout()
plt.savefig("figures/experiment_fiber_crosstalk.pdf")

# fibers to look at stopping
solids = []
for i in range(2*n_width):
	for j in range(n_depth):
		solids.append(Solid("box", x=0.094, y=10.0, z=0.094, z_position=0.1*(i + 1/2 - n_width/2), x_position=0.1*(j + 1/2 - n_depth/2) + Δx, material="EJ-100"))  # fiber
for i in range(2*n_width):
	for j in range(n_depth + 1):
		solids.append(Solid("box", x=0.003 if j == 0 or j == n_depth else 0.006, y=10.0, z=0.094, x_position=0.1*(j - n_depth/2) + Δx, z_position=0.1*(i + 1/2 - n_width/2), material="PMMA"))  # side cladding
for i in range(2*n_width + 1):
	solids.append(Solid("box", x=0.1*n_depth, y=10.0, z=0.003 if i == 0 or i == 2*n_width else 0.006, x_position=Δx, z_position=0.1*(i - n_width/2), material="PMMA"))  # top cladding
entries, _ = simulate(
	"EJ-100",
	solids,
	Beam("electron", 2.5, diameter=0.01),
	num_particles=10000, debug_mode=True,
)
plt.figure()
for detector_group in [(0, n_width*n_depth), (n_width*n_depth, 2*n_width*n_depth)]:
	here = (entries["detector"] >= detector_group[0]) & (entries["detector"] < detector_group[1])
	response = histogram(entries[here]["EventID"], weights=entries[here]["E_depositedMeV"], bins=arange(-1/2, num_particles))[0]
	plt.hist(response, bins=energy_bins)
plt.xlim(0, energy_bins[-1])
plt.xlabel("Energy deposited (MeV)")
plt.title("Stacked fiber blocks")
plt.tight_layout()
plt.savefig("figures/experiment_fiber_stopping.pdf")

# LaBr₃
entries, num_particles = simulate(
	"LaBr3",
	[
		Solid("tube", z=2.54, x_rotation=90., deltaphi=2*pi, rmax=1.27, material="LaBr3"),
		Solid("tube", z=2.54, x_rotation=90., deltaphi=2*pi, rmin=1.27, rmax=1.32, material="aluminum"),
	],
	Beam("electron", 2.5, diameter=0.01),
	num_particles=10000, debug_mode=True,
)
plt.figure()
counts, _, _ = plt.hist(entries["E_depositedMeV"], bins=energy_bins)
plt.xlim(0, energy_bins[-1])
plt.ylim(0, min(counts.max()*1.05, partition(counts, -2)[-2]*1.5))
plt.xlabel("Energy deposited (MeV)")
plt.title("Single encased LaBr₃ cylinder")
plt.tight_layout()
plt.savefig("figures/experiment_LaBr.pdf")
print(f"{entries["E_depositedMeV"].size/num_particles:.1%} of the electrons reach the crystal")

# silicon strip detector
entries, _ = simulate(
	"silicon",
	[Solid("box", x=2.0, y=2.0, z=0.036)],
	Beam("electron", 2.5, diameter=0.01),
	num_particles=10000, debug_mode=True,
)
plt.figure()
counts, _, _ = plt.hist(entries["E_depositedMeV"], bins=energy_bins)
plt.xlim(0, energy_bins[-1])
plt.ylim(0, min(counts.max()*1.05, partition(counts, -2)[-2]*1.5))
plt.xlabel("Energy deposited (MeV)")
plt.title("Single silicon strip")
plt.tight_layout()
plt.savefig("figures/experiment_silicon_strip.pdf")

plt.show()
