""" some cool plots of energy deposition by electron beams in solids """

import os

from matplotlib import pyplot as plt
from numpy import linspace, sqrt, diff, histogram, arange, reshape, partition, count_nonzero

from detector import Detector, response
from simulation import Beam, simulate, Solid


def moliere_radius(material: str) -> float:
	""" calculate the Moliere radius of a material """
	pass


def plot_histogram(detector: Detector, beam: Beam) -> None:
	"""
	visualize the energy distribution of energy deposition in this material
	:param detector: the detector the particles are hitting
	:param beam: the species and energy of the particles that are depositing energy
	"""
	energy_deposition = response(detector, beam)
	bins = linspace(0, beam.energy*1.05, 51)
	counts = histogram(energy_deposition, bins=bins, density=True)[0]

	full_deposition = count_nonzero(energy_deposition >= 0.99*beam.energy)/energy_deposition.size

	os.makedirs("figures", exist_ok=True)
	plt.figure()
	plt.stairs(counts, bins, fill=True, linewidth=0)
	plt.text(
		0.01, 0.99,
		f"{full_deposition:.1%} of electrons deposit their full energy",
		ha="left", va="top", transform=plt.gca().transAxes,
	)
	plt.xlim(bins[0], bins[-1])
	plt.ylim(0, min(counts.max()*1.05, partition(counts, -2)[-2]*1.5))
	plt.xlabel("Energy deposited (MeV)")
	plt.savefig("figures/deposition_histogram.pdf")
	plt.show()


def plot_heatmap(detector: Detector, beam: Beam) -> None:
	"""
	visualize the spacial distribution of energy deposition in this material
	:param detector: the detector to depict
	:param beam: the species and energy of the particles that are depositing energy
	"""
	x_edges = linspace(-detector.width/2, detector.width/2, round((10*sqrt(detector.width/detector.depth) + 1)/2)*2)  # make sure there's an odd number of x bins
	z_edges = linspace(0, detector.depth, round(10*sqrt(detector.depth/detector.width) + 1))
	x_positions = (x_edges[0:-1] + x_edges[1:])/2
	z_positions = (z_edges[0:-1] + z_edges[1:])/2
	x_sizes = diff(x_edges)
	z_sizes = diff(z_edges)

	grid = []
	for i in range(x_positions.size):
		for j in range(z_positions.size):
			grid.append(Solid(
				"box",
				x_position=x_positions[i],
				z_position=z_positions[j],
				x=x_sizes[i],
				y=100,
				z=z_sizes[j],
			))

	tracks = simulate(detector.material_name, grid, beam)
	deposition = reshape(
		histogram(tracks["detector"], weights=tracks["E_depositedMeV"], bins=arange(len(grid) + 1))[0],
		shape=(x_positions.size, z_positions.size))

	os.makedirs("figures", exist_ok=True)
	plt.figure()
	plt.imshow(deposition.T, extent=(-detector.width/2, detector.width/2, 0, detector.depth), cmap="inferno")
	plt.savefig("figures/deposition_heatmap.pdf")
	plt.show()


if __name__ == "__main__":
	print(moliere_radius("LYSO"))
	# plot_histogram(Detector("EJ-276", 50, 80), Beam("electron", 16.7))
	plot_heatmap(Detector("EJ-100", 6, 10), Beam("electron", 2.5))
