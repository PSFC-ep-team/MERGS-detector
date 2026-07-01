from math import floor
import os

from matplotlib import pyplot as plt
from numpy import linspace, sqrt, diff, histogram, arange, reshape

from simulation import Beam, simulate, Solid


def moliere_radius(material: str) -> float:
	""" calculate the Moliere radius of a material """
	pass


def plot_energy_deposition(material: str, beam: Beam, width: float, depth: float) -> None:
	"""
	generate some plots that show the distribution of energy deposition in this material
	:param material: the name of the detection material
	:param beam: the species and energy of the particles that are depositing energy
	:param width: the scale of the detector in the dispersive direction (mm)
	:param depth: the scale of the detector in the beam direction (mm)
	"""
	x_edges = linspace(-width/2, width/2, floor(10*sqrt(width/depth)) + 1)
	z_edges = linspace(0, depth, floor(10*sqrt(depth/width)) + 1)
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
				y_position=0,
				z_position=z_positions[j],
				x=x_sizes[i],
				y=100,
				z=z_sizes[j],
			))

	tracks = simulate(material, grid, beam)
	deposition = reshape(
		histogram(tracks["detector"], weights=tracks["E_depositedMeV"], bins=arange(len(grid) + 1))[0],
		shape=(x_positions.size, z_positions.size))

	os.makedirs("figures", exist_ok=True)
	plt.figure()
	plt.imshow(deposition.T, extent=(-width/2, width/2, 0, depth), cmap="inferno")
	plt.savefig("figures/deposition_heatmap.pdf")
	plt.close("all")


if __name__ == "__main__":
	print(moliere_radius("LYSO"))
	plot_energy_deposition("EJ-276", Beam("electron", 16.7), 50, 100)
