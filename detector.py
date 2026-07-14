""" a formalism for rectangular detectors with simple energy thresholding """

from __future__ import annotations

import os

from matplotlib import pyplot as plt
from numpy import count_nonzero, inf, linspace, empty_like, histogram, arange
from numpy.typing import NDArray

from data import MATERIAL_DATA
from simulation import Beam, simulate, Solid


def plot_sensitivity_curves(detector: Detector) -> None:
	""" calculate the sensitivity of a detector to all types and energies of radiation """
	energies = linspace(1, 17, 18)
	plt.figure()
	for particle, color in [("electron", "tab:orange"), ("photon", "tab:green"), ("neutron", "tab:gray")]:
		sensitivities = empty_like(energies)
		for i, energy in enumerate(energies):
			print(f"{energy:.2g} MeV {particle}s")
			sensitivities[i] = calculate_sensitivity(detector, Beam(particle, energy, ambient=(particle != "electron")))
		plt.plot(energies, sensitivities, color=color, label=particle)
	os.makedirs("figures", exist_ok=True)
	plt.legend()
	plt.grid()
	plt.xlabel("Incident energy")
	plt.ylabel("Sensitivity")
	plt.ylim(0, 1)
	plt.xlim(0, 18)
	plt.savefig("figures/sensitivity_curves.pdf")
	plt.show()


def calculate_sensitivity(detector: Detector, beam: Beam, num_particles=10000, ignore_misses=False, use_cache=False) -> float:
	""" calculate the fraction of these incident particles that are detected by this detector """
	cache_key = (f"{detector.material_name}, {detector.width}, {detector.depth}, "
	             f"{detector.lower_threshold}, {detector.upper_threshold}, "
	             f"{beam.particle_name}, {beam.energy}, {beam.width}, {'ambient' if beam.ambient else 'collimated'}, "
	             f"{'ignore misses' if ignore_misses else 'count misses'}")
	if use_cache:
		# first, try to load it from the cache
		try:
			with open("results/cache.txt", mode="r") as file:
				for line in file.readlines():
					input_string, output_string = line.split(" -> ")
					if input_string == cache_key:
						return float(output_string)
		except FileNotFoundError:
			pass

	# do the simulation
	energy_deposited = calculate_response(detector, beam, num_particles)

	# calculate the sensitivity
	num_detected = count_nonzero(
		(energy_deposited > 0) &
		(energy_deposited >= detector.lower_threshold) &
		(energy_deposited <= detector.upper_threshold)
	)
	if ignore_misses:
		num_total = count_nonzero(energy_deposited > 0)
	else:
		num_total = energy_deposited.size
	sensitivity = num_detected/num_total

	if use_cache:
		os.makedirs("result", exist_ok=True)
		with open("results/cache.txt", mode="a") as file:
			file.write(f"{cache_key} -> {sensitivity}\n")

	return sensitivity



def calculate_response(detector: Detector, beam: Beam, num_particles=10000) -> NDArray:
	""" run a simulation for this detector and extract the total energy deposition of each particle """
	tracks = simulate(
		detector.material_name,
		[Solid(
			"box",
			x=detector.width, y=100, z=detector.depth,
		)],
		beam,
		num_particles)
	return histogram(tracks["EventID"], weights=tracks["E_depositedMeV"], bins=arange(-1/2, num_particles))[0]


class Detector:
	def __init__(self, material: str, width: float, depth: float, lower_threshold=0., upper_threshold=inf):
		"""
		a single channel of an electron detector
		:param material: the name of the detection material
		:param width: the scale of the detector in the dispersive direction (mm)
		:param depth: the scale of the detector in the beam direction (mm)
		:param lower_threshold: the minimum amount of energy in a pulse to be detected (MeV)
		:param upper_threshold: the maximum amount of energy in a pulse to be detected (MeV)
		"""
		self.material_name = material
		self.density = MATERIAL_DATA[material]["density"]  # g/cm³
		self.elements = MATERIAL_DATA[material]["elements"]
		self.width = width
		self.depth = depth
		self.lower_threshold = lower_threshold
		self.upper_threshold = upper_threshold


if __name__ == "__main__":
	plot_sensitivity_curves(Detector("LaBr3", 10, 30, lower_threshold=8.25))
