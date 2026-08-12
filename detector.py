""" a formalism for rectangular detectors with simple energy thresholding """

from __future__ import annotations

import os

from numpy import count_nonzero, inf, histogram, array, isclose, sqrt, concatenate
from numpy.typing import NDArray

from data import MATERIAL_DATA
from simulation import Beam, simulate, Solid, Spectrum


def calculate_sensitivity(detector: Detector, beam: Beam, num_particles=10000, use_cache=False, skip_undetectable_tracks=True) -> tuple[float, float, float, float]:
	"""
	calculate the fraction of these incident particles that are detected by this detector and by adjacent detectors
	:return: the direct sensitivity, the direct sensitivity uncertainty, the cross-talk sensitivity, and the cross-talk sensitivity uncertainty
	"""
	cache_key = (f"{detector.material_name}, {detector.width}, {detector.depth}, "
	             f"{detector.lower_threshold}, {detector.upper_threshold}, "
	             f"{beam.particle_name}, {beam.energy}, {beam.diameter}, {beam.width}, {beam.height}, {beam.distance}, {beam.shape}")
	if use_cache:
		# first, try to load it from the cache
		try:
			with open("results/cache.txt", mode="r") as file:
				for line in file.readlines():
					input_string, output_string = line.split(" -> ")
					if input_string == cache_key:
						results = output_string.split(",")
						return tuple(float(x) for x in results)
		except FileNotFoundError:
			pass

	# truncate the spectrum to save time, since nothing lower than the lower threshold matters
	if type(beam.energy) is Spectrum:
		if detector.lower_threshold > beam.energy.energies.max():
			return 0, 0, 0, 0
		if skip_undetectable_tracks:
			truncated_spectrum, simulated_fraction = beam.energy.truncate(detector.lower_threshold)
			beam = Beam(beam.particle_name, truncated_spectrum, beam.shape, beam.diameter, beam.width, beam.height, beam.distance)
		else:
			simulated_fraction = 1
	else:
		if detector.lower_threshold > beam.energy:
			return 0, 0, 0, 0
		simulated_fraction = 1

	# do the simulation
	energy_deposited_directly, energy_deposited_indirectly, num_particles = calculate_response(detector, beam, num_particles)

	# calculate the sensitivity to signal
	num_detected = count_nonzero(
		(energy_deposited_directly >= detector.lower_threshold) &
		(energy_deposited_directly <= detector.upper_threshold)
	)
	direct_sensitivity = num_detected/(num_particles/simulated_fraction)
	direct_sensitivity_error = sqrt(num_detected*(num_particles - num_detected)/num_particles)/(num_particles/simulated_fraction)

	# calculate the sensitivity to cross-talk
	num_detected = count_nonzero(
		(energy_deposited_indirectly >= detector.lower_threshold) &
		(energy_deposited_indirectly <= detector.upper_threshold)
	)
	cross_sensitivity = num_detected/(num_particles/simulated_fraction)
	cross_sensitivity_error = sqrt(num_detected*(num_particles - num_detected)/num_particles)/(num_particles/simulated_fraction)

	if use_cache:
		os.makedirs("results", exist_ok=True)
		with open("results/cache.txt", mode="a") as file:
			file.write(f"{cache_key} -> {direct_sensitivity}, {direct_sensitivity_error}, {cross_sensitivity}, {cross_sensitivity_error}\n")

	return direct_sensitivity, direct_sensitivity_error, cross_sensitivity, cross_sensitivity_error


def calculate_response(detector: Detector, beam: Beam, num_particles=10000) -> tuple[NDArray, NDArray, int]:
	""" run a simulation for this detector and extract the total energy deposition of each particle in both this and adjacent detectors """
	solids = []
	# instantiate three adjacent detectors
	for x in [-detector.width - detector.separation, 0, detector.width + detector.separation]:
		solids.append(Solid("box", x_position=x, x=detector.width, y=detector.length, z=detector.depth))
	# instantiate some thin foil between them
	if detector.separation != 0:
		for x in [-detector.width/2 - detector.separation/2, detector.width/2 + detector.separation/2]:
			solids.append(Solid("box", x_position=x, x=detector.separation, y=detector.length, z=detector.depth, material="aluminum"))

	# run the simulation
	entries = simulate(
		detector.material_name,
		solids,
		beam,
		num_particles)

	# sort particles by detector
	responses = []
	for detector_index in range(3):
		responses.append(entries[entries["detector"] == detector_index]["E_depositedMeV"])  # TODO: account for finite photon statistics

	return responses[1], concatenate([responses[0], responses[2]]), num_particles  # combine the two adjacent detectors when you return


class Detector:
	def __init__(self, material: str, width: float, depth: float, length=10.0, separation=0, lower_threshold=0., upper_threshold=inf):
		"""
		a single channel of an electron detector
		:param material: the name of the detection material
		:param width: the scale of the detector in the dispersive direction (cm)
		:param depth: the scale of the detector in the beam direction (cm)
		:param length: the scale of the detector in the nondispersive direction (cm)
		:param separation: the amount of aluminum to put between adjacent detectors (cm)
		:param lower_threshold: the minimum amount of energy in a pulse to be detected (MeV)
		:param upper_threshold: the maximum amount of energy in a pulse to be detected (MeV)
		"""
		self.material_name = material
		self.density = MATERIAL_DATA[material]["density"]  # g/cm³
		self.elements = MATERIAL_DATA[material]["elements"]
		self.width = width
		self.depth = depth
		self.length = length
		self.separation = separation
		self.lower_threshold = lower_threshold
		self.upper_threshold = upper_threshold


def test_spectral_truncation():
	detector = Detector("EJ-276", 5.0, 10.0, lower_threshold=18)
	spectrum = Spectrum("uniform", array([10., 20.]), array([1., 1.]))
	num_particles = 1_000_000
	pure_sensitivity, _, _, _ = calculate_sensitivity(
		detector, Beam("electron", spectrum),
		num_particles=num_particles, skip_undetectable_tracks=False)
	clever_sensitivity, _, _, _ = calculate_sensitivity(
		detector, Beam("electron", spectrum),
		num_particles=num_particles, skip_undetectable_tracks=True)
	assert isclose(pure_sensitivity, clever_sensitivity, rtol=0.005)
