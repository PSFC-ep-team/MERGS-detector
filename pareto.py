import os
import logging

import matplotlib.pyplot as plt
from numpy import pi, inf, array, linspace, savetxt, loadtxt, sqrt, concatenate
from scipy import optimize

from detector import calculate_sensitivity, Detector
from simulation import Beam, Spectrum


logging.basicConfig(
	level=logging.INFO, filename="results/out.log", encoding="utf-8",
	datefmt="%m-%d %H:%M:%S", format="%(asctime)s %(levelname)-5.5s %(message)s")
logging.getLogger().addHandler(logging.StreamHandler())


background_energy = linspace(0.5, 14, 21)
background_spectrum = Spectrum("E⁻²", background_energy, background_energy**-2)


def find_pareto_front(material: str) -> list[tuple[float, float, float, float, float]]:
	"""
	find the pareto front of designs with high sensitivity to signal and low sensitivity to background
	:param material: the material out of which the detector is made
	:return: a bunch of designs specified by their width (mm), depth (mm), lower threshold (MeV), upper threshold (MeV),
	         background sensitivity (mm²), and signal sensitivity
	"""
	try:
		results = loadtxt(f"results/pareto_{material}.txt")
	except FileNotFoundError:
		logging.info(f"starting pareto front calculation for {material}...")
		signal_sensitivities = 1 - linspace(1, 0, 9)[1:-1]**2
		results = []
		for target_signal_sensitivity in signal_sensitivities:
			width, depth, lower_threshold, upper_threshold, background_sensitivity = optimize_detector(material, target_signal_sensitivity)
			logging.info(f"found optimum that achieves {target_signal_sensitivity:.3g} for signal, {background_sensitivity:.3g} mm² for background")
			results.append((width, depth, lower_threshold, upper_threshold, background_sensitivity, target_signal_sensitivity))
		os.makedirs("result", exist_ok=True)
		savetxt(f"results/pareto_{material}.txt", results)
		logging.info(f"done!  saved to results/pareto_{material}.txt")
	return results


def optimize_detector(material: str, min_sensitivity: float) -> tuple[float, float, float, float, float]:
	"""
	get the optimal dimensions and thresholds for a detector of the given material with at least the given signal sensitivity
	:return: the width (mm), the depth (mm), the lower threshold (MeV), the upper threshold (MeV), and the achieved background sensitivity
	"""
	result = optimize.minimize(
		lambda x: calculate_background_sensitivity(material, *x),  # find the lowest background sensitivity
		constraints=[
			optimize.NonlinearConstraint(
				lambda x: calculate_signal_sensitivity(material, *x),  # for a given signal sensitivity
				lb=min_sensitivity, ub=inf),
			optimize.LinearConstraint(
				array([[0, 0, -1, 1]]),  # (lower threshold must < upper threshold)
				lb=0, keep_feasible=True),
		],
		x0=[15., 40., 8.0, 16.8],
		bounds=[
			(1., 50.),
			(0.2, 100.),
			(0.0, 16.6),
			(0.0, 16.8),
		],
		method="cobyqa",
		options=dict(
			initial_tr_radius=10.,
			final_tr_radius=1.e-3,
		),
	)

	print(result)
	width, depth, lower_threshold, upper_threshold = result.x
	return width, depth, lower_threshold, upper_threshold, result.fun


def calculate_signal_sensitivity(
		material: str, width: float, depth: float, lower_threshold: float, upper_threshold: float
) -> float:  # TODO: account for energy spread in the electron bean
	"""
	the detection efficiency of this detector assuming the beam is shaped to the detector
	"""
	detector = Detector(
		material=material, width=width, depth=depth,
		lower_threshold=lower_threshold, upper_threshold=upper_threshold)
	beam = Beam("electron", 16.7, diameter=2*width)
	signal_sensitivity = calculate_sensitivity(detector, beam, num_particles=100_000, use_cache=True, ignore_misses=True)
	return signal_sensitivity


def calculate_background_sensitivity(
		material: str, width: float, depth: float, lower_threshold: float, upper_threshold: float,
		include_neutrons=True, include_photons=True
) -> float:
	"""
	the background sensitivity of this detector assuming ambient neutrons and photons with a 1/E² spectrum,
	in counts per particle/mm²
	"""
	detector = Detector(
		material=material, width=width, depth=depth,
		lower_threshold=lower_threshold, upper_threshold=upper_threshold)
	world_radius = sqrt(width**2 + depth**2 + detector.length**2)/2
	neutron_beam = Beam("neutron", background_spectrum, distance=world_radius, ambient=True)
	photon_beam = Beam("photon", background_spectrum, distance=world_radius, ambient=True)
	total_detection_rate = 0.
	total_incidence_rate = 0.
	if include_neutrons:
		total_detection_rate += calculate_sensitivity(detector, neutron_beam, num_particles=1_000_000, use_cache=True)
		total_incidence_rate += 1/(4*pi*world_radius**2)
	if include_photons:
		total_detection_rate += calculate_sensitivity(detector, photon_beam, num_particles=1_000_000, use_cache=True)
		total_incidence_rate += 1/(4*pi*world_radius**2)
	return total_detection_rate/total_incidence_rate


if __name__ == "__main__":
	materials = ["EJ-276", "EJ-100", "LaBr3", "silicon"]
	styles = {"EJ-276": "C2.-", "EJ-100": "C2--", "LaBr3": "C0-", "silicon": "C1:"}

	fronts = {}
	for material in materials:
		fronts[material] = array(find_pareto_front(material))  # TODO: account for pulse-shape discrimination

	# plot the pareto fronts of performance
	plt.figure()
	for material in materials:
		plt.plot(
			concatenate([[0], fronts[material][:, 4]]),
			concatenate([[0], fronts[material][:, 5]]),
			styles[material], label=material)
	plt.grid()
	plt.xlim(0, git )
	plt.ylim(0, 1)
	plt.xlabel("Background sensitivity (mm²)")
	plt.ylabel("Signal sensitivity")
	plt.legend()
	plt.tight_layout()
	plt.savefig("figures/pareto.pdf")

	# plot the actual design variables
	fig, axs = plt.subplots(3, 1, sharex=True, gridspec_kw=dict(hspace=0))
	for material in materials:
		axs[0].plot(fronts[material][:, 5], fronts[material][:, 0], styles[material], label=material)
		axs[1].plot(fronts[material][:, 5], fronts[material][:, 1], styles[material])
		axs[2].plot(fronts[material][:, 5], fronts[material][:, 2], styles[material])
		axs[2].plot(fronts[material][:, 5], fronts[material][:, 3], styles[material])
	axs[0].legend()
	axs[0].grid()
	axs[0].set_ylabel("Width (mm)")
	axs[0].set_ylim(0, None)
	axs[1].grid()
	axs[1].set_ylabel("Depth (mm)")
	axs[1].set_ylim(0, None)
	axs[2].grid()
	axs[2].set_ylabel("Thresholds (MeV)")
	axs[2].set_ylim(0, 16.7)
	axs[2].set_xlabel("Signal sensitivity")
	axs[2].set_xlim(None, 1)
	fig.tight_layout()

	plt.show()
