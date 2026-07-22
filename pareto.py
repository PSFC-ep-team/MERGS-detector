import os
import logging

import matplotlib.pyplot as plt
from numpy import pi, inf, array, linspace, savetxt, loadtxt, sqrt, concatenate, stack, zeros, full
from scipy import optimize

from detector import calculate_sensitivity, Detector, calculate_response
from simulation import Beam, Spectrum


os.makedirs("results", exist_ok=True)
logging.basicConfig(
	level=logging.INFO, filename="results/out.log", encoding="utf-8",
	datefmt="%m-%d %H:%M:%S", format="%(asctime)s %(levelname)-5.5s %(message)s")
logging.getLogger().addHandler(logging.StreamHandler())


monoenergetic_spectrum = Spectrum("16.5–16.9", array([16.5, 16.9]), array([1., 1.]))
background_energy = linspace(0.5, 14, 21)
background_spectrum = Spectrum("E⁻²", background_energy, background_energy**-2)


def plot_responses(detector: Detector):
	""" plot the response of a given detector design to all three kinds of radiation """
	background_fluence = 0.01  # neutron/mm²/electron

	electron_beam = Beam("electron", monoenergetic_spectrum, diameter=2*detector.width)
	electron_response = calculate_response(detector, electron_beam, num_particles=100_000)
	electron_weight = 1/(100_000*(sqrt(3)/(2*pi) + 1/3))
	world_radius = sqrt(detector.width**2 + detector.depth**2 + detector.length**2)/2
	neutron_beam = Beam("neutron", background_spectrum, distance=world_radius, ambient=True)
	neutron_response = calculate_response(detector, neutron_beam, num_particles=1_000_000)
	neutron_weight = background_fluence*4*pi*world_radius**2/1_000_000
	photon_beam = Beam("photon", background_spectrum, distance=world_radius, ambient=True)
	photon_response = calculate_response(detector, photon_beam, num_particles=1_000_000)
	photon_weight = background_fluence*4*pi*world_radius**2/1_000_000

	energy_bins = linspace(0.05, 17.05, 86)
	plt.figure()
	plt.hist(
		electron_response, energy_bins, weights=full(electron_response.shape, electron_weight),
		color="tab:orange", alpha=0.5, label="Electrons")
	plt.hist(
		photon_response, energy_bins, weights=full(photon_response.shape, photon_weight),
		color="tab:green", alpha=0.5, label="Photons")
	plt.hist(
		neutron_response, energy_bins, weights=full(neutron_response.shape, neutron_weight),
		color="tab:gray", alpha=0.5, label="Neutrons")
	plt.axvline(detector.lower_threshold, linestyle="--", color="k")
	plt.axvline(detector.upper_threshold, linestyle="--", color="k")
	plt.xlim(0, 17)
	plt.legend()
	plt.xlabel("Deposited energy (MeV)")
	plt.title(f"{detector.width:.0f} mm × {detector.depth:.0f} mm {detector.material_name} detector")
	plt.tight_layout()
	plt.savefig(f"figures/{detector.material_name}_response.pdf")


def find_pareto_front(material: str, pulse_shape_discrimination: bool) -> list[tuple[float, float, float, float, float]]:
	"""
	find the pareto front of designs with high sensitivity to signal and low sensitivity to background
	:param material: the material out of which the detector is made
	:param pulse_shape_discrimination: whether we think there's PSD
	:return: a bunch of designs specified by their width (mm), depth (mm), lower threshold (MeV), upper threshold (MeV),
	         background sensitivity (mm²), and signal sensitivity
	"""
	os.makedirs("results", exist_ok=True)

	if pulse_shape_discrimination:
		try:
			parameters = loadtxt(f"results/pareto_{material}.txt")
		except FileNotFoundError:
			raise FileNotFoundError("you have to calculate the pareto front without PSD before you can calculate it with.")
		results = []
		for width, depth, lower_threshold, upper_threshold, _, signal_sensitivity in parameters:
			background_sensitivity = calculate_background_sensitivity(
				material, width, depth, lower_threshold, upper_threshold,
				include_photons=True, include_neutrons=False)
			results.append((width, depth, lower_threshold, upper_threshold, background_sensitivity, signal_sensitivity))

	else:
		try:
			results = loadtxt(f"results/pareto_{material}.txt")
		except FileNotFoundError:
			logging.info(f"starting pareto front calculation for {material}...")
			signal_sensitivities = 1 - linspace(1, 0, 9)[1:-1]**2
			results = []
			for target_signal_sensitivity in signal_sensitivities:
				width, depth, lower_threshold, upper_threshold, background_sensitivity, signal_sensitivity = optimize_detector(
					material, target_signal_sensitivity)
				logging.info(f"found optimum that achieves {signal_sensitivity:.3g} for signal, {background_sensitivity:.3g} mm² for background")
				results.append((width, depth, lower_threshold, upper_threshold, background_sensitivity, signal_sensitivity))
			savetxt(f"results/pareto_{material}.txt", results)
			logging.info(f"done!  saved to results/pareto_{material}.txt")

	return results


def optimize_detector(material: str, min_sensitivity: float) -> tuple[float, float, float, float, float, float]:
	"""
	get the optimal dimensions and thresholds for a detector of the given material with at least the given signal sensitivity
	:return: the width (mm), the depth (mm), the lower threshold (MeV), the upper threshold (MeV), the achieved background sensitivity (mm²), and the achieved signal sensitivity
	"""
	if material != "silicon":
		# optimize with freely varying thickness
		result = None
		for initial_depth in [11., 50.]:
			new_result = optimize.minimize(
				lambda x: calculate_background_sensitivity(material, *x),  # find the lowest background sensitivity
				constraints=[
					optimize.NonlinearConstraint(
						lambda x: calculate_signal_sensitivity(material, *x),  # for a given signal sensitivity
						lb=min_sensitivity, ub=inf),
					optimize.LinearConstraint(
						array([[0, 0, -1, 1]]),  # (lower threshold must < upper threshold)
						lb=0, keep_feasible=True),
				],
				x0=[15., initial_depth, 1.0, 16.7],
				bounds=[
					(1., 50.),
					(0.1, 100.),
					(0.0, 16.7),
					(0.0, 17.0),
				],
				method="cobyqa",
				options=dict(
					initial_tr_radius=10.,
					final_tr_radius=1.e-3,
				),
			)
			logging.debug(f"starting with {initial_depth} mm after {new_result.nfev} steps we ended up at {new_result.x[1]:.3g} mm for ({calculate_signal_sensitivity(material, *new_result.x):.3g}, {new_result.fun:.3g})")
			if result is None or new_result.fun < result.fun:
				result = new_result
		width, depth, lower_threshold, upper_threshold = result.x

	else:
		# optimize with fixed thickness
		depth = 1.0
		result = optimize.minimize(
			lambda x: calculate_background_sensitivity(material, x[0], depth, x[1], x[2]),  # find the lowest background sensitivity
			constraints=[
				optimize.NonlinearConstraint(
					lambda x: calculate_signal_sensitivity(material, x[0], depth, x[1], x[2]),  # for a given signal sensitivity
					lb=min_sensitivity, ub=inf),
				optimize.LinearConstraint(
					array([[0, -1, 1]]),  # (lower threshold must < upper threshold)
					lb=0, keep_feasible=True),
			],
			x0=[15., 1.0, 16.8],
			bounds=[
				(1., 50.),
				(0.0, 16.6),
				(0.0, 16.8),
			],
			method="cobyqa",
			options=dict(
				initial_tr_radius=10.,
				final_tr_radius=1.e-3,
			),
		)
		width, lower_threshold, upper_threshold = result.x

	print(result)
	signal_sensitivity = calculate_signal_sensitivity(material, width, depth, lower_threshold, upper_threshold)
	return width, depth, lower_threshold, upper_threshold, result.fun, signal_sensitivity


def calculate_signal_sensitivity(
		material: str, width: float, depth: float, lower_threshold: float, upper_threshold: float
) -> float:
	"""
	the detection efficiency of this detector assuming the beam is shaped to the detector
	"""
	width = max(0.01, width)
	depth = max(0.01, depth)
	detector = Detector(
		material=material, width=width, depth=depth,
		lower_threshold=lower_threshold, upper_threshold=upper_threshold)
	beam = Beam("electron", monoenergetic_spectrum, diameter=2*width)
	signal_sensitivity = calculate_sensitivity(detector, beam, num_particles=100_000, use_cache=True)
	valid_incidence_fraction = sqrt(3)/(2*pi) + 1/3
	return signal_sensitivity/valid_incidence_fraction


def calculate_background_sensitivity(
		material: str, width: float, depth: float, lower_threshold: float, upper_threshold: float,
		include_neutrons=True, include_photons=True
) -> float:
	"""
	the background sensitivity of this detector assuming ambient neutrons and photons with a 1/E² spectrum,
	in counts per particle/mm²
	"""
	width = max(0.01, width)
	depth = max(0.01, depth)
	detector = Detector(
		material=material, width=width, depth=depth,
		lower_threshold=lower_threshold, upper_threshold=upper_threshold)
	world_radius = sqrt(width**2 + depth**2 + detector.length**2)/2
	neutron_beam = Beam("neutron", background_spectrum, distance=world_radius, ambient=True)
	photon_beam = Beam("photon", background_spectrum, distance=world_radius, ambient=True)
	total_detection_rate = 0.
	if include_neutrons:
		total_detection_rate += calculate_sensitivity(detector, neutron_beam, num_particles=1_000_000, use_cache=True)
	if include_photons:
		total_detection_rate += calculate_sensitivity(detector, photon_beam, num_particles=1_000_000, use_cache=True)
	return total_detection_rate*4*pi*world_radius**2


if __name__ == "__main__":
	materials = ["EJ-276", "EJ-100", "LaBr3", "silicon"]
	styles = {"EJ-276": "C2.-", "EJ-100": "C2--", "LaBr3": "C0-", "silicon": "C1:"}

	fronts = {}
	for material in materials:
		fronts[material] = {}
		for pulse_shape_discrimination in [False, True]:
			fronts[material][pulse_shape_discrimination] = array(find_pareto_front(
				material, pulse_shape_discrimination))
			if not pulse_shape_discrimination:
				i = len(fronts[material][pulse_shape_discrimination])//2
				width, depth, lower_threshold, upper_threshold, _, _ = fronts[material][pulse_shape_discrimination][i, :]
				plot_responses(Detector(material, width, depth, lower_threshold=lower_threshold, upper_threshold=upper_threshold))

	fronts["silicon"][True] = fronts["silicon"][False]  # silicon detectors can't have PSD

	# plot the pareto fronts of performance
	plt.figure()
	for material in materials:
		plt.errorbar(
			x=concatenate([[0], fronts[material][False][:, 4]]),
			y=concatenate([[0], fronts[material][False][:, 5]]),
			xerr=stack([
				concatenate([[0], fronts[material][True][:, 4] - fronts[material][False][:, 4]]),
				zeros(len(fronts[material][False]) + 1)],
				axis=0,
			),
			fmt=styles[material], label=material)
	plt.grid()
	plt.xlim(0, 10)
	plt.ylim(0, 1)
	plt.xlabel("Background sensitivity (mm²)")
	plt.ylabel("Signal sensitivity")
	plt.legend()
	plt.tight_layout()
	plt.savefig("figures/pareto.pdf")

	# plot the actual design variables
	fig, axs = plt.subplots(3, 1, sharex=True, gridspec_kw=dict(hspace=0))
	for material in materials:
		axs[0].plot(fronts[material][False][:, 5], fronts[material][False][:, 0], styles[material], label=material)
		axs[1].plot(fronts[material][False][:, 5], fronts[material][False][:, 1], styles[material])
		axs[2].plot(fronts[material][False][:, 5], fronts[material][False][:, 2], styles[material])
		axs[2].plot(fronts[material][False][:, 5], fronts[material][False][:, 3], styles[material])
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
	plt.savefig("figures/pareto-parameters.pdf")

	plt.show()
