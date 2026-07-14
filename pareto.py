from numpy import pi, inf, array, linspace
from scipy import optimize

from detector import calculate_sensitivity, Detector
from simulation import Beam, Spectrum


background_energy = linspace(0.5, 14, 21)
background_spectrum = Spectrum("E⁻²", background_energy, background_energy**-2)


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
		x0=[20., 80., 8.0, 16.8],
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
	beam = Beam("electron", 16.7, width=2*width)
	signal_sensitivity = calculate_sensitivity(detector, beam, num_particles=100_000, use_cache=True, ignore_misses=True)
	return signal_sensitivity


def calculate_background_sensitivity(
		material: str, width: float, depth: float, lower_threshold: float, upper_threshold: float,
		include_neutrons=True, include_photons=True
) -> float:
	"""
	the background sensitivity of this detector assuming ambient neutrons and photons with a 1/E² spectrum,
	in counts per particle/cm²
	"""
	detector = Detector(
		material=material, width=width, depth=depth,
		lower_threshold=lower_threshold, upper_threshold=upper_threshold)
	neutron_beam = Beam("neutron", background_spectrum, width=100., ambient=True)
	photon_beam = Beam("photon", background_spectrum, width=100., ambient=True)
	total_detection_rate = 0.
	total_incidence_rate = 0.
	if include_neutrons:
		total_detection_rate += calculate_sensitivity(detector, neutron_beam, num_particles=1_000_000, use_cache=True)
		total_incidence_rate += 1/(pi*100**2)
	if include_photons:
		total_detection_rate += calculate_sensitivity(detector, photon_beam, num_particles=1_000_000, use_cache=True)
		total_incidence_rate += 1/(pi*100**2)
	return total_detection_rate/total_incidence_rate


if __name__ == "__main__":
	optimize_detector("EJ-276", 0.5)
