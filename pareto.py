import os
import logging
from typing import Callable

import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator
from numpy import pi, array, linspace, savetxt, loadtxt, sqrt, concatenate, stack, zeros, full, interp, \
	quantile, nanmax, geomspace, empty, percentile
from scipy import optimize
from scipy.special import erf

from data import MATERIAL_DATA
from detector import calculate_sensitivity, Detector, calculate_response
from simulation import Beam, Spectrum


plt.rcParams["font.size"] = 12

os.makedirs("results", exist_ok=True)
logging.basicConfig(
	level=logging.DEBUG, filename="results/out.log", encoding="utf-8",
	datefmt="%m-%d %H:%M:%S", format="%(asctime)s %(levelname)-5.5s %(message)s")
logging.getLogger().addHandler(logging.StreamHandler())
logging.getLogger("PIL").setLevel(logging.WARNING)
plt.set_loglevel(level="warning")


LENGTH = 10  # cm
INCIDENT_ENERGY = 16.7
MONOENERGETIC_SPECTRUM = Spectrum("16.5–16.9", array([INCIDENT_ENERGY - 0.2, INCIDENT_ENERGY + 0.2]), array([1., 1.]))
BACKGROUND_FLUENCE = 1e+1  # particle/cm²/electron

data = loadtxt("data/background-spectrum.csv", skiprows=1, delimiter=",", quotechar='"')
BACKGROUND_NEUTRON_SPECTRUM = Spectrum(
	"scattered neuts", (data[:, 0] + data[:, 1])/2, data[:, 2])
BACKGROUND_PHOTON_SPECTRUM = Spectrum(
	"scattered phots", (data[:, 0] + data[:, 1])/2, data[:, 4])
neutron_sum = sum(data[:, 2]*(data[:, 1] - data[:, 0]))
photon_sum = sum(data[:, 4]*(data[:, 1] - data[:, 0]))
NEUTRON_FRACTION = neutron_sum/(neutron_sum + photon_sum)
PHOTON_FRACTION = photon_sum/(neutron_sum + photon_sum)


def plot_pareto_fronts(materials: list[str], styles: dict[str, str]):
	os.makedirs("figures", exist_ok=True)

	fronts = {}
	for material in materials:
		fronts[material] = {}
		for optimistic in [False, True]:
			fronts[material][optimistic] = array(find_pareto_front(
				material, optimistic))
			if not optimistic:
				i = len(fronts[material][optimistic])//2
				width, depth, lower_threshold, upper_threshold, _, _ = fronts[material][optimistic][i, :]
				plot_responses(Detector(material, width, depth, LENGTH, lower_threshold=lower_threshold, upper_threshold=upper_threshold))

	# plot the pareto fronts of performance
	plt.figure()
	for material in materials:
		plt.errorbar(
			x=concatenate([[0], fronts[material][False][:, 4]]),
			y=concatenate([[0], fronts[material][False][:, 5]]),
			xerr=stack([
				concatenate([[0], fronts[material][False][:, 4] - fronts[material][True][:, 4]]),
				zeros(len(fronts[material][False]) + 1)],
				axis=0,
			),
			fmt=styles[material], label=material)
	plt.grid()
	plt.xlim(0, 1000)
	plt.ylim(0, 1)
	plt.xlabel("Background sensitivity (counts per signal electron)")
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
	axs[0].set_ylabel("Width (cm)")
	axs[0].set_ylim(0, None)
	axs[1].grid()
	axs[1].set_ylabel("Depth (cm)")
	axs[1].set_ylim(0, None)
	axs[2].grid()
	axs[2].set_ylabel("Thresholds (MeV)")
	axs[2].set_ylim(0, INCIDENT_ENERGY)
	axs[2].set_xlabel("Signal sensitivity")
	axs[2].set_xlim(None, 1)
	fig.tight_layout()
	plt.savefig("figures/pareto_parameters.pdf")


def plot_responses(detector: Detector):
	""" plot the response of a given detector design to all three kinds of radiation """
	electron_beam = Beam("electron", MONOENERGETIC_SPECTRUM, width=detector.width, height=LENGTH, shape="rectangular")
	electron_response, crosstalk_response, num_electrons = calculate_response(detector, electron_beam, num_particles=100_000)
	electron_weight = 1/num_electrons
	world_radius = sqrt(detector.width**2 + detector.depth**2 + detector.length**2)/2
	neutron_beam = Beam("neutron", BACKGROUND_NEUTRON_SPECTRUM, distance=world_radius, shape="ambient")
	neutron_response, _, num_neutrons = calculate_response(detector, neutron_beam, num_particles=1_000_000)
	neutron_weight = BACKGROUND_FLUENCE*4*pi*world_radius**2/num_neutrons
	photon_beam = Beam("photon", BACKGROUND_PHOTON_SPECTRUM, distance=world_radius, shape="ambient")
	photon_response, _, num_photons = calculate_response(detector, photon_beam, num_particles=1_000_000)
	photon_weight = BACKGROUND_FLUENCE*4*pi*world_radius**2/num_photons

	energy_bins = linspace(0.05, min(17.05, 1.5*detector.upper_threshold), 86)
	plt.figure()
	# plot the histograms
	for histogram_type, opacity, attach_label in [("stepfilled", 1/4, False), ("step", 1, True)]:
		counts, _, _ = plt.hist(
			[electron_response, crosstalk_response, photon_response, neutron_response],
			energy_bins,
			weights=[full(electron_response.size, electron_weight), full(crosstalk_response.size, electron_weight), full(photon_response.size, neutron_weight), full(neutron_response.size, photon_weight)],
			color=["tab:orange", "tab:red", "tab:green", "tab:gray"],
			label=["Signal", "Cross-talk", "Photons", "Neutrons"] if attach_label else None,
			histtype=histogram_type, alpha=opacity,
		)
	# plot the thresholds
	plt.axvline(detector.lower_threshold, linestyle="--", color="k")
	plt.axvline(detector.upper_threshold, linestyle="--", color="k")
	# plot the energy uncertainty at each threshold
	efficiency = MATERIAL_DATA[detector.material_name]["efficiency"]
	plt.errorbar(
		detector.lower_threshold, counts[0].max()*2/3,
		xerr=sqrt(detector.lower_threshold/efficiency), color="k", capsize=5)
	plt.errorbar(
		detector.upper_threshold, counts[0].max()*2/3,
		xerr=sqrt(detector.upper_threshold/efficiency), color="k", capsize=5)
	# adjust the axes
	plt.xlim(0, min(1.5*detector.upper_threshold, 18))
	plt.ylim(0, counts[0].max()*1.05)
	plt.legend()
	plt.xlabel("Deposited energy (MeV)")
	plt.title(f"{detector.width:.1f} cm × {detector.depth:.1f} cm {detector.material_name} detector")
	plt.tight_layout()
	plt.savefig(f"figures/{detector.material_name}_response.pdf")


def find_pareto_front(material: str, optimistic: bool) -> list[tuple[float, float, float, float, float]]:
	"""
	find the pareto front of designs with high sensitivity to signal and low sensitivity to background
	:param material: the material out of which the detector is made
	:param optimistic: whether we assume we can use pulse shape discrimination and coincidence subtraction
	:return: a bunch of designs specified by their width (cm), depth (cm), lower threshold (MeV), upper threshold (MeV),
	         background sensitivity, and signal sensitivity
	"""
	os.makedirs("results", exist_ok=True)

	if optimistic:
		try:
			parameters = loadtxt(f"results/pareto_{material}.txt", skiprows=1)
		except FileNotFoundError:
			raise FileNotFoundError("you have to calculate the conservative pareto front before you can calculate the optimistic pareto front.")
		results = []
		for width, depth, lower_threshold, upper_threshold, _, signal_sensitivity in parameters:
			coincidence_subtraction = True
			pulse_shape_discrimination = material.startswith("EJ")
			background_sensitivity = calculate_background_sensitivity(
				material, width, depth, lower_threshold, upper_threshold,
				include_photons=True,
				include_neutrons=not pulse_shape_discrimination,
				include_crosstalk=not coincidence_subtraction)
			results.append((width, depth, lower_threshold, upper_threshold, background_sensitivity, signal_sensitivity))

	else:
		try:
			results = loadtxt(f"results/pareto_{material}.txt", skiprows=1)
		except FileNotFoundError:
			logging.info(f"starting pareto front calculation for {material}...")
			signal_sensitivities = 1 - linspace(1, 0, 9)[1:-1]**2
			results = []
			for target_signal_sensitivity in signal_sensitivities:
				width, depth, lower_threshold, upper_threshold, background_sensitivity, signal_sensitivity = optimize_detector(
					material, target_signal_sensitivity)
				logging.info(f"found optimum that achieves {signal_sensitivity:.3g} for signal, {background_sensitivity:.3g} for background")
				results.append((width, depth, lower_threshold, upper_threshold, background_sensitivity, signal_sensitivity))
			savetxt(
				f"results/pareto_{material}.txt", results, delimiter="\t",
				header="width (cm)\tdepth (cm)\tlower threshold (MeV)\tupper threshold (MeV)\tbackground sensitivity\tsignal_sensitivity\n")
			logging.info(f"done!  saved to results/pareto_{material}.txt")

	return results


def optimize_detector(material: str, signal_sensitivity: float) -> tuple[float, float, float, float, float, float]:
	"""
	get the optimal dimensions and thresholds for a detector of the given material with at least the given signal sensitivity
	:return: the width (cm), the depth (cm), the lower threshold (MeV), the upper threshold (MeV), the achieved background sensitivity, and the achieved signal sensitivity
	"""
	if material != "silicon":
		# optimize with freely varying thickness
		result = None
		for initial_depth in [0.6, 5.0]:
			new_result = optimize.minimize(
				lambda x: calculate_background_sensitivity(material, x[0], x[1], x[2], x[2] + 100*signal_sensitivity),  # find the lowest background sensitivity
				x0=[1.5, initial_depth, 50.*(1 - signal_sensitivity)],
				bounds=[
					(0.1, 5.0),
					(0.1, 10.0),
					(0., 100.*(1 - signal_sensitivity)),
				],
				method="cobyqa",
				options=dict(
					initial_tr_radius=0.5,
					final_tr_radius=1.e-4,
				),
			)
			logging.debug(f"starting with {initial_depth} cm after {new_result.nfev} steps we ended up at {new_result.x[1]:.3g} cm for ({new_result.fun:.3g})")
			if result is None or new_result.fun < result.fun:
				result = new_result
		width, depth, lower_percentile = result.x

	else:
		# optimize with fixed thickness
		depth = 0.1
		result = optimize.minimize(
			lambda x: calculate_background_sensitivity(material, x[0], depth, x[1], x[1] + 100*signal_sensitivity),  # find the lowest background sensitivity
			x0=[1.5, 50.*(1 - signal_sensitivity)],
			bounds=[
				(0.1, 5.0),
				(0., 100.*(1 - signal_sensitivity)),
			],
			method="cobyqa",
			options=dict(
				initial_tr_radius=0.5,
				final_tr_radius=1.e-4,
			),
		)
		width, lower_percentile = result.x
	upper_percentile = lower_percentile + 100*signal_sensitivity

	print(result)
	lower_threshold, upper_threshold = calculate_thresholds(material, width, depth, lower_percentile, upper_percentile)
	return width, depth, lower_threshold, upper_threshold, result.fun, signal_sensitivity


def calculate_thresholds(
		material: str, width: float, depth: float, lower_percentile: float, upper_percentile: float
) -> tuple[float, float]:
	"""
	the thresholds that achieve the given percentiles
	"""
	width = max(0.001, width)
	depth = max(0.001, depth)
	detector = Detector(
		material=material, width=width, depth=depth, length=LENGTH)
	beam = Beam("electron", MONOENERGETIC_SPECTRUM, width=width, height=LENGTH, shape="rectangular")
	energies, _, _ = calculate_response(detector, beam, num_particles=100_000)
	efficiency = MATERIAL_DATA[material]["efficiency"]

	def fraction_below(threshold):
		energy_uncertainty = sqrt(energies/efficiency)
		score = (threshold - energies)/energy_uncertainty
		probability_below = 1/2 + 1/2*erf(score/sqrt(2))  # approximate the Poisson distribution as Gaussian so that it's continuus
		return probability_below.sum()/energies.size

	thresholds = []
	for percentile_value in [lower_percentile, upper_percentile]:
		threshold = find_root(
			lambda threshold: 100*fraction_below(threshold) - percentile_value,
			bracket=(0.0, 17.0),
			x0=percentile(energies, percentile_value),
		)
		thresholds.append(threshold)
	return tuple(thresholds)


def calculate_background_sensitivity(
		material: str, width: float, depth: float, lower_percentile: float, upper_percentile: float,
		include_neutrons=True, include_photons=True, include_crosstalk=True,
) -> float:
	"""
	the background sensitivity of this detector assuming ambient neutrons and photons with a 1/E² spectrum
	"""
	width = max(0.001, width)
	depth = max(0.001, depth)
	lower_threshold, upper_threshold = calculate_thresholds(material, width, depth, lower_percentile, upper_percentile)
	detector = Detector(
		material=material, width=width, depth=depth, length=LENGTH,
		lower_threshold=lower_threshold, upper_threshold=upper_threshold)
	world_radius = sqrt((3*width)**2 + depth**2 + detector.length**2)/2
	neutron_beam = Beam("neutron", BACKGROUND_NEUTRON_SPECTRUM, distance=world_radius, shape="ambient")
	photon_beam = Beam("photon", BACKGROUND_PHOTON_SPECTRUM, distance=world_radius, shape="ambient")
	electron_beam = Beam("electron", MONOENERGETIC_SPECTRUM, width=width, height=LENGTH, shape="rectangular")
	total_detection_rate = 0.
	total_detection_rate_var = 0.
	if include_neutrons:
		neutron_sensitivity, neutron_sensitivity_unc, _, _ = calculate_sensitivity(detector, neutron_beam, num_particles=1_000_000, use_cache=True)
		total_detection_rate += BACKGROUND_FLUENCE*4*pi*world_radius**2*neutron_sensitivity
		total_detection_rate_var += (BACKGROUND_FLUENCE*4*pi*world_radius*neutron_sensitivity_unc)**2
	if include_photons:
		photon_sensitivity, photon_sensitivity_unc, _, _ = calculate_sensitivity(detector, photon_beam, num_particles=1_000_000, use_cache=True)
		total_detection_rate += BACKGROUND_FLUENCE*4*pi*world_radius**2*photon_sensitivity
		total_detection_rate_var += (BACKGROUND_FLUENCE*4*pi*world_radius*photon_sensitivity_unc)**2
	if include_crosstalk:
		_, _, crosstalk_sensitivity, crosstalk_sensitivity_unc = calculate_sensitivity(detector, electron_beam, num_particles=100_000, use_cache=True)
		total_detection_rate += crosstalk_sensitivity
		total_detection_rate_var += crosstalk_sensitivity_unc**2

	total_detection_rate_unc = sqrt(total_detection_rate_var)
	if total_detection_rate_unc > .10*total_detection_rate:
		logging.warning(
			f"when calculating the sensitivity of a {detector.width:.2g}×{detector.depth:.2g} cm "
			f"{detector.material_name} detector to background, counting only particles between "
			f"{detector.lower_threshold:.2g} and {detector.upper_threshold:.2g} MeV, we got an unacceptably "
			f"uncertain anser of {total_detection_rate:.3g} ± {total_detection_rate_unc:.3g}.")

	return total_detection_rate


def find_root(f: Callable[[float], float], bracket: tuple[float, float], x0: float, **kwargs) -> float:
	"""
	it's like Scipy's quadratic Brent root-finding algorithm but it takes an initial gess,
	and returns the bounds if it seems out of bounds.
	and it assumes that the function is monotonicly increasing.
	"""
	if bracket[0] >= bracket[1]:
		raise ValueError(f"the bounds must be ascending, not {bracket}.")
	if x0 < bracket[0] or x0 > bracket[1]:
		raise ValueError(f"the initial gess {x0} is not in the feasible range {bracket}.")
	# check the bounds to make sure there's actually a root in this bracket
	left, right = bracket
	if f(left) >= 0:
		return left
	if f(right) <= 0:
		return right
	# shift x0 a bit if it's redundant with one of the bounds
	if x0 == left:
		x0 = 0.9*left + 0.1*right
	elif x0 == right:
		x0 = 0.1*left + 0.9*right
	# change one of the bounds to x0 to incorporate the initial gess into the search
	y0 = f(x0)
	if y0 < 0:
		left = x0
	elif y0 > 0:
		right = x0
	else:
		return x0
	# run Scipy's secant algorithm
	return optimize.root_scalar(f, bracket=(left, right), **kwargs).root


def test_plot_responses():
	plot_responses(Detector("EJ-276D", width=2, depth=5, lower_threshold=5, upper_threshold=17))


def test_objective_space():
	n = 9
	material = "EJ-276D"

	widths = linspace(0.1, 5.0, n)
	depths = linspace(0.1, 10.0, n)
	lower_percentiles = linspace(0., 50., n)
	upper_percentiles = linspace(50., 100., n)

	signal_sensitivities = empty((n, n))
	background_sensitivities = empty((n, n))

	lower_percentile = lower_percentiles[n//2]
	upper_percentile = upper_percentiles[n//2]
	for i, width in enumerate(widths):
		for j, depth in enumerate(depths):
			signal_sensitivities[i, j] = (upper_percentile - lower_percentile)/100
			background_sensitivities[i, j] = calculate_background_sensitivity(material, width, depth, lower_percentile, upper_percentile)
	plot_objective_space_slice(
		widths, depths, signal_sensitivities, background_sensitivities,
		"Width (cm)", "Depth (cm)")
	plt.savefig("figures/objective_slice_width-depth.pdf")

	width = widths[n//2]
	for i, lower_percentile in enumerate(lower_percentiles):
		if i == n//2: pass
		for j, depth in enumerate(depths):
			signal_sensitivities[i, j] = (upper_percentile - lower_percentile)/100
			background_sensitivities[i, j] = calculate_background_sensitivity(material, width, depth, lower_percentile, upper_percentile)
	plot_objective_space_slice(
		lower_percentiles, depths, signal_sensitivities, background_sensitivities,
		"Lower threshold (%)", "Depth (cm)")
	plt.savefig("figures/objective_slice_lower-depth.pdf")

	depth = depths[n//2]
	for i, lower_percentile in enumerate(lower_percentiles):
		for j, upper_percentile in enumerate(upper_percentiles):
			if j == n//2: pass
			signal_sensitivities[i, j] = (upper_percentile - lower_percentile)/100
			background_sensitivities[i, j] = calculate_background_sensitivity(material, width, depth, lower_percentile, upper_percentile)
	plot_objective_space_slice(
		lower_percentiles, upper_percentiles, signal_sensitivities, background_sensitivities,
		"Lower threshold (%)", "Upper threshold (%)")
	plt.savefig("figures/objective_slice_lower-upper.pdf")

	plt.close("all")


def plot_objective_space_slice(x, y, signal_sensitivities, background_sensitivities, x_label, y_label):
	fig = plt.figure()
	ax = fig.add_subplot()
	vmin = quantile(background_sensitivities[background_sensitivities > 0], .1)/6
	vmax = nanmax(background_sensitivities)
	mesh = ax.contourf(
		x, y, background_sensitivities.T, locator=LogLocator(),
		levels=geomspace(vmin, vmax, 21),
	)
	mesh.set_edgecolor("face")
	contours = ax.contour(x, y, signal_sensitivities, levels=[0.25, 0.5, 0.75, 0.875], colors="k")
	ax.clabel(contours)
	ax.set_xlabel(x_label)
	ax.set_ylabel(y_label)
	plt.colorbar(mesh, ticks=LogLocator().tick_values(vmin, vmax), extend="min").set_label("Background sensitivity")
	fig.tight_layout()


if __name__ == "__main__":
	plot_pareto_fronts(
		["EJ-276D", "EJ-100", "LaBr3", "silicon"],
		{"EJ-276D": "C2.-", "EJ-100": "C2--", "LaBr3": "C0-", "silicon": "C1:"})
	plt.show()
