import argparse
import os
import logging
from multiprocessing import Pool, cpu_count
from typing import Callable, Literal

import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator
from numpy import pi, array, linspace, savetxt, loadtxt, sqrt, concatenate, full, interp, \
	quantile, nanmax, geomspace, percentile, inf
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
logging.getLogger("filelock").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)


FOCAL_PLANE_HEIGHT = 10  # cm
BACKGROUND_FLUENCE = 1e+3  # particle/cm²/electron

data = loadtxt("data/background-spectrum.csv", skiprows=1, delimiter=",", quotechar='"')
BACKGROUND_NEUTRON_SPECTRUM = Spectrum(
	"scattered neuts", (data[:, 0] + data[:, 1])/2, data[:, 2])
BACKGROUND_PHOTON_SPECTRUM = Spectrum(
	"scattered phots", (data[:, 0] + data[:, 1])/2, data[:, 4])
neutron_sum = sum(data[:, 2]*(data[:, 1] - data[:, 0]))
photon_sum = sum(data[:, 4]*(data[:, 1] - data[:, 0]))
NEUTRON_FRACTION = neutron_sum/(neutron_sum + photon_sum)
PHOTON_FRACTION = photon_sum/(neutron_sum + photon_sum)


def plot_pareto_fronts(materials: list[str], styles: dict[str, str], spectrometric: bool):
	os.makedirs("figures", exist_ok=True)

	fronts = {}
	for material in materials:
		if spectrometric and material == "silicon":
			logging.warning("Silicon strips cannot be a spectrometer")
			continue
		fronts[material] = {}
		for optimistic in [False, True]:
			fronts[material][optimistic] = array(find_pareto_front(
				material, optimistic, spectrometric))

	# plot the pareto fronts of performance
	logging.info("Generating final plots...")
	for material in fronts.keys():
		i = len(fronts[material][False])//2
		width, depth, length, lower_threshold, upper_threshold, _, _ = fronts[material][False][i, :]
		plot_responses(Detector(material, width, depth, length, lower_threshold=lower_threshold, upper_threshold=upper_threshold), incident_energy=16.7)

	for optimistic in [True, False]:
		plt.figure()
		for material in fronts.keys():
			plt.plot(
				concatenate([[0], fronts[material][optimistic][:, 4]]),
				concatenate([[0], fronts[material][optimistic][:, 5]]),
				styles[material], label=material)
		plt.grid()
		plt.xlim(0, 10)
		plt.ylim(0, 1)
		plt.xlabel("Background sensitivity (counts per signal electron)")
		plt.ylabel("Signal sensitivity")
		plt.legend()
		plt.tight_layout()
		plt.savefig(f"figures/pareto_{'optimistic' if optimistic else 'conservative'}_{'spectrometer' if spectrometric else 'detector'}.pdf")

		# plot the actual design variables
		fig, axs = plt.subplots(3, 1, sharex=True, gridspec_kw=dict(hspace=0))
		for material in fronts.keys():
			axs[0].plot(fronts[material][optimistic][:, 5], fronts[material][optimistic][:, 0], styles[material], label=material)
			axs[1].plot(fronts[material][optimistic][:, 5], fronts[material][optimistic][:, 1], styles[material])
			axs[2].plot(fronts[material][optimistic][:, 5], fronts[material][optimistic][:, 2], styles[material])
			axs[2].plot(fronts[material][optimistic][:, 5], fronts[material][optimistic][:, 3], styles[material])
		axs[0].legend()
		axs[0].grid()
		axs[0].set_ylabel("Width (cm)")
		axs[0].set_ylim(0, None)
		axs[1].grid()
		axs[1].set_ylabel("Depth (cm)")
		axs[1].set_ylim(0, None)
		axs[2].grid()
		axs[2].set_ylabel("Thresholds (MeV)")
		axs[2].set_ylim(0, 16.7)
		axs[2].set_xlabel("Signal sensitivity")
		axs[2].set_xlim(None, 1)
		fig.tight_layout()
		plt.savefig(f"figures/pareto_parameters_{'optimistic' if optimistic else 'conservative'}_{'spectrometer' if spectrometric else 'detector'}.pdf")


def plot_responses(detector: Detector, incident_energy: float, num_background_particles=5_000_000):
	""" plot the response of a given detector design to all three kinds of radiation """
	num_electrons, num_neutrons, num_photons = 1_000_000, num_background_particles, num_background_particles
	electron_beam = Beam("electron", tight_spectrum(incident_energy), width=detector.width, height=FOCAL_PLANE_HEIGHT, shape="rectangular")
	electron_response, crosstalk_response = calculate_response(detector, electron_beam, num_particles=num_electrons)
	electron_weight = 1/num_electrons
	world_radius = sqrt(detector.width**2 + detector.length**2 + detector.depth**2)/2
	neutron_beam = Beam("neutron", BACKGROUND_NEUTRON_SPECTRUM, distance=world_radius, shape="ambient")
	neutron_response, _ = calculate_response(detector, neutron_beam, num_particles=num_neutrons)
	neutron_weight = BACKGROUND_FLUENCE*4*pi*world_radius**2/num_neutrons
	photon_beam = Beam("photon", BACKGROUND_PHOTON_SPECTRUM, distance=world_radius, shape="ambient")
	photon_response, _ = calculate_response(detector, photon_beam, num_particles=num_photons)
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
	plt.ylim(0, max(counts[i][energy_bins[1:] > detector.lower_threshold].max() for i in range(4))*1.05)
	plt.legend()
	plt.xlabel("Deposited energy (MeV)")
	plt.title(f"{detector.width:.1f} cm × {detector.length:.1f} cm × {detector.depth:.1f} cm {detector.material_name} detector")
	plt.tight_layout()
	filename = f"figures/{detector.material_name}_{detector.width:.1f}cmx{detector.length:.1f}cmx{detector.depth:.1f}cm_response.pdf"
	plt.savefig(filename)
	logging.info(f"Saved response plot to {filename}")


def find_pareto_front(material: str, optimistic: bool, spectrometric: bool) -> list[tuple[float, float, float, float, float, float, float]]:
	"""
	find the pareto front of designs with high sensitivity to signal and low sensitivity to background
	:param material: the material out of which the detector is made
	:param optimistic: whether we assume we can use pulse shape discrimination and coincidence subtraction
	:param spectrometric: whether to require that most electrons be fully stopped
	:return: a bunch of designs specified by their width (cm), length (cm), depth (cm),
	         lower threshold (MeV), upper threshold (MeV),
	         background sensitivity, and signal sensitivity
	"""
	os.makedirs("results", exist_ok=True)

	filename = f"results/pareto_{material}_{'optimistic' if optimistic else 'conservative'}_{'spectrometer' if spectrometric else 'detector'}.txt"
	try:
		results = loadtxt(filename, skiprows=1)
		logging.info(f"loaded pareto front from {filename}")
	except FileNotFoundError:
		logging.info(f"starting {'optimistic' if optimistic else 'conservative'} pareto front calculation for a {material} {'spectrometer' if spectrometric else 'detector'}...")
		signal_sensitivities = 1 - linspace(1, 0, 9)[1:-1]**2
		num_processes = min(len(signal_sensitivities), cpu_count())
		logging.debug(f"running on {num_processes} parallel processes")
		with Pool(processes=9) as executor:
			results = executor.map(
				optimize_detector_star,
				[(material, sensitivity, 0.5 if spectrometric else 0.0, optimistic, "any", 16.7) for sensitivity in signal_sensitivities],
			)
		savetxt(
			filename, results, delimiter="\t",
			header="width (cm)\tlength(cm)\tdepth (cm)\tlower threshold (MeV)\tupper threshold (MeV)\tbackground sensitivity\tsignal_sensitivity\n")
		logging.info(f"done!  saved to {filename}")

	return results


def optimize_detector_star(args: tuple[str, float, float, bool, str, float]):
	return optimize_detector(*args)


def optimize_detector(material: str, signal_sensitivity: float, spectroscopic_quality: float, optimistic: bool, mode: Literal["block", "slab", "strip", "any"], incident_energy: float) -> tuple[float, float, float, float, float, float, float]:
	"""
	get the optimal dimensions and thresholds for a detector of the given material with at least the given signal sensitivity
	:param material: the name of the active volume material
	:param signal_sensitivity: the required fraction of signal electrons that generate pulses within the thresholds
	:param spectroscopic_quality: the required fraction of electrons that we stop completely or almost completely
	:param optimistic: whether we assume we can use pulse shape discrimination and coincidence subtraction
	:param mode: which set of free parameters and constraints to use.  one of:
	             - "block" – a large chunky detector designed to fully stop the electrons, which will probably not get much spacial information
	             - "slab" – a deep but thin detector designed to get some spacial information and also get spectral information in combination with its neibors
	             - "strip" – a tiny detector designed to forsake spectral information and minimize background
	             - "any" – it will pick whichever one has the best backgroud performance
	:param incident_energy: the electron energy being optimized for (MeV)
	:return: the width (cm), the length (cm), the depth (cm), the lower threshold (MeV), the upper threshold (MeV), the achieved background sensitivity, and the achieved signal sensitivity
	"""
	coincidence_counting = optimistic
	pulse_shape_discrimination = optimistic and material.startswith("EJ")
	if mode == "any":
		solutions = []
		for new_mode in ["block", "slab", "strip"]:
			try:
				solutions.append(optimize_detector(material, signal_sensitivity, spectroscopic_quality, optimistic, new_mode, incident_energy))
			except RuntimeError:
				pass
		if len(solutions) == 0:
			raise RuntimeError("all of the optimizations failed.")
		else:
			return min(solutions, key=lambda solution: solution[-2])

	elif mode == "block":
		if material == "silicon":
			raise RuntimeError("silicon detectors can't be manufactured that thick.")

		lower_percentile = 100*(1 - signal_sensitivity)
		# constrain the thresholds
		result = optimize.minimize(
			lambda x: calculate_background_sensitivity(
				material, x[0], x[1], x[2], lower_percentile, 100, incident_energy,
				include_photons=True,
				include_neutrons=not pulse_shape_discrimination,
				include_crosstalk=not coincidence_counting),  # find the lowest background sensitivity
			constraints=[optimize.NonlinearConstraint(
				lambda x: calculate_thresholds(
					material, x[0], x[1], x[2], incident_energy, 100*(1 - spectroscopic_quality))[0],
				lb=incident_energy - .6, ub=inf,
			)],
			x0=[4.0, 14.0, 6.0],
			bounds=[
				(0.1, 10.0),
				(FOCAL_PLANE_HEIGHT, FOCAL_PLANE_HEIGHT + 10.0),
				(0.1, 10.0),
			],
			method="cobyqa",
			options=dict(
				initial_tr_radius=0.5,
				final_tr_radius=1.e-4,
			),
		)
		width, length, depth = result.x

	elif mode == "slab":
		if material == "silicon":
			raise ValueError("silicon detectors can't be manufactured that thick.")

		# optimize with fixed width
		width = 1.0
		# optimize with freely varying thickness and thresholds
		result = optimize.minimize(
			lambda x: calculate_background_sensitivity(
				material, width, x[1], x[2], x[3], x[3] + 100*signal_sensitivity, incident_energy,
				include_photons=True,
				include_neutrons=not pulse_shape_discrimination,
				include_crosstalk=not coincidence_counting),  # find the lowest background sensitivity
			constraints=[optimize.NonlinearConstraint(
				lambda x: calculate_thresholds(
					material, x[0], x[1], x[2], incident_energy, 100*(1 - spectroscopic_quality))[0],
				lb=incident_energy - .6, ub=inf,
			)],
			x0=[14.0, 4.0, 6.0],
			bounds=[
				(FOCAL_PLANE_HEIGHT, FOCAL_PLANE_HEIGHT + 10.0),
				(0.1, 10.0),
				(1., 100.*(1 - signal_sensitivity)),
			],
			method="cobyqa",
			options=dict(
				initial_tr_radius=0.5,
				final_tr_radius=1.e-4,
			),
		)
		width, length, depth, lower_percentile = result.x

	elif mode == "strip":
		if spectroscopic_quality > 0:
			raise RuntimeError("we can't make a spectrometer this thin; no way")
		# optimize with fixed thickness, length, and width
		width, length, depth = 0.1, 10.0, 0.1
		result = optimize.minimize_scalar(
			lambda x: calculate_background_sensitivity(
				material, width, length, depth, x, x + 100*signal_sensitivity, incident_energy,
				include_photons=True,
				include_neutrons=not pulse_shape_discrimination,
				include_crosstalk=not coincidence_counting),  # find the lowest background sensitivity
			bounds=(0., 100.*(1 - signal_sensitivity)),
			options=dict(
				xatol=1.e-4,
			),
		)
		lower_percentile = result.x

	else:
		raise ValueError(f"undefined mode, {mode!r}; what _is_ that?")

	upper_percentile = lower_percentile + 100*signal_sensitivity

	if not result.success:
		logging.info(f"the optimization failed for signal sensitivity of {signal_sensitivity:.3g}; {result.message}")
		raise RuntimeError(result.message)
	else:
		logging.info(f"after {result.nfev} steps, we found an optimum that achieves {signal_sensitivity:.3g} for signal, {result.fun:.3g} for background")
	lower_threshold, upper_threshold = calculate_thresholds(material, width, length, depth, incident_energy, lower_percentile, upper_percentile)
	return width, length, depth, lower_threshold, upper_threshold, result.fun, signal_sensitivity


def calculate_thresholds(
		material: str, width: float, length: float, depth: float, incident_energy: float, *percentile_values: float,
) -> tuple[float, ...]:
	"""
	the thresholds that achieve the given percentiles
	"""
	thresholds = []
	for i, percentile_value in enumerate(percentile_values):
		cache_key = (f"{material}, {width:.12g}, {length:.12g}, {depth:.12g}, "
		             f"{incident_energy:.12g}, {percentile_value:.12g}, thresholds")
		threshold = None
		# first, try to load it from the cache
		try:
			with open("results/cache.txt", mode="r") as file:
				for line in file.readlines():
					input_string, output_string = line.split(" -> ")
					if input_string == cache_key:
						threshold = float(output_string)
						break
		except FileNotFoundError:
			pass
		thresholds.append(threshold)

	if not any(threshold is None for threshold in thresholds):
		return tuple(thresholds)

	width = max(0.001, width)
	depth = max(0.001, depth)
	detector = Detector(
		material=material, width=width, length=length, depth=depth)
	beam = Beam("electron", tight_spectrum(incident_energy), width=width, height=FOCAL_PLANE_HEIGHT, shape="rectangular")
	energies, _ = calculate_response(detector, beam, num_particles=1_000_000)
	efficiency = MATERIAL_DATA[material]["efficiency"]

	def fraction_below(threshold):
		energy_uncertainty = sqrt(energies/efficiency)
		score = (threshold - energies)/energy_uncertainty
		probability_below = 1/2 + 1/2*erf(score/sqrt(2))  # approximate the Poisson distribution as Gaussian so that it's continuus
		return probability_below.sum()/energies.size

	for i, percentile_value in enumerate(percentile_values):
		if thresholds[i] is None:
			threshold = find_root(
				lambda threshold: 100*fraction_below(threshold) - percentile_value,
				bracket=(0.0, incident_energy + 1.0),
				x0=percentile(energies, percentile_value),
			)
			thresholds[i] = threshold

			if abs(threshold - percentile(energies, percentile_value)) > 1 or abs(percentile_value - 100*fraction_below(threshold)) > 1:
				bottom = min(percentile(energies, 1.), percentile(energies, percentile_value)*0.9, threshold*0.9)
				top = max(percentile(energies, 99.), percentile(energies, percentile_value)*1.1, threshold*1.1)
				plt.figure()
				plt.hist(energies, bins=linspace(bottom, top, 101))
				plt.axvline(percentile(energies, percentile_value), color="blue", label="initial gess")
				plt.axvline(threshold, color="orange", linestyle="--", label="final anser")
				plt.legend()
				plt.xlim(bottom, top)
				plt.ylim(0, None)
				plt.savefig(f"problem {percentile_value:.2f} density.pdf")
				plt.figure()
				xx = linspace(bottom, top, 201)
				cum = 100*array([fraction_below(x) for x in xx])
				plt.plot(xx, cum)
				plt.axhline(percentile_value)
				plt.axvline(percentile(energies, percentile_value), color="blue", label="initial gess")
				plt.axvline(threshold, color="orange", linestyle="--", label="final anser")
				plt.legend()
				plt.xlim(bottom, top)
				plt.ylim(0, 100)
				plt.savefig(f"problem {percentile_value:.2f} cumulative.pdf")
				logging.warning(f"something went wrong with the percentile calculation for {percentile_value:.2f}%.  I tried to save a plot to illustrate the issue.")

			os.makedirs("results", exist_ok=True)
			with open("results/cache.txt", mode="a") as file:
				cache_key = (f"{material}, {width:.12g}, {length:.12g}, {depth:.12g}, "
				             f"{incident_energy:.12g}, {percentile_value:.12g}, thresholds")
				file.write(f"{cache_key} -> {threshold}\n")

	return tuple(thresholds)


def calculate_background_sensitivity(
		material: str, width: float, length: float, depth: float, lower_percentile: float, upper_percentile: float, incident_energy: float,
		include_neutrons=True, include_photons=True, include_crosstalk=True,
		use_percentiles=True, num_background_particles=5_000_000,
) -> float:
	"""
	the background sensitivity of this detector assuming ambient neutrons and photons with a 1/E² spectrum
	"""
	width = max(0.001, width)
	depth = max(0.001, depth)
	if use_percentiles:
		lower_threshold, upper_threshold = calculate_thresholds(material, width, length, depth, lower_percentile, upper_percentile, incident_energy)
	else:
		lower_threshold, upper_threshold = lower_percentile, upper_percentile
	detector = Detector(
		material=material, width=width, length=length, depth=depth,
		lower_threshold=lower_threshold, upper_threshold=upper_threshold)
	world_radius = sqrt((3*width)**2 + length**2 + depth**2)/2
	neutron_beam = Beam("neutron", BACKGROUND_NEUTRON_SPECTRUM, distance=world_radius, shape="ambient")
	photon_beam = Beam("photon", BACKGROUND_PHOTON_SPECTRUM, distance=world_radius, shape="ambient")
	electron_beam = Beam("electron", tight_spectrum(incident_energy), width=width, height=FOCAL_PLANE_HEIGHT, shape="rectangular")
	total_detection_rate = 0.
	total_detection_rate_var = 0.
	if include_crosstalk:
		_, _, crosstalk_sensitivity, crosstalk_sensitivity_unc = calculate_sensitivity(detector, electron_beam, num_particles=1_000_000, use_cache=True)
		total_detection_rate += crosstalk_sensitivity
		total_detection_rate_var += crosstalk_sensitivity_unc**2
	if include_neutrons:
		neutron_sensitivity, neutron_sensitivity_unc, _, _ = calculate_sensitivity(detector, neutron_beam, num_particles=num_background_particles, use_cache=True)
		total_detection_rate += BACKGROUND_FLUENCE*4*pi*world_radius**2*neutron_sensitivity
		total_detection_rate_var += (BACKGROUND_FLUENCE*4*pi*world_radius**2*neutron_sensitivity_unc)**2
	if include_photons:
		photon_sensitivity, photon_sensitivity_unc, _, _ = calculate_sensitivity(detector, photon_beam, num_particles=num_background_particles, use_cache=True)
		total_detection_rate += BACKGROUND_FLUENCE*4*pi*world_radius**2*photon_sensitivity
		total_detection_rate_var += (BACKGROUND_FLUENCE*4*pi*world_radius**2*photon_sensitivity_unc)**2

	total_detection_rate_unc = sqrt(total_detection_rate_var)
	if total_detection_rate_unc > .10*total_detection_rate:
		logging.warning(
			f"when calculating the sensitivity of a {detector.width:.2g}×{detector.length:.2g}×{detector.depth:.2g} cm "
			f"{detector.material_name} detector to background, counting only particles between "
			f"{detector.lower_threshold:.2g} and {detector.upper_threshold:.2g} MeV, we got an unacceptably "
			f"uncertain anser of {total_detection_rate:.3g} ± {total_detection_rate_unc:.3g}.")

	return total_detection_rate


def tight_spectrum(central_energy: float, width=0.3) -> Spectrum:
	lower_bound, upper_bound = central_energy - width/2, central_energy + width/2
	return Spectrum(f"{lower_bound:.1f}–{upper_bound:.1f}", array([lower_bound, upper_bound]), array([1., 1.]))


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
	if x0 < 0.99*left + 0.01*right:
		x0 = 0.99*left + 0.01*right
	elif x0 > 0.01*left + 0.99*right:
		x0 = 0.01*left + 0.99*right
	# change one of the bounds to x0 to incorporate the initial gess into the search
	y0 = f(x0)
	if y0 < 0:
		left = x0
	elif y0 > 0:
		right = x0
	else:
		return x0
	# run Scipy's Brent algorithm
	solution = optimize.root_scalar(f, bracket=(left, right), **kwargs)
	if not solution.converged:
		logging.warning(f"root_scalar did not converge; it returned a result of {solution.flag}")
	return solution.root


def test_plot_responses():
	plot_responses(Detector("EJ-276D", width=2, length=10, depth=5, lower_threshold=5, upper_threshold=17), incident_energy=16.7)


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
	contours = ax.contour(x, y, signal_sensitivities.T, levels=[0.25, 0.5, 0.75, 0.875], colors="k")
	ax.clabel(contours)
	ax.set_xlabel(x_label)
	ax.set_ylabel(y_label)
	plt.colorbar(mesh, ticks=LogLocator().tick_values(vmin, vmax), extend="min").set_label("Background sensitivity")
	fig.tight_layout()


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--require-spectrometry", action="store_true")
	args = parser.parse_args()

	plot_pareto_fronts(
		["EJ-276D", "EJ-100", "LaBr3", "silicon"],
		{"EJ-276D": "C2.-", "EJ-100": "C2--", "LaBr3": "C0-", "silicon": "C1:"},
		args.require_spectrometry,
	)
	plt.show()
