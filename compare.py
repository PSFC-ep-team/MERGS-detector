from multiprocessing import Pool, cpu_count

import matplotlib.pyplot as plt
from numpy import inf, array, empty

from detector import Detector
from pareto import calculate_background_sensitivity, plot_responses, calculate_thresholds, optimize_detector

SIGNAL_RATE = 1  # electron/s
IGNORABLE_PULSE_HEIGHT = 0.1  # MeV


def compare():
	material = "EJ-276D"
	signal_sensitivities = array([.25, .50, .90])
	incident_energies = array([10, 12, 14, 16, 16.7, 18])
	optimistic_background_levels = empty(
		(signal_sensitivities.size, incident_energies.size),
		dtype=[("strip", float), ("slab", float), ("block", float)])
	conservative_background_levels = empty(
		(signal_sensitivities.size, incident_energies.size),
		dtype=[("strip", float), ("slab", float), ("block", float)])
	count_rates = empty(
		(signal_sensitivities.size, incident_energies.size),
		dtype=[("strip", float), ("slab", float), ("block", float)])
	for mode in ["strip", "slab", "block"]:
		for i, signal_sensitivity in enumerate(signal_sensitivities):
			num_processes = min(len(signal_sensitivities), cpu_count())
			with Pool(processes=num_processes) as pool:
				# determine the detector parameters
				results = pool.map(
					evaluate_detector_concept_star,
					[(material, signal_sensitivity, mode, energy) for energy in incident_energies],
				)
			for j, (optimistic_background_level, conservative_background_level, count_rate) in enumerate(results):
				optimistic_background_levels[mode][i, j] = optimistic_background_level
				conservative_background_levels[mode][i, j] = conservative_background_level
				count_rates[mode][i, j] = count_rate

	for i, signal_sensitivity in enumerate(signal_sensitivities):
		plot_detector_concept(
			material, signal_sensitivity, incident_energies,
			optimistic_background_levels[i, :],
			conservative_background_levels[i, :],
			count_rates[i, :])
	plt.show()


def evaluate_detector_concept_star(args):
	return evaluate_detector_concept(*args)


def evaluate_detector_concept(material, signal_sensitivity, mode, incident_energy):
	try:
		width, length, depth, lower_percentile, upper_percentile, _, _ = optimize_detector(
			material, signal_sensitivity, mode=mode,
			spectroscopic_quality=.5 if mode != "strip" else .0, optimistic=True,
			incident_energy=incident_energy,
		)
	except RuntimeError:
		return inf, inf, inf
	else:
		plot = incident_energy == 16.7 and signal_sensitivity == .80
		return evaluate_detector(
			material, width, length, depth, lower_percentile, upper_percentile, incident_energy, plot=plot,
		)


def plot_detector_concept(
		material, signal_sensitivity, incident_energies,
		optimistic_background_levels, conservative_background_levels, count_rates):
	fig, axs = plt.subplots(nrows=2, ncols=1, sharex="all", gridspec_kw=dict(hspace=0), figsize=(6, 6))
	axs[0].set_title(f"Counting {signal_sensitivity:.0%} of electrons")
	axs[0].fill_between(incident_energies, optimistic_background_levels["block"], conservative_background_levels["block"], facecolor="C1", edgecolor="none", alpha=1/4)
	axs[0].fill_between(incident_energies, optimistic_background_levels["slab"], conservative_background_levels["slab"], facecolor="C2", edgecolor="none", alpha=1/4)
	axs[0].fill_between(incident_energies, optimistic_background_levels["strip"], conservative_background_levels["strip"], facecolor="C0", edgecolor="none", alpha=1/4)
	axs[0].plot(incident_energies, conservative_background_levels["block"], "C1-", label="Large detector")
	axs[0].plot(incident_energies, optimistic_background_levels["block"], "C1--")
	axs[0].plot(incident_energies, conservative_background_levels["slab"], "C2--", label="Deep detector")
	axs[0].plot(incident_energies, optimistic_background_levels["slab"], "C2--")
	axs[0].plot(incident_energies, conservative_background_levels["strip"], "C0-", label="Tiny detector")
	axs[0].plot(incident_energies, optimistic_background_levels["strip"], "C0--")
	axs[0].grid()
	axs[0].xaxis.set_visible(False)
	axs[0].set_yscale("log")
	axs[0].legend()
	axs[0].set_ylabel("Background/signal ratio")
	axs[1].plot(incident_energies, count_rates["block"], "C1-")
	axs[1].plot(incident_energies, count_rates["slab"], "C2-")
	axs[1].plot(incident_energies, count_rates["strip"], "C0-")
	axs[1].grid()
	axs[1].set_yscale("log")
	axs[1].set_ylabel("Count rate (cps)")
	axs[1].set_xlim(10, 18)
	axs[1].set_xlabel("Electron energy (MeV)")
	fig.savefig(f"figures/comparison-{material}-{signal_sensitivity*100:.0f}.pdf")


def evaluate_detector(material: str, width: float, length: float, depth: float, lower_percentile: float, upper_percentile: float, incident_energy: float, plot=False) -> tuple[float, float, float]:
	""" calculate the range of possible background levels (per signal particle) and the total count rate of this detector """
	lower_threshold, upper_threshold = calculate_thresholds(material, width, length, depth, incident_energy, lower_percentile, upper_percentile)
	num_background_particles = round(10_000_000/(width*depth*length)**(1/3))
	optimistic_background = calculate_background_sensitivity(
		material, width, length, depth, lower_threshold, upper_threshold, incident_energy,
		include_photons=True, include_neutrons=False, include_crosstalk=False,
		num_background_particles=num_background_particles,
		use_percentiles=False)
	conservative_background = optimistic_background + calculate_background_sensitivity(
		material, width, length, depth, lower_threshold, upper_threshold, incident_energy,
		include_photons=False, include_neutrons=True, include_crosstalk=True,
		num_background_particles=num_background_particles,
		use_percentiles=False)
	count_rate = SIGNAL_RATE*calculate_background_sensitivity(
		material, width, length, depth, IGNORABLE_PULSE_HEIGHT, inf, incident_energy,
		include_photons=True, include_neutrons=True, include_crosstalk=True,
		num_background_particles=num_background_particles,
		use_percentiles=False)

	if plot:
		plot_responses(
			Detector(
				material, width, depth, length,
				lower_threshold=lower_threshold, upper_threshold=upper_threshold,
			),
			num_background_particles=num_background_particles,
			incident_energy=incident_energy,
		)
		print(f"{width:.1f} cm × {depth:.1f} cm detector: B/S = {optimistic_background:.2g}–{conservative_background:.2g}; total signal rate = {count_rate:.2g} cps")

	return optimistic_background, conservative_background, count_rate


if __name__ == "__main__":
	compare()