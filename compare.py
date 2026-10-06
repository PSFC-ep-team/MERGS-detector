from concurrent.futures import ProcessPoolExecutor
import logging
import os

import matplotlib.pyplot as plt
from numpy import inf, array, empty

from detector import Detector
from pareto import calculate_background_sensitivity, plot_responses, calculate_thresholds, optimize_detector

SIGNAL_RATE = 1  # electron/s
IGNORABLE_PULSE_HEIGHT = 0.1  # MeV



def compare():
	material = "EJ-276D"
	modes = ["strip", "slab", "block"]
	signal_sensitivities = array([.25, .50, .90])
	incident_energies = array([10, 12, 14, 16, 16.7, 18])

	logging.info(f"designing {len(modes)*len(signal_sensitivities)*len(incident_energies)} {material} detectors...")
	detector_designs = {mode: [None]*len(signal_sensitivities) for mode in modes}
	for mode in modes:
		for i, signal_sensitivity in enumerate(signal_sensitivities):
			num_processes = min(len(incident_energies), os.cpu_count())
			with ProcessPoolExecutor(max_workers=num_processes) as pool:
				# determine the detector parameters
				detector_designs[mode][i] = pool.map(
					optimize_detector_star,
					[(material, signal_sensitivity, .5 if mode != "strip" else .0, True, mode, energy) for energy in incident_energies],
				)
	logging.info(f"all jobs submitted")

	logging.info(f"waiting for and evaluating {len(modes)*len(signal_sensitivities)*len(incident_energies)} {material} detectors...")
	optimistic_background_levels = empty(
		(signal_sensitivities.size, incident_energies.size),
		dtype=[(mode, float) for mode in modes])
	conservative_background_levels = empty(
		(signal_sensitivities.size, incident_energies.size),
		dtype=[(mode, float) for mode in modes])
	count_rates = empty(
		(signal_sensitivities.size, incident_energies.size),
		dtype=[(mode, float) for mode in modes])
	for mode in modes:
		for i, signal_sensitivity in enumerate(signal_sensitivities):
			for j, incident_energy in enumerate(incident_energies):
				try:
					width, length, depth, lower_percentile, upper_percentile, _, _ = detector_designs[mode][i].__next__()
				except RuntimeError:
					optimistic_background_level, conservative_background_level, count_rate = inf, inf, inf
				else:
					plot = incident_energy == 16.7 and signal_sensitivity == .80
					optimistic_background_level, conservative_background_level, count_rate = evaluate_detector(
						material, width, length, depth, lower_percentile, upper_percentile, incident_energy, plot=plot,
					)
				optimistic_background_levels[mode][i, j] = optimistic_background_level
				conservative_background_levels[mode][i, j] = conservative_background_level
				count_rates[mode][i, j] = count_rate
			logging.info(f"done with {signal_sensitivity:.0%} {mode}s")

	logging.info(f"plotting {len(modes)*len(signal_sensitivities)*len(incident_energies)} {material} detectors...")
	for i, signal_sensitivity in enumerate(signal_sensitivities):
		plot_detector_concept(
			material, signal_sensitivity, incident_energies,
			optimistic_background_levels[i, :],
			conservative_background_levels[i, :],
			count_rates[i, :])

	logging.info(f"done!")
	plt.show()


def optimize_detector_star(args):
	return optimize_detector(*args)


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


if __name__ == "__main__" or __name__ == "__mp_main__":
	os.makedirs("results", exist_ok=True)
	logging.basicConfig(
		level=logging.DEBUG, filename="results/compare.log", encoding="utf-8",
		datefmt="%m-%d %H:%M:%S", format="%(asctime)s %(levelname)-5.5s %(message)s")
	logging.getLogger().addHandler(logging.StreamHandler())
	logging.getLogger("filelock").setLevel(logging.WARNING)
	logging.getLogger("matplotlib").setLevel(logging.WARNING)
	logging.getLogger("PIL").setLevel(logging.WARNING)

if __name__ == "__main__":
	compare()
