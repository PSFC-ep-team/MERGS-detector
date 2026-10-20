from concurrent.futures import ProcessPoolExecutor
import logging
import os
from typing import Literal

import matplotlib.pyplot as plt
from numpy import inf, array, empty, interp, loadtxt

from detector import Detector
from pareto import calculate_background_sensitivity, plot_responses, calculate_thresholds, optimize_detector, \
	ELECTRON_RATE

plt.rcParams["font.size"] = 11

IGNORABLE_PULSE_HEIGHT = 0.1  # MeV

_, spectrum_0 = loadtxt("data/gamma0_spectrum_interpol.dat", unpack=True)
spectrum_energy, spectrum_1 = loadtxt("data/gamma1_spectrum_interpol.dat", unpack=True)
spectrum = spectrum_0 + spectrum_1


def compare():
	material = "EJ-276D"
	modes = ["strip", "block"]
	signal_sensitivities = array([.25, .50, .90])
	incident_energies = array([10, 12, 14, 16, 16.7, 18])

	logging.info(f"designing {len(modes)*len(signal_sensitivities)*len(incident_energies)} {material} detectors...")
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
			num_processes = min(len(incident_energies), os.cpu_count())
			for optimistic in [True, False]:
				# determine the detector parameters
				logging.info(f"launching {num_processes} simultaneus {'optimistic' if optimistic else 'conservative'} {signal_sensitivity:.0%} {mode} optimizations...")
				with ProcessPoolExecutor(max_workers=num_processes) as pool:
					results = []
					for energy in incident_energies:
						results.append(pool.submit(
							optimize_and_evaluate_detector,
							material, signal_sensitivity, .5 if mode != "strip" else .0, optimistic, mode, energy,
						))
					for j, (energy, result) in enumerate(zip(incident_energies, results)):
						try:
							background_level, count_rate = result.result(timeout=1800)
						except RuntimeError:
							background_level, count_rate = inf, inf
						except TimeoutError:
							logging.warning("I think one of the optimizations may have crashed.  I'm sorry; I don't know what to do about that.")
							background_level, count_rate = inf, inf
						if optimistic:
							optimistic_background_levels[mode][i, j] = background_level
							count_rates[mode][i, j] = count_rate
						else:
							conservative_background_levels[mode][i, j] = background_level
				logging.info(f"done with {'optimistic' if optimistic else 'conservative'} {signal_sensitivity:.0%} {mode}s!")
	logging.info(f"all jobs completed!")

	logging.info(f"plotting summary of {len(modes)*len(signal_sensitivities)*len(incident_energies)} {material} detectors...")
	for i, signal_sensitivity in enumerate(signal_sensitivities):
		plot_detector_concept(
			material, signal_sensitivity, incident_energies,
			optimistic_background_levels[i, :],
			conservative_background_levels[i, :],
			count_rates[i, :])

	logging.info(f"done!")
	plt.show()


def optimize_and_evaluate_detector(material: str, signal_sensitivity: float, spectrometric_quality: float, optimistic: bool, mode: Literal["any", "strip", "block"], energy: float) -> tuple[float, float]:
	width, length, depth, lower_percentile, upper_percentile, _, _ = optimize_detector(material, signal_sensitivity, spectrometric_quality, optimistic, mode, energy)
	return evaluate_detector(
		material, width, length, depth, lower_percentile, upper_percentile, energy, optimistic=optimistic,
		plot=energy == 16.7 and signal_sensitivity == .80 and optimistic)


def plot_detector_concept(
		material, signal_sensitivity, incident_energies,
		optimistic_background_levels, conservative_background_levels, count_rates):
	fig, axs = plt.subplots(nrows=2, ncols=1, sharex="all", gridspec_kw=dict(hspace=0), figsize=(5.0, 5.0))

	for mode, color, label in [("block", "#dfa11a", "Large detector"), ("strip", "#3382d7", "Strip detector")]:
		axs[0].plot(incident_energies, count_rates[mode], color, label=label)
	axs[0].grid()
	axs[0].legend()
	axs[0].set_yscale("log")
	axs[0].set_ylabel("Total count rate (s⁻¹)")
	axs[0].xaxis.set_ticks_position("none")

	for mode, color in [("block", "#dfa11a"), ("strip", "#3382d7")]:
		axs[1].fill_between(
			incident_energies,
			ELECTRON_RATE*optimistic_background_levels[mode],
			ELECTRON_RATE*conservative_background_levels[mode],
			facecolor=color, edgecolor="none", alpha=1/4)
	for mode, color, label in [("block", "#dfa11a", "Large detector"), ("strip", "#3382d7", "Strip detector")]:
		axs[1].plot(
			incident_energies,
			ELECTRON_RATE*conservative_background_levels[mode],
			color, linewidth=1, label=f"{label} background")
		axs[1].plot(
			incident_energies,
			ELECTRON_RATE*optimistic_background_levels[mode],
			color, linewidth=1)
	axs[1].plot(
		spectrum_energy,
		ELECTRON_RATE*signal_sensitivity*spectrum/spectrum.max(),
		color="k", linewidth=1.5, linestyle="dashed", label="Signal")
	axs[1].grid()
	axs[1].legend()
	axs[1].set_yscale("log")
	axs[1].set_ylabel("Thresholded count rate (s⁻¹)")
	axs[1].set_xlim(10, 18)
	axs[1].set_xlabel("Electron energy (MeV)")

	fig.tight_layout()
	fig.savefig(f"figures/comparison-{material}-{signal_sensitivity*100:.0f}.pdf")


def evaluate_detector(material: str, width: float, length: float, depth: float, lower_percentile: float, upper_percentile: float, incident_energy: float, optimistic: bool, plot=False) -> tuple[float, float]:
	""" calculate the range of possible background levels (per signal particle) and the total count rate of this detector """
	lower_threshold, upper_threshold = calculate_thresholds(material, width, length, depth, incident_energy, lower_percentile, upper_percentile)
	num_background_particles = round(10_000_000/(width*depth*length)**(1/3))
	background = calculate_background_sensitivity(
		material, width, length, depth, lower_threshold, upper_threshold, incident_energy,
		include_photons=True, include_neutrons=not optimistic, include_crosstalk=not optimistic,
		num_background_particles=num_background_particles,
		use_percentiles=False)
	count_rate = ELECTRON_RATE*(1 + calculate_background_sensitivity(
		material, width, length, depth, IGNORABLE_PULSE_HEIGHT, inf, incident_energy,
		include_photons=True, include_neutrons=True, include_crosstalk=True,
		num_background_particles=num_background_particles,
		use_percentiles=False))

	if plot:
		logging.debug("plotting the histogram for a detector...")
		plot_responses(
			Detector(
				material, width, length, depth,
				lower_threshold=lower_threshold, upper_threshold=upper_threshold,
			),
			num_background_particles=num_background_particles,
			incident_energy=incident_energy,
		)
		logging.info(f"{width:.1f} cm × {depth:.1f} cm detector: B/S = {background:.2g}; total signal rate = {count_rate:.2g} cps")

	return background, count_rate


if __name__ == "__main__" or __name__ == "__mp_main__":
	os.makedirs("results", exist_ok=True)
	logging.basicConfig(
		level=logging.DEBUG, filename="results/compare.log", encoding="utf-8",
		datefmt="%m-%d %H:%M:%S", format="%(asctime)s %(process)-05d %(levelname)-5.5s %(message)s")
	logging.getLogger().addHandler(logging.StreamHandler())
	logging.getLogger("filelock").setLevel(logging.WARNING)
	logging.getLogger("matplotlib").setLevel(logging.WARNING)
	logging.getLogger("PIL").setLevel(logging.WARNING)

if __name__ == "__main__":
	compare()
