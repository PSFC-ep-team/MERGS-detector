import matplotlib.pyplot as plt
from numpy import inf, array, empty

from detector import Detector
from pareto import LENGTH, calculate_background_sensitivity, plot_responses, calculate_thresholds


SIGNAL_RATE = 1  # electron/s
IGNORABLE_PULSE_HEIGHT = 0.1  # MeV


def compare():
	material = "EJ-276D"
	signal_sensitivities = array([.25, .50, .90])
	incident_energies = array([10, 12, 14, 16, 16.7, 18])
	optimistic_background_levels = {}
	conservative_background_levels = {}
	count_rates = {}
	for strategy in ["tiny", "large"]:
		optimistic_background_levels[strategy] = empty((signal_sensitivities.size, incident_energies.size))
		conservative_background_levels[strategy] = empty((signal_sensitivities.size, incident_energies.size))
		count_rates[strategy] = empty((signal_sensitivities.size, incident_energies.size))
		for i, signal_sensitivity in enumerate(signal_sensitivities):
			for j, incident_energy in enumerate(incident_energies):
				# determine the detector parameters
				if strategy == "tiny":
					width, depth = 0.1, 0.1
					lower_percentile, upper_percentile = 20*(1 - signal_sensitivity), 20 + 80*signal_sensitivity
				else:
					# width, depth, _, _ = optimize_detector(
					# 	material, signal_sensitivity, optimistic=True, spectrometric=True)
					width, depth = 6, 8
					lower_percentile, upper_percentile = 100*(1 - signal_sensitivity), 100
				plot = incident_energy == 16.7 and ((strategy == "tiny" and signal_sensitivity == .80) or (strategy == "large" and signal_sensitivity == .50))

				optimistic_background_level, conservative_background_level, count_rate = evaluate_detector(
					material, width, depth, lower_percentile, upper_percentile, incident_energy, plot=plot,
				)
				optimistic_background_levels[strategy][i, j] = optimistic_background_level
				conservative_background_levels[strategy][i, j] = conservative_background_level
				count_rates[strategy][i, j] = count_rate

	for i, signal_sensitivity in enumerate(signal_sensitivities):
		fig, axs = plt.subplots(nrows=2, ncols=1, sharex="all", gridspec_kw=dict(hspace=0), figsize=(6, 6))
		axs[0].set_title(f"Counting {signal_sensitivity:.0%} of electrons")
		axs[0].fill_between(incident_energies, optimistic_background_levels["large"][i], conservative_background_levels["large"][i], facecolor="C1", edgecolor="none", alpha=1/4, label="Large detector")
		axs[0].fill_between(incident_energies, optimistic_background_levels["tiny"][i], conservative_background_levels["tiny"][i], facecolor="C0", edgecolor="none", alpha=1/4, label="Tiny detector")
		axs[0].plot(incident_energies, conservative_background_levels["large"][i], "C1-", label="Large detector")
		axs[0].plot(incident_energies, optimistic_background_levels["large"][i], "C1--")
		axs[0].plot(incident_energies, conservative_background_levels["tiny"][i], "C0-")
		axs[0].plot(incident_energies, optimistic_background_levels["tiny"][i], "C0--")
		axs[0].grid()
		axs[0].xaxis.set_visible(False)
		axs[0].set_yscale("log")
		axs[0].legend()
		axs[0].set_ylabel("Background/signal ratio")
		axs[1].plot(incident_energies, count_rates["tiny"][i], "C0-")
		axs[1].plot(incident_energies, count_rates["large"][i], "C1-")
		axs[1].grid()
		axs[1].set_yscale("log")
		axs[1].set_ylabel("Count rate (cps)")
		axs[1].set_xlim(10, 18)
		axs[1].set_xlabel("Electron energy (MeV)")
		fig.savefig(f"figures/comparison-{material}-{signal_sensitivity*100:.0f}.pdf")

	plt.show()


def evaluate_detector(material: str, width: float, depth: float, lower_percentile: float, upper_percentile: float, incident_energy: float, plot=False) -> tuple[float, float, float]:
	""" calculate the range of possible background levels (per signal particle) and the total count rate of this detector """
	lower_threshold, upper_threshold = calculate_thresholds(material, width, depth, lower_percentile, upper_percentile, incident_energy)
	num_background_particles = round(10_000_000/(width*depth*LENGTH)**(1/3))
	optimistic_background = calculate_background_sensitivity(
		material, width, depth, lower_threshold, upper_threshold, incident_energy,
		include_photons=True, include_neutrons=False, include_crosstalk=False,
		num_background_particles=num_background_particles,
		use_percentiles=False)
	conservative_background = optimistic_background + calculate_background_sensitivity(
		material, width, depth, lower_threshold, upper_threshold, incident_energy,
		include_photons=False, include_neutrons=True, include_crosstalk=True,
		num_background_particles=num_background_particles,
		use_percentiles=False)
	count_rate = SIGNAL_RATE*calculate_background_sensitivity(
		material, width, depth, IGNORABLE_PULSE_HEIGHT, inf, incident_energy,
		include_photons=True, include_neutrons=True, include_crosstalk=True,
		num_background_particles=num_background_particles,
		use_percentiles=False)

	if plot:
		plot_responses(
			Detector(
				material, width, depth,
				lower_threshold=lower_threshold, upper_threshold=upper_threshold,
			),
			num_background_particles=num_background_particles,
			incident_energy=incident_energy,
		)
		print(f"{width:.1f} cm × {depth:.1f} cm detector: B/S = {optimistic_background:.2g}–{conservative_background:.2g}; total signal rate = {count_rate:.2g} cps")

	return optimistic_background, conservative_background, count_rate


if __name__ == "__main__":
	compare()