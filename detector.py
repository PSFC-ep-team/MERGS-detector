from __future__ import annotations

from numpy import count_nonzero, inf

from data import MATERIAL_DATA
from simulation import Beam, simulate, Solid


def plot_sensitivity_curves(detector: Detector) -> None:
	""" calculate the sensitivity of a detector to all types and energies of radiation """
	pass


def sensitivity(detector: Detector, particle: Beam) -> float:
	""" calculate the fraction of these incident particles that are detected by this detector """
	tracks = simulate(
		detector.material_name,
		[Solid(
			"box",
			x=detector.width, y=100, z=detector.depth,
			x_position=0, y_position=0, z_position=0,
		)],
		particle)
	return count_nonzero(
		(tracks["E_deposited"] >= detector.lower_threshold) &
		(tracks["E_deposited"] <= detector.upper_threshold)
	)/tracks.size


class Detector:
	def __init__(self, material: str, width: float, depth: float, lower_threshold=0., upper_threshold=inf):
		"""
		a single channel of an electron detector
		:param material: the name of the detection material
		:param width: the scale of the detector in the dispersive direction (mm)
		:param depth: the scale of the detector in the beam direction (mm)
		:param lower_threshold: the minimum amount of energy in a pulse to be detected (MeV)
		:param upper_threshold: the maximum amount of energy in a pulse to be detected (MeV)
		"""
		self.material_name = material
		self.density = MATERIAL_DATA[material]["density"]  # g/cm³
		self.elements = MATERIAL_DATA[material]["elements"]
		self.width = width
		self.depth = depth
		self.lower_threshold = lower_threshold
		self.upper_threshold = upper_threshold


if __name__ == "__main__":
	print(sensitivity(Detector("LaBr₃", 10, 30, lower_threshold=8.25), Beam("electron", 16.5)))
