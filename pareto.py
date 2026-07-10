from numpy import linspace, where

from detector import sensitivity, Detector
from simulation import Beam, Spectrum

if __name__ == "__main__":
	energy = linspace(1., 14, 14)
	background_spectrum = Spectrum(energy, energy**-2)

	detector = Detector(material="EJ-276", width=20., depth=100., lower_threshold=2.0)
	print("electrons:", sensitivity(detector, Beam("electron", 16.7, width=100.), num_particles=1_000_000))
	print("neutrons:", sensitivity(detector, Beam("neutron", background_spectrum, width=100., ambient=True), num_particles=1_000_000))
	print("photons:", sensitivity(detector, Beam("photon", background_spectrum, width=100., ambient=True), num_particles=1_000_000))
