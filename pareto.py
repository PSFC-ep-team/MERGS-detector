from detector import sensitivity, Detector
from simulation import Beam

if __name__ == "__main__":
	detector = Detector(material="EJ-276", width=20., depth=100., lower_threshold=2.0)
	print("electrons:", sensitivity(detector, Beam("electron", 16.7, width=20.)))
