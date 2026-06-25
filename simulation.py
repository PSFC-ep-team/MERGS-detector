from __future__ import annotations

import os
import subprocess
import xml.etree.ElementTree as xml

from numpy import inf, genfromtxt, count_nonzero

from data import PARTICLE_DATA, MATERIAL_DATA, ELEMENT_DATA


def moliere_radius(material: str) -> float:
	""" calculate the Moliere radius of a material """
	pass


def plot_sensitivity_curves(detector: Detector) -> None:
	""" calculate the sensitivity of a detector to all types and energies of radiation """
	pass


def sensitivity(detector: Detector, particle: Beam) -> float:
	""" calculate the fraction of these incident particles that are detected by this detector """
	tracks = simulate(detector, particle)
	return count_nonzero(
		(tracks["E_deposited"] >= detector.lower_threshold) &
		(tracks["E_deposited"] <= detector.upper_threshold)
	)/tracks.size


def simulate(detector: Detector, particle: Beam):
	""" run a Geant4 simulation of a beam of these particles hitting this detector """
	input_deck = xml.Element("gdml", {
		"xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
		"xsi:noNamespaceSchemaLocation": os.path.expanduser("~/grasshopper/schema/gdml.xsd"),
	})

	# specify the detector material
	materials = xml.SubElement(input_deck, "materials")
	for element in detector.elements.keys() | {"N"}:
		element_info = xml.SubElement(
			materials, "element", Z=f"{ELEMENT_DATA[element][0]}", name=element)
		xml.SubElement(
			element_info, "atom", value=f"{ELEMENT_DATA[element][1]}", unit="g/mole")
	material_info = xml.SubElement(
		materials, "material", name=detector.material_name, state="solid")
	xml.SubElement(material_info, "D", value=f"{detector.density}", unit="g/cm3")
	for element, abundance in detector.elements.items():
		xml.SubElement(material_info, "composite", ref=element, n=f"{abundance}")
	material_info = xml.SubElement(
		materials, "material", name="vacuum", state="gas")
	xml.SubElement(material_info, "D", value="1e-25", unit="g/cm3")
	xml.SubElement(material_info, "composite", ref="N", n="1.0")

	# I don't know what any of these settings are
	definitions = xml.SubElement(input_deck, "define")
	# output settings?
	xml.SubElement(definitions, "constant", name="TextOutputOn", value="1")
	xml.SubElement(definitions, "constant", name="BriefOutputOn", value="0")
	xml.SubElement(definitions, "constant", name="VRMLvisualizationOn", value="0")
	xml.SubElement(definitions, "constant", name="EventsToAccumulate", value="100")
	# physics cuts?
	xml.SubElement(definitions, "constant", name="LightProducingParticle", value="0")
	xml.SubElement(definitions, "constant", name="LowEnergyCutoff", value="0")
	xml.SubElement(definitions, "constant", name="KeepOnlyMainParticle", value="0")
	xml.SubElement(definitions, "quantity",
	               name="ProductionLowLimit", type="threshold", value="1", unit="keV")
	# output filters?
	xml.SubElement(definitions, "constant", name="SaveSurfaceHitTrack", value="0")
	xml.SubElement(definitions, "constant", name="SaveTrackInfo", value="0")
	xml.SubElement(definitions, "constant", name="SaveEdepositedTotalEntry", value="1")
	# bean definition?
	xml.SubElement(definitions, "constant", name="RandomGenSeed", value="0")
	xml.SubElement(definitions, "quantity",
	               name="BeamOffsetX", type="coordinate", value="0", unit="mm")
	xml.SubElement(definitions, "quantity",
	               name="BeamOffsetY", type="coordinate", value="0", unit="mm")
	xml.SubElement(definitions, "quantity",
	               name="BeamOffsetZ", type="coordinate", value="-100", unit="mm")
	xml.SubElement(definitions, "quantity",
	               name="BeamSize", type="coordinate", value="0", unit="mm")
	xml.SubElement(definitions, "quantity",
	               name="BeamEnergy", type="energy", value=f"{particle.energy}", unit="MeV")
	xml.SubElement(definitions, "constant", name="EventsToRun", value="10000")
	xml.SubElement(definitions, "constant", name="ParticleNumber", value=f"{particle.number}")

	# specify the detector geometry
	solids = xml.SubElement(input_deck, "solids")
	xml.SubElement(solids, "box", name="detector",
	               x=f"{detector.width}", y="100", z=f"{detector.depth}", lunit="mm")
	xml.SubElement(solids, "box", name="infinite_void",
	               x="300", y="300", z="300", lunit="mm")

	# fill in the remaining information
	structure = xml.SubElement(input_deck, "structure")
	detector_volume = xml.SubElement(structure, "volume", name="detector_log")
	xml.SubElement(detector_volume, "materialref", ref=detector.material_name)
	xml.SubElement(detector_volume, "solidref", ref="detector")
	world_volume = xml.SubElement(structure, "volume", name="world_log")
	xml.SubElement(world_volume, "materialref", ref="vacuum")
	xml.SubElement(world_volume, "solidref", ref="infinite_void")
	detector_specification = xml.SubElement(world_volume, "physvol", name="det_phys69")
	xml.SubElement(detector_specification, "volumeref", ref="detector_log")
	xml.SubElement(detector_specification, "position", name="detector_pos", x="0", y="0", z="0", unit="mm")

	# and then whatever this is
	setup = xml.SubElement(input_deck, "setup", name="Default", version="1.0")
	xml.SubElement(setup, "world", ref="world_log")

	# write to disc
	os.makedirs("run", exist_ok=True)
	tree = xml.ElementTree(input_deck)
	xml.indent(tree)
	tree.write("run/input.gdml", xml_declaration=True, encoding="UTF-8")

	subprocess.run(["grasshopper", "input.gdml", "output"], cwd="run")

	columns = [  # normally genfromtxt would be able to do this automaticly but Areg made the header wrong so I have to manually specify what the columns actually are
		("E_beam", float), ("E_incident", float), ("E_deposited", float),
		("x_incident", float), ("y_incident", float), ("z_incident", float), ("theta", float),
		("Time", float), ("EventID", int), ("TrackID", int), ("ParticleID", int),
		("ParticleName", str), ("CreatorProcessName", str),
		("IsEdepositedTotalEntry", int), ("IsSurfaceHitTrack", int),
	]
	output_data = genfromtxt("run/output.dat", skip_header=1, dtype=columns)
	return output_data


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


class Beam:
	def __init__(self, particle: str, energy: float):
		"""
		a type of radiation
		:param particle: the name of the particle
		:param energy: the energy of each particle (MeV)
		"""
		self.particle_name = particle
		self.rest_mass = PARTICLE_DATA[particle]["rest_mass"]  # MeV/c²
		self.charge = PARTICLE_DATA[particle]["charge"]  # e
		self.number = PARTICLE_DATA[particle]["number"]
		self.energy = energy


if __name__ == "__main__":
	print(sensitivity(Detector("LaBr₃", 10, 30, lower_threshold=8.25), Beam("electron", 16.5)))
