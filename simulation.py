""" a function to facilitate building simple Grasshopper input decks """

from __future__ import annotations

import os
import subprocess
import xml.etree.ElementTree as xml

from numpy import genfromtxt
from numpy.typing import NDArray

from data import PARTICLE_DATA, MATERIAL_DATA, ELEMENT_DATA


def simulate(detector_material: str, solids: list[Solid], beam: Beam) -> NDArray:
	""" run a Geant4 simulation of a beam of these particles hitting a detector """
	input_deck = xml.Element("gdml", {
		"xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
		"xsi:noNamespaceSchemaLocation": os.path.expanduser("~/grasshopper/schema/gdml.xsd"),
	})

	# first apply detector_material to every detector solid
	for solid in solids:
		if solid.material == "detector":
			solid.material = detector_material

	# specify the materials
	for material in {solid.material for solid in solids}:
		density = MATERIAL_DATA[material]["density"]
		elements = MATERIAL_DATA[material]["elements"]
		materials = xml.SubElement(input_deck, "materials")
		for element in elements.keys() | {"N"}:
			element_info = xml.SubElement(
				materials, "element", Z=f"{ELEMENT_DATA[element][0]}", name=element)
			xml.SubElement(
				element_info, "atom", value=f"{ELEMENT_DATA[element][1]}", unit="g/mole")
		material_info = xml.SubElement(
			materials, "material", name=material, state="solid")
		xml.SubElement(material_info, "D", value=f"{density}", unit="g/cm3")
		for element, abundance in elements.items():
			xml.SubElement(material_info, "composite", ref=element, n=f"{abundance}")
	material_info = xml.SubElement(
		materials, "material", name="vacuum", state="gas")
	xml.SubElement(material_info, "D", value="1e-25", unit="g/cm3")
	xml.SubElement(material_info, "composite", ref="N", n="1.0")

	# some miscillaneus settings
	definitions = xml.SubElement(input_deck, "define")
	# output settings
	xml.SubElement(definitions, "constant", name="TextOutputOn", value="1")
	xml.SubElement(definitions, "constant", name="BriefOutputOn", value="0")
	xml.SubElement(definitions, "constant", name="VRMLvisualizationOn", value="1")
	xml.SubElement(definitions, "constant", name="EventsToAccumulate", value="100")
	# particle selections
	xml.SubElement(definitions, "constant", name="LightProducingParticle", value="0")
	xml.SubElement(definitions, "constant", name="LowEnergyCutoff", value="0")
	xml.SubElement(definitions, "constant", name="KeepOnlyMainParticle", value="0")
	xml.SubElement(definitions, "quantity",
	               name="ProductionLowLimit", type="threshold", value="1", unit="keV")
	# output filters
	xml.SubElement(definitions, "constant", name="SaveSurfaceHitTrack", value="0")
	xml.SubElement(definitions, "constant", name="SaveTrackInfo", value="1")
	xml.SubElement(definitions, "constant", name="SaveEdepositedTotalEntry", value="0")
	# bean definition
	xml.SubElement(definitions, "constant", name="RandomGenSeed", value="0")
	xml.SubElement(definitions, "quantity",
	               name="BeamOffsetX", type="coordinate", value="0", unit="mm")
	xml.SubElement(definitions, "quantity",
	               name="BeamOffsetY", type="coordinate", value="0", unit="mm")
	xml.SubElement(definitions, "quantity",
	               name="BeamOffsetZ", type="coordinate", value="-100", unit="mm")
	xml.SubElement(definitions, "quantity",
	               name="BeamSize", type="coordinate", value=f"{beam.width}", unit="mm")
	xml.SubElement(definitions, "quantity",
	               name="BeamEnergy", type="energy", value=f"{beam.energy}", unit="MeV")
	xml.SubElement(definitions, "constant", name="EventsToRun", value="10000")
	xml.SubElement(definitions, "constant", name="ParticleNumber", value=f"{beam.number}")

	# specify the geometry
	solid_group = xml.SubElement(input_deck, "solids")
	for i, solid in enumerate(solids):
		xml.SubElement(solid_group, solid.kind, name=f"solid{i}",
		               lunit="mm", **{key: f"{value}" for key, value in solid.kwargs.items()})
	xml.SubElement(solid_group, "box", name="infinite_void",
	               x="300", y="300", z="300", lunit="mm")

	# fill in the remaining information
	structure = xml.SubElement(input_deck, "structure")
	for i, solid in enumerate(solids):
		volume = xml.SubElement(structure, "volume", name=f"solid{i}_log")
		xml.SubElement(volume, "materialref", ref=solid.material)
		xml.SubElement(volume, "solidref", ref=f"solid{i}")
	world_volume = xml.SubElement(structure, "volume", name="world_log")
	xml.SubElement(world_volume, "materialref", ref="vacuum")
	xml.SubElement(world_volume, "solidref", ref="infinite_void")
	for i, solid in enumerate(solids):
		name = f"det_phys{i}" if solid.material == detector_material else f"body{i}"
		volume_specification = xml.SubElement(world_volume, "physvol", name=name)
		xml.SubElement(volume_specification, "volumeref", ref=f"solid{i}_log")
		xml.SubElement(
			volume_specification, "position", name=f"solid{i}_pos", unit="mm",
			x=f"{solid.x_position}", y=f"{solid.y_position}", z=f"{solid.z_position}")
		if solid.x_rotation != 0 or solid.y_rotation != 0 or solid.z_rotation != 0:
			xml.SubElement(
				volume_specification, "rotation", name=f"solid{i}_rot", unit="deg",
				x=f"{solid.x_rotation}", y=f"{solid.y_rotation}", z=f"{solid.z_rotation}")

	# and then whatever this is
	setup = xml.SubElement(input_deck, "setup", name="Default", version="1.0")
	xml.SubElement(setup, "world", ref="world_log")

	# write to disc
	os.makedirs("run", exist_ok=True)
	tree = xml.ElementTree(input_deck)
	xml.indent(tree)
	tree.write("run/input.gdml", xml_declaration=True, encoding="UTF-8")

	# clear previus output
	try:
		os.remove("run/output.dat")
	except FileNotFoundError:
		pass

	# call the executable
	subprocess.run(["grasshopper", "input.gdml", "output"], cwd="run")

	# read the output
	try:
		output_data = genfromtxt("run/output.dat", names=True, comments=None)
	except FileNotFoundError:
		raise RuntimeError("Geant4 failed to run.")
	return output_data


class Solid:
	def __init__(
			self, kind: str, material="detector",
			x_position=0., y_position=0., z_position=0.,
			x_rotation=0., y_rotation=0., z_rotation=0., **kwargs: float):
		self.kind = kind
		self.material = material
		self.x_position = x_position
		self.y_position = y_position
		self.z_position = z_position
		self.x_rotation = x_rotation
		self.y_rotation = y_rotation
		self.z_rotation = z_rotation
		self.kwargs = kwargs


class Beam:
	def __init__(self, particle: str, energy: float, width=0.0):
		"""
		a type of radiation
		:param particle: the name of the particle
		:param energy: the energy of each particle (MeV)
		:param width: the diameter of the beam (mm)
		"""
		self.particle_name = particle
		self.rest_mass = PARTICLE_DATA[particle]["rest_mass"]  # MeV/c²
		self.charge = PARTICLE_DATA[particle]["charge"]  # e
		self.number = PARTICLE_DATA[particle]["number"]
		self.energy = energy
		self.width = width
