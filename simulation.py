""" a function to facilitate building simple Grasshopper input decks """

from __future__ import annotations

import os
import subprocess
import xml.etree.ElementTree as xml

from numpy import genfromtxt, savetxt, concatenate, sin, cos, array, stack, interp, isclose, hypot, count_nonzero
from numpy.typing import NDArray
from scipy import integrate

from data import PARTICLE_DATA, MATERIAL_DATA, ELEMENT_DATA


def simulate(detector_material: str, solids: list[Solid], beam: Beam, num_particles: int, debug_mode=False) -> NDArray:
	""" run a Geant4 simulation of a beam of these particles hitting a detector """
	# start by instantiating the input deck
	input_deck = xml.Element("gdml", {
		"xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
		"xsi:noNamespaceSchemaLocation": os.path.expanduser("~/grasshopper/schema/gdml.xsd"),
	})
	materials = xml.SubElement(input_deck, "materials")
	definitions = xml.SubElement(input_deck, "define")
	solid_group = xml.SubElement(input_deck, "solids")
	structure = xml.SubElement(input_deck, "structure")
	setup = xml.SubElement(input_deck, "setup", name="Default", version="1.0")

	# check the ambient flag.  if so, we need to use an "omnidirectional" beam.
	if beam.ambient:
		xml.SubElement(definitions, "quantity",
		               name="WorldRadius", type="coordinate", value=f"{beam.distance}", unit="mm")
		source_position = "0"
		source_code = "-3"
	else:
		source_position = f"{-beam.distance}"
		source_code = f"{beam.diameter/2}"

	# check if there are multiple energies.  if so, we need to use `input_spectrum.txt`.
	if type(beam.energy) is Spectrum:
		savetxt("run/input_spectrum.txt", stack([beam.energy.energies, beam.energy.probabilities], axis=1))
		energy_code = "-2"
	else:
		try:
			os.remove("run/input_spectrum.txt")
		except FileNotFoundError:
			pass
		energy_code = f"{beam.energy}"

	# first apply detector_material to every detector solid
	for solid in solids:
		if solid.material == "detector":
			solid.material = detector_material

	# specify the materials
	for material in {solid.material for solid in solids}:
		density = MATERIAL_DATA[material]["density"]
		elements = MATERIAL_DATA[material]["elements"]
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
	xml.SubElement(material_info, "D", value="0", unit="g/cm3")
	xml.SubElement(material_info, "composite", ref="N", n="1.0")

	# specify output settings
	xml.SubElement(definitions, "constant", name="TextOutputOn", value="1")
	xml.SubElement(definitions, "constant", name="BriefOutputOn", value="0")
	xml.SubElement(definitions, "constant", name="VRMLvisualizationOn", value="1" if debug_mode else "0")
	xml.SubElement(definitions, "constant", name="EventsToAccumulate", value="100" if debug_mode else "0")
	# specify particle selections
	xml.SubElement(definitions, "constant", name="LightProducingParticle", value="0")
	xml.SubElement(definitions, "constant", name="LowEnergyCutoff", value="0")
	xml.SubElement(definitions, "constant", name="KeepOnlyMainParticle", value="0")
	xml.SubElement(definitions, "quantity",
	               name="ProductionLowLimit", type="threshold", value="1", unit="keV")
	# specify output filters
	xml.SubElement(definitions, "constant", name="SaveSurfaceHitTrack", value="0")
	xml.SubElement(definitions, "constant", name="SaveTrackInfo", value="1")
	xml.SubElement(definitions, "constant", name="SaveEdepositedTotalEntry", value="0")
	# specify the bean
	xml.SubElement(definitions, "constant", name="RandomGenSeed", value="0")
	xml.SubElement(definitions, "quantity",
	               name="BeamOffsetX", type="coordinate", value="0", unit="mm")
	xml.SubElement(definitions, "quantity",
	               name="BeamOffsetY", type="coordinate", value="0", unit="mm")
	xml.SubElement(definitions, "quantity",
	               name="BeamOffsetZ", type="coordinate", value=source_position, unit="mm")
	xml.SubElement(definitions, "quantity",
	               name="BeamSize", type="coordinate", value=source_code, unit="mm")
	xml.SubElement(definitions, "quantity",
	               name="BeamEnergy", type="energy", value=energy_code, unit="MeV")
	xml.SubElement(definitions, "constant", name="EventsToRun", value=f"{num_particles}")
	xml.SubElement(definitions, "constant", name="ParticleNumber", value=f"{beam.number}")

	# specify the geometry
	for i, solid in enumerate(solids):
		xml.SubElement(solid_group, solid.kind, name=f"solid{i}",
		               lunit="mm", **{key: f"{value}" for key, value in solid.kwargs.items()})
	xml.SubElement(solid_group, "box", name="infinite_void",
	               x="300", y="300", z="300", lunit="mm")

	# fill in the remaining information
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


def rotation_matrix(θ):
	return array([
		[cos(θ), sin(θ)],
		[-sin(θ), cos(θ)],
	])


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
	def __init__(self, particle: str, energy: float | Spectrum, diameter=0.0, distance=100.0, ambient=False):
		"""
		a type of radiation
		:param particle: the name of the particle
		:param energy: the energy of each particle (MeV)
		:param diameter: the diameter of the beam (mm)
		:param distance: the standoff distance of the source from the origin
		:param ambient: whether particles should come from all directions instead of just z-
		"""
		self.particle_name = particle
		self.rest_mass = PARTICLE_DATA[particle]["rest_mass"]  # MeV/c²
		self.charge = PARTICLE_DATA[particle]["charge"]  # e
		self.number = PARTICLE_DATA[particle]["number"]
		self.energy = energy
		self.diameter = diameter
		self.distance = distance
		self.ambient = ambient
		if ambient and diameter != 0:
			raise ValueError("you can't pass a diameter when the source is ambient because that doesn't make any sense")


class Spectrum:
	def __init__(self, name: str, energies: NDArray, probabilities: NDArray):
		self.name = name
		self.energies = energies
		self.probabilities = probabilities

	def __str__(self):
		return self.name

	def truncate(self, lower_bound: float) -> tuple[Spectrum, float]:
		""" cut off the part of the spectrum below lower_bound, and return the factor by which this changes the normalization """
		total_sum = integrate.trapezoid(self.probabilities, self.energies)
		above_lower_bound = self.energies > lower_bound
		p_bound = interp(lower_bound, self.energies, self.probabilities)
		new_energies = concatenate([[lower_bound], self.energies[above_lower_bound]])
		new_probabilities = concatenate([[p_bound], self.probabilities[above_lower_bound]])
		truncated_sum = integrate.trapezoid(new_probabilities, new_energies)
		new_spectrum = Spectrum(f"{self.name} above {lower_bound} MeV", new_energies, new_probabilities)
		return new_spectrum, truncated_sum/total_sum


def test_simulation():
	uniform_spectrum = Spectrum("uniform", array([0., 14.]), array([1., 1.]))
	simple_box = Solid("box", x=2, y=2, z=2, x_position=1.0)
	num_particles = 10_000
	tracks = simulate(
		"silicon", [simple_box],
		Beam("proton", uniform_spectrum, diameter=2),
		num_particles=num_particles)
	incident_tracks = tracks[tracks["TrackID"] == 1]
	assert isclose(incident_tracks.size, num_particles/2, atol=100)
	assert all(incident_tracks["x_incident"] >= 0)
	assert all(hypot(incident_tracks["x_incident"], incident_tracks["y_incident"]) <= 1.)
	assert all(incident_tracks["E_beamMeV"] <= 14.)
	assert isclose(count_nonzero(incident_tracks["E_beamMeV"] > 10.), incident_tracks.size*2/7, atol=100)
	assert all(incident_tracks["theta"] == 0.)


def test_spectrum():
	whole_spectrum = Spectrum(
		"test",
		array([5., 15., 20.]),
		array([1.0, 0.0, 0.0]),
	)
	truncated_spectrum, truncated_fraction = whole_spectrum.truncate(10.)
	assert isclose(truncated_fraction, 1/4)
	assert all(isclose(truncated_spectrum.energies, [10., 15., 20.]))
	assert all(isclose(truncated_spectrum.probabilities, [0.5, 0.0, 0.0]))
