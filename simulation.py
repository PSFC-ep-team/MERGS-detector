""" a function to facilitate building simple Grasshopper input decks """

from __future__ import annotations

import os
import subprocess
import xml.etree.ElementTree as xml

from numpy import genfromtxt, savetxt, concatenate, sin, cos, array, stack, interp, isclose, hypot, count_nonzero, diff, \
	sqrt, arcsin, pi, unique, inf
from numpy.typing import NDArray
from scipy import integrate

from data import PARTICLE_DATA, MATERIAL_DATA, ELEMENT_DATA


def simulate(detector_material: str, solids: list[Solid], beam: Beam, num_particles: int, debug_mode=False, full_output=False) -> tuple[NDArray, int]:
	"""
	run a Geant4 simulation of a beam of these particles hitting a detector.
	if the beam has an x_limit, the number of particles you get may not be exactly the number you requested.
	:return: the track data from Grasshopper and the total number of tracks simulated
	"""
	os.makedirs("run", exist_ok=True)

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

	# check if we're applying an x-limit to the bean.  if so, we need to augment the number of simulated particles.
	if beam.x_limit < beam.diameter/2:
		x_twiddle = beam.x_limit/(beam.diameter/2)
		valid_fraction = 2/pi*(arcsin(x_twiddle) + x_twiddle*sqrt(1 - x_twiddle**2))
		if valid_fraction < 0.1:
			raise ValueError("oh no, I don't like how much the runtime just got augmented.")
		num_particles = round(num_particles/valid_fraction)

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
	xml.SubElement(definitions, "constant", name="SaveTrackInfo", value="1" if full_output else "0")
	xml.SubElement(definitions, "constant", name="SaveEdepositedTotalEntry", value="0" if full_output else "1")
	# specify the bean
	xml.SubElement(definitions, "constant", name="RandomGenSeed", value="0")
	xml.SubElement(definitions, "constant", name="EventsToRun", value=f"{num_particles}")
	xml.SubElement(definitions, "constant", name="ParticleNumber", value=f"{beam.number}")
	xml.SubElement(definitions, "quantity",
	               name="BeamOffsetX", type="coordinate", value="0", unit="cm")
	xml.SubElement(definitions, "quantity",
	               name="BeamOffsetY", type="coordinate", value="0", unit="cm")
	if beam.ambient:
		xml.SubElement(definitions, "quantity",
		               name="WorldRadius", type="coordinate", value=f"{beam.distance}", unit="cm")
		xml.SubElement(definitions, "quantity",
		               name="BeamOffsetZ", type="coordinate", value="0", unit="cm")
		xml.SubElement(definitions, "quantity",
		               name="BeamSize", type="coordinate", value="-3", unit="mm")
	else:
		xml.SubElement(definitions, "quantity",
		               name="BeamOffsetZ", type="coordinate", value=f"{-beam.distance}", unit="cm")
		xml.SubElement(definitions, "quantity",
		               name="BeamSize", type="coordinate", value=f"{beam.diameter/2}", unit="cm")
	if type(beam.energy) is Spectrum:
		savetxt("run/input_spectrum.txt", stack([beam.energy.energies, beam.energy.probabilities], axis=1))
		xml.SubElement(definitions, "quantity",
		               name="BeamEnergy", type="energy", value="-1", unit="MeV")
	else:
		try:
			os.remove("run/input_spectrum.txt")
		except FileNotFoundError:
			pass
		xml.SubElement(definitions, "quantity",
		               name="BeamEnergy", type="energy", value=f"{beam.energy}", unit="MeV")

	# specify the geometry
	for i, solid in enumerate(solids):
		xml.SubElement(solid_group, solid.kind, name=f"solid{i}",
		               lunit="cm", **{key: f"{value}" for key, value in solid.kwargs.items()})
	xml.SubElement(solid_group, "box", name="infinite_void",
	               x="20", y="20", z="20", lunit="cm")

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
			volume_specification, "position", name=f"solid{i}_pos", unit="cm",
			x=f"{solid.x_position}", y=f"{solid.y_position}", z=f"{solid.z_position}")
		if solid.x_rotation != 0 or solid.y_rotation != 0 or solid.z_rotation != 0:
			xml.SubElement(
				volume_specification, "rotation", name=f"solid{i}_rot", unit="deg",
				x=f"{solid.x_rotation}", y=f"{solid.y_rotation}", z=f"{solid.z_rotation}")

	# and then whatever this is
	xml.SubElement(setup, "world", ref="world_log")

	# write to disc
	tree = xml.ElementTree(input_deck)
	xml.indent(tree)
	tree.write("run/input.gdml", xml_declaration=True, encoding="UTF-8")

	# clear previus output
	try:
		os.remove("run/output.dat")
	except FileNotFoundError:
		pass

	# call the executable
	print(f"Simulating {num_particles} {beam.particle_name}s...", end=" ")
	subprocess.run(["grasshopper", "input.gdml", "output"], cwd="run/", stdout=subprocess.DEVNULL)
	print(f"done!")

	# read the output
	try:
		output_data = genfromtxt("run/output.dat", names=True, comments=None)
	except FileNotFoundError:
		raise RuntimeError("Geant4 failed to run.")
	# convert mm to cm
	output_data["x_incident"] /= 10
	output_data["y_incident"] /= 10
	output_data["z_incident"] /= 10

	# filter out phony particles if we wanted a non-circular beam
	if beam.x_limit < beam.diameter/2:
		num_particles_recorded = len(unique(output_data["EventID"]))
		if num_particles_recorded/num_particles < 0.99:
			raise ValueError(
				f"because of the way grasshopper is, I can't limit the beam unless all of the particles are incident on "
				f"more or less the same solid material (because any geometric correlations will cause me to incorrectly "
				f"estimate how many particles were simulated in the specified beam limits that we _didn't_ see).  "
				f"however, the fact that {1 - num_particles_recorded/num_particles:.1%} of the particles did not "
				f"interact with any matter leads me to suspect that some of them are missing the detector entirely.")
		valid = abs(output_data["x_incident"]) <= beam.x_limit
		output_data = output_data[valid]
		num_particles_recorded_in_bounds = len(output_data)
		num_particles = round(num_particles_recorded_in_bounds/num_particles_recorded*num_particles)
		reduced_old_event_indices, new_event_indices = unique(output_data["EventID"], return_inverse=True)
		output_data["EventID"] = new_event_indices  # remember to re-index the events

	return output_data, num_particles


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
	def __init__(self, particle: str, energy: float | Spectrum, diameter=0.0, x_limit=inf, distance=10.0, ambient=False):
		"""
		a type of radiation
		:param particle: the name of the particle
		:param energy: the energy of each particle (MeV)
		:param diameter: the diameter of the beam (cm)
		:param x_limit: a limit on the absolute value of x, which
		:param distance: the standoff distance of the source from the origin
		:param ambient: whether particles should come from all directions instead of just z-
		"""
		self.particle_name = particle
		self.rest_mass = PARTICLE_DATA[particle]["rest_mass"]  # MeV/c²
		self.charge = PARTICLE_DATA[particle]["charge"]  # e
		self.number = PARTICLE_DATA[particle]["number"]
		self.energy = energy
		self.diameter = diameter
		self.x_limit = x_limit
		self.distance = distance
		self.ambient = ambient
		if ambient and diameter != 0:
			raise ValueError("you can't pass a diameter when the source is ambient because that doesn't make any sense")


class Spectrum:
	def __init__(self, name: str, energies: NDArray, probabilities: NDArray):
		if any(diff(energies) < 0):
			raise ValueError("spectrum energies must be monotonicly increasing.")
		if energies[-1] == energies[0]:
			raise ValueError("the spectrum must have some extent or it's not really normalizable")
		if any(probabilities < 0):
			raise ValueError("probability density cannot be negative")
		if not any(probabilities > 0):
			raise ValueError("this spectrum is unnormalizable")
		self.name = name
		self.energies = energies
		self.probabilities = probabilities

	def __str__(self):
		return self.name

	def truncate(self, lower_bound: float) -> tuple[Spectrum, float]:
		""" cut off the part of the spectrum below lower_bound, and return the factor by which this changes the normalization """
		if lower_bound <= self.energies[0]:
			return self, 1.0
		elif lower_bound >= self.energies[-1]:
			raise ValueError("you're trying to truncate the whole spectrum away.  that would make an unnormalizable spectrum.")
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
	tracks, _ = simulate(
		"silicon", [simple_box],
		Beam("proton", uniform_spectrum, diameter=2),
		num_particles=num_particles, full_output=True)
	incident_tracks = tracks[tracks["TrackID"] == 1]
	assert isclose(incident_tracks.size, num_particles/2, atol=100)
	assert all(incident_tracks["x_incident"] >= 0)
	assert all(hypot(incident_tracks["x_incident"], incident_tracks["y_incident"]) <= 1. + 1e-7)
	assert all(incident_tracks["E_beamMeV"] <= 14.)
	assert isclose(count_nonzero(incident_tracks["E_beamMeV"] > 10.), incident_tracks.size*2/7, atol=100)
	assert all(incident_tracks["theta"] == 0.)


def test_beam_limiting():
	simple_box = Solid("box", x=2, y=2, z=2)
	num_particles = 10_000
	tracks, _ = simulate(
		"silicon", [simple_box],
		Beam("proton", energy=14, diameter=2, x_limit=0.5),
		num_particles=num_particles, full_output=True)
	incident_tracks = tracks[tracks["TrackID"] == 1]
	assert isclose(len(incident_tracks), 10_000, atol=100)


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
