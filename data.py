MATERIAL_DATA = {
	"LaBr3": {
		"density": 5.06,
		"efficiency": 63_000.*.10,
		"elements": {
			"La": 1,
			"Br": 3,
		},
	},
	"LYSO": {
		"density": 7.1,
		"efficiency": 33_200.*.10,
		"elements": {
			"Lu": 18,
			"Y": 2,
			"Si": 10,
			"O": 50,
		},
	},
	"EJ-276D": {
		"density": 1.099,
		"efficiency": 8_600.*.10,
		"elements": {
			"C": 4944,
			"H": 4647,
		},
	},
	"EJ-100": {
		"density": 1.03,
		"efficiency": 10_000.*.10,
		"elements": {
			"C": 468,
			"H": 516,
		},
	},
	"quartz": {
		"density": 2.65,
		"efficiency": 69.*.10,
		"elements": {
			"Si": 1,
			"O": 2,
		},
	},
	"aluminum": {
		"density": 2.699,
		"efficiency": 0,
		"elements": {
			"Al": 1,
		},
	},
	"silicon": {
		"density": 2.329,
		"efficiency": 276_000.,
		"elements": {
			"Si": 1,
		},
	},
	"PMMA": {
		"density": 1.18,
		"efficiency": 69.*.10,
		"elements": {
			"C": 5,
			"H": 8,
			"O": 2,
		},
	},
}

ELEMENT_DATA = {
	"H": (1, 1.008),
	"C": (6, 12.001),
	"N": (7, 14.007),
	"O": (8, 15.999),
	"Al": (13, 26.982),
	"Si": (14, 28.086),
	"Br": (35, 79.904),
	"Y": (39, 88.906),
	"La": (57, 138.91),
	"Lu": (71, 174.97),
}

PARTICLE_DATA = {
	"electron": {
		"rest_mass": 0.511,
		"charge": -1,
		"number": 11,
	},
	"proton": {
		"rest_mass": 938.272,
		"charge": +1,
		"number": 2212,
	},
	"neutron": {
		"rest_mass": 939.565,
		"charge": 0,
		"number": 2112,
	},
	"photon": {
		"rest_mass": 0,
		"charge": 0,
		"number": 22,
	},
}
