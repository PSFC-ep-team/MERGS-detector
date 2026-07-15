MERGS-detector
================

This is a collection of Python scripts for simulating detector behavior using Grasshopper/Geant4.
The main ones are as follows:
* `detector.py` will compute the sensitivity of a detector of a given material and dimensions to electrons, photons, and neutrons.
* `visualization.py` will make a heatmap of electrons entering a detector.
* `experiment.py` will run a bunch of simulations with 2.5 MeV electrons that are supposed to represent a real experimental setup.
* `pareto.py` will optimize a bunch of detectors for 16.7 MeV electrons and plot the pareto front of best detectors.
* `simulation.py` is not meant to be run on its own, but rather contains the underlying Python–Grasshopper interface.

None of them do anything fancy with the command line so you can just run them as Python scripts.  For example:
```bash
python experiment.py
```

There are also some tests in `simulation.py` that you can run with pytest:
```bash
python -m pytest simulation.py
```

A few directories are automaticly generated sometimes.  Specificly:
* `figures/` is where all generated plots go.
* `results/` is where caches and calculated pareto fronts go.
* `run/` is where grasshopper is actually run, so that's where you'll find its raw outputs and any WRL files.
