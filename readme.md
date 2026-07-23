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
python -m pytest detector.py
python -m pytest pareto.py
```

A few directories are automaticly generated sometimes.  Specificly:
* `figures/` is where all generated plots go.
* `results/` is where caches and calculated pareto fronts go.
* `run/` is where grasshopper is actually run, so that's where you'll find its raw outputs and any WRL files.

## Installation

I mean there's no installation needed but if you're installing Grasshopper the instructions on GitHub are really not sufficient.
For example, following them verbatim throws the error that you're missing "FindGeant4.cmake", and I don't even know what that is.
So here's a quick guide for installing Grasshopper:

### Using CMake (I hate software developers)
First of all, if you haven't used CMake, you need to use it thrice here, and I hate it because it's terrible so I'm just going to summarize CMake here.
In the year 2000, computer programmers had invented the computer program but hadn't yet invented computer programs that do two things in sequence.
So instead of shipping software with a script to install the software like modern Python libraries, they shipped it with a bunch of folders full of files that you install in three separate steps using two separate build tools.
And lots of C++ libraries still use this because everything about C++ is terrible.

First, you create a "build" directory in which you'll dump all of the trash that needs to be passed between the build tools.
```bash
mkdir program-build
```
Then you run CMake to configure the source code into the build directory, and pass most of your compilation options using the `-D` flag.
One of the more important compilation options is `CMAKE_INSTALL_PREFIX` which is the folder where you want to put the built binaries and libraries.
If you leave it blank it'll put them somewhere high up in your file system where they're hidden away and presumably on your paths,
but then you'll need to use `sudo` when you install.  If you don't have super-user rights, you'll have to specify a directory to which you have write access.
The locations of dependencies are also often compilation options.
```bash
cd program-build
cmake -D OPTION=ON -D ANOTHER_OPTION=OFF -D CMAKE_INSTALL_PREFIX=~/program-install ~/program-src
```
Then you run regular Make to build the binaries and libraries.
```bash
make
```
Then you run regular Make in "install" mode to move the binaries and libraries from wherever it put them into the correct folder.
```bash
make install
```

If either of the Make steps gets interrupted or you make a mistake and have to start over, make sure to call Make in "clean" mode to make it overwrite its last attempt rather than trying and invariably failing to reuse it.
```bash
make clean
```

### Installing Xerces

First, you need Apache Xerces, an XML parser, which is a prerequisite for Geant4 with GDML, which is a prerequisite for Grasshopper.
This one doesn't require any weird CMake arguments.

```bash
mkdir ~/xerces-build
cd ~/xerces-build
cmake -D CMAKE_INSTALL_PREFIX=~/xerces-install ~/xerces-build
make
make install
```

After it's installed, if you changed the install directory, you need to add the install's library subdirectory to your library path.
```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/xerces-install/lib64
```
Note that sometimes the library subdirectory is `lib/` and sometimes it's `lib64/`.
I don't know what determines when it's which; you just have to look inside the installation and check the name of the folder that contains the shared object files.

### Installing Geant4

Now you need to install Geant4.  For this, you'll need Expat.  IDK what Expat is, but most systems seem to have it already installed so you probably don't have to worry about installing it.
But if you're on the engaging cluster you need to load it as a module, and it's hidden.  So first you have to search for it with
```bash
module --show_hidden spider expat
```
and then you have to load one of the available versions that comes up.
If you have issues, an alternative approach is to configure Geant4 to use internal Expat with `-D GEANT4_USE_SYSTEM_EXPAT=OFF`

Then you can move on to Geant itself.  Download the source, then configure it with CMake, then build it with Make, then install it with Make.
There are many CMake options, but the important ones are:
* `GEANT4_USE_GDML` which Grasshopper requires be set to `ON`
* `GEANT4_INSTALL_DATA` which must be set to `ON`, unless you don't want data which I guess is the default for some reason?
* `GEANT4_BUILD_MULTITHREADED` which you presumably want `ON`, especially if you're building on a cluster
* `CMAKE_PREFIX_PATH` which must point to Xerces if Xerces wasn't installed in the default place

The build step takes a while here, so you might want to move it into a Slurm partition if you're on a shared computing cluster.

```bash
mkdir ~/geant4-build
cd ~/geant4-build
cmake -D GEANT4_USE_GDML=ON -D GEANT4_INSTALL_DATA=ON -D GEANT4_BUILD_MULTITHREADED=ON -D CMAKE_PREFIX_PATH=~/xerces-install -D CMAKE_INSTALL_PREFIX=~/geant4-install ~/geant4-src
sbatch --time=2:00:00 --wrap="make && make install"
```

After it's installed, if you changed the install directory, you need to add the install's library subdirectory to your library path.
```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/geant4-install/lib64
```
Note that sometimes the library subdirectory is `lib/` and sometimes it's `lib64/`.
I don't know what determines when it's which; you just have to look inside the installation and check the name of the folder that contains the shared object files.

### Installing Grasshopper

Finally, Grasshopper.  Download the source, then configure it with CMake, then build it with Make, then install it with Make.
The only CMake option you need is `DGeant4_DIR` which should point to the `lib/cmake/Geant4/` subfolder of the Geant4 installation,
if Geant4 wasn't installed in the default place.

Areg says you also need to pass a `-jN` flag to Make to specify how many cores it can use, but you don't need to do that as long as you're either in a partition or on a personal device.
Just let it infer the number of cores you have (this is one of the few actually good defaults).

```bash
mkdir ~/grasshopper-build
cd ~/grasshopper-build
cmake -D Geant4_DIR=~/geant4-install/lib/cmake/Geant4 -D CMAKE_INSTALL_PREFIX=~/grasshopper-install ~/grasshopper-src
make
make install
```

After it's installed, you need to manually add the Grasshopper executable to your path.  For example:
```bash
export PATH=$PATH:~/grasshopper-install/bin
```

Once this is done, you can remove all of the build directories, plus all of the source codes if you don't plan to make changes.
