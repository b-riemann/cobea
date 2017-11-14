## Version 0.24 ##

* postprocessing is now ~4x faster for typical problem sizes (using EIGH instead of SVD in error computation)
* clean-up in [MCS](cobea/mcs.py), now also uses EIGH.

## Version 0.23 ##

* new testing folder (moved [reference_tool](testing/reference_tool.py) there)
* internal timing in [cobea](cobea/__init__.py) function has been removed, use [cobea_timing](testing/cobea_timing.py)
  for this in the future.
* internals of error computation changed (moving to ErrorModel object)

## Version 0.22 ##

* Error margins for all quantities could be significantly reduced due to an error found
  in computing the variance expectation value.
* paper format in plot_results changed to DIN.
* errors are plotted as boxes, plots are using standard matplotlib colors.
* Response and Result now have a name attribute (shown when printing results).

## Version 0.21 ##

* Bug in error computation for missing dispersion has been corrected.
* DriftSpace class, plotting of continuous optical functions in known elements.
* low-level index functions have been simplified (part of topology objects and/or numpy.in1d)

### Version 0.20a ###

* drift space information is now included in the Response input object
* A bug in error computation has been discovered for include_dispersion=False, it will be corrected in the next release.

## Version 0.20 ##

* filter for corrector sets in [Topology](cobea/model.py) and Response class.
* Some plotting outputs have been modified accordingly.
* modified Result summary and error attributes.
* error handling in [Response](cobea/model.py) for missing corr_names in line.

## Version 0.19 ##

* slightly improved reference_tool
* cleaned up formatting and result.additional

## Version 0.18 ##

* improved optimization layer execution time by about 20% (array ops in _from_statevec, _to_statevec)

## Version 0.17 ##

* wrapper for DELTA storage ring (standard response files) has been added
* Some cleanup happened for the interiors of cobea.mcs and reference_tool

## Version 0.16 ##

* A [reference tool](testing/reference_tool.py) has been added to compare results of different runs.
* Output of cobea.mcs.layer has been reduced.
* Postprocessing layer is now a separate function (cobea.pproc.layer).

## Version 0.15 ##

* A usage example has been added in form of a random response object generator
  in [model_generator.py](examples/model_generator.py). This allows extensive testing for end-users
  (thanks to P. Towalski for the suggestion).
* A more stable quadrant guessing mechanism in the case of unavailable drift space information has been implemented.
* There are now more plots for multi-mode monitor quantities, reducing the cluttered look of monitor output.

### Version 0.14a ###

* typo in plotting.cbeta_km removed.

## Version 0.14 ##

* A simple bug in load_result (binary mode) has been removed.
* Plotting of corrector quantities is now possible in principle
  (error margins not yet included for their cbeta and delphi values)
* Plot output is now easier to customize (see plot_flags in plotting.plot_result)

## Version 0.13 ##

* Gradient computation now largely done using numpy.einsum
  (thanks to B. Isbarn for the suggestion).
  Gradient evaluation time has been observed to decrease by a factor 1.5 - 2. 
* A simple bug that caused the main cobea function to fail when no drift space was given has been removed.

## Version 0.12 ##

* The code is now also compatible with Python 3.6 and above
  (this only required minimal changes regarding xrange,range,float,and int)
* Some renaming of variables and function names below the user interface has occured.

## Version 0.11 ##

* First public upload. Hello, world!
