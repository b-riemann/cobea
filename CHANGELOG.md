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
