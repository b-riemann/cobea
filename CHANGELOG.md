## Version 0.13 ##

* Gradient computation now largely done using numpy.einsum, thanks to B. Isbarn for the suggestion. Gradient evaluation time has been observed to decrease by a factor 1.5 - 2. 
* A simple bug that caused the main cobea function to fail when no drift space was given has been removed.

## Version 0.12 ##

* The code is now also compatible with Python 3.6 and above (this only required minimal changes regarding xrange,range,float,and int)
* Some renaming of variables and function names below the user interface has occured.

## Version 0.11 ##

* First public upload. Hello, world!