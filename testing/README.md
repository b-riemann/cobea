### Scripts for testing COBEA execution ###

#### [reference_tool.py](reference_tool.py) ####

A script that compares pickled results, e.g.

    python reference_tool.py ../examples/delta_output/response.171027-2-nobump/{response_input,result}.pickle

To overwrite result.pickle instead of comparing, type

    python reference_tool.py ../examples/delta_output/response.171027-2-nobump/{response_input,result}.pickle ref

#### [cobea_timing.py](cobea_timing.py) ####

    python cobea_timing.py ../examples/delta_output/response.171027-2-nobump/response_input.pickle