
import os

'''
Submits a job for every sample defined in the info dict
'''

script_path = "/lustre/home/ekoch/code_repos/BaSiCs/Examples/THINGS/"

submit_file = os.path.join(script_path, "submit_THINGS.pbs")

# Load in the info dict for the names
execfile(os.path.join(script_path, "info_THINGS.py"))

datapath = "/lustre/home/ekoch/THINGS/"

for name in galaxy_props:
    galaxy_path = os.path.join(datapath, name)
    # Now submit it!
    os.system("qsub -v INP={1} {0}".format(submit_file, galaxy_path))
