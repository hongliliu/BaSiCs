
import os
from datetime import datetime

'''
Submits a job for every sample defined in the info dict
'''


def timestring():
    return datetime.now().strftime("%Y%m%d%H%M%S%f")

script_path = "/lustre/home/ekoch/code_repos/BaSiCs/Examples/THINGS/"

submit_file = os.path.join(script_path, "submit_THINGS.pbs")

# Load in the info dict for the names
execfile(os.path.join(script_path, "info_THINGS.py"))

datapath = "/lustre/home/ekoch/THINGS/"

for name in galaxy_props:
    galaxy_path = os.path.join(datapath, name)
    now_time = timestring()
    error_file = \
        os.path.join(galaxy_path, "{0}_bubbles_{1}.err".format(name, now_time))
    output_file = \
        os.path.join(galaxy_path, "{0}_bubbles_{1}.out".format(name, now_time))
    # Now submit it!
    os.system("qsub -e {2} -o {3} -v INP={1} {0}".format(submit_file,
                                                         galaxy_path,
                                                         error_file,
                                                         output_file))
