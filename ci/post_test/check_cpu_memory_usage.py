import time
import sys
import pandas as pd
import re


# read top output log file
if len(sys.argv) == 3:
    print("Top command output file path is " + sys.argv[1])
    original_data = pd.read_csv(sys.argv[1], sep="\s+", header=None)
else:
    print("Wrong input arguments, at least two arguments")
    sys.exit(-1)


if original_data.shape[1] == 12:
    original_data.columns = [
        "PID",
        "USER",
        "PR",
        "NI",
        "VIRT",
        "RES",
        "SHR",
        "S",
        "%CPU",
        "%MEM",
        "TIME+",
        "COMMAND",
    ]
else:
    print("Please check top command output format")
    sys.exit(-1)
# Sort by cpu%
original_data.sort_values(by="%MEM", ascending=False, inplace=True)
original_data.reset_index(inplace=True)
# Get the maximum CPU physical memory usage
searchObj_forT = re.search(r"(.*)(t|T)", original_data["RES"][0], re.M | re.I)
searchObj_forG = re.search(r"(.*)(G|g)", original_data["RES"][0], re.M | re.I)
searchObj_forM = re.search(r"(.*)", original_data["RES"][0], re.M | re.I)
cpu_usage_gb = 0
if searchObj_forT:
    cpu_usage_gb = float(searchObj_forT.group(1)) * 1000
elif searchObj_forG:
    cpu_usage_gb = float(searchObj_forG.group(1))
else:
    cpu_usage_gb = float(searchObj_forM.group(1)) / 1000 / 1000

if cpu_usage_gb > float(sys.argv[2]):
    print("The maximum physical memory usage exceeds the threshold " + sys.argv[2])
    sys.exit(-1)
