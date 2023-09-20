import os, psutil
process = psutil.Process(14828)
print(process.memory_info())  # in bytes
print(process._proc._get_raw_meminfo())  # in bytes
