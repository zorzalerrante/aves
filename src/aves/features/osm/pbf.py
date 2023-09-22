import os
from pathlib import Path


def clip_file(path, clipped_filename, bounds):
    print(
        f"osmconvert {path} -b={bounds[0]},{bounds[1]},{bounds[2]},{bounds[3]} -o={clipped_filename}"
    )
    os.system(
        f"osmconvert {path} -b={bounds[0]},{bounds[1]},{bounds[2]},{bounds[3]} -o={clipped_filename}"
    )
