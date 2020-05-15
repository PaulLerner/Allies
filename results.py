#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Prints human readable results from beat output log and saves them as csv.
Both averaged and grouped by type of supervision.

Usage:
results.py <log>
"""

from docopt import docopt
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
import re

def main(args):
    log_path=Path(args['<log>'])

    #open and parse results
    with open(log_path) as file:
        log=file.read()
    str_results,DERs=[],[]
    for line in log.split('\n'):
        if line == '' or line.isspace() or line.split()[0] != 'file':
            continue
        #I don't use file_uri for now but I guess we might study them later ?
        _,file_uri,DER,supervision_type=re.split('file |: DER = |, supervision: ',line)
        str_results.append((file_uri,supervision_type))
        DERs.append(float(DER))
    str_results,DERs=np.array(str_results),np.array(DERs)

    # Plot results
    plt.title(log_path.stem)
    plt.xlabel('File #')
    plt.ylabel('DER')
    plt.plot(DERs)
    plt.savefig(log_path.with_suffix('.png'))

    #prints results and write csv
    average = np.mean(DERs)
    print(f"Overall Average: {average:.2f}")
    with open(log_path.parent/f'{log_path.stem}.csv','w') as file:
        file.write(f'overall,{average:.2f}\n')
        supervision_types = np.unique(str_results[:,1])
        for supervision_type in supervision_types:
            indices=np.where(str_results[:,1]==supervision_type)[0]
            average=np.mean(DERs[indices])
            print(f'{supervision_type} supervision average: {average:.2f}')
            file.write(f'{supervision_type},{average:.2f}\n')

if __name__ == '__main__':
    args = docopt(__doc__)
    main(args)
