#!/usr/bin/env python

from metasyn.filehandler import get_file_handler
import sys


if __name__ == "__main__":
    dataset_fp = sys.argv[1]
    handler_class = get_file_handler(dataset_fp)
    df, _ = handler_class.from_file(dataset_fp)
    print(df.head(6))