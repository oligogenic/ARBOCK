import csv
import sys
import os

__author__      = "Alexandre Renaux"
__copyright__   = "Copyright (c) 2023 Alexandre Renaux - Universite Libre de Bruxelles - Vrije Universiteit Brussel"
__license__     = "MIT"
__version__     = "1.0.1"

csv.field_size_limit(sys.maxsize)


class CSVFileReader:
    def __init__(self, file_path, has_header=True, delimiter=',', skip_initial_comments_flag=None, doublequote=False):
        self.file_path = os.path.expanduser(file_path)
        self.file = open(self.file_path, 'r', newline='')
        self.reader = csv.reader(self.file, delimiter=delimiter, doublequote=doublequote)
        self.skip_initial_comments(skip_initial_comments_flag)
        self.header = next(self.reader, None) if has_header else None

    def skip_initial_comments(self, flag):
        if flag is not None:
            line_offset = []
            offset = 0
            line_count = 0
            while True:
                line = self.file.readline()
                line_offset.append(offset)
                offset += len(line)
                line_count += 1
                if not line.startswith(flag):
                    self.file.seek(line_offset[line_count - 1])
                    break

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()

    def __iter__(self):
        return self

    def __next__(self):
        if self.header:
            return {self.header[col[0]]:col[1] for col in enumerate(next(self.reader))}
        else:
            return next(self.reader)


class CSVFileWriter:
    def __init__(self, file_path, delimiter):
        self.file_path = os.path.expanduser(file_path)
        self.file = open(self.file_path, 'w', newline='\n', encoding='utf-8')
        self.writer = csv.writer(self.file, delimiter=delimiter, quoting=csv.QUOTE_MINIMAL, lineterminator='\n')

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()

    def write_row(self, line):
        self.writer.writerow(line)

    def close(self):
        self.file.close()