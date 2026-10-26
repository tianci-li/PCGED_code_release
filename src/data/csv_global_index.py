import json
import os
import re
import csv
import numpy as np

csv.field_size_limit(1000000000)











_index_cache: dict = {}


def _get_cached_offsets(index_filepath: str, local_index: int) -> int:

    offsets = _index_cache.get(index_filepath)
    if offsets is None:
        with open(index_filepath, mode='r') as f:
            offsets = np.array([int(line.strip()) for line in f], dtype=np.int64)
        _index_cache[index_filepath] = offsets
    return int(offsets[local_index])


def create_global_index(line_count_file: str):
    with open(line_count_file, 'r') as f:
        file_line_counts=json.load(f)

    global_index = []
    start_line = 0
    for filename, line_count in file_line_counts.items():
        global_index.append((filename, start_line, line_count))
        start_line += line_count
    return global_index










_file_lookup_cache: dict = {}


def _get_file_lookup(global_index: list):

    cache_key = id(global_index)
    lookup = _file_lookup_cache.get(cache_key)
    if lookup is None:
        starts = np.array([entry[1] for entry in global_index], dtype=np.int64)
        filenames = [entry[0] for entry in global_index]
        line_counts = [entry[2] for entry in global_index]
        lookup = (starts, filenames, line_counts)
        _file_lookup_cache[cache_key] = lookup
    return lookup


def find_file_and_local_index(global_index: list, global_line_index: int):
    starts, filenames, line_counts = _get_file_lookup(global_index)


    file_idx = int(np.searchsorted(starts, global_line_index, side='right')) - 1
    if file_idx < 0 or file_idx >= len(filenames):
        return None, None
    local_line_index = global_line_index - int(starts[file_idx])
    if local_line_index >= line_counts[file_idx]:
        return None, None
    return filenames[file_idx], local_line_index

def get_global_len(global_index: list):



    start = global_index[-1][1]
    length = global_index[-1][2]
    return start+length

def read_csv_by_local_index(index_folder, csv_filepath, local_index):
    csv_fn = re.split('/', csv_filepath)
    folder = csv_fn[-2]
    index_fn = re.sub(r'\.csv$', '.index', csv_fn[-1])

    index_filepath = os.path.join(index_folder, folder, index_fn)

    offset = _get_cached_offsets(index_filepath, local_index) + local_index
    with open(csv_filepath, mode='r') as csv_file:
        csv_file.seek(offset)
        csv_reader = csv.reader([csv_file.readline().strip()])
        l = next(csv_reader)
        return l[:-1], l[-1].strip('[]').replace('\'','').split(', ')

def get_line_by_local_index(index_folder ,csv_filepath, local_index):
    csv_fn = re.split('/', csv_filepath)
    folder = csv_fn[-2]
    index_fn = re.sub(r'\.csv$', '.index', csv_fn[-1])

    index_filepath = os.path.join(index_folder, folder, index_fn)

    offset = _get_cached_offsets(index_filepath, local_index) + local_index
    with open(csv_filepath, mode='r') as csv_file:
        csv_file.seek(offset)
        line = csv_file.readline().strip()
        return line


if __name__ == '__main__':
    import time

    csv_folder = 'sample_csv'
    index_folder = 'sample_csv_index'
    line_count_file = 'line_count.json'
    with open(line_count_file, mode='r') as f:
        file_line_counts = json.load(f)
    global_index = create_global_index(file_line_counts)
    length = get_global_len(global_index)
    print('global_index_lenght:', length)


    global_line_index = 249
    start = time.time()
    filename, local_line_index = find_file_and_local_index(global_index, global_line_index)



    label, expr = read_csv_by_local_index(index_folder, filename, local_line_index)
    end = time.time()
    print(expr)
    print(type(label))
    print(end-start)

    label_list =  [
            'tissue',
            'suspension_type',
            'cell_type',

            'disease',
            'sex',
            'development_stage'
        ]

    def label_sort(label_list,label_raw):

        return_list = []

        for label in label_list:
            for i in range(len(label_raw)):
                if label_raw[i].startswith(label):
                    if label == "development_stage":
                        return_list.append(label_raw[i+1])
                        return_list.append(label_raw[i+2])
                    else:
                        return_list.append(label_raw[i])


        return return_list
    print(label_sort(label_list,label))





