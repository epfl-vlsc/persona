
import gzip
import os
import argparse

class ListComparer:
    # Comparing two lists of elements by casting them to a set => each element occurs only once
    def __init__(self, list1, list2):
        self.list1 = list1
        self.list2 = list2
        self.set1 = set(list1)
        self.set2 = set(list2)

    def get_union(self):
        return list(self.set1 | self.set2)

    def get_intersection(self):
        return list(self.set1 & self.set2)

    def only_in_list1(self):
        return list(self.set1 - self.set2)

    def only_in_list2(self):
        return list(self.set2 - self.set1)

    def get_total_number(self):
        return len(self.get_union())

    def get_total_number_in_both(self):
        return len(self.get_intersection())

    def get_total_number_in_list1(self):
        return len(self.set1)

    def get_total_number_in_list2(self):
        return len(self.set2)

    def list1_has_duplicates(self):
        return len(self.get_duplicates_of_list1()) > 0

    def list2_has_duplicates(self):
        return len(self.get_duplicates_of_list2()) > 0

    def get_duplicates_of_list1(self):
        return list(set([x for x in self.list1 if self.list1.count(x) > 1]))

    def get_duplicates_of_list2(self):
        return list(set([x for x in self.list2 if self.list2.count(x) > 1]))

def darwin_intervall_to_list(intervall):
    # Conwerts a Darwin range into a tuple of (MIN, MAX)
    entries = intervall.split('.')
    ilow = int(entries[0])
    ihigh = int(entries[-1])
    return ilow, ihigh


def candidate_pair_from_string(string):
    # Extracts the needed data of a pair from a string (from the AllAll output)
    astring = string
    string = string.translate(None, '[]')
    entries = string.split(',')
    if not len(entries) == 7:
        print(entries)
        print(astring)
    seq1 = int(entries[0])
    seq2 = int(entries[1])
    score = float(entries[2])
    distance = float(entries[3])
    int1 = darwin_intervall_to_list(entries[4])
    int2 = darwin_intervall_to_list(entries[5])
    variance = float(entries[6])
    return seq1, seq2, int1, int2


def process_compressed_file(filename):
    # Read the lines of a zipped AllAll-output file
    f = gzip.open(filename)
    f_content = f.readlines()
    f.close()
    data_lines = [s.split(',\n')[0].translate(None, '[]):\n') for s in f_content if '[' in s]
    data_lines = [s for s in data_lines if not s.startswith("#")]
    candidates = []
    for d in data_lines:
        if not d.strip():
            continue
        cp = candidate_pair_from_string(d)
        candidates.append(cp)
    return candidates

def process_file(filename):
    # Read the lines of a unzipped AllAll-output file
    f = open(filename)
    f_content = f.readlines()
    f.close()
    data_lines = [s.split(',\n')[0].translate(None, '[]):\n') for s in f_content if '[' in s]
    data_lines = [s for s in data_lines if not s.startswith("#")]
    candidates = []
    for d in data_lines:
        if not d.strip():
            continue
        cp = candidate_pair_from_string(d)
        candidates.append(cp)
    return candidates

def fill_dictionary(folderpath, dictionary):
    # Fills up the dictionary (based on the folder name)
    folders = [x[0] for x in os.walk(folderpath) if x[0] != folderpath]
    for folder in folders:
        genome1 = folder.split('/')[-1]
        files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
        file_names = [f for f in files if f.endswith('.gz')]
        for compressed_file in file_names:
            genome2 = compressed_file.split('/')[-1].split('_')[0].split('.gz')[0]
            full_file_path = os.path.join(folder, compressed_file)
            candidates = process_compressed_file(full_file_path)
            key = tuple([genome1, genome2])
            if not key in dictionary:
                dictionary[key] = list()
            dictionary[key].extend(candidates)

def fill_dictionary_nogz(folderpath, dictionary):
    # Fills up the dictionary (based on the folder name)
    folders = [x[0] for x in os.walk(folderpath) if x[0] != folderpath]
    for folder in folders:
        genome1 = folder.split('/')[-1].upper()
        files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
        file_names = files #[f for f in files if f.endswith('.gz')]
        for compressed_file in file_names:
            genome2 = compressed_file.split('/')[-1].split('_')[0].upper()#.split('.gz')[0]
            print("genome 2 is {}".format(genome2))
            full_file_path = os.path.join(folder, compressed_file)
            candidates = process_file(full_file_path)
            key = tuple([genome1, genome2])
            if not key in dictionary:
                dictionary[key] = list()
            dictionary[key].extend(candidates)

def main():
    parser = argparse.ArgumentParser(description='Compare two Darwin result DBs.')
    parser.add_argument('allalldir', help='The dir with gz files from allall')
    parser.add_argument('protclusterdir', help='The dir with files from protcluster')

    args = parser.parse_args()

    ref_dict = dict()
    data_dict = dict()

    fill_dictionary(args.allalldir, ref_dict)
    fill_dictionary_nogz(args.protclusterdir, data_dict)

    if set(ref_dict) != set(data_dict):
        print(set(ref_dict))
        print(set(data_dict))
        print('Lengths: {} and {}'.format(len(set(ref_dict)), len(set(data_dict))))
        print(set(ref_dict).symmetric_difference(set(data_dict)))
        raise ValueError('The dictionaries do not have the same keys (genomes)!')

    reported_by_both = 0
    reported_by_ref_only = 0
    reported_by_data_only = 0
    total_pairs_in_ref = 0
    total_pairs_in_data = 0

    additional_pairs_in_ref = dict()
    additional_pairs_in_data = dict()

    for ref_key in ref_dict:
        list_comp = ListComparer(ref_dict[ref_key], data_dict[ref_key])

        reported_by_both += list_comp.get_total_number_in_both()
        reported_by_ref_only += len(list_comp.only_in_list1())
        reported_by_data_only += len(list_comp.only_in_list2())
        total_pairs_in_ref += list_comp.get_total_number_in_list1()
        total_pairs_in_data += list_comp.get_total_number_in_list2()

        additional_pairs_in_ref[ref_key] = list_comp.only_in_list1()
        additional_pairs_in_data[ref_key] = list_comp.only_in_list2()

    print(additional_pairs_in_data)
    percent_both = round(float(reported_by_both)/total_pairs_in_ref, 7) * 100
    percent_reference = round(float(reported_by_ref_only)/total_pairs_in_ref, 7) * 100
    percent_data = round(float(reported_by_data_only)/total_pairs_in_ref, 7) * 100

    output = ''
    output += 'Reported by both: {} ({}%)\n'.format(reported_by_both, percent_both)
    output += 'Reported by reference only: {} ({}%)\n'.format(reported_by_ref_only, percent_reference)
    output += 'Reported by new only: {} ({}%)\n'.format(reported_by_data_only, percent_data)
    output += 'Total reported significands: {}'.format(total_pairs_in_data)

    print(output)

if __name__ == "__main__":
    main()
