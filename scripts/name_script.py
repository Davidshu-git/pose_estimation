import os
import argparse
import shutil
import re


def get_child_list(root_dir):
    return os.listdir(root_dir)


def get_class_list(root_dir):
    children_name_list = os.listdir(root_dir)
    destination_dir = os.path.join(root_dir, children_name_list[0])
    return os.listdir(destination_dir)


def rename_file_for_origin(root_dir):
    child_name_list = get_child_list(root_dir)
    class_name_list = get_class_list(root_dir)
    for child_name in child_name_list:
        for class_name in class_name_list:
            destination_dir = os.path.join(root_dir, child_name, class_name)
            file_name_list = os.listdir(destination_dir)
            if (len(file_name_list) != 0):
                os.chdir(destination_dir)
                i = 0
                for old_name in file_name_list:
                    new_name = class_name + '_' + child_name + '_' + 'right' + f'_{i}' + '.mp4'
                    i += 1
                    os.rename(old_name, new_name)


def move_file_from_origin_to_destinaion(root_dir, destination_dir):
    child_name_list = get_child_list(root_dir)
    class_name_list = get_class_list(root_dir)
    for child_name in child_name_list:
        for class_name in class_name_list:
            child_class_dir = os.path.join(root_dir, child_name, class_name)
            file_name_list = os.listdir(child_class_dir)
            if (len(file_name_list) != 0):
                for file_name in file_name_list:
                    file_dir = os.path.join(root_dir, child_name, class_name, file_name)
                    shutil.move(file_dir, destination_dir)


def file_name_mask(file_name, i):
    file_name_split_list = re.split('，|_|\.', file_name)
    if (file_name_split_list[3] == 'right'):
        file_name_split_list[3] = '1'
    else:
        file_name_split_list[3] = '0'
    file_name_result = (str(i).zfill(6) + '_' + file_name_split_list[1] + '_'
                        + file_name_split_list[3])
    return file_name_result


def data_name_mask(root_dir):
    file_name_list = os.listdir(root_dir)
    os.chdir(root_dir)
    i = 0
    for file_name in file_name_list:
        new_name = file_name_mask(file_name, i) + '.mp4'
        i += 1
        os.rename(file_name, new_name)


def main(root_dir, destination_dir):
    rename_file_for_origin(root_dir)
    move_file_from_origin_to_destinaion(root_dir, destination_dir)
    # data_name_mask(root_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("origin_dir", help="origin path")
    parser.add_argument("destination_dir", help="destination path")
    args = parser.parse_args()
    main(args.origin_dir, args.destination_dir)
