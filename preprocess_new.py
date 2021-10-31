# for case 42 ~

import pydicom
import os
import os.path as osp
import shutil
from glob import glob
from tqdm import tqdm
import utils


study_vet_map = {
    '1': 'W',
    '2': 'S',
    '3': 'JH',      # from 42~
    '4': 'JY'       # from 42~
}


def copy_images(src_dir, tgt_dir):
    case_names = _get_case_list(src_dir)

    for case_name in case_names:
        tgt_case_dir = osp.join(tgt_dir, case_name)
        if osp.exists(tgt_case_dir):
            print(f'{case_name}: folder already exists. copy skipped!')
            continue

        img_file_paths = _get_image_files(src_dir, case_name)
        if len(img_file_paths) == 0:
            print(f'{case_name}: no images. copy skipped!')
            continue

        dcm = pydicom.read_file(img_file_paths[0])
        first_series_uid = dcm.SeriesInstanceUID

        src_files = []
        for img_file_path in img_file_paths:
            dcm = pydicom.read_file(img_file_path)
            if dcm.SeriesInstanceUID == first_series_uid:
                src_files.append(img_file_path)
            else:
                break

        os.makedirs(tgt_case_dir)
        for src_file in src_files:
            shutil.copy(src_file, tgt_case_dir)
        print(f'{case_name}: {len(src_files)} copied.')

    print('dcm image copy completed.')


def get_case_annostr_dict(src_dir):
    return {c: utils.get_annotation_list_v2(_get_anno_file(src_dir, c)) for c in _get_case_list(src_dir)}


def parse_annotations(case_annos_dict):
    annotations = []
    for case_name, annos in tqdm(case_annos_dict.items()):
        for anno in annos:
            study_name, series_desc, z_global, shape_text = anno
            assert case_name == study_name
            assert case_name == series_desc[:len(case_name)]

            vet_code = series_desc.split('-')[-1]
            vet = study_vet_map[vet_code]

            xc, yc, r = utils.get_annotation_info(shape_text)
            # xc, yc, r:  pixel (voxel) coordinate
            # z:  global coordinate

            annotations.append((case_name, vet, z_global, xc, yc, r))

    print(f'{len(annotations)} annotations are converted.')
    return annotations


def write_csv(annotations, csv_filepath):
    with open(csv_filepath, 'w') as file:
        # header
        header = ['case', 'vet', 'z_global', 'x_pixel', 'y_pixel', 'r_pixel']
        file.write(','.join(header) + '\n')

        # body
        lines = map(lambda t: ','.join(map(_str, t)) + '\n', annotations)
        file.writelines(lines)

    print('csv file generated.')


def _get_case_list(root_dir):
    return sorted(filter(lambda d: osp.isdir(osp.join(root_dir, d)), os.listdir(root_dir)))


def _get_image_files(root_dir, case_name):
    return sorted(glob(osp.join(root_dir, case_name, 'IMAGE', 'DCM', '*', 'I*')))


def _get_anno_file(root_dir, case_name):
    return osp.join(root_dir, case_name, 'CDViewer', 'burn.xml')


def _str(elem):
    if isinstance(elem, float):
        return f'{elem:.4f}'.rstrip('0').rstrip('.')
    else:
        return str(elem)


def build_dataset(source_dir, data_dir, csv_filepath):
    # copy dicom images (without duplication)
    copy_images(source_dir, data_dir)

    # .xml files -> .csv file
    case_annos_dict = get_case_annostr_dict(source_dir)
    annotations = parse_annotations(case_annos_dict)
    write_csv(annotations, csv_filepath)


if __name__ == '__main__':
    _source_dir = '../source_data'
    _data_dir = '../anno_data'
    _csv_filepath = osp.join(_data_dir, 'annotations.csv')

    # navigate case dirs, collect annotations from .xmls, and save them to a .csv file.
    build_dataset(_source_dir, _data_dir, _csv_filepath)

