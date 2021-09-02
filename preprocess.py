import pydicom
import pandas as pd
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt
from matplotlib_venn import venn2_unweighted
from glob import glob
from tqdm import tqdm
import utils
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)

study_vet_map = {
    '1': 'W',
    '2': 'S',
    '3': 'JY',
    '4': 'JH'
}


def _get_dir_structure(root_dir):
    # get sub-dirs
    case_dirs = os.listdir(root_dir)
    case_dirs = sorted(filter(lambda d: os.path.isdir(os.path.join(root_dir, d)), case_dirs))

    # read annotation files
    case_imgdir_dict = {}
    case_annolist_dict = {}
    anno_count = 0
    for case_name in case_dirs:
        xmlfile = os.path.join(root_dir, case_name, 'CDViewer', 'burn.xml')

        annolist = utils.get_annotation_list(xmlfile)
        # [(study_no, study_name, series_desc, filename, anno_text), ...]

        if len(annolist) > 0:
            case_imgdir_dict[case_name] = _get_study_dir(os.path.join(root_dir, case_name, 'IMAGE', 'DCM'))
            case_annolist_dict[case_name] = annolist
            anno_count += len(annolist)

    print(f'{anno_count} annotations found in {len(case_annolist_dict)} cases.')

    return case_imgdir_dict, case_annolist_dict


def _get_study_dir(dcm_dir):
    study_dirs = os.listdir(dcm_dir)
    study_dirs = sorted(filter(lambda d: os.path.isdir(os.path.join(dcm_dir, d)), study_dirs))

    dcm_count = -1
    first_study_dir = ''
    for sd in study_dirs:
        filenames = glob(os.path.join(dcm_dir, sd, 'I*'))
        if len(filenames) > 0:
            if dcm_count < 0:
                first_study_dir = os.path.join(dcm_dir, sd)
                dcm_count = len(filenames)
                # print(dcm_count)
            else:
                assert dcm_count == len(filenames)

    return first_study_dir


def _copy_images(case_imgdir_dict, data_dir):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print('target root folder created.')

        for case_name, imgsrc_dir in tqdm(case_imgdir_dict.items()):
            shutil.copytree(imgsrc_dir, os.path.join(data_dir, case_name))
        print('dcm image copy completed.')


def _parse_annotations(case_annolist_dict, data_dir):
    annodata_list = []
    for case_name, anno_list in tqdm(case_annolist_dict.items()):
        for annotation in anno_list:
            study_no, study_name, series_desc, filename, annotation_text = annotation
            assert case_name == study_name[:len(case_name)]

            vet_code = study_name.split('-')[-1]
            vet = study_vet_map[vet_code]

            seq = None
            if 'Post' in series_desc:
                seq = 'Ps'
            elif 'Pre' in series_desc:
                # skip pre-contrast sequence images
                continue

            filepath = os.path.join(data_dir, case_name, filename)

            dcm = pydicom.read_file(filepath)
            z = float(dcm.ImagePositionPatient[2])

            xc, yc, r = utils.get_annotation_info(annotation_text)

            annodata_list.append((case_name, study_no, vet, seq, xc, yc, z, r, filepath))

    print(f'{len(annodata_list)} annotations are converted.')

    return annodata_list


def _write_csv(annodata_list, data_dir):
    csv_file = os.path.join(data_dir, 'annotations.csv')
    with open(csv_file, 'w') as file:
        # header
        header = ['case', 'study', 'vet', 'seq', 'x', 'y', 'z', 'r', 'filepath']
        file.write(','.join(header) + '\n')

        # body
        lines = map(lambda t: ','.join(map(_str, t)) + '\n', annodata_list)
        file.writelines(lines)

    print('csv file created.')


def _str(elem):
    if isinstance(elem, float):
        return f'{elem:.4f}'.rstrip('0').rstrip('.')
    else:
        return str(elem)


##################################################################


def _get_distance(p1, p2):
    return np.sqrt(np.sum((p2 - p1) ** 2, axis=-1))


def _get_intersection_area(c1, c2):
    p1 = c1[..., :-1]
    p2 = c2[..., :-1]
    r1 = c1[..., -1]
    r2 = c2[..., -1]
    d = _get_distance(p1, p2)
    # print('p1', p1.shape, p1)
    # print('p2', p2.shape, p2)
    # print('r1', r1.shape, r1)
    # print('r2', r2.shape, r2)
    # print('d', d.shape, d)

    area = np.zeros(d.shape)

    # one circle is entirely enclosed in the other
    mask = d <= np.abs(r1 - r2)
    area[mask] = np.pi * np.minimum(r1, r2)[mask] ** 2
    # print('r1 - r2', np.abs(r1 - r2))
    # print(d <= np.abs(r1 - r2))
    # print(idxs)
    # print(area)

    # intersection
    mask = np.logical_and(d > np.abs(r1 - r2), d < (r1 + r2))
    r1_2 = r1 ** 2
    r2_2 = r2 ** 2
    d_2 = d ** 2
    alpha = np.arccos((d_2 + r1_2 - r2_2) / (d * r1 * 2))
    beta = np.arccos((d_2 + r2_2 - r1_2) / (d * r2 * 2))
    inter = r1_2 * alpha + r2_2 * beta - 0.5 * (r1_2 * np.sin(alpha * 2) + r2_2 * np.sin(beta * 2))
    area[mask] = inter[mask]
    # print('inter', inter)
    # print(d > np.abs(r1 - r2))
    # print(d < (r1 + r2))
    # print(np.logical_and(d > np.abs(r1 - r2), d < (r1 + r2)))
    # print(idxs)
    # print(area)

    # intersection over union
    iou = area / ((r1**2 + r2**2) * np.pi - area)

    return area, iou


def _compare_two(df_src, vet_a, vet_b, verbose=0, outcsv=None):
    # generate sub data frame
    mask = (df_src['vet'] == vet_a) | (df_src['vet'] == vet_b)
    df_ab = df_src.loc[mask, :]
    print(f'{len(df_ab)} size of filtered data frame for "{vet_a}" and "{vet_b}" has been generated.')

    anno_a_count = 0
    anno_b_count = 0
    anno_ab_count = 0

    matched_annotations = []

    groups = df_ab.groupby(['case', 'z'])

    # for the same case and slice(z)
    for key, df in groups:

        # if key[0] in ['LN0026', 'LN0002', 'LN0018']:
        #     continue

        anno_a_list, anno_b_list = [], []

        for index, row in df.iterrows():
            anno = list(row[['x', 'y', 'z', 'r']])
            if row['vet'] == vet_a:
                anno_a_list.append(anno)
            elif row['vet'] == vet_b:
                anno_b_list.append(anno)
            else:
                print('error')
                return

        a_and_b = 0
        if len(anno_a_list) > 0 and len(anno_b_list) > 0:
            anno_a_arr = np.array(anno_a_list)
            anno_b_arr = np.array(anno_b_list)
            area, iou = _get_intersection_area(np.expand_dims(anno_a_arr, axis=1), anno_b_arr)
            if verbose > 2:
                print('A array:', anno_a_arr.shape, anno_a_arr)
                print('B array:', anno_b_arr.shape, anno_b_arr)
                print('Intersection area:', area.shape, area)
                print('IOU:', iou.shape, iou)

            while np.count_nonzero(iou) > 0:
                i, j = np.unravel_index(np.argmax(iou), iou.shape)
                matched_annotations.append((*key, anno_a_arr[i], anno_b_arr[j], area[i, j], iou[i, j]))
                a_and_b += 1

                area[i, :] = 0
                area[:, j] = 0
                iou[i, :] = 0
                iou[:, j] = 0

        if verbose > 1:
            print(f'{key[0]}, {key[1]}, {vet_a}: {len(anno_a_list)}, {vet_b}: {len(anno_b_list)}, matched: {a_and_b}'
                  + '\n' * (verbose > 2))

        anno_a_count += len(anno_a_list)
        anno_b_count += len(anno_b_list)
        anno_ab_count += a_and_b

    agreement_rate = anno_ab_count / (anno_a_count + anno_b_count - anno_ab_count)

    # print numbers by case
    matched_df = None
    grp = None
    if verbose:
        matched_df = pd.DataFrame(
            matched_annotations, columns=['case', 'z', 'circle1', 'circle2', 'intersect', 'iou']
        )
        grp = df_ab.groupby('case')
        print('\nCases')
        for case, df in grp:
            # if case in ['LN0026', 'LN0002', 'LN0018']:
            #     continue
            a_num = len(df.loc[df['vet'] == vet_a])
            b_num = len(df.loc[df['vet'] == vet_b])
            matched_num = len(matched_df.loc[matched_df['case'] == case])
            print(f'Case: {case},  {vet_a}: {a_num},  {vet_b}: {b_num},  matched: {matched_num}')
        print('')

    # write csv file
    if outcsv:
        if matched_df is None:
            matched_df = pd.DataFrame(
                matched_annotations, columns=['case', 'z', 'circle1', 'circle2', 'intersect', 'iou']
            )
        if grp is None:
            grp = df_ab.groupby('case')

        with open(outcsv, 'w') as fp:
            # header
            header = ['case', 'A', 'B', 'nA', 'nB', 'nAB']
            fp.write(','.join(header) + '\n')

            # body
            for case, df in grp:
                a_num = len(df.loc[df['vet'] == vet_a])
                b_num = len(df.loc[df['vet'] == vet_b])
                matched_num = len(matched_df.loc[matched_df['case'] == case])
                fp.write(f'{case}, {vet_a}, {vet_b}, {a_num}, {b_num}, {matched_num}\n')

        print(f'{outcsv} file saved.')

    print(vet_a, anno_a_count, ',', vet_b, anno_b_count, ', matched', anno_ab_count)
    print(f'agreement rate: {agreement_rate * 100: .2f}%')
    print(f'matched / {vet_a}: {anno_ab_count / anno_a_count * 100: .2f}%')
    print(f'matched / {vet_b}: {anno_ab_count / anno_b_count * 100: .2f}%')
    print('')

    if verbose:
        out = venn2_unweighted(
            subsets=(anno_a_count-anno_ab_count, anno_b_count-anno_ab_count, anno_ab_count),
            set_labels=(vet_a, vet_b),
            set_colors=('skyblue', 'g'),
            alpha=0.4
        )
        for text in out.set_labels:
            text.set_fontsize(16)
        for text in out.subset_labels:
            text.set_fontsize(16)
        plt.show()

    # for elem in matched_annotations:
    #     print(elem)


def build_dataset(source_dir, data_dir):
    case_imgdir_dict, case_annolist_dict = _get_dir_structure(source_dir)
    _copy_images(case_imgdir_dict, data_dir)
    annodata_list = _parse_annotations(case_annolist_dict, data_dir)
    _write_csv(annodata_list, data_dir)


def analyze_annotations(csv_path):
    df = pd.read_csv(csv_path)
    print(f'Data frame of size {len(df)} has been loaded.\n')

    verbose = 1
    _compare_two(df, 'W', 'S', verbose=verbose, outcsv='./result_W_S.csv')
    _compare_two(df, 'W', 'JY', verbose=verbose, outcsv='./result_W_JY.csv')
    _compare_two(df, 'W', 'JH', verbose=verbose, outcsv='./result_W_JH.csv')
    _compare_two(df, 'S', 'JY', verbose=verbose, outcsv='./result_S_JY.csv')
    _compare_two(df, 'S', 'JH', verbose=verbose, outcsv='./result_S_JH.csv')
    _compare_two(df, 'JY', 'JH', verbose=verbose, outcsv='./result_JY_JH.csv')


if __name__ == '__main__':
    _source_dir = './20210514_Unblind_LN0001_0030'
    _data_dir = './data/20210514_preA_30_cases_unblind'
    _csv_path = os.path.join(_data_dir, 'annotations.csv')

    # 1. navigate case dirs, collect annotations from .xmls, and save them into a .csv file.
    # build_dataset(_source_dir, _data_dir)

    # 2. load annotation from .csv file and analyze
    analyze_annotations(_csv_path)

