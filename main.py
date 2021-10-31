import argparse
import os
import cv2
import pandas as pd
import numpy as np
import pydicom
from tqdm import tqdm
from glob import glob


### KEY MAPPING
### Mac
# ARROW_LEFT = 63234
# ARROW_RIGHT = 63235
# ARROW_UP = 63232
# ARROW_DOWN = 63233
# TAB = 9
# SPACE = 32
### Windows
ARROW_LEFT = 2424832
ARROW_RIGHT = 2555904
ARROW_UP = 2490368
ARROW_DOWN = 2621440
TAB = 9
SPACE = 32

# DATA DIR
# for development
DATA_ROOT = '../anno_data'
# DATA_ROOT = './data/20210514_preA_30_cases_unblind'

# for distribution
# DATA_ROOT = '../../data/20210514_preA_30_cases_unblind'


WINDOWED = 0x0001
ANNO1 = 0x0002
ANNO2 = 0x0004
ANNO3 = 0x0008
ANNO4 = 0x0010
ANNOALL = ANNO1 | ANNO2 | ANNO3 | ANNO4
ANNOFILL = 0x0020
ZOOM = 0x0040

ZOOM_SIZE_FROM = 200
ZOOM_SIZE_TO = 512

zoom_x = 0
zoom_y = 0
display_mode = WINDOWED | ANNOALL
anno_memory = display_mode & ANNOALL


COLOR_BGR_ANNO1 = (.3, .3, 1.)  # RED
COLOR_BGR_ANNO2 = (.3, 1., 1.)  # YELLOW
COLOR_BGR_ANNO3 = (.3, 1., .3)  # GREEN
COLOR_BGR_ANNO4 = (1., 1., .3)  # CYAN

VET_MAP = {
    'W': (ANNO1, COLOR_BGR_ANNO1),
    'S': (ANNO2, COLOR_BGR_ANNO2),
    'JH': (ANNO3, COLOR_BGR_ANNO3),     # from 42~
    'JY': (ANNO4, COLOR_BGR_ANNO4),     # from 42~
}


eps = 1e-4


def numerical_equals(a: float, b: float):
    diff = abs(a - b)
    return diff < eps


def load_data(data_path):
    # read dirs
    listdir = os.listdir(data_path)
    casename_list = sorted(filter(lambda d: os.path.isdir(os.path.join(data_path, d)), listdir))
    print(f'{len(casename_list)} cases are found in data path "{data_path}"')

    # read csv
    anno_filepath = os.path.join(data_path, 'annotations.csv')
    anno_df = pd.read_csv(anno_filepath)
    print(f'{anno_df["case"].nunique()} cases and {len(anno_df)} annotations are loaded from .csv file.')

    return casename_list, anno_df


def load_images_and_annotations(imgdir, df):
    # load dicom datasets
    dcm_paths = sorted(glob(os.path.join(imgdir, 'I*')))
    dcm_dss = [pydicom.read_file(f) for f in dcm_paths]

    # load images
    print('reading pixel arrays and converting to HU ...')
    imgs = [ds.pixel_array for ds in tqdm(dcm_dss)]
    imgs = np.asarray(imgs, dtype=np.float32)
    print('done. \nimage shape:', imgs.shape)

    # convert to HU
    slope, icpt, w_min, w_max = get_image_info(dcm_dss[0])
    imgs = convert_to_HU(imgs, slope, icpt)

    hu_min, hu_max = np.min(imgs), np.max(imgs)
    info = (hu_min, hu_max, w_min, w_max)

    # load annotations
    annotations = []
    z_poss = []
    for ds in dcm_dss:
        slice_annos = []
        try:
            z_pos = abs(float(ds.ImagePositionPatient[2]))
        except:
            z_pos = abs(float(ds.SliceLocation))

        slice_df = df.loc[numerical_equals(df['z_global'], z_pos)]
        slices = slice_df[['vet', 'x_pixel', 'y_pixel', 'z_global', 'r_pixel']].values.tolist()
        if len(slices) > 0:
            slice_annos.extend(slices)

        annotations.append(slice_annos)
        z_poss.append(f'{z_pos:.2f}')

    print('loaded annotations for current case.')
    return dcm_dss, imgs, annotations, z_poss, info


def get_image_info(ds):
    slope = ds.RescaleSlope
    icpt = ds.RescaleIntercept

    wc = ds.WindowCenter
    ww = ds.WindowWidth
    if isinstance(wc, pydicom.multival.MultiValue):
        wc = wc[0]
    if isinstance(ww, pydicom.multival.MultiValue):
        ww = ww[0]
    wl = wc - ww // 2
    wh = wc + ww // 2

    return slope, icpt, wl, wh


def convert_to_HU(images, slope, icpt):
    assert isinstance(images, np.ndarray)
    images[images == -2000] = 0
    HU = images * slope + icpt
    return HU


def put_annotation(image, annotation, dispmode=0):
    # draw annotation
    anno_count = {'W': 0, 'S': 0, 'JY': 0, 'JH': 0}
    for anno in reversed(annotation):
        vet, x, y, z, r = anno
        anno_count[vet] += 1
        dispmask, color = VET_MAP[vet]
        if dispmode & dispmask:
            cv2.circle(image, (round(x), round(y)), round(r), color,
                       thickness=-1 if dispmode & ANNOFILL else 1)

    return anno_count


def print_info(image, casename, i, zpos, anno_count, dispmode=0):
    # print case name and z position
    cv2.putText(image, casename, (15, 30), cv2.FONT_HERSHEY_DUPLEX, 0.6, (.1, 1., 1.))
    cv2.putText(image, f'SP: {zpos}', (15, 55), cv2.FONT_HERSHEY_DUPLEX, 0.6, (.1, 1., 1.))
    cv2.putText(image, f'Img: {i+1}', (15, 80), cv2.FONT_HERSHEY_DUPLEX, 0.6, (.1, 1., 1.))

    # print annotation display info
    tx = image.shape[1] - 100
    for i, vet in enumerate(['W', 'S', 'JY', 'JH']):
        dispmask, color = VET_MAP[vet]
        cv2.putText(image, f'{vet}: {anno_count[vet]}', (tx, i * 25 + 30), cv2.FONT_HERSHEY_DUPLEX, 0.7,
                    color if dispmode & dispmask else (.4, .4, .4))


def main(args):
    global display_mode, anno_memory, zoom_x, zoom_y

    casename = args.case
    h = w = args.width
    resized = False

    # load all case names and annotation data
    casename_list, anno_df = load_data(args.path)

    dss = None
    images = None
    annotations = None
    z_poss = None
    w_min_norm, w_max_norm = None, None
    j_img = None

    cv2.namedWindow('CT', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('CT', mouse_function)

    while True:

        if dss is None:
            # load dicom images and annotations of the selected case.
            caseimgdir = os.path.join(args.path, casename)
            case_df = anno_df[anno_df['case'] == casename]

            dss, images, annotations, z_poss, info = load_images_and_annotations(caseimgdir, case_df)
            # print(len(dss), images.shape, len(annotations))

            # normalize with global min, max
            hu_min, hu_max, w_min, w_max = info
            w_min_norm = (w_min - hu_min) / (hu_max - hu_min)
            w_max_norm = (w_max - hu_min) / (hu_max - hu_min)

            images = (images - hu_min) / (hu_max - hu_min)
            print(info, f'{w_min_norm:.4f}, {w_max_norm:.4f}')

            j_img = 0

        # load current image sequence
        # ds = dss[j_img]
        img = images[j_img]
        anno = annotations[j_img]
        z_pos = z_poss[j_img]

        # apply lung window?
        if display_mode & WINDOWED:
            img = (img - w_min_norm) / (w_max_norm - w_min_norm)

        # draw annotation
        img = np.expand_dims(img, axis=-1)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        anno_count = put_annotation(img, anno, dispmode=display_mode)

        # flip horizontal
        img = cv2.flip(img, 1)

        # zoom?
        if display_mode & ZOOM:
            zoom(img, zoom_x, zoom_y)

        # slice info
        print_info(img, casename, j_img, z_pos, anno_count, dispmode=display_mode)

        # show image
        cv2.imshow('CT', img)

        if not resized:
            cv2.resizeWindow('CT', w, h)
            resized = True

        k = cv2.waitKeyEx(100)

        # escape
        if k == 27:
            break
        # lung window
        elif k == ord('w'):
            display_mode = display_mode ^ WINDOWED

        # select annotations
        elif k == ord('1'):
            display_mode = display_mode ^ ANNO1
        elif k == ord('2'):
            display_mode = display_mode ^ ANNO2
        elif k == ord('3'):
            display_mode = display_mode ^ ANNO3
        elif k == ord('4'):
            display_mode = display_mode ^ ANNO4
        elif k == ord('5'):
            display_mode = display_mode | ANNOALL
        elif k == ord('`'):
            if display_mode & ANNOALL:
                anno_memory = display_mode & ANNOALL
                display_mode = display_mode & ~ANNOALL
        elif k == TAB:
            anno_mode = display_mode & ANNOALL
            bitcnt = bin(anno_mode).count('1')
            if bitcnt == 4:
                anno_mode = ANNO1
            elif bitcnt == 1:
                anno_mode = (anno_mode << 1) & ANNOALL
            else:
                anno_mode = ANNOALL
            display_mode = display_mode & ~ANNOALL | anno_mode
        elif k == SPACE:
            if display_mode & ANNOALL:
                anno_memory = display_mode & ANNOALL
                display_mode = display_mode & ~ANNOALL
            else:
                display_mode = display_mode | anno_memory

        # fill
        elif k == ord('f'):
            display_mode = display_mode ^ ANNOFILL

        # slice navigation
        elif k == ARROW_LEFT or k == ord(','):          # left
            j_img = max(j_img - 1, 0)
        elif k == ARROW_RIGHT or k == ord('.'):         # right
            j_img = min(j_img + 1, len(images) - 1)
        elif k == ord('<'):
            j_img = 0
        elif k == ord('>'):
            j_img = len(images) - 1
        elif k == ARROW_UP:                             # up
            for j in range(j_img-1, -1, -1):
                if len(annotations[j]) > 0:
                    j_img = j
                    break
        elif k == ARROW_DOWN:                           # down
            for j in range(j_img+1, len(images)):
                if len(annotations[j]) > 0:
                    j_img = j
                    break

        # case navigation
        elif k == ord('['):
            idx = casename_list.index(casename)
            idx = (idx - 1) % len(casename_list)
            casename = casename_list[idx]
            dss = None
        elif k == ord(']'):
            idx = casename_list.index(casename)
            idx = (idx + 1) % len(casename_list)
            casename = casename_list[idx]
            dss = None

        elif k != -1:
            print(k)

    cv2.destroyAllWindows()


def mouse_function(event, x, y, flags, param):
    global display_mode, zoom_x, zoom_y

    if event == cv2.EVENT_LBUTTONDOWN:
        zoom_x, zoom_y = x, y
        display_mode = display_mode ^ ZOOM


def zoom(img, x, y):
    h, w = img.shape[0:2]
    src_x_min = max(min(x - ZOOM_SIZE_FROM // 2, w - ZOOM_SIZE_FROM), 0)
    src_y_min = max(min(y - ZOOM_SIZE_FROM // 2, h - ZOOM_SIZE_FROM), 0)

    src_img = img[src_y_min: src_y_min + ZOOM_SIZE_FROM, src_x_min: src_x_min + ZOOM_SIZE_FROM, :]
    zoom_img = cv2.resize(src_img, (ZOOM_SIZE_TO, ZOOM_SIZE_TO))

    tgt_x_min = max(min(x - ZOOM_SIZE_TO // 2, w - ZOOM_SIZE_TO), 0)
    tgt_y_min = max(min(y - ZOOM_SIZE_TO // 2, h - ZOOM_SIZE_TO), 0)

    tgt_x_max = min(tgt_x_min + ZOOM_SIZE_TO, w)
    tgt_y_max = min(tgt_y_min + ZOOM_SIZE_TO, h)

    img[tgt_y_min: tgt_y_max, tgt_x_min: tgt_x_max, :] = zoom_img


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('case', help='case name')
    parser.add_argument('--width', '-w', type=int, default=1280, help='window width')
    parser.add_argument('--path', '-p', default=DATA_ROOT, help='data folder')
    args = parser.parse_args()

    main(args)

