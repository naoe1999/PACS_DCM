import xml.etree.ElementTree as et
import numpy as np
import pydicom
import cv2
import csv
import os
from tqdm import tqdm


def readAnnoXml(xmlfile):
    tree = et.parse(xmlfile)
    study = tree.find('study')
    series_list = study.findall('series')

    anno_dict = {}

    for series in series_list:
        inst_list = series.findall('instance')

        for inst in inst_list:
            filename = inst.find('file_name').text
            annotxt = inst.find('annotation_lob').text

            if annotxt is not None:
                txts = annotxt.split('\\')
                n = int(txts[0])
                shape_list = txts[1:n+1]
                anno_dict[filename] = shape_list

    return anno_dict


def readCaseAnnoXml(xmlfile):
    tree = et.parse(xmlfile)
    case_list = tree.findall('study')

    case_anno_dict = {}
    for case in case_list:
        case_num = case.find('study_key').text
        series_list = case.findall('series')

        anno_dict = {}
        for series in series_list:
            inst_list = series.findall('instance')

            for inst in inst_list:
                filename = inst.find('file_name').text
                annotxt = inst.find('annotation_lob').text

                if annotxt is not None:
                    txts = annotxt.split('\\')
                    n = int(txts[0])
                    shape_list = txts[1:n+1]
                    anno_dict[filename] = shape_list

        case_anno_dict[case_num] = anno_dict

    return case_anno_dict


def get_annotation_list(xmlfile):
    anno_list = []

    tree = et.parse(xmlfile)
    study_list = tree.findall('study')
    for study in study_list:
        study_no = study.find('study_key').text
        study_name = study.find('origin_pid').text

        series_list = study.findall('series')
        for series in series_list:
            series_desc = series.find('series_desc').text

            inst_list = series.findall('instance')
            for inst in inst_list:
                filename = inst.find('file_name').text
                annotxt = inst.find('annotation_lob').text

                if annotxt is not None:
                    txts = annotxt.split('\\')
                    n = int(txts[0])
                    shape_list = txts[1:n+1]
                    for shape in shape_list:
                        anno_list.append((study_no, study_name, series_desc, filename, shape))

    return anno_list


def get_annotation_info(annotxt):
    [xc, yc], r = _getBoundingCircle(annotxt)
    return xc, yc, r


def getPatientCaseSet(path):
    class Patient:
        def __init__(self, case_num, patient_id, patient_copy):
            self.case_num = case_num
            self.patient_id = patient_id
            self.patient_copy = patient_copy

    # 1. folder --> Patient list
    case_nums = sorted(os.listdir(path))

    patients = []
    for cnum in case_nums:
        cpath = os.path.join(path, cnum)
        if os.path.isdir(cpath):
            dcm_sample_file = sorted(os.listdir(cpath))[0]
            dcm_sample_file = os.path.join(cpath, dcm_sample_file)
            ds = pydicom.read_file(dcm_sample_file)

            pstr = ds.PatientID
            pid = pstr[:pstr.index('-')]
            pcp = pstr[pstr.index('-') + 1:]
            # print(cnum, pid, pcp)

            new_patient = Patient(cnum, pid, pcp)
            patients.append(new_patient)

    patients.sort(key=lambda x: x.patient_id + ' ' + x.patient_copy)

    # 2. Patient list --> {patient_id: [(patient_copy, case_num), ... ], ...}
    patient_case_dict = {}
    patient_id_list = []
    for p in patients:
        cnum = p.case_num
        pid = p.patient_id
        pcp = p.patient_copy

        if pid in patient_case_dict:
            patient_case_dict[pid].append((pcp, cnum))
        else:
            patient_case_dict[pid] = [(pcp, cnum)]
            patient_id_list.append(pid)

    return patient_case_dict, patient_id_list


def getDicomImageList(path):
    dcm_files = sorted(os.listdir(path))

    slice_ds_image_list = []
    for filename in tqdm(dcm_files):
        ds = pydicom.read_file(os.path.join(path, filename))
        slice_ds_image_list.append((filename, ds, ds.pixel_array))

    return slice_ds_image_list


def getImageList(slice_ds_image_list, anno_dict_list=None, window=True):
    gmin, gmax = 10000, 0
    if anno_dict_list is None:
        anno_dict_list = []

    slice_img_list = []
    for (slice_name, ds, pxarr) in slice_ds_image_list:
        img = _getHU(ds, pixelarray=pxarr, window=window)
        gmax = np.maximum(gmax, np.max(img))
        gmin = np.minimum(gmin, np.min(img))
        slice_img_list.append((slice_name, img))

    img_list = []
    for (slice_name, img) in slice_img_list:
        img = _normalize(img, gmax, gmin)
        img = cv2.merge((img, img, img))
        for (anno_dict, color) in anno_dict_list:
            if slice_name in anno_dict:
                img = _drawAnnotations(img, anno_dict[slice_name], color=_colormap[color])

        # horizontal flip
        img = cv2.flip(img, 1)
        cv2.putText(img, slice_name, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (1., 1., 1.), 1)
        img = (img * 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img_list.append(img)

    return img_list


def generateCsv(ds_dict, anno_dict, caseno, csvout):
    bc_dict = {}

    with open(csvout, 'w', encoding='utf-8') as cf:
        wr = csv.writer(cf)

        for f, shape_list in anno_dict.items():
            ds, _ = ds_dict[f]
            z = float(ds.SliceLocation)

            bc_list = []
            for shapetxt in shape_list:
                center2D, r = _getBoundingCircle(shapetxt)
                if r is None:
                    continue
                wr.writerow([caseno, f'{center2D[0]:.2f}', f'{center2D[1]:.2f}', f'{z:.0f}', f'{r:.4f}', f])
                bc_list.append(center2D + [z, r])

            if len(bc_list) > 0:
                bc_dict[f] = bc_list

    return bc_dict


def generateImageDict(ds_dict, window=False, annotation=False, bcircle=False, anno_dict=None, bc_dict=None):
    gmax = 0
    gmin = 1e5

    img_dict = {}
    for f, (ds, pa) in ds_dict.items():
        img = _getHU(ds, pixelarray=pa, window=window)
        gmax = np.maximum(gmax, np.max(img))
        gmin = np.minimum(gmin, np.min(img))
        img_dict[f] = img

    for f, img in img_dict.items():
        img = _normalize(img, gmax, gmin)
        img = cv2.merge((img, img, img))

        if annotation and anno_dict is not None:
            if f in anno_dict:
                img = _drawAnnotations(img, anno_dict[f])

        if bcircle and bc_dict is not None:
            if f in bc_dict:
                img = _drawBCircles(img, bc_dict[f])

        img = (img * 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = cv2.flip(img, 1)
        img_dict[f] = img

    return img_dict


###############################################################
# internal functions

_colormap = {
    'red': (255, 0, 0),
    'yellow': (255, 255, 0),
    'green': (0, 255, 0),
    'cyan': (0, 255, 255),
    'blue': (0, 0, 255),
    'magenta': (255, 0, 255),
    'white': (255, 255, 255),
    'black': (0, 0, 0)
}


def _getHU(ds, pixelarray=None, window=False):
    slope = ds.RescaleSlope
    icpt = ds.RescaleIntercept

    if pixelarray is not None:
        img = pixelarray
    else:
        img = ds.pixel_array

    img[img == -2000] = 0

    HU = img * slope + icpt

    if window:
        wc = ds.WindowCenter
        if type(wc) is pydicom.multival.MultiValue:
            wc = wc[0]
        ww = ds.WindowWidth
        if type(ww) is pydicom.multival.MultiValue:
            ww = ww[0]
        wl = wc - ww//2
        wh = wc + ww//2
        HU = np.minimum(np.maximum(HU, wl), wh)

    return HU


def _normalize(img, gmax=None, gmin=None):
    if gmax is None:
        gmax = np.max(img)
    if gmin is None:
        gmin = np.min(img)
    return (img - gmin) / (gmax - gmin)


def _drawAnnotations(img, shapetxt_list, color=None):
    linewidth = 1
    for shapetxt in shapetxt_list:
        shape_type, points, color256 = _parseAnnoText(shapetxt)
        if color:
            color256 = color    # override
        color_norm = np.array(color256) / 255.

        # circle
        if shape_type == 'CC':
            x1, y1 = points[0]
            x2, y2 = points[1]
            r = int(np.sqrt((x2 - x1)**2 + (y2 - y1)**2))
            cv2.circle(img, (x1, y1), r, color_norm, linewidth)

        # rectangle
        elif shape_type == 'RT':
            cv2.rectangle(img, points[0], points[1], color_norm, linewidth)

        # free line
        elif shape_type == 'FL' or shape_type == 'F2':
            points = np.asarray(points)
            cv2.polylines(img, [points], False, color_norm, linewidth)

        # arrow
        elif shape_type == 'AR':
            x1, y1 = points[0]
            cv2.line(img, points[0], points[1], color_norm, linewidth)
            cv2.putText(img, '*', (x1-4, y1+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_norm, linewidth)

        # ellipse (measure area)
        elif shape_type == 'ME':
            x1, y1 = points[0]
            x2, y2 = points[1]
            bx, by = (x1 + x2)//2, (y1 + y2)//2
            bw, bh = abs(x1 - x2), abs(y1 - y2)
            cv2.ellipse(img, ((bx, by), (bw, bh), 0), color_norm, linewidth)

        # line (measure length)
        elif shape_type == 'L2':
            cv2.line(img, points[0], points[1], color_norm, linewidth)

    return img


def _drawBCircles(img, bcircle_list):
    for bc in bcircle_list:
        x, y, z, r = bc
        color = (1.0, 0.5, 0.5)
        cv2.circle(img, (round(x), round(y)), round(r), color, 1, cv2.LINE_AA)

    return img


def _getBoundingCircle(shapetxt):
    shape_type, points, _ = _parseAnnoText(shapetxt)

    if shape_type == 'CC':
        xc, yc = points[0]
        xr, yr = points[1]
        r = np.sqrt((xr - xc) ** 2 + (yr - yc) ** 2)

    elif shape_type == 'RT' or shape_type == 'ME':
        x1, y1 = points[0]
        x2, y2 = points[1]
        xc = (x1 + x2) / 2
        yc = (y1 + y2) / 2
        r = max(abs(xc-x1), abs(yc-y1))

    elif shape_type == 'AR' or shape_type == 'L2':
        x1, y1 = points[0]
        x2, y2 = points[1]
        xc = (x1 + x2) / 2
        yc = (y1 + y2) / 2
        r = np.sqrt((x2 - xc) ** 2 + (y2 - yc) ** 2)

    elif shape_type == 'FL':
        points = np.asarray(points)
        x1, y1 = np.min(points, axis=0)
        x2, y2 = np.max(points, axis=0)
        xc = (x1 + x2) // 2
        yc = (y1 + y2) // 2
        r = max(abs(xc-x1), abs(yc-y1))

    else:
        xc, yc, r = None, None, None

    return [xc, yc], r


def _parseAnnoText(shapetxt):
    shape_type = shapetxt[:2]
    contents = shapetxt[2:]

    subcontents = contents.split('|')
    point_str = subcontents[0]
    color_str = subcontents[1]

    # points
    point_elem = point_str.split()
    if shape_type == 'FL':
        point_elem = point_elem[1:]
    elif shape_type == 'ME':
        point_elem = point_elem[:6]
    elif shape_type == 'F2':
        point_count = int(point_elem[0]) - 1
        point_elem = point_elem[1: point_count * 2 + 1]

    xs = point_elem[0::2]
    ys = point_elem[1::2]
    points = [(int(float(x)), int(float(y))) for x, y in zip(xs, ys)]

    # color
    r = int(color_str[0:3])
    g = int(color_str[3:6])
    b = int(color_str[6:9])
    color = (r, g, b)

    return shape_type, points, color

