import os
import sys
import cv2
import utils


IMGROOT = './20210308_LN0002-2/IMAGE/DCM'
ANNOFILE = './20210308_LN0002-2/CDViewer/burn.xml'


def mouse_function(event, x, y, flags, param):
    # print(x, y, event, flags, param)

    global slice_idx, img_list

    if event == cv2.EVENT_MOUSEWHEEL:
        if flags > 0:
            slice_idx = min(slice_idx + 1, len(img_list) - 1)
        elif flags < 0:
            slice_idx = max(slice_idx - 1, 0)


def main(pidx=0):
    print('\nloading patient list ... ', end='', flush=True)
    patient_case_dict, patient_id_list = utils.getPatientCaseSet(IMGROOT)
    # patient_case_dict = {'p_id': [('copy', 'case_num'), ...], ... }
    # patient_id_list = ['p_id', ...]
    print('done')
    print(patient_id_list)
    print(patient_case_dict)

    print('\nloading annotation ... ', end='', flush=True)
    case_anno_dict = utils.readCaseAnnoXml(ANNOFILE)
    # {'case_num': {'slice_file_name': [shape_str, ...], ... }, ... }
    print('done')

    for i, (k, v) in enumerate(sorted(case_anno_dict.items())):
        print(i//2 +1, k)
        for k_, v_ in v.items():
            print(k_, v_)
        print('----------'*10)

    patient_idx = pidx
    slice_idx = 0

    slice_img_list = None
    img_list = None

    window_on = True
    anno_1_on = True
    anno_2_on = True
    bcircle_on = False

    onf = lambda x: 'ON' if x else 'OFF'

    cv2.namedWindow('CT', cv2.WINDOW_GUI_NORMAL)
    cv2.setMouseCallback('CT', mouse_function)

    while True:
        if slice_img_list is None:
            pid = patient_id_list[patient_idx]
            _, case_num = patient_case_dict[pid][0]

            print('\nloading dicom files ... ', end='\n', flush=True)
            path = os.path.join(IMGROOT, case_num)
            slice_img_list = utils.getDicomImageList(path)
            # [(dicom_file_name, dicom_ds, image_array), ...]
            img_list = None
            slice_idx = 0
            print('done')

        if img_list is None:
            pid = patient_id_list[patient_idx]

            copy_case_list = patient_case_dict[pid]
            anno_dict_list = []
            for (cp, case_num) in copy_case_list:
                if cp == '1' and anno_1_on:
                    anno_dict_list.append((case_anno_dict[case_num], 'red'))
                if cp == '2' and anno_2_on:
                    anno_dict_list.append((case_anno_dict[case_num], 'yellow'))

            print('updating images ... ', end='', flush=True)
            img_list = utils.getImageList(slice_img_list, anno_dict_list, window_on)
            print('done')

        img = img_list[slice_idx]
        cv2.imshow('CT', img)

        k = cv2.waitKeyEx(30)

        if k in map(ord, map(str, range(10))):
            new_p_idx = k - ord('1')
            if new_p_idx < 0:
                new_p_idx += 10

            if new_p_idx != patient_idx:
                patient_idx = new_p_idx
                slice_img_list = None

        elif k in map(ord, ['a', 'w', 's', 'd']):
            anno_1_on = k == ord('w') or k == ord('a')
            anno_2_on = k == ord('s') or k == ord('a')
            img_list = None
            print(f'Annotation 1: {onf(anno_1_on)}, Annotation 2: {onf(anno_2_on)}')

        elif k == ord('x'):
            window_on = not window_on
            img_list = None
            print(f'Window: {onf(window_on)}')

        elif k == ord('c'):
            bcircle_on = not bcircle_on
            img_list = None
            print(f'Bounding circle: {onf(bcircle_on)}')

        elif k == 63234 or k == ord('['):
            slice_idx = max(slice_idx - 1, 0)
        elif k == 63235 or k == ord(']'):
            slice_idx = min(slice_idx + 1, len(img_list) - 1)
        elif k == ord('{'):
            slice_idx = 0
        elif k == ord('}'):
            slice_idx = len(img_list) - 1

        elif k == 27:
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    idx = int(sys.argv[1]) - 1 if len(sys.argv) > 1 else 0
    main(idx)

