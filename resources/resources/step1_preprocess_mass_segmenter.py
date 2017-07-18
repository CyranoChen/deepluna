import settings
import helpers
import cv2
import os
import glob
import ntpath
import numpy
import dicom



def preprocess_extradata(start_dir):
    #src_dir = settings.BASE_DIR + "extrasets/"
    dirs_and_files = os.listdir(start_dir)
    dirs = []
    dicom_paths = []
    for dir_or_file in dirs_and_files:
        if os.path.isdir(start_dir + dir_or_file):
            dirs.append(start_dir + dir_or_file + "/")
        else:
            if dir_or_file.endswith(".dcm") or "CT." in dir_or_file:
                dicom_paths.append(start_dir + dir_or_file)

    # patient_imgs = load_patient(dicom_files)
    #patient_pixels = get_pixels_hu(patient_imgs)

    for dicom_path in dicom_paths:
        try:
            dicom_slice = dicom.read_file(dicom_path)
            slice_location = 0
            slice_location = str(int(round(dicom_slice.SliceLocation * 1000)))
            image = dicom_slice.pixel_array
            image = image.astype(numpy.int16)
            image[image == -2000] = 0
            intercept = dicom_slice.RescaleIntercept
            slope = dicom_slice.RescaleSlope
            if slope != 1:
                print("Slope <> 1!!!")
                image = slope * image.astype(numpy.float64)
                image = image.astype(numpy.int16)
            image += numpy.int16(intercept)

            image = helpers.normalize_hu(image)
            tmp = dicom_path.replace(settings.BASE_DIR + "extradata/", "")
            file_name = ntpath.basename(dicom_path)
            patient_id = tmp.replace(file_name, "").replace("/", "")
            file_name = file_name.replace(".dcm", "")
            target_dir = settings.BASE_DIR + "extradata_processed/" + patient_id + "/"
            if not os.path.exists(target_dir):
                print("Patient: ", patient_id)
                os.mkdir(target_dir)
            target_path = target_dir + slice_location + "_" + file_name + "_i.png"
            cv2.imwrite(target_path, image * 255)
        except:
            print("EXCEPTION")


    for sub_dir in dirs:
        preprocess_extradata(sub_dir)


def replace_color(src_image, from_color, to_color):
    data = numpy.array(src_image)   # "data" is a height x width x 4 numpy array
    r1, g1, b1 = from_color  # Original value
    r2, g2, b2 = to_color  # Value that we want to replace it with

    red, green, blue = data[:, :, 0], data[:, :, 1], data[:, :, 2]
    mask = (red == r1) & (green == g1) & (blue == b1)
    data[:, :, :3][mask] = [r2, g2, b2]

    return data


def prepare_overlay_image(src_overlay_path, target_size, antialias=False):
    if os.path.exists(src_overlay_path):
        overlay = cv2.imread(src_overlay_path)
        overlay = replace_color(overlay, (255, 255, 255), (0, 0, 0))
        overlay = replace_color(overlay, (0, 255, 255), (255, 255, 255))
        overlay = overlay.swapaxes(0, 2)
        overlay = overlay.swapaxes(1, 2)
        overlay = overlay[0]
        # overlay = overlay.reshape((overlay.shape[1], overlay.shape[2])
        interpolation = cv2.INTER_AREA if antialias else cv2.INTER_NEAREST
        overlay = cv2.resize(overlay, (target_size, target_size), interpolation=interpolation)
    else:
        overlay = numpy.zeros((target_size, target_size), dtype=numpy.uint8)
    return overlay


def make_segmenter_train_images(model_type):
    dst_dir = settings.SEGMENTER_TRAIN_DIR + model_type + "/"
    src_dir = settings.MANUAL_MASSES_DIR if model_type == "masses" else settings.MANUAL_EMPHYSEMA_DIR
    for file_path in glob.glob(dst_dir + "*.*"):
        os.remove(file_path)

    for overlay_path in glob.glob(src_dir + "overlays/*.png"):
        file_name = ntpath.basename(overlay_path)
        image_path = src_dir + "images/" + file_name
        print(image_path)
        img = cv2.imread(image_path)
        img = cv2.resize(img, dsize=(settings.SEGMENTER_IMG_SIZE, settings.SEGMENTER_IMG_SIZE))
        overlay_img = prepare_overlay_image(overlay_path, settings.SEGMENTER_IMG_SIZE)
        cv2.imwrite(dst_dir + file_name.replace(".png", "_1.png"), img)
        cv2.imwrite(dst_dir + file_name.replace(".png", "_o.png"), overlay_img)


if __name__ == "__main__":
    # preprocess_extradata(settings.BASE_DIR + "extradata/")
    make_segmenter_train_images("masses")

