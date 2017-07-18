import settings
import helpers
import SimpleITK  # conda install -c https://conda.anaconda.org/simpleitk SimpleITK
import numpy
import ntpath
import cv2  # conda install -c https://conda.anaconda.org/menpo opencv3
import shutil
import random
import multiprocessing
import os
import glob
import pandas
import math

random.seed(1321)
numpy.random.seed(1321)

TIANCHI_WORK_DIR = "train_subset"


def normalize(image):
    MIN_BOUND = -1000.0
    MAX_BOUND = 400.0
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image > 1] = 1.
    image[image < 0] = 0.
    return image


def process_image(src_path):
    patient_id = ntpath.basename(src_path).replace(".mhd", "")
    print("Patient: ", patient_id)

    dst_dir = settings.TIANCHI_EXTRACTED_IMAGE_DIR + patient_id + "/"
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)

    itk_img = SimpleITK.ReadImage(src_path)
    img_array = SimpleITK.GetArrayFromImage(itk_img)
    print("Img array: ", img_array.shape)

    origin = numpy.array(itk_img.GetOrigin())      # x,y,z  Origin in world coordinates (mm)
    print("Origin (x,y,z): ", origin)

    direction = numpy.array(itk_img.GetDirection())      # x,y,z  Origin in world coordinates (mm)
    print("Direction: ", direction)


    spacing = numpy.array(itk_img.GetSpacing())    # spacing of voxels in world coor. (mm)
    print("Spacing (x,y,z): ", spacing)
    rescale = spacing / settings.TARGET_VOXEL_MM
    print("Rescale: ", rescale)

    img_array = helpers.rescale_patient_images(img_array, spacing, settings.TARGET_VOXEL_MM)

    img_list = []
    for i in range(img_array.shape[0]):
        img = img_array[i]
        seg_img, mask = helpers.get_segmented_lungs(img.copy())
        img_list.append(seg_img)
        img = normalize(img)
        cv2.imwrite(dst_dir + "img_" + str(i).rjust(4, '0') + "_i.png", img * 255)
        cv2.imwrite(dst_dir + "img_" + str(i).rjust(4, '0') + "_m.png", mask * 255)
        

def process_images(delete_existing=False, only_process_patient=None):
    if delete_existing and os.path.exists(settings.TIANCHI_EXTRACTED_IMAGE_DIR):
        print("Removing old stuff..")
        if os.path.exists(settings.TIANCHI_EXTRACTED_IMAGE_DIR):
            shutil.rmtree(settings.TIANCHI_EXTRACTED_IMAGE_DIR)

    if not os.path.exists(settings.TIANCHI_EXTRACTED_IMAGE_DIR):
        os.mkdir(settings.TIANCHI_EXTRACTED_IMAGE_DIR)
        os.mkdir(settings.TIANCHI_EXTRACTED_IMAGE_DIR + "_labels/")

    for subject_no in range(settings.TIANCHI_SUBSET_START_INDEX, settings.TIANCHI_SUBSET_END_INDEX):
        src_dir = settings.TIANCHI_RAW_SRC_DIR + TIANCHI_WORK_DIR + str(subject_no) + "/"
        src_paths = glob.glob(src_dir + "*.mhd")

        if only_process_patient is None and True:
            for src_path in src_paths:
                print(src_path)
                process_image(src_path)
#            pool = multiprocessing.Pool(2)
#            pool.map(process_image, src_paths)
        else:
            for src_path in src_paths:
                print(src_path)
                if only_process_patient is not None:
                    if only_process_patient not in src_path:
                        continue
                process_image(src_path)
                

def gen_mhd_row(mhd_path):
    print(mhd_path)
    
    patient_id = ntpath.basename(mhd_path).replace(".mhd", "")
    
    itk_img = SimpleITK.ReadImage(mhd_path)
    
    img_array = SimpleITK.GetArrayFromImage(itk_img)
    print("Img array shape: ", img_array.shape)
    
    origin = numpy.array(itk_img.GetOrigin())      # x,y,z  Origin in world coordinates (mm)
    print("Origin (x,y,z): ", origin)
    print("Origin (z,y,x): ", origin[::-1])

    direction = numpy.array(itk_img.GetDirection())      # x,y,z  Origin in world coordinates (mm)
    print("Direction: ", direction)

    spacing = numpy.array(itk_img.GetSpacing())    # spacing of voxels in world coor. (mm)
    print("Spacing (x,y,z): ", spacing)
    
    mhd_csv_line = [patient_id, img_array.shape[2], img_array.shape[1], img_array.shape[0], 
                    spacing[0], spacing[1], spacing[2], origin[0], origin[1], origin[2], direction]
    
    return mhd_csv_line


def process_auto_candidates_patient(src_path, patient_id, sample_count=100, candidate_type="white"):
    dst_dir = settings.TIANCHI_EXTRACTED_IMAGE_DIR + "_labels/"
    img_dir = settings.TIANCHI_EXTRACTED_IMAGE_DIR + patient_id + "/"
    src_dir = settings.TIANCHI_RAW_SRC_DIR + "csv/train/"
    
    df_pos_annos = pandas.read_csv(src_dir + "annotations.csv")
    df_pos_annos_patient = df_pos_annos[df_pos_annos["seriesuid"] == patient_id]
#    df_pos_annos = pandas.read_csv(dst_dir + patient_id + "_annos_pos_lidc.csv")

#    pos_annos_manual = None
#    manual_path = settings.EXTRA_DATA_DIR + "luna16_manual_labels/" + patient_id + ".csv"
#    if os.path.exists(manual_path):
#        pos_annos_manual = pandas.read_csv(manual_path)

    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)

    itk_img = SimpleITK.ReadImage(src_path)
    img_array = SimpleITK.GetArrayFromImage(itk_img)
    print("Img array: ", img_array.shape)
    print("Pos annos: ", len(df_pos_annos_patient))

    num_z, height, width = img_array.shape        #heightXwidth constitute the transverse plane
    origin = numpy.array(itk_img.GetOrigin())      # x,y,z  Origin in world coordinates (mm)
    print("Origin (x,y,z): ", origin)
    spacing = numpy.array(itk_img.GetSpacing())    # spacing of voxels in world coor. (mm)
    print("Spacing (x,y,z): ", spacing)
    rescale = spacing / settings.TARGET_VOXEL_MM
    print("Rescale: ", rescale)

    if candidate_type == "white":
        wildcard = "*_c.png"
    else:
        wildcard = "*_m.png"

    src_files = glob.glob(img_dir + wildcard)
    src_files.sort()
    src_candidate_maps = [cv2.imread(src_file, cv2.IMREAD_GRAYSCALE) for src_file in src_files]
    
    print(len(src_candidate_maps))

    candidate_list = []
    tries = 0
    while len(candidate_list) < sample_count and tries < 1000:
        tries += 1
        coord_z = int(numpy.random.normal(len(src_files) / 2, len(src_files) / 6))
        coord_z = max(coord_z, 0)
        coord_z = min(coord_z, len(src_files) - 1)
        candidate_map = src_candidate_maps[coord_z]
        if candidate_type == "edge":
            candidate_map = cv2.Canny(candidate_map.copy(), 100, 200)

        non_zero_indices = numpy.nonzero(candidate_map)
        if len(non_zero_indices[0]) == 0:
            continue
        nonzero_index = random.randint(0, len(non_zero_indices[0]) - 1)
        coord_y = non_zero_indices[0][nonzero_index]
        coord_x = non_zero_indices[1][nonzero_index]
        ok = True
        candidate_diameter = 6
        
        print("candidate coord x,y,z: ", coord_x, coord_y, coord_z)
        
        # df_pos_annos -> df_pos_annos_patient
        for index, row in df_pos_annos_patient.iterrows():
            pos_coord_x = float(row["coordX"] - origin[0])
            pos_coord_y = float(row["coordY"] - origin[1])
            pos_coord_z = float(row["coordZ"] - origin[2])
            
            diameter = float(row["diameter_mm"])
            
#            pos_coord_x = row["coordX"] * src_candidate_maps[0].shape[1]
#            pos_coord_y = row["coordY"] * src_candidate_maps[0].shape[0]
#            pos_coord_z = row["coordZ"] * len(src_files)
#            diameter = row["diameter"] * src_candidate_maps[0].shape[1]
            dist = math.sqrt(math.pow(pos_coord_x - coord_x, 2) + math.pow(pos_coord_y - coord_y, 2) + math.pow(pos_coord_z - coord_z, 2))
            if dist < (diameter + 48): #  make sure we have a big margin
                ok = False
                print("# Too close", (coord_x, coord_y, coord_z))
                break

#        if pos_annos_manual is not None:
#            for index, row in pos_annos_manual.iterrows():
#                pos_coord_x = row["x"] * src_candidate_maps[0].shape[1]
#                pos_coord_y = row["y"] * src_candidate_maps[0].shape[0]
#                pos_coord_z = row["z"] * len(src_files)
#                diameter = row["d"] * src_candidate_maps[0].shape[1]
#                # print((pos_coord_x, pos_coord_y, pos_coord_z))
#                # print(center_float_rescaled)
#                dist = math.sqrt(math.pow(pos_coord_x - coord_x, 2) + math.pow(pos_coord_y - coord_y, 2) + math.pow(pos_coord_z - coord_z, 2))
#                if dist < (diameter + 72):  #  make sure we have a big margin
#                    ok = False
#                    print("#Too close",  (coord_x, coord_y, coord_z))
#                    break

        if not ok:
            continue


        perc_x = round(float(coord_x) / src_candidate_maps[coord_z].shape[1], 4)
        perc_y = round(float(coord_y) / src_candidate_maps[coord_z].shape[0], 4)
        perc_z = round(float(coord_z) / len(src_files), 4)
        candidate_list.append([len(candidate_list), perc_x, perc_y, perc_z, round(float(candidate_diameter) / src_candidate_maps[coord_z].shape[1], 4), 0])

    if tries > 999:
        print("****** WARING!! TOO MANY TRIES ************************************")
    df_candidates = pandas.DataFrame(candidate_list, columns=["anno_index", "coord_x", "coord_y", "coord_z", "diameter", "malscore"])
    df_candidates.to_csv(dst_dir + patient_id + "_candidates_" + candidate_type + ".csv", index=False)


def process_auto_candidates_patients():        
    for subject_no in range(10, 15):
        src_dir = settings.TIANCHI_RAW_SRC_DIR + TIANCHI_WORK_DIR + str(subject_no) + "/"
        
        print(src_dir)
        
        for patient_index, src_path in enumerate(glob.glob(src_dir + "*.mhd")):
            # if not "100621383016233746780170740405" in src_path:
            #     continue
            patient_id = ntpath.basename(src_path).replace(".mhd", "")
            print("Patient: ", patient_index, " ", patient_id)
            # process_auto_candidates_patient(src_path, patient_id, sample_count=500, candidate_type="white")
            process_auto_candidates_patient(src_path, patient_id, sample_count=20, candidate_type="edge")


def gen_all_mhd_csv():
    all_mhd = []
    
    for subject_no in range(settings.TIANCHI_SUBSET_START_INDEX, settings.TIANCHI_SUBSET_END_INDEX):
        mhd_dir = settings.TIANCHI_RAW_SRC_DIR + TIANCHI_WORK_DIR + str(subject_no) + "/"
        mhd_paths = glob.glob(mhd_dir + "*.mhd")
        all_mhd.extend(mhd_paths)
    
    print("Total mhd: ", len(all_mhd))
    
    all_mhd_csv = []
    
    for mhd in all_mhd:
        mhd_csv_line = gen_mhd_row(mhd)
        all_mhd_csv.append(mhd_csv_line)
        
    df = pandas.DataFrame(all_mhd_csv, columns=
                          ["patient_id", "shape_x", "shape_y", "shape_z", "spacing_x", "spacing_y", "spacing_z", "origin_x", "origin_y", "origin_z", "direction"])
    
    dst_dir = settings.TIANCHI_RAW_SRC_DIR + "csv/train/"
    df.to_csv(dst_dir + "all_mhd.csv", index=False)
                


if __name__ == "__main__":    
    if False:
        only_process_patient = None
        process_images(delete_existing=False, only_process_patient=only_process_patient)
        
    if False:
        gen_all_mhd_csv()

    if True:
        process_auto_candidates_patients()