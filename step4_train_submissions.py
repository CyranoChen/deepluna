import settings
import helpers
import sys
import os
import glob
import pandas
import numpy
import SimpleITK
import ntpath
import math

P_TH = 0.3
MAX_NODULE_COUNT = 10000
MIN_DISTANCE = 18

def get_distance(x1,y1,z1,x2,y2,z2):
    return math.sqrt((x1-x2)**2+(y1-y2)**2+(z1-z2)**2)

#change world coordination to voxel space 
def voxel_to_world(voxel, origin, spacing):
    voxel = voxel * spacing
    voxel = voxel + origin
    return voxel

def convert_csv_coord_to_world():
    csv_path = settings.TIANCHI_NODULE_DETECTION_DIR + "predictions10_tianchi_val_fs_final/"
    df = pandas.read_csv(csv_path+"all_predictions_candidates_falsepos.csv")
    
    df_all_mhd = pandas.read_csv(settings.TIANCHI_RAW_SRC_DIR + "csv/val/all_mhd.csv")
    print("df_all_mhd count: ", len(df_all_mhd))
    
    rows = []
    for row_index, row in df.iterrows():   
        patient_id = row["patient_id"]        
        print(patient_id)
        
        mhd = df_all_mhd[df_all_mhd["patient_id"] == patient_id]
        
        row["coord_x"] = float(row["coord_x"]*mhd["shape_x"]*mhd["spacing_x"])+float(mhd["origin_x"])
        row["coord_y"] = float(row["coord_y"]*mhd["shape_y"]*mhd["spacing_y"])+float(mhd["origin_y"])
        row["coord_z"] = float(row["coord_z"]*mhd["shape_z"]*mhd["spacing_z"])+float(mhd["origin_z"])
        
        row["diameter_mm"] =  round(float(row["diameter_mm"]) / float(mhd["spacing_x"]),4)
        
        rows.append(row)
    
    df = pandas.DataFrame(rows, columns=["patient_id", "anno_index", "coord_x", "coord_y", "coord_z", "diameter", "nodule_chance", "diameter_mm"])
    df.to_csv(csv_path+"all_predictions_candidates_falsepos_world_coord.csv", index=False)


def convert_nodules_coord_to_world(mhd, dir_name, magnification=1):    
    patient_id = mhd["patient_id"]
    
    print("patient_id: ", patient_id)
    
    img_array_shape = [int(mhd["shape_x"]), int(mhd["shape_y"]), int(mhd["shape_z"])]
    print("Img array shape: ", img_array_shape)
    
    origin = [float(mhd["origin_x"]), float(mhd["origin_y"]), float(mhd["origin_z"])]
    print("Origin (x,y,z): ", origin)
    
    spacing = [float(mhd["spacing_x"]), float(mhd["spacing_y"]), float(mhd["spacing_z"])]
    print("Spacing (x,y,z): ", spacing)
    
    print("Direction: ", mhd["direction"])

    
#    patient_img = helpers.load_patient_images(patient_id, settings.TIANCHI_EXTRACTED_IMAGE_DIR, "*_i.png", [])

    patient_img_shape = [int(img_array_shape[0]*spacing[0]/float(magnification)), 
                         int(img_array_shape[1]*spacing[1]/float(magnification)), 
                         int(img_array_shape[2]*spacing[2]/float(magnification))]

    print("patient_img shape: ", patient_img_shape)
        
    csv_path = settings.TIANCHI_NODULE_DETECTION_DIR + dir_name + "/" + patient_id + ".csv"
    
    print(csv_path)
    
    rows = []
    
    if not os.path.exists(csv_path):
        return rows
    
#    pred_df_list = []
    pred_nodules_df = pandas.read_csv(csv_path)
#    pred_df_list.append(pred_nodules_df)
#    pred_nodules_df = pandas.concat(pred_df_list, ignore_index=True)

    nodule_count = len(pred_nodules_df)
    
    print("nodule_count: ", nodule_count)

    nodule_count = 0
    if len(pred_nodules_df) > 0:
        pred_nodules_df = pred_nodules_df.sort_values(by="nodule_chance", ascending=False)        
        
        for row_index, row in pred_nodules_df.iterrows():
            if float(row["nodule_chance"]) < P_TH:
                continue
            
            dia = float(row["diameter_mm"]) / spacing[0]
            
#            if dia < 3.0:
#                continue
                        
            p_x = float(row["coord_x"])*patient_img_shape[0]
            p_y = float(row["coord_y"])*patient_img_shape[1]
            p_z = float(row["coord_z"])*patient_img_shape[2]
            
            x, y, z = [p_x+origin[0],p_y+origin[1],p_z+origin[2]]
#            z, y, x = voxel_to_world([p_z, p_y, p_x], origin[::-1], spacing[::-1])
            
            row["coord_z"] = z
            row["coord_y"] = y
            row["coord_x"] = x 
            row["diameter_mm"] = dia
            row["patient_id"] = patient_id         
            
            rows.append(row)
            
            nodule_count += 1
            
            if nodule_count >= MAX_NODULE_COUNT:
                break
        
        print(nodule_count)
        
    return rows


def convert_all_nodules_coord_to_world(csv_dir_name, magnification=1):       
    df_all_mhd = pandas.read_csv(settings.TIANCHI_RAW_SRC_DIR + "csv/val/all_mhd.csv")
    print("df_all_mhd count: ", len(df_all_mhd))
    
    all_predictions_world_coord_csv = []
    
    for index, mhd in df_all_mhd.iterrows():
#        if mhd["patient_id"] != "LKDS-00006":
#            continue
        rows = convert_nodules_coord_to_world(mhd, csv_dir_name, magnification)
        all_predictions_world_coord_csv.extend(rows)
       
    df = pandas.DataFrame(all_predictions_world_coord_csv, columns=["patient_id", "anno_index", "coord_x", "coord_y", "coord_z", "diameter", "nodule_chance", "diameter_mm"])
    
    dst_dir = settings.TIANCHI_NODULE_DETECTION_DIR + csv_dir_name + "/"
    df.to_csv(dst_dir + "all_predictions_world_coord.csv", index=False)


def combine_nodule_predictions(dirs, train_set=True, nodule_th=0.5, extensions=[""]):
    print("Combining nodule predictions: ", "Train" if train_set else "Submission")
    if train_set:
        labels_df = pandas.read_csv("resources/val/seriesuids.csv")
    else:
        labels_df = pandas.read_csv("resources/test2/seriesuids.csv")

#    mass_df = pandas.read_csv(settings.BASE_DIR + "masses_predictions.csv")
#    mass_df.set_index(["patient_id"], inplace=True)

    # meta_df = pandas.read_csv(settings.BASE_DIR + "patient_metadata.csv")
    # meta_df.set_index(["patient_id"], inplace=True)

    data_rows = []
    for index, row in labels_df.iterrows():
        patient_id = row["id"]
        # mask = helpers.load_patient_images(patient_id, settings.EXTRACTED_IMAGE_DIR, "*_m.png")
        print(len(data_rows), " : ", patient_id)
        # if len(data_rows) > 19:
        #     break
#        cancer_label = row["cancer"]
#        mass_pred = int(mass_df.loc[patient_id]["prediction"])
        # meta_row = meta_df.loc[patient_id]
        # z_scale = meta_row["slice_thickness"]
        # x_scale = meta_row["spacingx"]
        # vendor_low = 1 if "1.2.276.0.28.3.145667764438817.42.13928" in meta_row["instance_id"] else 0
        # vendor_high = 1 if "1.3.6.1.4.1.14519.5.2.1.3983.1600" in meta_row["instance_id"] else 0
        #         row_items = [cancer_label, 0, mass_pred, x_scale, z_scale, vendor_low, vendor_high] # mask.sum()

        row_items = [] # mask.sum()

        for magnification in [1, 1.5, 2]:
            pred_df_list = []
            for extension in extensions:
                src_dir = settings.TIANCHI_NODULE_DETECTION_DIR + "predictions" + str(int(magnification * 10)) + extension + "/"
                pred_nodules_df = pandas.read_csv(src_dir + patient_id + ".csv")
                pred_nodules_df = pred_nodules_df[pred_nodules_df["diameter_mm"] > 0]
                pred_nodules_df = pred_nodules_df[pred_nodules_df["nodule_chance"] > nodule_th]
                pred_df_list.append(pred_nodules_df)

            pred_nodules_df = pandas.concat(pred_df_list, ignore_index=True)

            nodule_count = len(pred_nodules_df)
            nodule_max = 0
            nodule_median = 0
            nodule_chance = 0
            nodule_sum = 0
            coord_z = 0
            second_largest = 0
            nodule_wmax = 0

            count_rows = []
            coord_y = 0
            coord_x = 0

            if len(pred_nodules_df) > 0:
                max_index = pred_nodules_df["diameter_mm"].argmax
                max_row = pred_nodules_df.loc[max_index]
                nodule_max = round(max_row["diameter_mm"], 2)
                nodule_chance = round(max_row["nodule_chance"], 2)
                nodule_median = round(pred_nodules_df["diameter_mm"].median(), 2)
                nodule_wmax = round(nodule_max * nodule_chance, 2)
                coord_z = max_row["coord_z"]
                coord_y = max_row["coord_y"]
                coord_x = max_row["coord_x"]


                rows = []
                for row_index, row in pred_nodules_df.iterrows():
                    dist = helpers.get_distance(max_row, row)
                    if dist > 0.2:
                        nodule_mal = row["diameter_mm"]
                        if nodule_mal > second_largest:
                            second_largest = nodule_mal
                    rows.append(row)

                count_rows = []
                for row in rows:
                    ok = True
                    for count_row in count_rows:
                        dist = helpers.get_distance(count_row, row)
                        if dist < 0.2:
                            ok = False
                    if ok:
                        count_rows.append(row)
            nodule_count = len(count_rows)
            row_items += [nodule_max, nodule_chance, nodule_count, nodule_median, nodule_wmax, coord_z, second_largest, coord_y, coord_x]

        row_items.append(patient_id)
        data_rows.append(row_items)

    # , "x_scale", "z_scale", "vendor_low", "vendor_high"
    columns = []
    for magnification in [1, 1.5, 2]:
        str_mag = str(int(magnification * 10))
        columns.append("nodule_max_" + str_mag) # 
        columns.append("nodule_chance_" + str_mag)
        columns.append("nodule_count_" + str_mag)
        columns.append("nodule_median_" + str_mag)
        columns.append("nodule_wmax_" + str_mag)
        columns.append("coord_z_" + str_mag)
        columns.append("second_largest_" + str_mag)
        columns.append("coord_y_" + str_mag)
        columns.append("coord_x_" + str_mag)

    columns.append("patient_id")
    res_df = pandas.DataFrame(data_rows, columns=columns)

    if not os.path.exists(settings.TIANCHI_NODULE_DETECTION_DIR + "submission/"):
        os.mkdir(settings.TIANCHI_NODULE_DETECTION_DIR + "submission/")
    target_path = settings.TIANCHI_NODULE_DETECTION_DIR + "submission/" + "submission" + extension + ".csv"
    res_df.to_csv(target_path, index=False)


def filter_submission():
    df_all_mhd = pandas.read_csv(settings.TIANCHI_RAW_SRC_DIR + "csv/test2/all_mhd.csv")
    print("df_all_mhd count: ", len(df_all_mhd))
    
    df_nodules = pandas.read_csv(settings.TIANCHI_NODULE_DETECTION_DIR + "predictions10_tianchi_test2_fs_final/all_predictions_world_coord_merge.csv")
    
    rows = []
    for index, mhd in df_all_mhd.iterrows():
        patient_id = mhd["patient_id"]
        print(patient_id)
        
        df = df_nodules[df_nodules["seriesuid"] == patient_id]
        
        count = 0
        if len(df) > 0:
            df = df.sort_values(by="probability", ascending=False)
            
            for index, row in df.iterrows():
#                if row["diameter_mm"] < 3:
#                    continue
                
                if row["probability"] < P_TH:
                    continue
                
                rows.append(row)
                count +=1
                
                if count >= 50:
                    break
        
        print(count)
     
    res_df = pandas.DataFrame(rows, columns=["seriesuid","coordX", "coordY", "coordZ","probability"])
    target_path = settings.TIANCHI_NODULE_DETECTION_DIR + "predictions10_tianchi_test2_fs_final/all_predictions_world_coord_merge_filter.csv"
    res_df.to_csv(target_path, index=False)

            
def merge_submission():
    #read the annotations.csv that contains the nodules info
    df_node = pandas.read_csv(settings.TIANCHI_NODULE_DETECTION_DIR + "predictions10_tianchi_test2_fs_final/all_predictions_world_coord.csv")
    
    df_node = df_node.dropna()

    seriesuids_csv = pandas.read_csv(settings.TIANCHI_RAW_SRC_DIR + "csv/test2/all_mhd.csv")
    seriesuids = seriesuids_csv['patient_id'].values

    x = []
    y = []
    z = []
    p = []
    user_id = []
    uid_done = []
    for seriesuid in seriesuids:
        if seriesuid in uid_done:
            continue
        uid_done.append(seriesuid)
        mini_node = df_node[df_node['patient_id'] == seriesuid]

        print(seriesuid)
    
        uid = mini_node["patient_id"].values
        node_x = mini_node["coord_x"].values
        node_y = mini_node["coord_y"].values
        node_z = mini_node["coord_z"].values
        probability = mini_node['nodule_chance'].values


        print(len(node_x))
        mat = numpy.zeros([len(node_x),len(node_x)])
        for i in range(len(node_x)):
            for j in range(len(node_x)):
                mat[i,j]=get_distance(node_x[i],node_y[i],node_z[i],node_x[j],node_y[j],node_z[j])
                if i == j:
                    mat[i,j] = 80

        for i in range(len(node_x)):
            num = 1
            print("node index: ",i)
            for j in range(len(node_x)):
                if mat[i,j] < MIN_DISTANCE:
                    print("distance",mat[i,j])
                    num += 1
                    node_x[i] += node_x[j]
                    node_y[i] += node_y[j]
                    node_z[i] += node_z[j]
                    probability[i] += probability[j]
                    print(probability[i])
                    #if probability[j] > probability[i]:
                    #    probability[i] = probability[j]
                    print('add one',j,node_x[j])
            print('whole',node_x[i],num)
            node_x[i] /= num
            node_y[i] /= num
            node_z[i] /= num
            probability[i] /= num
            user_id.append(uid[i])
            x.append(node_x[i])
            y.append(node_y[i])
            z.append(node_z[i])
            p.append(probability[i])
            print(node_x[i])
            #raw_input()
    
    x1 = []
    y1 = []
    z1 = []
    p1 = []
    u = []
    for i in range(len(x) -1):
        if get_distance(x[i],y[i],z[i],x[i+1],y[i+1],z[i+1]) < 3:
            x[i+1] = x[i]/2 + x[i+1]/2
            y[i+1] = y[i]/2 + y[i+1]/2
            z[i+1] = z[i]/2 + z[i+1]/2
            p[i+1] = p[i]/2 + p[i+1]/2
        else:
            x1.append(x[i])
            y1.append(y[i])
            z1.append(z[i])
            p1.append(p[i])
            u.append(user_id[i])

    dataframe = pandas.DataFrame({'seriesuid':u,'coordX':x1,'coordY':y1,'coordZ':z1,'probability':p1})
    #dataframe = pd.DataFrame({'seriesuid':user_id,'coordX':x,'coordY':y,'coordZ':z,'probability':p})
    dataframe.to_csv(settings.TIANCHI_NODULE_DETECTION_DIR + "predictions10_tianchi_test2_fs_final/all_predictions_world_coord_merge.csv",index=False)


if __name__ == "__main__":
    if False:
        combine_nodule_predictions(None, train_set=False, nodule_th=0.5, extensions=["_tianchi_test2_fs_final"])
    
    if False:
        convert_csv_coord_to_world()
        
    if False:
        for magnification in [1, 1.5, 2]:
            convert_all_nodules_coord_to_world("predictions"+str(int(magnification*10))+"_tianchi_test2_fs_final", magnification)
            convert_all_nodules_coord_to_world("predictions"+str(int(magnification*10))+"_tianchi_test2_fs_final", magnification)
    
    if True:
        convert_all_nodules_coord_to_world("predictions"+str(int(1*10))+"_tianchi_val_fs_final_new", 1)

    if False:
        filter_submission()
    
    if False:
        merge_submission()
        
#    if False:
#        for model_variant in ["_luna16_fs", "_luna_posnegndsb_v1", "_luna_posnegndsb_v2"]:
#            print("Variant: ", model_variant)
#            if True:
#                combine_nodule_predictions(None, train_set=False, nodule_th=0.7, extensions=[model_variant])
#            if True:
#                train_xgboost_on_combined_nodules(fixed_holdout=False, submission=True, submission_is_fixed_holdout=False, extension=model_variant)
#                train_xgboost_on_combined_nodules(fixed_holdout=True, extension=model_variant)
#
#    combine_submissions(level=1, model_type="luna_posnegndsb")
#    combine_submissions(level=1, model_type="luna16_fs")
#    combine_submissions(level=1, model_type="daniel")
#    combine_submissions(level=2)
