import subprocess
import os


out_files_dir = "/home/sixue/FaceDetection/cascade_cnn/matlab/result_orig"
#out_files_dir = "/home/sixue/FaceDetection/cascade_cnn/caffe/CNN_face_detection/face_detection/detections"
save_DiscROC_dir = out_files_dir + "/"

evaluate_dir="/home/sixue/2T/FaceDataSet/evaluation/evaluate"
FDDB_root="/home/sixue/2T/FaceDataSet/FDDB"
FDDB_fold=FDDB_root+"/FDDB-folds"


for current_file in range(1,11):
    # run evaluation
    subprocess.call([evaluate_dir,
                           "-a", FDDB_fold + "/FDDB-fold-" + str(current_file).zfill(2) + "-ellipseList.txt",
                           "-d", out_files_dir + "/fold-" + str(current_file).zfill(2) + "-out.txt",
                           "-i", FDDB_root + "/",
                           "-l", FDDB_fold +"/FDDB-fold-" + str(current_file).zfill(2) + ".txt",
                           "-r", save_DiscROC_dir,
                           "-z", ".jpg"])

    # rename file
    os.rename(save_DiscROC_dir + "DiscROC.txt",
              save_DiscROC_dir + "DiscROC-" + str(current_file).zfill(2) + ".txt")
