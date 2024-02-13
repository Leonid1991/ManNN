import os
import shutil
# paths to the "initial" data to operate with files
# old patients
path_old  = "Z:\VisualStudio\Python\Manchester_Working\TREND\images\old"
files_old = os.listdir(path_old)   # List all files in the folder        
files_old = [file for file in files_old if file.lower().endswith(('.png')) ] # Filter only png images
len_old   = len(files_old)
# young patients
path_young= "Z:\VisualStudio\Python\Manchester_Working\TREND\images\young"
files_young = os.listdir(path_young)   # List all files in the folder        
files_young = [file for file in files_young if file.lower().endswith(('.png')) ] # Filter only png images
len_young = len(files_young)
# calculation of indexes for validation, testing and training
nb_images = min(len_old,len_young) # common min
train_dir, val_dir, test_dir = 'train', 'val', 'test'  # naming of the furture folders fro training 
test_data_portion, val_data_portion = 0.15, 0.15       # propotion of files
start_val_data_idx = int(nb_images * (1 - val_data_portion - test_data_portion))
start_test_data_idx = int(nb_images * (1 - test_data_portion))
# function to create folders, where we send images
def create_directory(dir_name):
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)
    os.makedirs(dir_name)
    os.makedirs(os.path.join(dir_name, "old"))
    os.makedirs(os.path.join(dir_name, "young"))    
# folders' creation
create_directory(train_dir)
create_directory(val_dir)
create_directory(test_dir)
# copying of images
def copy_images(start_index, end_index, source_folder, dest_dir, name):
    files = os.listdir(source_folder)       
    for i in range(start_index, end_index):
        file = files[i]
        shutil.copy2(os.path.join(source_folder, file), os.path.join(dest_dir, name))
def copy_img_general(start_val, start_test, end_nb, source_folder, dest_tr, dest_val, dest_test, name):
    copy_images(0, start_val, source_folder, dest_tr, name)
    copy_images(start_val, start_test, source_folder, dest_val, name)
    copy_images(start_test, end_nb, source_folder, dest_test, name)
# separation data of elderly people  
copy_img_general(start_val_data_idx, start_test_data_idx, nb_images, path_old, train_dir, val_dir, test_dir, "old")
# separation data of young people  
copy_img_general(start_val_data_idx, start_test_data_idx, nb_images, path_young, train_dir, val_dir, test_dir, "young")
