import os
import cv2
import glob
from tqdm import tqdm

import numpy as np
import tifffile
import tifffile as tf
import shutil
from tifffile import imread, imsave





def upscale_image(image_path, target_width):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    # Upscale the image
    if image.shape[0] != target_width or image.shape[1] != target_width:
        image = cv2.resize(image, (target_width, target_width), interpolation=cv2.INTER_LANCZOS4)

   
    return image

def crop_image_if_needed(image,tw,th):
    # If the image width is 1200, crop it to 1200x782
    target_width = tw
    target_height = th
    if image.shape[0] == target_height:
        return image
    elif image.shape[1] == target_width:
        pixels_to_remove = image.shape[0] - target_height
        pixels_to_remove_from_top = pixels_to_remove // 2
        pixels_to_remove_from_bottom = pixels_to_remove - pixels_to_remove_from_top

        image = image[pixels_to_remove_from_top:-pixels_to_remove_from_bottom, :]
    return image

def crop_images_in_folder(folder_path):
    for image_name in os.listdir(folder_path):
        if image_name.endswith('.tif'):
            image_path = os.path.join(folder_path, image_name)
            image = tf.imread(image_path)
            cropped_image = crop_image_if_needed(image,tw,th)
            tf.imsave(image_path, cropped_image) 

def crop_directory(base_directory):
    sub_folders = [name for name in os.listdir(base_directory) if os.path.isdir(os.path.join(base_directory, name))]
    
    for sub_folder in tqdm(sorted(sub_folders)):
        if sub_folder in ["00", "00_ST","00_GT"]:
            continue
        #if not sub_folder.endswith("_GT"):
        print('Processing sub_folder:', sub_folder)
        if sub_folder.endswith("_ST"):
            seg_folder = os.path.join(base_directory, sub_folder, 'SEG')
        elif sub_folder.endswith("_GT"):
            seg_folder = os.path.join(base_directory, sub_folder, 'TRA')
        else:
            seg_folder = os.path.join(base_directory, sub_folder)
                
        crop_images_in_folder(seg_folder)

def upscale_images_in_folder(folder_path, target_width, path_pretrained, model_name):
    #files_in_folder = glob.glob(os.path.join(folder_path, '*'))

  
    png_files = glob.glob(os.path.join(folder_path, '*.png'))
    for png_file in png_files:
        image = upscale_image(png_file, target_width)
        new_filename = "t" + str(os.path.basename(png_file).split('.')[0]).zfill(3) + '.tif'
        cv2.imwrite(os.path.join(folder_path, new_filename), image)
        os.remove(png_file)
    
   
    print('cellpose current path',folder_path)
    # After processing the folder, run the Cellpose command
    os.system(f"python -m cellpose --dir '{folder_path}' --pretrained_model '{path_pretrained + model_name}' --save_png")
    
    # Clean up after Cellpose
    npy_files = glob.glob(os.path.join(folder_path, '*.npy'))
    for npy_file in npy_files:
        os.remove(npy_file)

    output_png_files = glob.glob(os.path.join(folder_path, '*_output.png'))
    for output_png in output_png_files:
        os.remove(output_png)

    mask_files = glob.glob(os.path.join(folder_path, '*_masks.png'))
    seg_folder_path = os.path.join(os.path.dirname(folder_path), os.path.basename(folder_path) + "_ST", "SEG")
    os.makedirs(seg_folder_path, exist_ok=True)
    #print('saved to',seg_folder_path)
    for mask_file in mask_files:
        filename = os.path.basename(mask_file)  # Extract only the filename from the full path
        new_name = "man_seg" + filename.split('_')[0][1:] + ".tif"

        # Load the PNG image
        image = cv2.imread(mask_file, cv2.IMREAD_UNCHANGED)

        # Save the image as a TIFF file
        cv2.imwrite(os.path.join(seg_folder_path, new_name), image)
        #print('saved to',seg_folder_path)
        # Remove the original PNG file
        os.remove(mask_file)
    
def process_directory(base_directory, target_width=1024, path_pretrained='', model_name='',r=10):
    sub_folders = [name for name in os.listdir(base_directory) if os.path.isdir(os.path.join(base_directory, name))]
    for sub_folder in tqdm(sorted(sub_folders)):
        if not sub_folder.endswith("_GT") and not sub_folder.endswith("_ST"):
            #print('sub_folder',sub_folder)
            #if sub_folder=='1':
            #    print(stop)
            upscale_images_in_folder(os.path.join(base_directory, sub_folder), target_width, path_pretrained, model_name)
            
            # Paths for segmentation and ground truth
            seg_folder = os.path.join(base_directory, sub_folder+ "_ST", 'SEG')
            gt_folder = os.path.join(base_directory, sub_folder + "_GT", 'TRA')
            # Loop through each segmentation mask
            for seg_file in glob.glob(os.path.join(seg_folder, 'man_seg*.tif')):
                # Extract timeframe from the filename
                timeframe = str(int(os.path.basename(seg_file).split('seg')[1].split('.')[0]))
                #print('tf',timeframe)
                # Corresponding ground truth detection filename
                gt_file_name = 'man_track' + timeframe + '.tif'
                gt_file_path = os.path.join(gt_folder, gt_file_name)
                
                new_timeframe = str(timeframe).zfill(3) # Ensure three digits
                new_name = 'man_track' + new_timeframe + '.tif'
                #print('nn',new_name,gt_file_path)
                new_path = os.path.join(os.path.dirname(gt_file_path), new_name)
                try:
                    os.rename(gt_file_path, new_path)
                except:
                    pass
                
                
                # Rename the detection file to the desired format
                #new_gt_file_name = 'man_seg' + timeframe + '.tif'
                new_gt_file_path = os.path.join(gt_folder, new_name)
                #os.rename(gt_file_path, new_gt_file_path)
                
                # Process masks
                corrected_mask = process_masks(seg_file, new_gt_file_path,radius=r)
                cv2.imwrite(seg_file, corrected_mask)  # Overwrite the segmentation mask in the SEG folder
            
    clean_up(base_directory)
            
def process_masks(segmentation_mask_path, detection_gt_path, radius=10):
    # Load the images
    mask_img = tifffile.imread(segmentation_mask_path)
    gt_img = tifffile.imread(detection_gt_path)

    # Ensure the images are of the same shape
    if mask_img.shape != gt_img.shape:
        raise ValueError("Segmentation mask and ground truth images must have the same shape.")

    unique_masks = np.unique(mask_img)
    #print('seg,det,unique',segmentation_mask_path,detection_gt_path,unique_masks)
    #print(stop)
    # For each unique value in segmentation mask, check if it overlaps with the detection ground truth
    for mask_val in unique_masks:
        if mask_val == 0:  # Ignore background
            continue

        mask_coords = np.where(mask_img == mask_val)
        gt_overlap = gt_img[mask_coords]

        # Check if any of the mask's coordinates overlap with the ground truth
        if np.sum(gt_overlap) == 0:
            mask_img[mask_coords] = 0
    
    
    unique_circs = np.unique(gt_img)

    # For each unique value in segmentation mask, check if it overlaps with the detection ground truth
    for mask_val in unique_circs:
        if mask_val == 0:  # Ignore background
            continue

        mask_coords = np.where(gt_img == mask_val)
        seg_overlap = mask_img[mask_coords]

        # Check if any of the mask's coordinates overlap with the ground truth
        if np.sum(seg_overlap) == 0:
            
            x_min, x_max = np.min(mask_coords[1]), np.max(mask_coords[1])
            y_min, y_max = np.min(mask_coords[0]), np.max(mask_coords[0])

            # Calculate the center of the bounding box
            center_x = (x_max + x_min) // 2
            center_y = (y_max + y_min) // 2

            # Create a new unique label and draw a circle at the bounding box's center
            new_label = int(mask_img.max() + 1)
            cv2.circle(mask_img, (center_x, center_y), radius, new_label, -1)


    return mask_img


def clean_up(base_directory):
    # Rename folders with names '1' to '9' to add leading zeros
    try:
        for i in range(1, 10):
            old_name = str(i)
            new_name = str(i).zfill(2)

            # Check if folders exist before renaming
            for suffix in ["", "_GT", "_ST"]:
                old_folder = os.path.join(base_directory, old_name + suffix)
                new_folder = os.path.join(base_directory, new_name + suffix)

                if os.path.exists(old_folder):
                    os.rename(old_folder, new_folder)

    except:
        pass

    # Deleting specified folders and files
    gt_folders = [name for name in os.listdir(base_directory) if name.endswith("_GT")]
    for folder in gt_folders:
        path_to_folder = os.path.join(base_directory, folder)
        
        # Delete COND and ID subfolders if they exist
        for subfolder in ["COND", "ID"]:
            full_path = os.path.join(path_to_folder, subfolder)
            if os.path.exists(full_path):
                shutil.rmtree(full_path)
                
        # Delete pos_GT.txt in TRA subfolder
        pos_gt_file = os.path.join(path_to_folder, "TRA", "pos_GT.txt")
        if os.path.exists(pos_gt_file):
            os.remove(pos_gt_file)
            
    all_folders = [name for name in os.listdir(base_directory) if not name.endswith(("_GT", "_ST"))]
    for folder in all_folders:
        path_to_folder = os.path.join(base_directory, folder)
        print('folder',folder)
        # Open and save .tif files
        tif_files = [file for file in os.listdir(path_to_folder) if file.endswith(".tif")]
        for file in tif_files:
            full_path = os.path.join(path_to_folder, file)
            img = imread(full_path)
            
            # Check if the image has more than one channel
            if len(img.shape) > 2 and img.shape[2] > 1:
                # Taking only the first channel
                img_one_channel = img[:, :, 0]
                imsave(full_path, img_one_channel)




def move_directory(base_directory_path, output_directory):
    # Get the name of the base directory (e.g., 'PSC')
    base_directory_name = os.path.basename(base_directory_path)

    # Construct the path of the new directory in the output location
    new_directory_path = os.path.join(output_directory, base_directory_name)

    # Remove the destination directory if it already exists
    if os.path.exists(new_directory_path):
        try:
            shutil.rmtree(new_directory_path)
        except OSError as e:
            print(f"Error: {new_directory_path} : {e.strerror}")
            return

    # Copy the entire directory tree to the new location
    try:
        shutil.copytree(base_directory_path, new_directory_path)
        print(f"Directory {base_directory_name} successfully copied to {output_directory}")
    except Exception as e:
        print(f"An error occurred while copying: {e}")



def get_pp_params(name):

    sample_dir='./ControlNet/sampling/'+name
    stat_path = sample_dir+'/statistics.txt'

    r = None
    max_shape_width = None

    # Open and read the file
    try:
        with open(stat_path, 'r') as file:
            lines = file.readlines()

            # Process each line to find the required values
            for line in lines:
                if 'r:' in line:
                    r = float(line.split(':')[1].strip())
                elif 'Shape:' in line:
                    shape_values = line.split(':')[1].strip().strip('()').split(',')
                    max_shape_width = max(int(shape_values[0].strip()), int(shape_values[1].strip()))

    except FileNotFoundError:
        print(f"File not found: {stat_path}")
        return None, None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None

   

    return sample_dir, max_shape_width, r 


