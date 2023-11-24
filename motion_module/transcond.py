import numpy as np
import cv2
import os
import re

from tqdm import tqdm




def make_ini_pic(path_GT, r,pix,t): 
    file_path = os.path.join(path_GT, 'TRA', 'pos_GT.txt')
    xl, yl, rl, idel, split_l, s_pr_l, t_vl = np.loadtxt(file_path, skiprows=2, delimiter='\t', usecols=(0, 1, 2, 3, 4, 5, 6), unpack=True)
    
    
    mask = split_l != -1
    xl = xl[mask]
    yl = yl[mask]
    rl = rl[mask]
    idel = idel[mask]
    split_l = split_l[mask]
    s_pr_l = s_pr_l[mask]
    t_vl = t_vl[mask]
    
    
    y = (xl[t_vl==t] * pix)
    x = ((1-yl[t_vl==t]) * pix)
    rl = rl[t_vl==t]

    # Get the flags for each position using the check_split_timeframe function
    flags = [check_split_timeframe(t, int(id), t_vl, idel, split_l) for id in idel[t_vl==t]]

    
    colors = [get_gradient_color(4, flag) for flag in flags]
    positions = list(zip(x, y))
    
    # Pass the positions list and colors to the ind2CNet function
    image_tensor = ind2CNet(positions, rl, colors,r,pix)
    
    # Create the ID folder if it doesn't exist
    id_dir = os.path.join(path_GT, 'ID')
    if not os.path.exists(id_dir):
        os.makedirs(id_dir)

    output_path = os.path.join(id_dir, str(t) + '.jpg')
    #print('out',output_path)
    image_tensor=resize_to_512_if_needed(image_tensor)
    cv2.imwrite(output_path, image_tensor)
    
    #plt.imshow(image_tensor)
    #plt.axis('off')
    #plt.show()

    
    
    
def get_gradient_color(d, flag):
    base_color = [0, 255, 0]  # Pure green as base color

    if flag == 0:
        return base_color
    elif flag > 0:
        return [255-((flag-1)*(255/(d-1))),255,0]
    else:  # flag < 0
        return [255,50+((abs(flag)-1)*(150/(d-1))),0]    


def resize_to_512_if_needed(image):
    """
    Resizes a squared image to 512x512 if its resolution is bigger than 512x512.
    
    Args:
        image (np.ndarray): The input image.

    Returns:
        np.ndarray: The possibly resized image.
    """
    H, W = image.shape[:2]
    if H > 512 and W > 512:
        return cv2.resize(image, (512, 512), interpolation=cv2.INTER_LANCZOS4)
    else:
        return image
    


def ind2CNet(ind_list, values, colors,r,pix):
    image_tensor = np.zeros((int(pix), int(pix), 3),dtype=np.uint8)  # Initialize a 3-channel image

    radius=r

    for (x, y), color in zip(ind_list, colors):
        x, y = int(x), int(y)
        for i in range(-radius, radius+1):
            for j in range(-radius, radius+1):
                if i**2 + j**2 <= radius**2:
                    try:
                        image_tensor[x+i, y+j] = color  # Set the pixel value to the specified color
                    except IndexError:
                        pass

    return image_tensor




def check_split_timeframe(t, cell_id, t_vl, idel, split_l):
    # Search for the cell with the given cell_id
    indices = np.where(idel == cell_id)[0]  # Find all indices where idel matches cell_id
    
    for index in indices:
        if split_l[index] != 0.0:  # If the cell has a split_id
            split_time = t_vl[index]  # Extract the time of splitting
            time_difference = t - split_time
            
            # Check splitting timeframes
            if time_difference == 1:
                return -1
            elif time_difference == 2:
                return -2
            elif time_difference == 3:
                return -3
            elif time_difference == 4:
                return -4
            elif time_difference == 0:
                return 99
            
    
    # Check for merges
    child_indices = np.where(split_l == cell_id)[0]  # Find all indices where the cell is a parent
    for index in child_indices:
        merge_time = t_vl[index]  # Extract the time of merging
        time_difference = merge_time - t
        
        # Check merging timeframes
        if time_difference == 1:
            return 1
        elif time_difference == 2:
            return 2
        elif time_difference == 3:
            return 3
                
    # If neither condition is met, return flag=0
    return 0




def get_num_timeframes(path_GT):
    file_path = os.path.join(path_GT, 'TRA', 'pos_GT.txt')
    t_vl = np.loadtxt(file_path, skiprows=2, delimiter='\t', usecols=6)
    return len(np.unique(t_vl))


def transition2condition(mm_path, A_mu,pix,d_mito):


    base_folder = mm_path
    gt_folders = [f for f in os.listdir(base_folder) if '_GT' in f]


    r=int(np.sqrt(A_mu / np.pi) / 4) if int(np.sqrt(A_mu / np.pi) / 4) > 2 else 2
    #print('r',int(np.sqrt(A_mu / np.pi)))


    for gt_folder in tqdm(gt_folders, desc="Creating Conditioning out of Motion Module"):
        path_GT = os.path.join(base_folder, gt_folder)
        num_timeframes = get_num_timeframes(path_GT)
        #print('#',num_timeframes)
        for z in range(num_timeframes):
            make_ini_pic(path_GT,r,pix, t=z)