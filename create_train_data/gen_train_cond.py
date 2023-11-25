import numpy as np
import os
#from cellpose import utils, io, models
#from cellpose import plot
import matplotlib.pyplot as plt
from PIL import Image
from scipy.optimize import curve_fit
from scipy.integrate import quad
from skimage.io import imread
import imgaug.augmenters as iaa
import cv2
import json
from scipy.spatial import KDTree
import random
from scipy.special import gamma
from tqdm import tqdm
#import seaborn as sns
#import pandas as pd



def make_square(image):
    """
    Takes an image and pads it to be square, using a black background.
    """
    h, w = image.shape[:2]
    
    # Create padding instructions for grayscale (2D) and color (3D) images
    if len(image.shape) == 2:  # Grayscale image
        padding_dims = ((0, 0), (0, 0))
    else:  # RGB or similar 3D image
        padding_dims = ((0, 0), (0, 0), (0, 0))

    # If the image's height is less than its width
    if h < w:
        padding_top = (w - h) // 2
        padding_bottom = w - h - padding_top
        padding_dims = ((padding_top, padding_bottom), (0, 0))
    # Else, pad horizontally
    elif h > w:
        padding_left = (h - w) // 2
        padding_right = h - w - padding_left
        padding_dims = ((0, 0), (padding_left, padding_right))
        
        
    padded_image = np.pad(image, padding_dims, mode='constant', constant_values=0)
    
    # If the padded image is smaller than 512x512, resize it to 256x256
    #h_padded, w_padded = padded_image.shape[:2]
    #if h_padded < 512 and w_padded < 512:
    #    padded_image = cv2.resize(padded_image, (256, 256))
    
    return padded_image


# Test the function
# image = np.random.rand(500, 412, 3)  # Example non-square image
# result = crop_to_power_of_two(image)



def gamma_distribution(x, k, theta, A):
    """Gamma distribution function scaled by amplitude A"""
    return A * (x**(k-1) * np.exp(-x/theta)) / (theta**k * gamma(k))


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
    if H<512 and W<512:
        return cv2.resize(image, (256, 256), interpolation=cv2.INTER_LANCZOS4)
    else:
        return image



def ctc_2_CNet_track(path_X,suff_X,path_Y,suff_Y,start_ind,mu,shape,half=True,prompt='hela_ctc',alt=0,save_dir_org='/home/mo/Desktop/IWR/Cell_GT_Proj/CNet_cells_track/',test=False,step=1,specific_i=None,mito_d=4):
    
    #data=[]
    test_entries = []
    train_entries = []
    l_ind=start_ind
    if specific_i is not None:
        end=1
    else:
        end=999
        
    save_dir=save_dir_org
    save_dir_test=save_dir.replace('training', 'testing')
    save_dir_test=save_dir_test.replace('train', 'test')
    
    if not os.path.exists(save_dir + '/target/'):
        os.makedirs(save_dir + '/target/')
        
    if not os.path.exists(save_dir + '/source/'):
        os.makedirs(save_dir + '/source/')
        
    if not os.path.exists(save_dir_test + '/target/'):
        os.makedirs(save_dir_test + '/target/')
        
    if not os.path.exists(save_dir_test + '/source/'):
        os.makedirs(save_dir_test + '/source/')
        
    for i in range(end):
        test=False
        save_dir=save_dir_org
        
        
        try:
            ind=i+step
            path_r_X0 = path_X +'/' +suff_X + '{:03d}'.format(ind) + '.tif'
            path_r_Y0 = path_Y+'/TRA/'+suff_Y+'{:03d}'.format(ind)+'.tif'
            path_r_X1 = path_X +'/' +suff_X + '{:03d}'.format(i) + '.tif'
            path_r_Y1 = path_Y+'/TRA/'+suff_Y+'{:03d}'.format(i)+'.tif'

            if specific_i is not None:

                path_r_X0 = path_X + suff_X + '{:03d}'.format(specific_i+1) + '.tif'
                path_r_Y0 = path_Y + suff_Y + '{:03d}'.format(specific_i+1) + '.tif'
                path_r_X1 = path_X + suff_X + '{:03d}'.format(specific_i) + '.tif'
                path_r_Y1 = path_Y + suff_Y + '{:03d}'.format(specific_i) + '.tif'





            img_r_X0 = imread(path_r_X0)
            img_r_Y0 = imread(path_r_Y0)
            img_r_X1 = imread(path_r_X1)
            img_r_Y1 = imread(path_r_Y1)

            max_X = np.max(img_r_X0)
            img_r_X0 = (img_r_X0/max_X)*255
            img_r_X1 = (img_r_X1/max_X)*255


            
            
            
           
            if half:
                #print('shape',shape)
                start_x = np.random.randint(0, int(shape[1]/2))
                end_x = start_x + int(shape[1]/2)
                start_y = np.random.randint(0, int(shape[0]/2))
                end_y = start_y + int(shape[0]/2)
 
                img_r_X0 = img_r_X0[start_y:end_y, start_x:end_x]
                img_r_X1 = img_r_X1[start_y:end_y, start_x:end_x]
                img_r_Y0 = img_r_Y0[start_y:end_y, start_x:end_x]
                img_r_Y1 = img_r_Y1[start_y:end_y, start_x:end_x]
            
                
                img_r_X0=make_square(img_r_X0)
                img_r_X1=make_square(img_r_X1)
                img_r_Y0=make_square(img_r_Y0)
                img_r_Y1=make_square(img_r_Y1)
                
            else:
                img_r_X0=make_square(img_r_X0)
                img_r_X1=make_square(img_r_X1)
                img_r_Y0=make_square(img_r_Y0)
                img_r_Y1=make_square(img_r_Y1)




            probability = 0.1  # 10% probability

            # Generate a random number between 0 and 1
            random_number = random.random()

            # Check if the random number is less than the desired probability
            if random_number < probability:
                test=True
                #print('is_True',)

            if test:
                save_dir=save_dir_test

            with open(path_Y+'/TRA/'+'man_track.txt', 'r') as file:
                lineage = file.readlines()


            img_r_Y1=connect_matching_dots(img_r_Y0,img_r_Y1,path_Y+'/TRA/',ind,img_r_X0,i+step,mu,lineage=lineage,d=mito_d)
            
            
            number_of_images = count_images_in_directory(save_dir + '/target/')
            if (save_dir == save_dir_test and number_of_images < 20) or save_dir != save_dir_test:
            
                output_path_X0 = save_dir + '/target/' + str(start_ind + i + 1)+'_'+ str(i)+ '.png'
                output_path_Y0 = save_dir + '/source/' + str(start_ind + i + 1) +'_'+ str(i)+ '.png'
                output_path_X1 = save_dir + '/target/' + str(start_ind + i) + '_'+ str(i)+'.png'
                output_path_Y1 = save_dir + '/source/' + str(start_ind + i) + '_'+ str(i)+'.png'


                img_r_Y1 = (img_r_Y1).astype(np.uint8)


                img_r_X1,img_r_Y1=augment_images(img_r_X1, img_r_Y1, alt)
                
                            
                img_r_Y1=resize_to_512_if_needed(img_r_Y1)
                img_r_X1=resize_to_512_if_needed(img_r_X1)

                #cv2.imwrite(output_path_Y0, combined_imag0)
                cv2.imwrite(output_path_Y1, img_r_Y1)
                #cv2.imwrite(output_path_X0, img_r_X0)
                cv2.imwrite(output_path_X1, img_r_X1)
                l_ind = l_ind + 1
                #print('out',output_path_X1)
                #print(stop)




                entry = {
                "source": 'source/' + str(start_ind + i) + '_'+ str(i)+'.png',
                "target": 'target/' + str(start_ind + i) + '_'+ str(i)+'.png',
                "prompt": prompt
                }

                # Append the entry to the data list
                #data.append(entry)

                if test:
                    test_entries.append(entry)

                else:
                    train_entries.append(entry)




                        # Write the data list as JSON to a file



        except Exception as e:
            if not str(e).startswith("[Errno 2]"):
                 traceback.print_exc()
                    #print("An error occurred:", str(e))

    save_list=[save_dir,save_dir_test]    
    for h in range(2): 
        save_dir=save_list[h]
        if h==0:
            data = train_entries
        else:
            data = test_entries
    
        output_path = save_dir+"/prompt.json"    
        existing_data = []



        # Read existing data from the JSON file, if it exists
        try:
            with open(output_path, "r") as infile:
                for line in infile:
                    line = line.strip()  # Remove leading/trailing whitespaces
                    if line:
                        data_j = json.loads(line)
                        existing_data.append(data_j)
        except FileNotFoundError:
            pass

        #print('!!!!!!ex',existing_data)
        #print('!!!!!!dat',data)
        # Append new entries to the existing data
        existing_data=existing_data+data
        #print('!!!!!!ex2',existing_data)

        # Write the updated data to the JSON file
        with open(output_path, "w") as outfile:
            for entry in existing_data:
                #print('ent',entry)
                json.dump(entry, outfile)
                outfile.write('\n') 

    return l_ind



def count_images_in_directory(directory_path):
    return sum([len(files) for r, d, files in os.walk(directory_path)])


def connect_matching_dots(img1, img2, path, t, cells, frame,mu,d,test=False, deep=False, lineage=None):
    # Find unique pixel values and their counts in both images
    unique_values_1, counts_1 = np.unique(img1, return_counts=True)
    unique_values_2, counts_2 = np.unique(img2, return_counts=True)
    counts_2_dict = dict(zip(unique_values_2, counts_2))
    
    # Create an output image for dots and lines separately
    output_img_dots = np.zeros_like(img1)
    output_img_lines = np.zeros_like(img1)
    output_img_splits = np.zeros_like(img1)

    # Initialize 3-channel image with zeros
    output_img_dots = np.stack((output_img_dots,)*3, axis=-1)
    output_img_lines = np.stack((output_img_lines,)*3, axis=-1)
    output_img_splits = np.stack((np.zeros_like(img1),)*3, axis=-1)

    # Draw dots in blue
    output_img_dots[:, :, 2] = cells  
    radius = int(np.sqrt(mu / np.pi) / 4) if int(np.sqrt(mu / np.pi) / 4) > 2 else 2
    
    #rint('!!!!radius',radius)
    
    # Draw lines and circles in green
    for value in unique_values_1[unique_values_1 > 0]:
        # Find the coordinates of matching dots in both images
        coordinates_img1 = np.argwhere(img1 == value)
        coordinates_img2 = np.argwhere(img2 == value)

        if coordinates_img2.size > 0: 
            
            coord1 = np.mean(coordinates_img1, axis=0, dtype=int)
            coord2 = np.mean(coordinates_img2, axis=0, dtype=int)

            flag = check_split_timeframe(frame, value, lineage, d)

            if test:
                coord2 = test_transform(coord1, coord2)

            line_color = [55, 200, 0]
            circle_color = get_gradient_color(d, flag)

            cv2.line(output_img_lines, tuple(coord1[::-1]), tuple(coord2[::-1]), color=line_color, thickness=3)
            circle_color = [int(val) for val in circle_color]
            cv2.circle(output_img_lines, tuple(coord2[::-1]), radius, color=circle_color, thickness=-1)
            # Define the radius for this specific point

        else:
            cs = check_split(path, t, value)
                
            if cs[0]:
                coordinates_img1 = np.argwhere(img1 == value)
                coordinates_img2 = np.argwhere(img2 == cs[1])
                if coordinates_img2.size > 0: 
                    coord1 = np.mean(coordinates_img1, axis=0, dtype=int)
                    coord2 = np.mean(coordinates_img2, axis=0, dtype=int)
                    cv2.line(output_img_splits, tuple(coord1[::-1]), tuple(coord2[::-1]), color=[200, 55, 0], thickness=3)
                    cv2.circle(output_img_splits, tuple(coord2[::-1]), radius, color=[255, 0, 0], thickness=-1)  # Filled circle

    # Stack images along the channel dimension: Red for splits, Green for lines, Blue for dots
    output_img = output_img_dots + output_img_lines + output_img_splits
    return output_img


def check_split(path,t,value):
    data = np.loadtxt(path+'man_track.txt')

    # Split the data into columns
    start = data[:, 1]
    ide = data[:, 0][start==t]
    
    end = data[:, 2][start==t]
    parent = data[:, 3][start==t]
    
    try:
        par = parent[ide==value]
        #print('p_t_pa_v',par,t,path,value)
        
        if len(ide[parent==par])==2:
            #print('ide',ide[parent==par])
            return True,par[0]
        else:
            return False,0
        
    except:
        return False,0
    
def parse_lineage_data(data):
    """
    Convert the given lineage data to a list of dictionaries.
    """
    result = []
    for line in data:
        L, B, E, P = map(int, line.split())
        result.append({
            'label': L,
            'frame_start': B,
            'frame_end': E,
            'parent': P
        })
    return result



# Reading the file
def get_gradient_color(d, flag):
    base_color = [0, 255, 0]  # Pure green as base color

    if flag == 0:
        return base_color
    elif flag > 0:
        return [255-((flag-1)*(255/(d-1))),255,0]
    else:  # flag < 0
        return [255,50+((abs(flag)-1)*(150/(d-1))),0]
    


def check_split_timeframe(t, cell_id, lineage_data, d, pos=False):
    #print('mito_d',d)
    parsed_data = parse_lineage_data(lineage_data)
    
    current_cell = next((cell for cell in parsed_data if cell['label'] == cell_id), None)
    if not current_cell:
        print('Cell not found', cell_id)
        return 0
    
    if current_cell['parent'] != 0:
        time_difference_from_merge = t - current_cell['frame_start']
        if 0 < time_difference_from_merge <= d:
            return -time_difference_from_merge
        elif time_difference_from_merge == 0 and pos:
            return 99
    
    for cell in parsed_data:
        if cell['parent'] == current_cell['label']:
            merge_point = cell['frame_start']
            time_difference_to_next_merge = merge_point - t
            if 0 < time_difference_to_next_merge <= d:
                return time_difference_to_next_merge
    return 0




def ctc_2_CNet(path_X,path_Y,suff_X,suff_Y,start_ind,mu,shape,half=True,prompt='hela_ctc',alt=0,save_dir='/home/mo/Desktop/IWR/Cell_GT_Proj/CNet_cells/',test=False,mito_d=4):
    
    
    end=999
    
    if test:
        save_dir=save_dir.replace('training', 'testing')
        save_dir=save_dir.replace('train', 'test')
        
        end=10
    
    if not os.path.exists(save_dir + '/target/'):
        os.makedirs(save_dir + '/target/')
        
    if not os.path.exists(save_dir + '/source/'):
        os.makedirs(save_dir + '/source/')
    
    
    data=[]
    l_ind=start_ind
    for i in range(end):
        
        try:
            path_r_X=path_X+'/'+suff_X+'{:03d}'.format(i)+'.tif'
            path_r_Y=path_Y+'/TRA/'+suff_Y+'{:03d}'.format(i)+'.tif'
            # read the tif file
            #print('p',path_r)
            #try:
            #img_r_X = tiff.imread(path_r_X)
            #img_r_Y = tiff.imread(path_r_Y)
            img_r_X = imread(path_r_X)
            img_r_Y = imread(path_r_Y)


            max_X=np.max(img_r_X)
            img_r_X = (img_r_X/max_X)*255
            #max_y=np.max(img_r_Y)
            #img_r_Y = (img_r_Y/max_y)*255
            

            if half:
                #print('shape',shape)
                start_x = np.random.randint(0, int(shape[1]/2))
                end_x = start_x + int(shape[1]/2)
                start_y = np.random.randint(0, int(shape[0]/2))
                end_y = start_y + int(shape[0]/2)
                #print(start_y,end_y, start_x,end_x)
                # Crop the random region from img_r_X
                img_r_X = img_r_X[start_y:end_y, start_x:end_x]

                # Crop the random region from img_r_Y
                img_r_Y = img_r_Y[start_y:end_y, start_x:end_x]
                
                img_r_X=make_square(img_r_X)
                img_r_Y=make_square(img_r_Y)
            else:
                img_r_X=make_square(img_r_X)
                img_r_Y=make_square(img_r_Y)
                




            #print('both shapes',img_r_X.shape,img_r_Y.shape)
            img_r_X,img_r_Y=augment_images(img_r_X, img_r_Y,alt=alt)


            #if alt==1:
            #    cv2.imwrite(output_path_Y_c, img_r_Y)
            with open(path_Y+'/TRA/'+'man_track.txt', 'r') as file:
                lineage = file.readlines()


            img_r_Y,pos_list = source2CNet(img_r_Y, mu,i,lineage,mito_d,test=test)
            
            if test:
                file_path = os.path.join(save_dir, 'numbers.txt')
                if not os.path.exists(file_path):
                    with open(file_path, 'w') as f:
                        f.write("image\tnumber of cells\n")

                        # Determine number of cells
                        if pos_list is not None and len(pos_list) > 0:
                            num_cells = len(pos_list)
                        else:
                            num_cells = 0

                        f.write(f"{start_ind+i}\t{num_cells}\n")
                else:
                    with open(file_path, 'a') as f:
                        # Determine number of cells
                        if pos_list is not None and len(pos_list) > 0:
                            num_cells = len(pos_list)
                        else:
                            num_cells = 0

                        f.write(f"{start_ind+i}\t{num_cells}\n")

                #print('pos',pos_list,len(pos_list))

            #print('source worked')

            # Specify the output file path





            output_path_X = save_dir+'/target/'+str(start_ind+i)+'.png'
            output_path_Y = save_dir+'/source/'+str(start_ind+i)+'.png'
            #output_path_Y_c = save_dir+'source/'+str(start_ind+i)+'contr.png'

                            
            img_r_X=resize_to_512_if_needed(img_r_X)
            img_r_Y=resize_to_512_if_needed(img_r_Y)

            # Save the resized image as PNG
            cv2.imwrite(output_path_X, img_r_X)
            cv2.imwrite(output_path_Y, img_r_Y)


            entry = {
            "source": "source/" + str(start_ind+i) + ".png",
            "target": "target/" + str(start_ind+i) + ".png",
            "prompt": prompt
            }

            # Append the entry to the data list
            data.append(entry)

            l_ind=l_ind+1

                # Write the data list as JSON to a file


        
        except Exception as e:
            #print("An error occurred:", str(e))
            if not str(e).startswith("[Errno 2]"):
                traceback.print_exc()
            #
        
        
        
    output_path = save_dir + "/prompt.json"

    # Check if file exists, if not create it
    if not os.path.exists(output_path):
        open(output_path, 'a').close()   
    existing_data = []



    # Read existing data from the JSON file, if it exists
    try:
        with open(output_path, "r") as infile:
            for line in infile:
                line = line.strip()  # Remove leading/trailing whitespaces
                if line:
                    data_j = json.loads(line)
                    existing_data.append(data_j)
    except FileNotFoundError:
        pass

    #print('!!!!!!ex',existing_data)
    #print('!!!!!!dat',data)
    # Append new entries to the existing data
    existing_data=existing_data+data
    #print('!!!!!!ex2',existing_data)

    # Write the updated data to the JSON file
    with open(output_path, "w") as outfile:
        for entry in existing_data:
            #print('ent',entry)
            json.dump(entry, outfile)
            outfile.write('\n')    
        
    return l_ind





def mean_ind(indices):
    """Get the mean indices from a list of indices."""
    return np.mean(indices, axis=0)

def source2CNet(img, mu,t,lineage_data,mito_d,test=False):
    
    # Find unique values and their counts, ignoring zeros
    unique_values, unique_counts = np.unique(img, return_counts=True)
    non_zero_mask = unique_values != 0
    unique_values = unique_values[non_zero_mask]
    unique_counts = unique_counts[non_zero_mask]

    # Lists to store the mean position of each unique value and their respective counts
 
    pos_list = []
    flag_list = []

    for value in unique_values:
        mask = img == value
        indices = np.argwhere(mask)
        m_ind = mean_ind(indices)
        pos_list.append(m_ind)
        flag_list.append(check_split_timeframe(t, value, lineage_data, mito_d, pos=True))

    # Initialize an output image of zeros
    image_tensor = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    if test:
        pos_list = generate_similar_positions2(pos_list)
        flag_list = generate_flag_list(flag_list)

    # Determine the radius for the painted dots
    radius = int(np.sqrt(mu / np.pi) / 4) if int(np.sqrt(mu / np.pi) / 4) > 2 else 2

    # Iterate over the mean positions and draw the dots
    for position, flag in zip(pos_list, flag_list):
        x, y = int(position[0]), int(position[1])
        color = get_color(flag, mito_d)
        #print('col',color,'flag',flag)
        #color=[55, 200, 0]#[255, 255, 255]
        color = [int(val) for val in color]
        #print('col2',color)
        cv2.circle(image_tensor, (y, x), radius, color, -1)  # Filled circle
    return image_tensor, pos_list

def get_color(flag, d):
    gradient_color = get_gradient_color(d, flag)

    # Handle the special case where flag is 99
    if flag == 99:
        gradient_color = [255, 0, 0]

    return gradient_color  # Return the color as a list

def generate_similar_positions2(positions):
    # Calculate pairwise distances
    positions = np.array(positions)
    try:
        pairwise_distances = np.linalg.norm(positions[:, None] - positions, axis=2)
    except:
        return positions

    # Get the minimum distance from the pairwise_distances
    min_distance = np.min(pairwise_distances + np.eye(pairwise_distances.shape[0]) * np.max(pairwise_distances))

    # Create an empty list for new positions
    new_positions = []

    # Initial random position
    new_positions.append(np.random.uniform(low=positions.min(axis=0), high=positions.max(axis=0)))

    # Maximal number of attempts to generate a point that keeps minimal distance
    max_attempts = 1000
    attempts = 0

    while len(new_positions) < len(positions):
        if attempts > max_attempts:
            #print("Cannot generate desired number of positions with the current minimal distance.")
            break

        # Sample random point
        random_point = np.random.uniform(low=positions.min(axis=0), high=positions.max(axis=0))

        # Construct KDTree
        tree = KDTree(new_positions)

        # Get the distances and indices of the closest points to the random_point
        distances, indices = tree.query(random_point, k=1)

        # Check if the minimum distance condition holds
        if distances >= min_distance:
            new_positions.append(random_point.tolist())
        else:
            attempts += 1

    # Convert list to numpy array
    if len(new_positions) > 1 and random.random() < 0.70:
        n = np.random.randint(1, len(new_positions))
        new_positions = np.array(new_positions[0:n])

        
    else:
        new_positions = np.array(new_positions)
        

    return new_positions

def generate_flag_list(input_list):
    # Define probabilities and values
    probabilities = [0.8, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02]
    values = [0, 1, 2, 3, -1, -2, -3, -4, 99]

    # Generate the new list
    new_list = [random.choices(values, probabilities)[0] for _ in input_list]

    return new_list

def list_target_directories(path):
    # A set of names we are looking for
    target_names = {f"{i:02}" for i in range(1, 100)} # assuming 01 to 99 as the possible names

    # List to store the directories found
    found_dirs = []

    # Walk through the root directory
    for dirpath, dirnames, filenames in os.walk(path):
        for dirname in dirnames:
            if dirname in target_names:
                found_dirs.append(os.path.join(dirpath, dirname))
    
    return found_dirs


def generate_CTC_datasets(path_stats,path_orig,name,save_dir,d):
    #cellpose_segmentation(path,diameter=diameter)
    mu,sig,shape=get_stats(path_stats)
    #print(stop)
    generate_XY_pairs(path_orig,mu,sig,shape,name, save_dir,d)
    

def get_stats(path):
    file_path = path

    # Initialize a dictionary to hold the variables
    variables = {}

    # Open and read the file line by line
    with open(file_path, 'r') as file:
        for line in file:
            # Split each line into key and value
            if ': ' in line:
                key, value = line.split(': ', 1)
                key = key.strip()
                value = value.strip()
                # Store in the dictionary
                variables[key] = value

    # Now you can access the variables
    # For example, to print them:


   
    # Convert to specific data types
    A_mu = float(variables['A_mu']) if 'A_mu' in variables else None
    A_sigma = float(variables['A_sigma']) if 'A_sigma' in variables else None

    # For 'Shape', convert the string to a tuple of integers
    if 'Shape' in variables:
        shape = tuple(map(int, variables['Shape'].strip('()').split(',')))
    else:
        shape = None

    return A_mu, A_sigma, shape



def one_gaussian(x, A1, mu1, sigma1):
    g1 = A1 * np.exp(-(x - mu1)**2 / (2 * sigma1**2))
    return g1
   


def inverse_power_law(x, A, n):
    return A / (x ** n)


def power_law_pdf(x, n, A):
    return A / (x ** n)

# Normalization constant A for defined interval [1, b]
def find_A(n, b=10):
    integral, _ = quad(power_law_pdf, 1, b, args=(n, 1))
    return 1 / integral

# CDF of the power-law function
def power_law_cdf(x, n, A):
    def integrand(t):
        return power_law_pdf(t, n, A)
    result, _ = quad(integrand, 1, x)
    return result

# Inverse of CDF
def inverse_cdf(u, n, A, b=10):
    # Using a simple binary search for the inverse
    low, high = 1, b
    while high - low > 1e-6:
        mid = (low + high) / 2
        if power_law_cdf(mid, n, A) < u:
            low = mid
        else:
            high = mid
    return (low + high) / 2



#samples = sample_from_power_law(n, b,size=5000)

    
def calculate_displacement(img1, img2):
    displacements = []

    unique_values1 = np.unique(img1)
    unique_values1 = unique_values1[unique_values1 != 0]

    unique_values2 = np.unique(img2)
    unique_values2 = unique_values2[unique_values2 != 0]

    for val in unique_values1:
        if val in unique_values2:
            y1, x1 = np.where(img1 == val)
            y2, x2 = np.where(img2 == val)

            mean_coord1 = [np.mean(x1), np.mean(y1)]
            mean_coord2 = [np.mean(x2), np.mean(y2)]

            distance = np.sqrt((mean_coord1[0] - mean_coord2[0])**2 + (mean_coord1[1] - mean_coord2[1])**2)
            displacements.append(distance)

    return displacements

def augment_images(image_X, image_Y, alt):

    # Rotate the images based on the 'alt' value
    rotate_degrees = alt * 90  # Rotate by 90, 180, 270, or 360 degrees
    aug = iaa.Affine(rotate=rotate_degrees)
    augmented_images = aug(images=[image_X, image_Y])
    
    # Return the augmented images
    augmented_image_X, augmented_image_Y = augmented_images
    return augmented_image_X, augmented_image_Y

def generate_pos_pairs(path, mu, sig, shape, name, save_dir, d, num_cut_outs=3):
    suff_X = 't'
    suff_Y = 'man_track'
    path_X, path_Y = find_folders(path)
    prompt_ctc = 'cell, microscopy, image'
    l_ind = 0
    alt = [0, 1, 2, 3]
    save_dir_0 = save_dir + '_' + str(int(shape[1]/2)) + '_' + str(int(shape[0]/2))

    total_iterations = num_cut_outs * len(path_X) * len(alt) + len(path_X) * 1 + len(path_X) * len(alt) + len(path_X) * 1

    with tqdm(total=total_iterations, desc="Generating Positional Conditioning") as pbar:
        for g in range(num_cut_outs):
            for k in range(len(path_X)):
                for h in range(len(alt)):
                    l_ind = ctc_2_CNet(path_X[k], path_Y[k], suff_X, suff_Y, start_ind=l_ind, mu=mu, shape=shape, half=True, prompt=prompt_ctc, alt=alt[h], save_dir=save_dir_0, test=False, mito_d=d)
                    pbar.update(1)

        l_ind = 0
        for k in range(len(path_X)):
            for h in range(1):
                l_ind = ctc_2_CNet(path_X[k], path_Y[k], suff_X, suff_Y, start_ind=l_ind, mu=mu, shape=shape, half=True, prompt=prompt_ctc, alt=0, save_dir=save_dir_0, test=True, mito_d=d)
                pbar.update(1)

        l_ind = 0
        if shape[1] != shape[0]:
            alt = [0, 2]
        else:
            alt = [0, 1, 2, 3]
        save_dir_0 = save_dir + '_' + str(int(shape[1])) + '_' + str(int(shape[0]))
        for k in range(len(path_X)):
            for h in range(len(alt)):
                l_ind = ctc_2_CNet(path_X[k], path_Y[k], suff_X, suff_Y, start_ind=l_ind, mu=mu, shape=shape, half=False, prompt=prompt_ctc, alt=alt[h], save_dir=save_dir_0, test=False, mito_d=d)
                pbar.update(1)

        l_ind = 0
        for k in range(len(path_X)):
            for h in range(1):
                l_ind = ctc_2_CNet(path_X[k], path_Y[k], suff_X, suff_Y, start_ind=l_ind, mu=mu, shape=shape, half=False, prompt=prompt_ctc, alt=0, save_dir=save_dir_0, test=True, mito_d=d)
                pbar.update(1)
def find_folders(root_path):
    path_X = []
    path_Y = []
    
    for dirpath, dirnames, filenames in os.walk(root_path):
        for dirname in dirnames:
            # Check if folder name matches the pattern XX
            if len(dirname) == 2 and dirname.isdigit():
                path_X.append(os.path.join(dirpath, dirname))
            # Check if folder name matches the pattern XX_GT
            elif dirname.endswith('_GT') and len(dirname) == 5 and dirname[:2].isdigit():
                path_Y.append(os.path.join(dirpath, dirname))
                
    return path_X, path_Y
    
    
def generate_track_pairs(path, mu, sig, shape, name, save_dir, d, num_cut_outs=3):
    suff_X = 't'
    suff_Y = 'man_track'
    prompt_ctc = 'cell, microscopy, image'
    path_X, path_Y = find_folders(path)
    l_ind = 0
    alt = [0, 1, 2, 3]
    steps = [1]

    # Calculate total iterations for both parts of the process
    total_iterations = 2 * num_cut_outs * len(path_X) * len(steps) * len(alt)

    save_dir_0 = save_dir + '_' + 'track' + '_' + str(int(shape[1]/2)) + '_' + str(int(shape[0]/2))

    with tqdm(total=total_iterations, desc="Generating Track Conditioning") as pbar:
        for g in range(num_cut_outs):
            for k in range(len(path_X)):
                for q in steps:
                    for h in range(len(alt)):
                        l_ind = ctc_2_CNet_track(path_X[k], suff_X, path_Y[k], suff_Y, start_ind=l_ind, mu=mu, shape=shape, half=True, prompt=prompt_ctc, alt=alt[h], save_dir_org=save_dir_0, step=q, mito_d=d)
                        pbar.update(1)  # Update progress bar
                        #print(l_ind)

        l_ind = 0
        save_dir_0 = save_dir + '_' + 'track' + '_' + str(int(shape[1])) + '_' + str(int(shape[0]))
        for k in range(len(path_X)):
            for q in steps:
                for h in range(len(alt)):
                    l_ind = ctc_2_CNet_track(path_X[k], suff_X, path_Y[k], suff_Y, start_ind=l_ind, mu=mu, shape=shape, half=False, prompt=prompt_ctc, alt=alt[h], save_dir_org=save_dir_0, step=q, mito_d=d)
                    pbar.update(1)  # Continue updating the same progress bar
                    #print(l_ind)
    
def generate_XY_pairs(path,mu,sig,shape,name, save_dir,d):
    #print(stop)
    #print('##pos##')
    generate_pos_pairs(path,mu,sig,shape,name, save_dir,d)
    #print(stop)
    #print('##track##')
    generate_track_pairs(path,mu,sig,shape,name, save_dir,d)

