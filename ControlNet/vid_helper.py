import os
import glob
import cv2
import numpy as np

def delete_files_in_folder(directory):
    """Delete all files in a specified directory."""
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

def check_split(path,t,value):
    xl,yl,rl,idel,split_l,s_pr_l,t_vl= np.loadtxt(path+'pos_GT.txt',skiprows=1, delimiter='\t', usecols=(0,1,2,3,4,5,6), unpack=True)
    id_t = idel[t_vl==t]
    parent = split_l[t_vl==t]
    num= int(np.max(idel))
    ide=value
    

    par = parent[id_t==ide]
    print('ide,par',ide,parent,t)
    if par != 0:
        print('ide,par',ide,par)
        return True,par[0]

    else:
        return False,0
    

    





def connect_matching_dots(img1, img2, path, t, cells, test=False):
    # Find unique pixel values in both images
    print('poss_val',np.linspace(100,255,12))
    unique_values_1 = np.unique(img1)
    unique_values_2 = np.unique(img2)

    # Create an output image for dots and lines separately
    output_img_dots = np.zeros_like(img1)
    output_img_lines = np.zeros_like(img1)
    output_img_splits = np.zeros_like(img1)

    # Initialize 3-channel image with zeros
    output_img_dots = np.stack((output_img_dots,)*3, axis=-1)
    output_img_lines = np.stack((output_img_lines,)*3, axis=-1)
    output_img_splits = np.stack((np.zeros_like(img1),)*3, axis=-1)

    # Draw dots in blue
    output_img_dots[:, :, 2] = cells  # RGB color for Blue 

    # Draw lines and circles in green
    for value in unique_values_1[unique_values_1 > 0]:
        # Find the coordinates of matching dots in both images
        coordinates_img1 = np.argwhere(img1 == value)
        coordinates_img2 = np.argwhere(img2 == value)

        if coordinates_img2.size > 0: 
            coord1 = np.mean(coordinates_img1, axis=0, dtype=int)
            coord2 = np.mean(coordinates_img2, axis=0, dtype=int)
            print('working...',coord1,coord2,value)
            cv2.line(output_img_lines, tuple(coord1[::-1]), tuple(coord2[::-1]), color=[0, 255, 0], thickness=2)
            cv2.circle(output_img_lines, tuple(coord2[::-1]), 3, color=[0, 255, 0], thickness=-1)  # Filled circle
        else:
            cs = check_split(path, t, value)
            print('cs', cs)
            if cs[0]:
                coordinates_img1 = np.argwhere(img1 == value)
                coordinates_img2 = np.argwhere(img2 == cs[1])
                if coordinates_img2.size > 0: 
                    coord1 = np.mean(coordinates_img1, axis=0, dtype=int)
                    coord2 = np.mean(coordinates_img2, axis=0, dtype=int)
                    
                    cv2.line(output_img_splits, tuple(coord1[::-1]), tuple(coord2[::-1]), color=[255, 0, 0], thickness=2)
                    cv2.circle(output_img_splits, tuple(coord2[::-1]), 3, color=[255, 0, 0], thickness=-1)  # Filled circle

    # Stack images along the channel dimension: Red for splits, Green for lines, Blue for dots
    output_img = output_img_dots + output_img_lines + output_img_splits
    return output_img




def ctc_2_track(i, res_path, id_path, save_path,r_a):
     # File names
    x_img_path = os.path.join(res_path, f"{i}.png")
    #y_img_path = os.path.join(id_path, f"{i}.jpg")
    #y_img2_path = os.path.join(id_path, f"{i-1}.jpg")

    # Load the images
    x_img = cv2.imread(x_img_path, 0)
    #y_img = cv2.imread(y_img_path, 0)
    #y_img2 = cv2.imread(y_img2_path, 0)

    if x_img.shape[:2] != (512, 512):
        x_img_resized = cv2.resize(x_img, (512, 512), interpolation=cv2.INTER_AREA)
    else:
        x_img_resized = x_img
    x_img=x_img_resized
    # Error checking
    #if x_img is None or y_img is None or y_img2 is None:
    #    raise ValueError('One or more images could not be read')

    # Call connect_matching_dots function
     
    out = connect_matching_coords(x_img, x_img, id_path, i, x_img,r_a)

    output_file = os.path.join(save_path, f"{i}.jpg")
    cv2.imwrite(output_file, out)

    print('save_path',output_file)




    return 0




    
def revert_val_id(start,end,value,num):
    
    c_list = np.linspace(start,end,int(num))
    ind = np.abs(c_list - value).argmin()
    ind += 1

    return ind




def connect_matching_coords(img1, img2, path, t, cells, r_a,test=False):



    xl,yl,rl,idel,split_l,s_pr_l,t_vl= np.loadtxt(path+'pos_GT.txt',skiprows=1, delimiter='\t', usecols=(0,1,2,3,4,5,6), unpack=True)




    mask = split_l != -1
    xl = xl[mask]
    yl = yl[mask]
    rl = rl[mask]
    idel = idel[mask]
    split_l = split_l[mask]
    s_pr_l = s_pr_l[mask]
    t_vl = t_vl[mask]

    y0 = xl[t_vl==t]
    y1 = xl[t_vl==(t-1)]
    rl=rl[t_vl==(t-1)]

    x0 = 1-yl[t_vl==t]
    x1 = 1-yl[t_vl==(t-1)]

    id0 = idel[t_vl==t]
    id1 = idel[t_vl==(t-1)]











    # Find unique pixel values in both images



    # Create an output image for dots and lines separately
    output_img_dots = np.zeros_like(img1)
    output_img_lines = np.zeros_like(img1)
    output_img_splits = np.zeros_like(img1)

    # Initialize 3-channel image with zeros
    output_img_dots = np.stack((output_img_dots,)*3, axis=-1)
    output_img_lines = np.stack((output_img_lines,)*3, axis=-1)
    output_img_splits = np.stack((np.zeros_like(img1),)*3, axis=-1)

    # Draw dots in blue
    output_img_dots[:, :, 2] = cells  # RGB color for Blue 

    # Draw lines and circles in green
    for value in id0:

        # Find the coordinates of matching dots in both images
        coordinates_img1 = (x0[id0==value][0]*512,y0[id0==value][0]*512)
        try:
            r=rl[id1==value]*512
            r = r_a#int(map_value_linear(np.pi*(r**2), 10, 1500, 2, 8))
            coordinates_img2 = (x1[id1==value][0]*512,y1[id1==value][0]*512)
            coord1 = tuple(map(int, coordinates_img1))
            coord2 = tuple(map(int, coordinates_img2))
            flag=check_split_timeframe(t, value, t_vl, idel, split_l)
            print('working...',coord1,coord2,value)
            line_color = [55, 200, 0]
            circle_color = get_gradient_color(4, flag)

            cv2.line(output_img_lines, tuple(coord1[::-1]), tuple(coord2[::-1]), color=line_color, thickness=3)
            circle_color = [int(val) for val in circle_color]
            cv2.circle(output_img_lines, tuple(coord2[::-1]), r, color=circle_color, thickness=-1)
            '''
            if flag==0:
                cv2.line(output_img_lines, tuple(coord1[::-1]), tuple(coord2[::-1]), color=[55, 200, 0], thickness=3)
                cv2.circle(output_img_lines, tuple(coord2[::-1]), r, color=[0, 255, 0], thickness=-1) # Filled circle
            elif flag==1:
                cv2.line(output_img_lines, tuple(coord1[::-1]), tuple(coord2[::-1]), color=[55, 200, 0], thickness=3)
                cv2.circle(output_img_lines, tuple(coord2[::-1]), r, color=[255, 255, 0], thickness=-1) # Filled circle
            elif flag==2:
                #print('flag2',t,frame)
                cv2.line(output_img_lines, tuple(coord1[::-1]), tuple(coord2[::-1]), color=[55, 200, 0], thickness=3)
                cv2.circle(output_img_lines, tuple(coord2[::-1]), r, color=[170, 255, 0], thickness=-1) # Filled
            elif flag==3:
                #print('flag2',t,frame)
                cv2.line(output_img_lines, tuple(coord1[::-1]), tuple(coord2[::-1]), color=[55, 200, 0], thickness=3)
                cv2.circle(output_img_lines, tuple(coord2[::-1]), r, color=[85, 255, 0], thickness=-1) # Filled 
            elif flag==-1:
                #print('flag2',t,frame)
                cv2.line(output_img_lines, tuple(coord1[::-1]), tuple(coord2[::-1]), color=[55, 200, 0], thickness=3)
                cv2.circle(output_img_lines, tuple(coord2[::-1]), r, color=[255, 50, 0], thickness=-1) # Filled 
                
            elif flag==-2:
                #print('flag2',t,frame)
                cv2.line(output_img_lines, tuple(coord1[::-1]), tuple(coord2[::-1]), color=[55, 200, 0], thickness=3)
                cv2.circle(output_img_lines, tuple(coord2[::-1]), r, color=[255, 100,0],thickness=-1) # Filled
                
            elif flag==-3:
                #print('flag2',t,frame)
                cv2.line(output_img_lines, tuple(coord1[::-1]), tuple(coord2[::-1]), color=[55, 200, 0], thickness=3)
                cv2.circle(output_img_lines, tuple(coord2[::-1]), r, color=[255, 150, 0], thickness=-1) # Filled 
            elif flag==-4:
                #print('flag2',t,frame)
                cv2.line(output_img_lines, tuple(coord1[::-1]), tuple(coord2[::-1]), color=[55, 200, 0], thickness=3)
                cv2.circle(output_img_lines, tuple(coord2[::-1]), r, color=[255, 200, 0], thickness=-1) # Filled 
                
            #cv2.line(output_img_lines, tuple(coord1[::-1]), tuple(coord2[::-1]), color=[55, 200, 0], thickness=3)
            #cv2.circle(output_img_lines, tuple(coord2[::-1]), r, color=[0, 255, 0], thickness=-1)  # Filled circle
            '''
       
        except:
            cs = check_split(path, t, value)
            print('cs', cs)
            if cs[0]:
                r=rl[id1==cs[1]]*512
                r = r_a#int(map_value_linear(np.pi*(r**2), 10, 1500, 2, 8))
                coordinates_img2 = (x1[id1==cs[1]][0]*512,y1[id1==cs[1]][0]*512)
                coord1 = tuple(map(int, coordinates_img1))
                coord2 = tuple(map(int, coordinates_img2))
                    
                cv2.line(output_img_splits, tuple(coord1[::-1]), tuple(coord2[::-1]), color=[200, 55, 0], thickness=3)
                cv2.circle(output_img_splits, tuple(coord2[::-1]), r, color=[255, 0, 0], thickness=-1)  # Filled circle

       

    # Stack images along the channel dimension: Red for splits, Green for lines, Blue for dots
    output_img = output_img_dots + output_img_lines + output_img_splits
    return output_img

def get_gradient_color(d, flag):
    base_color = [0, 255, 0]  # Pure green as base color

    if flag == 0:
        return base_color
    elif flag > 0:
        return [255-((flag-1)*(255/(d-1))),255,0]
    else:  # flag < 0
        return [255,50+((abs(flag)-1)*(150/(d-1))),0]

def get_color(flag, d):
    gradient_color = get_gradient_color(d, flag)

    # Handle the special case where flag is 99
    if flag == 99:
        gradient_color = [255, 0, 0]

    return gradient_color  # Return the color as a list
def map_value_linear(value, in_min, in_max, out_min, out_max):
    # Ensure the input value is within the input range
    value = max(min(value, in_max), in_min)
    
    if value==in_min:
        return 1
    elif value== in_max:
        return 8

    # Map the value to the output range using linear interpolation
    return int((value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min)



'''

def check_split_timeframe(t, cell_id, t_vl, idel, split_l):
    
    # Search for the cell with the given cell_id
    indices = np.where(idel == cell_id)[0]  # Find all indices where idel matches cell_id
    
    for index in indices:
        if split_l[index] != 0.0:  # If the cell has a split_id
            split_time = t_vl[index]  # Extract the time of splitting
            time_difference = t - split_time
            
            if time_difference == 1:
                return 1
            elif time_difference == 2:
                return 2
                
    # If neither condition is met, return flag=0
    return 0

'''

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

