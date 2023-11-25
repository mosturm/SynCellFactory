import math
import os
import random

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.integrate import quad
from scipy.optimize import curve_fit
from skimage.draw import disk
from tifffile import imsave
from scipy.special import gamma
from tqdm import tqdm
plt.close('all')





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



    
    


def circle(x0,y0,r):
    
    x1=np.linspace(x0-r,x0+r,3000,endpoint=True)
    x2=np.linspace(x0-r,x0+r,3000,endpoint=True)
    y1=np.sqrt(r**2-(x1-x0)**2)+y0
    y2=-1*np.sqrt(r**2-(x1-x0)**2)+y0
    x=np.append(x1,x2)
    y=np.append(y1,y2)
        
    return x,y 

def circletest(x0, y0, r0, x1, y1, r1):
    # circle 1: (x0, y0), radius r0
    # circle 2: (x1, y1), radius r1

    d=np.sqrt((x1-x0)**2 + (y1-y0)**2)
    
    #print('d',d,r1,r0)
    
    # non intersecting
    if d > r0 + r1 and r0 != r1:
        #print('1')
        return 0
    # One circle within other
    if d < abs(r0-r1):
        #print('2')
        return 1
    # coincident circles
    if d == 0 and r0 == r1:
        #print('3')
        return 0
    else:
        #print('4')
        return 1

    
def splitprob(r,low,high):
    if r <= low:
        diff = 0
    else:
        diff = ((1/(high-low))**3 * (r-low)**3)*100
    return 100-diff

def distance(circle_a, circle_b):
    return ((circle_a.x - circle_b.x) ** 2 + (circle_a.y - circle_b.y) ** 2) ** 0.5

    
def ini_pic(b, r_mu, r_sig, pix_max,pix_min, t, run, path='.',bg=False):
    my_dpi = 96

    if not os.path.isdir(path):
        os.mkdir(path)
    if not os.path.isdir(path+'/' + str(run)):

        os.mkdir(path+'/' + str(run))
        os.mkdir(path+'/' + str(run) + '_GT')
        os.mkdir(path+'/' + str(run) + '_GT'+'/TRA')
    
    fig = plt.figure(figsize=(pix_max / my_dpi, pix_max / my_dpi), dpi=my_dpi)
    ax = fig.gca()
    
    circles = []
    
    x=np.array([])
    y=np.array([])
    x0c=np.array([2])
    y0c=np.array([0.5])
    rc=np.array([0])
    split_prob=np.array([0])
    
    
        # Calculate the margin or padding on the y-axis
    y_margin = (pix_max - pix_min) / 2

    # Normalize the start and stop values to the range [0, 1]


    x_pos = np.linspace(1 * r_mu, 1 - 1 * r_mu, 1500)
    y_pos = np.linspace(1 * r_mu + (y_margin / pix_max),1 - (y_margin / pix_max) - 1 * r_mu, int((pix_min/pix_max)*1500))
    
    for _ in range(b):
        x0 = np.random.choice(x_pos)
        y0 = np.random.choice(y_pos)
        r_com = np.random.normal(r_mu, r_sig, 1)
        #while r_com < 34/512:
        #    r_com = np.random.normal(r_mu, r_sig, 1)
        circle = Circle(x0, y0, r_com)
        circles.append(circle)
    
    resolve_all_overlaps(circles)
    
    # Extract updated x, y, r values for plotting
    xu, yu, ru = zip(*[(circle.x if isinstance(circle.x, float) else circle.x[0], 
                   circle.y if isinstance(circle.y, float) else circle.y[0], 
                   circle.radius if isinstance(circle.radius, float) else circle.radius[0]) for circle in circles])
    #print('xu',xu)
    x0c = np.append(x0c,xu)
    y0c = np.append(y0c,yu)
    rc = np.append(rc,ru)
    
    split_prob = [100 - splitprob(radius, 0.035, 0.06) for radius in rc]
    
    for circle in circles:
        circ = plt.Circle((circle.x, circle.y), circle.radius, color='red', fill=False)
        ax.add_patch(circ)
        
    plt.gca().set_aspect('equal', adjustable='box')
    plt.axis('off')
    plt.xlim(0, 1)
    plt.ylim(0, (pix_min/pix_max))
    plt.savefig(path+'/'+str(run)+'/'+str(t)+'.png',dpi=my_dpi,transparent=False) 
    plt.close()
    
    r_count = np.unique(rc)
    ide = np.append(-1, np.arange(1, b + 1))
    '''
    if len(r_count) < b:
        print('!!!!!!!!!!!!!!!!!', len(r_count))
        x0c, y0c, rc, ide, split_prob = ini_pic(b, r_i, pix, t, run, bg=False)
    '''
    return x0c, y0c, rc, ide, split_prob

def gt(x,y,r,t,run,ide,split_prob,path='.',split_id=0,savefile=True,shuffle=False):
    
    
    #plt.rcParams['axes.facecolor'] = 'black'
    my_dpi=96
    
    #print('r',r,np.mean(r))


    t_v=np.ones(len(x))*t
    
    if t==0:
        split_id=np.zeros(len(x))
    
    if shuffle:
        ind = np.argsort(x) 
        #print('ind',ind)
        random.shuffle(ind)
        #print('ind',ind)
        x=np.array(x)
        y=np.array(y)
        r=np.array(r)
        #print('x',x)
        #print('y',y)
        x= np.take_along_axis(x, ind, axis=-1)  
        #print('x2',x)
        
        y=np.take_along_axis(y, ind, axis=-1) 
        
        #print('y2',y)
        #print('ind2',ind)
        r=np.take_along_axis(r, ind, axis=-1)
        ide=np.take_along_axis(ide, ind, axis=-1)
        split_id=np.take_along_axis(split_id, ind, axis=-1)
        split_prob=np.take_along_axis(split_prob, ind, axis=-1)
    
    
    
    if savefile:
        if t==0:
            np.savetxt(path+'/'+str(run)+'_GT'+'/TRA/'+'pos_GT.txt', np.c_[x,y,r,ide,split_id,split_prob,t_v],delimiter='\t',header='x'+'\t'+'y'+'\t'+'r'+'\t'+'id'+'\t'+'split_id'+'\t'+'s_prob'+'\t'+'t')
        else:    
            xl,yl,rl,idel,split_l,s_pr_l,t_vl= np.loadtxt(path+'/'+str(run)+'_GT'+'/TRA/'+'pos_GT.txt',skiprows=1, delimiter='\t', usecols=(0,1,2,3,4,5,6), unpack=True)
            
            xl=np.append(xl,x)
            yl=np.append(yl,y)
            rl=np.append(rl,r)
            t_vl=np.append(t_vl,t_v)
            idel=np.append(idel,ide)
            split_l=np.append(split_l,split_id)
            s_pr_l=np.append(s_pr_l,split_prob)
            np.savetxt(path+'/'+str(run)+'_GT'+'/TRA/'+'pos_GT.txt', np.c_[xl,yl,rl,idel,split_l,s_pr_l,t_vl],delimiter='\t',header='x'+'\t'+'y'+'\t'+'r'+'\t'+'id'+'\t'+'split_id'+'\t'+'s_prob'+'\t'+'t')
    plt.close()

def move(x,y,r,mv_mu,mv_std,sp_prob,t,r_split,pix_max,pix_min,run,ide,path='.',reco=True,forget=True,directed=0):
    
    

    
    vec=np.linspace(0,2*np.pi,500)

    n=pix_max
    my_dpi=96
    fig = plt.figure(figsize=(pix_max/my_dpi, pix_max/my_dpi), dpi=my_dpi)
    
    plt.axis('off')
    
    plt.gca().set_aspect('equal', adjustable='box')
    ax = fig.gca()
    plt.xlim(0,1)
    plt.ylim(0,pix_min/pix_max)
    
    rec_flag=False
    xrec_l=[]
    yrec_l=[]
    rrec_l=[]
    
    for m in range(5):
        
        misdetect=np.random.random_integers(99)
        m_lim=120
    
        if misdetect >= m_lim and reco==True:
            
            coord=np.linspace(0,1,num=500)
            xrec=np.random.choice(coord)
            yrec=np.random.choice(coord)
            rrec=np.random.choice(coord)*10
            #rec=plt.Rectangle((xrec,yrec), rrec*10**(-2), rrec*10**(-2),fc=None,ec="blue") 
            #rec=plt.Circle((xrec,yrec), rrec*10**(-2), color='red',fill=False)
            #ax.add_patch(rec)
            
            xrec_l.append(xrec)
            yrec_l.append(yrec)
            rrec_l.append(rrec)
            
            print('rec!',len(xrec_l))
           
            
            rec_flag=True

    
    #xp=np.array([0,1])
    #yp=np.array([0,1])
    
    pl=True
    
    

        

    re = np.array([])
    x0c=np.array([])
    y0c=np.array([])
    r0c=np.array([])
    idec=np.array([])
    split_id=np.array([])
    split_prob=np.array([])
    flag=0

       
    for i in range(len(x)):
        phi=np.random.choice(vec)
        mv = np.random.gamma(shape=mv_mu, scale=mv_std, size=1)#sample_from_power_law(mv_mu, mv_std, size=1)#rejection_sample(mv_mu, mv_sig)
        x_n=x[i]+(mv/pix_max)*np.sin(phi)
        y_n=y[i]+(mv/pix_max)*np.cos(phi)
        
        
        r_e0=0.1 #!!!!!!!!
        #if r[i] < 0.35*r_mu:
        #    r_e0 = np.random.random_integers(8,12)
            
        re=np.append(re,r_e0)
        #print('ri0',r[i])
        r[i] = r[i]*(1+(r_e0/100))
        #print('ri1',r[i])

        if x_n < 0 or x_n > 1:
            x_n=x[i]-(mv/pix_max)*np.sin(phi)

            
        y_margin = (pix_max - pix_min) / 2

        
        if y_n < (y_margin / pix_max) or y_n > 1 - (y_margin / pix_max):
            y_n=y[i]-(mv/pix_max)*np.cos(phi)


        sp=np.random.random_integers(99)
        #print('ri',r[i])
        if percent_true(sp_prob) and t>7 and r[i]>r_split and t<10:###splitprob(r[i],0.035,0.06): ###96
            #print(r[i],'split',splitprob(r[i],0.035,0.06))
            flag=flag+1
            d=random.uniform(0.7, 0.8)
            x_n1, y_n1, x_n2, y_n2 = split_cell_axis(x_n, y_n, r[i], d)
            
            
            #x_n1=x_n-1.05*r[i]
            #x_n2=x_n+1.05*r[i]


            #r_e=np.random.random_integers(5)

            #r_n1= r[i]*(1+(r_e/100))
            #r_n2= r[i]*(1-(r_e/100))

            #print('r_n1')
            #r_e1=np.random.random_integers(15,40)
            #r_e2=np.random.random_integers(15,40)
            
            r_e1=np.random.random_integers(40,45)
            

            z=0
            while z<100:


                r_n1= r[i]*(1-(r_e1/100))
                r_n2= r[i]*(1-(r_e1/100))
                #print('r_n1',r_n1)
                #print('r_n2',r_n2)
                if r_n1 in r or r_n2 in r:
                    #print('while',r)
                    r_e1=np.random.random_integers(40,65)
                    #print('r_e',r_e)
                    #r_n1= r[i]*(1+(r_z/100))+r_n1/100
                    #r_n2= r[i]*(1-(r_z/100))+r_n2/100


                    #print()
                    z=z+1
                else:

                    break


            circ1=plt.Circle((x_n1,y_n1), r_n1, color='red',fill=False)
            circ2=plt.Circle((x_n2,y_n2), r_n2, color='red',fill=False)
            ax.add_patch(circ1)
            ax.add_patch(circ2)
            x0c=np.append(x0c,x_n1)
            y0c=np.append(y0c,y_n1)
            x0c=np.append(x0c,x_n2)
            y0c=np.append(y0c,y_n2)

            split_id=np.append(split_id,ide[i])
            split_id=np.append(split_id,ide[i])
            
            split_prob=np.append(split_prob,100-splitprob(r_n1,0.035,0.06))
            split_prob=np.append(split_prob,100-splitprob(r_n2,0.035,0.06))


            if len(idec)!=0:
                max_id=np.max(np.append(ide,idec))
            else:
                max_id=np.max(ide)


            idec=np.append(idec,max_id+1)
            idec=np.append(idec,max_id+2)

            r0c=np.append(r0c,r_n1)
            r0c=np.append(r0c,r_n2)
        else:
            #print(r[i],'nosplit',splitprob(r[i],0.035,0.06))
            split_id=np.append(split_id,0)
            circ=plt.Circle((x_n,y_n), r[i], color='red',fill=False)
            ax.add_patch(circ)
            x0c=np.append(x0c,x_n)
            y0c=np.append(y0c,y_n)
            r0c=np.append(r0c,r[i])
            split_prob=np.append(split_prob,100-splitprob(r[i],0.035,0.06))

            idec=np.append(idec,ide[i])

    circles = [Circle(x, y, r) for x, y, r in zip(x0c, y0c, r0c)]

    # Resolve overlaps
    resolve_all_overlaps(circles)

    # Update your circle lists with new values after resolving overlaps
    x0c = [circle.x for circle in circles]
    y0c = [circle.y for circle in circles]
    r0c = [circle.radius for circle in circles]

                
    
    if rec_flag:
        n_miss=len(xrec_l)
        for k in range(len(xrec_l)):
            
            xrec=xrec_l[k]
            yrec=yrec_l[k]
            rrec=rrec_l[k]
            
            x0c=np.append(x0c,xrec)
            y0c=np.append(y0c,yrec)
            r0c=np.append(r0c,rrec*10**(-2))
            #print('rec_split',rrec,100-splitprob(rrec*10**(-2),0.035,0.06))
            split_prob=np.append(split_prob,100-splitprob(rrec*10**(-2),0.035,0.06))
            split_id=np.append(split_id,-1)
            idec=np.append(idec,-2)
    
    
    n_miss=0
        
    #plt.scatter(xp,yp,s=pix**2/n**2,c='blue')
    
    if pl != False:
        plt.savefig(path+'/'+str(run)+'/'+str(t)+'.png',dpi=my_dpi,transparent=False) 
        plt.close()
        
    return x0c, y0c, r0c,idec,split_id,split_prob
        

#plt.ioff()      




   

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



def split_cell_axis(x_n, y_n, r, d):
    """
    Calculates new positions for the daughter cells of a splitting cell.

    Parameters:
    x_n (float): x position of the parent cell
    y_n (float): y position of the parent cell
    r (float): radius of the parent cell
    theta (float): orientation angle of the split in radians

    Returns:
    tuple: new x and y positions for both daughter cells
    """
    theta = random.uniform(0, 2 * np.pi)
    
    # Displacement along the axis
    delta_x = d * r * np.cos(theta)
    delta_y = d * r * np.sin(theta)

    x_n1 = x_n - delta_x
    y_n1 = y_n - delta_y

    x_n2 = x_n + delta_x
    y_n2 = y_n + delta_y

    return x_n1, y_n1, x_n2, y_n2




    
    
    
def resolve_all_overlaps(circles, max_iterations=100000, WIDTH=100, HEIGHT=100):
    for k in range(max_iterations):
        any_overlap = False
        for i, circle_a in enumerate(circles):
            for j, circle_b in enumerate(circles):
                if i != j and circle_a.intersects(circle_b):
                    any_overlap = True
                    overlap_distance = circle_a.radius + circle_b.radius - distance(circle_a, circle_b)
                    move_distance = (overlap_distance * (circle_a.radius + circle_b.radius) / 2) + 0.01

                    # Calculate angle of repulsion
                    angle = math.atan2(circle_a.y - circle_b.y, circle_a.x - circle_b.x)

                    # Add noise to the angle
                    noise = random.uniform(-0.1, 0.1)  # Adjust this value for more or less noise
                    angle += noise

                    # Calculate the forces for the two circles
                    force_a = move_distance + random.uniform(0, 0.2 * move_distance)
                    force_b = move_distance + random.uniform(0, 0.2 * move_distance)

                    # Update circle positions based on the repulsion force
                    circle_a.x += math.cos(angle) * force_a
                    circle_a.y += math.sin(angle) * force_a

                    circle_b.x -= math.cos(angle) * force_b
                    circle_b.y -= math.sin(angle) * force_b

                    # Ensure circle centers stay inside boundaries
                    circle_a.x = max(circle_a.radius, min(WIDTH - circle_a.radius, circle_a.x))
                    circle_a.y = max(circle_a.radius, min(HEIGHT - circle_a.radius, circle_a.y))

                    circle_b.x = max(circle_b.radius, min(WIDTH - circle_b.radius, circle_b.x))
                    circle_b.y = max(circle_b.radius, min(HEIGHT - circle_b.radius, circle_b.y))

        if not any_overlap:
            #print('K',k)
            break

# Test the function
class Circle:
    def __init__(self, x, y, radius):
        self.x = x
        self.y = y
        self.radius = radius

    def __repr__(self):
        return f"Circle({self.x}, {self.y}, {self.radius})"
    
    def intersects(self, other):
        """Check if this circle intersects with another circle."""
        distance = ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5
        return distance < (self.radius + other.radius)


        
def compute_radius_stats(mu_area, sigma_area):
    mu_radius = np.sqrt(mu_area / np.pi)
    sigma_radius = sigma_area / (2 * np.sqrt(np.pi * mu_area))
    return mu_radius, sigma_radius

def percent_true(p):
    return random.random() < p


def func_neg(x, a, b, c):
    return (1 - a * (b ** x)) + c

def lin_func_neg(x,m):
    return 1-m*x



def adjust_radius_using_lin_func(time_since_split, initial_radius, m):
    size_ratio = lin_func_neg(time_since_split,m)
    return initial_radius * size_ratio

def modify_file_radius(filepath, popt_neg):
    # Read the data from the file
    with open(filepath, 'r') as file:
        lines = file.readlines()

    # Remove header and store it
    header = lines.pop(0)

    # Process data lines
    data = [list(map(float, line.split())) for line in lines]

    # Find parent cells of all splits
    split_parents = {}
    for line in data:
        split_id = int(line[4])
        if split_id != 0 and split_id != -1:
            split_parents[split_id] = int(line[6])  # key = split_id, value = split time

    # Adjust radius for each parent cell
    for parent_id, split_time in split_parents.items():
        for line in data:
            cell_id = int(line[3])
            time = int(line[6])

            if cell_id == parent_id:
                time_since_split = time - split_time
                if time_since_split == -5:
                    initial_radius = line[2]
                if -5<= time_since_split < 0:
                    # Adjust the radius using the func_neg
                    line[2] = adjust_radius_using_lin_func(time_since_split, initial_radius, popt_neg)#adjust_radius_using_func(time_since_split, initial_radius, popt_neg)

    # Convert back to string format and write to file
    with open(filepath, 'w') as file:
        file.write(header)
        for line in data:
            str_line = "\t".join(map(str, line)) + '\n'
            file.write(str_line)


def convert_to_CTC(filename,image_shape,A_mu):
    
    dir_path = os.path.dirname(filename)
    
    # Load data from the file
    xl, yl, rl, idel, split_l, s_pr_l, t_vl = np.loadtxt(filename, skiprows=2, delimiter='\t', unpack=True)

    # Convert tracking format
    tracks = {}
    for x, y, r, id_, split_id, t in zip(xl, yl, rl, idel, split_l, t_vl):
        if id_ not in tracks:
            tracks[id_] = {'start': int(t), 'end': int(t), 'parent': int(split_id)}
        else:
            tracks[id_]['end'] = int(t)
    
    # Save the tracking format
    with open(os.path.join(dir_path, "man_track.txt"), "w") as out_file:
        for id_, details in tracks.items():
            out_line = f"{int(id_)} {details['start']} {details['end']} {details['parent']}\n"
            out_file.write(out_line)

    # Create a dictionary to track cells at each time point
    time_dict = {}
    for x, y, r, id_, t in zip(xl, yl, rl, idel, t_vl):
        if t not in time_dict:
            time_dict[t] = []
        time_dict[t].append((x, y, r, id_))

    width, height = image_shape

    for t, cells in time_dict.items():
        # Create a blank 16-bit image
        img = np.zeros((height, width), dtype=np.uint16)

        for x, y, r, id_ in cells:
            y_img = int(x * height)
            x_img = int((1 - y) * width)
            r= int(np.sqrt(A_mu / np.pi) / 4) if int(np.sqrt(A_mu / np.pi) / 4) > 2 else 2
            rr, cc = disk((x_img, y_img), r)
        
            # Clip the coordinates to ensure they are within the image boundaries
            rr = np.clip(rr, 0, height-1)
            cc = np.clip(cc, 0, width-1)

            img[rr, cc] = int(id_)
        # Save the image as a 16-bit TIFF
        imsave(os.path.join(dir_path, f"man_track{int(t)}.tif"), img)



def process_images(root_path):
    all_pixel_counts = []
    all_displacements = []

    # For area calculation
    for dirpath, dirnames, filenames in os.walk(root_path):
        if (os.path.basename(dirpath) == 'SEG' and 
            os.path.dirname(dirpath).endswith('_GT') and 
            os.path.basename(os.path.dirname(dirpath))[:2].isdigit()):
            
            for file in filenames:
                if file.endswith('.tif'):
                    img_path = os.path.join(dirpath, file)
                    img = np.array(Image.open(img_path))
                    unique_values, counts = np.unique(img, return_counts=True)
                    all_pixel_counts.extend(counts[unique_values != 0])

    first_image_shape = None
    
    for dirpath, dirnames, filenames in os.walk(root_path):
        if (os.path.basename(dirpath) == 'TRA' and 
            os.path.dirname(dirpath).endswith('_GT') and 
            os.path.basename(os.path.dirname(dirpath))[:2].isdigit()):

            filenames = sorted([f for f in filenames if f.startswith('man_track') and f.endswith('.tif')])
            print('filenames',filenames)
            for idx in range(len(filenames) - 1):
                img1_path = os.path.join(dirpath, filenames[idx])
                img2_path = os.path.join(dirpath, filenames[idx + 1])

                img1 = np.array(Image.open(img1_path))
                img2 = np.array(Image.open(img2_path))

                # Check image shapes
                if first_image_shape is None:
                    first_image_shape = img1.shape
                assert img1.shape == first_image_shape, f"Image {img1_path} has a different shape {img1.shape} than the first image shape {first_image_shape}"
                assert img2.shape == first_image_shape, f"Image {img2_path} has a different shape {img2.shape} than the first image shape {first_image_shape}"

                displacements = calculate_displacement(img1, img2)
                all_displacements.extend(displacements)


  
  

    # Second histogram
    hist_vals, bin_edges = np.histogram(all_displacements, bins=int(np.sqrt(len(all_displacements))))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    
 
    popt_gamma, _ = curve_fit(gamma_distribution, bin_centers[1:], hist_vals[1:], p0=[2, np.std(all_displacements), max(hist_vals)])

    mu, sigma = np.mean(all_pixel_counts), np.std(all_pixel_counts)
    print(popt_gamma)
    k,theta,_ = popt_gamma
    print('Statistics: mean_area: ',mu,' sig_area: ',sigma,' gamma:k,theta',k,theta,'mean_disp',np.mean(all_displacements) )


    return mu,sigma,k,theta,first_image_shape


def gamma_distribution(x, k, theta, A):
    """Gamma distribution function scaled by amplitude A"""
    return A * (x**(k-1) * np.exp(-x/theta)) / (theta**k * gamma(k))

def motion_module(input_path,mm_path,num_vid,sp_prob,t,n_cells):
    
    
    A_mu,A_sig,k,theta,img_shape=process_images(input_path)




 
    pix_max=img_shape[0]
    pix_min=img_shape[1]

    print('Processing images of shape ', img_shape)

    


    path0=mm_path
    combined_radii=[]
    for i in tqdm(range(1, num_vid + 1), desc="Motion Module Sampling"):
        
        r_split=np.sqrt(A_mu*0.75/np.pi)/pix_max
        b=random.randint(n_cells[0], n_cells[1])
        r_mu,r_sig= compute_radius_stats(A_mu, A_sig/10)
        r_mu=r_mu/pix_max
        r_sig=r_sig/pix_max
        run=i
        x_f, y_f, r_f,idec_f,split_prob_f=ini_pic(b,r_mu,r_sig,pix_max,pix_min,0,run,path=path0)
    
        #print(len(x), len(y), len(r), len(ide), len(split_prob))
        gt(x_f,y_f,r_f,0,run,idec_f,split_prob_f,path=path0)
        #mv=0.02
        

        x_f=x_f[idec_f>0]
        y_f=y_f[idec_f>0]
        r_f=r_f[idec_f>0]
        idec_f=idec_f[idec_f>0]

        
        split_id_f = np.zeros(len(x_f))
        # Convert to numpy array for boolean indexing

        for l in range(1, t):

            mask = split_id_f != -1
            x_f = np.array(x_f)[mask]
            y_f = np.array(y_f)[mask]
            r_f = np.array(r_f)[mask]
            idec_f = np.array(idec_f)[mask]
            split_id_f = np.array(split_id_f)[mask]


            
            x_f, y_f, r_f,idec_f,split_id_f,split_prob_f = move(x_f,y_f,r_f,k,theta,sp_prob,l,r_split,pix_max,pix_min,run,idec_f,forget=False,directed=0,path=path0)

                
            #print(l,len(x),ide)
            #x,y,r = ini_pic(b,r_i,pix,l,run)
            gt(x_f,y_f,r_f,l,run,idec_f,split_prob_f,split_id=split_id_f,path=path0,shuffle=True)
            
        filepath=path0+'/'+str(i)+'_GT'+'/TRA/'+'pos_GT.txt'
        #popt_neg=[0.55618812, 1.54775129, 0.01123954]
        popt_neg=0.05
        modify_file_radius(filepath, popt_neg)
        convert_to_CTC(filepath,(pix_max,pix_max),A_mu)
        data = np.loadtxt(filepath, delimiter='\t', skiprows=1, usecols=(2))
        
        combined_radii.extend(data)




    file_path = mm_path+'/statistics.txt'


    with open(file_path, 'w') as file:
        file.write(f"----Area (px)---\n")
        file.write(f"A_mu: {A_mu}\n")
        file.write(f"A_sigma: {A_sig}\n")
        file.write(f"----Gamma Function (Movement)----\n")
        file.write(f"k: {k}\n")
        file.write(f"theta: {theta}\n")
        file.write(f"----Image Shape----\n")
        file.write(f"Shape: {img_shape}\n")
        file.write(f"----Radii (px)----\n")
        file.write(f"r_cond: {int(np.sqrt(A_mu / np.pi) / 4) if int(np.sqrt(A_mu / np.pi) / 4) > 2 else 2}\n")
        file.write(f"r: {int(np.sqrt(A_mu / np.pi))}\n")



    return A_mu, pix_max


