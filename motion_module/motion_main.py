from motion_module.motion_help import motion_module
from motion_module.transcond import transition2condition



def motion_run(input_path,mm_path,num_vid,sp_prob,t,n_cells,d):
    A_mu,pix_max=motion_module(input_path,mm_path,num_vid,sp_prob,t,n_cells)
    transition2condition(mm_path, A_mu,pix_max,d)
