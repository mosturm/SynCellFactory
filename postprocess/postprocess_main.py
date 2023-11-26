from postprocess.postprocess_help import process_directory,crop_directory, move_directory, get_pp_params

def run_postproc(name,cellpose_path,cellpose_model):

    base_directory_path, target_width_value, r = get_pp_params(name)
    # Example usage:
    output_directory='./Outputs/'



    
    move_directory(base_directory_path,output_directory)
    process_directory(output_directory+name, target_width_value, cellpose_path, cellpose_model,r)
    crop_directory(output_directory+name)