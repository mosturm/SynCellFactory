
from create_train_data.gen_train_cond import generate_CTC_datasets
from create_train_data.auto_train import auto_train_main

def train_run(path_stats,path_orig,name,save_dir,d,n_cells,cuda_index):
    generate_CTC_datasets(path_stats,path_orig,name,save_dir,d)
    auto_train_main(name,n_cells,cuda_index)
