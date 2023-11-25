
from create_train_data.gen_train_cond import generate_CTC_datasets

def train_run(path_stats,path_orig,name,save_dir,d):
    generate_CTC_datasets(path_stats,path_orig,name,save_dir,d)

