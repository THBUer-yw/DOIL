import os

import pandas as pd
from collections import deque
import numpy as np

select_method_names = ['DOIL-v1','DOIL-v2','GAIL','DOIL-state']

def get_filename_dict(base_dir):
    file_name = []
    file_name_ant = []
    file_name_bipedalwalker = []
    file_name_halfcheetah = []
    file_name_hopper = []
    file_name_reacher = []
    file_name_walker2d = []
    for dir in os.listdir(base_dir):
        filename_dict = {}
        if os.path.isdir(os.path.join(base_dir, dir)):
            element = dir.split("_")
            env = element[0]
            seed = int(element[6])
            state_only = int(element[34])
            basic_method = element[2]
            if basic_method == "TD3":
                network = element[40]
                if int(network) and not state_only:
                    method = "DOIL-v2"
                elif state_only:
                    method = "DOIL-state"
                else:
                    method = "DOIL-v1"
            else:
                method = "GAIL"
            filename_dict.setdefault(method, []).append({"seed": seed, "env": env, "path": os.path.join(base_dir, dir, "eval_log.txt")})
            if env == "Ant-v2":
                file_name_ant.append(filename_dict)
            elif env == "BipedalWalker-v3":
                file_name_bipedalwalker.append(filename_dict)
            elif env == "HalfCheetah-v2":
                file_name_halfcheetah.append(filename_dict)
            elif env == "Hopper-v2":
                file_name_hopper.append(filename_dict)
            elif env == "Reacher-v2":
                file_name_reacher.append(filename_dict)
            elif env == "Walker2d-v2":
                file_name_walker2d.append(filename_dict)
    file_name.append(file_name_ant)
    file_name.append(file_name_bipedalwalker)
    file_name.append(file_name_halfcheetah)
    file_name.append(file_name_hopper)
    file_name.append(file_name_reacher)
    file_name.append(file_name_walker2d)
    return file_name


def plot(logdir):
    filename = get_filename_dict(logdir)
    env = None
    for i in range(len(filename)):
        file_name = filename[i]
        df_save = pd.DataFrame(columns=('method', 'seed', 'Return'))
        for element in file_name:
            for method, filepath_list in element.items():
                if method in select_method_names:
                    for path_dic in filepath_list:
                        slidwin = deque(maxlen=10)
                        seed = path_dic["seed"]
                        env = path_dic["env"]
                        logpath = path_dic["path"]
                        com_method = method
                        for line in open(logpath, "r"):
                            line_arr = line.split(":")
                            mean_reward = float(line_arr[1].split(",")[2].split(" ")[3])
                            slidwin.append(mean_reward)
                            plot_reward = np.mean(slidwin)
                        df_save = df_save.append([{'method': com_method, 'seed': seed, 'Return': plot_reward}],ignore_index=True, sort=True)
                        print("File {} done.".format(logpath))
        os.makedirs(r"./save_results", exist_ok=True)
        df_save.to_csv("./save_results/{}.csv".format(env))



if __name__ == '__main__':
    print(os.getcwd())
    log_dir = r"./save_results"
    plot(log_dir)

