import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from collections import deque
import numpy as np

select_method_names = ['TD3_mlp', 'TD3_dense']

def get_filename_dict(base_dir):
    file_name = []
    file_name_ant = []
    file_name_halfcheetah = []
    file_name_hopper = []
    file_name_walker2d = []
    for dir in os.listdir(base_dir):
        filename_dict = {}
        if os.path.isdir(os.path.join(base_dir, dir)):
            element = dir.split("_")
            basic_method = element[2]
            network = element[40]
            if int(network):
                method = basic_method+"_dense"
            else:
                method = basic_method+"_mlp"
            env = element[0]
            seed = int(element[6])
            filename_dict.setdefault(method, []).append(
                {"seed": seed, "env": env, "path": os.path.join(base_dir, dir, "eval_log.txt")})
            if env == "Ant-v2":
                file_name_ant.append(filename_dict)
            elif env == "HalfCheetah-v2":
                file_name_halfcheetah.append(filename_dict)
            elif env == "Hopper-v2":
                file_name_hopper.append(filename_dict)
            elif env == "Walker2d-v2":
                file_name_walker2d.append(filename_dict)
    file_name.append(file_name_ant)
    file_name.append(file_name_halfcheetah)
    file_name.append(file_name_hopper)
    file_name.append(file_name_walker2d)
    return file_name

def plot(logdir):
    filenames = get_filename_dict(logdir)
    env = None
    for i in range(len(filenames)):
        file_name_env = filenames[i]
        if not file_name_env:
            continue
        df = pd.DataFrame(columns=('method', 'seed', 'step', 'reward'))
        df_save = pd.DataFrame(columns=('method', 'seed', 'rard'))
        sns.set()
        fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=200)
        ax.xaxis.get_major_formatter().set_powerlimits((0, 1))
        for element in file_name_env:
            for method, filepath_list in element.items():
                if method in select_method_names:
                    for path_dic in filepath_list:
                        env = path_dic["env"]
                        slidwin = deque(maxlen=30)
                        seed = path_dic["seed"]
                        logpath = path_dic["path"]
                        com_method = method
                        for line in open(logpath, "r"):
                            if line == "\n":
                                continue
                            line_arr = line.split(":")
                            test_step = float(line_arr[1].split(",")[0])
                            mean_reward = float(line_arr[1].split(",")[2].split(" ")[3])
                            slidwin.append(mean_reward)
                            plot_reward = np.mean(slidwin)
                            df = df.append([{'method': com_method, 'seed': seed, 'step': test_step, 'reward': plot_reward}], ignore_index=True, sort=True)
                        df_save = df_save.append([{'method': com_method, 'seed': seed, 'reward': plot_reward}], ignore_index=True, sort=True)
                        print("File {} done.".format(logpath))

        os.makedirs(r"./plot_methods", exist_ok=True)
        palette = sns.color_palette("deep", 2)
        # df_save.to_csv("./plot_reward/{}.csv".format(env + "_" + "used_method" + "_" + basic_method))
        # palette = sns.hls_palette(4, l=.3, s=.8)
        g = sns.lineplot(x=df.step, y="reward", data=df, hue="method", style="method", dashes=False, palette=palette,
            hue_order=['TD3_mlp', 'TD3_dense'])
        plt.tight_layout()
        if env != "Ant-v2":
            ax.legend_.remove()
        fig.savefig("./plot_methods/{}.png".format(env), bbox_inches='tight')



if __name__ == '__main__':
    print(os.getcwd())
    log_dir = r"./results"
    plot(log_dir)

