import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from collections import deque
import numpy as np

select_method_names = ['n=2','n=4','n=8','n=16']

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
            num_hidden_layers = element[15]
            method = "n="+str(int(num_hidden_layers)+2)
            filename_dict.setdefault(method, []).append({"seed": seed, "env": env, "path": os.path.join(base_dir, dir, "eval_log.txt")})
            if env == "Ant-v2":
                file_name_hopper.append(filename_dict)
            elif env == "BipedalWalker-v3":
                file_name_bipedalwalker.append(filename_dict)
            elif env == "HalfCheetah-v2":
                file_name_halfcheetah.append(filename_dict)
            elif env == "Hopper-v2":
                file_name_walker2d.append(filename_dict)
            elif env == "Reacher-v2":
                file_name_ant.append(filename_dict)
            elif env == "Walker2d-v2":
                file_name_reacher.append(filename_dict)
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
        df = pd.DataFrame(columns=('num_layers', 'seed', 'Steps', 'Return'))
        sns.set()
        fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=500)
        ax.xaxis.get_major_formatter().set_powerlimits((0, 1))
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
                            test_step = float(line_arr[1].split(",")[0])
                            mean_reward = float(line_arr[1].split(",")[2].split(" ")[3])
                            slidwin.append(mean_reward)
                            plot_reward = np.mean(slidwin)
                            df = df.append([{'num_layers': com_method, 'seed': seed, 'Steps': test_step, 'Return': plot_reward}], ignore_index=True)
                        print("File {} done.".format(logpath))
        print(df)
        os.makedirs(r"./plot_ablation", exist_ok=True)
        palette = sns.color_palette("deep", 4)
        g = sns.lineplot(x=df.Steps, y="Return", data=df, hue="num_layers", style="num_layers", dashes=False, palette=palette, hue_order=['n=2','n=4','n=8','n=16'])
        plt.tight_layout()
        plt.legend(fontsize=10, loc='lower right')
        # if env != "Ant-v2":
        #     ax.legend_.remove()
        fig.savefig("./plot_ablation/{}.png".format(env + "_" + "ablation"), bbox_inches='tight')
        # plt.show()


if __name__ == '__main__':
    print(os.getcwd())
    log_dir = r"./plot_ablation"
    plot(log_dir)