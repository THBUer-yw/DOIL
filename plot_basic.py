import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from collections import deque
import numpy as np

select_method_names = ['GAIL']

def get_filename_dict(base_dir):
    file_name = []
    file_name_hopper = []
    file_name_halfcheetah = []
    file_name_walker2d = []
    file_name_ant = []
    file_name_reacher = []
    for dir in os.listdir(base_dir):
        filename_dict = {}
        if os.path.isdir(os.path.join(base_dir, dir)):
            element = dir.split("_")
            # wdail = int(element[25])
            # bcgail = int(element[15])
            # states_only = int(element[28])
            # if not wdail and not bcgail:
            #     method = "GAIL"
            # elif wdail:
            #     method = "WDAIL"
            # elif bcgail and not wdail:
            #     method = "BCGAIL"
            # method = method + "-states-only" if states_only else method
            method = "GAIL"
            env = element[0]
            seed = int(element[4])
            filename_dict.setdefault(method, []).append({"seed": seed, "env": env, "path": os.path.join(base_dir, dir, "eval_log.txt")})
            if env == "Hopper-v2":
                file_name_hopper.append(filename_dict)
            elif env == "HalfCheetah-v2":
                file_name_halfcheetah.append(filename_dict)
            elif env == "Walker2d-v2":
                file_name_walker2d.append(filename_dict)
            elif env == "Ant-v2":
                file_name_ant.append(filename_dict)
            elif env == "Reacher-v2":
                file_name_reacher.append(filename_dict)
    file_name.append(file_name_hopper)
    file_name.append(file_name_halfcheetah)
    file_name.append(file_name_walker2d)
    file_name.append(file_name_ant)
    file_name.append(file_name_reacher)
    return file_name


def plot(logdir):
    filename = get_filename_dict(logdir)
    env = None
    for i in range(len(filename)):
        file_name = filename[i]
        df = pd.DataFrame(columns=('method', 'seed', 'step', 'reward'))
        df_save = pd.DataFrame(columns=('method', 'seed', 'reward'))
        sns.set()
        fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=200)
        ax.xaxis.get_major_formatter().set_powerlimits((0, 1))
        for element in file_name:
            for method, filepath_list in element.items():
                if method in select_method_names:
                    for path_dic in filepath_list:
                        slidwin = deque(maxlen=10)
                        seed = path_dic["seed"]
                        logpath = path_dic["path"]
                        env = path_dic["env"]
                        # states_only = path_dic["states_only"]
                        plot_steps = 1e6
                        com_method = method
                        for line in open(logpath, "r"):
                            if line == "\n":
                                continue
                            line_arr = line.split(",")
                            test_step = float(line_arr[1].split(" ")[2])
                            if test_step >= plot_steps:
                                break
                            mean_reward = round(float(line_arr[0].split(":")[1]), 2)
                            slidwin.append(mean_reward)
                            plot_reward = np.mean(slidwin)
                            df = df.append([{'method': com_method, 'seed': seed, 'step': test_step, 'reward': plot_reward}], ignore_index=True)
                        # df_save = df_save.append([{'method': com_method, 'seed': seed, 'reward': plot_reward}], ignore_index=True)
                        print("File {} done.".format(logpath))
        print(df)
        os.makedirs(r"./plot_basic", exist_ok=True)
        # df_save.to_csv("./plot_basic/{}.csv".format(env + "_" + "basic"))
        palette = sns.color_palette("deep", 1)
        # palette = sns.hls_palette(4, l=.3, s=.8)

        g = sns.lineplot(x=df.step, y="reward", data=df, hue="method", style="method", dashes=False, palette=palette, hue_order=['GAIL'])
        plt.tight_layout()
        if env != "Ant-v2":
            ax.legend_.remove()
        fig.savefig("./plot_basic/{}.png".format(env + "_" + "basic"), bbox_inches='tight')
        # plt.show()


if __name__ == '__main__':
    print(os.getcwd())
    log_dir = r"./results"
    plot(log_dir)

