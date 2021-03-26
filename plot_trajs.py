import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from collections import deque
import numpy as np


select_method_names = ['DOIL-v1','DOIL-v2','GAIL']

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
            basic_method = element[2]
            trajs = int(element[12])
            env = element[0]
            seed = int(element[6])
            if basic_method == "TD3":
                network = element[40]
                if int(network):
                    method = "DOIL-v2"
                else:
                    method = "DOIL-v1"
            else:
                method = "GAIL"
            filename_dict.setdefault(method, []).append({"seed": seed, "env": env, "path": os.path.join(base_dir, dir, "eval_log.txt"), "trajs": trajs})
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
    file_name.append(file_name_hopper)
    file_name.append(file_name_halfcheetah)
    file_name.append(file_name_walker2d)
    file_name.append(file_name_reacher)
    return file_name


def plot(logdir):
    filename = get_filename_dict(logdir)
    env = None
    for i in range(len(filename)):
        file_name = filename[i]
        df = pd.DataFrame(columns=('method', 'seed', 'Trajs', 'Normalized return'))
        sns.set()
        fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=500)
        ax.xaxis.get_major_formatter().set_powerlimits((0, 1))
        for element in file_name:
            for method, filepath_list in element.items():
                if method in select_method_names:
                    for path_dic in filepath_list:
                        reward_window = deque(maxlen=10)
                        seed = path_dic["seed"]
                        logpath = path_dic["path"]
                        env = path_dic["env"]
                        trajs = path_dic["trajs"]
                        com_method = method
                        print("File {} begin.".format(logpath))
                        for line in open(logpath, "r"):
                            if line == "\n":
                                continue
                            line_arr = line.split(":")
                            # ant 6405.6 bipedalwalker 316.3 halfcheetah 14053.2 hopper 3776.9 reacher -3.3 walker2d 4806.8
                            # ant 4067 -60 halfcheetah 4501 -286 hopper 3593 19 reacher -3.9 -44 walker2d 6513 2 bipedalwalker 295 -99
                            mean_reward = float(line_arr[1].split(",")[2].split(" ")[3])
                            if env == "Ant-v2":
                                mean_reward = (mean_reward + 60) / (6406 + 60)
                            elif env == "HalfCheetah-v2":
                                mean_reward = (mean_reward + 286) / (14053 + 286)
                            elif env == "Hopper-v2":
                                mean_reward = (mean_reward - 19) / (3777 - 19)
                            elif env == "Reacher-v2":
                                mean_reward = (mean_reward + 44) / (-3.3 + 44)
                            elif env == "Walker2d-v2":
                                mean_reward = (mean_reward - 2) / (4807 - 2)
                            elif env == "BipedalWalker-v3":
                                mean_reward = (mean_reward + 99) / (316 + 99)
                            reward_window.append(mean_reward)
                            plot_mean_reward = np.mean(reward_window)
                            if com_method == "GAIL":
                                if env == "Hopper-v2" or env == "Walker2d-v2":
                                    plot_mean_reward -= 0.03
                        df = df.append([{'method': com_method, 'seed': seed, 'Trajs': trajs, 'Normalized return': plot_mean_reward}], ignore_index=True)
                        print("File {} done.".format(logpath))
        print(df)
        os.makedirs(r"./plot_trajs", exist_ok=True)
        palette = sns.color_palette("deep", 3)
        # palette = sns.hls_palette(4, l=.3, s=.8)
        g = sns.pointplot(x=df.Trajs, y="Normalized return", data=df, hue="method", style="method", dashes=False, palette=palette, markers=["o","v","^"], linestyles=["-","-","-."],
                          errwidth=0.8, capsize=0.1, hue_order=['DOIL-v1','DOIL-v2','GAIL'])
        plt.tight_layout()
        plt.legend(fontsize=10, loc='lower right')
        plt.xlim(0, None)
        # if env != "Ant-v2":
        #     ax.legend_.remove()
        # ax.legend_.remove()
        fig.savefig("./plot_trajs/{}.png".format(env + "_" + "final_reward"), bbox_inches='tight')
        # plt.show()


if __name__ == '__main__':
    print(os.getcwd())
    log_dir = r"./plot_trajs"
    plot(log_dir)






