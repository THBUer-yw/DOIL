import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from collections import deque
import numpy as np

select_method_names = ['reward-type-1', 'reward-type-2', 'reward-type-3', "reward-type-4", "reward-type-5", "reward-type-6", "reward-type-7", "reward-type-8"]

def get_filename_dict(base_dir):
    file_name = []
    file_name_ant = []
    ant_1 = []
    ant_2 = []
    file_name_halfcheetah = []
    halfcheetah_1 = []
    halfcheetah_2 = []
    file_name_hopper = []
    hopper_1 = []
    hopper_2 = []
    file_name_reacher = []
    reacher_1 = []
    reacher_2 = []
    file_name_walker2d = []
    walker2d_1 = []
    walker2d_2 = []
    file_name_bipedalwalker = []
    bipedalwalker_1 = []
    bipedalwalker_2 = []
    for dir in os.listdir(base_dir):
        filename_dict = {}
        if os.path.isdir(os.path.join(base_dir, dir)):
            element = dir.split("_")
            wdail = int(element[31])
            if not wdail:
                basic_method = "GAIL"
            else:
                basic_method = "WDAIL"
            env = element[0]
            seed = int(element[6])
            reward_type = element[37]
            method = "reward-type-"+reward_type
            filename_dict.setdefault(method, []).append(
                {"seed": seed, "env": env, "path": os.path.join(base_dir, dir, "eval_log.txt"),
                 "basic_method": basic_method})
            if env == "Ant-v2":
                if basic_method == "GAIL":
                    ant_1.append(filename_dict)
                elif basic_method == "WDAIL":
                    ant_2.append(filename_dict)
            elif env == "HalfCheetah-v2":
                if basic_method == "GAIL":
                    halfcheetah_1.append(filename_dict)
                elif basic_method == "WDAIL":
                    halfcheetah_2.append(filename_dict)
            elif env == "Hopper-v2":
                if basic_method == "GAIL":
                    hopper_1.append(filename_dict)
                elif basic_method == "WDAIL":
                    hopper_2.append(filename_dict)
            elif env == "Reacher-v2":
                if basic_method == "GAIL":
                    reacher_1.append(filename_dict)
                elif basic_method == "WDAIL":
                    reacher_2.append(filename_dict)
            elif env == "Walker2d-v2":
                if basic_method == "GAIL":
                    walker2d_1.append(filename_dict)
                elif basic_method == "WDAIL":
                    walker2d_2.append(filename_dict)
            elif env == "BipedalWalker-v3":
                if basic_method == "GAIL":
                    bipedalwalker_1.append(filename_dict)
                elif basic_method == "WDAIL":
                    bipedalwalker_2.append(filename_dict)
    file_name_ant.append(ant_1)
    file_name_ant.append(ant_2)
    file_name_halfcheetah.append(halfcheetah_1)
    file_name_halfcheetah.append(halfcheetah_2)
    file_name_hopper.append(hopper_1)
    file_name_hopper.append(hopper_2)
    file_name_reacher.append(reacher_1)
    file_name_reacher.append(reacher_2)
    file_name_walker2d.append(walker2d_1)
    file_name_walker2d.append(walker2d_2)
    file_name_bipedalwalker.append(bipedalwalker_1)
    file_name_bipedalwalker.append(bipedalwalker_2)
    file_name.append(file_name_ant)
    file_name.append(file_name_halfcheetah)
    file_name.append(file_name_hopper)
    file_name.append(file_name_reacher)
    file_name.append(file_name_walker2d)
    file_name.append(file_name_bipedalwalker)
    return file_name

def plot(logdir):
    filenames = get_filename_dict(logdir)
    env = None
    for i in range(len(filenames)):
        file_name_env = filenames[i]
        if not file_name_env:
            continue
        for method_files in file_name_env:
            if not method_files:
                continue
            df = pd.DataFrame(columns=('method', 'seed', 'step', 'reward'))
            df_save = pd.DataFrame(columns=('method', 'seed', 'rard'))
            sns.set()
            fig, ax = plt.subplots(1, 1, figsize=(10, 6), dpi=200)
            ax.xaxis.get_major_formatter().set_powerlimits((0, 1))
            for element in method_files:
                for method, filepath_list in element.items():
                    if method in select_method_names:
                        for path_dic in filepath_list:
                            env = path_dic["env"]
                            slidwin = deque(maxlen=30)
                            seed = path_dic["seed"]
                            logpath = path_dic["path"]
                            basic_method = path_dic["basic_method"]
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

            os.makedirs(r"./plot_reward", exist_ok=True)
            palette = sns.color_palette("deep", 8)
            # df_save.to_csv("./plot_reward/{}.csv".format(env + "_" + "used_method" + "_" + basic_method))
            # palette = sns.hls_palette(4, l=.3, s=.8)
            g = sns.lineplot(x=df.step, y="reward", data=df, hue="method", style="method", dashes=False, palette=palette,
                hue_order=['reward-type-1', 'reward-type-2', 'reward-type-3', "reward-type-4", "reward-type-5", "reward-type-6", "reward-type-7", "reward-type-8"])
            plt.tight_layout()
            if env != "Ant-v2":
                ax.legend_.remove()
            fig.savefig("./plot_reward/{}.png".format(env + "_" + "used_method" + "_" + basic_method), bbox_inches='tight')



if __name__ == '__main__':
    print(os.getcwd())
    log_dir = r"./results_reward"
    plot(log_dir)

