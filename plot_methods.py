import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from collections import deque
import numpy as np
from matplotlib.ticker import FuncFormatter

select_method_names = ['DOIL-v1','DOIL-v2','GAIL','BC','Random',"Expert"]

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
            if basic_method == "TD3":
                network = element[40]
                if int(network):
                    method = "DOIL-v2"
                else:
                    method = "DOIL-v1"
            else:
                method = "GAIL"
            env = element[0]
            seed = int(element[6])
            filename_dict.setdefault(method, []).append(
                {"seed": seed, "env": env, "path": os.path.join(base_dir, dir, "eval_log.txt"), "basic_method":basic_method})
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
    filenames = get_filename_dict(logdir)
    env = None
    bc_ant = 0
    bc_bipedalwalker = 0
    bc_halfcheetah = 0
    bc_hopper = 0
    bc_reacher = 0
    bc_walker2d = 0
    for i in range(len(filenames)):
        file_name_env = filenames[i]
        if not file_name_env:
            continue
        df = pd.DataFrame(columns=('method', 'seed', 'Steps', 'Return'))
        sns.set()
        # 5 3 ; 6 4
        fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=500)
        ax.xaxis.get_major_formatter().set_powerlimits((0, 1))
        # ax.set_xlabel(xlabel="Steps",fontsize=6)
        # ax.set_ylabel(ylabel="Return",fontsize=6)
        # plt.xticks(fontsize=6)
        # plt.yticks(fontsize=6)
        for element in file_name_env:
            for method, filepath_list in element.items():
                if method in select_method_names:
                    for path_dic in filepath_list:
                        env = path_dic["env"]
                        if env == "Ant-v2":
                            bc_ant += 1
                        if env == "BipedalWalker-v3":
                            bc_bipedalwalker += 1
                        if env == "HalfCheetah-v2":
                            bc_halfcheetah += 1
                        if env == "Hopper-v2":
                            bc_hopper += 1
                        if env == "Reacher-v2":
                            bc_reacher += 1
                        if env == "Walker2d-v2":
                            bc_walker2d += 1
                        slidwin = deque(maxlen=10)
                        seed = path_dic["seed"]
                        logpath = path_dic["path"]
                        basic_method = path_dic["basic_method"]
                        com_method = method
                        for line in open(logpath, "r"):
                            if line == "\n":
                                continue
                            line_arr = line.split(":")
                            test_steps = float(line_arr[1].split(",")[0])
                            if test_steps >= 1e6:
                                break
                            mean_reward = float(line_arr[1].split(",")[2].split(" ")[3])
                            if env == "Walker2d-v2":
                                mean_reward -= 200
                            # ant 6405.6 221.6  bipedalwalker 316.3 halfcheetah 14053.2 100.9 hopper 3776.9 26.4 reacher -3.3 walker2d 4806.8 12.4
                            slidwin.append(mean_reward)
                            plot_reward = np.mean(slidwin)
                            df = df.append([{'method': com_method, 'seed': seed, 'Steps': test_steps, 'Return': plot_reward}], ignore_index=True, sort=True)
                            if env == "Ant-v2":
                                if bc_ant == 0:
                                    df = df.append([{'method': "BC", 'seed': bc_ant, 'Steps': test_steps,'Return': 1861}], ignore_index=True, sort=True)
                                if bc_ant == 1:
                                    df = df.append([{'method': "BC", 'seed': bc_ant, 'Steps': test_steps,'Return': 1326}], ignore_index=True, sort=True)
                                if bc_ant == 2:
                                    df = df.append([{'method': "BC", 'seed': bc_ant, 'Steps': test_steps,'Return': 1667}], ignore_index=True, sort=True)
                                if bc_ant == 3:
                                    df = df.append([{'method': "BC", 'seed': bc_ant, 'Steps': test_steps,'Return': 3497}], ignore_index=True, sort=True)
                                if bc_ant == 4:
                                    df = df.append([{'method': "BC", 'seed': bc_ant, 'Steps': test_steps,'Return': 3705}], ignore_index=True, sort=True)
                                df = df.append([{'method': "Random", 'seed': bc_ant, 'Steps': test_steps, 'Return': -60}], ignore_index=True, sort=True)
                                df = df.append([{'method': "Expert", 'seed': bc_ant, 'Steps': test_steps, 'Return': 6406}], ignore_index=True, sort=True)
                            if env == "BipedalWalker-v3":
                                if bc_bipedalwalker == 0:
                                    df = df.append([{'method': "BC", 'seed': bc_ant, 'Steps': test_steps,'Return': -107}], ignore_index=True, sort=True)
                                if bc_bipedalwalker == 1:
                                    df = df.append([{'method': "BC", 'seed': bc_ant, 'Steps': test_steps,'Return': -115}], ignore_index=True, sort=True)
                                if bc_bipedalwalker == 2:
                                    df = df.append([{'method': "BC", 'seed': bc_ant, 'Steps': test_steps,'Return': -107}], ignore_index=True, sort=True)
                                if bc_bipedalwalker == 3:
                                    df = df.append([{'method': "BC", 'seed': bc_ant, 'Steps': test_steps,'Return': -97}], ignore_index=True, sort=True)
                                if bc_bipedalwalker == 4:
                                    df = df.append([{'method': "BC", 'seed': bc_ant, 'Steps': test_steps,'Return': 256}], ignore_index=True, sort=True)
                                df = df.append([{'method': "Random", 'seed': bc_ant, 'Steps': test_steps, 'Return': -99}], ignore_index=True, sort=True)
                                df = df.append([{'method': "Expert", 'seed': bc_ant, 'Steps': test_steps, 'Return': 316}], ignore_index=True, sort=True)
                            if env == "HalfCheetah-v2":
                                if bc_halfcheetah == 0:
                                    df = df.append([{'method': "BC", 'seed': bc_halfcheetah, 'Steps': test_steps,'Return': -1383}], ignore_index=True, sort=True)
                                if bc_halfcheetah == 1:
                                    df = df.append([{'method': "BC", 'seed': bc_halfcheetah, 'Steps': test_steps,'Return': -461}], ignore_index=True, sort=True)
                                if bc_halfcheetah == 2:
                                    df = df.append([{'method': "BC", 'seed': bc_halfcheetah, 'Steps': test_steps,'Return': -178}], ignore_index=True, sort=True)
                                if bc_halfcheetah == 3:
                                    df = df.append([{'method': "BC", 'seed': bc_halfcheetah, 'Steps': test_steps,'Return': -532}], ignore_index=True, sort=True)
                                if bc_halfcheetah == 4:
                                    df = df.append([{'method': "BC", 'seed': bc_halfcheetah, 'Steps': test_steps,'Return': -456}], ignore_index=True, sort=True)
                                df = df.append([{'method': "Random", 'seed': bc_halfcheetah, 'Steps': test_steps, 'Return': -286}], ignore_index=True, sort=True)
                                df = df.append([{'method': "Expert", 'seed': bc_halfcheetah, 'Steps': test_steps, 'Return': 14053}], ignore_index=True, sort=True)
                            if env == "Hopper-v2":
                                if bc_hopper == 0:
                                    df = df.append([{'method': "BC", 'seed': bc_hopper, 'Steps': test_steps,'Return': 66}], ignore_index=True, sort=True)
                                if bc_hopper == 1:
                                    df = df.append([{'method': "BC", 'seed': bc_hopper, 'Steps': test_steps,'Return': 634}], ignore_index=True, sort=True)
                                if bc_hopper == 2:
                                    df = df.append([{'method': "BC", 'seed': bc_hopper, 'Steps': test_steps,'Return': 322}], ignore_index=True, sort=True)
                                if bc_hopper == 3:
                                    df = df.append([{'method': "BC", 'seed': bc_hopper, 'Steps': test_steps,'Return': 266}], ignore_index=True, sort=True)
                                if bc_hopper == 4:
                                    df = df.append([{'method': "BC", 'seed': bc_hopper, 'Steps': test_steps,'Return': 487}], ignore_index=True, sort=True)
                                df = df.append([{'method': "Random", 'seed': bc_hopper, 'Steps': test_steps, 'Return': 19}], ignore_index=True, sort=True)
                                df = df.append([{'method': "Expert", 'seed': bc_hopper, 'Steps': test_steps, 'Return': 3777}], ignore_index=True, sort=True)
                            if env == "Reacher-v2":
                                if bc_reacher == 0:
                                    df = df.append([{'method': "BC", 'seed': bc_ant, 'Steps': test_steps,'Return': -10.7}], ignore_index=True, sort=True)
                                if bc_reacher == 1:
                                    df = df.append([{'method': "BC", 'seed': bc_ant, 'Steps': test_steps,'Return': -10.2}], ignore_index=True, sort=True)
                                if bc_reacher == 2:
                                    df = df.append([{'method': "BC", 'seed': bc_ant, 'Steps': test_steps,'Return': -6.5}], ignore_index=True, sort=True)
                                if bc_reacher == 3:
                                    df = df.append([{'method': "BC", 'seed': bc_ant, 'Steps': test_steps,'Return': -9.6}], ignore_index=True, sort=True)
                                if bc_reacher == 4:
                                    df = df.append([{'method': "BC", 'seed': bc_ant, 'Steps': test_steps,'Return': -11.5}], ignore_index=True, sort=True)
                                df = df.append([{'method': "Random", 'seed': bc_ant, 'Steps': test_steps, 'Return': -44}], ignore_index=True, sort=True)
                                df = df.append([{'method': "Expert", 'seed': bc_ant, 'Steps': test_steps, 'Return': -3.3}], ignore_index=True, sort=True)
                            if env == "Walker2d-v2":
                                if bc_walker2d == 0:
                                    df = df.append([{'method': "BC", 'seed': bc_walker2d, 'Steps': test_steps,'Return': 0}], ignore_index=True, sort=True)
                                if bc_walker2d == 1:
                                    df = df.append([{'method': "BC", 'seed': bc_walker2d, 'Steps': test_steps,'Return': 67}], ignore_index=True, sort=True)
                                if bc_walker2d == 2:
                                    df = df.append([{'method': "BC", 'seed': bc_walker2d, 'Steps': test_steps,'Return': 0}], ignore_index=True, sort=True)
                                if bc_walker2d == 3:
                                    df = df.append([{'method': "BC", 'seed': bc_walker2d, 'Steps': test_steps,'Return': 234}], ignore_index=True, sort=True)
                                if bc_walker2d == 4:
                                    df = df.append([{'method': "BC", 'seed': bc_walker2d, 'Steps': test_steps,'Return': 0}], ignore_index=True, sort=True)
                                df = df.append([{'method': "Random", 'seed': bc_walker2d, 'Steps': test_steps, 'Return': 2}], ignore_index=True, sort=True)
                                df = df.append([{'method': "Expert", 'seed': bc_walker2d, 'Steps': test_steps, 'Return': 4807}], ignore_index=True, sort=True)
                        print("File {} done.".format(logpath))

        os.makedirs(r"./plot_methods", exist_ok=True)
        palette = sns.color_palette("deep", 6)
        # palette = sns.hls_palette(4, l=.3, s=.8)
        g = sns.lineplot(x=df.Steps, y="Return", data=df, hue="method", sizes=3, style="method", dashes={'DOIL-v1':(2,0),'DOIL-v2':(2,0),'GAIL':(2,0),'BC':(2,0),'Random':(3,1),"Expert":(3,1)}, palette=palette,
            hue_order=['DOIL-v1','DOIL-v2','GAIL','BC','Random',"Expert"])
        plt.tight_layout()
        plt.legend(fontsize=10,loc='lower right')
        # plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        # box = ax.get_position()
        # ax.set_position([box.x0, box.y0, box.width, box.height])
        # ax.legend(loc='center', bbox_to_anchor=(0.5, 1.2), ncol=7)
        # if env != "Ant-v2":
        #     ax.legend_.remove()
        # ax.legend_.remove()
        fig.savefig("./plot_methods/{}.png".format(env), bbox_inches='tight')



if __name__ == '__main__':
    print(os.getcwd())
    log_dir = r"./plot_methods"
    plot(log_dir)

