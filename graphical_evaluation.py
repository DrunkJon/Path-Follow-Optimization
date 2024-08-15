import os
from turtle import color
import numpy as np
import pandas
from pandas import DataFrame
import matplotlib.pyplot as plt
from matplotlib import colormaps
import matplotlib.patches as mpatches


MAX_INDEX = 567
DT = 0.1

def read_data(experiment_path, depth=1):
    if not os.path.exists(experiment_path):
        raise Exception(f"{experiment_path} does not exist")
    data_dict = {}
    i = 0
    for dir_entry in os.scandir(experiment_path):
        if dir_entry.is_dir():
            data_dict = read_dir(dir_entry.path, data_dict, depth, i)
            i += 1
    return data_dict


def read_dir(folder_path, data_dict, depth, i):
    for dir_entry in os.scandir(folder_path):
        if depth <= 1 and dir_entry.is_file() and dir_entry.name.endswith(".h5"):
            key = dir_entry.name[:-3]
            if key not in data_dict.keys():
                data_dict[key] = init_df()
            append_to_total_df(data_dict[key], f'{folder_path}/{dir_entry.name}', MAX_INDEX, i)
        if depth > 1 and dir_entry.is_dir():
            key = dir_entry.name
            if key not in data_dict.keys():
                data_dict[key] = {}
            data_dict[key] = read_dir(f"{folder_path}/{dir_entry.name}", data_dict[key], depth-1, i)
    return data_dict


def fill_df(df, length, di):
    index = df.index[-1]
    last_loc = df.iloc[-1]
    while len(df) < length:
        index += di
        df.loc[index] = last_loc
    return df

def build_loc(goal_x_dists, goal_y_dists, obst_dists):
    return {
        "goal_x_dists": goal_x_dists,
        "goal_y_dists": goal_y_dists,
        "obst_dists": obst_dists
    }

def init_df():
    return DataFrame(columns=["goal_x_dists", "goal_y_dists", "obst_dists"])

def append_to_total_df(total_df, file_name, max_index, i):
    df = pandas.read_hdf(file_name)
    goal_x_dists = fill_df(abs(df["goal_x"] - df["robo_x"]), max_index, DT)
    goal_y_dists = fill_df(abs(df["goal_y"] - df["robo_y"]), max_index, DT)
    obst_dists = fill_df(df["obst_dist"], max_index, DT)
    total_df.loc[i] = build_loc(goal_x_dists, goal_y_dists, obst_dists)

def get_eval_arrays(total_df, max_index):
    avg_goal_x = list([np.average([iloci["goal_x_dists"].iloc[j] for iloci in total_df.iloc])for j in range(max_index)])
    avg_goal_y = list([np.average([iloci["goal_y_dists"].iloc[j] for iloci in total_df.iloc])for j in range(max_index)])
    avg_obst = list([np.average([iloci["obst_dists"].iloc[j] for iloci in total_df.iloc])for j in range(max_index)])
    min_obst = list([np.min([iloci["obst_dists"].iloc[j] for iloci in total_df.iloc])for j in range(max_index)])
    return avg_goal_x, avg_goal_y, avg_obst, min_obst

def plot_eval_row(axs, total_df, color=None):
    lw = 0.85

    avg_goal_x, avg_goal_y, avg_obst, min_obst = get_eval_arrays(total_df, MAX_INDEX)
    axs[0].plot(avg_goal_x, color=color, lw=lw)
    axs[0].set_ylim(top = max(axs[0].get_ylim()[1], max(avg_goal_x) * 1.1))

    axs[1].plot(avg_goal_y, color=color, lw=lw)
    axs[1].set_ylim(top = max(axs[1].get_ylim()[1], max(avg_goal_y) * 1.1))

    xs = list(range(MAX_INDEX))
    axs[2].plot(avg_obst, color=color, lw=lw)
    axs[3].plot(min_obst, color=color, lw=lw) 
    axs[2].set_ylim(top = max(axs[2].get_ylim()[1], max(avg_obst) * 1.1))
    axs[3].set_ylim(top = max(axs[3].get_ylim()[1], max(min_obst) * 1.1))
    axs[2].set_ylim(bottom = min(axs[2].get_ylim()[1], min(avg_obst) * 0.91))
    axs[3].set_ylim(bottom = min(axs[3].get_ylim()[1], min(min_obst) * 0.91))

    ylim2 = axs[2].get_ylim()
    ylim3 = axs[3].get_ylim()

    axs[2].fill_between(xs, [3]*len(avg_obst), 0.0, linestyle="-", color="cornsilk")
    axs[2].fill_between(xs, [1]*len(avg_obst), 0.0, linestyle="-", color="mistyrose")

    axs[3].fill_between(xs, [3]*len(min_obst), 0.0, linestyle="-", color="cornsilk")
    axs[3].fill_between(xs, [1]*len(min_obst), 0.0, linestyle="-", color="mistyrose")
    
    axs[2].set_ylim(ylim2)
    axs[3].set_ylim(ylim3)

    for ax in axs:
        # ax.set_ylim(bottom=0.0)
        ax.set_xlim(0,MAX_INDEX)
        ax.grid()

PROCEDURAL_COLOR_DICT = {}
xkcd_list = [
    "xkcd:purple", "xkcd:magenta", "xkcd:pink",
    "xkcd:dark blue", "xkcd:blue", "xkcd:sky blue",
    "xkcd:forest green", "xkcd:green", "xkcd:lime green"
]

def get_tab_color(i):
    step = 1 / 12
    return colormaps["Tab10"]((i + 0.5) * step)

def get_named_color(name:str, color_dict=PROCEDURAL_COLOR_DICT):
    if name not in color_dict:
        color_dict[name] = xkcd_list[len(color_dict)]
    return color_dict[name]


def plot_data_dict(data_dict, depth=1, fig_name=None, color_dict = PROCEDURAL_COLOR_DICT):
    if depth >= 3:
        raise Exception(f"plotting of depth {depth} is not implemented")
    rows = 1 if depth <= 1 else len(data_dict)
    fig, ax_rows = plt.subplots(rows, 4)
    if depth == 1:
        # note: ax_rows is just one row without nested list in this case
        ax_rows[1].axvline(x=MAX_INDEX-200, color="gray", ls="--", lw=0.7)
        ax_rows[0].axvline(x=MAX_INDEX-200, color="gray", ls="--", lw=0.7)
        plot_data_dict_row(data_dict, ax_rows, color_dict)
        add_legend(ax_rows[3], color_dict)
        add_labels(ax_rows)
    if depth == 2:
        for key, ax_row in zip(data_dict.keys(), ax_rows):
            ax_row[1].axvline(x=MAX_INDEX-200, color="gray", ls="--", lw=0.7)
            ax_row[0].axvline(x=MAX_INDEX-200, color="gray", ls="--", lw=0.7)
            ax_row[0].set_ylabel(key)
            plot_data_dict_row(data_dict[key], ax_row, color_dict)
        add_legend(ax_rows[0][3], color_dict)
        add_labels(ax_rows[0])

    if not fig_name is None:
        plt.savefig(f"./plot/{fig_name}.png", bbox_inches="tight", dpi=300)

    return fig, ax_rows

def plot_data_dict_row(data_dict, ax_row, color_dict=PROCEDURAL_COLOR_DICT):
    for i, (key, df) in enumerate(data_dict.items()):
        plot_eval_row(ax_row, df, color = get_named_color(key, color_dict))

def add_legend(top_right_ax, color_dict=PROCEDURAL_COLOR_DICT, label = "(o, g, s)"):
    patches = list([
        mpatches.Patch(color=color, label=key)
        for key, color in color_dict.items()
        ])
    patches = [mpatches.Patch(color="white", label=label)] + patches
    top_right_ax.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.)

def add_labels(top_row):
    top_row[0].set_title("avg x goal dist")
    top_row[1].set_title("avg y goal dist")
    top_row[2].set_title("avg obstacle dist")
    top_row[3].set_title("min obstacle dist")


if __name__ == '__main__':
    tri_color_dict = {
        "DWA_data": 'tab:green',
        "long_MultiPSO_data": 'tab:blue',
        "short_MultiPSO_data": 'tab:cyan'
    }
    tab_color_dict = {}
    xkcd_color_dict = {
    }
    dir_name = "param Experiment #7"
    depth = 1
    MAX_INDEX = 430 # 567 / 430
    DT = 0.16       # 0.1 / 0.16

    data_dict = read_data(f"./data/{dir_name}", depth)
    plot_data_dict(data_dict, depth, fig_name=dir_name, color_dict=tab_color_dict)
    plt.show()