import matplotlib.pyplot as plt


def dynamic_subplot(data_to_use, categories_list, figsize=(16.18, 10), figure_title=None, xlabel_title=None, ylabel_title=None):
    length_data = len(data_to_use)
    length_categories = len(categories_list)
    nrows = -int(-(length_data ** 1/2) // 1)
    ncols = length_categories

    fig, axs = plt.subplots(nrows, ncols, figsize)
    fig.suptitle(f"{figure_title}")

    for i, item in enumerate(categories_list):
        ax = axs[i // nrows, i % ncols]
        ax.plot(data_to_use[i], marker="o")
        ax.set_title(item)
        ax.set_xlabel(f"{xlabel_title}")
        ax.set_ylabel(f"{ylabel_title}")
    
    plt.tight_layout()
    plt.show()
