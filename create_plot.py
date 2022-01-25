import matplotlib.pyplot as plt
import numpy as np

def create_plot(init_train_acc, init_dev_acc, train_acc, dev_acc, n_data, lang, threshold, epoch, eval, symmetric=False, nfa=False):

    plt.style.use('ggplot')

    init_train_array = np.array(list(init_train_acc.values()))
    init_mean_train_acc = np.average(init_train_array, axis=0)
    init_std_train_acc = np.std(init_train_array, axis=0)
    init_dev_array = np.array(list(init_dev_acc.values()))
    init_mean_dev_acc = np.average(init_dev_array, axis=0)
    init_std_dev_acc = np.std(init_dev_array, axis=0)

    train_array = np.array(list(train_acc.values()))
    mean_train_acc = np.average(train_array, axis=0)
    std_train_acc = np.std(train_array, axis=0)
    dev_array = np.array(list(dev_acc.values()))
    mean_dev_acc = np.average(dev_array, axis=0)
    std_dev_acc = np.std(dev_array, axis=0)

    fig, ax = plt.subplots()

    ax.plot(n_data, mean_train_acc, linestyle='-', color='green', label='train')
    ax.plot(n_data, init_mean_train_acc, linestyle='--', lw=2, color='green', label='initial train')
    ax.plot(n_data, mean_dev_acc, linestyle='-', color='red', label='dev')
    ax.plot(n_data, init_mean_dev_acc, linestyle='--', lw=2, color='red', label='initial dev')

    ax.fill_between(n_data, mean_train_acc - std_train_acc, mean_train_acc + std_train_acc, alpha = 0.3)
    ax.fill_between(n_data, mean_dev_acc - std_dev_acc, mean_dev_acc + std_dev_acc, color='red', alpha = 0.3)
    ax.fill_between(n_data, init_mean_train_acc - init_std_train_acc, init_mean_train_acc + init_std_train_acc, color='green', alpha = 0.3)
    ax.fill_between(n_data, init_mean_dev_acc - init_std_dev_acc, init_mean_dev_acc + init_std_dev_acc, color='red', alpha = 0.3)
    ax.set_xlabel("#data")
    ax.set_ylabel("Accuracy")
    names_dict = {
        "Tom1": "Tomita 1",
        "Tom2": "Tomita 2",
        "Tom3": "Tomita 3",
        "Tom4": "Tomita 4",
        "Tom5": "Tomita 5",
        "Tom6": "Tomita 6",
        "Tom7": "Tomita 7"
    }
    if (epoch == "best"):
        # Do not mention epoch in the title
        title = f"{names_dict[lang]}"
    else:
        title = f"{names_dict[lang]}, {epoch}"
    plt.title(title)
    plt.tight_layout()
    ax.legend()
    plotname = f"./images_b4_ICML/acc-{lang}-{str(threshold)}-{epoch}-{eval}-{symmetric}-{nfa}.pdf"
    print(f"Saved {plotname}")
    plt.savefig(plotname)
    plt.show()
