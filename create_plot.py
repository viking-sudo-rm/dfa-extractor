import matplotlib.pyplot as plt
import numpy as np

def create_plot(train_acc, dev_acc, n_data, lang, threshold):

    train_array = np.array(list(train_acc.values()))
    mean_train_acc = np.average(train_array, axis=0)
    std_train_acc = np.std(train_array, axis=0)
    dev_array = np.array(list(dev_acc.values()))
    mean_dev_acc = np.average(dev_array, axis=0)
    std_dev_acc = np.std(dev_array, axis=0)

    fig, ax = plt.subplots()
    ax.plot(n_data, mean_train_acc, linestyle='-', marker='.', label='train acc')
    plt.plot(n_data, mean_dev_acc, linestyle='-', marker='.', label='dev acc')
    ax.fill_between(n_data, mean_train_acc - std_train_acc, mean_train_acc + std_train_acc, alpha = 0.3)
    plt.fill_between(n_data, mean_dev_acc - std_dev_acc, mean_dev_acc + std_dev_acc, alpha = 0.3)
    ax.set_xlabel("#data")
    ax.set_ylabel("Accuracy")
    plt.title(lang + ', threshold =' + str(threshold))
    ax.legend()
    plotname = f"./images/{lang + str(threshold)}.pdf"
    plt.savefig(plotname)
    plt.show()
