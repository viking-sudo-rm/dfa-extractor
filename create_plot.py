import matplotlib.pyplot as plt
import numpy as np

def create_plot(init_train_acc, init_dev_acc, train_acc, dev_acc, n_data, lang, threshold):


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

    ax.plot(n_data, init_mean_train_acc, linestyle='--', marker='.', label='initial train acc')
    ax.plot(n_data, init_mean_dev_acc, linestyle='--', marker='.', label='initial dev acc')
    ax.fill_between(n_data, init_mean_train_acc - init_std_train_acc, init_mean_train_acc + init_std_train_acc, alpha = 0.3)
    ax.fill_between(n_data, init_mean_dev_acc - init_std_dev_acc, init_mean_dev_acc + init_std_dev_acc, alpha = 0.3)

    ax.plot(n_data, mean_train_acc, linestyle='-', marker='.', label='train acc')
    ax.plot(n_data, mean_dev_acc, linestyle='-', marker='.', label='dev acc')
    ax.fill_between(n_data, mean_train_acc - std_train_acc, mean_train_acc + std_train_acc, alpha = 0.3)
    ax.fill_between(n_data, mean_dev_acc - std_dev_acc, mean_dev_acc + std_dev_acc, alpha = 0.3)
    ax.set_xlabel("#data")
    ax.set_ylabel("Accuracy")
    plt.title(lang + ', threshold =' + str(threshold))
    ax.legend()
    plotname = f"./images/acc-{lang}-{str(threshold)}.pdf"
    print(f"Saved {plotname}")
    plt.savefig(plotname)
    # plt.show()
