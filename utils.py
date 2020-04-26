import os
import matplotlib.pyplot as plt

def plot_data_histograms(train_histograms, test_histograms, round, max_round, boolean):
    # plt.ion()

    for i in range(len(train_histograms)):
        # plt.ioff()
        """
        if round==max_round and i==len(train_histograms):#just if its the final round and the final histogram
            print('hgfdsdfghjkjhgfds')
            plt.ion()
        else:
            print('AAAAAAAAAAAAAAASASASASADAEADAFADASASASA')
            plt.ioff()
            """
        plt.figure()
        plt.plot(
            train_histograms[i][1][:-1],
            train_histograms[i][0],
            label="Train samples",
            color="darkgreen",
        )
        plt.plot(
            test_histograms[i][1][:-1],
            test_histograms[i][0],
            label="Test samples",
            color="firebrick",
        )
        x_step = 1
        # print(test_histograms[i][1][0:x_step:50])
        # plt.xticks(test_histograms[i][1][0:x_step:-1])
        plt.xlabel(switcher_xlabel_plot.get(i, "Invalid parameter name"))
        plt.title(title_switcher.get(i, "hgfds") + " histogram for round " + str(round))
        plt.ylabel("Number of samples")
        plt.legend()

        if boolean[1]:  # save them
            plt.savefig(
                boolean[1]
                + "/"
                + title_switcher.get(i, "hgfds")
                + " histogram for round "
                + str(round)
                + ".png",
                bbox_inches="tight",
            )
        if bool(boolean[0]):  # print them
            if (
                round == max_round and i == len(train_histograms) - 1
            ):  # just if its the final round and the final histogram
                plt.show()
            else:
                plt.draw()
        else:
            plt.close()

def check_folder_and_create(path_to_folder, print_b=0):
    if not os.path.exists(path_to_folder):
        try:
            os.makedirs(path_to_folder)
            if print_b:
                print("Folder " + path_to_folder + " created")
        except OSError as e:
            # path=check_folder_and_create(path_to_folder+'_', print_b=0)
            print(e)
            # return path
    return path_to_folder
