import argparse
import datetime
import os
import pickle
import sys
import time
from datetime import date

import numpy as np
import scipy.io
from sklearn import linear_model
from sklearn import svm

import monitor_parser as mp
import pollution_parser as pp
import utils as u
from corss_validation import cross_validation


def run_entire_pipeline(data_folder,
                        regressor,
                        folds,
                        print_hist,
                        save_gmm,
                        use_latest_gmm,
                        save_hist_folder):
    

    todays_date = str(datetime.datetime.now()).split(" ")[0]
    hour_minute = str(datetime.datetime.now()).split(" ")[1]
    sub_folder_name = (
        str(todays_date.split("-")[0])
        + str(todays_date.split("-")[1])
        + str(todays_date.split("-")[2])
        + "_"
        + str(hour_minute.split(":")[0])
        + str(hour_minute.split(":")[1])
    )
    hist_folder = u.check_folder_and_create(save_hist_folder + "/" + sub_folder_name)
    hist_bool = [print_hist, hist_folder]  # first argument is for showing them on screen, second for saving them''
    

    data_samples, sample_classes = get_data(data_folder)    
    fited_model_for_each_round, results_for_each_round, result_names = cross_validation(regressor, data_samples, sample_classes, folds, hist_bool)


if __name__ == "__main__":
    # default parametters
    folds = 3
    save_gmm = 1
    save_hist_folder = (
        "/Volumes/Data_HD/Users/Carles/TFG/codes/try_github/TFG_database/results"
    )
    default_image_folder = "/Volumes/Data_HD/Users/Carles/TFG/codes/try_github/TFG_database/image_data/00000398"
    default_samples_per_class = -1

    use_latest_gmm = 1  # the index says the number of gmm that wants to be used
    default_regresor = ["LIN"]

    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", help=f'specify the folder with the image_data (deafault = {default_image_folder})',
        default=default_image_folder,
        type=str
    )

    parser.add_argument("-F","--folds",help=f'number of folds performed in cross validation (default = {str(folds)})',
        default=folds,
        type=int,
    )

    parser.add_argument(
        "-R",
        "--regresor",
        help=f'regresor used (can be: "SVM"(support vector machine), "ELN" (elasticnet), "BAY" (bayesian ridge), "LIN" (linear regression), "LASSLAR" (lasso least angle regression), "LASS"  (lasso regression), "THE" (theil sen regression), "ARDR" (bayesian ARD regression), deafult is {default_regresor[0]})',
        default=default_regresor,
        nargs="+",
        action="store",
    )

    parser.add_argument(
        "--print_hist",
        help="Show histograms (deafult = 0)",
        default=0,
        type=int
    )

    parser.add_argument(
        "--save_hist_folder",
        help="folder to save histograms (deafult = " + str(save_hist_folder) + ")",
        default=save_hist_folder,
        type=str,
    )



    args = parser.parse_args()

    data_folder = args.image_folder
    regresor = args.regresor:


    folds = args.folds
    print_hist = args.print_hist
    save_hist_folder = args.save_hist_folder


    run_entire_pipeline(data_folder,
                        regresor,
                        folds,
                        print_hist,
                        save_hist_folder)