from sklearn import metrics
import numpy as np

###COMPUTE METRICS
def compute_mean(predicted):
    number_samples = float(len(predicted))
    summ = float(np.sum(np.asarray(predicted)))
    return summ / number_samples


def compute_error_stdv(expected_vector, predicted_vector, mean_squared_error):
    errors = abs((expected_vector) - (predicted_vector))
    error_mean_error_squared_difference = (errors - mean_squared_error) ** 2
    # print(error_mean_error_squared_difference)
    stdv = sqrt(sum(error_mean_error_squared_difference) / len(predicted_vector))
    return stdv


def compute_stdv(expected_vector, sample_mean):
    errors = abs((expected_vector) - (sample_mean)) ** 2
    # print(error_mean_error_squared_difference)
    stdv = sqrt(sum(errors) / len(expected_vector))
    return stdv


def compute_s_ress(predicted_vector, expected_vector):
    diff_squares = (expected_vector - predicted_vector) ** 2
    residual = np.sum(diff_squares)
    return np.asarray(residual)


def compute_s_tot(expected_vector):
    diff_squares = (expected_vector - compute_mean(expected_vector)) ** 2
    [print(expected_vector[i]) for i in range(len(diff_squares)) if diff_squares[i] < 1]
    residual = np.sum(diff_squares)
    return np.asarray(residual)


def compute_r2_manually(predicted_labels, test_labels):
    number_of_outputs = test_labels[0].shape[0]
    
    s_ress = [compute_s_ress(predicted_labels[:, i], np.asarray(test_labels)[:, i])for i in range(number_of_outputs)]
    
    s_tot = [compute_s_tot(np.asarray(test_labels)[:, i])for i in range(number_of_outputs)]
    
    computed_r2 = 1.0 - (np.asarray(s_ress) / np.asarray(s_tot))

    return s_ress, s_tot, computed_r2

def compute_metrics(
    predicted_labels,
    test_labels,
):
    number_of_outputs = test_labels[0].shape[0]
    

    #RMSE
    root_mean_squared_errors = np.sqrt(metrics.mean_squared_error((test_labels), (predicted_labels), multioutput="raw_values"))
    
    #PREDICTED MEAN
    predicted_mean = [compute_mean(predicted_labels[:, i]) for i in range(number_of_outputs)]
    
    #ERROR STDV
    error_stdv = [compute_error_stdv(predicted_labels[:, i],np.asarray(test_labels)[:, i],root_mean_squared_errors[i])for i in range(number_of_outputs)]
    
    #R² using sklearn utils
    sklearn_r_squares = metrics.r2_score(np.asarray(test_labels), (predicted_labels), multioutput="raw_values")
    
    #R² using "manual" utils so we can get s_ress and s_tot separatelly
    s_ress, s_tot, computed_r2 = compute_r2_manually(predicted_labels, test_labels)

    #RESULTS STDV
    stdv = [compute_stdv(predicted_labels[:, i], predicted_mean[i])for i in range(number_of_outputs)]

    round_results = np.stack(
        (
            predicted_mean,
            stdv,
            root_mean_squared_errors,
            error_stdv,
            s_ress,
            s_tot,
            computed_r2,
            sklearn_r_squares,
        )
    )
    result_names = [
        "Predicted Mean",
        "Predicted stdv",
        "RootMeanSquareError",
        "Error stdv",
        "Residual Sum Squares (SSres)",
        "Data variance (SStot)",
        "Computed R^2",
        "R^2",
    ]
    print_results(round_results, result_names, fold)

    return (round_results, result_names)
    
    