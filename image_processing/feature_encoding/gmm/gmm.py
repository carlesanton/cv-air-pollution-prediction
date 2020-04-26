from utils import check_folder_and_create
from sklearn.mixture import GaussianMixture

def retrieve_or_compute_gmm(feature_samples, number_of_gaussians, gaussian_mixtures_folder):
    if compute_new_gmm: 
        # Compute gmm
        gmm = generate_gaussian_mixture_model(feature_samples, number_of_gaussians)
        # Save GMM if wanted
        check_folder_and_create(gaussian_mixtures_folder)
        save_gmm(gaussian_mixtures_folder,gmm)
    else:  # if we did not find a stored gm model
        gmm = load_gmm(gaussian_mixtures_folder)
        
    return gmm

def generate_gaussian_mixture_model(samples, number_of_gaussians):
    fv_gmm = GaussianMixture(
        n_components=number_of_gaussians, covariance_type="diag"
    )
    fv_gmm.fit(samples)

    return fv_gmm

def save_gmm(folder_to_save, gmm: sklearn.mixture.GaussianMixture):

    number_of_gmm_on_folder = len(os.listdir(folder_to_save))
    gmm_name = f'gmm_model_{number_of_gmm_on_folder+1}.npz'
    print("Saving GMM " + gmm_name + " into folder " + folder_to_save)
    
    # Setting up variables to save into
    means = np.float32(gmm.means_)
    covs = np.float32(gmm.covariances_)
    weights = np.float32(gmm.weights_)
    precisions_cholesky = np.float32(gmm.precisions_cholesky_)

    filepath = os.path.join(folder_to_save, gmm_name)

    np.savez(
        filepath,
        means=means,
        covs=covs,
        weights=weights,
        precisions_cholesky=precisions_cholesky,
    )

def load_gmm(gaussian_mixtures_folder, model_number = -1):
    """
    Load a Gaussian Mixture Model from the gaussian_mixtures_folder.
    If model_number == -1 it will load the latest created model of that folder.
    Otherwise it will load the model specified by model_number
    """
    # Get file name
    if model_number <= len(os.listdir(gaussian_mixtures_folder)) and number > 0:
        # If number is grater than 0 and in the range of valid files
        number_of_gmm = model_number
    else:
        # otherwise we take the latest gmm created
        number_of_gmm = len(os.listdir(gaussian_mixtures_folder))

    gmm_model_name = f'gmm_model_{number_of_gmm_on_folder+1}.npz'
    gmm_filepath = os.path.join(gaussian_mixtures_folder,gmm_model_name)
    
    # Load model file
    npzfile = np.load(gmm_filepath)
    print("Got mixture model for " + descriptor_info + " from " + filepath)

    # Convert model file to usefull GaussianMixture object
    model = GaussianMixture(
        n_components=number_of_gaussians, covariance_type="diag"
    )
    model.means_ = npzfile["means"]
    model.covariances_ = npzfile["covs"]
    model.weights_ = npzfile["weights"]
    model.precisions_cholesky_ = npzfile["precisions_cholesky"]

    return model