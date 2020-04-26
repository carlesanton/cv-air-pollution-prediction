import numpy as np
import sklearn.mixture
from sklearn import decomposition
from sklearn import metrics

from gmm.gmm import retrieve_or_compute_gmm
from utils import check_folder_and_create


def encode_features(visual_features, meteorological_samples, gmm_number_tuple, PCA):
    """
    if PCA is a tuple it means that visual and meteorological features must be reduced by PCA separately.
    """
    encoded_feature_vector = np.array([],dtype=float) # vector to contain all feature vectors one next to another
    gmm_root_folder = "gmm/gmmmodels"
    number_of_gaussians_visual_features = gmm_number_tuple[0]
    number_of_gaussians_meteorological_features = gmm_number_tuple[1]
    for visual_feature in visual_features:
        # compute gmm
        feature_gmm_folder = os.path.join(gmm_root_folder,
                                          "visual_features",
                                          number_of_gaussians_visual_features,
                                          visual_feature.name)
        encoded_visual_feature = encode_feature(feature_gmm_folder, visual_feature.data, number_of_gaussians_visual_features)
        encoded_feature_vector = np.concatenate((encoded_feature_vector,encoded_visual_feature),axis=0)

    # Encode meteorological data
    number_of_gaussians_met_features = gmm_number_tuple[1]
    meteorological_gmm_folder = os.path.join(gmm_root_folder,
                                            "meteorological",
                                            number_of_gaussians_met_features)
    encoded_meteorological_feature = encode_feature(meteorological_gmm_folder,
                                                    meteorological_samples,
                                                    number_of_gaussians_met_features)

    encoded_feature_vector = np.concatenate((encoded_feature_vector,encoded_meteorological_feature),axis=0)
    #PERFORM PCA AND FIT DATA TO REGRESSOR, RETURNING AS OUTPUT THE METRICS COMPUTED
    if bool(PCA):
        encoded_feature_vector = perform_pca_on_data(encoded_feature_vector, PCA, "PCA_models")
        
    return encoded_feature_vector

def encode_feature(feature_gmm_folder, feature_data, number_of_gaussians):
    gmm_model = retrieve_or_compute_gmm(feature_data,
                                        number_of_gaussians,
                                        feature_gmm_folder)
    
    # use gmm to compute fisher vector encoding of the feature and
    # concatenate it to the current vector of encoded features
    encoded_feature = fisher_vector(np.asarray(feature_data), gmm_model)

    return encoded_feature

def perform_pca_on_data(data, pca_components, folder_to_save_pca_model):
    
    print(f'Data dimension beofre PCA: {len(data[0])}')
    if pca_components<1 and pca_components>0:
        # if we are telling what portion of the dimensions we want to keep
        pca_components=np.round(len(data[1])/pca_components)

    pca = decomposition.PCA(n_components=int(pca_components))
    reduced_data = pca.fit_transform(data)
    print(f'Data dimension after PCA: {len(data[0])}')

    #save PCA model
    check_folder_and_create(folder_to_save_pca_model)
    print("Saving PCA model")
    models_on_folder = len(os.listdir(folder_to_save_pca_model))
    filename = f'PCA_basis_{models_on_folder+1}.sav'
    pickle.dump(pca, open(os.path.join(folder_to_save_pca_model,filename), 'wb'))


    return reduced_data

def fisher_vector(xx, gmm: sklearn.mixture.GaussianMixture):
    """Computes the Fisher vector on a set of descriptors.
    Parameters
    ----------
    xx: array_like, shape (N, D) or (D, )
        The set of descriptors
    gmm: instance of sklearn mixture.GMM object
        Gauassian mixture model of the descriptors.
    Returns
    -------
    fv: array_like, shape (K + 2 * D * K, )
        Fisher vector (derivatives with respect to the mixing weights, means
        and variances) of the given descriptors.
    Reference
    ---------
    J. Krapac, J. Verbeek, F. Jurie.  Modeling Spatial Layout with Fisher
    Vectors for Image Categorization.  In ICCV, 2011.
    http://hal.inria.fr/docs/00/61/94/03/PDF/final.r1.pdf
    """
    xx = np.atleast_2d(xx)
    N = xx.shape[0]

    # Compute posterior probabilities.
    Q = gmm.predict_proba(xx)  # NxK

    # Compute the sufficient statistics of descriptors.
    Q_sum = np.sum(Q, 0)[:, np.newaxis] / N
    Q_xx = np.dot(Q.T, xx) / N
    Q_xx_2 = np.dot(Q.T, xx ** 2) / N

    # Compute derivatives with respect to mixing weights, means and variances.
    d_pi = Q_sum.squeeze() - gmm.weights_
    d_mu = Q_xx - Q_sum * gmm.means_
    d_sigma = (
        -Q_xx_2
        - Q_sum * gmm.means_ ** 2
        + Q_sum * gmm.covariances_
        + 2 * Q_xx * gmm.means_
    )

    # print(d_sigma.shape)
    """
    print('xx shape ' + str(xx.shape))
    print('d_sigma shape ' + str(d_sigma.shape))
    print('d_sigmaflat shape ' + str(d_sigma.flatten().shape))
    """
    # Merge derivatives into a vector.
    return np.hstack(
        (d_mu.flatten(), d_sigma.flatten())
    )  # np.hstack((d_pi, d_mu.flatten(), d_sigma.flatten()))
