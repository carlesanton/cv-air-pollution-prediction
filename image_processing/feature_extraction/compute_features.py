def compute_image_features(feature_list):

    return

def normalize_data(data, norm_type="norm"):
    # NORMALIZATION
    if norm_type == "norm":
        scaler = Normalizer().fit(data)
        rescaled_samples = scaler.transform(data)
    elif norm_type == "scale":
        scaler = MinMaxScaler(feature_range=(0, 1))
        rescaled_samples = scaler.fit_transform(data)
    else:
        rescaled_samples = data

    return rescaled_samples
