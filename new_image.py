import matlab.engine
import sqlite3
import pollution_parser as pp
import download_amos as da
import utils as u
import os
import pickle
import numpy as np
from sklearn.mixture import GaussianMixture
import scipy
import matplotlib.pyplot as plt

parameters_for_saving= ['Ozone','SO2','CO','NO2', 'PM2.5', 'PM10']
switcher_xlabel_plot = {
    0: "Ozone concentration (ppm mg/km)",
    1: "SO2 concentration (ppb)",
    2: "CO concentration (ppm mg/km)",
    3: "NO2 concentration (ppb)",
    4: "PM2.5 concentration (mg/m3)",
    5: "PM10 concentration (mg/m3)",
}

title_switcher = {
    0: "Ozone",
    1: "SO2",
    2: "CO",
    3: "NO2",
    4: "PM2.5",
    5: "PM10",
}

def get_features_from_sample_image(eng:matlab.engine,image_path,folder_to_save):
    image_name=image_path.split('/')[-1]
    samples=[]
    classes=[]
    folder_path=image_path.replace(image_name,'')
    eng.cd('mcloud')
    if image_name.endswith('.jpg'):
        print("Getting features from: " + image_path, end='\r')
        data_path=folder_to_save
        u.check_folder_and_create(data_path,0)
        raw_desc_feat=eng.first_test(folder_path,image_name,data_path,[12,16,20], 8, 300)
        print('Saved faetures into: '+ data_path)
        final_fisher_vector=[]
        print(len(raw_desc_feat))
        [print(len(f)) for f in raw_desc_feat]
        '''
        for raw in raw_desc_feat:
            final_fisher_vector.append(raw)
        samples.append(final_fisher_vector)
        classes.append(int(folder_name))
        '''

    #return (samples,classes)

def parse_pollution_txt(final_path):
    myfile=open(final_path,"r")
    lines = myfile.read().split('[')
    measure_values=[]
    count=0
    for f in lines:
        if not f:
            continue
        else:

            ff=f.replace(']','')
            fff=ff.split(',')
            if len(fff)>1:
                if count==6:
                    for ffff in fff:
                        try:
                            measure_values.append(float(ffff))
                        except:
                            
                            print("NO HI HA VALOR")
                            print(ffff)
                            
                            #return np.zeros(6)
                else:
                    measure_values.append(float(fff[0])) 

            else:
                try:
                    measure_values.append((float(fff[0])))
                except:
                    
                    print("NO HI HA VALOR")
                    print(fff)
                    
                    # return np.zeros(6)
                    #measure_values.append((float(fff[0])))
            #print(len(f))
            #print(ff)
            #[print(char) for char in lines]
            count+=1
    '''
    print(str(len(measure_values)) + ':     ',end='')
    [print(str(s) + ', ',end='') for s in measure_values]
    print('')
    '''
    #print(np.asarray(measure_values).shape)

    return(np.asarray(measure_values))

def new_image_fv(visual_features,met_data,gaussian_models,multi_modal_weights):
    count=0
    weigth_selector=0
    sample_fisher_vector=[]
    retrieve_count=0
    scale=100
    for data_array in visual_features:
        fv=u.fisher_vector(np.asarray(data_array), gaussian_models[count])
        sample_fisher_vector.append(fv)
        count+=1

    '''    
    for descriptor_name in descriptors:
        for patch_size in patches:
            data_array=sample_features[count]
            #retrieve_feature_from_sample(sample_name,pollution_data_paths,descriptor_name,patch_size)
            #DATA NORMALIZATION
            fv=fisher_vector(np.asarray(data_array), gaussian_models[count])

            #square-rooting normalization
                #en teoria la funcio fisher_vector ja la fa
            #l2 normalization

            #fisher vector append
            
            # print("Fisher vector " + str(count) + ' has lenght: ' + str(len(fv)))
            # print("Fisher vector " + str(count) + ' has shape: ' + str((fv.shape)))
            
            sample_fisher_vector.append(fv)

            count+=1
    '''
    visual_fv_length=np.sum(np.concatenate([f.shape for f in sample_fisher_vector]))
    visual_fisher_vector=np.concatenate([(multi_modal_weights[weigth_selector]/visual_fv_length)*f for f in sample_fisher_vector])
    weigth_selector+=1
        
    
    poll_data_from_sample=met_data
    poll_data_length=len(poll_data_from_sample)
    met_fisher_vector=(multi_modal_weights[weigth_selector]/poll_data_length)*u.fisher_vector(np.asarray(poll_data_from_sample), gaussian_models[count])
    final_fisher_vector=np.concatenate([visual_fisher_vector,met_fisher_vector])
    #get data from txt
    #fv = fsh.fisher_vector(met_data, gaussian_models[count][0], gaussian_models[count][1], gaussian_models[count][2])
    #square-rooting normalization
        #en teoria la funcio fisher_vector ja la fa
    #l2 normalization

    return final_fisher_vector

def retrieve_feature_from_sample(feature_folder_path):
    #returns array with dimensions (NxD) where N is the number of feature points extracted with that patch size and D the dimensionality of the descriptor
    #print("retrieve_feature_from_sample MUST BE TESTED")
    feature=[]
    #print(sample_name)
    print(feature_folder_path)
    try:
        array=scipy.io.loadmat(feature_folder_path)
        fea_arr=array['feaSet']['feaArr'][0][0]
        #print(fea_arr.shape)
        feature.append(fea_arr)
    except Exception as e:
        print(e)
        print("Couldn't load feature " + feature_folder_path.split('/')[-1])
    print(feature)
    print(feature_folder_path)
    return feature[0]

def load_gmm(gaussian_mixture_path):
    npzfile = np.load(gaussian_mixture_path)
    print("Got mixture model " + gaussian_mixture_path.split('/')[-1])

    number_of_gaussians=int(gaussian_mixture_path.split('/')[-1].split('_')[2].replace('N',''))
    model = GaussianMixture(n_components=number_of_gaussians, covariance_type='diag')
    model.means_ = npzfile["means"]
    model.covariances_ = npzfile["covs"]
    model.weights_ = npzfile["weights"]
    model.precisions_cholesky_ = npzfile["precisions_cholesky"]


    return model

def fit_new_patch(patch_path,linear_models,gaussian_models,multimodal_weigths,pca_model,met_data,visual_features):
    # u.delete_last_lines()
    # print("Fitting patch " + patch_path.split('/')[-1])
    image_feature=new_image_fv(visual_features,met_data,gaussian_mixture_models,multimodal_weigths)
    fuck_you_ostia_puta=image_feature.reshape(1, -1)
    pca_features = pca_model.transform(fuck_you_ostia_puta)

    parameter_count=0
    name_of_pollutant=[]
    predicted_pollution=[]
    for linear_model in linear_models:
        result=linear_model.predict(pca_features.reshape(1, -1))
        predicted_pollution.append(result)
        name_of_pollutant.append(parameters_for_saving[parameter_count])
        parameter_count+=1

    return (predicted_pollution,name_of_pollutant)


###################

def plot_histograms(histograms,boolean,folder_to_save,bins=50):
    print(len(histograms))
    print(len(histograms[0]))
    print(len(histograms[0][0]))
    u.check_folder_and_create(folder_to_save)
    i=0
    for h in histograms:
        histogram=np.asarray(h)
        plt.figure()
        print(histogram.shape)

        plt.plot(histogram[1][:-1],histogram[0],label='Train samples',color='darkgreen')
        x_step=1
        #print(test_histograms[i][1][0:x_step:50])
        #plt.xticks(test_histograms[i][1][0:x_step:-1])
        plt.xlabel(switcher_xlabel_plot.get(i,'Invalid parameter name'))
        plt.title(title_switcher.get(i,'hgfds') + " histogram")
        plt.ylabel('Number of samples')

        if boolean[1]:#save them
            plt.savefig(folder_to_save+'/'+title_switcher.get(i,'hgfds') + " histogram " + str(bins) +" bins.png", bbox_inches='tight')
        if bool(boolean[0]):#print them
            if i==len(histograms)-1:#just if its the final round and the final histogram
                plt.show()
            else:
                plt.draw()
        else:
            plt.close()
        i+=1

linear_models_path=[]
models_folder='/Volumes/Data_HD/Users/Carles/TFG/cluster files/results/goodmodels/20190710_2047-3/linear_models'
regresors=['BayesianRidge','BayesianRidge']
[linear_models_path.append(models_folder+'/'+parameters_for_saving[r]+'_'+ regresors[r]+ '.sav') for r in range(len(regresors))]




gaussian_models_paths=[]
gaussian_models_paths.append('/Volumes/Data_HD/Users/Carles/TFG/codes/try_github/TFG_database/gmm/65_gmm_N200_598997s_k1_centrist12_else.npz')
gaussian_models_paths.append('/Volumes/Data_HD/Users/Carles/TFG/codes/try_github/TFG_database/gmm/66_gmm_N10_3908s_k1_weather_else.npz')


multimodal_weigth=[0.8,1]

pca_model_path='/Volumes/Data_HD/Users/Carles/TFG/cluster files/results/goodmodels/20190617_2203-4/PCA/PCA_basis_round1.sav'



descriptors=['centrist']
patch_sizes=[12]

default_image_folder='/Volumes/Data_HD/Users/Carles/TFG/codes/try_github/TFG_database/image_data/00030877'
samples_per_class=-1
if samples_per_class<=0:#to get the number of samples that has the folder with less samples
    samples_per_class=min(np.asarray([len(os.listdir(os.path.join(default_image_folder,f))) for f in os.listdir(default_image_folder) if os.path.isdir(os.path.join(default_image_folder,f))]))
clases_to_get=[int(i) for i in os.listdir(default_image_folder) if i != '.DS_Store' and i!='420']#[42,129]    
clases_to_get=[34,51,55]
(sample_names,sample_classes,sample_pollution_paths,sample_features)=u.get_samples(clases_to_get,samples_per_class,default_image_folder,'pollution',descriptors,patch_sizes)


#get met data
met_test_samples=np.vstack([u.retrieve_weather_data_from_sample(sample_pollution_paths[i],sample_names[i])  for i in range(len(sample_names))])
n_bins=50

linear_models=[]
#LOAD LINEAR MODELS
for path_to_model in linear_models_path:
    reg=pickle.load(open(path_to_model, 'rb'))
    print(reg.coef_)
    linear_models.append(reg)

gaussian_mixture_models=[]
#LOAD GMM
for gaussian_model in gaussian_models_paths:
    descriptor=gaussian_model.split('_')[-2]
    if descriptor=='weather':
        met_gaussian_model=load_gmm(gaussian_model)
    else:
        gaussian_mixture_models.append(load_gmm(gaussian_model))
#SToRe MET GMM LAST
#store meteorological gmm with others
gaussian_mixture_models.append(met_gaussian_model)

#LOAD PCA MODEL
pca_model = pickle.load(open(pca_model_path, 'rb'))


predicted_pollution_list=[]
print()
for i in range(len(sample_names)):
    patch_path=os.path.join(sample_pollution_paths[i].replace('pollution_data','image_data'),sample_names[i])
    u.delete_last_lines()
    print("Fitting patch " + patch_path.split('/')[-1] + ' ('+ str(i)+'/'+str(len(sample_names))+')')
    (predicted_pollution,name_of_pollutant)= fit_new_patch(patch_path,linear_models,gaussian_mixture_models,multimodal_weigth,pca_model,met_test_samples[i],sample_features[i])
    predicted_pollution_list.append(predicted_pollution)


poll_predicted_array=np.asarray(predicted_pollution_list)
poll_truth_array=np.asarray(sample_classes)

print(poll_predicted_array.shape)
print(poll_truth_array.shape)
r2_scores=[]
s_tots=[]
s_resss=[]
results_histograms=[]
for p in range(len(name_of_pollutant)):
    samples_with_that_pollution_data=poll_truth_array[:,p]!=420
    pollutant_truth_samples=poll_truth_array[:,p][samples_with_that_pollution_data]
    pollutant_predicted_samples=poll_predicted_array[:,p][samples_with_that_pollution_data]
    print(np.mean(pollutant_truth_samples))
    print(np.mean(pollutant_predicted_samples))
    (n,b)=np.histogram(np.asarray(pollutant_predicted_samples),bins=n_bins)

    results_histograms.append((n,b))

    s_ress=u.compute_s_ress(pollutant_predicted_samples,pollutant_truth_samples)
    s_tot=u.compute_s_tot(pollutant_truth_samples)
    expected_r2=1.-(np.asarray(s_ress)/np.asarray(s_tot))
    r2_scores.append(expected_r2)
    s_resss.append(s_ress)
    s_tots.append(s_tot)


print(r2_scores)
plot_histograms(results_histograms,[1,1],'/Volumes/Data_HD/Users/Carles/TFG/codes/histograms/30877',bins=n_bins)
