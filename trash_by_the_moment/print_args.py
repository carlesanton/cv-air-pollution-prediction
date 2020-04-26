import argparse

def main():

if __name__ == "__main__":
    #default parametters

    parser = argparse.ArgumentParser()
    default_samples_per_class=20
    parser.add_argument("-N","--samples_per_calss", help = "number of samples per class (AQI index) used (default = " + str(default_samples_per_class) + ")",default=default_samples_per_class,type=int)
    parser.add_argument("-GMM","--number_of_gaussians", help = "number of gaussians used per gaussian mixture (default = " + str(number_of_gaussians) + ")",default=number_of_gaussians,type=int)

    args = parser.parse_args()

    print(args.samples_per_calss)
