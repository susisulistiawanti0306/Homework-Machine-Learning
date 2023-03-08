import getopt
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
import sys


def run_clustering(n_cluster):
    sns.set()

    data = pd.read_csv('data/country_location.csv')

    print(data)

    # select logitude dan latitude
    x = data.iloc[:,1:3]

    print(x)

    # modelling
    kmeans = KMeans(n_cluster)
    kmeans.fit(x)

    # calculate prediction
    identified_clusters = kmeans.fit_predict(x)

    print(identified_clusters)

    # copy data ke variable baru
    data_with_clusters = data.copy()

    # bikin kolom baru bernama cluster
    data_with_clusters['Clusters'] = identified_clusters

    # bikin plot scatter
    plt.scatter(data_with_clusters['Longitude'], data_with_clusters['Latitude'], c=data_with_clusters['Clusters'], cmap='rainbow')

    # label plot
    plt.title('Clustered Country using K-Means')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')

    # anotasi plot
    for i, txt in enumerate(data_with_clusters['Country']):
        plt.annotate(txt, (x.loc[i, 'Longitude'], x.loc[i, 'Latitude']))

    # save image plot
    plt.savefig('output/clustered_country.png')

if __name__ == '__main__':
    argumentList = sys.argv[1:]

    options = "c:"

    long_options = ["cluster"]

    arguments, values = getopt.getopt(argumentList, options, long_options)

    for current_argument, current_value in arguments:
        if current_argument == "--cluster":
            n_cluster = values[0]
        elif current_argument == "-c":
            n_cluster = current_value

    run_clustering(eval(n_cluster))
