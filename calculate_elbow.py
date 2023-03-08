import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans


def elbow(max_cluster):
    sns.set()

    data = pd.read_csv('data/country_location.csv')

    print(data)

    x = data.iloc[:, 1:3]

    print(x)

    # within cluster sum of square (jumlah kuadrat dari node-node yang ada dalam cluster)
    wcss=[]
    number_clusters = range(1,max_cluster)

    for i in number_clusters:
        kmeans = KMeans(i)

        kmeans.fit(x)

        wcss_iter = kmeans.inertia_

        wcss.append(wcss_iter)

    print(wcss)

    plt.plot(number_clusters,wcss)
    plt.title('The Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')

    plt.show()
