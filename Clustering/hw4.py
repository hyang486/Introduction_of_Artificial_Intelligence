"""
Author : Hyeonmin  Yang
ID : 9081998412
Instructor : Yudong Chen
"""
import csv
import numpy as np
from numpy import linalg as LG
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt


"""
0.1 load_data(filepath)
input : string, the path to a file to be read
output : list, where each element is a dict representing one row of the file read.
return a list of dictionaries, where each row in the dataset is a dictionary with the cloumn
header as keys and the row elements as value
"""


def load_data(filepath):
    # create an empty list for return the list dictionary
    return_dictionary = []
    # open countries.csv file  read it line by line
    with open("countries.csv", newline="") as file:
        reader = csv.DictReader(file)
        # with using for loop append each row to return_dictionary
        for row in reader:
            return_dictionary.append(dict(row))
    # return a list of dictionaries
    return return_dictionary


"""
0.2 calc_features(row)
x1 = ‘Population’
2. x2 = ‘Net migration’
3. x3 = ‘GDP ($ per capita)’
4. x4 = ‘Literacy (%)’
5. x5 = ‘Phones (per 1000)’
6. x6 = ‘Infant mortality (per 1000 births)’
with these data, first append it to new dictionary
then change it to a numpy array with specific data type (np.float64)
"""


def calc_features(row):
    # first declare a new dictionary for datasets
    calc_features_dic = []
    # append data form row which has key 'Population'
    calc_features_dic.append(row["Population"])
    # append data form row which has key 'Net migration'
    calc_features_dic.append(row["Net migration"])
    # append data form row which has key 'GDP ($ per capita)'
    calc_features_dic.append(row["GDP ($ per capita)"])
    # append data form row which has key 'Literacy (%)'
    calc_features_dic.append(row["Literacy (%)"])
    # append data form row which has key 'Phones (per 1000)'
    calc_features_dic.append(row["Phones (per 1000)"])
    # append data form row which has key 'Infant mortality (per 1000 births)'
    calc_features_dic.append(row["Infant mortality (per 1000 births)"])

    # create a numpy array with specific data type
    numpy_version_array = np.array(calc_features_dic, dtype=np.float64)
    #  we need to reshape it to (6,)
    numpy_version_array = np.reshape(numpy_version_array, (6,))
    # return numpy_version_array
    return numpy_version_array


"""
0.3 hac(features)
input : list of NumPy arrays of shape (6,),
output : Z numpy array with info of clustering 
first make a distance matrix with using Euclidean distance
and do clustering with complete-linkage algorithm for all datapoints
and store clustering info to Z matrix
"""


def hac(features):
    # first we need a number of data point and it's same the
    # first dimension of features
    num_dataPoint = len(features)

    # Initialize the linkage matrix Z with 0
    # this matrix should be 2D array with the instruction (num_dataPoint - 1 * 4)
    # and it's elements' type should be float
    Z = np.zeros((num_dataPoint - 1, 4))

    # Also we need distance matrix, this should be contain all the distance
    # bewteen all th coountries datapoints, so this will be num_dataPoint * num_dataPoint
    distance_matrix = np.zeros((num_dataPoint, num_dataPoint))

    # put distance between each pair of data point into distance_matrix
    # first we need to loop all data point
    for itself in range(num_dataPoint):
        # but we need to compare without ith datapoint itself
        # so the range will be i + 1 ~ end of datapoint
        for fromNext in range(itself + 1, num_dataPoint):
            # with using np.linalg.norm() we can get Euclidean distance
            distance_matrix[itself, fromNext] = LG.norm(
                features[itself] - features[fromNext]
            )
            # we saw that distance matrix is symmetric in the class
            # because distance between 1,2 and 2,1 should be same
            distance_matrix[fromNext, itself] = distance_matrix[itself, fromNext]

    # Initialize a dictionary to keep track of clusters
    # in this dictionary the key will be the first index of each element
    # and the value will store the cluster of datapoint
    # first we need to declare edictionary
    tracing_clusters = {}
    # and then initialize the key of each dictionary as a list to it's index(key)
    for j in range(num_dataPoint):
        tracing_clusters[j] = [j]

    # because we set tracing_clusters as a dictionary, we need to
    # set the start cluster_index as the last index + 1 of tracing_clusters
    new_cluster_index = num_dataPoint

    # now start with Agglomerative Hierarchical Clustering with complet-linkage
    # outest loop is untill all cluster is merged as one so the numbe should be num_dataPoint - 1
    i = 0
    while i < num_dataPoint - 1:
        # first, we need to set first minimun distance -> it should be biggest float enough
        # because any distance should be smaller than fisrt min distance
        # minimun_distance will be return to infinit float for each loop
        minimun_distance = np.inf

        # declare pair variable this pair's element will be the index of clustered datapoint
        # this pair will be also return to None for each loop
        pair = None

        # Store the every key to list for using it to iteration
        # because we need to iterate the loop with the number of keys in tracing_clusters
        keys = list(tracing_clusters.keys())

        # then make loop to get mininum_distance and pair
        # range of outest loop should be 0 ~  len(keys)(exclusive)
        # and the len(keys) will be decreasing by 1 for each loop
        for fromDataIndex in range(len(keys)):
            # we need a key value of tracing_clusters for fromDataIndex
            # which is the element of keys list
            fromData_key_inDic = keys[fromDataIndex]
            # lterate start with the value fromDataIndex + 1, because
            # we don't need to compare same datapoint itself and the datapoints
            # which is alreay compared (ex: [1,2] and [2,1])
            for toDataIndex in range(fromDataIndex + 1, len(keys)):
                # we need a key value of tracing_clusters for toDataIndex
                # which is the element of keys list
                toData_key_inDic = keys[toDataIndex]

                # this is processing for get maximun distance because we are
                # goint to use complete-linkage algorithm
                # Set the max distance to 0.0 so any first distance can be
                # the max_distance and update it with compring
                max_distance = 0.0
                # outer loop is the elements of clustered data point or single datapoint(start)
                for values_in_clusData1 in tracing_clusters[fromData_key_inDic]:
                    # inner loop is the elements of clustered data point or single datapoint(destination)
                    for values_in_clusData2 in tracing_clusters[toData_key_inDic]:
                        # for each loop it will get distance of every element to every element
                        current_distance = distance_matrix[
                            values_in_clusData1, values_in_clusData2
                        ]
                        # if current_distance is bigger than max_distance update it
                        if current_distance > max_distance:
                            max_distance = current_distance
                        # if not continue to the next iteration
                        else:
                            continue
                # with result of above, set the distance between two datapoints
                distance = max_distance

                # if distance is smaller than minimun_distance,
                # update minimun_distance and pair
                if distance < minimun_distance:
                    minimun_distance = distance
                    pair = (fromData_key_inDic, toData_key_inDic)
                # if the distance is equal to minimun_distance do tie breaking
                elif distance == minimun_distance:
                    # if the first index is different
                    if pair[0] != fromData_key_inDic:
                        # update pair with smaller first index one
                        if pair[0] > fromData_key_inDic:
                            pair = (fromData_key_inDic, toData_key_inDic)
                        # if not keep current pair
                        else:
                            pair = pair
                    # if the first index is same but the second index is different
                    else:
                        # update pair with smaller second index one
                        if pair[1] > toData_key_inDic:
                            pair = (fromData_key_inDic, toData_key_inDic)
                        # if not keep current pair
                        else:
                            pair = pair

        # with combining two datapoints, create a new cluster
        new_cluster = tracing_clusters[pair[0]] + tracing_clusters[pair[1]]
        # delete duplicated element
        new_cluster_noDup = []
        for item in new_cluster:
            if item not in new_cluster_noDup:
                new_cluster_noDup.append(item)

        # Update Z matrix with info of new cluster
        # update first column with smaller index
        Z[i, 0] = int(min(pair[0], pair[1]))
        # update second column with bigger index
        Z[i, 1] = max(pair[0], pair[1])
        # update third column with distance
        Z[i, 2] = minimun_distance
        # update fourth column with number of elements in new cluster
        Z[i, 3] = len(new_cluster_noDup)

        # add the new cluster to tracing_clusters dictionary
        tracing_clusters[new_cluster_index] = new_cluster_noDup
        # delete the old cluster elements from tracing_clusters dictionary
        del tracing_clusters[pair[0]]
        del tracing_clusters[pair[1]]

        i += 1
        new_cluster_index += 1
    Z = np.array(Z)
    return Z


"""
04.fug_hac(Z, names)
This method is for visualizing hierachical clustering with using
matplotlib library. this will using dendrogram and this is visualzing
hierachical clustering with returned array of hec() method.
"""


def fig_hac(Z, names):
    # initialize a figure
    fig = plt.figure()
    # visualize with provided linkage matrix and each datapoint's
    # name with name  and lotate 90 degree for vertically view
    dendrogram(Z, labels=names, leaf_rotation=90)
    # with using tight_layout(), prevent overlapping view
    fig.tight_layout()
    return fig


"""
05.normalize_features(features)
first we need to get mean of x axis 
and startd deviation of x axis
because to calculate the nomalize feature values we need to do
(original value - column's mean) / and columns standard deviation

"""


def normalize_features(features):
    # becase we set input as numpy array so use np.mean
    # and np.std to get mean and standard deviation for each column
    mean = np.mean(features, axis=0)
    standard_deviation = np.std(features, axis=0)

    # now we can get nomalize feature values
    normalized_features = (features - mean) / standard_deviation

    # append nomrmalized features to list
    normalized_features_list = []
    for vector in normalized_features:
        normalized_features_list.append(vector)

    return normalized_features_list


if __name__ == "__main__":
    # # load data with contries.csv
    # data = load_data("countries.csv")
    # # in the csv file, get all the country names and
    # # store it to name list
    # country_names = [row["Country"] for row in data]

    # # append calc_fetures result with all features in data as input
    # # to features list
    # features = [calc_features(row) for row in data]

    # # store result of nomlaize version
    # features_normalized = normalize_features(features)

    # n = 50
    # # store Z array with 2 version (original and nomalized)
    # Z_raw = hac(features[:n])
    # print(Z_raw)
    # Z_normalized = hac(features_normalized[:n])

    # print("Testing for n ", n)
    # # show the graph with 2 version (original and nomalized)
    # fig_hac(Z_raw, country_names[:n])
    # fig_hac(Z_normalized, country_names[:n])
    # plt.show()

    data = load_data("countries.csv")
    country_names = [row["Country"] for row in data]
    features = [calc_features(row) for row in data]
    features_normalized = normalize_features(features)
    n = 10

    # Take the last n rows from the end
    data_last_n = data[-n:]
    features_last_n = features[-n:]
    features_normalized_last_n = features_normalized[-n:]
    country_names_last_n = country_names[-n:]

    Z_raw = hac(features_last_n)
    Z_normalized = hac(features_normalized_last_n)
    fig = fig_hac(Z_raw, country_names_last_n)
    fig = fig_hac(Z_normalized, country_names_last_n)
    plt.show()
