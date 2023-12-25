from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt

def load_and_center_dataset(filename):
    # Your implementation goes here!
    #first load data
    data = np.load(filename)
    #center data
    mean = np.mean(data, axis=0)
    centered_data = data - mean

    return centered_data

def get_covariance(dataset):
    # Your implementation goes here!
    # we have 2414 arrays
    number_of_arrays = dataset.shape[0]
    matrix_covariance = (1/(number_of_arrays - 1)) * np.dot(dataset.T, dataset)

    return matrix_covariance

def get_eig(S, m):
    # S : get_covariance(dataset), m is m by m matrix
    # get eigenvalues and eigenvectors of S
    eigen_values, eigen_vectors = eigh(S, subset_by_index=[S.shape[0] - m, S.shape[0] - 1])
    # Then we need the originial index of decending order matrix
    eigen_index = np.argsort(eigen_values)[::-1]
    
    # Make diagonal matrix with using eigen_values_index with speicific range 
    eigen_values_range = eigen_values[eigen_index]

    # Do same for eigen_vectors
    sorted_eigen_vectors = eigen_vectors[:, eigen_index]

  
    # Your implementation goes here!
    return np.diag(eigen_values_range), sorted_eigen_vectors

def get_eig_prop(S, prop):
#     # Your implementation goes here!
    sum_of_eigenValue = sum(eigh(S, eigvals_only = True))
    
    proportion_eigenValue, proportion_eigenVector = eigh(S, subset_by_value=[prop * sum_of_eigenValue, np.inf])
    proportion_eigen_value_index = np.argsort(proportion_eigenValue)[::-1]

    proportion_eigen = proportion_eigenValue[proportion_eigen_value_index]
    proportion_eigen_vectors = proportion_eigenVector[:, proportion_eigen_value_index]


    return np.diag(proportion_eigen), proportion_eigen_vectors


def project_image(image, U):
    # Your implementation goes here!
    projection = 0
    # recall that u(j) is the jth columo of U
    # To do projection: sigma(j = 1 to number of columns of U)
    #  -> j(th) column of u(transpose) * image * j(th) column of U
    # U.shape[1] represent the number of columns of U
    for j in range(U.shape[1]):
        projection += np.dot(np.dot(U[:, j].T, image), U[:, j])

    return projection


def display_image(orig, proj):
    # Your implementation goes here!
    # Please use the format below to ensure grading consistency
    # fig, (ax1, ax2) = plt.subplots(figsize=(9,3), ncols=2)
    # return fig, ax1, ax2
    orig = orig.reshape(32,32).T
    proj = proj.reshape(32,32).T
    
    fig, (ax1, ax2) = plt.subplots(figsize=(9,3), ncols=2)

    ax1.set(title='Original')
    ax2.set(title='Projection')

    pic1 = ax1.imshow(orig, aspect='equal')
    pic2 = ax2.imshow(proj, aspect='equal')

    fig.colorbar(pic1, ax=ax1)
    fig.colorbar(pic2, ax=ax2)

    return fig, ax1, ax2


