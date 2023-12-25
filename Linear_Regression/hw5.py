# Name: Hyeonmin Yang
import sys
import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    # Q1 
    # Did data curation manually

    # Q2 - Visualize Data
    # first we need to open the data with sys.argv[1] (this will contain year and days)
    mendota_ice_data = open(sys.argv[1], "r")
    # then we need to jump fist line because this is name of data
    mendota_ice_data.readline()
    # then make a list for year and data, we need to assign each arrays size for later interation
    # but we can make just one size variable because they are in same csv so, both size will be the same

    year = []
    days = []
    # iterate each data and append to each arrays
    for line in mendota_ice_data:
        # first split year and days with comma
        (year_data  , days_data) = line.split(',')
        # then Append year and days to each arrays and cast it to int because this is string
        year.append(int(year_data))
        days.append(int(days_data))


        
    # close the mendota_ice_data
    mendota_ice_data.close()

    # then we need to plot the data
    plt.plot(year, days)
    # label the flot for x-axis
    plt.xlabel('Year')
    # label the flot for y-axis
    plt.ylabel('Number fo Frozen Days')
    #save the graph to plot.jpg
    plt.savefig('plot.jpg')



    #Q3a
    # first we need to make Year data to [[1,x1],[1,x2],[1,x3]...[1,xn]] 
    # this will vertical way
    # make a new list that will contain [1,x1],[1,x2],[1,x3]...[1,xn]
    xi = []
    # Then with for-loop append [1, xi] to list xi
    for year_index in range(len(year)):
        xi.append([1, year[year_index]]) 
    # then make this list to np array with dtype : int64
    X = np.array(xi, dtype = 'int64')
    # print the result
    print("Q3a:")
    print(X)

    #Q3b
    # first we need to make Y data to [[y1],[y2],[y3]...[yn]]
    # make a new list that will contain(y1),(y2),(y3)...(yn)
    yi = []
    # then with for-loop append i th y to yi list
    for year_index in range(len(days)):
        yi.append(days[year_index])
    # then make this list to np array with dtype : int64
    Y = np.array(yi, dtype = 'int64')
    print("Q3b:")
    print(Y)

    #Q3c
    # assign Z with X^2 -> X (tanspose) * X
    Z = np.dot(X.T, X)
    # X's dtype is aleady int64 so we don't need to cast it
    print("Q3c:")
    print(Z)

    #Q3d
    # To get inverse matrix of Z, we need to use np.linalg.inv
    # I = (x^T * x)^-1
    I = np.linalg.inv(Z)
    print("Q3d:")
    print(I)


    #Q3e
    # PI = (X^T * X)^-1 * X^T 
    # And I is (X^T * X)^-1 so, this will be I * X^T
    PI = np.dot(I, X.T)
    print("Q3e:")
    print(PI)

    #Q3f
    # estimate_beta = (X^T * X)^-1 * X^T * Y
    # And PI is (X^T * X)^-1 * X^T so, this will be PI * Y
    estimate_beta = np.dot(PI, Y)
    print("Q3f:")
    print(estimate_beta)

    #Q4
    # y_test = estimate_beta[0] + estimate_beta[1] * x_test and x_test is 2022
    y_test = estimate_beta[0] + np.dot(estimate_beta[1],2022)
    print("Q4: " +  str(y_test))


    #Q5
    #Q5a
    # We need to make codition statement for sign of estaimate_beta[1]
    # if estimate_beta[1] is positive : symbol = ">"
    if estimate_beta[1] > 0:
        symbol = ">"
    # if estimate_beta[1] is negative : symbol = "<"
    elif estimate_beta[1] < 0:
        symbol = "<"
    # if estimate_beta[1] is zero : symbol = "="
    else:
        symbol = "="

    print("Q5a: " + symbol)
    #Q5b
    print("Q5b: # of frozen day is increase: >  | # of frozen day is decrease: <  |  # of frozen day is same: = ")

    #Q6
    # 0 = estimate_beta[0] + estimate_beta[1] * x^* 
    # -> estimate_beta[1] * x^*  = -estimate_beta[0]
    # -> x^* = -estimate_beta[0] / estimate_beta[1]
    x = -estimate_beta[0] / estimate_beta[1]
    #Q6a
    print("Q6a: " + str(x))
    #Q6b
    print("Q6b: I think this is not perfect compelling prediction, beacuse the plots on graph are so scattered. but with linear regression, the frozen day in graph is decreasing so it can be seen little bit compelling prediction.")
