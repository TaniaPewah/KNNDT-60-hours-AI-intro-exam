from math import log
import numpy as np
import pandas as pd
from math import sqrt
eps = np.finfo(float).eps


def calc_ent_ei(data, attribute_index, value_of_attr):

    temp_data = np.array(list(filter(lambda row: row[attribute_index] == value_of_attr, data.transpose())))
    return find_entropy(temp_data.transpose())

def find_entropy_of_attribute_with_threshold(data, attribute_index, poss_limit):
    target_variables = np.unique(data[-1])  # This gives all 1 and 0
    # save attribute values for later
    prev_values = data[attribute_index].copy()

    for sample_index in range(0, len(data[attribute_index])):
        data[attribute_index][sample_index] = 0 if poss_limit > data[attribute_index][sample_index] else 1
    # This gives different binary values of chosen attribute after classification
    variables = np.unique(data[attribute_index][1:])

    sum_entr = 0

    # foe every possible value of this attr, calc its entropy Ei and put in entropies_of_sons
    for idx, value_of_attr in enumerate(variables):
        denum = np.count_nonzero(data[attribute_index] == value_of_attr)

        ent_Ei = calc_ent_ei(data, attribute_index, value_of_attr)
        sum_entr += (denum / len(data[0])) * ent_Ei

    data[attribute_index] = prev_values

    return abs(sum_entr)

def find_entropy(data):

    entropy = 0
    values = np.unique(data[-1])

    for value in values:
        fraction = np.count_nonzero(data[-1] == value) / len(data[-1])
        entropy += -fraction * np.log2(fraction)
    return entropy

def find_averages_for_attr(sort_arr):
    poss_limit_values = []
    for idx, val in enumerate(sort_arr):
        if idx == len(sort_arr) - 1:
            continue
        poss_limit_values += [(val + sort_arr[idx + 1]) / 2]

    return poss_limit_values


def calc_IG_all_limit_vals_of_attr(poss_limit_values, data, column_idx):
    ig_vec =[]

    #for every possible threshold
    for lim_idx, poss_limit in enumerate(poss_limit_values):
        # calc IG = according to slide number 70(?)
        all_ent = find_entropy(data)
        other_ent = find_entropy_of_attribute_with_threshold(data, column_idx, poss_limit)
        ig = all_ent - other_ent
        ig_vec.append(ig)
    return ig_vec

def calc_all_IG(data):
    transpose_data = data.transpose()
    final_attr_and_limit_vals = []

    # for every column of data (for every attr)
    # calc best threshold and IG for this threshold
    for column_idx in range(1,len(transpose_data)-1):

        sort_arr = sorted(transpose_data[column_idx])

        # find k-1 averages (limit values) for every i, i+1 values
        poss_limit_values = find_averages_for_attr(sort_arr)
        ig_vec = calc_IG_all_limit_vals_of_attr(poss_limit_values, transpose_data, column_idx)

        # chose max IG and the limit val according to max
        max_ig_indx, max_ig = ig_vec.index(max(ig_vec)), max(ig_vec)  # get also index
        chosen_limit_val = poss_limit_values[max_ig_indx]  # should insert the best limit val corresponding to the max ig
        final_attr_and_limit_vals += [(column_idx, chosen_limit_val, max_ig)]

        # sort the data by this attr
    return final_attr_and_limit_vals

#find the next attr to split the tree with
def find_winner(data):

    attr_IGs = calc_all_IG(data)
    sorted_attr_IGs = sorted(attr_IGs, key=lambda item: item[-1])  # decreasing IG values

    if len(sorted_attr_IGs) == 1:
        return sorted_attr_IGs[0][0], sorted_attr_IGs[0][1]

    attr_idx_to_return = sorted_attr_IGs[-1][0]
    limit_val = sorted_attr_IGs[-1][1]

    return attr_idx_to_return, limit_val


def classify_vals_transposed(data, attr_index, limit_val):
    lower = []
    higher = []
    # for every row
    for index, row in enumerate(data):
        if row[attr_index] < limit_val:
            lower.append(row)
        else:
            higher.append(row)
    return np.array(lower), np.array(higher)

def check_class(neighbors):
    output_values = [neighbor[-1] for neighbor in neighbors]
    prediction = max(set(output_values), key=output_values.count)
    return prediction

def epsilon_range_mistake( branch_data, epsilon, K ):
    wrong_classes = 0
    for row in branch_data:
        neighbors = get_neighbors(branch_data, row, K)
        class_by_knn = check_class(neighbors)
        wrong_classes += 0 if class_by_knn == row[-1] else 1

    return (wrong_classes / len(branch_data)) < epsilon


# Find the min and max values for each column
def dataset_minmax(dataset):
    minmax = list()
    for i in range(len(dataset[0])):
        col_values = [row[i] for row in dataset]
        value_min = min(col_values)
        value_max = max(col_values)
        minmax.append([value_min, value_max])
    # returns a list of min and max vals for each attr
    return minmax

# Rescale dataset columns to the range 0-1
def normalize_dataset(dataset, minmax):
    norm_data =[]
    for row in dataset:
        for i in range(len(row)):
            high = minmax[i][1]
            low = minmax[i][0]
            norm_data[row][i] = (row[i] - low) / (high - low)
    return norm_data


def get_neighbors(data, test_row, num_neighbors):
    distances = list()
    minmax_list = dataset_minmax(data)
    normalized = normalize_dataset(data, minmax_list)

    for train_row in data:
        dist = euclidean_distance(test_row, train_row)
        distances.append((train_row, dist))

    distances.sort(key=lambda tup: tup[1])
    neighbors = list()

    for i in range(num_neighbors):
        neighbors.append(distances[i][0])

    return neighbors

# calculate the Euclidean distance between two vectors
def euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(1, len(row1)-1):
        distance += (float(row1[i]) - float(row2[i]))**2
    return sqrt(distance)


def buildTree(data, K, M, epsilon):

    #TODO check its NP
    #Create a list of results for the training data example classes
    classList = [example[-1] for example in data]

    # If all training data belongs to a category, return to the category
    if classList.count(classList[0]) == len(classList):   return classList[0]

    # Get attribute with maximum information gain
    attr_indx, limit_val = find_winner(data)

    lower, higher = classify_vals_transposed(data, attr_indx, limit_val)

    myTree = {attr_indx: {}}

    # if len lower or len higher == 0
    # there is no split make empty leaf
    if len(lower) == 0:
        myTree[attr_indx][0] = limit_val, None
        print("lower: ", attr_indx, " empty leaf ")
    else:

        class_vals_l, counts_l = np.unique(lower[:,-1], return_counts=True)

        # if all lower are same class - its a leaf, save all lower examples in the leaf
        if len(class_vals_l) == 1 or epsilon_range_mistake(lower, epsilon, K) or len(lower) <= M*K:
            myTree[attr_indx][0] = limit_val, lower
            print("lower: ", attr_indx, " ", len(lower))

        # keep building the tree
        else:
            myTree[attr_indx][0] = limit_val, buildTree(lower, K, M, epsilon)

    if len(higher) == 0:
        myTree[attr_indx][1] = limit_val, None
        print("higher: ", attr_indx, " empty leaf ")
    else:
        class_vals_h, counts_h = np.unique(higher[:, -1], return_counts=True)

        # if all higher are same class - its a leaf, save all higher examples in the leaf
        if len(class_vals_h) == 1 or epsilon_range_mistake(higher, epsilon, K) or len(higher) <= M * K:
            myTree[attr_indx][1] = limit_val, higher
            print("higher: ", attr_indx, " ", len(higher))

        # keep building the tree
        else:
            myTree[attr_indx][1] = limit_val, buildTree(higher, K, M, epsilon)

    return myTree


df = pd.read_csv('train_9.csv')
df_array = df.to_numpy()

#entropy = find_entropy(df_array)
tree = buildTree(df_array, K, M, epsilon )

print(tree)
print(tree)