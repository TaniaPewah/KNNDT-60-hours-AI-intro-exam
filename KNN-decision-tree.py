import numpy as np
import pandas as pd
from math import sqrt
from matplotlib import pyplot as plt
import random
import copy
eps = np.finfo(float).eps



def calc_ent_ei(data, attribute_index, value_of_attr):

    temp_data = np.array(list(filter(lambda row: row[attribute_index] == value_of_attr, data.transpose())))
    return find_entropy(temp_data.transpose())

def calc_mistake_ei(data, attribute_index, value_of_attr, K, cache, prev_values):

    transposed = copy.deepcopy(data.transpose())
    copy_data = copy.deepcopy(data)
    temp_data = np.array(list(filter(lambda row: row[attribute_index] == value_of_attr, transposed)))
    copy_data[attribute_index] = prev_values

    for temp_row in temp_data:
        prev_val = np.array(list(filter(lambda row: row[0] == temp_row[0], copy_data.transpose())))[0][attribute_index]
        temp_row[attribute_index] = prev_val

    return get_eval_mistake(temp_data, K, cache)

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

def find_mistake_of_attribute_with_threshold(data, attribute_index, poss_limit, K, cache):
    target_variables = np.unique(data[-1])  # This gives all 1 and 0
    # save attribute values for later
    prev_values = data[attribute_index].copy()

    for sample_index in range(0, len(data[attribute_index])):
        data[attribute_index][sample_index] = 0 if poss_limit > data[attribute_index][sample_index] else 1
    # This gives different binary values of chosen attribute after classification
    variables = np.unique(data[attribute_index][1:])

    sum_entr = 0

    # for every possible value of this attr,
    for idx, value_of_attr in enumerate(variables):
        denum = np.count_nonzero(data[attribute_index] == value_of_attr)

        #temp_data = np.array(list(filter(lambda row: row[attribute_index] < poss_limit, data.transpose())))
        mistake_Ei = calc_mistake_ei(data, attribute_index, value_of_attr, K, cache,  prev_values)
        #TODO check sum_entr
        sum_entr += (denum / len(data[0])) * (1 - mistake_Ei)

    data[attribute_index] = prev_values

    return abs(sum_entr)


def calc_weigt_mistakes_for_all_thresholds_of_attr(poss_limit_values, data, column_idx, K, cache):
    weighted_mistakes =[]

    #for every possible threshold
    for lim_idx, poss_limit in enumerate(poss_limit_values):
        # calc weighted mistakes
        # current_mistake = get_eval_mistake(data.transpose(), K)
        #print("calc_weigt_mistakes_for_all_thresholds_of_attr ", column_idx, " threshold: ", poss_limit)
        children_mistake_sum = find_mistake_of_attribute_with_threshold(data, column_idx, poss_limit, K, cache)
        weighted_mistake = children_mistake_sum
        weighted_mistakes.append(weighted_mistake)
    return weighted_mistakes

def calc_weigh_mistakes(data, K, cache):
    transpose_data = data.transpose()
    final_attr_thresholds_mistakes = []

    # for every column of data (for every attr)
    # calc best threshold and IG for this threshold
    for column_idx in range(1, len(transpose_data) - 1):
        #print("checking for attr: ", column_idx)

        #poss_limit_values = np.percentile(transpose_data[column_idx], [12.5, 25, 37.5, 50, 62.5, 75, 87.5])
        poss_limit_values = np.percentile(transpose_data[column_idx], [25,  50, 75])
        res_vec = calc_weigt_mistakes_for_all_thresholds_of_attr(poss_limit_values, transpose_data, column_idx, K, cache)

        # chose max IG and the limit val according to max
        max_ig_indx, max_ig = res_vec.index(max(res_vec)), max(res_vec)  # get also index
        chosen_limit_val = poss_limit_values[ max_ig_indx]
        final_attr_thresholds_mistakes += [(column_idx, chosen_limit_val, max_ig)]

        a = sorted(final_attr_thresholds_mistakes, key=lambda item: item[-1])
        # sort the data by this attr
    return final_attr_thresholds_mistakes


#find the next attr to split the tree with
def find_winner(data, K, cache):

    #attr_IGs = calc_all_IG(data)
    attr_weighted_mistakes = calc_weigh_mistakes(data, K, cache)
    sorted_attr_weighted_mistakes = sorted(attr_weighted_mistakes, key=lambda item: item[-1])  # decreasing IG values

    if len(sorted_attr_weighted_mistakes) == 1:
        return sorted_attr_weighted_mistakes[0][0], sorted_attr_weighted_mistakes[0][1]

    best_result = sorted_attr_weighted_mistakes[-1][2]
    poss_winners = np.array(list(filter(lambda row: row[2] == best_result, sorted_attr_weighted_mistakes)))
    winner = random.choice(poss_winners)
    limit_val = winner[1]
    attr_idx_to_return = winner[0]

    return int(attr_idx_to_return), limit_val


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
    if len(neighbors) == 0:
        print("WHY AM I HERE")
        return 0
    output_values = [neighbor[-1] for neighbor in neighbors]
    prediction = max(set(output_values), key=output_values.count)
    return prediction

def get_eval_mistake(branch_data, K, cache):

    # no mistake if only one example in the group
    if len(branch_data) == 1:
        #print("only 1 exmpl here")
        return 0

    wrong_classes = 0
    for row in branch_data:
        neighbors = get_neighbors(branch_data, row, K, cache)
        class_by_knn = check_class(neighbors)
        wrong_classes += 0 if class_by_knn == row[-1] else 1

    return wrong_classes / len(branch_data)


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
    norm_data = np.zeros(shape=dataset.shape)
    norm_data[:,0] = dataset[:,0]
    for idx, row in enumerate(dataset):
        for i in range(1,len(row)):
            high = minmax[i][1]
            low = minmax[i][0]
            norm_data[idx][i] = (row[i] - low) / (high - low)
    return norm_data

# Rescale row to the range 0-1
def normalize_row(row, minmax):
    norm_row = np.zeros(len(row))
    for i in range(len(row)):
        high = minmax[i][1]
        low = minmax[i][0]
        norm_row[i] = (row[i] - low) / (high - low)
    return norm_row

def get_neighbors(data, test_row, num_neighbors, cache):
    dists = list()

    for train_row in data:
        dist = euclidean_distance(test_row, train_row, cache)
        dists.append((train_row, dist))

    dists.sort(key=lambda tup: tup[1])
    neighbors = list()

    num_neighbors = data.shape[0] - 1 if num_neighbors >= data.shape[0] else num_neighbors

    #cut myself off
    dists = dists[1:]

    for i in range(0, num_neighbors):
        neighbors.append(dists[i][0])

    return np.array(neighbors)



# calculate the Euclidean distance between two vectors
def euclidean_distance(row1, row2, distances):
    distance = 0.0
    key = str(int(row1[0])) + " " + str(int(row2[0]))

    if key in distances.keys():
        return distances[key]
    for i in range(1, len(row1)-1):
        distance += (float(row1[i]) - float(row2[i]))**2

    distance = sqrt(distance)

    key2 = str(int(row2[0])) + " " + str(int(row1[0]))
    distances[key] = distance
    distances[key2] = distance
    return sqrt(distance)

def buildTree(data, K, M, epsilon, cache):

    #TODO check its NP
    #Create a list of results for the training data example classes
    classList = [example[-1] for example in data]

    # If all training data belongs to a category, return to the category
    if classList.count(classList[0]) == len(classList):   return classList[0]

    # Get attribute with maximum information gain
    attr_indx, limit_val = find_winner(data, K, cache)
    print("winner is: ", attr_indx)

    lower, higher = classify_vals_transposed(data, int(attr_indx), limit_val)

    myTree = {attr_indx: {}}

    # if len lower or len higher == 0
    # there is no split make empty leaf
    if len(lower) == 0:
        myTree[attr_indx][0] = limit_val, None
        #print("lower: ", attr_indx, " empty leaf ")
    else:

        class_vals_l, counts_l = np.unique(lower[:,-1], return_counts=True)

        mistake = get_eval_mistake(lower, K, cache)
        # if all lower are same class - its a leaf, save all lower examples in the leaf
        if len(class_vals_l) == 1 or mistake <= epsilon or len(lower) <= M*K:
            myTree[attr_indx][0] = limit_val, lower
            print("lower: ", attr_indx, " ", len(lower), "threshold: ", limit_val, "mistake: ", mistake)

        # keep building the tree
        else:
            myTree[attr_indx][0] = limit_val, buildTree(lower, K, M, epsilon, cache)

    if len(higher) == 0:
        myTree[attr_indx][1] = limit_val, None
        #print("higher: ", attr_indx, " empty leaf ")
    else:
        class_vals_h, counts_h = np.unique(higher[:, -1], return_counts=True)

        # if all higher are same class - its a leaf, save all higher examples in the leaf
        mistake = get_eval_mistake(higher, K, cache)
        if len(class_vals_h) == 1 or mistake <= epsilon or len(higher) <= M * K:
            myTree[attr_indx][1] = limit_val, higher
            print("higher: ", attr_indx, " ", len(higher), "threshold: ", limit_val, "mistake: ", mistake)

        # keep building the tree
        else:
            myTree[attr_indx][1] = limit_val, buildTree(higher, K, M, epsilon, cache)

    return myTree


def is_leaf(tree):
    return isinstance(tree[1], np.ndarray)

def find_leaf(tree, root_key, query, K, new_cache):

    # TODO if tree is leaf do KNN class
    if tree is None:
        print("something went wrong")
        return query[47]

    lower = tree[root_key][0]

    if is_leaf(lower):
        neighbors = get_neighbors(lower[1], query, K, new_cache)
        class_by_knn = check_class(neighbors)
        return class_by_knn

    threshold = lower[0]
    if query[root_key] < threshold:
        lower_key = list(lower[1].keys())[0]
        return find_leaf(lower[1], lower_key, query, K, new_cache)
    else:
        higher = tree[root_key][1]
        if is_leaf(higher):
            neighbors = get_neighbors(higher[1], query, K, new_cache)
            class_by_knn = check_class(neighbors)
            return class_by_knn

        higher_key = list(higher[1].keys())[0]
        return find_leaf( higher[1], higher_key, query, K, new_cache)


def predict(query, tree, K, new_cache):

    return find_leaf(tree, list(tree.keys())[0], query, K, new_cache)


def get_prediction( tree, test_data, K ):

    new_cache = {}
    prediction = []

    for idx, row in enumerate(test_data):
        predicted = predict(row, tree, K, new_cache)
        prediction += [predicted]

    return prediction


def calc_accuracy( tree, test_data, K):
    correct_pred_sum = 0

    predictions = get_prediction( tree, test_data, K )
    # TODO check if predictions and test data are of the same size

    for idx, row in enumerate(test_data):
        correct_pred_sum += 1 if row[-1] == predictions[idx] else 0

    accuracy = correct_pred_sum / len(test_data)
    return accuracy


def KNN_build_and_test( train, test, K, M, epsilon):
    train = train.to_numpy()
    test = test.to_numpy()
    tree = buildTree(train, K, M, epsilon, {})
    return calc_accuracy(tree, test, K)

def experiments():
    Ms = [1, 2, 3, 4, 5]
    default_K = 4
    default_M = 2
    default_eps = 0.01
    accuracies = []
    sizes = [1, 3, 9, 27]

    train = pd.read_csv('train_9.csv')
    test_df = pd.read_csv('test_9.csv')

    for m in Ms:

        accuracy = KNN_build_and_test(train, test_df, default_K, m, default_eps)
        accuracies.append(accuracy)
        print(m, ' M accuracy:', accuracy)

    # Plot results:
    plt.plot(Ms, accuracies)
    plt.title('M Accuracies')
    plt.xlabel('M')
    plt.ylabel('Accuracies(%)')
    plt.show()




def calc_accuracy_of_forest( results, test_data ):
    # calc majority of votes fore every test item,
    # calc accuracy
    correct_pred_sum = 0

    sum_of_results = np.array(results).sum(axis=0)
    for idx, row in enumerate(test_data):
        prediction = 1 if sum_of_results[idx] / len(results) > 0.5 else 0
        correct_pred_sum += 1 if row[-1] == prediction else 0

    accuracy = correct_pred_sum / len(test_data)
    return accuracy

def forest_comitee1(train, test):

    train = train.to_numpy()
    test = test.to_numpy()
    params = [(4,2,0.01), (4,2,0.01), (4,2,0.01), (4,2,0.01), (4,2,0.01), (4,2,0.01), (4,2,0.01)]
    results = []
    for param_set in params:
        K = param_set[0]
        M = param_set[1]
        epsilon = param_set[2]
        tree = buildTree(train, K, M, epsilon, {})
        results += [get_prediction(tree, test, K)]

    #TODO check results are as expected
    accuracy = calc_accuracy_of_forest( results, test )
    print(accuracy)
    return accuracy


train = pd.read_csv('train_9.csv')
test_df = pd.read_csv('test_9.csv')
#experiments()
forest_comitee1(train, test_df)