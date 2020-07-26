from math import log
import numpy as np
import pandas as pd
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
        else :
            higher.append(row)
    return np.array(lower), np.array(higher)


def get_subtable(data, attr_index, value):
    col_names = data[0]
    returned = data[data[:,attr_index] == value]
    returned = np.vstack((col_names, returned))
    return returned

def buildTree(data, tree=None):

    #TODO check its NP
    #Create a list of results for the training data example classes
    classList = [example[-1] for example in data]

    # If all training data belongs to a category, return to the category
    if classList.count(classList[0]) == len(classList):   return classList[0]


    # Get attribute with maximum information gain
    attr_indx, limit_val = find_winner(data)

    lower, higher = classify_vals_transposed(data, attr_indx, limit_val)

    # feat_values = [example[attr_indx] for example in data]
    # unique_attr_vals = set(feat_values)
    # Get distinct value of that attribute e.g Salary is node and Low, Med and High are values
    #attValues = np.unique(transposed[attr_indx])

    # Create an empty dictionary to create tree
    # if tree is None:
    #     tree = {}
    #     tree[attr_indx] = {}

    myTree = {attr_indx: {}}

    # if len lower or len higher == 0
    # there is no split make empty leaf
    if len(lower) == 0:
        myTree[attr_indx][0] = limit_val, None
        print("lower: ", attr_indx, " empty leaf ")
    else:
        class_vals_l, counts_l = np.unique(lower[:,-1], return_counts=True)

        # if all lower are same class - its a leaf, save all lower examples in the leaf
        if len(class_vals_l) == 1:
            myTree[attr_indx][0] = limit_val, lower
            print("lower: ", attr_indx, " ", len(lower))

        # keep building the tree
        else:
            myTree[attr_indx][0] = limit_val, buildTree(lower)

    if len(higher) == 0:
        myTree[attr_indx][1] = limit_val, None
        print("higher: ", attr_indx, " empty leaf ")
    else:
        class_vals_h, counts_h = np.unique(higher[:, -1], return_counts=True)

        # if all higher are same class - its a leaf, save all higher examples in the leaf
        if len(class_vals_h) == 1:
            myTree[attr_indx][1] = limit_val, higher
            print("higher: ", attr_indx, " ", len(higher))

        # keep building the tree
        else:
            myTree[attr_indx][1] = limit_val, buildTree(higher)

    return myTree











    # We make loop to construct a tree by calling this function recursively.
    # In this we check if the subset is pure and stops if it is pure.
    # print("bla---------------------")
    # for value in unique_attr_vals:
    #
    #    #root = DTNode(parent, p, n, 'testNode', attributeName=attribute, attributeIndex=attrIndex)
    #
    #     subtable = get_subtable(data, attr_indx, value)
    #     clValue, counts = np.unique(subtable.transpose()[0][1:], return_counts=True)
    #
    #     majority = 0
    #     # TODO check leaf creation
    #     if( majority ):
    #         zero_diagnosis = counts[int(clValue[0])]
    #         other_diagnosis = counts[int(clValue[1])]
    #         if (zero_diagnosis > other_diagnosis):
    #             majority = clValue[0]
    #             myTree[attr_indx][value] = limit_val, majority
    #         else:
    #             majority = clValue[1]
    #             myTree[attr_indx][value] = limit_val, majority
    #         return tree
    #
    #     if len(counts) == 1:   # Checking purity of subset
    #         myTree[attr_indx][value] = limit_val, clValue[0]
    #         print("creating leaf")
    #     else:
    #         print("building tree")
    #         #add coulumn names to subtable
    #         zero_diagnosis = counts[int(clValue[0])]
    #         other_diagnosis = counts[int(clValue[1])]
    #         if (zero_diagnosis > other_diagnosis):
    #             majority = clValue[0]
    #         else:
    #             majority = clValue[1]
    #
    #         myTree[attr_indx][value] = limit_val, majority, buildTree(subtable)  # Calling the function recursively





df = pd.read_csv('train_9.csv')
df_array = df.to_numpy()

#entropy = find_entropy(df_array)
tree = buildTree(df_array )

print(tree)
print(tree)