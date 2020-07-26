from math import log
import numpy as np
import pandas as pd
import operator

def find_entropy(data):

    entropy = 0
    transposed = data.transpose()
    values = np.unique(transposed[-1])

    for value in values:
        fraction = np.count_nonzero(transposed[-1] == value) / len(transposed[-1])
        entropy += -fraction * np.log2(fraction)
    return entropy

def calc_all_IG(data):
    return[(1,0.4)]

#find the next attr to split the tree with
def find_winner(data):

    attr_IGs = calc_all_IG(data)
    sorted_attr_IGs = sorted(attr_IGs, key=lambda item: item[1])  # decreasing IG values

    if len(sorted_attr_IGs) == 1:
        return sorted_attr_IGs[0][0], sorted_attr_IGs[0][1]

    attr_idx_to_return = sorted_attr_IGs[-1][0]
    limit_val = sorted_attr_IGs[-1][1]

    print(" winner is: ", attr_idx_to_return)
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

    #TODO check if needed
    attr_indx += 1

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
    else:
        class_vals_l, counts_l = np.unique(lower[:,-1], return_counts=True)

        # if all lower are same class - its a leaf, save all lower examples in the leaf
        if len(class_vals_l) == 1:
            myTree[attr_indx][0] = limit_val, lower

        # keep building the tree
        else:
            myTree[attr_indx][0] = limit_val, buildTree(lower)

    if len(higher) == 0:
        myTree[attr_indx][1] = limit_val, None
    else:
        class_vals_h, counts_h = np.unique(higher[:, -1], return_counts=True)

        # if all higher are same class - its a leaf, save all higher examples in the leaf
        if len(class_vals_h) == 1:
            myTree[attr_indx][1] = limit_val, higher

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

print("hello")