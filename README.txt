#param.py at the bottom of the file run the tests for ranges of params:
train = pd.read_csv('train_9.csv')
test_df = pd.read_csv('test_9.csv')

experimentsMs(train, test_df) # אפשר לערוך את הפרמטרים בפנים
experimentsKs(train, test_df)

train = pd.read_csv('train_12.csv')
test_df = pd.read_csv('test_12.csv')

experimentsMs(train, test_df)
experimentsKs(train, test_df)

# הפונקציות המרכזיות:
accuracy = KNN_build_and_test(train, test_df, default_K, m, default_eps, cache)
tree = buildTree(train, K, M, epsilon, cache)

# Get attribute with maximum accuracy gain
attr_indx, limit_val = find_winner(data, K, cache)

# למצוא דיוק ממושקל לכל התכונות
attr_weighted_mistakes = calc_weigh_mistakes(data, K, cache)

# למצוא דיוק ממוקשל לכל ערך סף אפשרי של תכונה מסויימת
res_vec = calc_weigt_mistakes_for_all_thresholds_of_attr(poss_limit_values, transpose_data, column_idx, K, cache)

# למצוא דיוק ממושקל
children_mistake_sum = find_mistake_of_attribute_with_threshold(data, column_idx, poss_limit, K, cache)

#for all values after possible split of data calculate mistake of possible child:
mistake_Ei = calc_mistake_ei(data, attribute_index, value_of_attr, K, cache,  prev_values)

#calc the mistake in leaf according to the requirements
get_eval_mistake(temp_data, K, cache)

# then splitting the data to lower and higher and if not leaf countinue building the tree with lower data or higher data
___________________________________________________
#exp.py: at the buttom:

train = pd.read_csv('train_9.csv')
test = pd.read_csv('test_9.csv')

run_increasing_test_folds(train, test)

train = pd.read_csv('train_12.csv')
test = pd.read_csv('test_12.csv')

run_increasing_test_folds(train, test)
_____________________________________________
#improvements.py: at the buttom:

train = pd.read_csv('train_9.csv')
test_df = pd.read_csv('test_9.csv')

#כל העצים אותו הדבר מחזיר את הדיוק לפי החלטת הרוב
forest_comitee1(train, test_df)

# העצים שונים מחזיר את החלטת הרוב
forest_comitee2(train, test_df)

train = pd.read_csv('train_12.csv')
test_df = pd.read_csv('test_12.csv')

forest_comitee1(train, test_df)
forest_comitee2(train, test_df)