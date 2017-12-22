import sys
import csv
from itertools import chain, combinations
import math
from collections import defaultdict, Counter
from operator import itemgetter
import random

def extract_data_from_file(input_file):
    reader = csv.reader(open(input_file), delimiter = ' ')
    data = []
    for line in reader:
        formatted = []
        label = int(line[0])
        formatted.append(label)
        attributes = line[1:]
        for attr in attributes:
            key, value = attr.split(':')
            formatted.append(int(value))
        data.append(formatted)
    return data

def set_num_trees(set_name):
    num_trees = 0
    if 'balance' in set_name:
        num_trees = 500
    elif 'nursery' in set_name:
        num_trees = 50
    elif 'led' in set_name:
        num_trees = 100
    else:
        num_trees = 100
    return num_trees

#Randomly sample with replacement for data size equivalent to data extracted. Use 63.2% ?
def bootstrap(data):
    size = int(0.632 * float(len(data)))
    bootstrapped_data = random.choices(data, k=size)
    #print(bootstrapped_data)
    return bootstrapped_data

class Node(object):
    def __init__(self):
        self.attribute = 0
        self.attribute_value = 0
        self.class_label = 0
        self.children = []
        self.isLeafNode = 0
        
    def add(self, child):
        self.children.append(child)
        
    def get_kids(self):
        return self.children
    
    def get_class_label(self):
        return self.class_label
    
    def verify_leaf(self):
        return self.isLeafNode
    
    def get_attribute(self):
        return self.attribute
    
    def get_attribute_value(self):
        return self.attribute_value
    
def create_attribute_list(data_partition):
    attribute_list = [i for i in range(1, len(data_partition[0]))]
    return attribute_list

def check_if_same_class_label(data_partition, counter):
    #print(counter, len(counter))
    if len(counter) == 1:
        return counter.most_common(1)[0][0]
    return 0

def majority_vote_class_label(data_partition, counter):
    #Returns a list of tuples, get 0 entry of first tuple for most common class label.
    return counter.most_common(1)[0][0]

#Makes a leaf node with indicated class node for decision tree.
def makeLeafNode(target_node, class_label):
    target_node.class_label = class_label
    target_node.isLeafNode = 1
    
def calc_num_values_in_attributes(data):
    num_values_in_attributes = []
    for i in range(len(data[0])):
        values = Counter([data[x][i] for x in range(len(data))])
        num_values_in_attributes.append(len(values))
        #print(values)
    #print(num_values_in_attributes)
    return num_values_in_attributes

def get_best_split_attribute(data_partition, attribute_list, num_values_in_attributes):
    #Get number of tuples in data partition and list of class labels for dataset.
    gini_attribute_splits = []
    num_data = len(data_partition)
    num_class_labels = [data_partition[x][0] for x in range(num_data)]
    
    #Random Forest modification: Random Attribute Selection. Choose a sample of n**0.5 attributes from overall list to 
    #decide to split on.
    k = math.floor(len(attribute_list) ** 0.5)
    random_attributes = random.sample(attribute_list, k)
    #print('random attribute selection: ', random_attributes)
    #random_attributes = attribute_list
    
    #Go through each individual attribute. Start at column 1, since column 0 is class labels.
    for i in random_attributes:
        gini_values = []
        attribute_value_data_count = []
        #Get all values for attribute i. Make a Counter class dictionary out of it (keys are attribute values, value is count of them.)
        attribute_values = [data_partition[x][i] for x in range(num_data)]
        attribute_value_counter = Counter(attribute_values)
        #print('attribute values:', attribute_values, 'attribute value counter: ', attribute_value_counter)
        #attribute_values = num_values_in_attributes[i - 1]
        #Go through each unique value in attribute i. 
        for value in attribute_value_counter.keys():
        #for value in range_len(1, attribute_values + 1):
            #If value matches attribute i, then append to sublist. The count of ALL values in attribute i should equal to length of dataset input.
            value_sublist = [[attribute_values[a], num_class_labels[a]] for a in range(len(attribute_values)) if attribute_values[a] == value]
            #print('length of attribute_values: ', len(attribute_values), 'length of class labels: ', len(num_class_labels))
            #print('current value in attribute: ', value, 'Values sublist:', value_sublist)
            #value_sublist = []
            count = len(value_sublist)
            #print('length of sublist: ', count)
            #Get all class labels and make counter out of them. Will use for calculating gini index.
            value_class_labels_counter = Counter([value_sublist[a][1] for a in range(count)])
            attribute_value_data_count.append(count)
            #Calculate gini index of this value for attribute i, for all possible class labels.
            local_gini_index = 1 - sum((local_label/count)**2 for local_label in value_class_labels_counter.values())
            gini_values.append(local_gini_index)
            
        gini_split = sum((attribute_value_data_count[a]/num_data) * gini_values[a] for a in range(len(attribute_value_data_count))) 
        gini_attribute_splits.append(gini_split)

    smallest_gini_value = min(gini_attribute_splits)
    index = gini_attribute_splits.index(smallest_gini_value)
    random_attribute_select = random_attributes[index]
    #print('gini attribute splits: ', gini_attribute_splits, 'index: ', index, 'randomly selected attribute: ', random_attribute_select)
    return random_attribute_select

#Creates decision tree to use on the testing set. No continuous values used, as stated in HW4, only discrete.
#Need to prune tree after generating tree in order to address overfitting issues.
def generate_decision_tree(data_partition, attribute_list, num_values_in_attributes, depth_lim = 0):
    currNode = Node()
    #currNode.attribute_list = attribute_list
    class_labels = [data_partition[x][0] for x in range(len(data_partition))]
    labels_counter = Counter(class_labels)
    #print('Current attribute list: ', attribute_list)
    #Get number of variables for each attribute, so we can use them to generate appropriate number of branches. Starts at
    #index 1, with class labels with index 0 in the list.
    #print(num_values_in_attributes)
    
    #Step 1: If class label on tuples are same, return currNode as leafnode with class label.
    class_label = check_if_same_class_label(data_partition, labels_counter)
    if(class_label != 0):
        #print('reached same class label base case')
        makeLeafNode(currNode, class_label)
        return currNode
    
    #Step 2: If attribute list is empty, then do majority vote for class label, then return currNode as leafnode with class label.
    if not attribute_list or depth_lim == 10:
        class_label = majority_vote_class_label(data_partition, labels_counter)
        #print('reached empty attribute label base case. Majority class vote label is: ', class_label)
        makeLeafNode(currNode, class_label)
        return currNode
    
    #Step 3: Use gini impurity to choose best splitting criterion.
    best_split_attr = get_best_split_attribute(data_partition, attribute_list, num_values_in_attributes)
    currNode.attribute = best_split_attr
    
    #Step 4: Delete attribute from tree, as we do not want to use it again.
    #attr_data_split = Counter([data_partition[x][best_split_attr] for x in range(len(data_partition))])
    #print('best split attribute: ', best_split_attr)
    #list_attribute_values = list(attr_data_split.keys())
    values_in_curr_attr = num_values_in_attributes[best_split_attr]
    #range_values = [i for i in range(1, values_in_curr_attr + 1)]
    #not_in_list = list(set(range_values) ^ set(list_attribute_values))
    #print(values_in_curr_attr)
    #print('range values: ', range_values)
    attribute_list.remove(best_split_attr)
    #print(attr_data_split)
    #print(best_split_attr)
    #print(attribute_list)
    
    for attr_value in range(1, values_in_curr_attr + 1):
        attr_data_partition = [data_partition[x] for x in range(len(data_partition)) if data_partition[x][best_split_attr] == attr_value]
        if(len(attr_data_partition) == 0):
            leafNode = Node()
            leafNode.isLeafNode = 1
            leafNode.attribute_value = attr_value
            leafNode.class_label = labels_counter.most_common(1)[0][0]
            currNode.add(leafNode)
        else:
            kid_attr = list(attribute_list)
            nextNode = generate_decision_tree(attr_data_partition, kid_attr, num_values_in_attributes, depth_lim+1)
            nextNode.attribute_value = attr_value
            currNode.add(nextNode)
    
    #print(vars(currNode))
    return currNode

def predict_class_label(data_tuple, currNode):
    #print(vars(currNode))
    #Base Case. If not leaf node, the no class label either.
    if currNode.verify_leaf():
        return currNode.get_class_label()
    #Recursive Case. Happens, because it is not a leaf node.
    assigned_attr = currNode.get_attribute()
    attr_value = data_tuple[assigned_attr]
    kids = currNode.get_kids()
    kids_list = [vars(kids[a]) for a in range(len(kids))]
    target_kid = 0
    for a in range(len(kids)):
        if kids_list[a]['attribute_value'] == attr_value:
            target_kid = kids[a]
    class_label = predict_class_label(data_tuple, target_kid)
    return class_label

#Get all predicted labels for data set. Use this to calculate metrics and square.
def predict_labels(data_partition, root):
    predicted_labels = []
    for i, data_tuple in enumerate(data_partition):
        #print(data_tuple)
        tuple_label = predict_class_label(data_tuple, root)
        #Each predicted label has the test set's original class label, and the predicted class label.
        predicted_labels.append([data_partition[i][0], tuple_label])
        #predicted_labels.sort(key=lambda x: x[0])
    return predicted_labels

#Use only for testing sets so we get accurate number of class labels. Looking at you, Nursery test data set.
def get_num_class_labels(data):
    class_labels = [data[x][0] for x in range(len(data))]
    labels_counter = Counter(class_labels)
    return list(labels_counter.keys())

#Fixed to not have Numpy used.
def create_confusion(data_partition, predicted_labels, num_labels):
    #class_labels = [data_partition[x][0] for x in range(len(data_partition))]
    #labels_counter = Counter(class_labels)
    confusion_matrix = [[0] * len(num_labels) for i in range(len(num_labels))] 
    for labels in predicted_labels:
        confusion_matrix[labels[0] - 1][labels[1] - 1] += 1
    for index in confusion_matrix:
        row = [str(a) for a in index]
        print(' '.join(row))
    return confusion_matrix

#Taken from bagging algorithm in book, chapter 8 page 55. Modified for random forest
#With random attribute selection in choosing best splitting criteria.
def generate_random_forest(original_data, num_trees):
    random_forest = []
    num_values_in_attributes = calc_num_values_in_attributes(original_data)
    for i in range(num_trees):
        bootstrap_data = bootstrap(original_data)
        attribute_list = create_attribute_list(bootstrap_data)
        random_forest.append(generate_decision_tree(bootstrap_data, attribute_list, num_values_in_attributes, 0))
    #print(random_forest)
    return random_forest

def rf_predict_labels(test_data, random_forest):
    all_predicted_labels = []
    rf_predicted_labels = []
    #Get ALL predicted labels for test set for random forest.
    for tree in random_forest:
        all_predicted_labels.append(predict_labels(test_data, tree))
    #Use majority voting on the labels.
    #To access every data tuple.
    #print(len(all_predicted_labels))
    #print(len(all_predicted_labels[0]))
    #print(all_predicted_labels[0][0][1])
    for j in range(len(all_predicted_labels[0])):
        #print(list(all_predicted_labels[i][j][0] for i in range(len(all_predicted_labels))))
        tuple_pred_label_list = [all_predicted_labels[i][j][1] for i in range(len(all_predicted_labels))]
        label_list_counter = Counter(tuple_pred_label_list)
        #Majority vote based on counter.
        winning_label = label_list_counter.most_common(1)[0][0]
        #print(label_list_counter)
        #print(winning_label)
        #print('majority vote: ', [all_predicted_labels[0][j][0], winning_label])
        rf_predicted_labels.append([all_predicted_labels[0][j][0], winning_label])
    return rf_predicted_labels

if __name__ == '__main__':
    test_set = sys.argv[2]
    training_set = sys.argv[1]
    data = extract_data_from_file(training_set)
    test_data = extract_data_from_file(test_set)
    num_labels = get_num_class_labels(data)
    num_trees = set_num_trees(training_set)
    random_forest = generate_random_forest(data, num_trees)
    predicted_labels = rf_predict_labels(test_data, random_forest)
    #bootstrap_data = bootstrap(data)
    #attribute_list = create_attribute_list(bootstrap_data)
    #root = generate_decision_tree(bootstrap_data, attribute_list)
    #predicted_labels = predict_labels(test_data, root)
    confusion_matrix = create_confusion(test_data, predicted_labels, num_labels)
    #print(confusion_matrix)