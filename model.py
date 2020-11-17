#!/usr/bin/python3
# -*- coding: utf-8 -*-

""" Implementation of ID3 Decision Tree
""" 
import numpy as np, sys



# Create a Class of the Tree
class id3:
    
    def __init__(self, data = None):
        """ Create an instance of the Tree, shape of the data
            and the classification
        """
        if data is not None:
            self._data = data
            self._classEntropy = self.Entropy(self._data)
            self._class = self.probability_distribution(self._data, -1).keys()
            self._Tree = None
            self._shape = data.shape[1] if data.ndim>1 else len(data)
       

    def Entropy(self, data):
        """ Calculate and return the Entropy value
        """
        entropy = 0
        probability = self.probability_distribution(data, -1)
        for value in probability.values():
            entropy += (value * (-np.log2(value)))

        return entropy 
    



    def probability_distribution(self, array, index):
        """ Calculate the probability distribution of the Array using the column index
        """
      
        total = array.shape[0] # Total number of element in the column
        distinct = {}
        
        if array.ndim == 1:
            distinct[array[index]] = 1
            return distinct
        
        # Iterate over rows in the data
        for row in array:
            # Determine if the element exist and increment by 1
            if row[index] in distinct:
                distinct[row[index]] += 1
            # If element does not exist, count element by 1
            else:
                distinct[row[index]] = 1
        # Caluculate probability per element in the column index
        for key, value in distinct.items():
            distinct[key] = value/total
            
        return distinct
    

    def sort(self, data, index):
        """ Sort data with a given index and return sorted data"""
        return data[np.argsort(data[:,index],kind='mergesort')]

    def split(self, data, index):
        """ Split data at a given index,
            Iterate over the data when the elemnet at the index is the same as the element before it.
            Split the data into two when the elements are different
        """
        data = self.sort(data, index) # Sort data
        n = 0
        before = data[n-1, index].item() if n > 0 else data[n, index].item() # Get the element before
        while n < data.shape[0] and data[n, index].item() == before:
            before = data[n-1, index].item() if n > 0 else data[n, index].item()
            n += 1
        
        if n < data.shape[0]:
            """
                Determine if all rows are not iterated
            """
            avg = (data[n-1, index].item() + data[n, index].item()) / 2
            return avg, data[ :n, :] , data[n: data.shape[0], :] # Return the split datas

        # If all datas are similar return the data and the splitting data
        return data[n-1: data.shape[0], index].item()+1, data 

             
    def Gain(self, data, index):
        """ Calulate and return the entropy of each different group of datas in a column index 
        """
        splitted = self.split(data, index) # Split data
        
        entropy = 0
        if len(splitted) > 2: # Determine the data was splitted
            _, left, right = splitted
            total = left.shape[0] + right.shape[0]
            entropy += self.Entropy(left) * left.shape[0]/total # Calculate Entropy
            entropy += self.Entropy(right) * right.shape[0]/total 
            return entropy 
        else:
            return self.Entropy(splitted[1])


    def find(self, data):
        """ Calculate individual Information gain and return the maximum gain
        """
        maxGain = -1
        gain = -1
        for i in range(data.shape[1]-1):
            newGain = self.Entropy(data) - self.Gain(data, i)
            if newGain > gain:
                maxGain = i
                gain = newGain


        return maxGain



    def train_model(self, data = None):
        """
            Build Decision Tree
        """
        if data is not None:
            self._data = data
            self._classEntropy = self.Entropy(self._data)
            self._class = self.probability_distribution(self._data, -1).keys()
            self._Tree = None
            self._shape = data.shape[1] if data.ndim>1 else len(data)

        self._Tree = Tree()
        self.recTree(self._Tree, self._data)
        return self._Tree


    
    def recTree(self, tree, data):
        """ Helper Recursion method for Building the Tree
        """
        if self.Entropy(data) == 0:
            tree.decision = data[0, -1] if data.ndim >1 else data[-1]
            return 

        else:
            tree.column = self.find(data) # Save Column to root
            splitted = self.split(data, tree.column)
            tree.value = splitted[0] # Save splitting value
            tree.less = Tree() # Create a leaf node for element less than the spliting value
            
            self.recTree(tree.less, splitted[1])
            if len(splitted) > 2:
                tree.greater = Tree() # Create a leaf node for element greater than the spliting value
                self.recTree(tree.greater, splitted[2])

     
    def makeDecision(self, data):
        """ Classify data based on decision tree
        """
        assert data.shape[0] == self._shape or data.shape[0] == self._shape-1,\
            "Data not in Shape"
        return self.recTest(self._Tree, data)


    def recTest(self, tree, data):
        """ Helper Recursion method for Classifying the data
        """
        if tree.decision is not None:
            return tree.decision
        else:
            if data[tree.column] < tree.value:
                return self.recTest(tree.less, data)
            else:               
                return self.recTest(tree.greater, data)

    
    def iterate(self, tree):
        if tree.decision is not None:
            print(tree.decision)
            return 
        else:
            print(tree.value, tree.column)
            if tree.less is not None:
                self.iterate(tree.less)
            
            if tree.greater is not None:
                self.iterate(tree.greater)


        
# Tree Node
class Tree:
    """ Create an instance of the Tree
    """
    def __init__(self):
        self.column = None
        self.value = None
        self.less = None
        self.greater = None
        self.decision = None

