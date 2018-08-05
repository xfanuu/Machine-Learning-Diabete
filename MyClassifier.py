import math
import numpy as np
import operator
import sys

def createDataset(data):
    attribute = {}
    dataset = []
    for i in range(len(data)):
            attribute[i] = data[i][0:].split(',')
            dataset.append(attribute[i]) 
    return dataset
    
class NaiveBayes(object):
    
    def createTestingDataset(self, testingdata):
        attribute = {}
        testingdataset = []
        for i in range(len(testingdata)):
            attribute[i] = testingdata[i][0:].split(',')
            testingdataset.append(attribute[i]) 
        return testingdataset
    
    def createTrainingDataset(self, trainingdata):
        attribute = {}
        dataset = []
        for i in range(len(trainingdata)):
            attribute[i] = trainingdata[i][0:].split(',')
            dataset.append(attribute[i]) 
        
        trainingDatasetYes = []
        trainingDatasetNo = []       
        
        for example in dataset:
            if example[-1] == 'yes':
                trainingDatasetYes.append(example)    
            elif example[-1] == 'no':
                trainingDatasetNo.append(example)
        tempY =  [x[:] for x in trainingDatasetYes]
        for e in tempY:
            e.pop()
            trainingDatasetYes = tempY
        tempN =  [x[:] for x in trainingDatasetNo]
        for e in tempN:
            e.pop()
            trainingDatasetNo = tempN
    
        return trainingDatasetYes, trainingDatasetNo
        
    def probability_density(self, test_value, mean_value, stdDev_value):
       f = math.exp(-(test_value-mean_value)**2/(2*stdDev_value**2))/(stdDev_value*math.sqrt(2*math.pi))
       return f

    def probability(self, testingdataset, trainingdataset, e):
        np_testing = np.array(testingdataset)
        np_training = np.array(trainingdataset)
        testing = np_testing.astype('Float64')
        training = np_training.astype('Float64')
        f_pd = 1
        for attribute in range(8):
            f_pd = f_pd*self.probability_density(testing[e][attribute],np.mean(training[:,attribute]),np.std(training[:,attribute]))
        return f_pd
        
    def naive_bayes(self, testingdata, trainingdata): 
        testingdataset = self.createTestingDataset(testingdata)
        trainingdataset_yes = self.createTrainingDataset(trainingdata)[0]
        trainingdataset_no = self.createTrainingDataset(trainingdata)[1]
      
        ratioyes = len(trainingdataset_yes)/(len(trainingdata))
        rationo = len(trainingdataset_no)/(len(trainingdata))
        
        for example in range(len(testingdataset)):
            f_yes = self.probability(testingdataset, trainingdataset_yes, example)
            f_no = self.probability(testingdataset, trainingdataset_no, example)
            if f_yes*ratioyes >= f_no*rationo:
                print('yes')
            else:
                print('no')
                
                
class DecisionTree(object):
    
    def __init__(self, data):
        self.set = self.createTrainingDataset(data)
        self.dataSet = [x[:] for x in self.set]
        self.parentset = [x[:] for x in self.set] 
    
    def createTestingDataset(self, testingdata):
        attribute = {}
        testingdataset = []
        for i in range(len(testingdata)):
            attribute[i] = testingdata[i][0:].split(',')
            testingdataset.append(attribute[i]) 
        return testingdataset
    
    def createTrainingDataset(self, trainingdata):
        attribute = {}
        trainingDataset = []
        for i in range(len(trainingdata)):
                attribute[i] = trainingdata[i][0:].split(',')
                trainingDataset.append(attribute[i]) 
        return trainingDataset
    
    def createTree(self, dataset, labels):
        # extracting data

        #print(self.parentset)
        ####print('-----')
        classList = [example[-1] for example in dataset]
        if classList == []:
            classlistOfParent = [example[-1] for example in self.parentset[-1]]
            return self.majorityCnt(classlistOfParent)
            #return self.majorityCnt()
        else:
            if classList.count(classList[0]) == len(classList):
                return classList[0]  # stop splitting when all of the classes are equal
            if len(dataset[0]) == 1:  # stop splitting when there are no more features in dataset
                  return self.majorityCnt(classList) # stop splitting when there are no examples in dataset
                  
        #if len(dataset[0] == 0:
         #  return self.majorityCnt(parent_classlist)
        # use Information Gain
        bestAttribute = self.chooseBestAttribute(dataset)
        bestAttributeLabel = labels[bestAttribute]
    
        #build a tree recursively
        myTree = {bestAttributeLabel: {}}
        #print("myTree : "+labels[bestAttribute])
        del (labels[bestAttribute])
        attributeValues = [example[bestAttribute] for example in self.dataSet]
        #print("attributeValues: "+str(attributeValues))
        uniqueVals = set(attributeValues) #set(['a','a']) --> {'a'}
        #print("uniqueVals: " + str(uniqueVals))
        
        subset = []
        

        for value in uniqueVals:
            for example in self.splitDataset(dataset, bestAttribute, value):
                subset.append(example)
            
        ##print(subset)
        ##print('#####')
        self.parentset.append(subset)
        
        for value in uniqueVals:
            subLabels = labels[:]  # copy all of labels, so trees don't mess up existing labels
            #print("subLabels"+str(subLabels))
            myTree[bestAttributeLabel][value] = self.createTree(self.splitDataset(dataset, bestAttribute, value), subLabels)
            #print("myTree : " + str(myTree))
        return myTree

    def entropy(self, dataset):
        numEntries = len(dataset)
        classCounts = {}
        for example in dataset:
            class_curr = example[-1]
            if class_curr not in classCounts.keys(): classCounts[class_curr] = 0
            classCounts[class_curr] += 1
            
        ent = 0.0
        for key in classCounts:
            prob = classCounts[key] / numEntries
            #print(prob)
            ent -= prob * math.log2(prob)  # log base 2
        return ent

    def splitDataset(self, dataSet, attribute, value):
        subDataSet = []
        for example in dataSet:
            if example[attribute] == value:
                subdataset = example[:attribute]  # chop out axis used for splitting
                subdataset.extend(example[attribute+1:])
                subDataSet.append(subdataset)
        return subDataSet

    def chooseBestAttribute(self, dataset):
        parent_ent = self.entropy(dataset)
        numAttribute = len(dataset[0]) - 1
        bestInfoGain = 0.0
        bestAttribute = 0
        for i in range(numAttribute):
            AttributeList = [example[i] for example in dataset] # create a list of all the examples of this feature
            uniqueVals = set(AttributeList) # get a set of unique values
            child_ent = 0.0
            for val in uniqueVals:
                subDataset = self.splitDataset(dataset, i, val)
                prob = float(len(subDataset)) / float(len(dataset))
                child_ent += prob * self.entropy(subDataset)   
            infoGain = parent_ent - child_ent # calculate the info gain; ie reduction in entropy
            if infoGain > bestInfoGain: # compare this to the best gain so far
                bestInfoGain = infoGain # if better than current best, set to best
                bestAttribute = i 
        return bestAttribute # returns an integer 
    
    def majorityCnt(self, classList):
        classCount = {}
        classlist = list(filter(None,classList))
        for vote in classlist:
            if vote not in classCount.keys(): classCount[vote] = 0
            classCount[vote] += 1
        sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
        if sortedClassCount[0][1] == sortedClassCount[1][1]:
            return 'yes'
        else:
            return sortedClassCount[0][0]

    def classify(self, inputTree, attributeLabels, attributeValues):
        rootOfTree = list(inputTree.keys())[0]
        branch = inputTree[rootOfTree]
        attributeIndex = attributeLabels.index(rootOfTree)
        key = attributeValues[attributeIndex]
        subtree = branch[key]
        if isinstance(subtree, dict):
            classLabel = self.classify(subtree, attributeLabels, attributeValues)
        else:
            classLabel = subtree
        return classLabel
    
    def decision_tree(self, testingdata, trainingdata, labels):
        temp =  [x[:] for x in labels]
        temp2 = [x[:] for x in labels]
        testingdataset = self.createTestingDataset(testingdata)
        trainingdataset = self.createTrainingDataset(trainingdata)
        mytree = self.createTree(trainingdataset, temp)
#        print(mytree)
        for example in range(len(testingdataset)):
            ans = self.classify(mytree, temp2, testingdataset[example])
            print(ans)

        

if __name__ == "__main__":

    training = open(sys.argv[1],'r')
    trainingdata = training.read().split('\n')
    trainingdata = list(filter(None, trainingdata))
    testing= open(sys.argv[2],'r')
    testingdata = testing.read().split('\n')
    testingdata = list(filter(None, testingdata))
    classifier_type = sys.argv[3]
    if classifier_type == 'NB':
        m = NaiveBayes()
        m.naive_bayes(testingdata, trainingdata)
    if classifier_type == 'DT':
        if len(trainingdata[0].split(',')) ==  9:
            labels = ['preg', 'plas', 'pres', 'skin', 'insu', 'mass', 'pedi', 'age']   
            m = DecisionTree(trainingdata)
            m.decision_tree(testingdata, trainingdata, labels)
        if len(trainingdata[0].split(',')) ==  6:
            labels = ['plas', 'insu', 'mass', 'pedi', 'age']   
            m = DecisionTree(trainingdata)
            m.decision_tree(testingdata, trainingdata, labels)


 