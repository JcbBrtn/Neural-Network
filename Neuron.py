import random
import math
import numpy as np

class Neuron:
    """
    the inputs are recieved with masses already multiplied...
    the outputs must be given as an array,
    each output has its own mass it must multiply by to send to next neuron

    the Zeroth neuron is the user

    Each Neuron is given a Neuron Number.
    inp is an Array of all of the neuron numbers of the desired inputs
    out is an Array of all of the neuron numbers of the desired outputs
    """

    def __init__(self, num, inp, out):
        
        self.num = num
        self.inpDic = {}
        self.outDic = {}
        self.masses = []
        for i in inp:
            self.addInput(i)
        for o in out:
            self.addOutput(o)
        self.bias = random.uniform(-1, 1)
        self.preActivation = 0
        self.activation = 0
        self.error = 0
        self.learningRate = random.random()


    def sigmoid(self):
        #This will be the normilization function
        #sets the activation, returns nothing
        self.activation = round((1/(1+ np.exp(-1*self.preActivation))), 6)
        return

    def updateInpDic(self, inputs):
        self.inpDic = inputs

    def run(self):
        #from all inputs gets the neurons activation then gives outputs
        self.preActivation = 0
        for inp in self.inpDic.values():
            self.preActivation += inp
        self.preActivation += self.bias
        self.sigmoid()
        #Neuron now has an activation
        #Now update dictionary of outputs.
        for count, out in enumerate(self.outDic.keys()):
            output = self.activation * self.masses[count]
            self.outDic.setdefault(out, output)
            
    def addInput(self, inpNum):
        self.inpDic.setdefault(inpNum, 0.0)

    def addOutput(self, outNum):
        self.outDic.setdefault(outNum, 0.0)
        self.masses.append(random.uniform(-1,1))

    def mutate(self):
        for mass in self.masses:
            mass += random.uniform(-1,1) * self.learningRate
        self.bias += random.uniform(-1,1) * self.learningRate

    def popInput(self, num):
        return self.inpDic.pop(num)
        
    def popOut(self, num):
        for massNum, outNum in enumerate(self.outDic.keys()):
            if outNum == num:
                self.masses.remove(massNum)
                return self.outDic.pop(num)
            else:
                continue
    
    def getInputNums(self):
        return self.inpDic.keys()

    def getOutput(self, num):
        try:
            return self.inpDic[num]
        except:
            return 0
