import random
import Neuron

class NeuralNetwork:
    def __init__(self):
        self.neuronCounter = 1
        #network[0] is the input layer
        #network[-1] is the output layer
        self.network = [[],[],[]]
        self.neurons = []
        self.learningRate = random.random()

        #Start the neural Network with 1 input linked to 1 output
        self.addNeuron(0)
        print(self.neuronCounter)
        self.addNeuron(2)
        print(self.neuronCounter)
        self.link(1,2)

    def run(self, inpArr):
        for layer in self.network:        
            for count, neuron in enumerate(layer):
                #get and update inputs to each neuron before firing that neuron
                inputs = {}
                for inpNum in neuron.getInputNums():
                    if inpNum == 0:
                        inp = inpArr[count]
                    else:
                        inp = self.neurons[inpNum - 1].getOutput(neuron.num)

                    inputs.setdefault(inpNum, inp)
                neuron.updateInpDic(inputs)
                neuron.run()
                self.neurons[neuron.num - 1] = neuron

        self.updateNetwork()
        return self.getOutputs()

    def updateNetwork(self):
        for layerNum in range(len(self.network)):
            for neuronNum in range(len(self.network[layerNum])):
                self.network[layerNum][neuronNum] = self.neurons[self.network[layerNum][neuronNum].num - 1]
        return self.network
    
    def getOutputs(self):
        output = []
        for neuron in self.network[-1]:
            output.append(neuron.getOutput(0))
        return output

    def addNeuron(self, layerNum):
        if layerNum == 0:
            self.neurons.append(Neuron.Neuron(self.neuronCounter,[0],[]))
        elif layerNum == 1:
            self.neurons.append(Neuron.Neuron(self.neuronCounter,[],[]))
        else:
            self.neurons.append(Neuron.Neuron(self.neuronCounter,[],[0]))
            
        self.network[layerNum].append(self.neurons[self.neuronCounter - 1])
        self.neuronCounter += 1
        return self.updateNetwork()

    def link(self, beginNum, endNum):
        self.neurons[beginNum - 1].addOutput(endNum)
        self.neurons[endNum - 1].addInput(beginNum)
        return self.updateNetwork()

    def cut(self, beginNum, endNum):
        self.neurons[beginNum - 1].popOut(endNum)
        self.neurons[endNum - 1].popOut(beginNum)
        return self.updateNetwork()
                
    def mutate(self):
        """
        goes through each neuron, each neuron has a chance of mutating
        equal to the learning rate of the network.
        There is a 20% chance of a physical mutation.
        A physical mutation is:
            Adding a neuron
                Can add to input, hidden or Output layer, determined randomly
            Linking a neuron
                links are determined randomly, any neuron can link to anyother neuron
            Cutting a neuron link
                Cuts are determined randomly
        """
        #First, mutate masses
        for neuronNum in range(self.neuronCounter - 1):
            if self.learningRate > random.random():
                self.neurons[neuronNum].mutate()
            else:
                continue

        #Now determine physical mutations
        if random.random() < 0.2:
            try:
                physMutation = random.choice(['a','l','c'])
                if physMutation == 'a':
                    self.addNeuron(random.choice([0,1,2]))
                elif physMutation == 'l':
                    begin = random.randint(1,self.neuronCounter - 1)
                    end = random.randint(1, self.neuronCounter - 1)
                    self.link(begin, end)
                else:
                    begin = random.randint(1,self.neuronCounter - 1)
                    end = random.choice(self.neurons[begin].outDic.keys())
                    self.cut(begin, end)
            except:
                return self
        return self




