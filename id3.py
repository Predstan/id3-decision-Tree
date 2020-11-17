#!/usr/bin/python3
# -*- coding: utf-8 -*-

""" Implementation of ID3 Decision Tree
""" 


import numpy as np
import pandas as pd, sys
from model import id3



def main():
    if (len(sys.argv) != 3):
        print()
        print("Usage: %s [training Data] [Test Data]" %(sys.argv[0]))
        print()
        sys.exit(1)
  
    train, test = np.loadtxt(sys.argv[1]), np.loadtxt(sys.argv[2])
    model = id3()
    model.train_model(train)
    correct = 0
    if test.ndim < 2:
        classification = model.makeDecision(test)
        print(classification, test[-1])
        if test[-1] == classification:
            correct+=1
    else:
        for data in test:
            classification = model.makeDecision(data)
            print(classification, data[-1])
            if data[-1] == classification:
                correct+=1
            
    print(correct)


if __name__ == "__main__":
    main()

