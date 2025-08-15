from random import *

LAYER_NUMBER = 2
INPUT_NODES = 784

print(3123)
COUNTER = 784
for i in range (0, LAYER_NUMBER):
    tp = [k for k in range (i*INPUT_NODES+1, (i+1)*INPUT_NODES+1)]
    shuffle(tp)
    for x in tp:
        aus = randint((i)*INPUT_NODES+1, (i+1)*INPUT_NODES)
        while (aus == x):
            aus = randint((i)*INPUT_NODES+1, (i+1)*INPUT_NODES)
        print(x, aus)
        COUNTER += 1
    
last = [k for k in range (LAYER_NUMBER*INPUT_NODES+1, (LAYER_NUMBER+1)*INPUT_NODES+1)]
while (len(last)/2 > 10):
    nl = []
    for i in range (0, len(last)//2):
        if (i*2+1 < len(last)): print(last[i*2], last[i*2+1])
        else: print(last[i*2], randint(last[0], last[-1]+1))
        COUNTER += 1
        nl.append(COUNTER)
    last = nl

print("10")
