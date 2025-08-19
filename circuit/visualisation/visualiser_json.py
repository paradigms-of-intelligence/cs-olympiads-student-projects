import sys
import json

if not __name__ == "__main__": exit(0)
gate16 = False

def bytetoint(bytes) -> int:
    return int.from_bytes(bytes, "little")


size = int(sys.stdin.readline())

nodeid = 785
edges = []

for line in sys.stdin:
    nums = line.split(' ')
    if(len(nums) != 2): break
    a = int(nums[0])
    b = int(nums[1])
    if a == -1 and b == -1: continue
    edges.append({"from":str(a), "to":str(nodeid)})
    edges.append({"from":str(b), "to":str(nodeid)})
    nodeid += 1

nodes = []
for i in range(nodeid):
    if(i != 0): nodes.append({"id":str(i), "label":str(i)})

output = {
    "kind": {"graph": True},
    "nodes": nodes,
    "edges": edges,
}

print(json.dumps(output))
