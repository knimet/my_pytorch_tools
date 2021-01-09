import torch
def getIdxofSameItems(a):
#for a given vector, output the indexes of the items which have the same value
    co = a.unsqueeze(0)-a.unsqueeze(1)
    uniquer = co.unique(dim=0)
    out = []
    for r in uniquer:
        cover = torch.arange(a.size(0))
        mask = r==0
        idx = cover[mask]
        out.append(idx)
    return out
a = torch.Tensor([1,1,2,3,4,5,5,5,5])
idxs=getIdxofSameItems(a) 
#output: [tensor([5, 6, 7, 8]), tensor([4]), tensor([3]), tensor([2]), tensor([0, 1])]
