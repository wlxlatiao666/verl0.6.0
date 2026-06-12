
import torch

print("=== PyTorch std() test ===")
# Test 1: Two identical values - what's std?
x = torch.tensor([5.0, 5.0])
print(f"x = {x}")
print(f"std(unbiased=True) = {x.std(unbiased=True)}")  # default
print(f"std(unbiased=False) = {x.std(unbiased=False)}")
print()

# Test 2: Two very close values
x = torch.tensor([5.0, 5.0001])
print(f"x = {x}")
print(f"mean = {x.mean()}")
print(f"diff = {x - x.mean()}")
print(f"std(unbiased=True) = {x.std(unbiased=True)}")
print(f"std(unbiased=False) = {x.std(unbiased=False)}")
print()

# Test 3: What if we have parent score = mean(children), but with numerical error?
sibling_scores = torch.tensor([1.23456789, 1.23456789])
parent_score = sibling_scores.mean()
diff = sibling_scores - parent_score
print(f"sibling_scores = {sibling_scores}")
print(f"parent_score = {parent_score}")
print(f"diff = {diff}")
print(f"diff.abs().sum() = {diff.abs().sum()}")
print()

# Test 4: Numerical precision example
a = torch.tensor([0.1, 0.2])
print(f"a = {a}")
print(f"sum(a) = {a.sum()}")
print(f"sum(a) - 0.3 = {a.sum() - 0.3}")
print()

# Test 5: Simulate the actual scenario
print("=== Simulate tree scenario ===")
# Suppose we have:
# Parent has children [A, B], where A is a leaf, B has children [B1, B2]
leaf_scores = torch.tensor([10.0, 10.0])  # B1, B2
node_scores = torch.zeros(4)  # root, A, B, B1, B2
node_scores[3] = 10.0  # B1
node_scores[4] = 10.0  # B2
# Compute B's score
node_scores[2] = (node_scores[3] + node_scores[4]) / 2  # mean = 10
# Compute A's score (leaf)
node_scores[1] = 8.0  # different value
# Compute root's score
node_scores[0] = (node_scores[1] + node_scores[2]) / 2  # mean = 9

print(f"node_scores = {node_scores}")
# Now compute advantages for root's children (A and B)
children = [1, 2]
sibling_scores = node_scores[children]
parent_score = node_scores[0]
diff = sibling_scores - parent_score
sib_std = sibling_scores.std()  # default unbiased=True
print(f"\nRoot's children:")
print(f"  sibling_scores = {sibling_scores}")
print(f"  parent_score = {parent_score}")
print(f"  diff = {diff}")
print(f"  sib_std = {sib_std}")
print(f"  diff / (sib_std + 1e-6) = {diff / (sib_std + 1e-6)}")

# Now compute advantages for B's children (B1 and B2)
children = [3, 4]
sibling_scores = node_scores[children]
parent_score = node_scores[2]
diff = sibling_scores - parent_score
sib_std = sibling_scores.std()  # default unbiased=True
print(f"\nB's children:")
print(f"  sibling_scores = {sibling_scores}")
print(f"  parent_score = {parent_score}")
print(f"  diff = {diff}")
print(f"  sib_std = {sib_std}")
print(f"  diff / (sib_std + 1e-6) = {diff / (sib_std + 1e-6)}")
