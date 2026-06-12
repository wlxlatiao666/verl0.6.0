
import torch
import numpy as np

print("=== The REAL issue analysis ===")

# Let me create a scenario that matches the code more closely
# Suppose a tree where:
#
# Prompt -> seg0 (root) -> seg1 -> seg2 (leaf1, score=100)
#                      -> seg1 -> seg3 (leaf2, score=100)
#
# So unique_segments has seg0, seg1, seg2, seg3
# leaf_segment_indices for leaf1: [0, 1, 2]
# leaf_segment_indices for leaf2: [0, 1, 3]

n_unique = 4
leaf_segment_indices = [[0, 1, 2], [0, 1, 3]]
leaf_scores = torch.tensor([100.0, 100.0])  # both leaves have same score
device = 'cpu'

print(f"leaf_scores = {leaf_scores}")
print()

# Build parent_of, seg_depth, leaf_seg_to_leaf_idx
leaf_seg_to_leaf_idx = {}
parent_of = np.full(n_unique, -1, dtype=np.int64)
seg_depth = np.zeros(n_unique, dtype=np.int64)

for j, path in enumerate(leaf_segment_indices):
    if len(path) > 0:
        leaf_seg_to_leaf_idx[path[-1]] = j
    for depth, seg_idx in enumerate(path):
        seg_depth[seg_idx] = depth
        if depth > 0:
            parent_of[seg_idx] = path[depth - 1]

print("parent_of =", parent_of)
print("seg_depth =", seg_depth)
print("leaf_seg_to_leaf_idx =", leaf_seg_to_leaf_idx)
print()

# Step 1: Initialize node_scores
node_scores = torch.zeros(n_unique, dtype=torch.float32, device=device)
for seg_idx, leaf_j in leaf_seg_to_leaf_idx.items():
    node_scores[seg_idx] = leaf_scores[leaf_j]

print("Initial node_scores (only leaves filled):", node_scores)

# Step 2: Bottom-up propagation
children_sum = torch.zeros(n_unique, dtype=torch.float32, device=device)
children_count = torch.zeros(n_unique, dtype=torch.float32, device=device)
for i in np.argsort(-seg_depth):  # deepest first
    p = int(parent_of[i])
    if p >= 0:
        children_sum[p] += node_scores[i]
        children_count[p] += 1
        print(f"  node {i} -> parent {p}: children_sum[{p}] = {children_sum[p]}, children_count[{p}] = {children_count[p]}")

print("children_sum =", children_sum)
print("children_count =", children_count)

has_children = children_count > 0
node_scores[has_children] = children_sum[has_children] / children_count[has_children]

print("Final node_scores =", node_scores)
print()

# Step 3: Compute advantages
from collections import defaultdict
seg_advantages = torch.zeros(n_unique, dtype=torch.float32, device=device)
children_of = defaultdict(list)
for i in range(n_unique):
    p = int(parent_of[i])
    if p >= 0:
        children_of[p].append(i)

print("children_of =", dict(children_of))
print()

for p, children in children_of.items():
    children_t = torch.tensor(children, dtype=torch.long, device=device)
    sibling_scores = node_scores[children_t]
    sib_std = sibling_scores.std() if len(children) > 1 else torch.tensor(0.0, device=device)
    parent_score = node_scores[p]
    diff = sibling_scores - parent_score
    seg_advantages[children_t] = diff

    print(f"Parent {p}:")
    print(f"  children = {children}")
    print(f"  sibling_scores = {sibling_scores}")
    print(f"  parent_score = {parent_score}")
    print(f"  diff = {diff}")
    print(f"  sib_std = {sib_std}")
    if sib_std > 0:
        normalized = diff / (sib_std + 1e-6)
        print(f"  normalized = {normalized}")
    else:
        print(f"  normalized would be INF!")
    print()

print("seg_advantages =", seg_advantages)
print()

# ========================================
print("=== Now imagine a scenario with numerical error ===")
print()

# Suppose due to floating point, we have:
leaf_scores = torch.tensor([100.0, 100.0001])
node_scores = torch.zeros(n_unique, dtype=torch.float32, device=device)
for seg_idx, leaf_j in leaf_seg_to_leaf_idx.items():
    node_scores[seg_idx] = leaf_scores[leaf_j]

# Bottom-up
children_sum = torch.zeros(n_unique, dtype=torch.float32, device=device)
children_count = torch.zeros(n_unique, dtype=torch.float32, device=device)
for i in np.argsort(-seg_depth):
    p = int(parent_of[i])
    if p >= 0:
        children_sum[p] += node_scores[i]
        children_count[p] += 1

has_children = children_count > 0
node_scores[has_children] = children_sum[has_children] / children_count[has_children]

print("node_scores with tiny difference:", node_scores)

# Compute advantages again
for p, children in children_of.items():
    if p == 1:  # look at seg1 which has children 2 and 3
        children_t = torch.tensor(children, dtype=torch.long, device=device)
        sibling_scores = node_scores[children_t]
        sib_std = sibling_scores.std()
        parent_score = node_scores[p]
        diff = sibling_scores - parent_score
        print(f"\nParent {p} with slightly different children:")
        print(f"  sibling_scores = {sibling_scores}")
        print(f"  parent_score = {parent_score}")
        print(f"  diff = {diff}")
        print(f"  sib_std = {sib_std}")
        normalized = diff / (sib_std + 1e-6)
        print(f"  normalized = {normalized}")

print()

# ========================================
print("=== Another scenario: only ONE child! ===")
print()

# Suppose seg0 has only one child seg1 (len(children) == 1)
parent_of2 = np.array([-1, 0, 1, 1], dtype=np.int64)
children_of2 = defaultdict(list)
for i in range(4):
    p = int(parent_of2[i])
    if p >= 0:
        children_of2[p].append(i)

# seg0 has only one child seg1
node_scores2 = torch.tensor([50.0, 50.0, 100.0, 100.0])
print("node_scores2 =", node_scores2)

for p, children in children_of2.items():
    if p == 0:  # seg0 has only one child seg1
        children_t = torch.tensor(children, dtype=torch.long, device=device)
        sibling_scores = node_scores2[children_t]
        sib_std = sibling_scores.std() if len(children) > 1 else torch.tensor(0.0, device=device)
        parent_score = node_scores2[p]
        diff = sibling_scores - parent_score
        print(f"\nParent {p} with only ONE child:")
        print(f"  children = {children}")
        print(f"  sibling_scores = {sibling_scores}")
        print(f"  parent_score = {parent_score}")
        print(f"  diff = {diff}")
        print(f"  sib_std = {sib_std}")
        normalized = diff / (sib_std + 1e-6)
        print(f"  normalized = {normalized}  <-- THIS COULD BE HUGE!")

print()

# ========================================
print("=== What if sib_std is very small but NOT zero? ===")
print()

# Imagine:
diff = torch.tensor([-0.001, 0.001])
sib_std = torch.tensor(0.001)
normalized = diff / (sib_std + 1e-6)
print(f"diff = {diff}")
print(f"sib_std = {sib_std}")
print(f"normalized = {normalized}")

print()

# What if sib_std is 1e-6?
sib_std = torch.tensor(1e-6)
normalized = diff / (sib_std + 1e-6)
print(f"If sib_std = 1e-6:")
print(f"normalized = {normalized}")
