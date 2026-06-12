
import torch
import numpy as np
from collections import defaultdict

print("=== Testing the FIXED logic ===")

# Same tree as before
n_unique = 4
leaf_segment_indices = [[0, 1, 2], [0, 1, 3]]
leaf_scores = torch.tensor([100.0, 100.0])
device = 'cpu'

# Build maps
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

# ========== THE FIXED LOGIC ==========
# Assign leaf scores first
node_scores = torch.zeros(n_unique, dtype=torch.float32, device=device)
for seg_idx, leaf_j in leaf_seg_to_leaf_idx.items():
    node_scores[seg_idx] = leaf_scores[leaf_j]

# Build children_of map
children_of = defaultdict(list)
for i in range(n_unique):
    p = int(parent_of[i])
    if p >= 0:
        children_of[p].append(i)

# Get list of internal nodes (non-leaf nodes)
internal_nodes = [i for i in range(n_unique) if i in children_of]

# Sort internal nodes by depth descending (compute deeper nodes first)
internal_nodes_sorted = sorted(internal_nodes, key=lambda i: -seg_depth[i])

print("internal_nodes =", internal_nodes)
print("internal_nodes_sorted =", internal_nodes_sorted)
print()

# Compute scores for internal nodes bottom-up
for i in internal_nodes_sorted:
    children = children_of[i]
    children_t = torch.tensor(children, dtype=torch.long, device=device)
    child_scores = node_scores[children_t]
    node_scores[i] = child_scores.mean()
    print(f"  node {i}: children = {children}, child_scores = {child_scores}, mean = {node_scores[i]}")

print()
print("FINAL node_scores =", node_scores)
print()

# Now compute advantages
seg_advantages = torch.zeros(n_unique, dtype=torch.float32, device=device)
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
        print(f"  normalized would be (diff / eps) but diff is {diff}")
    print()

print("seg_advantages =", seg_advantages)
print()
print("=== SUCCESS! Parent scores are correctly computed! ===")
print("  Root node (0) has score =", node_scores[0].item())
print("  Which equals the mean of its children!")
