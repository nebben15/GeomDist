#!/usr/bin/env python3
"""
Export categorical features from a JSON segmentation file to a TXT file in the
same format as export_smpl_template_vertices.py (header + rows), and save a
pickled Python dict that maps feature indices to category names.

Input JSON format (example: smpl_vert_segmentation.json):
{
  "categoryA": [vertex_idx, ...],
  "categoryB": [vertex_idx, ...],
  ...
}

Output TXT format:
- First lines (header):
  FEATURES_COUNT <int>
  FEATURES_DIM <int>
  FEATURES_MAX_ABS <float ... float>
- Followed by one line per vertex: space-separated feature vector values

Each vertex feature vector is a one-hot vector of length = number of categories.
"""

import argparse
import json
import os
import pickle
from typing import Dict, List


from typing import Tuple


def build_one_hot(json_path: str, vertex_count: int) -> Tuple[List[List[float]], List[str]]:
	with open(json_path, "r", encoding="utf-8") as f:
		seg: Dict[str, List[int]] = json.load(f)

	categories = list(seg.keys())
	categories.sort()  # stable order for reproducibility
	dim = len(categories)
	# Initialize zero matrix [vertex_count, dim]
	features: List[List[float]] = [[0.0] * dim for _ in range(vertex_count)]

	cat_index = {name: i for i, name in enumerate(categories)}
	for name, indices in seg.items():
		ci = cat_index[name]
		for vidx in indices:
			if 0 <= vidx < vertex_count:
				features[vidx][ci] = 1.0
			else:
				# Ignore out-of-range entries but continue
				continue

	return features, categories


def build_categorical(json_path: str, vertex_count: int) -> Tuple[List[List[float]], List[str]]:
	"""
	Build true categorical features: a single dimension per vertex with the integer
	category index. Vertices not present in any category will get -1.
	Returns features as [[cat_idx], ...] and the ordered list of category names.
	"""
	with open(json_path, "r", encoding="utf-8") as f:
		seg: Dict[str, List[int]] = json.load(f)

	categories = list(seg.keys())
	categories.sort()
	cat_index = {name: i for i, name in enumerate(categories)}

	# Initialize with -1 meaning "no category"
	features: List[List[float]] = [[-1.0] for _ in range(vertex_count)]

	for name, indices in seg.items():
		ci = cat_index[name]
		for vidx in indices:
			if 0 <= vidx < vertex_count:
				features[vidx][0] = float(ci)
			else:
				continue

	return features, categories


def write_txt_with_header(out_path: str, features: List[List[float]]):
	count = len(features)
	dim = len(features[0]) if count > 0 else 0
	# Per-dimension max abs; for one-hot this is 1.0 if any vertex uses the dim
	max_abs = [0.0] * dim
	for v in features:
		for i, val in enumerate(v):
			a = abs(val)
			if a > max_abs[i]:
				max_abs[i] = a

	os.makedirs(os.path.dirname(out_path), exist_ok=True)
	with open(out_path, "w", encoding="utf-8") as f:
		f.write(f"FEATURES_COUNT {count}\n")
		f.write(f"FEATURES_DIM {dim}\n")
		f.write("FEATURES_MAX_ABS " + " ".join(f"{m:.8f}" for m in max_abs) + "\n")
		for v in features:
			f.write(" ".join(f"{x:.8f}" for x in v) + "\n")


def save_mapping_pickle(pkl_path: str, categories: List[str]):
	mapping = {
		"index_to_name": {i: name for i, name in enumerate(categories)},
		"name_to_index": {name: i for i, name in enumerate(categories)},
	}
	os.makedirs(os.path.dirname(pkl_path), exist_ok=True)
	with open(pkl_path, "wb") as f:
		pickle.dump(mapping, f, protocol=pickle.HIGHEST_PROTOCOL)


def main():
	parser = argparse.ArgumentParser(description="Export categorical features from JSON to TXT + mapping PKL")
	parser.add_argument("--json", required=True, help="Path to JSON file with category->vertex indices")
	parser.add_argument("--output", required=True, help="Output TXT path (header + features)")
	parser.add_argument("--mapping-pkl", required=True, help="Output PKL path to save feature index/name mapping")
	parser.add_argument("--vertex-count", type=int, default=6890, help="Total number of vertices; defaults to 6890 for SMPL")
	parser.add_argument("--one-hot", action="store_true", help="If set, export one-hot features; otherwise export true categorical (single dim with category index).")
	args = parser.parse_args()

	if args.one_hot:
		features, categories = build_one_hot(args.json, args.vertex_count)
	else:
		features, categories = build_categorical(args.json, args.vertex_count)
	write_txt_with_header(args.output, features)
	save_mapping_pickle(args.mapping_pkl, categories)

	exported_dim = len(features[0]) if len(features) > 0 else 0
	print(f"Exported categorical features for {len(features)} vertices with dim={exported_dim} to: {args.output}")
	print(f"Saved category mapping to: {args.mapping_pkl}")


if __name__ == "__main__":
	main()

