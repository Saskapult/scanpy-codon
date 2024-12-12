
import anndata as ad
import pooch
import time
import numpy as np 
import matplotlib.pyplot as plt


def main():
	n = 2
	
	def get_anndata():
		EXAMPLE_DATA = pooch.create(
			path=pooch.os_cache("scverse_tutorials"),
			base_url="doi:10.6084/m9.figshare.22716739.v1/",
		)
		EXAMPLE_DATA.load_registry_from_doi()

		samples = {
			"s1d1": "s1d1_filtered_feature_bc_matrix.h5",
			"s1d3": "s1d3_filtered_feature_bc_matrix.h5",
		}
		adatas = {}

		for sample_id, filename in samples.items():
			path = EXAMPLE_DATA.fetch(filename)
			sample_adata = sc.read_10x_h5(path)
			sample_adata.var_names_make_unique()
			adatas[sample_id] = sample_adata

		adata = ad.concat(adatas, label="sample")
		adata.obs_names_make_unique()
		return adata
	def pca(adata):
		st = time.time()
		sc.tl.pca(adata)
		en = time.time()
		dt = en - st
		return dt
	
	# print("Benching scanpy...")
	# import scanpy as sc
	# dts_scanpy = bench_n(get_anndata, pca, n)
	# # dts_scanpy = [4000.0, 26477.0]
	# print_dt_stats(dts_scanpy)


	print("Benching scanpy_codon...")
	# del sc
	import scanpy_codon as sc
	print(dir(sc))
	dts_scanpy_codon = bench_n(get_anndata, pca, n)
	print_dt_stats(dts_scanpy_codon)


def print_dt_stats(dts):
	worst = max(dts)
	best = min(dts)
	mean = sum(dts) / len(dts)
	median = sorted(dts)[len(dts)/2] if len(dts) % 2 == 1 else sum(sorted(dts)[int(len(dts)/2):int(len(dts)/2+1)])/2
	
	print(f"Best:   {best: >8.2f}s")
	print(f"Worst:  {worst: >8.2f}s")
	print(f"Mean:   {mean: >8.2f}s")
	print(f"Median: {median: >8.2f}s")


def bench_n(s, f, n):
	adata = s()
	dts = []
	# print(f"{0} / {n}", end="")
	print(f"{0} / {n}")
	for i in range(0, n):
		a = adata.copy() # Do not mutate adata
		dts.append(f(a))
		# print(f"{i+1} / {n}", end="\r")
		print(f"{i+1} / {n}")
	# print()
	return dts


if __name__ == "__main__":
	main()
