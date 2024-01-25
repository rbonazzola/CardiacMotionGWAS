# GWAS on cardiac motion phenotypes

### Class `EnsemblResults`

Initialize an `EnsemblResults` object by passing it a path to GWAS results. 
This folder needs to have a subfolder called `summaries`, with one file per phenotype, detailing the best association for each genomic region (one region per row).

##### Usage

```python
>>> gwas_dir = "GWAS_results"
>>> results = EnsembleResults(gwas_dir)

>>> results.get_significant_regions()
# Returns, e.g., ['chr4_77', 'chr17_27', 'chr22_7', 'chr17_40', 'chr12_67', 'chr15_35', ...]

>>> results.results.region_assocs_df
# Returns a dataframe with one z-SNP association per row,
# detailing experiment ID, run ID, SNP data, variable type (dyn/stat),
# and GWAS summary statistics.

>>> results.get_lead_snps()
# Returns a dataframe with one region per row, with details about the lead SNP in that region

```
