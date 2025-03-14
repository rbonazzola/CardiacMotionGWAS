# GWAS on cardiac motion phenotypes

### Class `EnsembleGWASResults`

Initialize an `EnsembleGWASResults` object by passing it a path to GWAS results. 
This folder needs to have a subfolder called `summaries`, with one file per phenotype, detailing the best association for each genomic region (one region per row).

##### Usage

```python
>>> from cardiac_motion_upe.EnsembleResults
>>> gwas_dir = "GWAS_results"
>>> results = EnsembleGWASResults(gwas_dir)

>>> results.get_significant_regions()
# Returns, e.g., ['chr4_77', 'chr17_27', 'chr22_7', 'chr17_40', 'chr12_67', 'chr15_35', ...]

>>> results.region_assocs_df
# Returns a dataframe with one z-SNP association per row,
# detailing experiment ID, run ID, SNP data, variable type (dyn/stat),
# and GWAS summary statistics.

>>> results.get_lead_snps()
# Returns a dataframe with one region per row, with details about the lead SNP in that region, including the minimum p-value found among the results.

>>> results.loci_summary()
# Returns a dataframe with (run, variable_type, region) as key and (count, min_P) as values.

>>> results.show_counts()
# Returns a dataframe with (region, variable_type) as key and (count, min_P) as values.
```
