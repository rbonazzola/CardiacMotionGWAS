import os, sys

GWAS_REPO = "/home/rodrigo/01_repos/GWAS_pipeline/"
CARDIAC_COMA_REPO = "/home/rodrigo/01_repos/CardiacCOMA/"
CARDIAC_GWAS_REPO = "/home/rodrigo/01_repos/CardiacGWAS/"

HOME = os.environ['HOME']

import mlflow

import os; os.chdir(CARDIAC_COMA_REPO)
from config.load_config import load_yaml_config, to_dict

import ipywidgets as widgets
from ipywidgets import interact
from IPython.display import Image
from mlflow.tracking import MlflowClient

import pickle as pkl
import pytorch_lightning as pl

from argparse import Namespace
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from IPython import embed
sys.path.insert(0, '..')

import model.Model3D
from copy import deepcopy
from pprint import pprint

from copy import deepcopy
from typing import List
from tqdm import tqdm
from IPython import embed

import matplotlib.pyplot as plt
import seaborn as sns
import glob
from functools import partial


class EnsembleResults:
    
    def __init__(self, root_dir, expid=None):
        
        self._rootdir = root_dir
        self._summary_file = f"{HOME}/01_repos/CardiacMotionGWAS/results/gwas_loci_summary_across_runs.csv"
        self.expid = expid
        self._collect_summaries()
        self.loci_mapping_df = fetch_loci_mapping()
        
        
    def _collect_summaries(self):
        
        summary_files = glob.glob(f"{HOME}/01_repos/GWAS_pipeline/output/All_partitions_spatiotemporal/summaries/*")

        zvar_gwas_assocs = []
        
        NSTATIC = 8
        
        for file in tqdm(summary_files):
            expid, run_id, zvar = file.split("_")[6], file.split("_")[7], file.split("_")[8]
            zvar_gwas_assocs.append(pd.read_csv(file).assign(expid=expid, run=run_id, pheno=zvar))
            
            # print(run_id, zvar, pd.read_csv(file))
            
        region_assocs_df = pd.concat(zvar_gwas_assocs)
        region_assocs_df = region_assocs_df.set_index(["expid", "run", "pheno", "region"])
        
        # region_assocs_df
        # sorted(region_assocs_df[region_assocs_df.P < 5e-8].index.get_level_values("region").unique()
        # region_assocs_df.iloc[region_assocs_df.index.get_level_values("region") == "chr1_78",].sort_values("P").head(15)
        
        # region_assocs_df.to_csv("/home/rodrigo/01_repos/CardiacMotionGWAS/results/gwas_loci_summary_across_runs.csv")
        region_assocs_df = pd.read_csv(self._summary_file)
        region_assocs_df = region_assocs_df.reset_index()
        region_assocs_df["variable_type"] = region_assocs_df.pheno.apply(
            lambda x: "static" if int(x[-2:]) < NSTATIC else "dynamic"
        )
        
        if self.expid is not None:
            region_assocs_df = region_assocs_df[region_assocs_df.expid == "X4"]
            
        self.region_assocs_df = region_assocs_df
        
        
    def loci_summary(self):
        
        loci_summary_df = region_assocs_df[region_assocs_df.P < 5e-8].\
            reset_index().\
            groupby(by=["run", "variable_type", "region"]).\
            aggregate({"CHR":"count", "P": "min"}).\
            rename({"CHR":"count", "P":"min_P"}, axis=1).\
            sort_values("count", ascending=False).\
            sort_values("min_P", ascending=True)
        
        return loci_summary_df
    
    
    def loci_count(self):
        
        loci_count_df = loci_summary_df.\
            groupby(["region", "variable_type"]).\
            aggregate({"count":"count", "min_P": "min"}).\
            rename({"CHR":"count", "P":"min_P"}, axis=1).\
            sort_values("count", ascending=False)
            
        return loci_count_df
    
    
    def show_counts(self, count_thr = 5, pvalue_thr=1.5e-10):
        region_count_df = self.loci_count()
        display( region_count_df[(region_count_df["count"] >= count_thr) | (region_count_df.min_P < pvalue_thr)])
    
    
    def fetch_loci_mapping():
    
        import requests
        from io import StringIO
        # https://docs.google.com/spreadsheets/d/1LbILFyaTHeRPit8v3gwx2Db4uS1Hnx6dibeGHK9zXcU/edit?usp=sharing
        # LINK = 'https://docs.google.com/spreadsheet/ccc?key=1LbILFyaTHeRPit8v3gwx2Db4uS1Hnx6dibeGHK9zXcU&output=csv'
        LINK = 'https://docs.google.com/spreadsheet/ccc?key=1XvVDFZSvcWWyVaLaQuTpglOqrCGB6Kdf6c78JJxymYw&output=csv'
        response = requests.get(LINK)
        assert response.status_code == 200, 'Wrong status code'
        loci_mapping_df = pd.read_csv(
            StringIO(response.content.decode()),
            sep=","
        ).set_index("region")
        
        return loci_mapping_df
    
    
    def get_results_for_region(self, region):
    
        def color_negative_red(val):
            color = 'red' if val < 0 else 'black'
            return 'color: %s' % color
        
        df = self.region_assocs_df[self.region_assocs_df.region == region].sort_values("P").head(20)
        # df = region_assocs_df.iloc[region_assocs_df.index.get_level_values("region") == region,].sort_values("P").head(15)
        df.style.background_gradient(cmap='Blues')
        df.style.set_properties(**{'text-align': 'center'}).set_table_styles([{
          'selector': 'th',
          'props': [('text-align', 'left')]
        }])
        df.style.applymap(color_negative_red)
        return df
    
    
    def get_suggestive_regions(self):
        
        region_count_df = self.loci_count()
        suggestive_regions = sorted(
            region_count_df[
                ((region_count_df.min_P < 5e-8) & (region_count_df["count"] >= 1)) &
                (region_count_df.min_P > 1.5e-10)
            ].index.get_level_values("region")
        )
        
        suggestive_regions = list(set(suggestive_regions))

        return suggestive_regions
            
        
    def get_significant_regions(self):

        region_count_df = self.loci_count()
        significant_regions = sorted(
            region_count_df[
                ((region_count_df.min_P < 5e-8) & (region_count_df["count"] >= 5)) |
                (region_count_df.min_P < 1.5e-10)
            ].index.get_level_values("region")
        )
        
        significant_regions = list(set(significant_regions))
        return significant_regions
    
    
    def create_count_table_tex(self, tex_file=None):
    
        '''
          counts_df: 
          tex_file:
        '''
        
        regions = self.get_significant_regions()
        snp_data = self._snp_data()
        counts_df = self.loci_count()
        
        # counts_df = deepcopy(counts_df)        
        
        counts_df = counts_df.loc[regions]
        counts_df.min_P = [f"${str(round(float(x[0]), 1))} \times 10^{{{x[1]}}}$" for x in counts_df.min_P.astype(str).str.split("e")]
        counts_df = counts_df.reset_index()
        counts_df["candidate gene"] = counts_df.region.apply(lambda region: loci_mapping_df.loc[region, "candidate_gene"])
        counts_df["chr."] = counts_df.region.apply(lambda region: regions_df.loc[region, "chr"])
        
        counts_df = counts_df.merge(snp_data, on="region")
        counts_df["region"] = counts_df.region.apply(lambda region: f'{regions_df.loc[region, "start"]}-{regions_df.loc[region, "stop"]}')
        counts_df = counts_df.sort_values("count", ascending=False)
        counts_df = counts_df[["chr.", "region", "candidate gene", "count", "min_P", "SNP", "a_0", "a_1", "AF", "BETA", "SE"]]
        counts_df = counts_df.rename({"min_P": "min. $p$-value", "a_0": "NEA", "a_1": "EA", "AF": "EAF"}, axis=1)
        
        table_code = counts_df.to_latex(escape=False, index=False)
        # table_code = counts_df.style.to_latex()
    
        table_code = table_code.replace("_", "\_")

        if tex_file is not None:
            print(f"Creating output file in {tex_file}")
            with open(tex_file, "wt") as table_f:    
                table_f.write(table_code)
        
        else: 
            return table_code
        
    
    def _snp_data(self):
        
        region_assocs_df = self.region_assocs_df
        # print(region_assocs_df.shape)
        region_assocs_df = region_assocs_df.loc[~region_assocs_df.sort_values("SNP").duplicated("SNP")]
        signif_regions = set(self.get_significant_regions())
        region_assocs_df = region_assocs_df.reset_index()
        rows_signif_regions = region_assocs_df.apply(lambda row: row.region in signif_regions, axis=1)
        
        region_assocs_df = region_assocs_df[rows_signif_regions]
        region_assocs_df = region_assocs_df[region_assocs_df.P < 5e-8]
        
        snp_data = region_assocs_df.loc[:,["region", "SNP", "BP", "AF", "a_0", "a_1", "BETA", "SE"]]
        return snp_data
    
    
    def assocs_per_variable_type(self, type="d"):
        
        by_variable_type_df = results.loci_count()[["min_P"]].reset_index().pivot(index="region", columns="variable_type", values="min_P")
        
        if type == "d":
            return by_variable_type_df.sort_values("dynamic")
        else:
            return by_variable_type_df.sort_values("static")
    
    
    def get_lead_snps(self):
        
        idx_min = self.region_assocs_df.groupby("region").P.idxmin()
        idx_min = idx_min[self.get_significant_regions()]        
        return self.region_assocs_df.iloc[idx_min, [3,4,5,6]].reset_index(drop=True).sort_values(["CHR", "region"])