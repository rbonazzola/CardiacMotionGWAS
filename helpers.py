import os, sys

# HOME = os.environ['HOME']
REPOS_ROOT = "/mnt/rodrigo/DGX" # f"{HOME}/01_repos"
os.chdir(REPOS_ROOT)

from paths import *

from CardiacCOMA.config.load_config import load_yaml_config, to_dict

import pickle as pkl

import numpy as np
import pandas as pd
from IPython import embed

from pprint import pprint
import logging

from copy import deepcopy
from typing import List
from tqdm.notebook import tqdm

import seaborn as sns
import glob
import re
from functools import partial
import ast

from CardiacMotionRL.utils.run_helpers import *

logging.basicConfig(
  level=logging.INFO, 
  format='%(asctime)s - %(levelname)s - %(message)s',
  datefmt='%Y-%m-%d %H:%M:%S'
)


EXPERIMENT_DICT = {
  "3": "RV",
  "4": "LV",
  "5": "BV",
  "6": "LA",
  "7": "RA",
  "8": "AO",
}

regions_df = pd.read_csv(f"{Paths.Repos.GWAS}/data/ld_indep_regions/fourier_ls-all_EUR_hg19_named.bed").set_index('id')
# latent_dim_c_dict = runs_df[["run_id", "params.latent_dim_c"]].reset_index(drop=True).set_index("run_id").to_dict()['params.latent_dim_c']
latent_dim_c_dict = runs_df[["params.latent_dim_c"]].to_dict()['params.latent_dim_c']
# rec_ratio_per_run = runs_df["metrics.val_rec_ratio_to_time_mean"].reset_index().drop("experiment_id", axis=1).set_index("run_id").to_dict()['metrics.val_rec_ratio_to_time_mean']
rec_ratio_per_run = runs_df[["metrics.val_rec_ratio_to_time_mean"]].to_dict()['metrics.val_rec_ratio_to_time_mean']

msd_static_per_run = runs_df[["metrics.val_recon_loss_c"]].to_dict()['metrics.val_recon_loss_c']
msd_dynamic_per_run = runs_df[["metrics.val_recon_loss_s"]].to_dict()['metrics.val_recon_loss_s']

run_to_expid_dict = runs_df.reset_index(drop=True).set_index('run_id')["experiment_id"].to_dict() # { k: EXPERIMENT_DICT[v] for k,v in runs_df.reset_index(drop=True).set_index('run_id')["experiment_id"].to_dict().items() }
run_to_chamber_dict = { k: EXPERIMENT_DICT[v] for k, v in run_to_expid_dict.items() }


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

loci_mapping = fetch_loci_mapping()

# TODO: use df.assign(**new_columns_dict) to parse columns
# with new_columns_dict={"col1": function1, "col2": function2}, etc.

class EnsembleResults:
    
    def __init__(self, root_dir, expid=None, top_n_runs_per_chamber=None, from_cached=False, cache=None):
        
        self._rootdir = root_dir        
        self._top_n_runs_per_chamber = top_n_runs_per_chamber
        
        # self.expid = expid
        self._collect_summaries(from_cached=from_cached, cache=cache)
        self.loci_mapping_df = fetch_loci_mapping()
        
        self._valid_regions = self.loci_mapping_df.index[self.loci_mapping_df["duplicated"].isnull()]
        
        
    def _collect_summaries(self, from_cached, cache):
        
        if not os.path.exists(self._rootdir):
            logging.error(f"Folder {self._rootdir} does not exist. Other folders under the same parent folder are: {', '.join(os.listdir(os.path.dirname(self._rootdir)))}")
        
        cache_file = f"{self._rootdir}/region_assocs_df.pkl"
        
        if from_cached and os.path.exists(cache_file):
            logging.info(f"Loading cached results from {cache_file}")
            self.region_assocs_df = pd.read_pickle(f"{cache_file}")

        else:     

            if not os.path.exists(cache_file):
                logging.info(f"File {cache_file} does not exist. Results will be cached.")
                cache = True

            summary_files = sorted(glob.glob(f"{self._rootdir}/summaries/*"))                        
            logging.info(f"Found {len(summary_files)} files under the folder {self._rootdir}/summaries")
            zvar_gwas_assocs = []
                        
            for file in tqdm(summary_files):            
                
                pheno = re.sub(".*GWAS__|.*GWAS_|__region.*tsv|_region.*tsv", "", file) # get basename w/o extension or prefix or suffix                        
                if "z0" in pheno:
                    try:
                        # embed()
                        expid, run_id, zvar = pheno.split("_")[0], pheno.split("_")[1], pheno.split("_")[2]
                        expid = expid.replace("X", "")
                        chamber = EXPERIMENT_DICT[expid]
                        full_phenoname = f"{run_id}_{zvar}"
                        logging.debug(f"Experiment: {expid} / Run: {run_id} / phenotype: {zvar}")
                        zvar_gwas_assocs.append(pd.read_csv(file).assign(expid=expid, chamber=chamber, run=run_id, pheno=zvar, full_pheno=full_phenoname))
                    except:
                        zvar, run_id = pheno.split("_")[0], pheno.split("_")[1]
                        chamber = run_to_chamber_dict[run_id]
                        full_phenoname = f"{run_id}_{zvar}"
                        logging.debug(f"Run: {run_id} / phenotype: {zvar}")
                        zvar_gwas_assocs.append(pd.read_csv(file).assign(run=run_id, pheno=zvar, full_pheno=full_phenoname, chamber=chamber))
    
                elif (("aha" in pheno) or ("AHA" in pheno)):                
                    magnitude, aha_segment = "_".join(pheno.split("_")[:-1]), pheno.split("_")[-1] # "_".join(pheno.split("_")[1:])
                    chamber = "LV"
                    logging.debug(f"Magnitude: {magnitude} / AHA segment: {aha_segment}")
                    zvar_gwas_assocs.append(pd.read_csv(file).assign(chamber=chamber, magnitude=magnitude, aha_segment=aha_segment, pheno=pheno))
                    
                elif  ("_ED_" not in pheno) and ("_MEAN_" not in pheno):
                    logging.debug(f"Phenotype: {pheno}")
                    
                    for possible_chamber in ["BV", "LV", "RV", "LA", "RA"]:
                      if possible_chamber in pheno:
                          chamber = possible_chamber
                    zvar_gwas_assocs.append(pd.read_csv(file).assign(pheno=pheno, chamber=chamber))
                                                
                else:                
                    run_id, pheno = pheno.split("_")[0], "_".join(pheno.split("_")[1:])
                    chamber = run_to_chamber_dict[run_id]
                    logging.debug(f"Run: {run_id} / phenotype: {pheno}")
                    zvar_gwas_assocs.append(pd.read_csv(file).assign(run=run_id, pheno=pheno, chamber=chamber))
                
            logging.info(f"Collected GWAS summary data for {len(zvar_gwas_assocs)} phenotypes.")
    
            logging.info(f"Concatenating...")
            region_assocs_df = pd.concat(zvar_gwas_assocs)
            region_assocs_df = region_assocs_df.set_index(["pheno", "region"])
            region_assocs_df = region_assocs_df.reset_index()
     
            # sorted(region_assocs_df[region_assocs_df.P < 5e-8].index.get_level_values("region").unique()
            # region_assocs_df.iloc[region_assocs_df.index.get_level_values("region") == "chr1_78",].sort_values("P").head(15)
            
            # This is a way to determine whether we are dealing with latent variables
            if any(region_assocs_df.pheno.apply(lambda x: "z0" in x)):
                logging.info(f"Assigning dynamic/static label to phenotypes...")
                try:
                    region_assocs_df["variable_type"] = region_assocs_df.apply(self._get_variable_type, axis=1)
                except Exception as e:
                    logging.error(e)
            
            if "run" in region_assocs_df.columns:
                region_assocs_df["msd_static"] = region_assocs_df.run.apply(lambda x: msd_static_per_run[x])
                region_assocs_df["msd_dynamic"] = region_assocs_df.run.apply(lambda x: msd_dynamic_per_run[x])
                region_assocs_df["rec_ratio"] = region_assocs_df.run.apply(lambda x: rec_ratio_per_run[x])
            
            if cache:
                logging.info(f"Caching results to {self._rootdir}/region_assocs_df.pkl")
                region_assocs_df.to_pickle(f"{self._rootdir}/region_assocs_df.pkl")
    
            self.region_assocs_df = region_assocs_df
         
        if self._top_n_runs_per_chamber is not None:
            logging.info(f"Only the best (at most) {self._top_n_runs_per_chamber} runs per chamber with the best performance will be kept. Filtering...")
            self.region_assocs_df = self._keep_top_n_per_chamber(n=self._top_n_runs_per_chamber)
    
    
    def _keep_top_n_per_chamber(self, n):
        
        rec_losses = []
        
        for chamber in self.region_assocs_df.chamber.unique():
            results_for_chamber = self.region_assocs_df[self.region_assocs_df.chamber == chamber]
            unique_values = results_for_chamber.msd_dynamic.unique()[:n]
            logging.info(f"Chamber {chamber}: Keeping {len(unique_values)} runs with the best performance.")
            rec_losses.extend(unique_values)
        
        rec_losses = set(rec_losses)
        self.region_assocs_df = self.region_assocs_df[self.region_assocs_df.rec_ratio.apply(lambda x: x in rec_ratios)]

        return self.region_assocs_df
        
    
    def _get_variable_type(self, row):

        n_static_variables = int(latent_dim_c_dict[row.run])
        zvar = int(row.pheno[-2:])
        return "static" if zvar < n_static_variables else "dynamic"
        
        
    def loci_summary(self, only_dynamic=False, only_static=False, per_chamber=False, chamber=None):
        
        assert chamber in [None, "BV", "LV", "RV", "LA", "RA", "AO"], f"Chamber not valid, got {chamber}."
        
        if "run" in self.region_assocs_df.columns:
            GROUPBY_COLS = ["run", "region"]
        else:
            GROUPBY_COLS = ["region"]
        if "variable_type" in self.region_assocs_df.columns: GROUPBY_COLS.append("variable_type")
        if per_chamber:                                      GROUPBY_COLS.append("chamber")
            
        loci_summary_df = self.region_assocs_df[self.region_assocs_df.P < 5e-8].\
            reset_index().\
            groupby(by=GROUPBY_COLS).\
            aggregate({"CHR":"count", "P": "min"}).\
            rename({"CHR":"count", "P":"min_P"}, axis=1).\
            sort_values("count", ascending=False).\
            sort_values("min_P", ascending=True)
        
        if "variable_type" in loci_summary_df.columns:
            if only_dynamic:
                loci_summary_df = loci_summary_df[loci_summary_df.index.get_level_values("variable_type") == "dynamic"]
            elif only_static:
                loci_summary_df = loci_summary_df[loci_summary_df.index.get_level_values("variable_type") == "static"]
        
        # Filter regions that are duplicated (according to the spreadsheet)
        loci_summary_df = loci_summary_df.loc[[x in self._valid_regions for x in loci_summary_df.index.get_level_values("region")]]
        
        return loci_summary_df
    
    
    def loci_count(self, per_chamber=False):
        
        loci_summary_df = self.loci_summary(per_chamber=per_chamber)
        
        GROUPBY_COLS = ["region", "variable_type"]
        
        if per_chamber:
            GROUPBY_COLS.append("chamber")
        
        try:
            loci_count_df = loci_summary_df.groupby(GROUPBY_COLS)
        except KeyError as e:
            print(e)
            loci_count_df = loci_summary_df.groupby(["region"])
        
        loci_count_df = loci_count_df.\
            aggregate({"count":"count", "min_P": "min"}).\
            rename({"CHR":"count", "P":"min_P"}, axis=1).\
            sort_values("count", ascending=False)
            
        return loci_count_df
    
    
    def show_counts(self, count_thr = 5, pvalue_thr=1.5e-10):
        region_count_df = self.loci_count()
        display( region_count_df[(region_count_df["count"] >= count_thr) | (region_count_df.min_P < pvalue_thr)])
    
    
    def get_results_for_region(self, region, top_n=20, only_dynamic=False, only_static=False, exp_ids=None):
    
        def color_negative_red(val):
            color = 'red' if val < 0 else 'black'
            return 'color: %s' % color
        
        df = self.region_assocs_df[self.region_assocs_df.region == region].sort_values("P").head(top_n)
        # df = region_assocs_df.iloc[region_assocs_df.index.get_level_values("region") == region,].sort_values("P").head(15)
        df.style.background_gradient(cmap='Blues')
        df.style.set_properties(**{'text-align': 'center'}).set_table_styles([{
          'selector': 'th',
          'props': [('text-align', 'left')]
        }])
        df.style.applymap(color_negative_red)
        
        df = df.rename({"expid": "experiment_id", "run": "run_id"}, axis=1)
        df.experiment_id = df.experiment_id.apply(lambda x: x.replace("X", ""))
        
        if only_dynamic:
            df = df[df.variable_type == "dynamic"]
        elif only_static:
            df = df[df.variable_type == "static"]
            
        if exp_ids is not None:
            df = df[df.experiment_id.apply(lambda x: x in exp_ids)]
            
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
        counts_df["candidate gene"] = counts_df.region.apply(lambda region: self.loci_mapping_df.loc[region, "candidate_gene"])
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
        
        return self.region_assocs_df.iloc[idx_min][["CHR", "BP", "region", "SNP", "AF", "P"]].\
            reset_index(drop=True).\
            sort_values(["CHR", "region"]).\
            sort_values(["CHR", "BP"])
    
    
    def counts_per_chamber(self):
        
        COL_ORDER = [(variable_type, chamber) for variable_type in ["dynamic", "static"] for chamber in ["LV", "RV", "LA", "RA"]]

        counts_per_chamber = self.loci_count(per_chamber=True).\
          reset_index().\
          pivot(index="region", values="count", columns=["variable_type", "chamber"]).\
          fillna(0).astype(int)[COL_ORDER]
        
        ordered_by_dynamic = counts_per_chamber["dynamic"].sum(axis=1).sort_values(ascending=False).index
        total_counts_dynamic = counts_per_chamber.loc[ordered_by_dynamic]["dynamic"].sum(axis=1)
        total_counts_static = counts_per_chamber.loc[ordered_by_dynamic]["static"].sum(axis=1)
        
        ratio_dyn_to_stat = (total_counts_dynamic - total_counts_static) / (total_counts_dynamic + total_counts_static)
        possible_order = (ratio_dyn_to_stat[(total_counts_dynamic+total_counts_static) > 3]).sort_values(ascending=False).index
        
        counts_per_chamber = counts_per_chamber.loc[possible_order]
        counts_per_chamber.index = [self.loci_mapping_df.loc[region, "candidate_gene"] for region in counts_per_chamber.index]
        
        return counts_per_chamber
    
# def get_significant_loci(
#     runs_df,
#     experiment_id, run_id, 
#     p_threshold=5e-8, 
#     client=mlflow.tracking.MlflowClient()
# ) -> pd.DataFrame:
#     
#     '''    
#     Returns a DataFrame with the loci that have a stronger p-value than a given threshold
#     '''
#     
#     def get_phenoname(path):        
#         filename = os.path.basename(path)
#         phenoname = filename.split("__")[0]
#         return phenoname
#         
#     run_info = runs_df.loc[(experiment_id, run_id)].to_dict()
#     artifact_uri = run_info["artifact_uri"].replace("file://", "")    
#            
#     gwas_dir_summaries = os.path.join(artifact_uri, "GWAS/summaries")
#     
#     try:
#         summaries_fileinfo = [ os.path.join(gwas_dir_summaries, x) for x in  os.listdir(gwas_dir_summaries) ]
#     except:
#         summaries_fileinfo = []
#     
#     if len(summaries_fileinfo) == 0:
#         return pd.DataFrame(columns=["run", "pheno", "region"])
#     
#     region_summaries = {get_phenoname(x): os.path.join(artifact_uri, x) for x in summaries_fileinfo}
#     dfs = [pd.read_csv(path).assign(pheno=pheno) for pheno, path in region_summaries.items()]
#     
#     df = pd.concat(dfs)
#     df['locus_name'] = df.apply(lambda row: REGION_TO_LOCUS.get(row["region"], "Unnamed"), axis=1)
#     df = df.set_index(["pheno", "region"])    
#     
#     df_filtered = df[df.P < p_threshold]
#     
#     return df_filtered.sort_values(by="P")
# 
# 
# def summarize_loci_across_runs(runs_df: pd.DataFrame):
# 
#     '''
#     Parameters: run_ids
#     Return: pd.DataFrame with .
#     '''
# 
#     # run_ids = sorted([x[1] for x in runs_df[runs_df["metrics.test_recon_loss"] < RECON_LOSS_THRES].index])
#     run_ids = sorted([x[1] for x in runs_df.index])
# 
#     all_signif_loci = []
#     
#     for run_id in tqdm(run_ids):
#         signif_loci_df = \
#             get_significant_loci(runs_df, experiment_id=1, run_id=run_id).\
#             assign(run=run_id).\
#             reset_index().\
#             set_index(["run", "pheno", "region"]
#         )                
#         all_signif_loci.append(signif_loci_df)        
#       
#     all_signif_loci = pd.concat(all_signif_loci)    
#     return all_signif_loci
# 
#     # df = all_signif_loci.\
#     #   groupby(["region", "locus_name"]).\
#     #   aggregate({"CHR":"count", "P": "min"}).\
#     #   rename({"CHR":"count", "P":"min_P"}, axis=1).\
#     #   sort_values("count", ascending=False)    
#     # 
#     # return df