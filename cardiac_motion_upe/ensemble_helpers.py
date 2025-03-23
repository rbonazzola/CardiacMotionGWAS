import os, sys
import glob
import re

from tqdm import tqdm
from pprint import pprint
import pickle as pkl

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import logging
LOGLEVEL = os.getenv("LOGLEVEL", "INFO")
logging.basicConfig(level=LOGLEVEL)

import mlflow

from cardiac_motion_upe import (
    BASE_DIR, GWAS_RESULTS_DIR)

from cardiac_motion.config.load_config import (
    load_yaml_config, 
    to_dict)

from cardiac_motion.utils.mlflow_read_helpers import (
    Run
)

from loci_helpers import (
    get_ld_indep_regions,
    fetch_loci_mapping
)

import gwas_pipeline


def generate_loci_summary(assocs_df, 
    only_dynamic=False, 
    only_static=False,     
    # chamber=None, 
    p_threshold=5e-8, 
    other_columns=['chamber']):

    assocs_df = assocs_df.copy()

    # assert chamber in { None } | EnsembleGWASResults.possible_chambers, f"Chamber not valid, got {chamber}."
    assert not (only_dynamic and only_static), "At most one of only_dynamic or only_static can be True."
        
    GROUPBY_COLS = ["region", "run", "variable_type"]
    
    # if "run" in assocs_df.columns:   
        # GROUPBY_COLS.append("run")
    # if "variable_type" in assocs_df.columns: 
        # GROUPBY_COLS.append("variable_type")
    GROUPBY_COLS.extend(other_columns)
        
    loci_summary_df = ( assocs_df
        .query("P < @p_threshold")
        .reset_index()
        .groupby(by=GROUPBY_COLS)
        .aggregate({"CHR": "count", "P": "min"}).rename({"CHR": "count", "P": "min_P"}, axis=1)
        .reset_index()
        .sort_values("count", ascending=False)
        .sort_values("min_P", ascending=True) ) 
    
    if "variable_type" in loci_summary_df.columns:
        if only_dynamic:
            loci_summary_df.query("variable_type == 'dynamic'")  
        elif only_static:
            loci_summary_df.query("variable_type == 'static'")

    # Filter regions that are duplicated (according to the spreadsheet)
    loci_summary_df = EnsembleGWASResults.filter_valid_regions(loci_summary_df)
    # loci_summary_df.loc[[x in EnsembleGWASResults.valid_regions for x in loci_summary_df.index.get_level_values("region")]]
    
    return loci_summary_df


class EnsembleGWASResults:

    runs_df = Run.get_runs()
    
    regions_df          = get_ld_indep_regions()
    latent_dim_c_dict   = runs_df[["params.latent_dim_c"]].to_dict()['params.latent_dim_c']
    
    rec_ratio_per_run   = runs_df[["metrics.val_rec_ratio_to_time_mean"]].to_dict()['metrics.val_rec_ratio_to_time_mean']
    msd_static_per_run  = runs_df[["metrics.val_recon_loss_c"]].to_dict()['metrics.val_recon_loss_c']
    msd_dynamic_per_run = runs_df[["metrics.val_recon_loss_s"]].to_dict()['metrics.val_recon_loss_s']

    run_to_expid_dict   = runs_df.reset_index(drop=True).set_index('run_id')["experiment_id"].to_dict()
    
    expid_to_partition_mapping = { str(k): v for k, v in Run.expid_to_partition_mapping.items() }

    # TODO: THIS IS A COMMON PROBLEM, SOLVE IT IN SOME ELEGANT WAY
    assert type(list(expid_to_partition_mapping.keys())[0]) == type(list(run_to_expid_dict.values())[0]), f"""
        Keys of expid_to_partition_mapping are not of the right type: f{expid_to_partition_mapping.keys()} vs. {set(run_to_expid_dict.values())}
    """
    
    run_to_chamber_dict = dict()
    for k, v in run_to_expid_dict.items():
        run_to_chamber_dict[str(k)] = expid_to_partition_mapping[v]

    loci_mapping_df = fetch_loci_mapping()
    valid_regions = loci_mapping_df.index[loci_mapping_df["duplicated"].isnull()]
    
    possible_chambers = { "BV", "LV", "RV", "LA", "RA", "AO" }

    RELEVANT_RUN_HPARAMS = [ "params.latent_dim_c", "params.latent_dim_s", "params.dataset_static_representative", "params.w_kl" ]
    RELEVANT_RUN_METRICS = [ "metrics.val_recon_loss_c", "metrics.val_recon_loss_s", "metrics.val_rec_ratio_to_time_mean" ]


    def __init__(self, root_dir, expid=None, from_cached=False, cache=None):
        
        self._rootdir = root_dir        
        self._cache_file = f"{self._rootdir}/region_assocs_df.pkl"
        self._collect_summaries(from_cached=from_cached, cache=cache)      
        
    
    @staticmethod
    def find_summary_files(root_dir):       
        summary_files = sorted(glob.glob(f"{root_dir}/summaries/*"))
        logging.info(f"Found {len(summary_files)} files under the folder {root_dir}/summaries")
        return summary_files
    

    @staticmethod
    def get_phenotype_name_summary_files(summary_file):
        # get basename w/o extension or prefix or suffix                        
        return re.sub(".*GWAS__|.*GWAS_|__region.*tsv|_region.*tsv", "", summary_file)


    @staticmethod
    def process_aha_summary_file(phenotype, file):
        magnitude, aha_segment = "_".join(phenotype.split("_")[:-1]), phenotype.split("_")[-1]
        logging.debug(f"Magnitude: {magnitude} / AHA segment: {aha_segment}")
        assocs_df = pd.read_csv(file).assign(chamber="LV", magnitude=magnitude, aha_segment=aha_segment, pheno=phenotype)
        return assocs_df

    
    @staticmethod
    def process_latent_variable_summary_file(phenotype, file):
        
        try:
            expid, run_id, zvar = phenotype.split("_")[0], phenotype.split("_")[1], phenotype.split("_")[2]
            expid = expid.replace("X", "")
            
            chamber = EnsembleGWASResults.expid_to_partition_mapping[expid]
            full_phenoname = f"{run_id}_{zvar}"
            logging.debug(f"Experiment: {expid} / Run: {run_id} / phenotype: {zvar}")
            assocs_df = pd.read_csv(file).assign(expid=expid, chamber=chamber, run=run_id, pheno=zvar, full_pheno=full_phenoname)
        
        except:
            zvar, run_id = phenotype.split("_")[0], phenotype.split("_")[1]
            if run_id not in EnsembleGWASResults.run_to_chamber_dict:
                print(f"{run_id} not in the run-to-chamber dictionary")
                return None
            chamber = EnsembleGWASResults.run_to_chamber_dict[run_id]
            full_phenoname = f"{run_id}_{zvar}"
            logging.debug(f"Run: {run_id} / phenotype: {zvar}")
            assocs_df = pd.read_csv(file).assign(run=run_id, pheno=zvar, full_pheno=full_phenoname, chamber=chamber)

        return assocs_df


    @staticmethod
    def process_traditional_phenotype_summary_file(phenotype, file):

        logging.debug(f"Phenotype: {phenotype}")
                    
        for possible_chamber in ["BV", "LV", "RV", "LA", "RA"]:
            if possible_chamber in phenotype:
                chamber = possible_chamber
                break
        assocs_df = pd.read_csv(file).assign(pheno=phenotype, chamber=chamber)
        return assocs_df


    @staticmethod
    def process_other_phenotype_summary_file(phenotype, file):

        run_id, pheno = pheno.split("_")[0], "_".join(pheno.split("_")[1:])
        chamber = EnsembleGWASResults.run_to_chamber_dict[run_id]
        logging.debug(f"Run: {run_id} / phenotype: {pheno}")
        assocs_df = pd.read_csv(file).assign(run=run_id, pheno=pheno, chamber=chamber)
        
        return assocs_df
    

    @staticmethod
    def _type_of_phenotype(phenotype):
        if "z0" in phenotype:
            return "latent"
        elif "aha" in phenotype.lower():
            return "aha"
        elif ("_ED_" not in phenotype) and ("_MEAN_" not in phenotype):
            return "traditional"
        else:
            return "other"
        

    @staticmethod
    def process_summary_file(file):

        phenotype = EnsembleGWASResults.get_phenotype_name_summary_files(file)
        
        type_of_phenotype = EnsembleGWASResults._type_of_phenotype(phenotype)
        
        if type_of_phenotype == "latent":
            assocs_df = EnsembleGWASResults.process_latent_variable_summary_file(phenotype, file)
        elif type_of_phenotype == "aha":
            assocs_df = EnsembleGWASResults.process_aha_summary_file(phenotype, file)
        elif type_of_phenotype == "traditional":
            assocs_df = EnsembleGWASResults.process_traditional_phenotype_summary_file(phenotype, file)
        elif type_of_phenotype == "other":
            assocs_df = EnsembleGWASResults.process_other_phenotype_summary_file(phenotype, file)
        
        return assocs_df
    

    @staticmethod
    def add_variable_type(df):
        
        # This is a way to determine whether we are dealing with latent variables
        assert "pheno" in df.columns, "Column 'pheno' not found in the DataFrame. Available columns are {df.columns}"

        if any(df.pheno.apply(lambda x: "z0" in x)):
            logging.info(f"Assigning dynamic/static label to phenotypes...")
            try:
                df["variable_type"] = df.apply(EnsembleGWASResults._get_variable_type, axis=1)
            except Exception as e:
                logging.error(e)        
        return df
    

    @staticmethod
    def add_is_variational_column(df):
        
        # This is a way to determine whether we are dealing with latent variables
        assert 'params.w_kl' in df.columns, f"Column 'params.w_kl' not found in the DataFrame. Available columns are {df.columns}"

        return df.assign(is_variational=lambda df: df['params.w_kl'].astype(float) > 0)


    def filter_results(self, query, inplace=True):       
        if inplace:
            self.region_assocs_df = self.region_assocs_df.query(query)
            return self.region_assocs_df
        else:
            return self.region_assocs_df.query(query)
        

    def count_runs(self):
        return ( self.region_assocs_df.
            drop_duplicates(subset=['run'], keep='first')[["chamber", "static_representative", "is_variational"]].
            value_counts().
            to_frame().reset_index().
            pivot(columns=["static_representative", "is_variational"], index="chamber", values="count").
            fillna(0).astype(int).
            rename_axis(columns=["static representative", "is variational?"]) )
    

    def _collect_summaries(self, from_cached, cache):
        
        assert os.path.exists(self._rootdir), f"""
            Folder {self._rootdir} does not exist. Other folders under the same parent folder are: {', '.join(os.listdir(os.path.dirname(self._rootdir)))}
        """
        
        if from_cached and os.path.exists(self._cache_file):
            logging.info(f"Loading cached results from {self._cache_file}")
            self.region_assocs_df = pd.read_pickle(f"{self._cache_file}")
        else:     
            if not os.path.exists(self._cache_file):
                logging.info(f"File {self._cache_file} does not exist. Results will be cached.")
                cache = True

            zvar_gwas_assocs = []
            for file in tqdm(EnsembleGWASResults.find_summary_files(self._rootdir)):            
                assocs_df = EnsembleGWASResults.process_summary_file(file)                
                if assocs_df is not None:
                    zvar_gwas_assocs.append(assocs_df)
                
            logging.info(f"Collected GWAS summary data for {len(zvar_gwas_assocs)} phenotypes.")
    
            logging.debug(f"Concatenating...")
            region_assocs_df = pd.concat(zvar_gwas_assocs) 
            region_assocs_df = EnsembleGWASResults.add_variable_type(region_assocs_df)            

            run_additional_info = self.runs_df[EnsembleGWASResults.RELEVANT_RUN_HPARAMS + EnsembleGWASResults.RELEVANT_RUN_METRICS]
            
            region_assocs_df = pd.merge(region_assocs_df, run_additional_info, left_on='run', right_on='run_id')
            region_assocs_df = EnsembleGWASResults.add_is_variational_column(region_assocs_df)

            region_assocs_df = region_assocs_df.rename({
                "params.dataset_static_representative": "static_representative",
                "metrics.val_recon_loss_c": "msd_static",
                "metrics.val_recon_loss_s": "msd_dynamic",
                "metrics.val_rec_ratio_to_time_mean": "rec_ratio"
            }, axis=1)

            if cache:
                logging.info(f"Caching results to {self._rootdir}/region_assocs_df.pkl")
                region_assocs_df.to_pickle(f"{self._rootdir}/region_assocs_df.pkl")
    
            self.region_assocs_df = region_assocs_df
            return self.region_assocs_df

             
    def keep_top_n_per_chamber(self, n, inplace=False):
        
        rec_losses = []
        
        for chamber in self.region_assocs_df.chamber.unique():
            results_for_chamber = self.region_assocs_df. query("chamber == @chamber")
            unique_values = results_for_chamber.msd_dynamic.unique()[:n]
            logging.info(f"Chamber {chamber}: Keeping {len(unique_values)} runs with the best performance.")
            rec_losses.extend(unique_values)
        
        # rec_losses = set(rec_losses)
        if inplace:
            logging.info(f"Only the best (at most) {n} runs per chamber with the best performance will be kept. Filtering...")
            self.region_assocs_df = self.region_assocs_df[self.region_assocs_df.msd_dynamic.apply(lambda x: x in rec_losses)]
        else:            
            return self.region_assocs_df[self.region_assocs_df.msd_dynamic.apply(lambda x: x in rec_losses)]


    @staticmethod
    def _get_variable_type(row):
                
        n_static_variables = int(EnsembleGWASResults.latent_dim_c_dict.get(row.run, -1))
        if n_static_variables == -1:
            print(f"Run {row.run} not found in the dictionary of content latent dimension.")
            return None
        zvar = int(row.pheno[-2:])
        return "static" if zvar < n_static_variables else "dynamic"


    @staticmethod
    def filter_valid_regions(df):
        print(f"{df.shape=}")
        if "region" in df.columns:
            df_filtered = df.where(df.region.isin(EnsembleGWASResults.valid_regions))
            return df_filtered
        # return df.loc[[x in EnsembleGWASResults.valid_regions for x in df.index.get_level_values("region")]]


    def loci_count(self, p_threshold=5e-8, attributes=['chamber']):
        
        loci_summary_df = generate_loci_summary(self.region_assocs_df, p_threshold=p_threshold) # self.loci_summary(p_threshold=p_threshold, )
        
        assert all([attributes in loci_summary_df.columns for attributes in attributes]), f"""
            Columns {set(attributes) - set(loci_summary_df.columns)} not found in the DataFrame.
            Columns are {loci_summary_df.columns.to_list()}.
        """

        ( GROUPBY_COLS := ["region", "variable_type"] ).extend( attributes )

        assert set(GROUPBY_COLS).issubset(loci_summary_df.columns), f"""
            Columns {set(GROUPBY_COLS) - set(loci_summary_df.columns)} not found in the DataFrame.
            Columns are {loci_summary_df.columns.to_list()}.
        """ 
        
        loci_count_df = ( loci_summary_df.
            groupby(GROUPBY_COLS).
            aggregate({"count":"count", "min_P": "min"}).
            rename({"CHR":"count", "P":"min_P"}, axis=1).
            sort_values("count", ascending=False) )
            
        return loci_count_df
    
    
    def show_counts(self, count_thr = 5, pvalue_thr=1.5e-10):
        region_count_df = self.loci_count()
        # display( region_count_df.query("count >= @count_thr or min_p < @pvalue_thr") )
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
    

    @classmethod
    def create_count_table_tex(counts_df, tex_file=None):
        
        '''
          Generates a LaTeX table for significant loci.
          tex_file: Optional filename to save the LaTeX table.
        '''
        
        assert isinstance(counts_df, pd.DataFrame), "counts_df must be a pandas DataFrame"
        if tex_file is not None:
            assert isinstance(tex_file, str), "tex_file must be a string or None"

        assert "min_P" in counts_df.columns, "'min_P' column is missing"

        regions   = self.get_significant_regions()
        snp_data  = self._snp_data()
    
        build_pvalue_str = lambda x: f"${round(float(x[0]), 1)} \\times 10^{{{x[1]}}}$"
        build_region_str = lambda region: f'{EnsembleGWASResults.regions_df.loc[region, "start"]}-{EnsembleGWASResults.regions_df.loc[region, "stop"]}'
        build_eaf_pctg_str    = lambda x: f"{(100*x):.1f}"
        region_to_candidate_gene = lambda gene: self.loci_mapping_df.loc[gene, "candidate_gene"]
        chromosome_from_region = lambda region: EnsembleGWASResults.regions_df.loc[region, "chr"] 

        counts_df = counts_df.loc[regions]
        counts_df = counts_df.assign(min_P=counts_df["min_P"].astype(str).str.split("e").apply(build_pvalue_str))
        counts_df = counts_df.reset_index()
        counts_df = counts_df.assign(candidate_gene=counts_df["region"].map(region_to_candidate_gene))
        counts_df = counts_df.assign(chr= counts_df.region.map(chromosome_from_region))
        counts_df = counts_df.merge(snp_data, on="region")
        counts_df = counts_df.assign(region=counts_df["region"].map(build_region_str))
        counts_df = counts_df.sort_values("count", ascending=False)

        counts_df = counts_df[["chr", "region", "candidate gene", "count", "min_P", "SNP", "a_0", "a_1", "AF", "BETA", "SE"]]
        counts_df = counts_df.rename(columns={"chr": "chr.", "min_P": "min. $p$-value", "a_0": "NEA", "a_1": "EA", "AF": "EAF"})
        counts_df = counts_df.assign(EAF=counts_df.EAF.apply(build_eaf_str))
    
        assert not counts_df.empty, "counts_df is empty after filtering"        
        assert hasattr(cls, 'loci_mapping_df'), "'loci_mapping_df' attribute is missing in the class"

        scale_beta = counts_df["BETA"].abs().between(0.01, 0.2).all()
    
        if scale_beta:
            counts_df["BETA"] = (counts_df["BETA"] * 100).round(3)
            counts_df["SE"] = (counts_df["SE"] * 100).round(3)
            beta_header = r"$\hat{\beta} \pm \text{se}(\hat{\beta})(\times 100)$ "
        else:
            counts_df["BETA"] = counts_df["BETA"].round(3)
            counts_df["SE"] = counts_df["SE"].round(3)
            beta_header = r"$\hat{\beta} \pm \text{se}(\hat{\beta})$"
    
        counts_df["BETA_SE"] = counts_df.apply(lambda row: f"${row['BETA']} \pm {row['SE']}$", axis=1)
        counts_df = counts_df.drop(columns=["BETA", "SE"])
    
        table_code = counts_df.to_latex(
            escape=False,
            index=False,
            column_format="rllrllllr",
            caption="GWAS Results"
        )
    
        table_code = table_code.replace("_", "\\_").replace("BETA\_SE", beta_header)
    
        if tex_file is not None:
            print(f"Creating output file in {tex_file}")
            with open(tex_file, "wt") as table_f:
                table_f.write(table_code)
    
        return table_code

    
    def _snp_data(self):
        
        region_assocs_df = self.region_assocs_df.copy()
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
        
        return by_variable_type_df
    
    
    def get_lead_snps(self):
        
        idx_min = self.region_assocs_df.groupby("region").P.idxmin()
        idx_min = idx_min[self.get_significant_regions()]        
        
        return self.region_assocs_df.iloc[idx_min][["CHR", "BP", "region", "SNP", "AF", "P"]].\
            reset_index(drop=True).\
            sort_values(["CHR", "region"]).\
            sort_values(["CHR", "BP"])


    def get_counts_per_attribute(self, p_threshold=5e-8, attributes=["variable_type"]):
        
        COL_ORDER = [(variable_type, chamber) for variable_type in ["dynamic", "static"] for chamber in ["BV", "LV", "RV", "LA", "RA"]]        

        counts_by_attribute = self.loci_count(p_threshold=p_threshold).\
            reset_index().\
            pivot(index="region", values="count", columns=attributes).\
            fillna(0).astype(int)
        
        COL_ORDER = [ indices for indices in COL_ORDER if indices in counts_by_attribute.columns ]
        counts_by_attribute = counts_by_attribute[COL_ORDER]
        
        ordered_by_dynamic   = counts_by_attribute["dynamic"].sum(axis=1).sort_values(ascending=False).index
        total_counts_dynamic = counts_by_attribute.loc[ordered_by_dynamic]["dynamic"].sum(axis=1)
        total_counts_static  = counts_by_attribute.loc[ordered_by_dynamic]["static"].sum(axis=1)
        
        ratio_dyn_to_stat = (total_counts_dynamic - total_counts_static) / (total_counts_dynamic + total_counts_static)
        possible_order = (ratio_dyn_to_stat[(total_counts_dynamic+total_counts_static) > 3]).sort_values(ascending=False).index
        
        counts_by_attribute = counts_by_attribute.loc[possible_order]
        counts_by_attribute.index = [self.loci_mapping_df.loc[region, "candidate_gene"] for region in counts_by_attribute.index]
        
        return counts_by_attribute


    def get_counts_per_chamber(self, p_threshold=5e-8):
        return self.get_counts_per_attribute(p_threshold=p_threshold, attributes=["variable_type", "chamber"])