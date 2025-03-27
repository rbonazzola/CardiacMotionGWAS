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


def summarize_loci_hits(
    assocs_df,
    p_threshold=5e-8,
    groupby=["region", "run", "variable_type"],
    collapse_attributes=None,
    extra_columns=None
):
    """
    Summarizes significant loci with P < p_threshold.

    Parameters
    ----------
    assocs_df : pd.DataFrame
        DataFrame containing GWAS association results.
    p_threshold : float
        Significance threshold for filtering loci.
    groupby : list of str
        Columns to group by in the initial summary.
    collapse_attributes : list of str or None
        If provided, perform a second aggregation grouped by these attributes.
        Useful for collapsing across multiple runs or configurations.
    extra_columns : list of str or None
        Additional columns to include in the grouping (e.g., 'chamber').

    Returns
    -------
    pd.DataFrame
        Summary of significant loci per group, with columns:
            - 'count': number of significant entries or grouped hits
            - 'min_P': minimum p-value in the group
    """
    df = assocs_df.query("P < @p_threshold")

    if extra_columns:
        groupby = groupby + extra_columns

    lead_snp_df = df.loc[df.groupby(groupby)["P"].idxmin()][groupby + ["SNP"]]
    lead_snp_df = lead_snp_df.rename(columns={ "SNP": "lead_SNP" })

    # Filter by significance threshold and group
    agg_df = ( df.
        groupby(groupby).
        agg(
            count=("CHR", "count"),            
            min_P=("P", "min")).
        reset_index()
    )

    summary_df = pd.merge(agg_df, lead_snp_df, on=groupby)

    # Optional: collapse across runs or other attributes
    if collapse_attributes:

        ignore_columns = ["min_P", "count", "lead_SNP"] + collapse_attributes
        groupby2 = summary_df.columns.difference( ignore_columns ).tolist()

        assert all(attr in summary_df.columns for attr in collapse_attributes), (
            f"Some collapse attributes not found: {set(collapse_attributes) - set(summary_df.columns)}"
        )

        # Find the best hit per group (across the collapse_attributes)        
        cols = list(set(groupby) - set(collapse_attributes))
        lead_snp = summary_df.loc[summary_df.groupby(cols)["min_P"].idxmin(), cols + ["lead_SNP"]]

        summary_df = ( summary_df
            .groupby(groupby2)
            .agg({"count": "count", "min_P": "min"})
            .reset_index()
            .sort_values("count", ascending=False) )
    else:
        summary_df = summary_df.sort_values("count", ascending=False).sort_values("min_P", ascending=True)

#    from IPython import embed; embed()
    summary_df = summary_df.merge(lead_snp, on=cols)

    # Optional filtering step for valid regions (if using EnsembleGWASResults class)
    if hasattr(EnsembleGWASResults, "filter_valid_regions"):
        summary_df = EnsembleGWASResults.filter_valid_regions(summary_df)

    summary_df = summary_df.dropna()
    summary_df = summary_df.assign(count=summary_df["count"].astype(int))

    return summary_df



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
    del k, v

    loci_mapping_df = fetch_loci_mapping()
    valid_regions = loci_mapping_df.index[loci_mapping_df["duplicated"].isnull()]
    
    possible_chambers = { "BV", "LV", "RV", "LA", "RA" } # , "AO" }

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
                
        assert "pheno" in df.columns, "Column 'pheno' not found in the DataFrame. Available columns are {df.columns}"
       
        # This is a way to determine whether we are dealing with latent variables
        is_latent_variable = lambda x: "z0" in x
                
        if any(df.pheno.apply(is_latent_variable)):
            logging.info(f"Assigning dynamic/static label to phenotypes...")
            try:
                df["variable_type"] = df.apply(EnsembleGWASResults._get_variable_type, axis=1)
            except Exception as e:
                logging.error(e)
                
        return df
    

    @staticmethod
    def add_is_variational_column(df):
                
        assert 'params.w_kl' in df.columns, f"Column 'params.w_kl' not found in the DataFrame. Available columns are {df.columns}"

        return df.assign(is_variational=lambda df: df['params.w_kl'].astype(float) > 0)
    

    def filter_results(self, query, inplace=True):
        
        """
        Filters the .region_assocs_df DataFrame according to the query. 
        It uses the same syntax as pd.DataFrame.query
        """
        
        if inplace:
            self.region_assocs_df = self.region_assocs_df.query(query)
            return self.region_assocs_df
        else:
            return self.region_assocs_df.query(query)
        

    def count_runs(self):

        """
        Returns a DataFrame with the number of runs per "chamber", "static representative" and "is_variational".
        """
        
        return ( self.region_assocs_df.
            drop_duplicates(subset=['run'], keep='first').
            loc[:, ["chamber", "static_representative", "is_variational"]].
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
                logging.info(f"Cache'ing results to {self._rootdir}/region_assocs_df.pkl")
                region_assocs_df.to_pickle(f"{self._rootdir}/region_assocs_df.pkl")
    
            self.region_assocs_df = region_assocs_df
            return self.region_assocs_df

             
    def keep_top_n_per_chamber(self, n, inplace=False):

        """
        Keeps the top n runs per chamber with the best reconstruction performance, as measured by the dynamic mean squared deviation.
        """
        
        rec_losses = []
        
        for chamber in self.region_assocs_df.chamber.unique():
            results_for_chamber = self.region_assocs_df. query("chamber == @chamber")
            unique_values = results_for_chamber.msd_dynamic.unique()[:n]
            logging.info(f"Chamber {chamber}: Keeping {len(unique_values)} runs with the best performance.")
            rec_losses.extend(unique_values)
        
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
        
        """
        Filters out regions that are duplicated in the spreadsheet
        """
        
        if "region" in df.columns:
            df_filtered = df.where(df.region.isin(EnsembleGWASResults.valid_regions))
            return df_filtered
        else:
            logging.error(f"Column 'region' not found in the DataFrame. Available columns are {df.columns}")
            return df

    
    def show_counts(self, count_thr = 5, pvalue_thr=1.5e-10):
        region_count_df = self.loci_count()
        # display( region_count_df.query("count >= @count_thr or min_p < @pvalue_thr") )
        display( region_count_df[(region_count_df["count"] >= count_thr) | (region_count_df.min_P < pvalue_thr)])
          

    def summarize_loci_hits(self, p_threshold=5e-8, groupby=["region", "run", "variable_type"], collapse_attributes=None, extra_columns=None):
        __doc__ = summarize_loci_hits.__doc__

        return summarize_loci_hits(self.region_assocs_df, p_threshold=p_threshold, groupby=groupby, collapse_attributes=collapse_attributes, extra_columns=extra_columns)


    def get_results_for_region(self, region, top_n=20, only_dynamic=False, only_static=False, exp_ids=None):
    
        def color_negative_red(val):
            color = 'red' if val < 0 else 'black'
            return 'color: %s' % color
        
        df = self.region_assocs_df.query("region == @region").sort_values("P").head(top_n)
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
    
    
    @staticmethod
    def get_suggestive_regions(counts_df, GW_P_THRESHOLD=5e-8, COUNT_THRESHOLD=1, SW_P_THRESHOLD=1.5e-10):
        
        # region_count_df = self.loci_count()
        sugg_condition = ((counts_df.min_P < GW_P_THRESHOLD) & (counts_df['count'] >= COUNT_THRESHOLD)) & (counts_df.min_P > SW_P_THRESHOLD)
        suggestive_regions = sorted(counts_df[sugg_condition]['region'])
        suggestive_regions = list(set(suggestive_regions))
        return suggestive_regions
            
    
    @staticmethod
    def get_significant_regions(counts_df, GW_P_THRESHOLD=5e-8, COUNT_THRESHOLD=5, SW_P_THRESHOLD=1.5e-10):

        signif_condition = ((counts_df.min_P < GW_P_THRESHOLD) & (counts_df['count'] >= COUNT_THRESHOLD)) | (counts_df.min_P < SW_P_THRESHOLD)
        significant_regions = sorted(counts_df[signif_condition]['region'])
        significant_regions = list(set(significant_regions))
        return significant_regions
    

    def create_count_table_tex(self, counts_df, tex_file=None, caption=None, label=None):
        
        '''
          Generates a LaTeX table for significant loci.
          tex_file: Optional filename to save the LaTeX table.
        '''
        
        assert isinstance(counts_df, pd.DataFrame), "counts_df must be a pandas DataFrame"
        if tex_file is not None:
            assert isinstance(tex_file, str), "tex_file must be a string or None"

        assert "min_P" in counts_df.columns, "'min_P' column is missing"

        # regions   = self.get_significant_regions(counts_df)
        snp_data  = self._snp_data()        
    
        build_pvalue_str         = lambda x:      f"${round(float(x[0]), 1)} \\times 10^{{{x[1]}}}$"
        build_region_str         = lambda region: f'{EnsembleGWASResults.regions_df.loc[region, "start"]}-{EnsembleGWASResults.regions_df.loc[region, "stop"]}'
        build_eaf_pctg_str       = lambda eaf:    f"{(100*eaf):.1f}"
        region_to_candidate_gene = lambda gene:   self.loci_mapping_df.loc[gene, "candidate_gene"]
        chromosome_from_region   = lambda region: EnsembleGWASResults.regions_df.loc[region, "chr"] 

        counts_df = ( counts_df.
            assign(
                min_P=lambda df: df["min_P"].astype(str).str.split("e").apply(build_pvalue_str),
                candidate_gene=lambda df: df["region"].map(region_to_candidate_gene),
                chr=lambda df: df["region"].map(chromosome_from_region),
                region=lambda df: df["region"].map(build_region_str)).
            merge(snp_data.drop("region", axis=1), left_on="lead_SNP", right_on="SNP").
            sort_values("count", ascending=False).
            loc[:, ["chr", "region", "candidate_gene", "count", "min_P", "SNP", "a_0", "a_1", "AF", "BETA", "SE"]].
            rename(columns={
                "chr": "chr.",
                "min_P": "min. $p$-value",
                "a_0": "NEA",
                "a_1": "EA",
                "AF": "EAF",
                "candidate_gene": "candidate gene"
            }).
            assign(EAF=lambda df: df["EAF"].apply(build_eaf_pctg_str))
        )
    
        assert not counts_df.empty, "counts_df is empty after filtering"        
        assert hasattr(self, 'loci_mapping_df'), "'loci_mapping_df' attribute is missing in the class"

        scale_beta = counts_df["BETA"].abs().between(0.01, 0.2).all()
    
        if scale_beta:
            counts_df["BETA"] = (counts_df["BETA"] * 100).round(2)
            counts_df["SE"] = (counts_df["SE"] * 100).round(2)
            beta_header = r"$\hat{\beta} \pm \text{se}(\hat{\beta})(\times 100)$ "
        else:
            counts_df["BETA"] = counts_df["BETA"].round(2)
            counts_df["SE"] = counts_df["SE"].round(2)
            beta_header = r"$\hat{\beta} \pm \text{se}(\hat{\beta})$"
    
        counts_df["BETA_SE"] = counts_df.apply(lambda row: f"${row['BETA']} \pm {row['SE']}$", axis=1)
        counts_df = counts_df.drop(columns=["BETA", "SE"])
    
        table_code_inner = counts_df.to_latex(
            escape=False,
            index=False,
            column_format="rccccccccc"
        )

        # Optional label + caption
        table_label = r"\label{table:gwas_cardiac_motion}"
        table_caption = r"\caption{GWAS summary statistics for cardiac motion phenotypes.}"

        # Clean column name
        table_code_inner = table_code_inner.replace("_", r"\_").replace("BETA\_SE", beta_header)

        # Wrap in table* + adjustbox
        table_code = (
            r"\begin{table*}" "\n"
            r"\begin{adjustbox}{width=\textwidth}%" "\n"
            r"\centering" "\n"
            + table_code_inner.strip() + "\n"
            r"\end{adjustbox}" "\n"
            + f"\\label{{{label}}}\n"
            + f"\\caption{{{caption}}}\n"
            r"\end{table*}"
        )

        if tex_file is not None:
            print(f"Creating output file in {tex_file}")
            with open(tex_file, "wt") as table_f:
                table_f.write(table_code)

        return table_code

    
    def _snp_data(self):
        
        if not hasattr(self, "snp_data"):
            region_assocs_df = self.region_assocs_df.copy()
            snp_data = region_assocs_df.loc[~region_assocs_df.sort_values("SNP").duplicated("SNP")]
            snp_data = snp_data.loc[:,["region", "SNP", "BP", "AF", "a_0", "a_1", "BETA", "SE"]]
            self.snp_data = snp_data
        return self.snp_data
        
    
    def assocs_per_variable_type(self, type="d"):
        
        by_variable_type_df = ( results.loci_count()[["min_P"]].
            reset_index().
            pivot(index="region", columns="variable_type", values="min_P")
        )
        
        if type == "d":
            return by_variable_type_df.sort_values("dynamic")
        else:
            return by_variable_type_df.sort_values("static")
        
        return by_variable_type_df
    
    
    def get_lead_snps(self):
        
        idx_min = self.region_assocs_df.groupby("region").P.idxmin()
        idx_min = idx_min[self.get_significant_regions()]        
        
        return ( self.region_assocs_df.
            iloc[idx_min].
            loc[:, ["CHR", "BP", "region", "SNP", "AF", "P"]].
            reset_index(drop=True).
            sort_values(["CHR", "region"]).
            sort_values(["CHR", "BP"]) ) 


    def get_counts_per_attribute(self, p_threshold=5e-8, attributes=["variable_type"]):
        
        """
        """
        
        COL_ORDER = [(variable_type, chamber) for variable_type in ["dynamic", "static"] for chamber in ["BV", "LV", "RV", "LA", "RA"]]        

        counts_by_attribute = self.loci_count(p_threshold=p_threshold).\
            reset_index().\
            pivot(index="region", values="count", columns=attributes).\
            fillna(0).astype(int)
        
        COL_ORDER = [ indices for indices in COL_ORDER if indices in counts_by_attribute.columns ]

        counts_by_attribute  = counts_by_attribute[COL_ORDER]
        ordered_by_dynamic   = counts_by_attribute["dynamic"].sum(axis=1).sort_values(ascending=False).index
        total_counts_dynamic = counts_by_attribute.loc[ordered_by_dynamic]["dynamic"].sum(axis=1)
        total_counts_static  = counts_by_attribute.loc[ordered_by_dynamic]["static"].sum(axis=1)
        
        ratio_dyn_to_stat = (total_counts_dynamic - total_counts_static) / (total_counts_dynamic + total_counts_static)
        possible_order    = (ratio_dyn_to_stat[(total_counts_dynamic+total_counts_static) > 3]).sort_values(ascending=False).index
        
        counts_by_attribute = counts_by_attribute.loc[possible_order]
        counts_by_attribute.index = [ self.loci_mapping_df.loc[region, "candidate_gene"] for region in counts_by_attribute.index ]
        
        return counts_by_attribute


    def get_counts_per_chamber(self, p_threshold=5e-8):
        return self.get_counts_per_attribute(p_threshold=p_threshold, attributes=["variable_type", "chamber"])