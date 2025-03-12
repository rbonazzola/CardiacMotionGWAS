import pandas as pd
import requests
import gwas_pipeline

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

get_ld_indep_regions = gwas_pipeline.get_ld_indep_regions