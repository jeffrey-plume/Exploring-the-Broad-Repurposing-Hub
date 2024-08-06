import base64
import pandas as pd
import requests
import numpy as np
from math import ceil

def getFPfromCID(listofCIDs):
    """
    This function takes a list of PubChem CID identifiers and returns a dataframe with the PubChem fingerprints associated with each CID
    """

    def get_pubchem_fingerprint(cid):
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/Fingerprint2D/JSON" #can load multiple comma seperated cids
        response = requests.get(url)
        if response.status_code == 200: # 200 indicates that the request was sucessful
            data = response.json()
            return(data)
        return None
    
    def decode_base64_pubchem_fingerprint(base64_fingerprint):  #this stupid thing took me forever.
        try:
            binary_data = base64.b64decode(base64_fingerprint) #FPs are base64 encoded and must be decoded
            binary_fingerprint = ''.join(f'{byte:08b}' for byte in binary_data)[32:913] # the first 32 and last 6 bits are padding
            bit_vector = np.array([int(bit) for bit in binary_fingerprint])
            return bit_vector
        except ValueError as e:
            print(f"Error decoding fingerprint: {e}")
            return None
    
    
    cidStr = list(map(lambda x: str(int(x)), listofCIDs)) #string like integers comma seperated.  I probably could have done this better with an f-string, maybe later.
    cidList=[]
    for i in range(int(ceil(len(cidStr)/100))): #bin into groups of 100
        cidList.append(",".join(cidStr[i*100:i*100+100]))
    
    base64_fingerprint=[]
    fingerprints = {}
    
    #This code loops through the binned cids the loops through the resulting json files to decode the base64 fingerprints.
    for cids in cidList:    
        base64_fingerprint.append(get_pubchem_fingerprint(cids))
    
    for j in range(len(base64_fingerprint)):  
        for k in range(len(base64_fingerprint[j]['PropertyTable']['Properties'])):
            if base64_fingerprint[j]: 
                fpEnc = base64_fingerprint[j]['PropertyTable']['Properties'][k]['Fingerprint2D']
                fingerprints[base64_fingerprint[j]['PropertyTable']['Properties'][k]['CID']] = decode_base64_pubchem_fingerprint(fpEnc)
    
    #combined an exported to csv.  This code should run fine, but heavy users will be throttle by Pubchem.
    df = pd.DataFrame.from_dict(fingerprints, orient='index')
    df.columns = pd.read_table('https://raw.githubusercontent.com/cdk/orchem/master/doc/pubchem_fingerprints.txt', skiprows = 36, nrows = 916).dropna(subset = 'Bit Substructure').query('`Bit Position` != "Bit Position"').loc[:, 'Bit Substructure'].values
    return(df)