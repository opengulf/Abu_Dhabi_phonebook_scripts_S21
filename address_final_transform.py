import os
import json
import unicodedata
import string
import re
from collections import Counter
from datetime import datetime
import Levenshtein as lev

directory = r'<PATH-TO-DIRECTORY-CONTAINING-DELIMITED-JSON-ENTRIES>'
par_directory = r'<PATH-TO-DIRECTORY-CONTAINING-SWAP-JSON-FILES>'


with open(os.path.join(par_directory, 'location_abbreviation_modified.json')) as f:
    abbs = json.loads(f.read())
    f.close()
with open(os.path.join(par_directory, 'common_corrections.json')) as f:
    corrections = json.loads(f.read())
    f.close()
with open(os.path.join(par_directory, 'normalized_locations.json')) as f:
    normalized = json.loads(f.read())
    f.close()
with open(os.path.join(par_directory, 'occupations_common_corrections.json')) as f:
    occ_corrections = json.loads(f.read())
    f.close()
with open(os.path.join(par_directory, 'occupations_abbreviations.json')) as f:
    occ_abbreviations = json.loads(f.read())
    f.close()
with open(os.path.join(par_directory, 'locations_cut.txt')) as f:
    token_ignore_list = [i.strip() for i in f.readlines()]


def simplify_token(in_token, tp='upper'):
    in_token = in_token.strip().replace('\n', '').replace('.','').replace("'","").replace(',','').replace('-','').replace(':','').replace('(','').replace(')','').replace('¢','')
    if tp == 'lower':
        return in_token.lower()
    else:
        return in_token

## Address swaps

def corr_swap(intoken):
    if intoken in corrections:
        return corrections[intoken]
    else:
        return intoken

def abbs_swap(intoken):
    if intoken in abbs:
        return abbs[intoken]
    else:
        return intoken

def sp_abb_swap(in_full_address):
    if re.search(r'^\d+\w', in_full_address):
        digits = re.findall(r'^\d+', in_full_address)[0]
        return in_full_address.replace(digits, digits + " ")
    else:
        for space_swap in spaced_normal_swaps:
            if re.search(space_swap, in_full_address):
                in_full_address = re.sub(space_swap, spaced_normal_swaps[space_swap], in_full_address)
        return in_full_address

def location_normalizer(inaddress):
    for space_swap in spaced_normal_swaps:
        if re.search(space_swap, inaddress):
            inaddress = re.sub(space_swap, spaced_normal_swaps[space_swap], inaddress)
    if re.search(r'\s\d+\sth\s+|\s\d+\sth$|\s\d+\snd\s+|\s\d+\snd$|\s\d+\sst\s+|\s\d+\sst$', inaddress, re.I):     # Fix for extra space showing up between number streets and the ordinal letters
        inaddress = re.sub(r'\sth$', 'th', inaddress)
        inaddress = re.sub(r'\sth\s', 'th ', inaddress)
        inaddress = re.sub(r'\snd$', 'nd', inaddress)
        inaddress = re.sub(r'\snd\s', 'nd ', inaddress)
        inaddress = re.sub(r'\sst$', 'st', inaddress)
        inaddress = re.sub(r'\sst\s', 'st ', inaddress)
    if re.search(r'^\d+\sh\s\d+|^\d+\s8\s\d+', inaddress):
        inaddress = re.sub(r'\sh\s|\s8\s', ' east ', inaddress)  # Fix for E as in East being OCRd as "H" or "8" after an address number
    ret_address = []
    for add_token in inaddress.split():
        if add_token != "g":       # A fix for a mysterious problem with an added "g" in Greenwich addresses
            if add_token in normalized:
                ret_address.append(normalized[add_token])
            else:
                ret_address.append(add_token)
    return ' '.join([i.title() if not re.search(r'^\d', i) else i for i in ret_address])



## Occupation swaps

def occ_corr_swap(intoken, checkcase=True):
    if intoken.lower() in occ_corrections:
        if re.search(r'^[A-Z]',intoken) and checkcase == True:
            return occ_corrections[intoken.lower()].title()
        else:
            return occ_corrections[intoken.lower()]
    else:
        return intoken

def occ_abbs_swap(intoken, checkcase=True):
    if intoken.lower() in occ_abbreviations:
        if re.search(r'^[A-Z]',intoken) and checkcase == True:
            return occ_abbreviations[intoken.lower()].title()
        else:
            return occ_abbreviations[intoken.lower()]
    else:
        return intoken

def occ_suffix_swap(intoken):
    for swap in [(r'mkr$', 'maker'), (r'gds$', 'goods'), (r'furngh$', 'furnishing'),
                (r'ingh$', 'inghouse'), (r'bldr', 'builder'), (r'bks$', 'books'),
                (r'hgr$','hanger'),(r'manuf$','manufacturer'),(r'manufs$','manufacturers'),
                (r'makr$','maker'),(r'matls$','materials'),(r'wkr$','worker'),(r'mker$','maker'),
                (r'fdr$','founder'),(r'bkrs$','brokers'),(r'mks$','makers'),(r'wshr$','washer')]:
        intoken = re.sub(swap[0], swap[1], intoken)
    return intoken


## N-Gram maker for addresses (that is, swaps are address-specific)

def ngram(in_token, n):
    address_tokens = in_token.strip().lower().replace('#','').translate(str.maketrans('', '', string.punctuation))
    address_tokens = unicodedata.normalize('NFKD', address_tokens).encode('ascii', 'ignore').decode()
    tokenjoin = sp_abb_swap(' '.join([abbs_swap(corr_swap(token)) for token in address_tokens.split()]))
    tokenjoin = tokenjoin.replace(' ', '')
    if len(tokenjoin) > 1:
        grams = []
        for i in range(0,len(tokenjoin)):
            try:
                if len(tokenjoin[i:i+n]) == n:
                    grams.append(tokenjoin[i:i+n])
            except:
                break
        grams = list(set(grams))
        grams.sort()
        return ''.join(grams)
    else:
        return in_token.lower()



spaced_normal_swaps = {
    r"g\swich": "greenwich",
    r"b\sway": "broadway",
    r"\se\sr": " east river",
    r"gd\sblvd": "grand boulevard and concourse",
    r"\sj\sc": " jersey city",
    r"\sl\si": " long island",
    r"\sl\si\sc": " long island city",
    r"\sn\sj": " new jersey",
    r"\sn\sr": " north river",
    r"\sp\so": " post office",
    r"\sr\sr": " railroad",
    r"\ss\si": " staten island",
    r"tre\smont": "tremont",
    r"gt\sjones": "great jones",
    r"\s8\si": " long island",
    r"staten\si$": "staten island",
    r"staten\sis$": "staten island",
    r"staten\sisl$": "staten island",
    r"stat\si$": "staten island",
    r"stat\sis$": "staten island",
    r"\sc\s": " corner ",
    r"\sn\s": " near "
}

dirs = ["4adf9ec0-317a-0134-03ad-00505686a51c",
        "4ae3cb40-317a-0134-489d-00505686a51c",
        "4ae76b60-317a-0134-b849-00505686a51c",
        "4aea8af0-317a-0134-2393-00505686a51c",
        "4aed8a80-317a-0134-28a4-00505686a51c",
        "4af0c6f0-317a-0134-e90c-00505686a51c",
        "4af3b880-317a-0134-bda8-00505686a51c",
        "4af6a690-317a-0134-5947-00505686a51c",
        "4afa0510-317a-0134-cf84-00505686a51c",
        "4afd6280-317a-0134-575a-00505686a51c",
        "4b00bf60-317a-0134-32d0-00505686a51c",
        "4b0419c0-317a-0134-7464-00505686a51c",
        "4b073d20-317a-0134-af68-00505686a51c",
        "4b0aa870-317a-0134-712b-00505686a51c",
        "4b0e13f0-317a-0134-6578-00505686a51c",
        "4b119360-317a-0134-9131-00505686a51c",
        "4b154340-317a-0134-afd3-00505686a51c",
        "4b18f080-317a-0134-fded-00505686a51c",
        "4b336e60-317a-0134-1e9b-00505686a51c",
        "4b36edd0-317a-0134-eedc-00505686a51c",
        "4b3a14d0-317a-0134-011c-00505686a51c",
        "4b3d0590-317a-0134-1631-00505686a51c",
        "4b4009d0-317a-0134-949b-00505686a51c",
        "4b437600-317a-0134-6db3-00505686a51c",
        "4b47b740-317a-0134-ad0b-00505686a51c",
        "4b4b2b90-317a-0134-6800-00505686a51c",
        "4b4e8300-317a-0134-fb8c-00505686a51c",
        "4b51d420-317a-0134-aa50-00505686a51c",
        "4b5532f0-317a-0134-52ca-00505686a51c",
        "4b58d200-317a-0134-d2aa-00505686a51c",
        "4b5c40e0-317a-0134-e9c9-00505686a51c",
        "4b5ff0e0-317a-0134-7e27-00505686a51c",
        "4b63a460-317a-0134-d3bd-00505686a51c",
        "4b66b460-317a-0134-8cb2-00505686a51c",
        "4b69a410-317a-0134-a570-00505686a51c",
        "4b6c95d0-317a-0134-f4e4-00505686a51c",
        "4b6f8210-317a-0134-ff86-00505686a51c",
        "4b728f10-317a-0134-8c07-00505686a51c",
        "4b8e3f70-317a-0134-721a-00505686a51c",
        "4b939190-317a-0134-d1d5-00505686a51c"]


yearcross = {"4adf9ec0-317a-0134-03ad-00505686a51c":"1850-51",
"4ae3cb40-317a-0134-489d-00505686a51c":"1851-52",
"4ae76b60-317a-0134-b849-00505686a51c":"1852-53",
"4aea8af0-317a-0134-2393-00505686a51c":"1853-54",
"4aed8a80-317a-0134-28a4-00505686a51c":"1854-55",
"4af0c6f0-317a-0134-e90c-00505686a51c":"1855/56",
"4af3b880-317a-0134-bda8-00505686a51c":"1856/57",
"4af6a690-317a-0134-5947-00505686a51c":"1857/58",
"4afa0510-317a-0134-cf84-00505686a51c":"1858/59",
"4afd6280-317a-0134-575a-00505686a51c":"1859/60",
"4b00bf60-317a-0134-32d0-00505686a51c":"1860/61",
"4b0419c0-317a-0134-7464-00505686a51c":"1861/62",
"4b073d20-317a-0134-af68-00505686a51c":"1862/63",
"4b0aa870-317a-0134-712b-00505686a51c":"1863/64",
"4b0e13f0-317a-0134-6578-00505686a51c":"1864/65",
"4b119360-317a-0134-9131-00505686a51c":"1865/66",
"4b154340-317a-0134-afd3-00505686a51c":"1866/67",
"4b18f080-317a-0134-fded-00505686a51c":"1867/68",
"4b336e60-317a-0134-1e9b-00505686a51c":"1868/69",
"4b36edd0-317a-0134-eedc-00505686a51c":"1869/70",
"4b3a14d0-317a-0134-011c-00505686a51c":"1870/71",
"4b3d0590-317a-0134-1631-00505686a51c":"1871-72",
"4b4009d0-317a-0134-949b-00505686a51c":"1872-73",
"4b437600-317a-0134-6db3-00505686a51c":"1873-74",
"4b47b740-317a-0134-ad0b-00505686a51c":"1874-75",
"4b4b2b90-317a-0134-6800-00505686a51c":"1875-76",
"4b4e8300-317a-0134-fb8c-00505686a51c":"1876-77",
"4b51d420-317a-0134-aa50-00505686a51c":"1877-78",
"4b5532f0-317a-0134-52ca-00505686a51c":"1878-79",
"4b58d200-317a-0134-d2aa-00505686a51c":"1879-80",
"4b5c40e0-317a-0134-e9c9-00505686a51c":"1880-81",
"4b5ff0e0-317a-0134-7e27-00505686a51c":"1881-82",
"4b63a460-317a-0134-d3bd-00505686a51c":"1882-83",
"4b66b460-317a-0134-8cb2-00505686a51c":"1883-84",
"4b69a410-317a-0134-a570-00505686a51c":"1884-85",
"4b6c95d0-317a-0134-f4e4-00505686a51c":"1885-86",
"4b6f8210-317a-0134-ff86-00505686a51c":"1886-87",
"4b728f10-317a-0134-8c07-00505686a51c":"1887-88",
"4b8e3f70-317a-0134-721a-00505686a51c":"1888-89",
"4b939190-317a-0134-d1d5-00505686a51c":"1889-90"}

urlcross = {"4adf9ec0-317a-0134-03ad-00505686a51c":"https://digitalcollections.nypl.org/items/7b3fbb00-5293-0134-b386-00505686a51c",
"4ae3cb40-317a-0134-489d-00505686a51c":"https://digitalcollections.nypl.org/items/023b8530-5295-0134-4c9c-00505686a51c",
"4ae76b60-317a-0134-b849-00505686a51c":"https://digitalcollections.nypl.org/items/3f790190-5298-0134-517e-00505686a51c",
"4aea8af0-317a-0134-2393-00505686a51c":"https://digitalcollections.nypl.org/items/d8b8ac20-5299-0134-e59e-00505686a51c",
"4aed8a80-317a-0134-28a4-00505686a51c":"https://digitalcollections.nypl.org/items/d73e7cd0-529b-0134-92d5-00505686a51c",
"4af0c6f0-317a-0134-e90c-00505686a51c":"https://digitalcollections.nypl.org/items/f283bb50-52ac-0134-0b4b-00505686a51c",
"4af3b880-317a-0134-bda8-00505686a51c":"https://digitalcollections.nypl.org/items/8f502510-52b4-0134-dacd-00505686a51c",
"4af6a690-317a-0134-5947-00505686a51c":"https://digitalcollections.nypl.org/items/6ec37860-52b6-0134-1782-00505686a51c",
"4afa0510-317a-0134-cf84-00505686a51c":"https://digitalcollections.nypl.org/items/83c244c0-52b8-0134-354a-00505686a51c",
"4afd6280-317a-0134-575a-00505686a51c":"https://digitalcollections.nypl.org/items/4f239540-52bb-0134-5039-00505686a51c",
"4b00bf60-317a-0134-32d0-00505686a51c":"https://digitalcollections.nypl.org/items/59b5b330-52be-0134-feef-00505686a51c",
"4b0419c0-317a-0134-7464-00505686a51c":"https://digitalcollections.nypl.org/items/23010ba0-52c0-0134-9308-00505686a51c",
"4b073d20-317a-0134-af68-00505686a51c":"https://digitalcollections.nypl.org/items/c2ab5490-5356-0134-0971-00505686a51c",
"4b0aa870-317a-0134-712b-00505686a51c":"https://digitalcollections.nypl.org/items/9574e160-535a-0134-fcf1-00505686a51c",
"4b0e13f0-317a-0134-6578-00505686a51c":"https://digitalcollections.nypl.org/items/7b8012d0-535c-0134-fe5e-00505686a51c",
"4b119360-317a-0134-9131-00505686a51c":"https://digitalcollections.nypl.org/items/258b7470-5361-0134-8a14-00505686a51c",
"4b154340-317a-0134-afd3-00505686a51c":"https://digitalcollections.nypl.org/items/f02a69a0-5363-0134-aba2-00505686a51c",
"4b18f080-317a-0134-fded-00505686a51c":"https://digitalcollections.nypl.org/items/46e14e50-536e-0134-b015-00505686a51c",
"4b336e60-317a-0134-1e9b-00505686a51c":"https://digitalcollections.nypl.org/items/c7aef1e0-5370-0134-55bb-00505686a51c",
"4b36edd0-317a-0134-eedc-00505686a51c":"https://digitalcollections.nypl.org/items/050245b0-5374-0134-ac00-00505686a51c",
"4b3a14d0-317a-0134-011c-00505686a51c":"https://digitalcollections.nypl.org/items/37dd46b0-58c9-0134-07e4-00505686a51c"}



add_gf = {}
total_entries = 0

for d in dirs:
    page_uuid_list = [i for i in os.listdir(os.path.join(directory, d, "final-entries")) if not i[0] == '.']
    for p_uuid in page_uuid_list:
        with open(os.path.join(directory, d, "final-entries", p_uuid)) as f:
            entries = f.readlines()
            for ent in entries:
                total_entries += 1
                ent_dict = json.loads(ent)
                for location in ent_dict['labeled_entry']['locations']:
                    or_token = ngram(location['value'], 2)
                    if or_token in add_gf:
                        add_gf[or_token].append({"original_entry": location['value'],
                                                 "directory_uuid": d,
                                                 "page_uuid": p_uuid,
                                                 "ent_uuid": ent_dict['entry_uuid'],
                                                 "current_corrected": location['value'],
                                                 "current_fingerprint": or_token,
                                                 "confidence_score": 0})
                    else:
                        add_gf[or_token] = [{"original_entry": location['value'],
                                             "directory_uuid": d,
                                             "page_uuid": p_uuid,
                                             "ent_uuid": ent_dict['entry_uuid'],
                                             "current_corrected": location['value'],
                                             "current_fingerprint": or_token,
                                             "confidence_score": 0}]
        f.close()


print("Finished building n-gram clusters (" + str(len(add_gf)) +  " clusters found)")
print("Starting analysis of cluster lengths")

# In[4]:

total_tokens_15more = 0
add_gram_current_new_val = {}
add_gram_cluster_counter = {"15more":0}
for i in range(1,15):
    add_gram_cluster_counter[i] = 0
add_gram_count_dict = {"15more":[]}
for i in range(1,15):
    add_gram_count_dict[i] = []

## Currently using a 2-gram...


for k in add_gf:
    num_clusters = len(add_gf[k])
    most_common_list = Counter([i["original_entry"] for i in add_gf[k]])
    most_common_token = most_common_list.most_common(1)[0][0]
    add_gram_current_new_val[k] = sp_abb_swap(' '.join([abbs_swap(corr_swap(simplify_token(i, 'lower'))) for i in most_common_token.split()]))
    if num_clusters < 15:
        add_gram_cluster_counter[num_clusters]+=1
        add_gram_count_dict[num_clusters].append(k)
    else:
        add_gram_cluster_counter["15more"]+=1
        add_gram_count_dict["15more"].append(k)
        total_tokens_15more+=num_clusters

# We now update our original dictionary of address information to reflect this
# ngram-fingerprint-derived most common value out of the clusters,
# regardless of how big the cluster is. We do this by working through all of the keys (=fingerprints)
# in the add_gram_current_new_val dictionary, then iterate through every fingerprint key
# in the original add_gf dictionary. At every key in add_gf (= common fingerprint), we iterate through the list of
# records that all have that same fingerprint and at that record's "current_corrected" value
# we set the current value associated with that fingerprint in the add_gram... dictionary.

for new_val_fp in add_gram_current_new_val:
    for record in add_gf[new_val_fp]:
        record["current_corrected"] = location_normalizer(add_gram_current_new_val[new_val_fp])

# And update the score based on the number of clusters. We iterate through all of the cluster sizes (= key)
# in add_gram_count_dict, then through the list of fingerprints associated with each cluster size.
# For every fingerprint, we then look up the associated list of records using the common fingerprint
# in add_gf, and then modify add_gf by adding the cluster size to the original score of 0

for score in add_gram_count_dict:
    for scored_fp in add_gram_count_dict[score]:
        for record in add_gf[scored_fp]:
            if record["current_corrected"] not in token_ignore_list:
                if not re.search(r'^\d+$|^\d+\sth$|between$|tith|sist|sith|tist|\%|\$|\scorner$', record["current_corrected"]):
                    if score != "15more":
                        record["confidence_score"]+=score
                    else:
                        record["confidence_score"]+=15
                else:
                    record["confidence_score"] = 0
            else:
                record["confidence_score"] = 0


print(add_gram_cluster_counter)
print("Starting Lev analysis of most common values")


# In[18]:


add_num_matched = 0
add_num_no_matched = 0
add_match_fail = []
lev_pull_lookup_by_fingerprint = {}
add_gram_low_count_list = []


for j in range(1,10):
    add_gram_low_count_list+=(add_gram_count_dict[j])

med_freq_grams = []
for j in range(10,15):
    med_freq_grams+=(add_gram_count_dict[j])


for add_l_freq_finger in add_gram_low_count_list:
    matched = False
    high_score_num = 0.92

    ## We first try to match against the high-freq tokens as they are deemed more likely to be correct

    for candidate in add_gram_count_dict["15more"]:
        ratio = lev.ratio(add_gram_current_new_val[candidate], add_gram_current_new_val[add_l_freq_finger])
        if ratio > high_score_num:
            matched = True
            high_score_num = ratio
            latest_match = [candidate, add_l_freq_finger]

    ## If we still don't have any good matches we try the next lower-freq tokens:

    if not matched:
        for candidate in med_freq_grams:
            ratio = lev.ratio(add_gram_current_new_val[candidate], add_gram_current_new_val[add_l_freq_finger])
            if ratio > high_score_num:
                matched = True
                high_score_num = ratio
                latest_match = [candidate, add_l_freq_finger]

    if matched:
        if latest_match[0] in lev_pull_lookup_by_fingerprint:
            lev_pull_lookup_by_fingerprint[latest_match[0]].append(latest_match[1])
        else:
            lev_pull_lookup_by_fingerprint[latest_match[0]] = [latest_match[1]]
        add_num_matched+=1
    else:
        add_num_no_matched+=1
        add_match_fail.append(add_l_freq_finger)


print("Number matched: " + str(add_num_matched))
print("Number not matched: " + str(add_num_no_matched))
print("Proceeding now to look for most common (largest) Levenshtein clusters")


lev_gram_count_dict = {"15more":[]}
lev_gram_cluster_counter = {"15more":0}

for j in range(1,15):
    lev_gram_cluster_counter[j] = 0
    lev_gram_count_dict[j] = []

for lev in lev_pull_lookup_by_fingerprint:
    num_clusters = len(lev_pull_lookup_by_fingerprint[lev])
    if num_clusters < 15:
        lev_gram_cluster_counter[num_clusters]+=1
        lev_gram_count_dict[num_clusters].append(lev)
    else:
        lev_gram_cluster_counter["15more"]+=1
        lev_gram_count_dict["15more"].append(lev)
    for record in add_gf[lev]:
        if add_gram_current_new_val[lev] not in token_ignore_list:
            if not re.search(r'^\d+$|^\d+\sth$|between$|tith|sist|sith|tist|\%|\$|\scorner$', add_gram_current_new_val[lev]):
                if record["confidence_score"] + num_clusters > 15:
                    record["confidence_score"] = 15
                else:
                    record["confidence_score"] += num_clusters
            else:
                record["confidence_score"] = 0
        else:
            record["confidence_score"] = 0

        record["current_corrected"] = location_normalizer(add_gram_current_new_val[lev])




## Write-outs

score_summary = {}
for i in range(0,16):
    score_summary[i] = 0


# We need to shuffle the dictionary of corrected entries in order
# to write it out sorted into directories
# Note that the dictionaries written to the text files are
# organized by entry_uuid in order to more quickly enable
# data merging down the line, but that entry_uuids will not be unique:
# some entries had multiple addresses so that when a record is appended
# here it maybe be the second or more case of that UUID.

writable_all_entries = {}

for clus in add_gf:
    for rec in add_gf[clus]:
        if rec["directory_uuid"] in writable_all_entries:
            writable_all_entries[rec["directory_uuid"]].append({rec["ent_uuid"]:rec})
        else:
            writable_all_entries[rec["directory_uuid"]] = [{rec["ent_uuid"]:rec}]
        score_summary[rec["confidence_score"]] += 1


for d_uuid in writable_all_entries:
    with open('<PATH-TO-OUTPUT-DIRECTORY>' + d_uuid + '-output-corrected-entries.ndjson', 'w') as outfile:
        for rec in writable_all_entries[d_uuid]:
            outfile.write(json.dumps(rec))
            outfile.write('\n')
    outfile.close()


with open('address_final_transform_messages.txt', 'a') as outfile:
    outfile.write("Run on " + datetime.now().strftime('%Y-%m-%d-%H:%M:%S') + "\n")
    outfile.write("Total number of address entries found: " + str(total_entries) + "\n")
    outfile.write("Total number of address entries in clusters of 15 more: " + str(total_tokens_15more) + "\n")
    for j in range(1,15):
        outfile.write("Total number of entries in clusters of " + str(j) + ": " + str(add_gram_cluster_counter[j] * j) + "\n")
    outfile.write("Cluster sizes: \n")
    for clus_size, clus_num in add_gram_cluster_counter.items():
        outfile.write("Clusters of size " + str(clus_size) + ": " + str(clus_num) + "\n")
    outfile.write("Following an attempt to match low frequency (9 entries clustered or less) \nmost common values against high-frequency tokens, \nthe following matches were made: \n")
    outfile.write("Number matched: " + str(add_num_matched) + "\n")
    outfile.write("Number not matched: " + str(add_num_no_matched) + "\n")
    outfile.write("Cluster sizes after a Levenshtein match: \n")
    for lev_clus_size, lev_clus_num in lev_gram_cluster_counter.items():
        outfile.write("Clusters of size " + str(lev_clus_size) + ": " + str(lev_clus_num) + "\n")
    for score_sum in score_summary:
        outfile.write("Number entries of score " + str(score_sum) + ": " + str(score_summary[score_sum]) + "\n")
    outfile.write('---------\n')
    outfile.close()

print("Completed run")



