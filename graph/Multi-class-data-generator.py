import pandas as pd

entry_pairs = pd.read_csv ('entry_pairs.csv')
pair_labels = pd.read_csv ('pair_labels.csv')
Liu_data_List = pd.read_csv ('Liu_data_List.csv')

Liu_data_List_df = pd.DataFrame(Liu_data_List)
entry_pairs['class']=pair_labels

Liu_data_List_df_index = Liu_data_List_df.index
drug_id_map = {}
for i in Liu_data_List_df_index:
    drug_id_map[Liu_data_List_df.iloc[i]['DrugBank ID']]=i

DDI_list = []
for i in entry_pairs.iloc:
    if i['entry1'] in drug_id_map.keys() and i['entry2'] in drug_id_map.keys():
        DDI_list.append([drug_id_map[i['entry1']], drug_id_map[i['entry2']], i['class']])

DDI_df = pd.DataFrame(DDI_list)
DDI_df.columns = ['entry1', 'entry2', 'class']
DDI_df.to_csv('DDI_df.csv')