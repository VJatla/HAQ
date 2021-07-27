import pandas as pd
import pdb
import numpy as np
import sys

if( len(sys.argv) != 2):
    print("USAGE:\n\t python3 analyse_txt_file.py <txt_file>\n")
    sys.exit(1)


df_kids = pd.DataFrame()
df_sessions = pd.DataFrame()
df_groups = pd.DataFrame()
act = 0
no_act = 0
f = open(sys.argv[1], "r")
for x in f:

# persons
    kid         = x.split("_")[8]
    row_kids    = pd.DataFrame({"kids" : kid,} , index=[0])
    df_kids     = df_kids.append(row_kids)
# session
    session_split = x.split("_")[2].split("-")
    session =  session_split[0]+"-"+session_split[1] +"-"+session_split[2] +"-"+session_split[3]
    row_session = pd.DataFrame({"session" : session,} , index=[0])
    df_sessions = df_sessions.append(row_session)
#group
    group = session.split("-")[1]+"-"+session.split("-")[3]
    row_session = pd.DataFrame({"group" : group,} , index=[0])
    df_groups = df_groups.append(row_session)
#typing-notyping instances
    row_act = x.split("/")[0]
    if row_act == "typing":
        act += 1
    if row_act == "notyping":
        no_act += 1
    if row_act == "writing":
        act += 1
    if row_act == "nowriting":
        no_act += 1
print("Num of Activity Samples:" , act)
print("Num of No-Activity Samples:" , no_act)
num_kids = len(np.unique(df_kids))
print("Num of Persons:" , num_kids)
print(np.unique(df_kids))
num_sessions = len(np.unique(df_sessions))
print("Num of sessions:" , num_sessions)
print(np.unique(df_sessions))
num_groups = len(np.unique(df_groups))
print("Num of groups:" , num_groups)
print(np.unique(df_groups))
