import teradatasql as tdapi
import pandas as pd
import os, sys
from datetime import datetime as dtt


### define connection variables
login = os.getenv('USER')
pw = os.getenv('PW')
host = 'tdprod1.ccf.org'
conn = tdapi.connect(host=host, user=login, password=pw, tmode='ANSI', sessions=16, 
                logmech='LDAP')

### start 
now = dtt.now().strftime('%H:%M:%S')
print(f'Start pull sql NICM at {now}')
sys.stdout.flush()

# iteratively update query for new dates
sql_query = 'SELECT CAST(pat.K_PAT_KEY AS BIGINT), CAST(e.K_PAT_ENC_KEY AS BIGINT), \
 	            e.ENC_DT, dx.ICD9_CD_LIST, dx.ICD10_CD_LIST, dx.DX_NM \
             FROM IHAA_EDV.BK_PAT_ENC AS e \
             JOIN IHAA_EDV.S_PAT_ENC_DX AS se ON se.K_PAT_ENC_KEY = e.K_PAT_ENC_KEY \
             JOIN IHAA_EDV.BK_DX AS dx ON dx.K_DX_KEY = se.K_DX_KEY \
             JOIN IHAA_EDV.BK_PATIENT AS pat ON pat.K_PAT_KEY = se.K_PAT_KEY \
             WHERE pat.K_PAT_KEY IN (SELECT K_PAT_KEY FROM DL_AIIIH.Manual_Disease_Labels_for_CMR)'
# pull data using pandas built in sql connectors
df = pd.read_sql_query(sql_query, conn)
#save data without index
df.to_csv(f'/data/aiiih/projects/ts_nicm/data/dx_nicm.csv', index=False)
now = dtt.now().strftime('%H:%M:%S')
print(f'Finished DX at {now}')
sys.stdout.flush()


sql_query = 'SELECT CAST(pat.K_PAT_KEY AS BIGINT), \
    ord.ORD_RSLT_COMP_ID, ord.ORD_RSLT_COMP_NM, ord.ORD_RSLT_LOINC_CD, \
    ord.ORD_RSLT_DTTM, ord.ORD_RSLT_VALUE \
    FROM IHAA_EDV.BK_PATIENT AS pat \
    JOIN IHAA_EDV.S_PAT_ENC_APPT AS senc ON senc.K_PAT_KEY = pat.K_PAT_KEY \
    JOIN IHAA_EDV.S_ORDER_PARENT_INFO AS sord ON sord.K_PAT_ENC_KEY = senc.K_PAT_ENC_KEY \
    JOIN IHAA_EDV.BK_ORD_RESULTS AS ord ON ord.K_ORD_KEY = sord.K_ORD_KEY \
    WHERE pat.K_PAT_KEY IN (SELECT K_PAT_KEY FROM DL_AIIIH.Manual_Disease_Labels_for_CMR)'

# pull data using pandas built in sql connectors
df = pd.read_sql_query(sql_query, conn)
#save data without index
df.to_csv(f'/data/aiiih/projects/ts_nicm/data/labs_nicm.csv', index=False)
now = dtt.now().strftime('%H:%M:%S')
print(f'Finished LABS at {now}')
sys.stdout.flush()

sql_query = 'SELECT CAST(pat.K_PAT_KEY AS BIGINT), \
    CAST(enc.K_PAT_ENC_KEY AS BIGINT), \
    enc.ENC_DT, bkord.ORD_PROC_DTTM, bkord.ORD_PROC_DESC \
    FROM IHAA_EDV.BK_PATIENT AS pat \
    JOIN IHAA_EDV.S_PAT_ENC_APPT AS senc ON senc.K_PAT_KEY = pat.K_PAT_KEY \
    JOIN IHAA_EDV.S_ORDER_PARENT_INFO AS sord ON sord.K_PAT_ENC_KEY = senc.K_PAT_ENC_KEY \
    JOIN IHAA_EDV.BK_PAT_ENC AS enc ON enc.K_PAT_ENC_KEY = senc.K_PAT_ENC_KEY \
    JOIN IHAA_EDV.BK_ORD_PROC AS bkord ON bkord.K_ORD_KEY = sord.K_ORD_KEY \
    WHERE pat.K_PAT_KEY IN (SELECT K_PAT_KEY FROM DL_AIIIH.Manual_Disease_LABELS_for_CMR)'
# pull data using pandas built in sql connectors
df = pd.read_sql_query(sql_query, conn)
#save data without index
df.to_csv(f'/data/aiiih/projects/ts_nicm/data/proc_nicm.csv', index=False)
now = dtt.now().strftime('%H:%M:%S')
print(f'Finished nicm data pull at {now}')
sys.stdout.flush()

conn.close()