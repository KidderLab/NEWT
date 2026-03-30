#!/usr/bin/env python3
"""
Creates a directory hierarchy under OUTPUT_DIR reflecting ATC levels:
  • Level-1 main classes (A, B, C, …): filtered top-10 network + CSVs
  • Level-2 subgroups (e.g. A01, B02…): filtered top-10 network + CSVs
  • Level-3 sub-subgroups (e.g. L01A, A02B…): BOTH filtered top-10 AND unfiltered full networks + CSVs

Targets CSVs have no header as requested.
"""
import os
import glob
import pandas as pd
import networkx as nx

# — USER CONFIGURATION —
INPUT_DIR  = "./ct_network_exports_shRNA"
ATC_FILE   = "data/WHO_ATC_DDD_2024-07-31.csv"
OUTPUT_DIR = "./ct_network_exports_ATC_subnetworks_shRNA"


def load_atc_mapping():
    atc_all = pd.read_csv(ATC_FILE, dtype=str)
    # Level-1 classes
    main = (
        atc_all[atc_all['atc_code'].str.len()==1]
        [['atc_code','atc_name']]
        .rename(columns={'atc_code':'class','atc_name':'class_name'})
    )
    # Level-5 substances
    subs7 = (
        atc_all[atc_all['atc_code'].str.len()==7]
        [['atc_code','atc_name']]
        .rename(columns={'atc_code':'atc7_code','atc_name':'atc7_name'})
    )
    subs7['class'] = subs7['atc7_code'].str[0]
    subs7['cmpd_norm'] = subs7['atc7_name'].str.lower().str.strip()
    return atc_all, main, subs7


def load_all_interactions():
    dfs=[]
    for f in sorted(glob.glob(os.path.join(INPUT_DIR,'*.csv'))):
        df=pd.read_csv(f)
        req={'Query Pert ID','Compound Name','Symbol','Probability'}
        if not req.issubset(df.columns):
            raise ValueError(f"{f} missing columns: {req - set(df.columns)}")
        df=df[['Query Pert ID','Compound Name','Symbol','Probability']].dropna()
        df=df.rename(columns={
            'Query Pert ID':'compound_id',
            'Compound Name':'compound_name',
            'Symbol':'target'
        })
        df['cmpd_norm']=df['compound_name'].str.lower().str.strip()
        dfs.append(df)
    if not dfs:
        raise FileNotFoundError(f"No CSVs in {INPUT_DIR}")
    return pd.concat(dfs,ignore_index=True)


def safe_name(s): return s.replace(' ','_').replace('/','_').replace(',','')


def filter_top10(grp):
    # top-10 targets per compound
    df_c=grp.sort_values(['compound_id','Probability'],ascending=[True,False])
    df_c=df_c.groupby('compound_id',as_index=False).head(10)
    # top-10 compounds per target
    df_t=grp.sort_values(['target','Probability'],ascending=[True,False])
    df_t=df_t.groupby('target',as_index=False).head(10)
    # intersection
    return pd.merge(
        df_c[['compound_id','compound_name','target']],
        df_t[['compound_id','target']],
        on=['compound_id','target']
    )


def export_graph(G, path):
    nx.write_graphml(G,path)


def export_csv(detail_df, targets, base, outdir, suffix=''):
    # details CSV
    det=detail_df.copy()
    det=det.rename(columns={
        'compound_id':'compound',
        'compound_name':'drug_name'
    })
    det=det[['atc_code','atc_name','drug_name','compound','target']]
    det_file=os.path.join(outdir,f"{base}{suffix}_details.csv")
    det.to_csv(det_file,index=False)
    # targets CSV (no header)
    tgt_df=pd.DataFrame({'target':sorted(targets)})
    tgt_file=os.path.join(outdir,f"{base}{suffix}_targets.csv")
    tgt_df.to_csv(tgt_file,index=False,header=False)


def build_and_export():
    atc_all, main, subs7 = load_atc_mapping()
    df = load_all_interactions().merge(subs7, on='cmpd_norm', how='left').dropna(subset=['class'])
    os.makedirs(OUTPUT_DIR,exist_ok=True)

    # Level-1
    for cls, grp in df.groupby('class'):
        cls_name=main.loc[main['class']==cls,'class_name'].iat[0]
        safe1=safe_name(cls_name)
        dir1=os.path.join(OUTPUT_DIR,f"{cls}_{safe1}")
        os.makedirs(dir1,exist_ok=True)
        df1_i=filter_top10(grp)
        if df1_i.empty: continue
        # graph
        G1=nx.Graph()
        for cid,cname in df1_i[['compound_id','compound_name']].drop_duplicates().itertuples(index=False,name=None):
            G1.add_node(cid,type='compound',name=cname)
        for t in df1_i['target'].unique():
            G1.add_node(t,type='target')
        G1.add_edges_from(df1_i[['compound_id','target']].itertuples(index=False,name=None))
        export_graph(G1,os.path.join(dir1,f"{cls}_{safe1}.graphml"))
        # csv + targets
        df1_i['atc_code']=cls; df1_i['atc_name']=cls_name
        export_csv(df1_i,df1_i['target'].unique(),f"{cls}_{safe1}",dir1)

        # Level-2
        dir2_root=os.path.join(dir1,'subgroups'); os.makedirs(dir2_root,exist_ok=True)
        atc2=atc_all[atc_all['atc_code'].str.len()==3][['atc_code','atc_name']].rename(columns={'atc_code':'code2','atc_name':'name2'})
        for _,r2 in atc2[atc2['code2'].str.startswith(cls)].iterrows():
            c2,name2=r2['code2'],r2['name2']; safe2=safe_name(name2)
            grp2=df[df['atc7_code'].str.startswith(c2)]
            df2_i=filter_top10(grp2)
            if df2_i.empty: continue
            dir2=os.path.join(dir2_root,f"{c2}_{safe2}"); os.makedirs(dir2,exist_ok=True)
            G2=nx.Graph()
            for cid,cname in df2_i[['compound_id','compound_name']].drop_duplicates().itertuples(index=False,name=None):
                G2.add_node(cid,type='compound',name=cname)
            for t in df2_i['target'].unique(): G2.add_node(t,type='target')
            G2.add_edges_from(df2_i[['compound_id','target']].itertuples(index=False,name=None))
            export_graph(G2,os.path.join(dir2,f"{c2}_{safe2}.graphml"))
            df2_i['atc_code']=c2; df2_i['atc_name']=name2
            export_csv(df2_i,df2_i['target'].unique(),f"{c2}_{safe2}",dir2)

            # Level-3
            dir3_root=os.path.join(dir2,'subgroups'); os.makedirs(dir3_root,exist_ok=True)
            atc3=atc_all[atc_all['atc_code'].str.len()==4][['atc_code','atc_name']].rename(columns={'atc_code':'code3','atc_name':'name3'})
            for _,r3 in atc3[atc3['code3'].str.startswith(c2)].iterrows():
                c3,name3=r3['code3'],r3['name3']; safe3=safe_name(name3)
                grp3=df[df['atc7_code'].str.startswith(c3)]
                # filtered
                df3_f=filter_top10(grp3)
                # unfiltered
                df3_u=grp3[['compound_id','compound_name','target']].drop_duplicates()
                if df3_f.empty and df3_u.empty: continue
                dir3=os.path.join(dir3_root,f"{c3}_{safe3}"); os.makedirs(dir3,exist_ok=True)
                # write filtered
                if not df3_f.empty:
                    G3_f=nx.Graph()
                    for cid,cname in df3_f[['compound_id','compound_name']].drop_duplicates().itertuples(index=False,name=None):
                        G3_f.add_node(cid,type='compound',name=cname)
                    for t in df3_f['target'].unique(): G3_f.add_node(t,type='target')
                    G3_f.add_edges_from(df3_f[['compound_id','target']].itertuples(index=False,name=None))
                    export_graph(G3_f,os.path.join(dir3,f"{c3}_{safe3}.graphml"))
                    df3_f['atc_code']=c3; df3_f['atc_name']=name3
                    export_csv(df3_f,df3_f['target'].unique(),f"{c3}_{safe3}",dir3)
                # write unfiltered
                if not df3_u.empty:
                    G3_u=nx.Graph()
                    for cid,cname in df3_u[['compound_id','compound_name']].drop_duplicates().itertuples(index=False,name=None):
                        G3_u.add_node(cid,type='compound',name=cname)
                    for t in df3_u['target'].unique(): G3_u.add_node(t,type='target')
                    G3_u.add_edges_from(df3_u[['compound_id','target']].itertuples(index=False,name=None))
                    export_graph(G3_u,os.path.join(dir3,f"{c3}_{safe3}_full.graphml"))
                    df3_u['atc_code']=c3; df3_u['atc_name']=name3
                    export_csv(df3_u,df3_u['target'].unique(),f"{c3}_{safe3}_full",dir3)

if __name__=='__main__':
    build_and_export()
