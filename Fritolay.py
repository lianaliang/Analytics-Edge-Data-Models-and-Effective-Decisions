# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 00:00:48 2019

@author: h'p
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 23:13:18 2019

@author: h'p
"""




def optimize(input_margins,input_sales,input_sales_products,input_distribution_cost,
             input_prices,input_inventory,input_chain,input_manufacture, 
             q, qm,cc,st,h,gr,outputFile):
    from gurobipy import Model, GRB
    import pandas as pd
    import numpy as np
    import statistics as sts
    from scipy import stats 
    
    mod = Model()

    margins=pd.read_csv(input_margins)
    sales=pd.read_csv(input_sales)
    sales_products=pd.read_csv(input_sales_products)
    
    # Select BDCs
    southwest_sale=sales[sales.region=="SOUTHWEST"]
    southwest_sale["sales"]=southwest_sale["sales"]-southwest_sale["returns"]
    product_sales=pd.DataFrame(southwest_sale.groupby(["BDC_description"])["sales"].sum())
    total_sales=product_sales["sales"].sum()
    product_sales["percent_sales"]=(product_sales["sales"])/total_sales
    product_sales=product_sales.reset_index()
    sw_growth=pd.DataFrame(southwest_sale.groupby(["BDC_description","sales_period"])["sales"].sum()).reset_index()
    product_list=pd.DataFrame(sw_growth["BDC_description"].value_counts()).reset_index()["index"]
    dic={}
    for i in product_list:
        sales=pd.DataFrame(sw_growth[sw_growth.BDC_description==i]["sales"])
        growth_rt=np.mean((np.exp(np.diff(np.log(sales["sales"]))) - 1)[1:])
        dic[i]=growth_rt

    growth=pd.Series(dic)
    growth=pd.DataFrame(growth).reset_index()
    growth=growth.rename(columns={"index":"BDC_description",0:"growthrate"})
    growth=growth.fillna(0)

    BCG_matrix=product_sales.merge(growth,on="BDC_description")
    BCG_matrix=BCG_matrix.sort_values(by=['growthrate'],ascending=False).iloc[10:]

    #Add BDC and Margins
    sales=pd.read_csv(input_sales)
    margins=pd.read_csv(input_margins)
    margins=margins[["BDC","estimated_margins"]]
    sales=sales[["BDC","BDC_description"]]
    sales=sales.drop_duplicates()

    BCG_matrix=pd.merge(BCG_matrix,sales,how='left',on="BDC_description")
    BCG_matrix=BCG_matrix.merge(margins,how='left',on="BDC")
    BCG_matrix=BCG_matrix.drop_duplicates()

    
    md_gr=sts.median(BCG_matrix["growthrate"])
    md_ms=sts.median(BCG_matrix["percent_sales"])
    

    BCG_selected=BCG_matrix[(BCG_matrix.percent_sales>=md_ms) | (BCG_matrix.growthrate>=md_gr)]


    BCG_selected["question_mark"]=np.where(BCG_selected["percent_sales"]<=md_ms,np.where(BCG_selected["growthrate"]>=md_gr,1,0),0)
    BCG_selected["cash_cow"]=np.where(BCG_selected["percent_sales"]>md_ms,np.where(BCG_selected["growthrate"]<md_gr,1,0),0)
    BCG_selected["star"]=np.where(BCG_selected["percent_sales"]>md_ms,np.where(BCG_selected["growthrate"]>md_gr,1,0),0)

    bcg=BCG_selected.fillna(np.mean(BCG_selected["estimated_margins"]))
    bcg=bcg[(bcg['star']!=0) |(bcg['question_mark']!=0)| (bcg['cash_cow']!=0)]
    
    CC=bcg['BDC'][bcg['cash_cow']==1]
    QM=bcg['BDC'][bcg['question_mark']==1]
    Star=bcg['BDC'][bcg['star']==1]

    # Healthy Products
    manufacture=pd.read_csv(input_manufacture)
    manufacture=manufacture[["brand","BDC"]]
    healthy=pd.merge(bcg,manufacture,how='left',on="BDC")
    healthy["brand"]=healthy["brand"].fillna("Lay's")
    hb=["Smartfood","Matador","Sun Chips","Nut Harvest","Miss Vickie's","Bare","Spitz","Off The Eaten Path"]
    healthy=healthy[healthy["brand"].isin(hb)]
    healthy=healthy[["BDC","sales"]]
    healthy=healthy.groupby(['BDC']).aggregate({'sales':'first'}).sort_values('sales',ascending=False).head(h)
    H=healthy.index

    transportation_cost=pd.read_excel(input_distribution_cost, sheet_name = 'transportation_cost')
    shipment=pd.read_excel(input_distribution_cost, sheet_name = 'shipment_quantities')
    SW_warehouseID_DC=shipment['destination_LMSID'][(shipment['destination_region']=='SOUTHWEST')&
                                                    (shipment['destination_type']=='DC')].reset_index()['destination_LMSID'].unique()
    ctable=transportation_cost[['source_LMSID','destination_LMSID','transportation_cost_per_standard_case']]\
    [transportation_cost['destination_LMSID'].isin(SW_warehouseID_DC)].sort_values('destination_LMSID')
    costtable=transportation_cost[['source_LMSID','destination_LMSID','transportation_cost_per_standard_case']]\
    [transportation_cost['destination_LMSID'].isin(SW_warehouseID_DC)].sort_values('destination_LMSID')\
    .set_index(['source_LMSID','destination_LMSID'])
    I=ctable['source_LMSID'].sort_values().unique()
    J=ctable['destination_LMSID'].sort_values().unique()
    SW_trans_cost=pd.DataFrame(columns=J,index=I)
    I=ctable['source_LMSID'].sort_values().unique()
    J=ctable['destination_LMSID'].sort_values().unique()
    SW_trans_cost=pd.DataFrame(columns=J,index=I)
    SW_trans_cost.head()
    for i in I:
        for j in J:
            try:
                SW_trans_cost.loc[i,j]=float(costtable.loc[i,j])
            except:
                continue
    cost1=SW_trans_cost

    source_cost=pd.read_excel(input_distribution_cost, sheet_name ='source_warehouse_cost')
    destination_cost=pd.read_excel(input_distribution_cost, sheet_name ='destination_warehouse_cost')
    all_source=SW_trans_cost.index
    all_destination=SW_trans_cost.columns
    index_names=all_source.sort_values().unique()
    column_names=all_destination.sort_values().unique()
    sc_plus_dc=pd.DataFrame(columns=column_names,index=index_names)
    for i in index_names:
        for j in column_names:
            try:
                sc_plus_dc.loc[i,j]=float(source_cost.loc[i,'source_cost_per_case']
                                          +destination_cost.loc[j,'destination_cost_per_case'])
            except:
                continue
    cost2=sc_plus_dc 

    case_and_standard=shipment[['BDC','cases_ordered','standard_cases_ordered']]
    case_and_standard['ratio']=round(case_and_standard.loc[:,'standard_cases_ordered']\
                                     /case_and_standard.loc[:,'cases_ordered'],2)
    case_and_standard=case_and_standard.dropna()
    ratio=case_and_standard.groupby(['BDC'])['ratio'].agg(lambda x:  stats.mode(x)[0] ).reset_index()

    # ratio of Standard case to unit
    prices=pd.read_csv(input_prices)
    unit_to_case=prices[['BDC','count_per_carton','cube']]
    unit_to_case['units_per_cube']=unit_to_case['count_per_carton']/unit_to_case['cube']
    unit_to_case['units_per_starardcase']=unit_to_case['units_per_cube']*1.51
    unit_to_case=unit_to_case[['BDC','units_per_starardcase']]
    ratio=ratio.reset_index()
    ratio=ratio.groupby(['BDC'])['ratio'].sum()
    ratio=pd.DataFrame(ratio)
    bcg1=bcg[['BDC','estimated_margins']]
    bizcode=sales_products[['BDC','business_unit_code']]
    bcg4=bcg1.merge(unit_to_case, on=['BDC'],how='left')
    bcg4=bcg4[['BDC','units_per_starardcase']]
    bcg4=bcg4.groupby(['BDC'])['units_per_starardcase'].mean()
    bcg4=pd.DataFrame(bcg4)
    bcg4=bcg4.merge(bizcode,on=['BDC'],how='left')
    ratio_unit_to_case=bizcode.merge(unit_to_case, on=['BDC'],how='left')
    ratio_unit_to_case=ratio_unit_to_case[['business_unit_code','units_per_starardcase']]
    ratio_unit_to_case=ratio_unit_to_case.groupby(['business_unit_code'])['units_per_starardcase'].mean()
    ratio_unit_to_case=pd.DataFrame(ratio_unit_to_case)
    ratio_unit_to_case=ratio_unit_to_case.reset_index()

    ratio_unit_to_case['units_per_starardcase'] = np.where(np.isnan(ratio_unit_to_case['units_per_starardcase']), np.mean(ratio_unit_to_case['units_per_starardcase']), ratio_unit_to_case['units_per_starardcase'])
    bcg5=bcg4.merge(ratio_unit_to_case,on=['business_unit_code'],how='left')
    bcg5['units_per_starardcase'] = np.where(np.isnan(bcg5['units_per_starardcase_x']), bcg5['units_per_starardcase_y'], bcg5['units_per_starardcase_x'])
    bcg5=bcg5[['BDC','units_per_starardcase']]

    ratio1=bizcode.merge(ratio, on=['BDC'],how='left')
    ratio_total=ratio1[['business_unit_code','ratio']]
    ratio_total=ratio_total.groupby(['business_unit_code'])['ratio'].mean()
    ratio_total=pd.DataFrame(ratio_total)
    ratio_total=ratio_total.reset_index()

    bcg2=bcg1.merge(bizcode, on=['BDC'],how='left')
    bcg3=bcg2.merge(ratio_total, on=['business_unit_code'],how='left')
    bcg3=bcg3.groupby(['BDC']).aggregate({'estimated_margins':'first','business_unit_code':'first','ratio':'first'})
    bcg3=bcg3.reset_index()
    bcg3['ratio'] = np.where(np.isnan(bcg3['ratio']), np.mean(bcg3['ratio']), bcg3['ratio'])
    bcg3=bcg3.merge(bcg5,on=['BDC'],how='left')
    bcg3=bcg3[['BDC','estimated_margins','ratio','units_per_starardcase']]
    bcg3=bcg3.groupby(['BDC']).aggregate({'estimated_margins':'first','ratio':'first','units_per_starardcase':'first'})
    bcg3['standardcase_per_unit']=1/bcg3['units_per_starardcase']
    bcg3=bcg3[['estimated_margins','ratio','standardcase_per_unit']]

    # turn NA cost to 0
    cost1=cost1.replace(np.nan,0)
    cost2=cost2.replace(np.nan,0)

    # some data preparation for Optimization
    

    ## Data table for constaints
    data=pd.read_csv(input_sales)
    chain_info=pd.read_csv(input_chain)
    product=pd.read_csv(input_sales_products)
    p= pd.merge(data,product,on='BDC', how='left')
    p=p[p["region"]=="SOUTHWEST"]
    p["new_cate"]=np.where(p["category_description"]=="OTHER",p["business_unit_description"],p["category_description"])
    p["new_cate"]=np.where(p["category_description"]=="SINGLE SERVE",p["business_unit_description"],p["new_cate"])
    BCG_list=bcg["BDC"]
    p=p[(p['BDC'].isin(BCG_list))]
    p1= pd.merge(p,chain_info,on='chain_id', how='left')
    p1["revenue"]=p1["sales"]-p1["returns"]
    p1["revenue"]=np.where(p1["revenue"]<0,0,p1["revenue"])
    price1=pd.DataFrame(prices.groupby("BDC")["price_on_bag"].mean())

    p2= p1.merge(price1,on='BDC', how='left')
    p2["price_on_bag"]=np.where(p2["price_on_bag"].isna(),p2["price_on_bag"].mean(),p2["price_on_bag"])
    p2["count"]=p2["sales"]/p2["price_on_bag"]
    psmall=p2.loc[p1["sales_channel"].isin(['C-STORE','SMALL GROCERY', 'INDEPENDENT BUSINESS',
           'DRUG STORE', 'DOLLAR STORE',
           'OTHER NON-UDS', 'FOOD SERVICE', 'VEND', 'ALL OTHER'])]
    popular_BDC=psmall.groupby("BDC")["revenue"].sum().sort_values(ascending=False)
    popular_BDC=pd.DataFrame(popular_BDC)
    popular_BDC['cum_sum'] = popular_BDC.revenue.cumsum()
    popular_BDC['cum_sum_percent']=popular_BDC["cum_sum"]/(popular_BDC["revenue"].sum())
    popular_BDC["product_count"]=range(len(popular_BDC["revenue"]))
    popular_BDC["product_count"]=popular_BDC["product_count"]+1
    popular_BDC["product_percentage"]=popular_BDC["product_count"]/len(popular_BDC["revenue"])
    popular_BDC=popular_BDC.reset_index()
    popular_BDC_sub=popular_BDC[['BDC','revenue']]
    popular_BDC_quantities=popular_BDC_sub.merge(price1,on='BDC', how='left')
    popular_BDC_quantities["price_on_bag"]=np.where(popular_BDC_quantities["price_on_bag"].isna(),
                                                    popular_BDC_quantities["price_on_bag"].mean(),popular_BDC_quantities["price_on_bag"])
    popular_BDC_quantities["count"]=popular_BDC_quantities["revenue"]/popular_BDC_quantities["price_on_bag"]
    popular_BDC_quantities=popular_BDC_quantities[['BDC','count']]
    popular_BDC_quantities=popular_BDC_quantities.set_index('BDC')

    proportion_small=pd.DataFrame(psmall.groupby("new_cate")["count"].sum())
    category_proportion=proportion_small.sort_values(by="count",ascending=False)
    category_proportion['proportion']=category_proportion['count']/sum(category_proportion['count'])

    ls=p2.groupby(["new_cate","BDC"])["count"].count()
    l=pd.DataFrame(ls)
    l=l.reset_index()
    BDC_newcate=l[['new_cate','BDC']]
    BDC_newcate=BDC_newcate[BDC_newcate['new_cate']!='DISCONTINUED']
    category_names=BDC_newcate.groupby(['new_cate']).aggregate({'BDC':'first'}).index

    # Gurobi Optmzation
    I=cost1.index
    J=cost1.columns
    N=BDC_newcate['BDC']
    
    ## Whether the plants(I) produce the BDC
    inventory=pd.read_excel(input_inventory,skiprows=2)
    Cons_produce=inventory[['LMSID','BDC','produced?']][inventory['type']=='PLANT']
    Cons_produce=Cons_produce[Cons_produce['produced?']=='NP']
    Cons_produce=Cons_produce[['LMSID','BDC']]
    Cons_produce=Cons_produce.set_index('LMSID')
    select_plants=[t for t in Cons_produce.reset_index().groupby(['LMSID']).aggregate({'BDC':'first'}).index if t in I]
    
    x=mod.addVars(N, lb=0,vtype = GRB.BINARY)
    mod.addConstr(sum(x[n] for n in N)<=250)
    mod.addConstr(sum(x[n] for n in N)>=200)
    a=mod.addVars(N,I,J, lb=0,vtype = GRB.INTEGER,name='a')

    bcg=bcg[(bcg['BDC'].isin(N))]
    CC=bcg['BDC'][bcg['cash_cow']==1]
    QM=bcg['BDC'][bcg['question_mark']==1]
    Star=bcg['BDC'][bcg['star']==1]

    for n in N :
        for i in I :
            for j in J:
                mod.addConstr(a[n,i,j] <=5000000000 )
                mod.addConstr(a[n,i,j] >=0 )

    for n in N:
        for j in range(len(J)-1):
            mod.addConstr(sum(a[n,i,J[j]] for i in I)==sum(a[n,i,J[j+1]] for i in I))

    for n in H:
        mod.addConstr(x[n]==1)

    for i in select_plants:
        for n in [t for t in Cons_produce.loc[i,'BDC'].values if t in N]:
            for j in J:
                mod.addConstr(a[n,i,j]==0)

    mod.addConstr(sum(x[n] for n in CC) >= (cc-0.05)*sum(x[n] for n in N))
    mod.addConstr(sum(x[n] for n in Star) >= (st-0.05)*sum(x[n] for n in N))
    mod.addConstr(sum(x[n] for n in QM) >= (qm-0.05)*sum(x[n] for n in N))

    # Constaints- core BDC
    PBDC=popular_BDC[popular_BDC['cum_sum_percent']<=q].set_index('BDC').index
    for n in PBDC:
        mod.addConstr(x[n]==1)
        mod.addConstr(sum(a[n,i,j] for i in I for j in J)>= popular_BDC_quantities.loc[n,'count'])
        mod.addConstr(sum(a[n,i,j] for i in I for j in J)<= (1+gr)*popular_BDC_quantities.loc[n,'count'])

    # Constraints- upper/lower
    category=BDC_newcate
    sw=southwest_sale[["sales_period","BDC","sales"]]
    sw_cat=pd.merge(sw,category,how='left',on="BDC")
    sw_cat=pd.DataFrame(sw_cat.groupby(["new_cate","sales_period"])["sales"].sum()).reset_index()
    sw_cat=pd.DataFrame(sw_cat.groupby(["new_cate","sales_period"])["sales"].sum()).reset_index()
    product_list=pd.DataFrame(sw_cat["new_cate"].value_counts()).reset_index()["index"]
    dic={}
    for i in product_list:
        growth_rt=np.mean(sw_cat[sw_cat.new_cate==i]["sales"].pct_change()[1:])
        dic[i]=growth_rt
    growth=pd.Series(dic)
    growth=growth.fillna(0)
    growth=pd.DataFrame(growth)
    growth.rename(columns={0:'gr'}, inplace=True)
    growth["gr"]=np.where(growth.gr>=0, growth["gr"], 0.1)
    dict_of_groups = {k: v for k, v in BDC_newcate.groupby('new_cate')}

    for name in category_names:
        mod.addConstr(sum(a[n,i,j] for n in dict_of_groups[name]['BDC'] for i in I for j in J)>= category_proportion.loc[name,'count'])
        mod.addConstr(sum(a[n,i,j] for n in dict_of_groups[name]['BDC'] for i in I for j in J)<= category_proportion.loc[name,'count']*(1+growth.loc[name,"gr"]))
        
    # Objective
    mod.setObjective(sum((x[n]*bcg3.loc[n,'estimated_margins']*a[n,i,j]
                          -cost1.loc[i,j]*x[n]*(a[n,i,j])*bcg3.loc[n,'standardcase_per_unit']
                          -cost2.loc[i,j]*x[n]*(a[n,i,j])*bcg3.loc[n,'ratio']*bcg3.loc[n,'standardcase_per_unit']) 
                         for n in N for i in I for j in J),sense=GRB.MAXIMIZE)

    mod.optimize()

    BDC_chosen=[]
    for n in N:
        if x[n].x==1:
            BDC_chosen.append(n)
    BDC_chosen=pd.Series(BDC_chosen)
    output=pd.DataFrame(BDC_chosen,columns=['BDC'])
    writer=pd.ExcelWriter(outputFile)
    pd.DataFrame([mod.objVal],columns=["Objective Value"]).to_excel(writer,sheet_name='Summary',index=False)
    pd.DataFrame(output).to_excel(writer,sheet_name='Solution',index=False)
    writer.save()

    
if __name__=='__main__':
    import sys, os
#     if len(sys.argv)!=14:
#         print('Correct syntax: python lab2_code.py inputFile outputFile')
#     else:
    input_margins=sys.argv[1]
    input_sales=sys.argv[2]
    input_sales_products=sys.argv[3]
    input_distribution_cost=sys.argv[4]
    input_prices=sys.argv[5]
    input_inventory=sys.argv[6]
    input_chain=sys.argv[7]
    input_manufacture=sys.argv[8]
    q=float(sys.argv[9])
    qm=float(sys.argv[10])
    cc=float(sys.argv[11])
    st=float(sys.argv[12])
    h=int(sys.argv[13])
    gr=float(sys.argv[14])
    outputFile=sys.argv[15]
#     if os.path.exists(input_margins,input_sales,input_sales_products,
#                       input_distribution_cost,input_prices,input_inventory,input_chain,input_manufacture,q,qm,cc,st,h,gr,outputFile):
    optimize(input_margins,input_sales,input_sales_products,
             input_distribution_cost,input_prices,input_inventory,input_chain,input_manufacture,q,qm,cc,st,h,gr,outputFile)
    print(f'Successfully optimized. Results in "{outputFile}"')
#     else:
#         print(f'File "{inputFile}" not found!')