from numpy_financial import npv, irr
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

def NPV(cflow, discount, investment):
	"""Compute NPV - Net Present Value.
	Parameters:
	cflow: array_like -- Contains yearly cash flows, excluding initial investment
	investment: (positive) float -- Initial investment
	discount: float in [0,1] -- Investment Discount Rate - Tasso inflazione
	Returns float NPV.
	"""
	cflow = list(cflow) #cast as list
	cflow = [-investment] + cflow #Add (negative) initial investment to cash flows vector
	result = npv(discount, cflow)
	return result

def IRR(cflow, investment):
	"""Compute IRR - Internal Rate of Return
	Parameters:
	cflow: array_like -- Contains yearly cash flows, excluding initial investment
	investment: (negative) float -- Initial investment
	Returns float IRR.
	"""
	cflow = list(cflow) #cast as list
	cflow = [-investment] + cflow #Add (negative) initial investment to cash flows vector
	result = irr(cflow)
	return result

def initial_investment(pv_size, battery_size, n_batteries = 1, capex_pv = 900, capex_batt = 509):
	"""Compute initial investment"""
	return pv_size*capex_pv + battery_size*capex_batt*n_batteries

def cash_flows(fed_energy, shared_energy, pv_size, beta, PR3 = 42,  CUAF = 8.56 , TP = 110, inf_rate = 0.02, OM = 30, n_years = 20):
	"""Compute cash flows over 20 years"""
	prezzi_ritiro = [PR3]*n_years
	for i in range(1, n_years):
		prezzi_ritiro[i] = (1 + inf_rate)*prezzi_ritiro[i-1] #prezzi minimi garantiti inflazionati
 	#Assumo CUAF costante nei prox 20 anni
	gse_refund = np.array([shared_energy * CUAF] * n_years)
	#TP is guaranteed constant
	premium = np.array([shared_energy * TP] * n_years)
	contributions = gse_refund + premium #Amount to be splitted
	contrib_prod = beta * contributions #Goes to producer
	contrib_other = (1 - beta) * contributions #Goes to non producer
	#Sales
	energy_sales = np.array([fed_energy * prezzo for prezzo in prezzi_ritiro])
	#Cash cash_flows (producer)
	cflows = np.array([rce + contr - OM*pv_size for rce, contr in zip(energy_sales, contrib_prod)])

	return cflows, energy_sales, contributions, contrib_prod   

def PCR(yearly_expense, contrib_prod, energy_sales, initial_investment, pv_size, OM = 30, n_years = 20):
	"""Percentage cost reduction"""
	#Average yearly returns
	community_returns =(contrib_prod.sum() + energy_sales.sum())/n_years
	yearly_amortations = initial_investment/n_years
	yearly_om = OM*pv_size
	pcr = 100*(community_returns - yearly_amortations - yearly_om)/yearly_expense
	return pcr

def PCR_community(yearly_expense, contrib_community, n_years = 20):
	"""Compute PCR for community"""
	pcr = contrib_community.sum()/n_years
	return pcr/yearly_expense*100

def PBT(initial_investment, cflow):
	"""Compute Payback Time for given cash flows and initial investment"""
	year = 0 
	amount = 0 #Total money
	while amount < initial_investment:
		amount += cflow[year]
		year += 1
		if year == len(cflow):
			return 'Infeasible' 
	return year
#%%

#UNIT TEST#
if __name__ == '__main__':
	pass