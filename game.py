# ___________                                ________  ________  
# \__    ___/  ____  _____     _____         \_____  \ \_____  \ 
#   |    |   _/ __ \ \__  \   /     \         /  ____/   _(__  < 
#   |    |   \  ___/  / __ \_|  Y Y  \       /       \  /       \
#   |____|    \___  >(____  /|__|_|  / ______\_______ \/______  /
#                 \/      \/       \/ /_____/        \/       \/
import pandas as pd
import numpy as np
from itertools import product
from math import factorial
from evaluation_quicky import configuration_evaluation, yearly_monthly_energy #To compute value function & yearly consumption
from systems_specifications import storage_system_specifications #Read battery specs
from plotter import shares_pie_plot
import os
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
from numpy_financial import irr

#Plot configuration
# =============================================================================
# sns.set_theme() #use to reset955
# =============================================================================
sns.set_style("whitegrid")
sns.set_context("notebook", font_scale= 2.5)
plt.rc('figure',figsize=(32,24))
plt.rc('font', size=20)
plt.ioff()

class BenefitDistributionGame:
	"""Class to represent benefit distribution as a game"""

#-----Nested Player class - Each player represents a participant in the REC----
	class _Player:
		"""Constructor should not be invoked by user"""
		def __init__(self, player_number, player_name, profile_wd = None, profile_we = None,
					 spec_capex_pv = (1100, 800), spec_capex_battery = 500):
			"""Parameters:
				player_number: int. 
				player_name: str.
				profile_we: np.array.
				profile_wd: np.array.
				spec_capex_pv: tuple of size 2 -- Specific cost for small / large plant.
							   Threshold is set to 20 KW.
				spec_capex_battery: float. -- Specific cost.
			"""
			#Store data - pd.DataFrames in usual format.
			if profile_we is None or profile_wd is None:
				profile_we, profile_wd = self._simulate_profiles()
			self._profile_wd = profile_wd
			self._profile_we = profile_we
			#Parameters
			self.spec_capex_pv = spec_capex_pv
			self.spec_capex_battery = spec_capex_battery
			#Assign player number
			self.player_number = player_number
			#Player name
			self.player_name = player_name
			#Display dumb menu
			print("Creating Player " + str(player_number) + ": " + player_name)
			#Insert PV Power and Battery Size for this user.
			self._pv_size = input("Insert PV Power for this player: ")
			if self._pv_size == '': 
				self._pv_size = 0
			else:
				self._pv_size = float(self._pv_size)
			self._battery_size = input("Insert battery size for this player: ")
			if self._battery_size == '':
				self._battery_size = 0
			else:
				self._battery_size = float(self._battery_size)
			#Compute capex for battery and/or PV
			self._capex = self._compute_capex()
			#Compute player max power
			self._grid_purchase_max = np.ceil(profile_wd.max())

			#Set player type - Useful for later usage
			if self._pv_size > 0:
				if self._grid_purchase_max == 0:
					self.player_type = "producer"
				else:
					self.player_type = "prosumer"
			else:
				self.player_type = "consumer"

			# Store shapley value
			self.shapley = None

		def _simulate_profiles(self):
			"""Run simulator to generate profiles"""
			raise NotImplementedError("Integrate Lorenti's simulation engine") 

		def _compute_capex(self):
			"""Compute capex for battery and/or PV"""
			#Solar
			if self._pv_size < 20:
				capex_pv = self.spec_capex_pv[0]*self._pv_size
			else:
				capex_pv = self.spec_capex_pv[1]*self._pv_size
			#Battery
			capex_battery = self.spec_capex_battery*self._battery_size
			#Total amount
			capex = capex_pv + capex_battery
			return capex

		def shapley_value(self, vfdb, approx_order = None):
			"""Compute shapley value of this user.
			Parameters:
			vfdb: value function database (dict)
			approx_order: approximation order
			Returns:
			shapley: float. Shapley value for this player
			"""
			n = len(list(vfdb.keys())[0]) #n_players
			#Compute all configurations excluding this players
			configs_without = [key for key in vfdb if key[self.player_number] == 0]
			shapley = 0
			if approx_order is None:
				order = 0
			else:
				order = n - approx_order
			
			for conf in configs_without:
				if sum(conf) < order:
					print("Skipping") #debug
					pass
				else:
					conf_with = list(conf)
					conf_with[self.player_number] = 1
					conf_with = tuple(conf_with) #configuration adding this player
					s = sum(conf)
					term = vfdb[conf_with] - vfdb[conf] #Added value
					weight = factorial(s)*factorial(n - s - 1)*(1/factorial(n)) #weight according to definition
					shapley += weight*term
			#Save shapley value for later usage
			self.shapley = shapley
			
			############################################ ADDED PLAYER'S PAYOFF
			self.payoff = shapley
			##################################################################

			return shapley


#Game Class - Private Methods
	def __init__(self):
		"""Create an instance of the game.
		"""

		#Create player list
		self.players = self._create_players()
		self._n_players = len(self.players)

		#Auxiliary variables
		# Seasons, storing for each season an id number and nickname
		self._seasons = {'winter': {'id': 0, 'nickname': 'w'},
						'spring': {'id': 1, 'nickname': 'ap'},
						'summer': {'id': 2, 'nickname': 's'},
						'autumn': {'id': 3, 'nickname': 'ap'},
						}
		# self._months, storing for each month an id number and nickname				
		self._months = {'january': {'id': 0, 'nickname': 'jan', 'season': 'winter'},
						'february': {'id': 1, 'nickname': 'feb', 'season': 'winter'},
						'march': {'id': 2, 'nickname': 'mar', 'season': 'winter'},
						'april': {'id': 3, 'nickname': 'apr', 'season': 'spring'},
						'may': {'id': 4, 'nickname': 'may', 'season': 'spring'},
						'june': {'id': 5, 'nickname': 'jun', 'season': 'spring'},
						'july': {'id': 6, 'nickname': 'jul', 'season': 'summer'},
						'august': {'id': 7, 'nickname': 'aug', 'season': 'summer'},
						'september': {'id': 8, 'nickname': 'sep', 'season': 'summer'},
						'october': {'id': 9, 'nickname': 'oct', 'season': 'autumn'},
						'november': {'id': 10, 'nickname': 'nov', 'season': 'autumn'},
						'december': {'id': 11, 'nickname': 'dec', 'season': 'autumn'},
						}
		# Day types, storing for each day type an id number and nickname
		self._day_types = {'work-day': {'id': 0, 'nickname': 'wd'},
							'weekend-day': {'id': 1, 'nickname': 'we'},
							}
		# Distribution of both day types among all self._months
		self._days_distr_months = {'january': {'work-day': 21, 'weekend-day': 10},
								'february': {'work-day': 20, 'weekend-day': 8},
								'march': {'work-day': 23, 'weekend-day': 8},
								'april': {'work-day': 22, 'weekend-day': 8},
								'may': {'work-day': 21, 'weekend-day': 10},
								'june': {'work-day': 22, 'weekend-day': 8},
								'july': {'work-day': 22, 'weekend-day': 9},
								'august': {'work-day': 22, 'weekend-day': 9},
								'september': {'work-day': 22, 'weekend-day': 8},
								'october': {'work-day': 21, 'weekend-day': 10},
								'november': {'work-day': 22, 'weekend-day': 8},
								'december': {'work-day': 23, 'weekend-day': 8},
								}
		# Number of seasons, months and day_types
		self._n_seasons, self._n_months, self._n_day_types = \
			len(self._seasons), len(self._months), len(self._day_types)

		# Distribution of both day types among all seasons
		self._days_distr_seasons = {}
		for month in self._months:
			season = self._months[month]['season']
			
			work_days_month = self._days_distr_months[month]['work-day']
			weekend_days_month = self._days_distr_months[month]['weekend-day']
			
			if season not in self._days_distr_seasons: 
				self._days_distr_seasons[season] = {'work-day': work_days_month,
											  'weekend-day':  weekend_days_month}
			else: 
				self._days_distr_seasons[season]['work-day'] += work_days_month
				self._days_distr_seasons[season]['weekend-day'] += weekend_days_month
				
		# Days distributions as arrays useful for quicker calculations
		self._days_distr_months_array = np.zeros((self._n_months, self._n_day_types))
		for month in self._months:
			mm = self._months[month]['id']
			for day_type in self._day_types:
				dd = self._day_types[day_type]['id']
				self._days_distr_months_array[mm, dd] = self._days_distr_months[month][day_type]
				
		self._days_distr_seasons_array = np.zeros((self._n_seasons, self._n_day_types))
		for season in self._seasons:
			ss = self._seasons[season]['id']
			for day_type in self._day_types:
				dd = self._day_types[day_type]['id']
				self._days_distr_seasons_array[ss, dd] = self._days_distr_seasons[season][day_type]

		# Auxiliary time dictionary
		self._auxiliary_dict = {'seasons': self._seasons,
							  'months': self._months,
							  'day_types': self._day_types,
							  'days_distr_seasons': self._days_distr_seasons,
							  'days_distr_months': self._days_distr_months,
							  'days_distr_months_array': self._days_distr_months_array,
				  			}

		# PV data
		filename = 'pv_production_unit.csv'
		self._pv_data = np.array(pd.read_csv(filename, sep=';'))
		self._pv_production_unit_months = self._pv_data[:, 1:]
		# Broadcasting the array in order to account for different day types
		self._pv_production_unit_months = self._pv_production_unit_months[:, :, np.newaxis]
		broadcaster = np.zeros((self._n_day_types,))
		self._pv_production_unit_months = self._pv_production_unit_months + broadcaster

		# Battery specs (constant)
		# Maximum and minimum states of charge (SOC) (%)
		# Minimum time of charge/discharge (h)
		# Charge, discharge and self-discharge efficiencies (-)
		# Size is not present as it varies according to subconfiguration
		self._bess = storage_system_specifications(default_flag=False)['bess']

		# Time discretization
		# Time-step (h)
		self._dt = 1
		# Total time of simulation (h)
		self._time = 24
		# Vector of time (h)
		self._time_sim = np.arange(0, self._time, self._dt)
		self._time_length = self._time_sim.size
		self._time_dict = {'dt': self._dt,
					  'time': self._time,
					  'time_sim': self._time_sim,
					  }
		
		#Store Shapley values once computed
		self.shapley_vals = None
		self.payoffs = None
							
	def _create_db(self):
		"""Create database with value function result foreach configuration"""
		#Create all configuration
		configs = list(product((0, 1),
								repeat = self._n_players)
								)
		#Create database
		#Here use dictionary comprehension; eventually use a starmap
		db = {key : self._value_function(key) for key in configs}
		return db

	def _value_function(self, config, mode = 'democratic'):
		"""Compute value of given config. Wrapper for
		Lorenti's module.
		Parameters:
		config: binary iterable of length self._n_players
		mode: string in {'democratic', 'republican'}
		Returns:
		float, positive value of config.
		"""
		#If configuration is empty, value is zero
		threshold = 0 if mode == 'republican' else 1
		if sum(config) <= threshold:
			return 0 #No consumption -> No shared energy
		profile_wd, profile_we, pv_size, bess_size, grid_purchase_max = self._subconfig_inputs(config)
		
		if pv_size == 0:
			return 0 #No shared energy, nor power sold

		# Optimization Setup
		# Size (kW, electric)
		pv = {'size': pv_size,}
		## Electricicty distribution grid (grid)
		# Maximum purchasable power (kW, electric)
		# Maximum feedable power (kW, electric) - Set equal to pv_size
		grid = {'purchase_max': grid_purchase_max,
				'feed_max': pv_size,
				}
		## Battery energy storage system (bess)
		# Size (kWh, electric)
		bess_flag = bess_size != 0 # (Is Battery present in config?)

		#Create technologies dictionary
		technologies_dict = {'pv': pv,
					 'grid': grid,                      
					}
		#Battery size - Update for each configuration
		self._bess['size'] = bess_size
		if bess_flag != 0: technologies_dict['bess'] = self._bess
		# Electric load (kWh/h) evaluated using the routine for work- and weekend-days
		ue_demand_months = np.zeros((self._time_length, self._n_months, self._n_day_types))
		for day_type, data in zip(self._day_types, [profile_wd, profile_we]):
			dd = self._day_types[day_type]['id']
			ue_demand_months[:, :, dd] = data

		# Load & Production dictionary
		profiles_months = {'ue_demand_months': ue_demand_months, #(24x12x2)
							'pv_production_unit_months': self._pv_production_unit_months,
							}	

		# Run Optimization
		yearly_energy = configuration_evaluation(self._time_dict, 
													profiles_months,
													technologies_dict,
													self._auxiliary_dict
													)

		#Extract quantities of interest
		shared_energy = yearly_energy['shared power']
		grid_feed = yearly_energy['injections']

		#Compute value
		value = self._economic_value(grid_feed, shared_energy)
		
		if sum(config) == self._n_players:
			print('Total value: {} euro/year'.format(value))

		return value
		
	def _create_players(self):
		"""Create players for game.
		Returns:
		List of _Player objects
		
		"""
		basepath= Path(__file__).parent
		datapath = basepath / 'Data'
		player_data_folders = os.listdir(datapath)
		player_list = [] #Store _Player objects
		for ix, folder in enumerate(player_data_folders):
			#Workday
			wd_path = datapath / folder / "consumption_profiles_month_wd.csv"
			wd_data = pd.read_csv(wd_path,
								  sep = ';',
								  decimal= ',',
								  ).dropna().values[:,1:]
			#Weekend
			we_path = datapath / folder / "consumption_profiles_month_we.csv"
			we_data = pd.read_csv(we_path,
								  sep = ';',
								  decimal= ',',
								  ).dropna().values[:,1:]

			#Instantiate new player
			newplayer = self._Player(player_number = ix, player_name = folder, 
									 profile_wd = wd_data, profile_we = we_data)
			player_list.append(newplayer)

		return player_list

	def _subconfig_inputs(self, config):
		"""Generate all subconfiguration inputs.
		Parameters:
		config: binary iterable.
		Returns:
		profile_wd: np.array of shape (24,12)
		profile_we: np.array of shape (24,12)
		pv_size: float
		battery_size : float
		"""
		profile_wd = np.zeros((24,12))
		profile_we = np.zeros((24,12))
		pv_size = 0
		battery_size = 0
		grid_purchase_max = 0
		sublist = [player for ix, player in enumerate(self.players) if config[ix] == 1]
		for player in sublist:
			profile_wd += player._profile_wd
			profile_we += player._profile_we
			pv_size += player._pv_size
			battery_size += player._battery_size
			grid_purchase_max += player._grid_purchase_max

		return profile_wd, profile_we, pv_size, battery_size, grid_purchase_max
#----------------------------------------Economic Functions---------------------------------------
	def _economic_value(self, grid_feed, shared_energy, PR3 = 42,  CUAF = 8.56 , TP = 110, feed_tax = 0.22):
		"""Return economic value of shared energy plus energy sales"""
		ritiro_energia = grid_feed/1000 * PR3 * (1-feed_tax)
		incentivo = shared_energy/1000 * (CUAF + TP)
		return (ritiro_energia + incentivo)*0.98

	def _cash_flows(self, player, OM = 30, n_years = 20):
		"""Compute simplified cash flows for given player"""
		yearly_revenue = player.payoff
		yearly_expense = player._pv_size * OM

		cash_flows = [yearly_revenue - yearly_expense] * n_years
		cash_flows = np.array(cash_flows)
		
		return cash_flows
	
	def _pbt(self, player):
		"""Pay-Back Time"""
		initial_investment = player._capex
		cash_flows = self._cash_flows(player)
		year = 0 
		amount = 0 #Total money
		while amount < initial_investment:
			amount += cash_flows[year]
			year += 1
			if year == len(cash_flows):
				return 'Infeasible' 
		return year
		
	def _pcr(self, player, n_years = 20, OM = 30, energy_price = 230):
		"""Percentage Cost Reduction"""
		#OM + Amortations
		yearly_expense = player._pv_size * OM + player._capex / n_years
		yearly_revenue = player.payoff

		#yearly energy consumption & expense
		profiles_months = np.stack([player._profile_wd, player._profile_we], axis = 2)
		yearly_consumption, _ = yearly_monthly_energy(self._time_dict,
													  profiles_months,	
													  self._auxiliary_dict) #KWh
		yearly_energy_expense = yearly_consumption/1000 * energy_price #KWh to MWh
		#Compute kpi
		pcr = 100 * (yearly_revenue - yearly_expense)/yearly_energy_expense
		return pcr

	def _irr(self, player):
		"""Internal Return Rate"""
		initial_investment = player._capex
		cash_flows = list(self._cash_flows(player))
		cash_flows = [- initial_investment] + cash_flows
		return irr(cash_flows)

		
#Game Class - Public Methods
	
	def play(self, approx_order = None, return_vals = True):
		"""Run Game and Plot results.
		Parameters:
			approx_order: int or None -- Order of approximation in shapley 
			value calculation. Default: None - No approximation
		Return:
		shapley_vals: list with shapley value for each player"""
		#Create value function database
		self._vfdb = self._create_db() #{"(00010000)":vf}
		#Compute Shapley values
		shapley_vals = [player.shapley_value(self._vfdb, approx_order) for player in self.players]
		#Save shapley values
		self.shapley_vals = shapley_vals
		
		############################ ADDED EVALUATION OF PLAYER'S PAYOFFS LIST
		payoffs = [player.payoff for player in self.players]
		self.payoffs = payoffs
		######################################################################
		
		if return_vals:
			# Plot results
			# Use Lorenti's module
			#Create input dataframe
			names = [player.player_name for player in self.players] #player names
			types = [player.player_type for player in self.players] #producers/consumers/prosumers
			distribution = shapley_vals / sum(shapley_vals)
			plot_input = pd.DataFrame({'player': names,
									  'share': distribution,
									  'type': types
								 	})
			figure = shares_pie_plot(plot_input) #Plot data 
			plt.show(figure)

			#Return benefit shares
			return shapley_vals

	def compute_kpis(self):
		"""Compute and return main KPIs for each player.
		Return pd.DataFrame with KPIs"""
		if self.payoffs is None: #Generate shapley values
			self.play(return_vals = False)

		# Compute KPIs for each player
		#columns = ["PBT", "PCR", "IRR"]
		kpi_array = np.empty(shape = (self._n_players, 3)) #store values
		kpi_array.fill(np.nan)
		for ix, player in enumerate(self.players):
			if player.player_type == "producer":
				kpi_array[ix, 0] = self._pbt(player)
				kpi_array[ix, 2] = self._irr(player)

			elif player.player_type == "consumer":
				kpi_array[ix, 1] = self._pcr(player)

			elif player.player_type == "prosumer":
				kpi_array[ix, 0] = self._pbt(player)
				kpi_array[ix, 1] = self._pcr(player)
				kpi_array[ix, 2] = self._irr(player)

		#Return result as a dataframe
		kpi_df = pd.DataFrame(data = kpi_array,
			   				 columns = ["PBT", "PCR", "IRR"]
			   				 )
		return kpi_df
	
	def find_ms_share(self, min_pcr_min=8, avg_irr_min=0.06, avg_pbt_max=15, delta_ms=0.001):
		"""
		Finds the maximum value for the REC's management service's share
		that allows to keep good KPIs for the members
		"""
		# Flag to continue exploring the reduction in players'payoffs
		keep_going = True
		# Quantity to remove from each payoff at each iteration
		payoffs_delta = [delta_ms*player.payoff for player in self.players]
		while keep_going:
			try:
				# Try to remove the delta from each pay-off and run the 
				# evaluation of the kpis
				payoffs = []
				for player, payoff_delta in zip(self.players, payoffs_delta):
					player.payoff = player.payoff - payoff_delta
					payoffs.append(player.payoff)
				self.payoffs = payoffs
				kpi_df = self.compute_kpis()
				# An average value of each indicator can be evaluated and 
				# compared with the related bound value
				# Pay-back time
				pbts = np.array(kpi_df['PBT'])
				avg_pbt = np.nanmean(pbts)
				# Percentage cost reduction
				pcrs = np.array(kpi_df['PCR'])
				min_pcr = np.nanmin(pcrs)
				# Internal rate of return
				irrs = np.array(kpi_df['IRR'])
				avg_irr = np.nanmean(irrs)
				
				keep_going = (avg_pbt <= avg_pbt_max) and \
					(min_pcr >= min_pcr_min) and \
					(avg_irr >= avg_irr_min)
			except:
				# If an error is raised during the evaluation of the kpis,
				# the flag to continue the exploration is deactivated
				keep_going = False
			
			if keep_going == False:
				# If the flag is deactivated (due to an error or due to a
				# kpi that reaches the bound) the payoffs are restored to 
				# the previous values
				payoffs = []
				for player, payoff_delta in zip(self.players, payoffs_delta):
					player.payoff = player.payoff + payoff_delta
					payoffs.append(player.payoff)
				self.payoffs = payoffs
		
		# Evaluate management service's payoff and share
		ms_value = sum(self.shapley_vals) - sum(self.payoffs)
		ms_share = ms_value / sum(self.shapley_vals)
		return ms_share
		

##### 	UNIT TEST #######
if __name__ == '__main__':
	# Run game
	game = BenefitDistributionGame()
	shapley_vals  = game.play()
	ms_share = game.find_ms_share()
	kpis = game.compute_kpis()


