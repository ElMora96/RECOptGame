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
from evaluation_quicky import configuration_evaluation #To compute value function
from systems_specifications import storage_system_specifications #Read battery specs
from plotter import shares_pie_plot
import os
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt

#Plot configuration
sns.set_theme() #use to reset955
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
		def __init__(self, player_number, player_name, profile_we = None, profile_wd = None, 
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
			#Compute approx yearly energy expense
			self._energy_expense = self._compute_energy_expense()
			#Compute player max power
			self._grid_purchase_max = np.ceil(profile_wd.max())

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

		def _compute_energy_expense(self)
			"""Compute approximate yearly expense for energy"""
			raise NotImplementedError("Do the trick with lorenti")

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

	def _value_function(self, config):
		"""Compute value of given config. Wrapper for
		Lorenti's module.
		Parameters:
		config: binary iterable of length self._n_players
		Returns:
		float, positive value of config.
		"""
		#If configuration is empty, value is zero
		if sum(config) <= 1:
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
		shared_energy = yearly_energy['shared power']*self._dt #Check better for different discretizations
		grid_feed = yearly_energy['injections']

		#Compute value
		value = self._economic_value(shared_energy, grid_feed)

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
			newplayer = self._Player(ix, folder, wd_data, we_data)
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

	def _economic_value(self, shared_energy, grid_feed, PR3 = 42,  CUAF = 8.56 , TP = 110):
		"""Return economic value of shared energy plus energy sales"""
		ritiro_energia = grid_feed/1000 * PR3
		incentivo = shared_energy/1000 * (CUAF + TP)
		return ritiro_energia + incentivo
		
#Game Class - Public Methods
	
	def play(self, approx_order = None):
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
		distribution = shapley_vals / sum(shapley_vals)
		# Plot results
		# Use Lorenti's module
		#Create input dataframe
		names = [player.player_name for player in self.players] #player names
		types = [] #producers/consumers/prosumers
		for player in self.players:
			if player._pv_size > 0:
				if player._grid_purchase_max == 0:
					types.append("producer")
				else:
					types.append("prosumer")
			else:
				types.append("consumer")
		plot_input = pd.DataFrame({'player': names,
								  'share': distribution,
								  'type': types
							 	})
		figure = shares_pie_plot(plot_input) #Plot data 
		plt.show(figure)
		
		#Save shapley values
		self.shapley_vals = shapley_vals

		#Return benefit shares
		return shapley_vals


##### 	UNIT TEST #######
if __name__ == '__main__':
	# Run game
	game = BenefitDistributionGame()
	shapley_vals  = game.play()

