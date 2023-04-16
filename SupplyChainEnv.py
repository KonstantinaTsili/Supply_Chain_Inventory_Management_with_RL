import gymnasium as gym
from gym import spaces
import numpy as np
import pandas as pd
import tiles3 as tc
import copy

class SupplyChainEnv(gym.Env):
    def __init__(self, sale_prices, warehouse_storage_costs, manufacturer_storage_costs, manufacturer_storage_cap, warehouse_storage_cap, production_costs, production_cap, shipping_costs, demand_distribution, shipping_capacity, num_products=2, num_distr_warehouses=2):

        ''' Define the supply chain parameters'''
        self.num_products = num_products
        self.num_distr_warehouses = num_distr_warehouses

        self.demand_distribution = demand_distribution
        self.sale_prices = sale_prices

        self.production_costs = production_costs
        self.production_cap = production_cap

        self.manufacturer_storage_costs = manufacturer_storage_costs
        self.manufacturer_storage_cap = manufacturer_storage_cap

        self.warehouse_storage_costs = warehouse_storage_costs
        self.warehouse_storage_cap = warehouse_storage_cap

        self.shipping_costs = shipping_costs
        self.shipping_capacity = shipping_capacity

        self.penalty_costs = 1.2 * self.sale_prices #for unsatisfied demand, 1 for the fleating cost plus 0.5 to account for unsatisfaction of client (may not come back)

        self.T = 12  # final time step (e.g., an episode takes 12 time steps = A Yearly Exercise of the activity)

        self.t=0  # Current time_step

        # Define the state space
        '''
        The state space is composed of:
        - The time step: min = 0, max = self.T
        - Manufacturer inventory (for each product): min = 0, max = self.manufacturer_storage_cap
        - Warehouses inventories (for each warehouse and product): min = 0, max = self.warehouse_storage_cap
        - The Customers' Demands (for each warehouse and product): min = 0, max = 400000
        '''
        self.observation_high = np.concatenate([[self.T],self.manufacturer_storage_cap, self.warehouse_storage_cap.flatten(),[4000]*self.num_products * self.num_distr_warehouses])
        self.observation_space = spaces.Box(low=np.zeros_like(self.observation_high), high=self.observation_high, dtype=np.int32) #lowest inventory level is zero

        #Define the action space
        '''
        The action space is composed of:
        - The quantities that were produced (for each product): min = 0, max = production capacity
        - The quantities shipped between the factory and the warehouses (for each warehouse and product): min = 0, max = total shipping capacity
        '''
        manufacturer_production_space = spaces.Box(low=0, high=self.production_cap, shape=(num_products,), dtype=np.int32)
        warehouse_shipping_space = spaces.Box(low=0, high=np.vstack([self.shipping_capacity]*self.num_products).T, shape=(self.num_distr_warehouses,self.num_products), dtype=np.int32)
        self.action_space = spaces.Tuple((manufacturer_production_space, warehouse_shipping_space))

        """
        Initializes the Tile Coder
        Initializers:
        iht_size -- int, the size of the index hash table, typically a power of 2
        num_tilings -- int, the number of tilings
        num_tiles -- int, the number of tiles. Here both the width and height of the tiles are the same
                            
        Class Variables:
        self.iht -- tc.IHT, the index hash table that the tile coder will use
        self.num_tilings -- int, the number of tilings the tile coder will use
        self.num_tiles -- int, the number of tiles the tile coder will use
        """
        iht_size=4096
        num_tilings=32
        num_tiles=8
        
        self.num_tilings = num_tilings
        self.num_tiles = num_tiles 
        self.iht = tc.IHT(iht_size)



        self.reset() # Reset the environment to its initial state at the beginning of a new episode (function defined just below)

    def reset(self):
        # Initialize the inventory levels and costs
        self.manufacturer_inventory = self.manufacturer_storage_cap * 0.5
        self.warehouse_inventories = self.warehouse_storage_cap * 0.5
        
        # Generate initial demand for each warehouse
        self.generate_demand()
        self.t = 0

        # Return the initial observation
        return self._get_observation()

    def step(self, action):

        # Fetch the actions
        manufacturer_production, warehouse_shipping = action
        
        # Take into account production capacity, make sure we havea n integer
        manufacturer_production = np.round(np.minimum(manufacturer_production,self.production_cap),decimals=0)

        # Take into account transport capacity
        shipping_ratios_by_warehouse = np.divide(warehouse_shipping, np.sum(warehouse_shipping, axis=0).reshape(-1,1), out=np.zeros_like(warehouse_shipping, dtype=float), where=np.sum(warehouse_shipping, axis=0).reshape(-1,1)!=0)
        warehouse_shipping = np.minimum(warehouse_shipping,(shipping_ratios_by_warehouse*self.shipping_capacity.reshape((-1,1))).astype(int))


        #-------------UPDATING INVENTORIES------------------------
             
        
        # Add new produced goods to inventory
        self.manufacturer_inventory += manufacturer_production 
        
        # Ship as many goods as decided, granted they are in inventory
        shipped_goods = np.minimum(np.sum(warehouse_shipping, axis=1), self.manufacturer_inventory)
        shipping_ratios_by_product = np.divide(warehouse_shipping, np.sum(warehouse_shipping, axis=1).reshape(-1, 1), out=np.zeros_like(warehouse_shipping, dtype=float), where=np.sum(warehouse_shipping, axis=1).reshape(-1,1)!=0)
        warehouse_shipping = np.round(shipping_ratios_by_product*shipped_goods.reshape(-1,1),decimals=0)
        self.manufacturer_inventory -= shipped_goods

        # All produced goods that cannot be stored after shipping are destroyed. Any over-production is penalized later in production costs.
        overcapacity_production = np.maximum(self.manufacturer_inventory - self.manufacturer_storage_cap, 0)
        self.manufacturer_inventory = np.minimum(self.manufacturer_inventory, self.manufacturer_storage_cap) 

        # Update warehouse inventories on shipped inventory. As for manufacturer inventory, all products that were produced but could not be stored will be destroyed.
        overcapacity_shipping = np.maximum(self.warehouse_inventories+ warehouse_shipping - self.warehouse_storage_cap, 0)
        self.warehouse_inventories = np.minimum(self.warehouse_inventories + warehouse_shipping, self.warehouse_storage_cap)
        self.available_product = copy.deepcopy(self.warehouse_inventories)

        #-------------CALCULATING REVENUE----------------------------

        # taking the minimum to account for the case we do not have enough to satisfy demand
        revenue = np.sum(self.sale_prices * np.sum(np.minimum(self.demands, self.warehouse_inventories),axis=0)) 

        # Compute Constrained THEORETICAL OPTIMAL REVENUES, while taking into account warehouse storage capacity
        constrained_max_demand = np.minimum(self.demands,self.warehouse_storage_cap) 
        rev_opt = np.sum(self.sale_prices * np.sum(constrained_max_demand,axis=0)) # Theoretical Max revenues
        
        # Update warehouse inventories (after satisfaction of demand)
        self.warehouse_inventories -= self.demands # this will result in negative values for unsatisfied demand


        #-------------CALCULATING COSTS------------------------------

        # Calculate production costs
        production_costs = np.sum(manufacturer_production * self.production_costs)
        opt_production_costs = np.sum(np.maximum(manufacturer_production-overcapacity_production,0) * self.production_costs) # In a perfect world, no overcapacity of production

        # Calculate shipping costs 
        shipping_costs = np.sum(self.shipping_costs*warehouse_shipping)
        opt_shipping_costs = np.sum(self.shipping_costs*np.maximum(warehouse_shipping-overcapacity_shipping,0)) # In a perfect world, no overcapacity of shipping

        # Calculate penalty costs for unsatisfied demand
        unsatisfied_demand = np.minimum(self.warehouse_inventories, 0)
        penalty_costs = -np.sum(np.sum(unsatisfied_demand, axis=1) * self.penalty_costs) # minus sign because stock levels would be already negative in case of unfulfilled demand
        opt_penalty_costs = -np.sum(np.sum(constrained_max_demand - self.demands, axis=1) * self.penalty_costs) # In a perfect world, penalties are only incurred due to the warehouses' capacity limits

        # Update warehouse inventories so that they do not have negative values anymore
        self.warehouse_inventories = np.maximum(self.warehouse_inventories, 0)

        # Calculate storage costs associated with the warehouse inventories, storage costs are for the inventory not yet sold
        warehouse_storage_costs = np.sum(self.warehouse_storage_costs * self.warehouse_inventories)  
        manufacturer_storage_costs = np.sum(self.manufacturer_storage_costs * self.manufacturer_inventory)   

        # Calculate total cost
        total_cost = production_costs + shipping_costs + penalty_costs + warehouse_storage_costs + manufacturer_storage_costs 
        cost_opt = opt_production_costs + opt_shipping_costs + opt_penalty_costs # In a perfect scenario, there would be almost no storage costs

        #-------------------CALCULATING GROSS INCOME = reward----------------------
        reward = revenue - total_cost
        income_opt = rev_opt - cost_opt

        self.t += 1

        
        # Return the new observation, reward, done flag, and info dictionary
        observation = self._get_observation()

        # Generate new demand for each warehouse
        self.generate_demand()

        if self.t >= self.T:
          done = True
        else:
          done = False

        info = {'theoretical_max_reward': income_opt, 'total_overcapacity': overcapacity_production + np.sum(overcapacity_shipping,axis=0),
                 'production_costs': production_costs, 'shipping_costs': shipping_costs, 'penalty_costs': penalty_costs,
                 'warehouse_storage_costs': warehouse_storage_costs, 'manufacturer_storage_costs' : manufacturer_storage_costs}

        return observation, reward, done, info

    def generate_demand(self):
        # self.demands = np.zeros((self.num_distr_warehouses, self.num_products))
        # for prod in range(self.num_products):
        #   for ware in range(self.num_distr_warehouses):
        #     demand_dist = self.demand_distribution[prod][ware][self.t]
        #     self.demands[prod][ware] = np.random.choice(demand_dist)
        
        #Copy the demande distribution
        demand_df = self.demand_distribution.copy()

        # compute the demand based from a time series model : Demand = Init + Trend * t + Seasonal(t) + Noise(t)
        demand_df['new'] = demand_df.apply(lambda row: row['initial'] + self.t * row['trend'] + np.random.default_rng().normal(0, row['std']),axis=1) 
        demand_df['new'] = demand_df.apply(lambda row: row['new'] if row['period']==0 else row['new']+ row['amplitude'] * np.cos( 2*np.pi*self.t/row['period']),axis=1) 

        # Round up
        demand_df['new'] = np.ceil(demand_df['new'])

        # Update the demands
        self.demands = np.maximum(0,demand_df['new'].to_numpy().astype(int).reshape((self.num_products, self.num_distr_warehouses)))


    def _get_observation(self):
        # Concatenate the manufacturer inventory and warehouse inventories
        inventories = np.concatenate([[self.t],self.manufacturer_inventory, self.warehouse_inventories.flatten()]) #.flatten returns a copy on an array collapsed into one dimension
        
        # Concatenate the inventories with the demands
        observation = np.concatenate([inventories, self.demands.flatten()])
        
        return observation 
  
    def get_tiles(self, state):
        """
        Takes in an angle and angular velocity from the pendulum environment
        and returns a numpy array of active tiles.
        
        Arguments:
        angle -- float, the angle of the pendulum between -np.pi and np.pi
        ang_vel -- float, the angular velocity of the agent between -2*np.pi and 2*np.pi
        
        returns:
        tiles -- np.array, active tiles
        
        """
        
        ### Use the ranges above and scale the angle and angular velocity between [0, 1]
        # then multiply by the number of tiles so they are scaled between [0, self.num_tiles]
        max_val = self.observation_high
        min_val = np.zeros_like(max_val)
        scaled = (state - min_val)/(max_val - min_val) *  self.num_tiles

        # Get tiles by calling tc.tileswrap method
        # wrapwidths specify which dimension to wrap over and its wrapwidth
        tiles = tc.tileswrap(self.iht, self.num_tilings, scaled, wrapwidths=[self.num_tiles, False])
                    
        return np.array(tiles)
