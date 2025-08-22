import numpy as np
import networkx as nx
import copy

class agent_profile():
    def __init__(self, stage_idx, agent_idx, role, prod_capacity, sale_price, init_inventory, backlog_cost, 
                 holding_cost, order_cost, lead_time, suppliers, customers):
        self.name = f"stage_{stage_idx}_agent_{agent_idx}"
        self.role = role
        self.stage = stage_idx
        self.agent = agent_idx
        self.prod_capacity = prod_capacity
        self.sale_price = sale_price
        self.sales = 0
        self.inventory = init_inventory
        self.backlog = 0
        self.backlog_cost = backlog_cost
        self.holding_cost = holding_cost
        self.order_cost = order_cost
        self.lead_time = lead_time
        self.fulfilled_rate = 1
        self.delivery = 1
        self.suppliers = suppliers
        self.customers = customers

    
    def update(self, sales, delivery, inventory, backlog, req_order, suppliers, customers):

        self.sales = sales
        self.delivery = delivery
        self.inventory = inventory
        self.backlog = backlog
        self.fulfilled_rate = sales / req_order
        self.suppliers = suppliers
        self.customers = customers

    def get_node_features(self):
        return np.array([self.prod_capacity, self.sale_price, self.sales, self.inventory, self.backlog, self.backlog_cost])
    
    def get_edge_features(self):
        return self.lead_time # order cost?
    
    

def create_agent_profiles(env_config: dict):

    agent_profiles = []
    num_stages = env_config['num_stages']
    num_agents_per_stage = env_config['num_agents_per_stage']
    for i in range(num_stages):
        for j in range(num_agents_per_stage):
            agent_profiles.append(agent_profile(stage_idx=i, agent_idx=j, role=env_config["stage_names"][i], prod_capacity=env_config['prod_capacities'][i*num_agents_per_stage+j],  
                                                sale_price=env_config['sale_prices'][i*num_agents_per_stage+j], init_inventory=env_config['init_inventories'][i*num_agents_per_stage+j], 
                                                backlog_cost=env_config['backlog_costs'][i*num_agents_per_stage+j], holding_cost=env_config['holding_costs'][i*num_agents_per_stage+j], 
                                                order_cost=env_config['order_costs'][i*num_agents_per_stage:(i+1)*num_agents_per_stage], lead_time=env_config['lead_times'][i][j],
                                                suppliers=env_config['supply_relations'][i][j], customers=env_config['demand_relations'][i][j]))
    
    return agent_profiles


class SupplyChain_Graph():

    def __init__(self, agent_profiles: list, num_stages: int, num_agents_per_stage: int):
        
        self.init_G = self._build_nx_graph(agent_profiles=agent_profiles)
        self.num_stages = num_stages
        self.num_agents_per_stage = num_agents_per_stage
        self.reset_G()

    def reset_G(self):
        self.G = copy.deepcopy(self.init_G)

    def _build_nx_graph(self, agent_profiles):

        G = nx.DiGraph()

        for ag in agent_profiles:
            G.add_node(ag.name, prod_capacity=ag.prod_capacity, sale_price=ag.sale_price, sales=ag.sales, 
                inventory=ag.inventory, backlog=ag.backlog, backlog_cost=ag.backlog_cost, stage=ag.stage)
            
        for ag in agent_profiles:
            if ag.role == "manufacturer":
                continue
            lead_time = ag.lead_time
            order_cost = ag.order_cost
            stage_idx = ag.stage
            for sup_idx in range(len(lead_time)):
                G.add_edge(u_of_edge=ag.name, v_of_edge=f"stage_{stage_idx+1}_agent_{sup_idx}", lead_time=lead_time[sup_idx], order_cost=order_cost[sup_idx])
            
            for sup_idx, label in enumerate(ag.suppliers):
                if label == 1:
                    G.add_edge(u_of_edge=f"stage_{stage_idx+1}_agent_{sup_idx}", v_of_edge=ag.name, supplier=True)
                    G.add_edge(u_of_edge=ag.name, v_of_edge=f"stage_{stage_idx+1}_agent_{sup_idx}", customer=True)

        return G
    
    def update_graph(self, state_dict: dict, past_req_orders: dict):
        # update dynamic info, such as sales, delievery, past_req_order, backlog, upstream_backlog
        for i in range(self.num_stages):
            for j in range(self.num_agents_per_stage):
                agent_state = state_dict[f"stage_{i}_agent_{j}"]
                sales = agent_state["sales"]
                backlog = agent_state["backlog"]
                upstream_backlog = agent_state["upstream_backlog"]
                deliveries = agent_state["deliveries"]
                inventory = agent_state["inventory"]
                pr_orders = past_req_orders.get(f"stage_{i}_agent_{j}", [])
                suppliers = agent_state["suppliers"]

                self.G.nodes[f"stage_{i}_agent_{j}"]["inventory"] = inventory
                self.G.nodes[f"stage_{i}_agent_{j}"]['backlog'] = backlog
                self.G.nodes[f"stage_{i}_agent_{j}"]['upstream_backlog'] = upstream_backlog
                self.G.nodes[f"stage_{i}_agent_{j}"]['sales'] = sales[-1] 
 
                if i < self.num_stages - 1:
                    for k in range(self.num_agents_per_stage): # to upstream suppliers except the manufacturers
                        # add new directional edge for indicating delivery
                        if sum(deliveries[k]):
                            for day in range(len(deliveries[k])):
                                if deliveries[k][-day] > 0:        
                                    self.G.add_edge(f"stage_{i+1}_agent_{k}", f"stage_{i}_agent_{j}")
                                    self.G[f"stage_{i+1}_agent_{k}"][f"stage_{i}_agent_{j}"][f'deliveries in {day} days'] = deliveries[k][-day]

                    if len(pr_orders) > 0:
                        for k in range(self.num_agents_per_stage):      
                            if pr_orders[k] > 0:
                                self.G[f"stage_{i}_agent_{j}"][f"stage_{i+1}_agent_{k}"]['past_req_orders'] = pr_orders[k]
                    
                    for k, label in enumerate(suppliers):
                        if label == 1:
                            self.G.add_edge(f"stage_{i+1}_agent_{k}", f"stage_{i}_agent_{j}", supplier=True)
                            self.G.add_edge(f"stage_{i}_agent_{j}", f"stage_{i+1}_agent_{k}", customer=True)
                        else: # label == 0
                            self.G.add_edge(f"stage_{i+1}_agent_{k}", f"stage_{i}_agent_{j}", supplier=False)
                            self.G.add_edge(f"stage_{i}_agent_{j}", f"stage_{i+1}_agent_{k}", customer=False)

