import torch
from torch_scatter import scatter
from torch.distributions.bernoulli import Bernoulli
from torch_geometric.data import Data
import yaml
import argparse
import time

'''
Sampler
'''
def sample_initializer(problem_type, probs, config,
                       device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), data=None):
    if problem_type in ["r_cheegercut", "n_cheegercut"]:
        samples = torch.zeros(config['total_mcmc_num'], data.num_nodes)
        index = data.sorted_degree_nodes[- config['total_mcmc_num']:]
        for i in range(config['total_mcmc_num']):
            samples[i][index[i]] = 1
        samples = samples.repeat(config['repeat_times'], 1)
        return samples.t()
    m = Bernoulli(probs)
    samples = m.sample([config['total_mcmc_num'] * config['repeat_times']])
    samples = samples.detach().to(device)
    return samples.t()

def metro_sampling(probs, start_status, max_transfer_time,
                   device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    num_node = len(probs)
    num_chain = start_status.shape[1]
    index_col = torch.tensor(list(range(num_chain))).to(device)

    probs = probs.detach().to(device)
    samples = start_status.bool().to(device)

    count = 0
    for t in range(max_transfer_time * 5):
        if count >= num_chain*max_transfer_time:
            break
        index_row = torch.randint(low=0, high=num_node, size=[
                                  num_chain], device=device)
        chosen_probs_base = probs[index_row]
        chosen_value = samples[index_row, index_col]
        chosen_probs = torch.where(
            chosen_value, chosen_probs_base, 1-chosen_probs_base)
        accept_rate = (1 - chosen_probs) / chosen_probs
        r = torch.rand(num_chain, device=device)
        is_accept = (r < accept_rate)
        samples[index_row, index_col] = torch.where(
            is_accept, ~chosen_value, chosen_value)

        count += is_accept.sum()

    return samples.float().to(device)

def mcpg_sampling_ising(data,
                         start_result, probs,
                         num_ls, change_times, total_mcmc_num,
                         device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    probs = probs.to(torch.device("cpu"))
    num_edges = data.num_edges
    edges = data.edge_index
    nlr_graph = edges[0]
    nlc_graph = edges[1]
    edge_weight = data.edge_attr
    edge_weight_sum = data.edge_weight_sum
    graph_probs = start_result.clone()
    # get probs
    graph_probs = metro_sampling(
        probs, graph_probs, change_times, device)
    start = graph_probs.clone()

    temp = graph_probs[data.sorted_degree_nodes[0]].clone()
    graph_probs += temp
    graph_probs = graph_probs % 2

    graph_probs = (graph_probs - 0.5) * 2 + 0.5

    # local search
    temp = torch.zeros(4, graph_probs.size(dim=1)).to(device)
    expected_cut = torch.zeros(graph_probs.size(dim=1))
    cnt = 0
    while True:
        cnt += 1
        for i in range(num_edges):
            index = data.sorted_degree_edges[i]
            node_r = nlr_graph[index]
            node_c = nlc_graph[index]
            edges_r = data.n0_edges[index]
            edges_c = data.n1_edges[index]
            add_0 = data.add[0][index]
            add_1 = data.add[1][index]
            add_2 = data.add[2][index]

            temp_r_v = torch.mm(edges_r, graph_probs[data.n0[index]])
            temp_c_v = torch.mm(edges_c, graph_probs[data.n1[index]])

            temp[1] = temp_r_v + torch.rand(graph_probs.size(dim=1), device=device) * 0.1 + add_0
            temp[2] = temp_c_v + torch.rand(graph_probs.size(dim=1), device=device) * 0.1 + add_1
            temp[0] = temp[1] + temp[2] + torch.rand(graph_probs.size(dim=1),
                                                     device=torch.device(device)) * 0.1 - add_2

            max_index = torch.argmax(temp, dim=0)
            graph_probs[node_r] = torch.floor(max_index / 2)
            graph_probs[node_c] = max_index % 2

        if cnt >= num_ls:
            break

    # Convert Spins back to Ising model conventions -1 or +1
    spins = 2 * graph_probs - 1  # [nvar, chains]

    # raw energy per chain: E(s) = - sum J_ij s_i s_j
    energy_vec = (edge_weight * (spins[nlr_graph] * spins[nlc_graph])).sum(dim=0)  # [chains]

    # full‐chain advantage for policy gradient
    adv = (energy_vec - energy_vec.mean()).to(device)  # [chains]

    # pick the best chain out of each repeat group
    energy_reshape = energy_vec.view(-1, total_mcmc_num)  # [repeat_times, total_mcmc_num]
    idx = torch.argmin(energy_reshape, dim=0)  # [total_mcmc_num]
    for i0 in range(total_mcmc_num):
            idx[i0] = i0 + idx[i0] * total_mcmc_num
    temp_min = energy_vec[idx]  # [total_mcmc_num] 
    temp_min_info = graph_probs[:, idx]  # [nvar, total_mcmc_num]
    start_samples = start  # [nvar, chains]
    
    # return best energy state, probs of the best energy states, the total probs, and the advantage 
    return temp_min, temp_min_info, start_samples, adv

'''
Network
'''
class simple(torch.nn.Module):
    def __init__(self, output_num):
        super(simple, self).__init__()
        self.lin = torch.nn.Linear(1,  output_num)
        self.sigmoid = torch.nn.Sigmoid()

    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, alpha = 0.1, start_samples=None, value=None, 
                device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        x = torch.ones(1).to(device)
        x = self.lin(x)
        x = self.sigmoid(x)

        x = (x-0.5) * 0.6 + 0.5
        probs = x
        probs = probs.squeeze()
        retdict = {}
        reg = probs * torch.log(probs) + (1-probs) * torch.log(1-probs )
        reg = torch.mean(reg)
        if start_samples == None:
            retdict["output"] = [probs.squeeze(-1), "hist"]  # output
            retdict["reg"] = [reg, "sequence"]
            retdict["loss"] = [alpha * reg, "sequence"]
            return retdict

        res_samples = value.t().detach()

        start_samples_idx = start_samples * \
            probs + (1 - start_samples) * (1 - probs)
        log_start_samples_idx = torch.log(start_samples_idx)
        log_start_samples = log_start_samples_idx.sum(dim=1)
        loss_ls = torch.mean(log_start_samples * res_samples)
        loss = loss_ls + alpha * reg

        retdict["output"] = [probs.squeeze(-1), "hist"]  # output
        retdict["reg"] = [reg, "sequence"]
        retdict["loss"] = [loss, "sequence"]
        return retdict

    def __repr__(self):
        return self.__class__.__name__

'''
Algorithm
'''
def mcpg_solver(nvar, config, data, verbose=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sampler = mcpg_sampling_ising
    change_times = int(nvar/10)  # transition times for metropolis sampling

    net = simple(nvar)
    net.to(device).reset_parameters()
    optimizer = torch.optim.Adam(net.parameters(), lr=config['lr_init'])

    start_samples = None
    for epoch in range(config['max_epoch_num']):

        if epoch % config['reset_epoch_num'] == 0:
            net.to(device).reset_parameters()
            regular = config['regular_init']

        net.train()
        if epoch <= 0:
            retdict = net(regular, None, None)
        else:
            retdict = net(regular, start_samples, value)

        retdict["loss"][0].backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1)
        optimizer.step()

        # get start samples
        if epoch == 0:
            probs = (torch.zeros(nvar)+0.5).to(device)
            tensor_probs = sample_initializer(
                config["problem_type"], probs, config, data=data)
            temp_max, temp_max_info, temp_start_samples, value = sampler(
                data, tensor_probs, probs, config['num_ls'], 0, config['total_mcmc_num'])
            now_max_res = temp_max
            now_max_info = temp_max_info
            tensor_probs = temp_max_info.clone()
            tensor_probs = tensor_probs.repeat(1, config['repeat_times'])
            start_samples = temp_start_samples.t().to(device)

        # get samples
        if epoch % config['sample_epoch_num'] == 0 and epoch > 0:
            probs = retdict["output"][0]
            probs = probs.detach()
            temp_max, temp_max_info, start_samples_temp, value = sampler(
                data, tensor_probs, probs, config['num_ls'], change_times, config['total_mcmc_num'])
            # update now_max
            for i0 in range(config['total_mcmc_num']):
                if temp_max[i0] < now_max_res[i0]:
                    now_max_res[i0] = temp_max[i0]
                    now_max_info[:, i0] = temp_max_info[:, i0]

            # update if min is too small
            now_max = min(now_max_res).item()
            now_max_index = torch.argmin(now_max_res)

            now_min = max(now_max_res).item()
            now_min_index = torch.argmax(now_max_res)

            now_max_res[now_min_index] = now_max

            now_max_info[:, now_min_index] = now_max_info[:, now_max_index]
            temp_max_info[:, now_min_index] = now_max_info[:, now_max_index]

            # select best samples
            tensor_probs = temp_max_info.clone()
            tensor_probs = tensor_probs.repeat(1, config['repeat_times'])
            # construct the start point for next iteration
            start_samples = start_samples_temp.t()
            if verbose:
                if config["problem_type"] == "maxsat" and len(data.pdata) == 7:
                    res = max(now_max_res).item()
                    if res > data.pdata[5] * data.pdata[6]:
                        res -= data.pdata[5] * data.pdata[6]
                        print("o {:.3f}".format(res))
                elif "obj_type" in config and config["obj_type"] == "neg":
                    print("o {:.3f}".format((-min(now_max_res).item())))
                else:
                    print("o {:f}".format((min(now_max_res).item())))
        del (retdict)

    total_max = now_max_res
    best_sort = torch.argsort(now_max_res, descending=False)
    total_best_info = torch.squeeze(now_max_info[:, best_sort[0]])

    return min(total_max).item(), total_best_info, now_max_res, now_max_info

'''
Dataloader
'''
def maxcut_dataloader(path, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    with open(path) as f:
        fline = f.readline()
        fline = fline.split()
        num_nodes, num_edges = int(fline[0]), int(fline[1])
        edge_index = torch.LongTensor(2, num_edges)
        edge_attr = torch.Tensor(num_edges, 1)
        cnt = 0
        while True:
            lines = f.readlines(num_edges * 2)
            if not lines:
                break
            for line in lines:
                line = line.rstrip('\n').split()
                edge_index[0][cnt] = int(line[0]) - 1
                edge_index[1][cnt] = int(line[1]) - 1
                edge_attr[cnt][0] = float(line[2])
                cnt += 1
        data_maxcut = Data(num_nodes=num_nodes,
                           edge_index=edge_index, edge_attr=edge_attr)
        data_maxcut = data_maxcut.to(device)
        data_maxcut.edge_weight_sum = float(torch.sum(data_maxcut.edge_attr))

        data_maxcut = append_neighbors(data_maxcut)

        data_maxcut.single_degree = []
        data_maxcut.weighted_degree = []
        tensor_abs_weighted_degree = []
        for i0 in range(data_maxcut.num_nodes):
            data_maxcut.single_degree.append(len(data_maxcut.neighbors[i0]))
            data_maxcut.weighted_degree.append(
                float(torch.sum(data_maxcut.neighbor_edges[i0])))
            tensor_abs_weighted_degree.append(
                float(torch.sum(torch.abs(data_maxcut.neighbor_edges[i0]))))
        tensor_abs_weighted_degree = torch.tensor(tensor_abs_weighted_degree)
        data_maxcut.sorted_degree_nodes = torch.argsort(
            tensor_abs_weighted_degree, descending=True)

        edge_degree = []
        add = torch.zeros(3, num_edges).to(device)
        for i0 in range(num_edges):
            edge_degree.append(abs(edge_attr[i0].item())*(
                tensor_abs_weighted_degree[edge_index[0][i0]]+tensor_abs_weighted_degree[edge_index[1][i0]]))
            node_r = edge_index[0][i0]
            node_c = edge_index[1][i0]
            add[0][i0] = - data_maxcut.weighted_degree[node_r] / \
                2 + data_maxcut.edge_attr[i0] - 0.05
            add[1][i0] = - data_maxcut.weighted_degree[node_c] / \
                2 + data_maxcut.edge_attr[i0] - 0.05
            add[2][i0] = data_maxcut.edge_attr[i0]+0.05

        for i0 in range(num_nodes):
            data_maxcut.neighbor_edges[i0] = data_maxcut.neighbor_edges[i0].unsqueeze(
                0)
        data_maxcut.add = add
        edge_degree = torch.tensor(edge_degree)
        data_maxcut.sorted_degree_edges = torch.argsort(
            edge_degree, descending=True)

        return data_maxcut, num_nodes
    
def append_neighbors(data, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    data.neighbors = []
    data.neighbor_edges = []
    num_nodes = data.num_nodes
    for i in range(num_nodes):
        data.neighbors.append([])
        data.neighbor_edges.append([])
    edge_number = data.edge_index.shape[1]

    for index in range(0, edge_number):
        row = data.edge_index[0][index]
        col = data.edge_index[1][index]
        edge_weight = data.edge_attr[index][0].item()

        data.neighbors[row].append(col.item())
        data.neighbor_edges[row].append(edge_weight)
        data.neighbors[col].append(row.item())
        data.neighbor_edges[col].append(edge_weight)

    data.n0 = []
    data.n1 = []
    data.n0_edges = []
    data.n1_edges = []
    for index in range(0, edge_number):
        row = data.edge_index[0][index]
        col = data.edge_index[1][index]
        data.n0.append(data.neighbors[row].copy())
        data.n1.append(data.neighbors[col].copy())
        data.n0_edges.append(data.neighbor_edges[row].copy())
        data.n1_edges.append(data.neighbor_edges[col].copy())
        i = 0
        for i in range(len(data.n0[index])):
            if data.n0[index][i] == col:
                break
        data.n0[index].pop(i)
        data.n0_edges[index].pop(i)
        for i in range(len(data.n1[index])):
            if data.n1[index][i] == row:
                break
        data.n1[index].pop(i)
        data.n1_edges[index].pop(i)

        data.n0[index] = torch.LongTensor(data.n0[index]).to(device)
        data.n1[index] = torch.LongTensor(data.n1[index]).to(device)
        data.n0_edges[index] = torch.tensor(
            data.n0_edges[index]).unsqueeze(0).to(device)
        data.n1_edges[index] = torch.tensor(
            data.n1_edges[index]).unsqueeze(0).to(device)

    for i in range(num_nodes):
        data.neighbors[i] = torch.LongTensor(data.neighbors[i]).to(device)
        data.neighbor_edges[i] = torch.tensor(
            data.neighbor_edges[i]).to(device)

    return data

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str,
                        help="input the configuration file for the mcpg solver")
    parser.add_argument("problem_instance", type=str,
                        help="input the data file for the problem instance")

    args = parser.parse_args()
    with open(args.config_file) as f:
        config = yaml.safe_load(f)

    path = args.problem_instance
    start_time = time.perf_counter()
    dataloader = maxcut_dataloader
    data, nvar = dataloader(path)
    dataloader_t = time.perf_counter()
    res, solutions, _, _ = mcpg_solver(nvar, config, data, verbose=True)
    mcpg_t = time.perf_counter()

    if config["problem_type"] == "maxsat" and len(data.pdata) == 7:
        if res > data.pdata[5] * data.pdata[6]:
            res -= data.pdata[5] * data.pdata[6]
            print("SATISFIED")
            print("SATISFIED SOFT CLAUSES:", res)
            print("UNSATISFIED SOFT CLAUSES:", data.pdata[1] - data.pdata[-1] - res)
        else: 
            res = res//data.pdata[5]-data.pdata[6]
            print("UNSATISFIED")
            
    elif "obj_type" in config and config["obj_type"] == "neg":
        print("OUTPUT: {:.2f}".format(-res))
    else:
        print("OUTPUT: {:f}".format(res))


    print("DATA LOADING TIME: {:f}".format(dataloader_t - start_time))
    print("MCPG RUNNING TIME: {:f}".format(mcpg_t - dataloader_t))