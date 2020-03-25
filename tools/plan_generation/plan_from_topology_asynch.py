#!/usr/bin/env python3

from timeit import default_timer

import numpy as np
from ortools.linear_solver import pywraplp
import json
import argparse

from topology_parser import get_topology_matrix, parse_conf

parser = argparse.ArgumentParser(description="create transfer plan.")
parser.add_argument("hugectr_conf", type=str, help="path to hugectr json file")
args=parser.parse_args()

modes = {"scatter":0, "gather":1, "all2all":2}
mode = "all2all"
main_gpu = 0

# dgx1 volta: 6 nvlink per gpu
gpu_list, plan_file = parse_conf(args.hugectr_conf)
capacities = get_topology_matrix("", gpu_list = gpu_list)
bisection_width = 1

# ps0001 pascal: 4 nvlink per gpu
# capacities = get_topology_matrix("ps0001_topology.txt")
# bisection_width = 4

# ps0001 pascal: 4 nvlink per gpu
# num_gpus = 4
# capacities = np.eye(num_gpus) * num_gpus
# capacities += np.array([[0,2,1,1],
#                         [2,0,1,1],
#                         [1,1,0,2],
#                         [1,1,2,0]])
# bisection_width = 4

# like ps0001 but volta: 6 nvlink per gpu
# num_gpus = 4
# capacities = np.eye(num_gpus) * num_gpus
# capacities += np.array([[0,2,2,2],
#                         [2,0,2,2],
#                         [2,2,0,2],
#                         [2,2,2,0]])
# bisection_width = 6

# half of dgx1 volta: 6 nvlink per gpu
# num_gpus = 4
# capacities = np.eye(num_gpus) * num_gpus
# capacities += np.array([[0,1,1,2],
#                         [1,0,2,1],
#                         [1,2,0,2],
#                         [2,1,2,0]])
# bisection_width = 5

# dgx1 volta: 6 nvlink per gpu
# num_gpus = 8
# capacities = np.eye(num_gpus) * num_gpus
# capacities += np.array([[0,1,1,2,2,0,0,0],
#                         [1,0,2,1,0,2,0,0],
#                         [1,2,0,2,0,0,1,0],
#                         [2,1,2,0,0,0,0,1],
#                         [2,0,0,0,0,1,1,2],
#                         [0,2,0,0,1,0,2,1],
#                         [0,0,1,0,1,2,0,2],
#                         [0,0,0,1,2,1,2,0]])
# bisection_width = 6

# like dgx1 volta: 6 nvlink per gpu, different ring structure
# num_gpus = 8
# capacities = np.eye(num_gpus) * num_gpus
# capacities += np.array([[0,2,1,1,2,0,0,0],
#                         [2,0,1,1,0,2,0,0],
#                         [1,1,0,2,0,0,2,0],
#                         [1,1,2,0,0,0,0,2],
#                         [2,0,0,0,0,1,1,2],
#                         [0,2,0,0,1,0,2,1],
#                         [0,0,2,0,1,2,0,1],
#                         [0,0,0,2,2,1,1,0]])
# bisection_width = 8

num_gpus = capacities.shape[0]

main_degree = int(np.sum(capacities[main_gpu, :]) - capacities[main_gpu,main_gpu])
print("main:", main_gpu, "degree:", main_degree)



max_capacity = np.max(capacities * (capacities < num_gpus))
if max_capacity > 2:
    print("topologies with more than 2 nvlinks at the same edge are not supported.")
    raise SystemExit()

capacities_non_zero = np.where(capacities == 0, 1.0E-7, capacities)
lengths = np.where(capacities <= max_capacity, max_capacity / capacities_non_zero, 1)
# print("lengths:")
# print(lengths)

print("topology:")
print(capacities)
print("max links:", max_capacity)

if modes[mode] == 0: # scatter
    num_commodities = 1
    parts_per_commodity = int(main_degree // np.gcd(main_degree, num_gpus-1))
    # one gpu starts with one chunk of the commodity
    source = main_gpu
    commodities_out = np.ones(num_gpus) * parts_per_commodity
    commodities_in = np.zeros(num_gpus)
    commodities_in[source] += np.sum(commodities_out)
elif modes[mode] == 1: # gather
    num_commodities = 1
    parts_per_commodity = int(main_degree // np.gcd(main_degree, num_gpus-1))
    # one gpu starts with all chunks of the commodity
    target = main_gpu
    commodities_in = np.ones(num_gpus) * parts_per_commodity
    commodities_out = np.zeros(num_gpus)
    commodities_out[target] += np.sum(commodities_in)
elif modes[mode] == 2: # all-to-all
    num_commodities = num_gpus
    parts_per_commodity = int(bisection_width // np.gcd(bisection_width, int(np.ceil(num_gpus/2)*np.floor(num_gpus/2))))
    # each gpu starts with one of each commodity
    commodities_in = np.ones((num_gpus,num_commodities)) * parts_per_commodity
    commodities_out = np.diagflat( np.sum(commodities_in, axis=0) )
else:
    raise SystemExit()

# parts_per_commodity = 1
capacities += np.eye(num_gpus) * num_gpus * parts_per_commodity

# each gpu wants to have all of its own commodity
print("commodities at begin:")
print(commodities_in)
print("commodities at end:")
print(commodities_out)
# storage size of each gpu in number of items
# max_space_per_gpu = num_commodities * parts_per_commodity

min_steps = np.ceil((num_commodities-1) * parts_per_commodity / main_degree * max_capacity).astype(int)
max_steps = num_gpus*num_commodities*parts_per_commodity

for steps in range(min_steps, max_steps+1):
    print("Creating flow problem for %i timesteps" %(steps))

    flows_per_gpu = num_gpus*num_commodities

    # solver = pywraplp.Solver('mcf', pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)
    solver = pywraplp.Solver('mcf', pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)
    # solver = pywraplp.Solver('mcf', pywraplp.Solver.CLP_LINEAR_PROGRAMMING)
    # solver = pywraplp.Solver('mcf', pywraplp.Solver.BOP_INTEGER_PROGRAMMING)

    objective = solver.Objective()
    objective.SetMinimization()

    # (1) conservation_constraints for each commodity flow at each gpu in each step
    conservation_constraints = np.empty((steps+1,num_gpus,num_commodities), dtype=pywraplp.Constraint)
    # Flow conservation on transit nodes: The amount of a flow entering is the same that exits the node.
    in_out_bounds = np.zeros_like(conservation_constraints)
    # Flow conservation at the source: A flow must exit its source node completely.
    in_out_bounds[0] = commodities_in
    # Flow conservation at the destination: A flow must enter its sink node completely.
    in_out_bounds[-1] = -1*commodities_out

    for i in range(conservation_constraints.size):
        conservation_constraints.flat[i] = solver.Constraint(in_out_bounds.flat[i], in_out_bounds.flat[i])

    # (2) space constraints: each gpu can hold exactly <num_gpus> commodities
    # space_constraints = np.empty((steps,num_gpus), dtype=pywraplp.Constraint)
    # for i in range(space_constraints.size):
    #     space_constraints.flat[i] = solver.Constraint(-solver.infinity(), max_space_per_gpu)

    flows = []
    edge_constraint = {}
    # create flows for each edge for each step
    for step in range(steps):
        for src in range(num_gpus):
            for trg in range(num_gpus):
                if capacities[src][trg] == 0:
                    continue

                # copying edges
                if src == trg:
                    # copy at most num_commodities items
                    edge_capacity = num_gpus*parts_per_commodity
                    cost = 0
                    length = 1
                # sending edges
                if (src != trg):
                    edge_capacity = 1
                    cost = 1
                    length = int(lengths[src][trg])

                if step+length <= steps:
                    # (3) Link capacity: The sum of all flows routed over a link does not exceed its capacity.
                    edge_name = (step,src,trg)
                    edge_constraint[edge_name] = solver.Constraint(-solver.infinity(), edge_capacity)

                    for c in range(num_commodities):
                        name = 't'+str(step)+' '+str(src)+'to'+str(trg)+' c'+str(c)
                        flow = solver.IntVar(0, edge_capacity, name)
                        flows.append(flow)
                        # increase cost with multiplicity
                        objective.SetCoefficient(flow, cost*length)
                        # sum flows of same edge
                        for prev in range(length):
                            if (step - prev) >= 0:
                                edge_name = (step - prev,src,trg)
                                edge_constraint[edge_name].SetCoefficient(flow, 1)
                        # sum flows of same gpu
                        # space_constraints[step][src].SetCoefficient(flow, 1)
                        # outgoing flow at src
                        conservation_constraints[step][src][c].SetCoefficient(flow, 1)
                        # incoming flow at trg
                        conservation_constraints[step+length][trg][c].SetCoefficient(flow, -1)


    print('Number of variables =', solver.NumVariables())
    print('Number of constraints =', solver.NumConstraints())

    status = solver.Solve()

    if status != solver.OPTIMAL:
        if status == solver.FEASIBLE:
            print('A potentially suboptimal solution was found\n.')
        else:
            print('The solver could not solve the problem in %i timesteps.\n' % (steps))
        continue
    else:
        print('A solution was found:')

        copies = 0
        transfers = 0

        flows_array = np.zeros((steps, num_gpus, num_gpus, num_commodities))

        step_time = np.ones(steps) / parts_per_commodity / max_capacity

        for flow in flows:
            # print(flow, flow.solution_value(), objective.GetCoefficient(flow))
            step, edge, commodity = str(flow).split()
            step = int(step[1:])
            src, trg = edge.split("to")
            src = int(src)
            trg = int(trg)
            commodity = int(commodity[1:])
            value = flow.solution_value()
            if (src == trg) and (value > 0):
                copies += 1
            if (src != trg) and (value > 0):
                transfers += 1
                # step_time[step] = max(step_time[step], value)
            flows_array[step, src, trg, commodity] += value

        print("copies:", copies)
        print("transfers:", transfers)
        print("time for each step:", step_time)
        print("total time:", np.sum(step_time))

        # print all flows
        for step in range(steps):
            print("\nstep",step)
            print(flows_array[step])
            # for gpu in range(num_gpus):
            #     print("from gpus",gpu,"send commodity (column) to gpu (row)")
            #     print(flows_array[step,gpu])


        # trace sequence of owners per commodity
        plan = []
        while np.any(flows_array[0] > 0):
            step = 0
            src, trg, commodity = np.transpose(np.nonzero(flows_array[step]))[0]
            flows_array[step,src,trg,commodity] -= 1
            owners = [src]
            for i in range(int(lengths[src][trg])):
                owners.append(int(trg))
            step += int(lengths[src][trg])
            while step < steps:
                src = owners[-1]
                trg = np.nonzero(flows_array[step,src,:,commodity])[0][0]
                flows_array[step,src,trg,commodity] -= 1
                for i in range(int(lengths[src][trg])):
                    owners.append(int(trg))
                step += int(lengths[src][trg])
            # print(owners)
            plan.append(owners)
        max_seq_len = 0
        for p in plan:
            max_seq_len = max(max_seq_len, len(p))
            # print(p)

        values, counts = np.unique(plan, return_counts=True, axis=0)

        print("num paths:", len(counts))

        data = {
            "type" : mode,
            "num_gpus" : num_gpus,
            "main_gpu" : main_gpu,
            "num_steps" : steps,
            "num_chunks" : parts_per_commodity,
            "plan" : values.tolist(),
            "chunks" : counts.tolist()
        }
        # print(data)

        json_string = json.dumps(data)
        print(json_string)
        # json_name = mode+"_plan.json"
        json_name = plan_file
        print("saving json to '%s'" %(json_name))
        with open(json_name, "w") as file:
            json.dump(data, file)

        break
