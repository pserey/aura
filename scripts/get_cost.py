"""
This script returns the cost of all csv 
allocation files in subdirectories with the
same name of the files. Gets a main directory
as an argument.

spot_all_ondemand
├── analytics
│   ├── analytics.csv
├── assisted_sales
│   ├── assisted_sales.csv
├── channel_supplier_management
│   ├── channel_supplier_management.csv
├── corp_systems
│   ├── corp_systems.csv
"""

import sys
import os
import argparse

import pandas as pd

def get_result_cost(result_cost):
    df = pd.read_csv(result_cost)

    return df['total_cost'][0]


def optimizer_cost(directory, filter_families=[]):
    total_cost = 0

    filtered = True
    if not filter_families:
        filtered = False

    for root, _, _ in os.walk(directory):
        if "output" in root:
            if filtered:
                # penultimate item is family name
                family = root.split('/')[-2]
                if family in filter_families:
                    for _, _, out_files in os.walk(root):
                        for filename in out_files:
                            if filename == "result_cost.csv":
                                result_cost = os.path.join(root, filename)

                                total_cost += get_result_cost(result_cost)
            else:
                for _, _, out_files in os.walk(root):
                    for filename in out_files:
                        if filename == "result_cost.csv":
                            result_cost = os.path.join(root, filename)

                            total_cost += get_result_cost(result_cost)

    return total_cost


def get_all_ondemand(main_dir, prices):
    
    results = []
    prices_flavors = set(prices['flavor'])

    for _, _, filenames in os.walk(main_dir):
        
        for file in filenames:
            if file.endswith('.csv'):
                demand = os.path.join(main_dir, file)
                total_demand = pd.read_csv(demand)
                flavors = list(total_demand.columns)
                flavors.pop(0)

                total = 0
                for flavor in flavors:
                    
                    if flavor not in prices_flavors:
                        print(f'Not getting cost for {flavor}')
                        continue
                    
                    price = float(prices['OnDemand'][prices['flavor'] == flavor].iloc[0])
                    value_flavor = total_demand[flavor].sum() * price
                    total += value_flavor

                results.append((file, total))

    results.sort()
    
    return results


def calculator_cost(demand):
    df = pd.read_csv(demand)
    sum = df['OnDemand'].sum()
    return sum


def advisor_cost(demand):
    df = pd.read_csv(demand)
    sum = df['AllMarkets'].sum()
    return sum


def dir_cost(main_dir, cost_function):
    """
    Calculates the cost for a directory with .csv
    files. Receives the cost calculation function.
    """

    results = []
    
    for file in os.listdir(main_dir):

        if file.endswith('.csv'):
            demand = os.path.join(main_dir, file)
            sum = cost_function(demand)
            results.append((file, sum))

    results.sort()
    
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Utility to get the cost of demands of the costplanner: Optimizer, Calculator and Advisor')
    parser.add_argument('input', type=str, help='The demand input file or directory.')
    parser.add_argument('mode', type=str, choices=['otm', 'calc', 'adv', 'all-od'],help='The input option.')
    parser.add_argument('-s', '--single', action='store_true', help='If demand is a directory or single csv (Only for "calc" option)')
    parser.add_argument('-sc', '--script', action='store_true', help='Flag for script output format.')
    parser.add_argument('--prices', help='prices.csv file path.')

    args = parser.parse_args()

    if args.prices:
        prices = pd.read_csv(args.prices)

    match args.mode:
        case 'otm':
            cost = optimizer_cost(args.input)
            # cost = optimizer_cost(args.input, filter_families=['c4', 'c5', 'c5a', 'c5ad', 'c5d', 'c6a', 'c6g', 'c6i', 'c7g', 'i3', 'm5', 'm5a', 'm5d', 'm6a', 'm6g', 'm6i', 'r5', 'r5a', 'r5d', 'r6a', 'r6g', 't2', 't3', 't3a', 't4g'])
            dir_name = args.input.split("/")[-1]

            if args.script:
                print(f"{cost:.2f}")
            else:
                print(f"Total cost of '{dir_name}': ${cost:.2f}")
        case 'calc':
            if args.single:
                sum = calculator_cost(args.input)
                file_name = args.input.split('/')[-1]
                
                print(f'Price for {file_name}: ${sum}')
            else:
                results = dir_cost(args.input, calculator_cost)
                
                if args.script:
                    for r in results:
                        print(r[1])
                else:
                    for r in results:
                        print(f'Price for {r[0]}: ${r[1]}')
        case 'adv':
            if args.single:
                sum = advisor_cost(args.input)
                file_name = args.input.split('/')[-1]
                
                print(f'Price for {file_name}: ${sum}')
            else:
                results = dir_cost(args.input, advisor_cost)
                
                if args.script:
                    for r in results:
                        print(r[1])
                else:
                    for r in results:
                        print(f'Price for {r[0]}: ${r[1]}')
        case 'all-od':
            if args.single:
                print('No single implementation for all on-demand.')
            else:
                results = get_all_ondemand(args.input, prices)
                
                if args.script:
                    for r in results:
                        print(r[1])
                else:
                    for r in results:
                        print(f'Price for {r[0]}: ${r[1]}')
