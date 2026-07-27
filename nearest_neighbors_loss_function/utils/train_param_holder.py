from itertools import product
from .train_model_params import TrainModelParams
from dataclasses import replace

class TrainParamHolder:

    def __init__(self, args):
        self.args = args
        self.combinations = self._generate_combinations()
        self.start = 0

    def __iter__(self):
        return self

    def _generate_combinations(self):

        combinations = []
        n_combinations = 0

        # check for lists inside of the input arguments
        list_params = []

        for param, argument in self.args.items():
            if isinstance(argument, list):
                list_params.append(param)
                n_combinations = n_combinations * len(argument)

        list_param_combinations = product([self.args[param_name] for param_name in list_params])
        starting_combination = {param: arguments for param, arguments in self.args.items() if param not in list_params}

        for args_combination in list_param_combinations:
            combination = TrainModelParams()

            # add arguments from list parameters
            for param_name, argument in zip(list_params, args_combination):
                combination = replace(combination, **{param_name: argument})
            
            # add arguments from non-list parameters
            combination = combination.replace(combination, **starting_combination)
            combinations.append(combination)

        self.end = n_combinations

        return combinations

    def __next__(self):

        combination = self.combinations[self.start]
        self.start += 1

        return combination