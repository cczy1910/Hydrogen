import logging
import random

import ConfigSpace.hyperparameters as CSH
import numpy as np
import pandas as pd
from ConfigSpace.configuration_space import Configuration
from smac.configspace import ConfigurationSpace
from smac.facade.smac_bo_facade import SMAC4BO
from smac.facade.smac_hpo_facade import SMAC4HPO
from smac.optimizer.acquisition import LCB, EI, PI, EIPS, IntegratedAcquisitionFunction
from smac.runhistory.runhistory2epm import RunHistory2EPM4Cost, RunHistory2EPM4LogCost
from smac.scenario.scenario import Scenario

from smac.configspace.util import convert_configurations_to_array
from smac.utils import constants


def readData(cs, file):
    data = pd.read_csv(file)
    configs = []
    values = {}
    for i in data.index:
        line = data.T[i][:-1]
        config = Configuration(cs, values=dict(line))
        configs.append(config)
        values[config] = 5000 - data.T[i][-1]
    return values, configs


results = {}


def optFunc(values):
    def res(config):
        if config in values.keys():
            return values[config]
        else:
            results[config] = getPrediction(config)
            raise Exception("Tak zadumano")

    return res


def getPrediction(config):
    pred = smac.solver.epm_chooser.model.predict(convert_configurations_to_array([config]))[0][0][0]
    rh2epm = smac.solver.epm_chooser.rh2EPM
    min_y = rh2epm.min_y - (
            rh2epm.perc - rh2epm.min_y)  # Subtract the difference between the percentile and the minimum
    min_y -= constants.VERY_SMALL_NUMBER  # Minimal value to avoid numerical issues in the log scaling below
    if min_y == rh2epm.max_y:
        min_y *= 1 - 10 ** -10
    pred = np.exp(pred)
    pred = pred * (rh2epm.max_y - min_y) + min_y
    return -pred


if __name__ == '__main__':
    # freeze_support()
    logging.basicConfig(level=logging.INFO)  # logging.DEBUG for debug output

    cs = ConfigurationSpace()
    starting_material_ID = CSH.CategoricalHyperparameter("starting_material_ID", choices=["S2", "S5"])
    fd_percent = CSH.UniformFloatHyperparameter("fd_percent", lower=0, upper=18)
    solvent = CSH.CategoricalHyperparameter("solvent", choices=["cyclohexane", "CHCl3", "toluene"])
    concentration = CSH.UniformFloatHyperparameter("concentration", lower=5, upper=30)
    thickness = CSH.UniformFloatHyperparameter("thickness", lower=25, upper=150)
    cs.add_hyperparameters([
        starting_material_ID,
        fd_percent,
        solvent,
        concentration,
        thickness
    ])

    values, configs = readData(cs, "data.csv")

    scenario = Scenario({"run_obj": "quality",
                         "runcount-limit": len(values) + 1,
                         "cs": cs,
                         "deterministic": "true",
                         "limit-resources": "false"
                         })
    for _ in range(100):
        smac = SMAC4HPO(scenario=scenario,
                        rng=np.random.RandomState(random.randint(0, 1000000)),
                        tae_runner=optFunc(values),
                        # acquisition_function=EI,
                        # acquisition_function_kwargs={'par': -0.5},
                        # runhistory2epm=RunHistory2EPM4LogCost,
                        initial_design=None,
                        initial_configurations=list(values.keys()),
                        )
        try:
            smac.optimize()
        except Exception:
            pass

    data = []
    for config in results.keys():
        res = dict(config._values)
        res['flow'] = results[config]
        data.append(res)
    data.sort(key=lambda d: -d['flow'])
    dataframe = pd.DataFrame(data, columns=[
        'starting_material_ID',
        'fd_percent',
        'solvent',
        'concentration',
        'thickness',
        'flow'])
    dataframe.to_csv("pred.csv")
    print(results)
