import pickle
import luigi
from pathlib import Path

import networkx as nx
from cls_luigi.search import ForbiddenActionFilter
from cls_luigi.tools.io_functions import dump_json, dump_pickle

import os

from implementations.template import *
from cls_luigi.inhabitation_task import RepoMeta, CLSLugiEncoder
from cls.debug_util import deep_str

from implementations.global_parameters import GlobalPipelineParameters
from import_pipeline_components import import_pipeline_components
from cls_luigi.tools.io_functions import load_json


def get_luigi_enumeration_time(cwd, ds_name, seed):

    return load_json(
        pjoin(
            cwd,
            "enumeration_outputs",
            f"seed-{seed}",
            f"{ds_name}",
            "logs",
            "train_time.json")
    )["total_seconds"]


if __name__ == "__main__":
    import argparse




    import_pipeline_components()
    import logging
    from cls_luigi.search.mcts import mcts_manager
    from cls_luigi.search.helpers import set_seed
    from cls_luigi.grammar import ApplicativeTreeGrammarEncoder
    from cls_luigi.grammar.hypergraph import get_hypergraph_dict_from_tree_grammar, build_hypergraph, \
        render_hypergraph_components
    from cls_luigi.tools.constants import MAXIMIZE, MINIMIZE

    from os.path import join as pjoin
    from os import makedirs, getcwd
    from cls.fcl import FiniteCombinatoryLogic
    from cls.subtypes import Subtypes
    from validators.not_forbidden_validator import FORBIDDEN

    logging.basicConfig(level=logging.DEBUG)
    
    CWD = getcwd()
    DATASETS_DIR = pjoin(str(Path(CWD).parent.absolute()), "datasets")
    DEFAULT_OUTPUT_DIR = pjoin(CWD, "search_outputs")
    
    
    parser = argparse.ArgumentParser(description='Run MCTS')
    parser.add_argument("--outputs_dir",
                        type=str,
                        default=DEFAULT_OUTPUT_DIR,
                        help="Directory for all pipeline enumeration outputs"
                        )

    parser.add_argument("--datasets_dir",
                        type=str,
                        default=DATASETS_DIR,
                        help="Directory for all datasets"
                        )

    parser.add_argument("--seed",
                        type=int,
                        default=123,
                        help="Seed for reproducibility")

    parser.add_argument("--metric",
                        type=str,
                        default="accuracy",
                        help="Metric for sorting the results")

    parser.add_argument('--ds_name',
                        type=str,
                        help='Dataset name')
    
    
    parser.add_argument("--workers",
                    type=int,
                    default=1,
                    help="Number of luigi workers")
    
    parser.add_argument("--n_jobs",
                        type=int,
                        default=1,
                        help="Number of jobs for pipeline compoenents"
                        )
    
    parser.add_argument("--debbug_mode",
                        action="store_true",
                        help="wheather to run in debugging mode (with no luigi server)"                        
    )
    
    parser.add_argument("--sense",
                        type=str,
                        choices=["MAX", "MIN"],
                        default="MAX"        
    )
    
    parser.add_argument("--time_factor",
                        type=float,
                        default=0.5,
                        help="this parameter will be multiplied by the time it took CLS-LUIGI to enumerate pipelines"
                        )
    
    parser.add_argument("--max_sec",
                        type=int,
                        default=0,
                        help="Max seconds to run the MCTS, If this time is 0, the enumeration time will be used"
                        )
    
    parser.add_argument("--mcts_explor",
                        type=float,
                        default=0.7,
                        help="exploration param,eter for the MCTS")
                        
    
    parser.add_argument("--pipeline_timeout",
                        type=int,
                        default=50,
                        help="pipeline evaluation will be cut after this time"
                        )
    
    parser.add_argument("--comp_timeout",
                        type=int,
                        default=0
                        )
    
    parser.add_argument("-punishment",
                        type=float,
                        default=0.0,
                        help="Punishment value for failed or timedout pipeline in the MCTS")
    
    parser.add_argument("prog_widening",
                        action="store_true",
                        help="wheather to use progressive widening or not")
    
    parser.add_argument("--pw_threshhold",
                        type=int,
                        default=2,
                        )    
    
    parser.add_argument("--pw_coeff",
                        type=float,
                        default=0.5,
                        )    
    
    
    parser.add_argument("--pw_max_child",
                        type=int,
                        default=6,
                        )    

    
    
                        
    config = parser.parse_args()
    



    # Configs
    # DEBBUG_MDOE = False
    # SEED = 123
    # N_JOBS = 1
    # PIPELINE_METRIC = "accuracy"
    # SENSE = MAXIMIZE

    # TIMING_FACTOR = .5
    # COMPONENT_TIMEOUT = None
    # PIPELINE_TIMEOUT = 3
    # PUNISHMENT_VALUE = 0.0

    forbidden_action_filter = ForbiddenActionFilter(FORBIDDEN)


    DS_DIR = pjoin(config.datasets_dir, config.ds_name)

    paths = {}
    paths["train_phase"] = {
        "x_train_path": pjoin(DS_DIR, f"seed-{config.seed}", "train_phase", "x_train.csv"),
        "y_train_path": pjoin(DS_DIR, f"seed-{config.seed}", "train_phase", "y_train.csv"),
        "x_valid_path": pjoin(DS_DIR, f"seed-{config.seed}", "train_phase", "x_valid.csv"),
        "y_valid_path": pjoin(DS_DIR, f"seed-{config.seed}", "train_phase", "y_valid.csv"),
    }

    paths["test_phase"] = {
        "x_train_path": pjoin(DS_DIR, f"seed-{config.seed}", "test_phase", "x_train.csv"),
        "y_train_path": pjoin(DS_DIR, f"seed-{config.seed}", "test_phase", "y_train.csv"),
        "x_test_path": pjoin(DS_DIR, f"seed-{config.seed}", "test_phase", "x_test.csv"),
        "y_test_path": pjoin(DS_DIR, f"seed-{config.seed}", "test_phase", "y_test.csv"),
    }


    MCTS_PARAMS = {
        "max_seconds": config.max_sec if config.max_sec > 0 else int(
            get_luigi_enumeration_time(
                CWD, config.ds_name, config.seed
                ) * config.time_factor),
        "exploration_param": config.mcts_explor
    }
    
    # Dirs
    OUTPUTS_DIR = pjoin(config.outputs_dir, config.ds_name)
    RUN_DIR = pjoin(OUTPUTS_DIR, f"seed-{config.seed}")
    CLS_LUIGI_OUTPUTS_DIR = pjoin(RUN_DIR, "cls_luigi_automl")
    CLS_LUIGI_PIPELINES_DIR = pjoin(CLS_LUIGI_OUTPUTS_DIR, "pipelines")
    LUIGI_OUTPUTS_DIR = pjoin(RUN_DIR, "luigi")
    LUIGI_PIPELINES_OUTPUTS_DIR = pjoin(LUIGI_OUTPUTS_DIR, "pipelines_outputs")
    LUIGI_INC_OUTPUT_DIR = pjoin(LUIGI_OUTPUTS_DIR, "incumbent_outputs")

    set_seed(config.seed)
    makedirs(OUTPUTS_DIR, exist_ok=True)
    makedirs(CLS_LUIGI_PIPELINES_DIR, exist_ok=False)
    makedirs(LUIGI_OUTPUTS_DIR, exist_ok=False)
    makedirs(LUIGI_PIPELINES_OUTPUTS_DIR, exist_ok=False)
    makedirs(LUIGI_INC_OUTPUT_DIR, exist_ok=False)

    # CLS Luigi
    target_class = Classifier
    target = target_class.return_type()
    repository = RepoMeta.repository
    fcl = FiniteCombinatoryLogic(repository, Subtypes(RepoMeta.subtypes), processes=1)
    inhabitation_result = fcl.inhabit(target)
    max_tasks_when_infinite = 10
    actual = inhabitation_result.size()
    max_results = max_tasks_when_infinite

    if actual > 0:
        max_results = actual

    pipelines_classes = [t for t in inhabitation_result.evaluated[0:max_results]]

    for p in pipelines_classes:
        with open(pjoin(CLS_LUIGI_PIPELINES_DIR, f"{p().task_id}.json"), "w") as f:
            json.dump(p, f, cls=CLSLugiEncoder)

    rtg = inhabitation_result.rules
    with open(pjoin(CLS_LUIGI_OUTPUTS_DIR, "applicative_regular_tree_grammar.txt"), "w") as f:
        f.write(deep_str(rtg))

    tree_grammar = ApplicativeTreeGrammarEncoder(rtg, target_class.__name__).encode_into_tree_grammar()
    dump_json(
        path=pjoin(CLS_LUIGI_OUTPUTS_DIR, "regular_tree_grammar.json"),
        obj=tree_grammar)

    hypergraph_dict = get_hypergraph_dict_from_tree_grammar(tree_grammar)
    hypergraph = build_hypergraph(hypergraph_dict)
    dump_pickle(
        obj=hypergraph,
        path=pjoin(CLS_LUIGI_OUTPUTS_DIR, "grammar_nx_hypergraph.pkl"))


    nx.write_graphml(hypergraph, pjoin(CLS_LUIGI_OUTPUTS_DIR, "grammar_nx_hypergraph.graphml"))
    render_hypergraph_components(hypergraph, pjoin(CLS_LUIGI_OUTPUTS_DIR, "grammar_hypergraph.png"),
                                 node_size=5000,
                                 node_font_size=11)

    _luigi_pipeline_params = {
        "x_train_path": paths["train_phase"]["x_train_path"],
        "x_test_path": paths["train_phase"]["x_valid_path"],
        "y_train_path": paths["train_phase"]["y_train_path"],
        "y_test_path": paths["train_phase"]["y_valid_path"],
        "luigi_outputs_dir": LUIGI_OUTPUTS_DIR,
        "pipelines_outputs_dir": LUIGI_PIPELINES_OUTPUTS_DIR,
        "seed": config.seed,
        "n_jobs": config.n_jobs
    }
    GlobalPipelineParameters().set_parameters(_luigi_pipeline_params)
    dump_json(pjoin(LUIGI_OUTPUTS_DIR, "luigi_pipeline_params.json"), _luigi_pipeline_params)

    prog_widening_params = {
         "threshold": config.pw_threshhold,
         "progressiv_widening_coeff": config.pw_coeff,
         "max_children": config.pw_max_child
    }

    pipeline_objects = [pipeline() for pipeline in pipelines_classes]
    opt = mcts_manager.MCTSManager(
        run_dir=RUN_DIR,
        pipeline_objects=pipeline_objects,
        mcts_params=MCTS_PARAMS,
        hypergraph=hypergraph,
        game_sense=MAXIMIZE if config.sense == "MAX" else MINIMIZE,
        pipeline_metric=config.metric,
        evaluator_punishment_value=config.punishment,
        pipeline_timeout=config.pipeline_timeout,
        component_timeout=config.comp_timeout if config.comp_timeout else None,
        prog_widening_params=prog_widening_params if config.prog_widening else None,
        pipeline_filters=[forbidden_action_filter],
        debugging_mode=config.debbug_mode
    )

    inc = opt.run_mcts()
    opt.save_results()

    incumbent_luigi_pipeline = opt.evaluator._get_luigi_task(inc["mcts_path"])

    _luigi_pipeline_params = {
        "x_train_path": paths["test_phase"]["x_train_path"],
        "x_test_path": paths["test_phase"]["x_test_path"],
        "y_train_path": paths["test_phase"]["y_train_path"],
        "y_test_path": paths["test_phase"]["y_test_path"],
        "luigi_outputs_dir": LUIGI_OUTPUTS_DIR,
        "pipelines_outputs_dir": LUIGI_INC_OUTPUT_DIR,
        "seed": config.seed,
        "n_jobs": config.n_jobs
    }
    GlobalPipelineParameters().set_parameters(_luigi_pipeline_params)
    luigi.build([incumbent_luigi_pipeline], local_scheduler=config.debbug_mode, detailed_summary=True)
    inc_score = {
        "luigi_task_id": inc["luigi_task_id"],
        "validation_score": inc["validation_score"],
        "test_score": incumbent_luigi_pipeline.get_score(config.metric)["test"]
    }
    dump_json(pjoin(RUN_DIR, "incumbent_score.json"), inc_score)
    opt.shut_down()


# python run_mcts.py --ds_name kc1