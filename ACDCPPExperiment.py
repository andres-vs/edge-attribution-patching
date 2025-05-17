import os
import sys
sys.path.append('Automatic-Circuit-Discovery/')

from acdc.TLACDCExperiment import TLACDCExperiment
from utils.prune_utils import acdc_nodes, get_nodes
from utils.graphics_utils import show

from typing import Callable, List, Literal

from transformer_lens import HookedTransformer
import torch as t
from torch import Tensor
import warnings
from tqdm import tqdm
import wandb

class ACDCPPExperiment():

    def __init__(
        self, 
        model: HookedTransformer,
        clean_data: Tensor,
        corr_data: Tensor,
        acdc_metric: Callable[[Tensor], Tensor],
        acdcpp_metric: Callable[[Tensor], Tensor],
        thresholds: List[float],
        save_graphs_after: float,
        wandb_run_name: str,
        using_wandb: bool = False,
        wandb_entity_name: str = None,
        wandb_project_name: str = None,
        proof_strategy: str = "all",
        proof_depth: int = None,
        num_examples: int = 1,
        verbose: bool = False,
        attr_absolute_val: bool = True,
        zero_ablation: bool = False,
        return_pruned_heads: bool = True,
        return_pruned_attr: bool = True,
        return_num_passes: bool = True,
        pass_tokens_to_metric: bool = False,
        pruning_mode: Literal["edge", "node"] = "node",
        no_pruned_nodes_attr: int = 10,
        **acdc_kwargs
    ):
        self.model = model

        self.clean_data = clean_data
        self.corr_data = corr_data

        
        self.using_wandb=using_wandb
        self.wandb_entity_name=wandb_entity_name
        self.wandb_project_name=wandb_project_name
        self.wandb_run_name=wandb_run_name
        self.verbose = verbose

        self.proof_strategy = proof_strategy
        self.proof_depth = proof_depth
        self.num_examples = num_examples
        
        self.acdc_metric = acdc_metric
        self.acdcpp_metric = acdcpp_metric
        self.pass_tokens_to_metric = pass_tokens_to_metric

        self.thresholds = thresholds 
        self.attr_absolute_val = attr_absolute_val
        self.zero_ablation = zero_ablation

        # For now, not using these (lol)
        self.return_pruned_heads = return_pruned_heads
        self.return_pruned_attr = return_pruned_attr
        self.return_num_passes = return_num_passes
        self.save_graphs_after = save_graphs_after
        self.pruning_mode: Literal["edge", "node"] = pruning_mode
        self.no_pruned_nodes_attr = no_pruned_nodes_attr

        if self.pruning_mode == "edge" and self.no_pruned_nodes_attr != 1:
            warnings.warn("I've been getting errors with no_pruned_nodes_attr > 1 with edge pruning, you may wish to switch to no_pruned_nodes_attr=1")

        self.acdc_args = acdc_kwargs
        if verbose:
            print('Set up model hooks')

    def setup_exp(self, threshold: float) -> TLACDCExperiment:
        exp = TLACDCExperiment(
            model=self.model,
            threshold=threshold,
            using_wandb=self.using_wandb,
            wandb_entity_name=self.wandb_entity_name,
            wandb_project_name=self.wandb_project_name,
            wandb_run_name=self.wandb_run_name,
            wandb_config={"threshold": threshold, "metric": self.acdc_metric, "proof_strategy": self.proof_strategy, "proof_depth": self.proof_depth, "num_examples": self.num_examples, "correct_metric_sign": True, "attr_absolute_val": self.attr_absolute_val},	
            ds=self.clean_data,
            ref_ds=self.corr_data,
            metric=self.acdc_metric,
            zero_ablation=self.zero_ablation,
            # save_graphs_after=self.save_graphs_after,
            online_cache_cpu=False,
            corrupted_cache_cpu=False,
            verbose=self.verbose,
            **self.acdc_args
        )
        exp.model.reset_hooks()
        exp.setup_model_hooks(
            add_sender_hooks=True,
            add_receiver_hooks=True,
            doing_acdc_runs=False
        )

        return exp
    
    def run_acdcpp(self, exp: TLACDCExperiment, threshold: float):
        if self.verbose:
            print('Running ACDC++')
            
        for _ in range(self.no_pruned_nodes_attr):
            pruned_nodes_attr = acdc_nodes(
                model=exp.model,
                clean_input=self.clean_data,
                corrupted_input=self.corr_data,
                metric=self.acdcpp_metric, 
                threshold=threshold,
                exp=exp,
                verbose=self.verbose,
                attr_absolute_val=self.attr_absolute_val,
                mode=self.pruning_mode,
            )
            t.cuda.empty_cache()
        return (get_nodes(exp.corr), pruned_nodes_attr)

    def run_acdc(self, exp: TLACDCExperiment):
        if self.verbose:
            print('Running ACDC')
            
        while exp.current_node:
            exp.step(testing=False)

        return get_nodes(exp.corr)
        # return (get_nodes(exp.corr), exp.num_passes)

    def run(self, save_after_acdcpp=True, save_after_acdc=True):
        os.makedirs(f'ims/{self.wandb_run_name}', exist_ok=True)

        pruned_heads = {}
        num_passes = {}
        pruned_attrs = {}

        for threshold in tqdm(self.thresholds):
            exp = self.setup_exp(threshold)
            acdcpp_nodes, pruned_nodes_attr = self.run_acdcpp(exp, threshold)
            # Only applying threshold to this one as these graphs tend to be HUGE
            if threshold >= self.save_graphs_after:
                print('Saving ACDC++ Graph')
                fname=f'ims/{self.wandb_run_name}/thresh{threshold}_before_acdc.png'
                show(exp.corr, fname=fname)
                if self.using_wandb:
                    try:
                        wandb.summary["acdcpp_final_graph"] = wandb.Image(fname)
                    except Exception as e:
                        pass # Usually a race condition when running many jobs. It's fine to not log an image.
            
            if self.using_wandb:
                wandb.summary["acdcpp_nodes"] = list(acdcpp_nodes),
                # wandb.summary["acdcpp_attrs"] = pruned_nodes_attr,
                wandb.summary["acdcpp_num_nodes"] = len(acdcpp_nodes),
                wandb.summary["acdcpp_num_edges"] = exp.corr.count_no_edges(),

            acdc_nodes = self.run_acdc(exp)
            # acdc_heads, passes = self.run_acdc(exp)

            print('Saving ACDC Graph')
            fname = f'ims/{self.wandb_run_name}/thresh{threshold}_after_acdc.png'
            show(exp.corr, fname=fname)
            if self.using_wandb:
                try:
                    wandb.summary["acdc_nodes"] = list(acdc_nodes),
                    wandb.summary["acdc_num_nodes"] = len(acdc_nodes),
                    wandb.summary["acdc_num_edges"] = exp.corr.count_no_edges(),
                    wandb.summary["acdc_final_graph"] = wandb.Image(fname)
                except Exception as e:
                    pass # Usually a race condition when running many jobs. It's fine to not log an image.


            pruned_heads[threshold] = [acdcpp_nodes, acdc_nodes]
            # num_passes[threshold] = passes
            pruned_attrs[threshold] = pruned_nodes_attr
            del exp
            t.cuda.empty_cache()
        t.cuda.empty_cache()
        return pruned_heads, num_passes, pruned_attrs