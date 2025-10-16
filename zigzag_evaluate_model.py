import pickle
from datetime import datetime

from zigzag import api
from zigzag.cost_model.cost_model import CostModelEvaluation
from zigzag.visualization import (
    bar_plot_cost_model_evaluations_breakdown,
    print_mapping,
)


def run_zigzag(
    yaml_accelerator: str,
    yaml_workload: str,
    output_folder: str = "outputs/",
    id: str = None,
    verbose: bool = False,
):
    """
    yaml_accelerator : path to accelerator file
    yaml_workload : path to workload file
    output_folder: path to output folder (needs to be created)
    id: sets the id of the experiment(can be set to a str of the hardware architecture + workload config)
    verbose: if True prints all Zigzag values
    """
    mapping_path = "inputs/a100_mapping.yaml"
    if id is None:
        experiment_id = datetime.now()
    else:
        experiment_id = id
    dump_folder = f"{output_folder}{experiment_id}"
    pickle_filename = f"{output_folder}{experiment_id}/cmes.pickle"

    energy, latency, cmes = api.get_hardware_performance_zigzag(
        workload=yaml_workload,
        accelerator=yaml_accelerator,
        mapping=mapping_path,
        opt="energy",
        dump_folder=dump_folder,
        pickle_filename=pickle_filename,
    )

    with open(pickle_filename, "rb") as fp:
        cmes = pickle.load(fp)
    cme: CostModelEvaluation = cmes[0]

    if verbose:
        print(f"Total network energy = {energy:.2e} pJ")
        print(f"Total network latency = {latency:.2e} cycles")

        bar_plot_cost_model_evaluations_breakdown(
            cmes[0:5], save_path=f"{dump_folder}/breakdown.png"
        )
        mem_names = [ml.memory_instance.name for ml in cme.mem_level_list]
        stall_slacks = cme.stall_slack_comb_collect
        print("Stall and slack per port of each memory instance:")
        for mem_name, ports_ss in zip(mem_names, stall_slacks):
            print(f"  {mem_name}: {ports_ss}")
        print(
            f"Latency: {cme.latency_total2:.3e} (bd: ideal -> {cme.ideal_temporal_cycle}, stall -> {cme.latency_total0 - cme.ideal_temporal_cycle} onload -> {cme.latency_total1 - cme.latency_total0}, offload -> {cme.latency_total2 - cme.latency_total1})"
        )
    return cme.latency_total2, cme.ideal_temporal_cycle


if __name__ == "__main__":
    workload_path = "inputs/gemm_layer.yaml"
    accelerator_path = "inputs/a100.yaml"

    latency, ideal_latency = run_zigzag(accelerator_path, workload_path, verbose=False)
    print(latency, ideal_latency)
