import math
import random
import yaml
import re
from pathlib import Path
import time

from zigzag_evaluate_model import run_zigzag

def simulated_annealing(
    objective_func,
    initial_state,
    neighbor_func,
    initial_temp=100.0,
    cooling_rate=0.99,
    min_temp=1e-3,
    max_iterations=1000
):
    """
    Generic simulated annealing algorithm.

    Parameters:
        objective_func (callable): Function to minimize.
        initial_state (any): Starting point of the search.
        neighbor_func (callable): Function that returns a neighbor of a given state.
        initial_temp (float): Starting temperature.
        cooling_rate (float): Factor by which temperature is multiplied each step.
        min_temp (float): Stop condition when temperature is below this.
        max_iterations (int): Maximum number of iterations.

    Returns:
        best_state, best_energy (tuple): Best found solution and its objective value.
    """
    current_state = initial_state
    current_energy = objective_func(current_state)
    best_state, best_energy = current_state, current_energy
    temperature = initial_temp

    for iteration in range(max_iterations):
        # Stop if the temperature is too low
        if temperature < min_temp:
            break

        # Generate a new neighbor solution
        new_state = neighbor_func(current_state)
        new_energy = objective_func(new_state)

        # Compute energy difference
        delta_energy = new_energy - current_energy

        # Decide whether to accept the new state
        if delta_energy < 0 or random.random() < math.exp(-delta_energy / temperature):
            current_state, current_energy = new_state, new_energy

            # Track the best solution found
            if current_energy < best_energy:
                best_state, best_energy = current_state, current_energy

        # Cool down
        temperature *= cooling_rate

        # Optional: print progress
        print(f"Iter {iteration}, Temp={temperature:.4f}, Best={best_energy:.4f}")

    return best_state, best_energy


#objective function should make a new yaml file with new inputs and call the zigzag_evaluate_model
def create_yaml_from_template(template_path, output_path, replacements):
    """
    Copy the yaml template file and replace placeholder tokens with actual values.

    Parameters
    ----------
    template_path : str or Path
        Path to the template YAML file (e.g. "inputs/a100_template.yaml")
    output_path : str or Path
        Path to save the customized YAML file (e.g. "inputs/a100_to_test.yaml")
    replacements : dict
        Dictionary mapping placeholders to replacement values.
        Example: {"D3_TEMP": 8, "D4_TEMP": 32}
    """

    template_path = Path(template_path)
    output_path = Path(output_path)

    # Read the YAML template as text (not parsed YAML yet)
    with open(template_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Replace placeholders like D3_TEMP with actual values
    for key, value in replacements.items():
        # Use regex to avoid partial matches, only whole words
        pattern = re.compile(rf"\b{re.escape(key)}\b")
        text = pattern.sub(str(value), text)

    # (Optional) Validate that resulting YAML is valid
    # try:
    #     parsed = yaml.safe_load(text)
    # except yaml.YAMLError as e:
    #     raise ValueError(f"Resulting YAML is invalid after replacements:\n{e}")

    # Write out the modified YAML
    with open(output_path, 'w', encoding='utf-8') as f:
        # yaml.dump(text, f, sort_keys=False)
        f.write(text)

    print(f"✅ New YAML file created: {output_path}")



def random_divisor_of(x):
    """
    Return a random integer that divides x evenly (i.e., x % n == 0).

    Returns
    -------
    int
        A random divisor of x.
    """
    divisors = [n for n in range(1, x+1) if x % n == 0]
    return random.choice(divisors), divisors





# Example usage — minimize f(x) = x²
if __name__ == "__main__":

    #This is the example for D3 and D4, where D3*D4=256 and both are integers. 
    #This will need to be extended for other parameters, but how will depend on the possible values you want to test out for those parameters and other constraints that link some of these parameters (like D3 and D4 are linked).
    #The "2" hyperparameter can be changed to have a bigger or smaller neighbourhood from which we take our next state.
    def neighbour_hardcoded(state):
        #D3 or state[0]:
        _, all_possible = random_divisor_of(256)
        #As neighbourhood function, we will go up to 2 possible values up or down, with the possible values for D3 given by all_possible
        D3_new = all_possible[(all_possible.index(state[0]) + random.randint(-2,2)) % (len(all_possible))]
        D4_new = int(256/D3_new)
        return [D3_new, D4_new]



    def objective_for_GPU_model(state):
        #create new yaml file with new parameters
        create_yaml_from_template("inputs/a100_template.yaml", "a100_to_test.yaml", {"D3_TEMP":state[0], "D4_TEMP":state[1]})

        #evaluate yaml file on zigzag
        latency, ideal_latency = run_zigzag(yaml_accelerator="a100_to_test.yaml", yaml_workload="inputs/gemm_layer.yaml")
        return abs(latency - ideal_latency)



    #Make initial state:
    D3, _ = random_divisor_of(256)
    init_state = [D3, int(256/D3)] #this incorporates the constraint that D3*D4 should be equal to 256

    #Simulated annealing:
    best_state, best_latency_difference = simulated_annealing(
        objective_func=objective_for_GPU_model,
        initial_state=init_state,
        neighbor_func=neighbour_hardcoded,
        initial_temp=100,
        cooling_rate=0.99,
        max_iterations=1000
    )

    print(f"Best solution: state = {best_state}, f(x) = {best_latency_difference}")





