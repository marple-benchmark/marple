from inference_utils import *

DATA_PATH = '/vision/u/emilyjin/mini-behavior-llm-baselines/data/'
#DATA_PATH = '/vision/u/emilyjin/marple_long/additional_data'

@hydra.main(version_base='1.1', config_path="config", config_name="rollout_low_policy.yaml")
def main(args): 
    num_samples = 100
    extra_steps = 1
    temp = 1.
    main_inference(args, DATA_PATH, num_samples, extra_steps, temp)


if __name__ == '__main__': 
    main()
