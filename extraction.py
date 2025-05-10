from utils_gpt import *
from openai import AzureOpenAI
import argparse
import subprocess

def main(args):
    example_env_string = extract_code('envs/go2_env.py')
    example_cfg_string = extract_code('cfgs/go2.yaml')

    vec_env_string = extract_code('envs/vec_env.py')
    vec_cfg_string = extract_code(f'cfgs/{args.task}.yaml')
    state_wrapper_string = extract_code('envs/state_wrapper.py')
    reward_wrapper_string = extract_code('envs/reward_wrapper.py')

    api_key = None
    endpoint = None

    # openai.api_key = os.getenv("OPENAI_API_KEY")
    client = AzureOpenAI(
        api_key=api_key,
        api_version="2024-12-01-preview",
        azure_endpoint=endpoint,
    )

    print("Starting Iteration:", 0)
    system_prompt = """
You are a helpful assistant.
You will be given a python class of a reinforcement learning environment and corresponding yaml configuration file as example environment.
Your task is to adapt the example environment to a new environment by writting two wrappers. Try to keep the functions in example environment as much as possible.
"""
    state_wrapper_prompt = f"""
The **example environment code** is:
```python
{example_env_string}
```
The **base class** of new environment is:
```python
{vec_env_string}
```
The **state wrapper** used to update the state is:
```python
{state_wrapper_string}
```
Your task is to fill **state wrapper** now.

**state wrapper** should contain same functions in **example environment code** except reward functions.
You can do copy the functions from **example environment code** to **reward wrapper**.
You can assume the self.env_cfg contains everything you need.
Ouput the extended **state wrapper** in the following format:
```python
...
```
"""
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": state_wrapper_prompt}]

    response = generate_state_wrapper(client, messages)
    messages.append({"role": "assistant", "content": response})

    reward_wrapper_prompt = f"""
The **reward wrapper** used to update the reward is:
```python
{reward_wrapper_string}
```
Your task is to fill **reward wrapper** now.
**reward wrapper** should contain all reward functions functions in **example environment code**.
You can do copy the reward functions from **example environment code** to **reward wrapper**.
You can assume the self.env_cfg contains everything you need.
Ouput the extended **reward wrapper** in the following format:
```python
...
```
"""
    messages.append({"role": "user", "content": reward_wrapper_prompt})

    response = generate_reward_wrapper(client, messages)
    messages.append({"role": "assistant", "content": response})

    cfg_prompt = f"""
After generating the code, generate additional configurations should be added.
The **example environment cfg** is:
{example_cfg_string}
The current **base environment cfg** is:
{vec_cfg_string}
Generate the new cfg in the following format:
```yaml
```
And also generate a description of the cfgs in the following format:
```text
```
Every block should end with ```.
"""
    messages.append({"role": "user", "content": cfg_prompt})

    response = generate_cfg(client, messages, args.task)
    messages.append({"role": "assistant", "content": response})

    print("Generating finished.")
    print("Running the code...")

    while True:
        result = subprocess.run(
            ["python", f"train_gpt.py", "--task", "leap_hand"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        if "ETA" in result.stdout:
            print("Code extraction finished.")
            break

        print(result.stdout)
        print(result.stderr)

        error_prompt = f"""
When running the code you generated, I got the following error:
{result.stdout}
{result.stderr}
Choose the file you want to modify:
1.state wrapper
2.reward wrapper
3.cfg file
Reply with the number of the file you want to modify, choose from (1/2/3)
"""

        messages.append({"role": "user", "content": error_prompt})
        response = complete(client, messages)
        messages.append({"role": "assistant", "content": response})
        print(response)

        if "1" in response:
            state_wrapper_prompt = """
Ouput the extended **state wrapper** in the following format:
```python
...
```
"""
            messages.append({"role": "user", "content": state_wrapper_prompt})
            response = generate_state_wrapper(client, messages)
            messages.append({"role": "assistant", "content": response})
        elif "2" in response:
            reward_wrapper_prompt = """
Ouput the extended **reward wrapper** in the following format:
```python
...
```
"""
            messages.append({"role": "user", "content": reward_wrapper_prompt})
            response = generate_reward_wrapper(client, messages)
            messages.append({"role": "assistant", "content": response})
        elif "3" in response:
            cfg_prompt = """
Generate the new cfg in the following format:
```yaml
```
And also generate a description of the cfgs in the following format:
```text
```
Every block should end with ```.
"""
            messages.append({"role": "user", "content": cfg_prompt})
            response = generate_cfg(client, messages, args.task)
            messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="leap_hand")
    parser.add_argument("-u", "--use_generated_response", action='store_true', default=False)
    args = parser.parse_args()

    main(args)