import argparse
import datetime
import yaml
import re
import statistics

import logging
import os
import pickle

import torch.multiprocessing as mp
import torch
from reward_tuning.prompt import INITIAL_SYSTEM, INITIAL_USER, JUDGE_SYSTEM, JUDGE_USER
from reward_tuning.prompt import TRAIN_FEEDBACK, EVAL_FEEDBACK, CODE_FEEDBACK, CODE_OUTPUT_TIP
from reward_tuning.example import RESPONSE_SAMPLE_REWARD
from reward_tuning.template import REWARD_CLASS_TEMPLATE, REWARD_CONFIG_TEMPLATE, METRIC_FUNCTION_TEMPLATE, TASK_NOTE_TEMPLATE
from reward_tuning.client import Client

from scripts.train import train_try
from scripts.eval import eval_try

def setup_logger(exp_name):
    log_dir = 'logs_run'
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        filename=f'{log_dir}/{exp_name}.log',
        level=logging.DEBUG,            
        format='%(asctime)s - %(processName)s - %(levelname)s - %(message)s',  
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    return logging.getLogger()

def train_eval(return_queue, args, response, iter_id, sample_id, train_cfg, env_cfg, tune_cfg, device_id):
    logger = setup_logger(args.exp_name)
    logger.info(f"Sample {sample_id} training start on device {device_id}")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)

    train_queue = mp.Queue()
    eval_queue = mp.Queue()

    train_process = mp.Process(
        target=train_try,
        args=(train_queue, args, response, iter_id, sample_id, train_cfg, env_cfg, tune_cfg),
    )
    train_process.start()
    train_return = train_queue.get()
    train_process.join()
    if 'error' in train_return.keys():
        logger.error(f"RuntimeError: Training process for sample {sample_id}\n" + train_return['error'])
        return_queue.put({"error": train_return['error']})
        raise RuntimeError(f"Training process for sample {sample_id} error.")

    eval_process = mp.Process(
        target=eval_try,
        args=(eval_queue, args, train_return['exp_name'])
    )
    eval_process.start()
    eval_return = eval_queue.get()
    eval_process.join()
    if 'error' in eval_return.keys():
        logger.error(f"RuntimeError: Evaluation process for sample {sample_id}\n" + eval_return['error'])
        return_queue.put({"error": eval_return['error']})
        raise RuntimeError(f"Evaluation process for sample {sample_id} error.")

    return_queue.put({
        'train': train_return,
        'eval': eval_return,
    })

    logger.info(f"Sample {sample_id} finish")

def get_eval_result(result):
    eval_result = ''
    idx = result['train']['sample_id']
    metric = result['eval']['metric']
    eval_result += f'Index: {idx}\n'
    for key in metric.keys():
        eval_result += f'   {key}: {metric[key]:.3f}\n'
    return eval_result

def get_best(client, results):
    # LLM-based judgement from evaluation result
    assert len(results) > 0
    
    eval_result = ''
    for result in results:
        eval_result += get_eval_result(result)

    message = [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": JUDGE_USER.format(task_note=TASK_NOTE_TEMPLATE) + eval_result}
    ]

    response = client.response(message)
    match = re.search(r'``` *\n*best\s*\n(.*?)\n```', response, re.DOTALL)
    if match : 
        idx = int(match.group(1))
    else:
        raise ValueError("No best index found in response")

    for result in results:
        if idx == result['train']['sample_id']:
            return result
    raise ValueError("Best index not match")

def get_reward_reflection(result):
    content = "" 
    content += TRAIN_FEEDBACK.format(epoch_freq=result['train']['log_frequency'])
    log_dict = result['train']['train_log']
    for key in log_dict[list(log_dict.keys())[0]]:
        values = [round(log_dict[i][key], 2) for i in log_dict.keys()]
        content += f"{key}: {values}, Max {max(values)}, Mean {statistics.mean(values)}, Min {min(values)}\n"

    content += EVAL_FEEDBACK.format(metric_function=METRIC_FUNCTION_TEMPLATE)
    content += get_eval_result(result)
    
    content += CODE_FEEDBACK.format(max_episode_length=result['train']['max_episode_length'])
    content += CODE_OUTPUT_TIP

    return content

def overload_prompt(args, cfg):
    # May be needed for REWARD_CLASS & METRIC_FUNCTION in the future
    global REWARD_CLASS_TEMPLATE, REWARD_CONFIG_TEMPLATE, METRIC_FUNCTION_TEMPLATE, TASK_NOTE_TEMPLATE
    
    if 'note' in cfg['reward_tuning']:
        TASK_NOTE_TEMPLATE = cfg['reward_tuning']['note']

    if args.reward_template != None:
        env_cfg, train_cfg = pickle.load(
           open(f'logs/{args.reward_template}/cfgs.pkl', 'rb')
        )
        reward_scales = {'reward_scales': env_cfg['reward']['reward_scales']}
        REWARD_CONFIG_TEMPLATE = yaml.dump(reward_scales, sort_keys=False)

def main(args):
    mp.set_start_method("spawn", force=True)

    logger = setup_logger(args.exp_name)

    client_reward = Client(disable=args.disable, template=RESPONSE_SAMPLE_REWARD)
    client_judge = Client()

    with open(f'./cfgs/{args.cfg}', 'r') as file:
        cfg = yaml.safe_load(file)
    train_cfg = cfg['learning']
    env_cfg = cfg['environment']
    tune_cfg = cfg['reward_tuning']

    overload_prompt(args, cfg)

    base_message = [
        {"role": "system", "content": INITIAL_SYSTEM.format(reward_class=REWARD_CLASS_TEMPLATE, reward_config=REWARD_CONFIG_TEMPLATE)},
        {"role": "user", "content": INITIAL_USER.format(task_note=TASK_NOTE_TEMPLATE)}
    ]

    best_result_list = []

    resume_iter = None
    if args.resume_from_iter != None:
        iter = args.resume_from_iter.rsplit('_', 1)[1]
        assert iter.startswith('it')   
        resume_iter = int(iter[2:])
        assert resume_iter in range(tune_cfg['num_iterations']), f'{resume_iter} not in range'

    for iter_id in range(tune_cfg['num_iterations']):
        logger.info(f"Iteration {iter_id} start")

        if resume_iter != None and iter_id <= resume_iter:
            # if iter_id < resume_iter: continue
            results = []
            for sample_id in range(tune_cfg['num_samples']):
                exp_name = f'{args.exp_name}_it{iter_id}_{sample_id}'
                try:
                    result_train = pickle.load(open(f'logs/{exp_name}/result_train.pkl', 'rb'))
                    result_eval = pickle.load(open(f'logs/{exp_name}/result_eval.pkl', 'rb'))
                    results.append({
                        'train': result_train,
                        'eval': result_eval,
                    })
                except:
                    print(f'Resuming from {sample_id} failed.')
        else :
            return_queue = mp.Queue()
            process = []

            for sample_id in range(tune_cfg['num_samples']):
                # try:
                response = client_reward.response(base_message, log=f'Iter{iter_id}_{sample_id}')
                sample_process = mp.Process(
                    target=train_eval,
                    args=(return_queue, args, response, iter_id, sample_id, train_cfg, env_cfg, tune_cfg, (sample_id + 1) % torch.cuda.device_count())
                )
                sample_process.start()
                process.append(sample_process)

                # except Exception as e:
                #     print(f"Iteration {iter_id}_{sample_id} Error: {str(e)}")
                #     print('Waiting for debugging...')
                #     import pdb; pdb.set_trace()
                #     continue
            
            # Queue.get before p.join to prevent deadlock caused by waiting to queue.put into a full buffer
            results = []
            for _ in range(tune_cfg['num_samples']):
                result = return_queue.get()
                if 'error' not in result.keys():
                    results.append(result)

            error_process = []
            for sample_id, p in enumerate(process):
                p.join()
                logger.info(f'Collect process {sample_id} : {p}')
                if p.exitcode != 0:
                    error_process.append(sample_id)
            
            logger.info(f"Iteration {iter_id} finished. Process {error_process} failed.")             
            
        logger.info(f'Get best result among {len(results)} results.')
        best_result = get_best(client_judge, results)
        best_result_list.append(best_result)

        logger.info(f"Evaluation finished. Sample {best_result['train']['sample_id']} wins. ")
        logger.debug(f"Details of best reward:\n{best_result}")

        assist_message = {"role": "assistant", "content": best_result['train']['response']['raw']}
        user_message = {"role": "user", "content": get_reward_reflection(best_result)}

        if len(base_message) == 2:
            base_message += [assist_message, user_message]
        else :
            base_message[-2:] = [assist_message, user_message]
    
    logger.info(f"Finish all iterations.")
    for id, result in enumerate(best_result_list):
        result['train']['sample_id'] = id
    best_of_all = get_best(client_judge, best_result_list)
    logger.info(f"Best result of all reward parameters: {best_of_all['train']['exp_name']}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--task', type=str, default='go2-gait')
    parser.add_argument('-e', '--exp_name', type=str, default=None)
    parser.add_argument('-c', '--cpu', action='store_true', default=False)
    parser.add_argument('-v', '--vis', action='store_true', default=False)
    parser.add_argument('-B', '--num_envs', type=int, default=15000)
    parser.add_argument('--max_iterations', type=int, default=1000)
    parser.add_argument('-o', '--offline', action='store_true', default=False)
    parser.add_argument('--time', action='store_true', default=False)
    parser.add_argument('-p', '--ppo', action='store_false', default=True)
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--disable', action='store_true', default=False)
    parser.add_argument('--eval', action='store_true', help='Turn off domain randomization.', default=False)

    # Eval
    parser.add_argument('--num_eval_step', type=int, default=6000)
    parser.add_argument('-r', '--record', action='store_true', default=False)
    parser.add_argument('--record_length', help='unit: second', type=int, default=10)
    parser.add_argument('--resample_time', help='unit: second', type=float, default=2)
    parser.add_argument('--resume_from_iter', help='resume from log prefix. (e.g. go2-gait_3-20-4-12_it0)', type=str, default=None)
    parser.add_argument('--real', action='store_true', help='Eval with noise.', default=False)
    parser.add_argument('--efficient', action='store_true', help='Eval with efficient mode for large scale experiment.', default=False)

    parser.add_argument('--reward_template', type=str, default=None,
                        help='Load reward template by exp_name for further training with a proper init reward. (e.g. go2-handstand-walk-llm-real_4-14-18-8_it2_0)',)

    args = parser.parse_args()

    if args.resume_from_iter != None:
        args.exp_name = args.resume_from_iter.rsplit('_', 1)[0]

    if args.exp_name == None:
        args.exp_name = args.task
        now = datetime.datetime.now()
        args.exp_name += f'_{now.month}-{now.day}-{now.hour}-{now.minute}'

    print(f'Exp_name : {args.exp_name}')

    args.cfg = args.task + '.yaml'
    args.robot = args.task.split('-')[0]
    
    if args.debug:
        args.vis = True
        args.offline = True
        args.num_envs = 1
        args.cpu = True

    if not torch.cuda.is_available():
        args.cpu = True

    main(args)
