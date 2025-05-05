import os 

'''
Check job status: 
    squeue -u $USER 
Cancel job:
    scancel <job_id>
'''


log_dir = 'logs_sbatch'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

command = 'sbatch -J {task} -o {log_dir}/{task}-%j.out -e {log_dir}/{task}-%j.err run_sbatch.sh {task} \
        -r \
        --eval \
        -B 4096 \
        --max_iterations 1000 \
        --record_length 20 \
        --num_eval_step 10000\
        --efficient'

task_list = [
    ('anymal-walk',),
    ('anymal-walk1',),
    ('anymal-walk2',),
    ('g1-hop',),
    ('g1-hop1',),
    ('g1-hop2',),
    ('go2-handstand',),
    ('go2-handstand1',),
    ('go2-handstand2',),
    ('h1_2-walk',),
    ('h1_2-walk1',),
    ('h1_2-walk2',),
    ('leap_hand-walk',),
    ('leap_hand-walk1',),
    ('leap_hand-walk2',),
    ('shadow_hand-walk',),
    ('shadow_hand-walk1',),
    ('shadow_hand-walk2',),
]

for task in task_list:
    os.system(command.format(task=task[0], log_dir=log_dir))