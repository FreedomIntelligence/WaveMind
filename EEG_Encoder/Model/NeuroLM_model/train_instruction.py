"""
by Wei-Bang Jiang
https://github.com/935963004/NeuroLM
"""

import os
import time
import argparse
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from .model_neurolm import NeuroLM
from .model import GPTConfig
from pathlib import Path
import tiktoken
# from utils import prepare_TUAB_dataset, prepare_TUEV_dataset, prepare_TUSL_dataset, prepare_HMC_dataset, prepare_Workload_dataset, cosine_scheduler, get_metrics
# from downstream_dataset import SEEDDataset
from torch.utils.data.dataset import ConcatDataset


master_process = None; device = None; dtype = None
ctx = None; ddp_rank = None; device_type = None
ddp = None; ddp_world_size = None; ddp_local_rank = None


def init(args):
    global ctx, master_process, ddp, ddp_world_size, ddp_rank, device, dtype, device_type, ddp_local_rank
    # various inits, derived attributes, I/O setup
    backend = 'nccl' # 'nccl', 'gloo', etc.
    device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
    
    ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
    if ddp:
        init_process_group(backend=backend)
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
        seed_offset = ddp_rank # each process gets a different seed
    else:
        # if not ddp, we are running on a single gpu, and one process
        master_process = True
        seed_offset = 0
        ddp_world_size = 1

    torch.manual_seed(args.seed + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
    device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
    # note: float16 data type will automatically use a GradScaler
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)


def get_instruct_datasets(args, downstream_dataset: str, eeg_max_len=-1, text_max_len=-1):
        dataset_info = {'name': downstream_dataset}
        if downstream_dataset == 'SEED':
            dataset_train = SEEDDataset(Path(args.dataset_dir, 'h5data/seed-3.hdf5'), window_size=800, stride_size=800, trial_start_percentage=0, 
                                        trial_end_percentage=0.6, is_instruct=True, eeg_max_len=eeg_max_len, text_max_len=text_max_len)
            dataset_val = SEEDDataset(Path(args.dataset_dir, 'h5data/seed-3.hdf5'), window_size=800, stride_size=800, trial_start_percentage=0.6, 
                                    trial_end_percentage=0.8, is_instruct=True, is_val=True, eeg_max_len=eeg_max_len, text_max_len=text_max_len)
            dataset_test = SEEDDataset(Path(args.dataset_dir, 'h5data/seed-3.hdf5'), window_size=800, stride_size=800, trial_start_percentage=0.8, 
                                    trial_end_percentage=1, is_instruct=True, is_val=True, eeg_max_len=eeg_max_len, text_max_len=text_max_len)
            
            dataset_info['metrics'] = ["accuracy", "balanced_accuracy", "cohen_kappa", "f1_weighted"]
            dataset_info['is_binary'] = False
            dataset_info['num_classes'] = 3
            dataset_info['result_idx'] = 11
            dataset_info['label_dic'] = {'Positive': 0, 'Neutral': 1, 'Negative': 2}
        elif downstream_dataset == 'TUAB':
            dataset_train, dataset_test, dataset_val = prepare_TUAB_dataset(Path(args.dataset_dir, 'TUAB/processed'), is_instruct=True, 
                                                                            eeg_max_len=eeg_max_len, text_max_len=text_max_len)

            dataset_info['metrics'] = ["pr_auc", "roc_auc", "accuracy", "balanced_accuracy"]
            dataset_info['is_binary'] = True
            dataset_info['result_idx'] = 7
            dataset_info['label_dic'] = {'Yes': 1, 'No': 0}
        elif downstream_dataset == 'TUEV':
            dataset_train, dataset_test, dataset_val = prepare_TUEV_dataset(Path(args.dataset_dir, 'TUEV'), is_instruct=True, 
                                                                            eeg_max_len=eeg_max_len, text_max_len=text_max_len)

            dataset_info['metrics'] = ["accuracy", "balanced_accuracy", "cohen_kappa", "f1_weighted"]
            dataset_info['is_binary'] = False
            dataset_info['num_classes'] = 6
            dataset_info['result_idx'] = 34
            dataset_info['label_dic'] = {'(A)': 0, '(B)': 1, '(C)': 2, '(D)': 3, '(E)': 4, '(F)': 5}
        elif downstream_dataset == 'TUSL':
            dataset_train, dataset_test, dataset_val = prepare_TUSL_dataset(Path(args.dataset_dir, 'TUSL'), is_instruct=True, 
                                                                            eeg_max_len=eeg_max_len, text_max_len=text_max_len)

            dataset_info['metrics'] = ["accuracy", "balanced_accuracy", "cohen_kappa", "f1_weighted"]
            dataset_info['is_binary'] = False
            dataset_info['num_classes'] = 3
            dataset_info['result_idx'] = 17
            dataset_info['label_dic'] = {'(A)': 0, '(B)': 1, '(C)': 2}
        elif downstream_dataset == 'HMC':
            dataset_train, dataset_test, dataset_val = prepare_HMC_dataset(Path(args.dataset_dir, 'HMC'), is_instruct=True, 
                                                                            eeg_max_len=eeg_max_len, text_max_len=text_max_len)

            dataset_info['metrics'] = ["accuracy", "balanced_accuracy", "cohen_kappa", "f1_weighted"]
            dataset_info['is_binary'] = False
            dataset_info['num_classes'] = 5
            dataset_info['result_idx'] = 22
            dataset_info['label_dic'] = {'(A)': 0, '(B)': 1, '(C)': 2, '(D)': 3, '(E)': 4}
        elif downstream_dataset == 'Workload':
            dataset_train, dataset_test, dataset_val = prepare_Workload_dataset(Path(args.dataset_dir, 'EEGWorkload'), is_instruct=True, 
                                                                            eeg_max_len=eeg_max_len, text_max_len=text_max_len)

            dataset_info['metrics'] = ["pr_auc", "roc_auc", "accuracy", "balanced_accuracy"]
            dataset_info['is_binary'] = True
            dataset_info['result_idx'] = 9
            dataset_info['label_dic'] = {'Yes': 1, 'No': 0}

        dataset_info['dataset_train'] = dataset_train
        dataset_info['dataset_val'] = dataset_val
        dataset_info['dataset_test'] = dataset_test

        if ddp:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=ddp_world_size, rank=ddp_rank, shuffle=True
            )
            data_loader_train = torch.utils.data.DataLoader(
                dataset_train, sampler=sampler_train,
                batch_size=args.eeg_batch_size,
                num_workers=10,
                pin_memory=True,
                drop_last=True,
            )
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
            data_loader_val = torch.utils.data.DataLoader(
                dataset_val, sampler=sampler_val,
                batch_size=int(args.eeg_batch_size * 1.5),
                num_workers=10,
                pin_memory=True,
                drop_last=False,
            )
            sampler_test = torch.utils.data.SequentialSampler(dataset_test)
            data_loader_test = torch.utils.data.DataLoader(
                dataset_test, sampler=sampler_test,
                batch_size=int(args.eeg_batch_size * 1.5),
                num_workers=10,
                pin_memory=True,
                drop_last=False,
            )
        else:
            data_loader_train = torch.utils.data.DataLoader(
                dataset_train,
                batch_size=args.eeg_batch_size,
                num_workers=10,
                pin_memory=True,
                drop_last=True,
                shuffle=True
            )
            data_loader_val = torch.utils.data.DataLoader(
                dataset_val,
                batch_size=int(args.eeg_batch_size * 1.5),
                num_workers=10,
                pin_memory=True,
                drop_last=False,
                shuffle=False
            )
            data_loader_test = torch.utils.data.DataLoader(
                dataset_test,
                batch_size=int(args.eeg_batch_size * 1.5),
                num_workers=10,
                pin_memory=True,
                drop_last=False,
                shuffle=False
            )
        dataset_info['data_loader_train'] = data_loader_train
        dataset_info['data_loader_val'] = data_loader_val
        dataset_info['data_loader_test'] = data_loader_test
        return dataset_info


def main(args):
    global ctx, master_process, ddp, ddp_world_size, ddp_rank, device, dtype, device_type, ddp_local_rank

    init(args)

    checkpoint_out_dir = os.path.join(args.out_dir, 'checkpoints/instruction-B')
    if master_process:
        os.makedirs(checkpoint_out_dir, exist_ok=True)

    # text data loader
    data_dir = os.path.join(args.out_dir, 'text')
    def get_batch(split):
        # We recreate np.memmap every batch to avoid a memory leak, as per
        # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
        if split == 'train':
            data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
        else:
            data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
        ix = torch.randint(len(data) - args.block_size, (args.text_batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i + args.block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i + 1:i + 1 + args.block_size]).astype(np.int64)) for i in ix])
        if device_type == 'cuda':
            # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        return x, y

    concat_datasets = True
    all_datasets = []
    for name in ['TUAB', 'TUEV', 'SEED', 'HMC', 'Workload', 'TUSL']:
        all_datasets.append(get_instruct_datasets(args, name, eeg_max_len=276, text_max_len=80))
    if concat_datasets:
        merge_datasets = ConcatDataset([dataset_info['dataset_train'] for dataset_info in all_datasets])
        if ddp:
            sampler_merge = torch.utils.data.DistributedSampler(
                merge_datasets, num_replicas=ddp_world_size, rank=ddp_rank, shuffle=True
            )
            data_loader_merge = torch.utils.data.DataLoader(
                merge_datasets, sampler=sampler_merge,
                batch_size=args.eeg_batch_size,
                num_workers=10,
                pin_memory=True,
                drop_last=True
            )
        else:
            data_loader_merge = torch.utils.data.DataLoader(
                merge_datasets,
                batch_size=args.eeg_batch_size,
                num_workers=10,
                pin_memory=True,
                drop_last=True,
                shuffle=True
            )
            
    # init these up here, can override if init_from='resume' (i.e. from a checkpoint)
    iter_num = 0

    tokenizer_ckpt_path = os.path.join(args.out_dir, args.tokenizer_path)

    if os.path.exists(os.path.join(checkpoint_out_dir, 'ckpt.pt')):
        init_from = 'resume'
    else:
        init_from = 'pretrained'
    # model init
    n_layer = 12
    n_head = 12
    n_embd = 768
    dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
    bias = False # do we use bias inside LayerNorm and Linear layers?
    model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=args.block_size,
                    bias=bias, vocab_size=50257, dropout=dropout) # start with model_args from command line
    if init_from == 'resume':
        print(f"Resuming training from {args.out_dir}")
        # resume training from a checkpoint.
        ckpt_path = os.path.join(checkpoint_out_dir, 'ckpt.pt')
        checkpoint = torch.load(ckpt_path, map_location=device)
        checkpoint_model_args = checkpoint['model_args']
        # force these config attributes to be equal otherwise we can't even resume training
        # the rest of the attributes (e.g. dropout) can stay as desired from command line
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
            model_args[k] = checkpoint_model_args[k]
        # create the model
        gptconf = GPTConfig(**model_args)
        model = NeuroLM(gptconf, init_from='gpt2')
        state_dict = checkpoint['model']
        # fix the keys of the state dictionary :(
        # honestly no idea how checkpoints sometimes get this prefix, have to debug more
        unwanted_prefix = '_orig_mod.'
        for k,v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        iter_num = checkpoint['iter_num']
        start_epoch = checkpoint['epoch'] + 1
    elif init_from == 'gpt':
        print(f"Initializing from tokenizer weights: {init_from}")
        # initialize from EEGPT weights
        model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=args.block_size,
                        bias=bias, vocab_size=50257, dropout=dropout) # start with model_args from command line
        # create the model
        gptconf = GPTConfig(**model_args)
        model = NeuroLM(gptconf, tokenizer_ckpt_path, init_from='gpt2')
        start_epoch = 0
    elif init_from == 'pretrained':
        print(f"Initializing training from {args.NeuroLM_path}")
        # resume training from a checkpoint.
        ckpt_path = os.path.join(args.out_dir, args.NeuroLM_path)
        checkpoint = torch.load(ckpt_path, map_location=device)
        checkpoint_model_args = checkpoint['model_args']
        # force these config attributes to be equal otherwise we can't even resume training
        # the rest of the attributes (e.g. dropout) can stay as desired from command line
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
            model_args[k] = checkpoint_model_args[k]
        # create the model
        gptconf = GPTConfig(**model_args)
        model = NeuroLM(gptconf, init_from='scratch')
        state_dict = checkpoint['model']
        # fix the keys of the state dictionary :(
        # honestly no idea how checkpoints sometimes get this prefix, have to debug more
        unwanted_prefix = '_orig_mod.'
        for k,v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        start_epoch = 0

    model.to(device)

    # initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.amp.GradScaler(device_type, enabled=(dtype == 'float16'))

    # optimizer
    optimizer = model.configure_optimizers(args.weight_decay, args.learning_rate, (args.beta1, args.beta2), device_type)
    if init_from == 'resume':
        optimizer.load_state_dict(checkpoint['optimizer'])
    checkpoint = None # free up memory

    # compile the model
    if compile:
        print("compiling the model... (takes a ~minute)")
        unoptimized_model = model
        model = torch.compile(model) # requires PyTorch 2.0

    # wrap model into DDP container
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])

    # logging

    num_training_steps_per_epoch = sum([len(dataset['dataset_train']) for dataset in all_datasets]) // args.batch_size // ddp_world_size
    lr_schedule_values = cosine_scheduler(
        args.learning_rate, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=int(args.warmup_ratio * num_training_steps_per_epoch * args.epochs)
    )

    enc = tiktoken.get_encoding("gpt2")
    decode = lambda l: enc.decode(l)
    
    # training loop
    datasets = [{'data_loader_train': data_loader_merge}] if concat_datasets else all_datasets
    X_text2, Y_text2 = get_batch('train') # fetch the very first batch
    t0 = time.time()
    local_iter_num = 0 # number of iterations in the lifetime of this process
    raw_model = model.module if ddp else model # unwrap DDP container if needed
    if args.eval_only:
        start_epoch = 0
    for epoch in range(start_epoch, args.epochs):
        for dataset_info in datasets:
            if args.eval_only:
                break
            for step, (batch) in enumerate(dataset_info['data_loader_train']):
                # determine and set the learning rate for this iteration
                lr = lr_schedule_values[iter_num] if args.decay_lr else args.learning_rate
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

                X_eeg, X_text, Y_text, input_chans, input_time, input_mask, gpt_mask = batch
                X_eeg = X_eeg.float().to(device, non_blocking=True)
                X_text = X_text.to(device, non_blocking=True)
                Y_text = Y_text.to(device, non_blocking=True)
                input_chans = input_chans.to(device, non_blocking=True)
                input_time = input_time.to(device, non_blocking=True)
                gpt_mask = gpt_mask.to(device, non_blocking=True)
                if input_mask is not None:
                    input_mask = input_mask.to(device, non_blocking=True)

                Y_eeg = torch.full((X_eeg.size(0), X_eeg.size(1)), fill_value=-1-raw_model.GPT2.config.vocab_size).to(device, non_blocking=True)

                # forward backward update, with optional gradient accumulation to simulate larger batch size
                # and using the GradScaler if data type is float16
                if ddp:
                    # in DDP training we only need to sync gradients at the last micro step.
                    # the official way to do this is with model.no_sync() context manager, but
                    # I really dislike that this bloats the code and forces us to repeat code
                    # looking at the source of that context manager, it just toggles this variable
                    model.require_backward_grad_sync = (step + 1) % args.gradient_accumulation_steps == 0

                with ctx:
                    loss1, log1, logits = model(X_eeg, Y_eeg, X_text, Y_text, input_chans, input_time, input_mask, eeg_text_mask=gpt_mask)
                    loss2, log2, _ = model(None, None, X_text2, Y_text2)
                    
                    model.train()

                    loss = (loss1 + loss2) / args.gradient_accumulation_steps # scale the loss to account for gradient accumulation
                # immediately async prefetch next batch while model is doing the forward pass on the GPU
                # backward pass, with gradient scaling if training in fp16
                scaler.scale(loss).backward()

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    # clip the gradient
                    if args.grad_clip != 0.0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    # step the optimizer and scaler if training in fp16
                    scaler.step(optimizer)
                    scaler.update()
                    # flush the gradients as soon as we can, no need for this memory anymore
                    optimizer.zero_grad(set_to_none=True)
                
                X_text2, Y_text2 = get_batch('train')

                # evaluate the loss on train/val sets and write checkpoints
                if (iter_num + 1) % args.log_interval == 0 and master_process:
                    print(f"epoch {epoch} step [{step + 1}/{num_training_steps_per_epoch}]: train total loss {log1['train/loss'] + log2['train/loss']:.4f}, instruction loss {log1['train/loss']:.4f}, text loss {log2['train/loss']:.4f}")
                if iter_num == 0 and args.eval_only:
                    break

                # timing and logging
                t1 = time.time()
                dt = t1 - t0
                t0 = t1
                iter_num += 1
                local_iter_num += 1
        
        if master_process and (not args.eval_only):
            checkpoint = {
                'model': raw_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'model_args': model_args,
                'iter_num': iter_num,
                'epoch': epoch
            }
            print(f"saving checkpoint to {checkpoint_out_dir}")
            torch.save(checkpoint, os.path.join(checkpoint_out_dir, f'ckpt.pt'))
            if (epoch + 1) % args.save_ckpt_freq == 0:
                print(f"saving checkpoint to {checkpoint_out_dir}")
                torch.save(checkpoint, os.path.join(checkpoint_out_dir, f'ckpt-{epoch}.pt'))
        
        # validation and test
        for dataset_info in all_datasets:
            print('Dataset:', dataset_info['name'])
            results_val = evaluate(raw_model, dataset_info, dataset_info['data_loader_val'], decode)
            print('=' * 10)
            print('Eval:')
            for metric in results_val.keys():
                print(metric + ':', results_val[metric])
            results_test = evaluate(raw_model, dataset_info, dataset_info['data_loader_test'], decode)
            print('=' * 10)
            print('Test:')
            for metric in results_test.keys():
                print(metric + ':', results_test[metric])
            print('=' * 10)
        if args.eval_only:
            break

    if ddp:
        destroy_process_group()


def get_pred(pred_string, dataset_info):
    if dataset_info['name'] == 'zuco':
        pred = pred_string[17:].split('<|endoftext|>')[0]
    else:
        pred = -1
        try:
            pred = pred_string.split(' ')[dataset_info['result_idx']]
            if pred.startswith('('):
                pred = pred[:3]
            pred = dataset_info['label_dic'][pred]
        except:
            # if master_process:
            #     print(f'label {pred} not found')
            pred = -1
    return pred

@torch.no_grad()
def evaluate(model, dataset_info, dataloader, decode):
    model.eval()
    preds = []
    targets = []
    for _, (batch) in enumerate(dataloader):
        X_eeg, X_text, label, input_chans, input_time, input_mask, gpt_mask = batch
        X_eeg = X_eeg.float().to(device, non_blocking=True)
        X_text = X_text.to(device, non_blocking=True)
        input_chans = input_chans.to(device, non_blocking=True)
        input_time = input_time.to(device, non_blocking=True)
        gpt_mask = gpt_mask.to(device, non_blocking=True)
        if input_mask is not None:
            input_mask = input_mask.to(device, non_blocking=True)

        with ctx:
            text = model.generate(X_eeg, X_text, input_chans, input_time, input_mask, eeg_text_mask=gpt_mask, max_new_tokens=5)
            text = text[:, 1:] # remove [SEP] token
            for i, t in enumerate(text):
                pred_string = decode(t.tolist())

                pred = get_pred(pred_string, dataset_info)
                if not dataset_info['is_binary']:
                    pred = np.eye(dataset_info['num_classes'])[pred]
                preds.append(pred)

            targets.append(label)
    
    model.train()

    targets = torch.cat(targets, dim=0).numpy()
    preds = np.array(preds)
    results = get_metrics(preds, targets, dataset_info['metrics'], dataset_info['is_binary'])

    return results


def get_args():
    parser = argparse.ArgumentParser('VQ training script', add_help=False)
    parser.add_argument('--out_dir', default='./', help='path where to save, empty for no saving')
    parser.add_argument('--dataset_dir', default='./', help='path where to save, empty for no saving')
    parser.add_argument('--tokenizer_path', default='checkpoints/VQ.py', help='path where tokenizer is')
    parser.add_argument('--NeuroLM_path', default='checkpoints/NeuroLM-B.pt', help='path where NeuroLM model is')
    parser.add_argument('--log_interval', default=10, type=int)
    parser.add_argument('--eval_only', default=False, action='store_true')
    # training args
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int)
    parser.add_argument('--eeg_batch_size', default=64, type=int)
    parser.add_argument('--text_batch_size', default=16, type=int)
    parser.add_argument('--epochs', default=5, type=int)
    parser.add_argument('--warmup_epochs', default=1, type=int)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--save_ckpt_freq', default=5, type=int)
    parser.add_argument('--block_size', default=1024, type=int)

    parser.add_argument('--learning_rate', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--min_lr', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-1,
                        help='weight decay (default: 1e-1)')
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.95)
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='clip gradients at this value, or disable if == 0.0')
    parser.add_argument('--decay_lr', default=True, action='store_false')
    parser.add_argument('--seed', default=1337, type=int)

    parser.add_argument('--compile', default=False, action='store_true')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    main(args)
